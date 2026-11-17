import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../worker/index.js', import.meta.url),'utf8');
const {default: worker} = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const env = {GEMINI_API_KEY:'fixture-google-key',SESSION_SECRET:'fixture-session-secret-1234567890',GEMINI_MODEL:'gemini-3.5-flash-lite'};
const origin = 'https://fixture.example';
const item = {evidence_id:'w0:tci',team:'Team A',start:0,end:900,metric:'tci',value:50};
const claim = {evidence_id:item.evidence_id,value:item.value,alternative_explanation:'No causal inference is supported.'};
let calls=0;
const originalFetch=globalThis.fetch;
globalThis.fetch=async (url,init)=>{
 calls++;
 assert.equal(init.headers['x-goog-api-key'],env.GEMINI_API_KEY);
 assert.ok(!url.includes(env.GEMINI_API_KEY));
 const body=JSON.parse(init.body);
 assert.equal(body.generationConfig.responseMimeType,'application/json');
 return Response.json({candidates:[{finishReason:'STOP',content:{parts:[{text:JSON.stringify({claims:[claim]})}]}}],usageMetadata:{promptTokenCount:20,candidatesTokenCount:10}});
};
function request(path,body,token,ip='fixture-1',extra={}) {
 const headers={'content-type':'application/json','cf-connecting-ip':ip,...extra};
 if(token)headers.authorization='Bearer '+token;
 return new Request(origin+path,{method:'POST',headers,body:JSON.stringify(body||{})});
}
try {
 const home=await worker.fetch(new Request(origin),env);
 assert.match(await home.text(),/MaxwellGerhart\/DS340W-Project/);
 const health=await (await worker.fetch(new Request(origin+'/api/health'),env)).json();
 assert.equal(health.available,true);
 assert.ok(!JSON.stringify(health).includes(env.GEMINI_API_KEY));
 let response=await worker.fetch(request('/api/select',{role:'single_analyst',evidence:[item],seed:42}),env);
 assert.equal(response.status,401);
 const session=await (await worker.fetch(request('/api/session'),env)).json();
 assert.ok(session.token);
 response=await worker.fetch(request('/api/select',{role:'single_analyst',evidence:[item],seed:42},session.token),env);
 assert.equal(response.status,200);
 const result=await response.json();
 assert.deepEqual(result.claims,[claim]);assert.equal(calls,1);
 assert.ok(!JSON.stringify(result).includes(env.GEMINI_API_KEY));
 response=await worker.fetch(request('/api/select',{role:'single_analyst',evidence:[item],seed:42},session.token),env);
 assert.equal((await response.json()).cached,true);assert.equal(calls,1);
 response=await worker.fetch(request('/api/select',{role:'arbitrary_chat',evidence:[item],seed:42},session.token),env);
 assert.equal(response.status,400);assert.equal(calls,1);
 response=await worker.fetch(request('/api/select',{role:'single_analyst',evidence:[{...item,value:'not numeric'}],seed:42},session.token),env);
 assert.equal(response.status,400);assert.equal(calls,1);
 response=await worker.fetch(request('/api/select',{role:'single_analyst',evidence:[item],seed:42},session.token+'tampered'),env);
 assert.equal(response.status,401);
 response=await worker.fetch(request('/api/session',{},null,'fixture-other',{'origin':'https://unrelated.example'}),env);
 assert.equal(response.status,403);
 globalThis.fetch=async()=>Response.json({error:{message:'Busy '+env.GEMINI_API_KEY}},{status:503});
 response=await worker.fetch(request('/api/select',{role:'single_analyst',evidence:[item],seed:43},session.token),env);
 assert.equal(response.status,503);assert.ok(!(await response.text()).includes(env.GEMINI_API_KEY));
 console.log('Worker checks passed: anonymous scoped sessions, key isolation, request validation, tamper rejection, caching and provider errors.');
} finally {globalThis.fetch=originalFetch;}
