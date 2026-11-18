const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[2],'utf8');
const report=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
async function check(provider='shared',configured=true){
 const elements=new Map(),requests=[];let reportForCase=structuredClone(report),outlookCalls=0,release=null,deferNext=false,runNumber=0;
 function get(id){if(!elements.has(id))elements.set(id,{value:'',textContent:'',innerHTML:'',files:[],hidden:false,disabled:false,checked:false,listeners:{},
  addEventListener(name,callback){this.listeners[name]=callback;},dispatchEvent(event){this.listeners[event.type]?.(event);},
  getBoundingClientRect:()=>({top:1000}),getClientRects:()=>[{}],classList:{toggle(){}}});return elements.get(id);}
 for(const [id,value] of Object.entries({source:'demo',system:'single',window:'15',writing:'free',critic:'hybrid','tci-mode':'fixed'}))get(id).value=value;
 const model={weights:{territory_share:.2,progressive_pass_rate:.3,completion_rate:.1,shot_share:.4},training_scores:{learned:{matches:3}}};
 get('auto-outlook').checked=true;let apiError=false;
 const context=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[],documentElement:{scrollHeight:2000}},
  window:{innerHeight:800,scrollY:0,addEventListener(){}},Event:class{constructor(type){this.type=type;}},setTimeout,clearTimeout,
  fetch:async(url,options)=>{requests.push({url,options});
   if(url==='/api/config')return {json:async()=>({token:'fixture-token',preferred_backend:provider,gemini_configured:configured,match_catalog_available:false})};
   if(url.endsWith('-status'))return {ok:true,json:async()=>({available:true,provider:'Shared Google Gemini',model:'fixture'})};
   if(url==='/api/train-tci')return {ok:true,json:async()=>({model,test:{fixed:{mse:.2},learned:{mse:.1,matches:1}}})};
   if(url==='/api/analyze'){
    runNumber++;reportForCase=structuredClone(report);reportForCase.analysis.backend=provider;
    return {ok:!apiError,json:async()=>apiError?{error:'Fixture upstream quota'}:{report:reportForCase,report_handle:'fixture-'+runNumber,markdown:'Fixture report'}};
   }
   if(url==='/api/outlook'){
    outlookCalls++;const payload=JSON.parse(options.body);
    if(deferNext){deferNext=false;await new Promise(resolve=>{release=resolve;});}
    const outlook=reportForCase.outlooks.find(o=>o.team===payload.team&&o.selected_interval_start===payload.interval_start);
    outlook.ai={summary:'If patterns continue, '+outlook.team+' could retain its recent approach.',model:'fixture-coach',verification:'Citations checked.',recommendations:outlook.priorities.slice(0,2).map(p=>({...p,rationale:'Consider <script>unsafe text</script> as untrusted model text.'}))};
    return {ok:true,json:async()=>({outlook,outlooks:reportForCase.outlooks,markdown:'Outlook export for '+outlook.team})};
   }
   throw Error('Unexpected request '+url);
  }});
 vm.runInContext(source,context);await new Promise(resolve=>setImmediate(resolve));
 assert.equal(requests.length,1,'Opening the dashboard must never generate AI text');
 assert.equal(get('backend').value,provider);
 if(provider==='gemini'&&!configured)assert.match(get('message').textContent,/GEMINI_API_KEY/);
 if(provider==='offline'){assert.equal(get('writing').value,'template');assert.equal(get('critic').value,'numeric');assert.equal(get('critic').disabled,true);}
 else{await get('check-ollama').onclick();assert.equal(requests.at(-1).url,'/api/'+provider+'-status');}
 get('team').value=report.summary.teams[0];get('interval').value='0';
 await context.run();
 let payload=JSON.parse(requests.findLast(r=>r.url==='/api/analyze').options.body);
 assert.equal(payload.generation_mode,provider==='offline'?'template':'free');
 assert.equal(payload.critic_mode,provider==='offline'?'numeric':'hybrid');
 assert.equal(get('results').hidden,false);assert.match(get('tci-note').textContent,/Original fixed/);
 assert.match(get('budget-label').textContent,/Critic:/);
 assert.ok(get('source-help').textContent);assert.ok(get('backend-help').textContent);
 assert.ok(get('metric-glossary').innerHTML.includes('Tactical control index'));
 assert.ok(get('outlook-projections').innerHTML.includes('Estimated next phase'));
 if(provider!=='offline'){
  assert.equal(outlookCalls,1,'A successful analysis automatically expands the selected team outlook');
  assert.match(get('outlook-kind').textContent,/AI EXPLANATION/);
  assert.match(get('outlook-priorities').innerHTML,/&lt;script&gt;/);assert.ok(!get('outlook-priorities').innerHTML.includes('<script>'));
  const before=requests.length;get('team').value=report.summary.teams[1];context.render();
  assert.equal(requests.length,before,'Changing a filter alone does not spend an AI call');
  assert.match(get('outlook-title').textContent,new RegExp(report.summary.teams[1]));
  deferNext=true;const pending=get('outlook-ai').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  get('team').value=report.summary.teams[0];context.render();release();await pending;
  assert.match(get('outlook-title').textContent,new RegExp(report.summary.teams[0]),'A late response must not replace the selected team');
  assert.ok(get('outlook-summary').textContent.includes(report.summary.teams[0]));
 }
 else assert.equal(outlookCalls,0);
 apiError=true;await context.run();assert.match(get('message').textContent,/Fixture upstream quota/);assert.equal(get('run').disabled,false);apiError=false;
 get('training-events').files=[{name:'training.json',size:10,text:async()=>'[]'}];
 get('test-events').files=[{name:'test.csv',size:10,text:async()=>'fixture'}];
 await get('train-tci').onclick();
 payload=JSON.parse(requests.at(-1).options.body);assert.equal(payload.training_json,'[]');assert.equal(payload.test_csv,'fixture');
 assert.equal(get('model-download').disabled,false);assert.equal(get('tci-mode').value,'learned');assert.match(get('training-status').textContent,/Validation was not opened/);
 assert.match(get('training-results').innerHTML,/0.1000/);
 await context.run();payload=JSON.parse(requests.findLast(r=>r.url==='/api/analyze').options.body);assert.equal(payload.tci_model.weights.shot_share,.4);
 get('training-events').files=[];await get('train-tci').onclick();assert.match(get('training-status').textContent,/Choose a Training/);
}
(async()=>{for(const mode of ['shared','gemini','groq','ollama','offline'])await check(mode);await check('gemini',false);
 console.log('Frontend execution passed: help text, default AI outlook, team filters, late response isolation, escaped AI text, offline controls, errors and training/model reuse.');
})().catch(error=>{console.error(error);process.exitCode=1;});
