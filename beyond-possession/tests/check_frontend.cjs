const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[2],'utf8');
const report=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
async function check(provider='shared',configured=true){
 const elements=new Map(),requests=[];
 function get(id){if(!elements.has(id))elements.set(id,{value:'',textContent:'',innerHTML:'',files:[],hidden:false,disabled:false,listeners:{},
  addEventListener(name,callback){this.listeners[name]=callback;},dispatchEvent(event){this.listeners[event.type]?.(event);},
  getBoundingClientRect:()=>({top:1000}),getClientRects:()=>[{}],classList:{toggle(){}}});return elements.get(id);}
 for(const [id,value] of Object.entries({source:'demo',system:'single',window:'15',writing:'free',critic:'hybrid','tci-mode':'fixed'}))get(id).value=value;
 const model={weights:{territory_share:.2,progressive_pass_rate:.3,completion_rate:.1,shot_share:.4},training_scores:{learned:{matches:3}}};
 let apiError=false;
 const context=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[],documentElement:{scrollHeight:2000}},
  window:{innerHeight:800,scrollY:0,addEventListener(){}},Event:class{constructor(type){this.type=type;}},setTimeout,clearTimeout,
  fetch:async(url,options)=>{requests.push({url,options});
   if(url==='/api/config')return {json:async()=>({token:'fixture-token',preferred_backend:provider,gemini_configured:configured,match_catalog_available:false})};
   if(url.endsWith('-status'))return {ok:true,json:async()=>({available:true,provider:'Shared Google Gemini',model:'fixture'})};
   if(url==='/api/train-tci')return {ok:true,json:async()=>({model,test:{fixed:{mse:.2},learned:{mse:.1,matches:1}}})};
   if(url==='/api/analyze')return {ok:!apiError,json:async()=>apiError?{error:'Fixture upstream quota'}:{report,markdown:'Fixture report'}};
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
 let payload=JSON.parse(requests.at(-1).options.body);
 assert.equal(payload.generation_mode,provider==='offline'?'template':'free');
 assert.equal(payload.critic_mode,provider==='offline'?'numeric':'hybrid');
 assert.equal(get('results').hidden,false);assert.match(get('tci-note').textContent,/Original fixed/);
 assert.match(get('message').textContent,/Critic:/);
 apiError=true;await context.run();assert.match(get('message').textContent,/Fixture upstream quota/);assert.equal(get('run').disabled,false);apiError=false;
 get('training-events').files=[{name:'training.json',size:10,text:async()=>'[]'}];
 get('test-events').files=[{name:'test.csv',size:10,text:async()=>'fixture'}];
 await get('train-tci').onclick();
 payload=JSON.parse(requests.at(-1).options.body);assert.equal(payload.training_json,'[]');assert.equal(payload.test_csv,'fixture');
 assert.equal(get('model-download').disabled,false);assert.equal(get('tci-mode').value,'learned');assert.match(get('training-status').textContent,/Validation was not opened/);
 assert.match(get('training-results').innerHTML,/0.1000/);
 await context.run();payload=JSON.parse(requests.at(-1).options.body);assert.equal(payload.tci_model.weights.shot_share,.4);
 get('training-events').files=[];await get('train-tci').onclick();assert.match(get('training-status').textContent,/Choose a Training/);
}
(async()=>{for(const mode of ['shared','gemini','groq','ollama','offline'])await check(mode);await check('gemini',false);
 console.log('Frontend execution passed: approach defaults, offline controls, provider routing, report rendering, errors, Training/Test uploads and learned-model reuse.');
})().catch(error=>{console.error(error);process.exitCode=1;});
