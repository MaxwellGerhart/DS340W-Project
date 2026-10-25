'use strict';
const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let sessionToken='',current=null,reportMarkdown='',matchCatalog=null,savedMatchesAvailable=false,setupMessage='';
const nice=k=>k.replaceAll('_',' ');
const metricValue=(k,v)=>v==null?'Unavailable':k==='tci'?v.toFixed(2):k.endsWith('_m')?v.toFixed(2)+' m':k==='pressure_per100'?v.toFixed(2):(v*100).toFixed(1)+'%';
const svg=(content,w=640,h=300)=>`<svg viewBox="0 0 ${w} ${h}" role="img">${content}</svg>`;
function pitch(content){return svg(`<rect x="0" y="0" width="640" height="410" fill="#1b2a20" rx="4"/><path d="M320 10V400 M10 10H630V400H10Z M10 100H110V310H10 M630 100H530V310H630" fill="none" stroke="#63795b" stroke-width="1"/><circle cx="320" cy="205" r="52" fill="none" stroke="#63795b"/>${content}`,640,410)}
function timeline(){
 const s=current.summary,colors=['#d4ed88','#e6a883'];let content='';
 for(let v=0;v<=100;v+=25){let y=235-v*2;content+=`<path d="M40 ${y}H610" stroke="#304034" stroke-dasharray="3 5"/><text x="8" y="${y+4}" fill="#91a28e" font-size="10">${v}</text>`;}
 s.teams.forEach((team,i)=>{let segments=[],segment=[];for(const w of s.windows.filter(w=>w.team===team)){
  if(w.metrics.tci==null){if(segment.length)segments.push(segment);segment=[];continue}
  const x=40+((w.start+w.end)/2)/s.duration_seconds*570,y=235-w.metrics.tci*2;
  segment.push(`${x},${y}`);content+=`<circle cx="${x}" cy="${y}" r="4" fill="${colors[i]}"><title>${esc(team)}: ${w.metrics.tci.toFixed(2)}</title></circle>`;
 }if(segment.length)segments.push(segment);segments.forEach(points=>content+=`<polyline points="${points.join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="2.5"/>`)});
 for(let m=0;m<=s.duration_seconds/60;m+=15){let x=40+m*60/s.duration_seconds*570;content+=`<text x="${x}" y="262" text-anchor="middle" fill="#91a28e" font-size="10">${m}′</text>`;}
 $('timeline').innerHTML=svg(content,640,280);$('legend-home').textContent=s.teams[0];$('legend-away').textContent=s.teams[1];
}
function render(){
 if(!current)return;let s=current.summary,a=current.analysis,team=$('team').value,start=Number($('interval').value);
 const w=s.windows.find(w=>w.team===team&&w.start===start);if(!w)return;
 const m=w.metrics;
 $('metrics').innerHTML=[['tci','Tactical control index','Experimental score out of 100'],['action_share','Action share','Recorded actions, not possession time'],['progressive_pass_rate','Progressive passing','Completed progressive / attempted'],['central_control','Central spatial control',w.n_frames+' full snapshots in interval']].map(([key,title,note])=>`<div class="metric"><div class="label">${title}</div><div class="value">${metricValue(key,m[key])}</div><small>${note}</small></div>`).join('');
 $('score-context').textContent=`Score at interval start: ${w.contexts.score_at_start.join(' : ')} · ${w.n_events} team events`;
 $('components').innerHTML=Object.entries(s.tci_weights).map(([k,v])=>`<div class="component"><div class="component-title">${nice(k)} <span>${metricValue(k,m[k])}</span></div><div class="bar"><i style="width:${Math.max(0,Math.min(100,(m[k]||0)*100))}%"></i></div><small>${Math.round(v*100)}% of index</small></div>`).join('');
 let net=w.network,nodes=Object.fromEntries(net.nodes.map(n=>[n.id,n]));
 let lines=net.edges.map(e=>{let x=nodes[e.source],y=nodes[e.target];return `<line x1="${10+x.x*6.2}" y1="${400-x.y*3.9}" x2="${10+y.x*6.2}" y2="${400-y.y*3.9}" stroke="#d4ed88" stroke-opacity=".28" stroke-width="${Math.min(8,1+e.count*.7)}"><title>${esc(e.source)} → ${esc(e.target)}: ${e.count}</title></line>`}).join('');
 let points=net.nodes.map(n=>`<circle cx="${10+n.x*6.2}" cy="${400-n.y*3.9}" r="${Math.min(15,5+Math.sqrt(n.received))}" fill="#d4ed88" stroke="#17211b" stroke-width="2"><title>${esc(n.id)} · ${n.received} received</title></circle><text x="${10+n.x*6.2}" y="${384-n.y*3.9}" text-anchor="middle" fill="#edf2e7" font-size="10">${esc(n.id)}</text>`).join('');
 $('network').innerHTML=net.nodes.length?pitch(lines+points):'<p class="empty">No completed pass connections in this interval.</p>';
 $('network-label').textContent=net.nodes.length+' OBSERVED PLAYERS';
 $('surface').innerHTML=w.surface.length?pitch(w.surface.map(c=>`<rect x="${c.x*6.4-640/36}" y="${410-c.y*4.1-410/24}" width="${640/18}" height="${410/12}" fill="${c.control===1?'#d4ed88':c.control===0?'#e6a883':'#ddd'}" opacity=".4"><title>Control: ${c.control}</title></rect>`).join('')):'<p class="empty">Tracking unavailable.<br>Spatial control and team compactness require full-team snapshots.</p>';
 $('surface-note').textContent=w.surface.length?`Representative snapshot at ${(w.snapshot_time/60).toFixed(1)} minutes. Lime: selected team. Peach: opponent. Snapshot uses home-attacks-right coordinates. Summary metrics average ${w.n_frames} snapshots.`:'No spatial metrics are inferred from event locations.';
 let claims=a.claims.filter(c=>c.evidence.team===team&&c.evidence.start===start);
 $('claim-count').textContent=claims.length+' FINDINGS IN THIS INTERVAL';
 $('patterns').innerHTML=(s.patterns||[]).filter(p=>p.team===team&&p.start===start).map(p=>`<article class="claim"><div class="claim-role">Calculated comparison</div><div><p>${esc(p.text)}</p><small>${esc(p.warning)} Evidence: ${esc(p.evidence_ids.join(', '))}</small></div></article>`).join('');
 $('claims').innerHTML=claims.length?claims.map(c=>`<article class="claim"><div class="claim-role">${esc(nice(c.agent))}</div><div><p>${esc(c.text)}<span class="badge ${c.verification==='unchecked'?'unchecked':''}">${esc(c.verification)}</span></p><small>${esc(c.alternative_explanation)}</small><small>Evidence ${esc(c.evidence.evidence_id)} · ${c.evidence.sample_count} ${esc(c.evidence.sample_unit)} · Examples: ${esc(c.evidence.supporting_events.join(', '))}</small><details><summary>How this was calculated</summary><p>${esc(c.evidence.definition)}</p><p>${esc(c.review_note)}</p></details></div></article>`).join(''):'<p class="empty">No accepted observations for this selection. See critic review or choose another interval.</p>';
 $('agent-trace').innerHTML=a.trace.map(t=>`<span title="${esc(t.action)}">${esc(nice(t.agent))}</span>`).join('');
 $('rejected-label').textContent=`Critic review · ${a.rejected.length} rejected candidates across the match`;
 $('rejected').innerHTML=a.rejected.length?a.rejected.map(r=>`<p>${esc(r.agent)}: ${esc(r.reason)}</p>`).join(''):'<p>No findings were flagged. The football interpretation still needs a human review.</p>';
 $('raw-metrics').innerHTML='<table><thead><tr><th>Metric</th><th>Value</th><th>Definition</th></tr></thead><tbody>'+Object.entries(s.metric_definitions).map(([k,d])=>`<tr><td>${esc(nice(k))}</td><td>${metricValue(k,m[k])}</td><td>${esc(d)}</td></tr>`).join('')+'</tbody></table>';
}
async function run(event){
 if(event)event.preventDefault();$('run').disabled=true;$('run').textContent='Analyzing…';$('message').textContent='Calculating match statistics. Local AI can take a few minutes.';
 try{
 const source=$('source').value;let payload={source,system:$('system').value,backend:$('backend').value,window_minutes:Number($('window').value),seed:42};
 if(source==='upload'){
 const events=$('events-file').files[0];if(!events)throw Error('Choose an event CSV or JSON file.');
 if(events.size>20*1024*1024)throw Error('Event file must be below 20 MiB.');
 payload[events.name.toLowerCase().endsWith('.json')?'events_json':'events_csv']=await events.text();payload.match_id=$('match-id').value.trim();
 const tracking=$('tracking-file').files[0];if(tracking){if(tracking.size>20*1024*1024)throw Error('Tracking JSON must be below 20 MiB.');payload.tracking_json=await tracking.text()}
 }
 if(source==='matches'){await loadMatches();return;}
 const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json','X-Session-Token':sessionToken},body:JSON.stringify(payload)});
 const data=await response.json();if(!response.ok)throw Error(data.error||'Analysis failed.');
 current=data.report;reportMarkdown=data.markdown;$('results').hidden=false;$('catalog-results').hidden=true;
 $('match-title').textContent=current.summary.teams.join(' vs ');
 $('run-label').textContent=`${current.summary.match_id} / ${current.analysis.system} / ${current.analysis.backend}${current.analysis.model?' / '+current.analysis.model:''} / Run ${current.run_id}`;
 $('team').innerHTML=current.summary.teams.map(t=>`<option>${esc(t)}</option>`).join('');
 $('interval').innerHTML=current.summary.windows.filter(w=>w.team===current.summary.teams[0]).map(w=>`<option value="${w.start}">${(w.start/60).toFixed(0)}–${(w.end/60).toFixed(0)} min</option>`).join('');
 $('source-note').textContent=source==='demo'?'SYNTHETIC DATA · Seed 42 · Workflow demonstration, not a research finding':'UPLOADED DATA · Validate provider mapping and licensing before research use';
 $('message').textContent=`Ready. ${current.analysis.claims.length} observations · ${current.analysis.rejected.length} rejected · ${current.analysis.latency_seconds.toFixed(2)}s · ${(current.analysis.model_calls||[]).filter(c=>!c.cached).length} AI requests · ${(current.analysis.model_calls||[]).filter(c=>c.cached).length} cached responses. ${current.analysis.backend==='offline'?'Offline rules, no LLM called.':''}`;
 $('message').textContent+=setupMessage;timeline();render();
 }catch(err){$('message').textContent=err.message+' Previous results, if visible, remain from the previous successful run.'}
 finally{$('run').disabled=false;$('run').textContent=$('source').value==='matches'?'Load match files':'Analyze match ↗'}
}
function download(name,text,type){let url=URL.createObjectURL(new Blob([text],{type}));let a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('source').addEventListener('change',()=>{$('upload').hidden=$('source').value!=='upload';$('match-upload').hidden=$('source').value!=='matches';$('run').textContent=$('source').value==='matches'?'Load match files':'Analyze match ↗';$('message').textContent='';});
$('controls').addEventListener('submit',run);$('team').addEventListener('change',render);$('interval').addEventListener('change',render);
$('json-download').onclick=()=>current&&download('beyond-possession-report.json',JSON.stringify(current,null,2),'application/json');
$('md-download').onclick=()=>current&&download('beyond-possession-report.md',reportMarkdown,'text/markdown');
fetch('/api/config').then(r=>r.json()).then(c=>{sessionToken=c.token;setupMessage=c.ai_setup_error?' Local AI setup did not finish; built-in analysis is available. Check the VS Code terminal.':'';$('backend').value=c.preferred_backend||'ollama';if(c.preferred_backend==='groq'&&!c.groq_configured)setupMessage=' To use AI, add your Groq key to .env and restart. Built-in analysis works without a key.';savedMatchesAvailable=!!c.match_catalog_available;if(savedMatchesAvailable){$('source').value='matches';$('source').dispatchEvent(new Event('change'));return loadMatches();}$('message').textContent='Ready. Choose your data, then click Analyze match. Opening this page does not call the AI.'+setupMessage;return;}).catch(e=>{$('message').textContent=e.message});

$('check-ollama').onclick=async()=>{
 $('check-ollama').disabled=true;$('message').textContent='Checking the selected AI connection…';
 try{if($('backend').value==='offline'){$('message').textContent='Built-in analysis is ready. No API key or model needed.';return;}const isGroq=$('backend').value==='groq';const response=await fetch(isGroq?'/api/groq-status':'/api/ollama-status');if(!response.ok)throw Error('Connection check failed. Restart the updated dashboard.');const result=await response.json();$('message').textContent=result.available?`${result.provider||'Ollama'} is ready. Model: ${result.model}. ${result.note||''}`:result.error;}
 catch(error){$('message').textContent=error.message;}
 finally{$('check-ollama').disabled=false;}
};

async function loadMatches(){
 const selected=[...$('match-files').files];if(!selected.length&&!savedMatchesAvailable)throw Error('Select your match JSON files first.');
 if(selected.reduce((n,f)=>n+f.size,0)>20*1024*1024)throw Error('Keep the selected files below 20 MiB in total.');
 const files=await Promise.all(selected.map(async f=>({name:f.name,text:await f.text()})));
 const response=selected.length?await fetch('/api/combine-matches',{method:'POST',headers:{'Content-Type':'application/json','X-Session-Token':sessionToken},body:JSON.stringify({files})}):await fetch('/api/matches');
 const result=await response.json();if(!response.ok)throw Error(result.error||'Could not read the match files.');
 matchCatalog=result.catalog;const summary=matchCatalog.summary;$('catalog-results').hidden=false;$('results').hidden=true;$('source-note').textContent='YOUR MATCH DATA · '+summary.unique_matches+' matches · Match details, not event data';
 $('catalog-summary').textContent=`${summary.unique_matches} matches from ${summary.source_files} files. ${summary.identical_duplicates_removed} identical duplicates removed.`;
 renderCatalogue();
 $('message').textContent='Match files loaded. Download the combined JSON to keep a copy. Tactical analysis needs a separate event file. The first 200 matches are shown below.'+setupMessage;
}
$('catalog-download').onclick=()=>matchCatalog&&download('combined_matches.json',JSON.stringify(matchCatalog,null,2),'application/json');
const sectionLinks=[...document.querySelectorAll('nav a[href^="#"]')];
function highlightSection(){
 let active='overview';
 for(const id of ['overview','evidence','method']){const section=document.getElementById(id);if(section&&section.getBoundingClientRect().top<=window.innerHeight*.35&&section.getClientRects().length)active=id;}
 if(window.scrollY+window.innerHeight>=document.documentElement.scrollHeight-4)active='method';
 for(const link of sectionLinks){const selected=link.getAttribute('href')==='#'+active;link.classList.toggle('active',selected);if(selected)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');}
}
sectionLinks.forEach(link=>link.addEventListener('click',()=>{sectionLinks.forEach(other=>{other.classList.toggle('active',other===link);if(other===link)other.setAttribute('aria-current','location');else other.removeAttribute('aria-current');});}));
window.addEventListener('scroll',highlightSection,{passive:true});window.addEventListener('resize',highlightSection);highlightSection();

function renderCatalogue(){
 if(!matchCatalog)return;
 const query=$('catalog-search').value.trim().toLowerCase();
 const records=matchCatalog.matches.filter(m=>String(m.label??m.name??'').toLowerCase().includes(query)||String(m.wyId??m.match_id??m.id).includes(query));
 $('catalog-table').innerHTML='<p class="chart-note">'+records.length+' matching records. Showing up to 200.</p><table><thead><tr><th>Match ID</th><th>Match</th><th>Date</th></tr></thead><tbody>'+records.slice(0,200).map(m=>`<tr><td>${esc(m.wyId??m.match_id??m.id)}</td><td>${esc(m.label??m.name??'No match label')}</td><td>${esc(m.dateutc??m.date??'')}</td></tr>`).join('')+'</tbody></table>';
}
$('catalog-search').addEventListener('input',renderCatalogue);
