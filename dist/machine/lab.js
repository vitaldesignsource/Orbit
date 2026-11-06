const base = new URL('.',import.meta.url);
const url = path => new URL(path,base);
const h = s => String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => Number(n||0).toLocaleString('en-US');
const PYTHON_LIMIT=409;
let manifestPromise,graphPromise,pythonTemplatePromise;
async function getBytes(path){const r=await fetch(url(path));if(!r.ok)throw Error(`Could not load ${path} (${r.status})`);return new Uint8Array(await r.arrayBuffer());}
async function getText(path){return new TextDecoder().decode(await getBytes(path));}
function pythonBytes(source){const normalized=source.replace(/\r\n?/g,'\n');return new TextEncoder().encode(normalized.endsWith('\n')?normalized:normalized+'\n');}
async function pythonTemplate(){
 if(!pythonTemplatePromise)pythonTemplatePromise=Promise.all([getBytes('fixtures/mpy-s5.elf'),getBytes('fixtures/s5.py')]).then(([elf,script])=>{
  if(script.length!==PYTHON_LIMIT)throw Error('The embedded Python script capacity has changed.');
  let offset=-1,matches=0;
  for(let at=elf.indexOf(script[0]);at!==-1;at=elf.indexOf(script[0],at+1)){
   if(at+script.length>elf.length)continue;let equal=true;
   for(let i=0;i<script.length;i++)if(elf[at+i]!==script[i]){equal=false;break;}
   if(equal){offset=at;matches++;}
  }
  if(matches!==1)throw Error('Could not uniquely locate the embedded Python script.');
  return {elf,offset,capacity:script.length};
 }).catch(error=>{pythonTemplatePromise=null;throw error;});
 return pythonTemplatePromise;
}
async function patchPython(source){
 const bytes=pythonBytes(source);if(bytes.length>PYTHON_LIMIT)throw Error(`Python source is ${bytes.length} UTF-8 bytes; the limit is ${PYTHON_LIMIT}, including a final newline.`);
 const {elf,offset,capacity}=await pythonTemplate(),patched=elf.slice();
 // Preserve the ELF layout and the guest's fixed script size.
 patched.fill(32,offset,offset+capacity);patched.set(bytes,offset);patched[offset+capacity-1]=10;return patched;
}
async function getJSON(path){
 const r=await fetch(url(path));if(!r.ok)throw Error(`Could not load ${path} (${r.status})`);
 const bytes=new Uint8Array(await r.arrayBuffer());
 // Static hosts may already decode Content-Encoding before fetch exposes bytes.
 if(bytes[0]===0x1f&&bytes[1]===0x8b)return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
 return JSON.parse(new TextDecoder().decode(bytes));
}
function getManifest(){return manifestPromise??=getJSON('programs.json').catch(e=>{manifestPromise=null;throw e;});}
function getGraph(path='neural-circuit.json'){return graphPromise??=getJSON(path).catch(e=>{graphPromise=null;throw e;});}
export function createComputer(api){
 if(!document.querySelector('#rv-lab-css')){const link=document.createElement('link');link.rel='stylesheet';link.id='rv-lab-css';link.href=url('lab.css');document.head.append(link);}
 const root=document.createElement('div');root.className='rv-lab';
 root.innerHTML=`<header class="rv-header"><div class="rv-chipmark">▧</div><div class="rv-heading"><h1>A computer in the weights.</h1><p>RV32IM + Zba · Compiled for Qwen3.5‑4B</p></div><span class="rv-badge">11 NEURAL LAYERS</span></header><nav class="rv-tabs" aria-label="Computer views"><button class="rv-tab active" data-tab="programs">Programs</button><button class="rv-tab" data-tab="tests">ISA tests</button><button class="rv-tab" data-tab="state">Memory & registers</button><button class="rv-tab" data-tab="architecture">Inside the machine</button></nav><div class="rv-view"><div class="rv-loader"><div class="rv-loading-orbit spinning"></div><strong>Bringing the circuit into orbit.</strong><p>Loading program information. The compiled circuit runs on your device.</p></div></div>`;
 const view=root.querySelector('.rv-view');let manifest,programs=[],current=null,cpu=null,backend='neural',tab='programs',running=false,disposed=false,runId=0,loaded=false,totalStart=0,lastPaint=0,executionMs=0,consoleText='',testsRunning=false,testCancel=false,loadGeneration=0,replacementDestroy=null,loadingProgram=false,customCount=0;const testResults=new Map();
 const $=s=>root.querySelector(s);
 function print(text){consoleText+=text;const c=$('.rv-console');if(c){c.textContent=consoleText;c.scrollTop=c.scrollHeight;}}
 function metrics(){if(!cpu)return;const values={tokens:fmt(cpu.tokens),instructions:fmt(cpu.instructions),pc:'0x'+Number(cpu.pc??0).toString(16).padStart(8,'0'),state:cpu.state??'Ready',time:(executionMs/1000).toFixed(2)+' s'};for(const [key,value] of Object.entries(values)){const e=$(`[data-metric="${key}"]`);if(e)e.textContent=value;}const status=$('.rv-status');if(status){status.classList.toggle('running',running);status.lastChild.textContent=running?'Running':loaded?(cpu.state==='halted'?'Halted':cpu.state==='faulted'||cpu.state==='fault'?'Fault':'Ready'):'Not loaded';}const run=$('[data-action="run"]');if(run)run.textContent=running?'Ⅱ Pause':'▶ Run';updateEditorStatus();}
 function stop(){running=false;runId++;metrics();}
 function showError(error){stop();print('\n'+(error.message||error)+'\n');api.toast(error.message||String(error));}
 function architecture(){view.innerHTML=`<article class="rv-architecture"><h2>Real arithmetic. Neural memory.</h2><p>This computer uses Qwenputer’s compiled RV32IM + Zba circuit. The instruction logic is expressed as sparse weights in 11 Qwen3.5-compatible Gated DeltaNet and SwiGLU layers. Registers, program code, and RAM live in recurrent attention state.</p><div class="rv-arch-grid"><div class="rv-arch-card"><label>CODE ROM</label><strong>256 KB</strong><small>Predecoded RISC‑V instructions</small></div><div class="rv-arch-card"><label>READ-ONLY DATA</label><strong>128 KB</strong><small>Constants and program strings</small></div><div class="rv-arch-card"><label>RAM</label><strong>32 KB</strong><small>Registers, stack, and working data</small></div></div><div class="rv-pipeline"><strong>Program snapshot</strong><span>→</span><strong>Weighted neural layers</strong><span>→</span><strong>Next trace token</strong><span>→</span><strong>Recurrent state</strong><span>↻</span></div><h3>What is executing here?</h3><p><b>Neural circuit mode</b> evaluates the compiled circuit’s sparse weighted equations and recurrent state directly in JavaScript. It does not interpret instruction opcodes. This is the CPU circuit targeted to Qwen3.5‑4B, without the original language model’s 32 text layers or its 18.7 GB merged checkpoint.</p><p><b>Reference mode</b> is a conventional RV32 interpreter for fast comparison. Its test results are shown separately. Changing modes changes the execution engine; it never silently replaces neural execution.</p><h3>C and MicroPython</h3><p>Included programs are real compiled RISC‑V ELF binaries, with their source shown alongside. The editable MicroPython scripts run inside the actual RISC‑V guest interpreter: Apply script patches its fixed 409-byte embedded source slot, then Run executes that guest. The limit includes a final newline; only the interpreter’s built-ins and 32 KB RAM are available, without stdin. Load ELF accepts static RV32IM + Zba executables using the same memory map and ecall conventions. This is a bare-metal computer, without a Linux kernel.</p><h3>Test coverage</h3><p>The supplied official user-level ISA fixtures cover RV32UI, RV32UM, and Zba. Passing these does not certify privileged instructions, atomics, compressed instructions, floating point, or a complete RISC‑V operating system.</p><div class="rv-links"><a href="https://github.com/divagr18/qwenputer/tree/main/experimental/riscv" target="_blank" rel="noopener">Qwenputer source ↗</a><a href="https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer" target="_blank" rel="noopener">Full merged checkpoint ↗</a><a href="${h(url('../THIRD-PARTY-NOTICES.md'))}" target="_blank" rel="noopener">Third-party notices ↗</a></div></article>`;}
 async function setupProgram(p,{applyScript=false}={}){
  stop();const generation=++loadGeneration,engine=backend;
  current=p;loaded=false;loadingProgram=true;cpu=null;consoleText='';executionMs=0;renderPrograms();
  const isCurrent=()=>!disposed&&generation===loadGeneration&&current===p&&tab==='programs';
  try{
   const source=typeof p.editorSource==='string'?p.editorSource:typeof p.source==='string'?await getText(p.source):(p.code||'// Source not supplied');
   if(!isCurrent())return;
   if(p.language==='python')p.editorSource=source;
   $('.rv-code').value=source;updateEditorStatus();
   let bytes=p.elfBytes;
   if(applyScript){bytes=await patchPython(source);if(!isCurrent())return;}
   let nextCPU;
   if(engine==='neural'){
    const [{NeuralRV32},graph]=await Promise.all([import('./neural.js'),getGraph(manifest.circuit)]);
    if(!isCurrent())return;
    let snapshot;
    if(bytes){const {buildSnapshot}=await import('./snapshot.js');if(!isCurrent())return;snapshot=buildSnapshot(bytes,graph,{name:p.name});}
    else{snapshot=await getJSON(p.snapshot);if(!isCurrent())return;}
    nextCPU=new NeuralRV32(graph,{onOutput:print});nextCPU.loadSnapshot(snapshot);
   }else{
    const {RV32}=await import('./cpu.js');if(!isCurrent())return;
    bytes??=await getBytes(p.elf);if(!isCurrent())return;
    nextCPU=new RV32({memoryMap:'qwenputer',onOutput:print});nextCPU.loadELF(bytes);
   }
   if(!isCurrent())return;
   cpu=nextCPU;
   if(applyScript){p.elfBytes=bytes;p.appliedSource=source;p.customScript=true;}
   else if(p.language==='python'&&p.appliedSource==null)p.appliedSource=source;
   loaded=true;loadingProgram=false;
   print(`Orbit machine · ${engine==='neural'?'compiled Qwen circuit':'reference interpreter'}\n${p.name}${p.customScript?' · custom Python':''}\n────────────────────────────\n`);
   metrics();updateEditorStatus();
  }catch(error){if(isCurrent()){loadingProgram=false;showError(error);updateEditorStatus();}}
 }
 function updateEditorStatus(){
  if(tab!=='programs')return;
  const python=current?.language==='python',source=current?.editorSource;
  const editor=$('.rv-code');if(editor)editor.readOnly=!python||loadingProgram;
  const dirty=python&&typeof source==='string'&&source!==current.appliedSource;
  const size=python&&typeof source==='string'?pythonBytes(source).length:0;
  const status=$('[data-source-status]');
  if(status)status.textContent=python?`${typeof source!=='string'?'Loading script':dirty?'Edited script · apply before running':current.customScript?'Custom Python loaded':'Bundled Python script'} · ${size} / ${PYTHON_LIMIT} UTF-8 bytes (including final newline)`:current?.elfBytes?'Uploaded static RV32 ELF · source not included':'Precompiled example · source is read-only';
  const apply=$('[data-action="apply-script"]');
  if(apply)apply.disabled=loadingProgram||running||typeof source!=='string'||size>PYTHON_LIMIT;
  const run=$('[data-action="run"]'),step=$('[data-action="step"]');
  if(run)run.disabled=!loaded||dirty;
  if(step)step.disabled=!loaded||running||dirty;
 }
 async function loadUploadedELF(file){
  if(!file)return;
  stop();const generation=++loadGeneration;loaded=false;loadingProgram=true;metrics();updateEditorStatus();
  try{
   if(file.size>8*1024*1024)throw Error('Choose a static RV32 ELF smaller than 8 MB.');
   const bytes=new Uint8Array(await file.arrayBuffer());
   const {parseELF}=await import('./snapshot.js');
   if(disposed||generation!==loadGeneration||tab!=='programs')return;
   const image=parseELF(bytes),header=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
   if(header.getUint16(16,true)!==2)throw Error('Load a statically linked ELF executable, not a shared object.');
   if(image.flags&1)throw Error('Compressed instructions are unsupported. Compile for RV32IM + Zba.');
   const p={id:`uploaded-${++customCount}`,name:file.name||'Uploaded ELF',language:'elf',elfBytes:bytes,code:`// Uploaded static RISC-V executable\n// File: ${file.name}\n// Size: ${fmt(bytes.length)} bytes\n// Entry: 0x${image.entry.toString(16)}\n//\n// Neural mode builds a program snapshot from this ELF.\n// Reference mode executes the same binary directly.\n//\n// Required memory map:\n//   RAM   0x00000–0x07fff\n//   DROM  0x20000–0x3ffff\n//   ROM   0x40000–0x7ffff\n//\n// Use the supplied Qwenputer ecall conventions.\n// No Linux kernel, dynamic linker, or stdin is provided.\n// The original source is not embedded in this ELF.\n`};
   programs.push(p);await setupProgram(p);
  }catch(error){if(!disposed&&generation===loadGeneration&&tab==='programs'){loadingProgram=false;showError(error);updateEditorStatus();}}
 }
 function downloadProgram(event){
  if(!current?.elfBytes)return;
  event.preventDefault();const link=document.createElement('a'),objectURL=URL.createObjectURL(new Blob([current.elfBytes],{type:'application/x-elf'}));
  link.href=objectURL;link.download=current.customScript?'orbit-python.elf':current.name;link.click();setTimeout(()=>URL.revokeObjectURL(objectURL),1000);
 }
 function renderPrograms(){
  const demos=programs.filter(p=>p.language!=='test'),python=current?.language==='python';
  view.innerHTML=`<div class="rv-toolbar"><select class="rv-select" aria-label="Choose program">${demos.map(p=>`<option value="${h(p.id)}" ${p.id===current?.id?'selected':''}>${h(p.name)}</option>`).join('')}</select><select class="rv-select rv-backend" aria-label="Execution engine"><option value="neural" ${backend==='neural'?'selected':''}>Neural circuit</option><option value="reference" ${backend==='reference'?'selected':''}>Reference · fast</option></select><button class="rv-button" data-action="load-elf">Load ELF</button><input type="file" class="rv-elf-input" aria-label="Upload RISC-V ELF" accept=".elf,application/x-elf,application/octet-stream" hidden><span class="rv-spacer"></span><button class="rv-button primary" data-action="run" ${!loaded?'disabled':''}>▶ Run</button><button class="rv-button" data-action="step" ${!loaded?'disabled':''}>Step</button><button class="rv-button" data-action="reset">↺ Reset</button><span class="rv-status"><i></i><span>Not loaded</span></span></div><div class="rv-workspace"><section class="rv-source-panel"><div class="rv-panel-label"><span>${python?'EDITABLE MICROPYTHON':current?.language==='c'?'C SOURCE':current?.language==='elf'?'ELF IMAGE':'ASSEMBLY SOURCE'}</span><span>${python?'409-BYTE SCRIPT SLOT':'RV32IM ELF'}</span></div><textarea class="rv-code" aria-label="Program source" spellcheck="false" ${python?'':'readonly'} placeholder="Loading source…">${h(current?.editorSource??current?.code??'')}</textarea><div class="rv-source-footer"><span data-source-status></span>${python?'<button class="rv-button" data-action="apply-script">Apply script</button>':''}<a href="${current?.elf?h(url(current.elf)):'#'}" download class="rv-button rv-download-elf" style="text-decoration:none">ELF ↓</a></div></section><section class="rv-console-panel"><div class="rv-panel-label"><span>STANDARD OUTPUT</span><span>${backend==='neural'?'NEURAL CIRCUIT':'REFERENCE CPU'}</span></div><pre class="rv-console" role="log" aria-live="polite">${h(consoleText)}</pre><form class="rv-console-entry"><span>›</span><input aria-label="Program input" placeholder="No guest stdin · output only" disabled><button type="button" data-action="clear">Clear</button></form></section></div><div class="rv-metrics"><div class="rv-metric"><label>TRACE TOKENS</label><strong data-metric="tokens">0</strong></div><div class="rv-metric"><label>INSTRUCTIONS</label><strong data-metric="instructions">0</strong></div><div class="rv-metric"><label>PROGRAM COUNTER</label><strong data-metric="pc">—</strong></div><div class="rv-metric"><label>EXECUTION</label><strong data-metric="state">Ready</strong></div><div class="rv-metric"><label>COMPUTE TIME</label><strong data-metric="time">0.00 s</strong></div></div><div class="rv-note">${python?'Edit up to 409 UTF-8 bytes, then Apply script and Run. Real guest MicroPython; available built-ins only, 32 KB RAM, no stdin. ':current?.language==='elf'?'Uploaded ELF must use the Qwenputer ROM/RAM map and ecall conventions. ' :''}${backend==='neural'?'Executing the compiled 11-layer CPU circuit; the original 4B language model is not loaded. Neural execution can be slow, especially for MicroPython.':'Conventional instruction interpreter for comparison. This engine does not execute neural weights.'}</div>`;
  $('.rv-select').onchange=e=>{const p=programs.find(p=>p.id===e.target.value);if(p)setupProgram(p);};
  $('.rv-backend').onchange=e=>{backend=e.target.value;setupProgram(current);};
  $('[data-action="load-elf"]').onclick=()=>$('.rv-elf-input').click();
  $('.rv-elf-input').onchange=e=>loadUploadedELF(e.target.files?.[0]);
  $('.rv-download-elf').onclick=downloadProgram;
  if(python){
   $('.rv-code').oninput=e=>{stop();current.editorSource=e.target.value;updateEditorStatus();};
   $('[data-action="apply-script"]').onclick=()=>setupProgram(current,{applyScript:true});
  }
  $('[data-action="run"]').onclick=()=>running?stop():startRun();
  $('[data-action="step"]').onclick=()=>{if(!loaded||running)return;try{execute(1);metrics();}catch(error){showError(error);}};
  $('[data-action="reset"]').onclick=()=>setupProgram(current);
  $('[data-action="clear"]').onclick=()=>{consoleText='';$('.rv-console').textContent='';};metrics();updateEditorStatus();
 }
 function execute(budget){const t=performance.now();const before=cpu.output||'';cpu.run(budget);executionMs+=performance.now()-t;
 // Executors may expose buffered output instead of callbacks.
 if(cpu.output&&cpu.output!==before&&!cpu.usesOutputCallback){/* callback is the canonical path */}
 }
 function startRun(){if(!loaded||!cpu)return;if(cpu.state==='halted'||cpu.state==='faulted'||cpu.state==='fault'){api.toast('Reset this program before running it again.');return;}running=true;const id=++runId;totalStart=performance.now();metrics();
 const tick=()=>{if(!running||id!==runId||disposed)return;try{const start=performance.now();do{execute(backend==='neural'?1:25000);}while(performance.now()-start<18&&cpu.state!=='halted'&&cpu.state!=='faulted'&&cpu.state!=='fault');if(cpu.state==='halted'||cpu.state==='faulted'||cpu.state==='fault'){running=false;print(`\n────────────────────────────\n${cpu.state==='halted'?'Halted':'Fault'}${cpu.exitCode!=null?' · exit '+cpu.exitCode:''}\n`);}metrics();if(running)setTimeout(tick,0);}catch(error){showError(error);}};setTimeout(tick,0);}
 function renderTests(){const tests=programs.filter(p=>p.language==='test');const passed=[...testResults.values()].filter(r=>r.state==='pass').length;view.innerHTML=`<header class="rv-tests-head"><div class="rv-heading"><h2>Put the circuit to the test.</h2><p>Official RV32UI · RV32UM · Zba fixtures.<br>Results below come from this session’s actual execution.</p></div><span class="rv-badge">${passed} / ${tests.length} PASSED</span><button class="rv-button primary" data-action="test-all">${testsRunning?'■ Stop':'▶ Run all'}</button></header><div class="rv-toolbar"><select class="rv-select rv-test-backend" aria-label="Test execution engine" ${testsRunning?'disabled':''}><option value="neural" ${backend==='neural'?'selected':''}>Neural circuit</option><option value="reference" ${backend==='reference'?'selected':''}>Reference · fast</option></select><span class="rv-status">${testsRunning?'Tests are running…':'No result is assumed before execution.'}</span><span class="rv-spacer"></span><button class="rv-button" data-action="test-download">Results ↓</button></div><div class="rv-test-list">${tests.map(p=>{const r=testResults.get(p.id);return `<div class="rv-test-row" data-test="${h(p.id)}"><span class="rv-test-name">${h(p.name)}</span><span class="rv-test-state ${r?.state||''}">${r?`${h(r.state)} · ${h(r.engine)}${r.instructions?' · '+fmt(r.instructions)+' instr.':''}`:'Not run'}</span><button class="rv-button" data-run-test="${h(p.id)}" ${testsRunning?'disabled':''}>Run</button></div>`;}).join('')}</div><div class="rv-note">Neural tests evaluate compiled circuit weights. Reference tests evaluate a conventional CPU interpreter. This suite covers the included user-level integer fixtures, not the entire RISC‑V architecture. Upstream excludes fence_i and ma_data.</div>`;
 $('[data-action="test-all"]').onclick=()=>{if(testsRunning){testCancel=true;return;}runTests(tests);};$('.rv-test-backend').onchange=e=>{backend=e.target.value;};for(const b of root.querySelectorAll('[data-run-test]'))b.onclick=()=>runTests([programs.find(p=>p.id===b.dataset.runTest)]);$('[data-action="test-download"]').onclick=()=>{const blob=new Blob([JSON.stringify({engine:backend,results:[...testResults.entries()],generated:new Date().toISOString()},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='orbit-riscv-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};}
 async function runTests(tests){if(testsRunning)return;stop();testsRunning=true;testCancel=false;const engine=backend;renderTests();for(const p of tests){if(testCancel||disposed)break;testResults.set(p.id,{state:'running',engine});if(tab==='tests')renderTests();let count=0;try{let m;if(engine==='neural'){const [{NeuralRV32},graph,snapshot]=await Promise.all([import('./neural.js'),getGraph(manifest.circuit),getJSON(p.snapshot)]);m=new NeuralRV32(graph);m.loadSnapshot(snapshot);}else{const [{RV32},r]=await Promise.all([import('./cpu.js'),fetch(url(p.elf))]);m=new RV32({memoryMap:'qwenputer'});m.loadELF(new Uint8Array(await r.arrayBuffer()));}const start=performance.now();while(!['halted','fault','faulted'].includes(m.state)&&count<1000000&&!testCancel&&!disposed){const t=performance.now();do{m.run(engine==='neural'?1:10000);count+=engine==='neural'?1:10000;}while(performance.now()-t<20&&!['halted','fault','faulted'].includes(m.state)&&count<1000000);await new Promise(resolve=>setTimeout(resolve,0));}const pass=m.state==='halted'&&(m.exitCode===0||m.exitCode===1&&p.tohost===true);testResults.set(p.id,{state:testCancel?'stopped':pass?'pass':'fail',engine,instructions:m.instructions,tokens:m.tokens,exitCode:m.exitCode,timeMs:Math.round(performance.now()-start),machineState:m.state});}catch(error){testResults.set(p.id,{state:'fail',engine,error:error.message});}if(tab==='tests')renderTests();}testsRunning=false;if(tab==='tests')renderTests();}
 function renderState(){
 if(!cpu){view.innerHTML='<div class="rv-loader"><strong>No program loaded yet.</strong><p>Open Programs and choose a machine image first.</p></div>';return;}
 const names=['zero','ra','sp','gp','tp','t0','t1','t2','s0','s1','a0','a1','a2','a3','a4','a5','a6','a7','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','t3','t4','t5','t6'];
 view.innerHTML=`<article class="rv-state"><h2>The state of your machine.</h2><p>Execution is paused while you inspect. ${backend==='neural'?'Values are read from the recurrent head matrices.':'Values are read from the reference interpreter.'}</p><div class="rv-registers">${Array.from(cpu.regs??new Uint32Array(32)).map((v,i)=>`<div class="rv-register"><span>x${i} / ${names[i]}</span><strong>${Number(v).toString(16).padStart(8,'0')}</strong></div>`).join('')}</div><h2>RAM</h2><div class="rv-memory-controls"><label>Start address <input aria-label="Memory start address" value="0x0100" spellcheck="false"></label><button class="rv-button" data-memory-view>View</button><span>32 KB address space · 128 bytes shown</span></div><div class="rv-memory-error" role="status"></div><pre class="rv-memory"></pre><h2 style="margin-top:24px">Recent trace tokens</h2><div class="rv-trace">${cpu.trace?.length?cpu.trace.slice(-40).map(t=>h(cpu.tokenName?.(t)??t)).join(' · '):'Trace tokens are available in neural circuit mode.'}</div></article>`;
 const renderMemory=()=>{const input=$('.rv-memory-controls input').value.trim(),address=Number(input);if(!Number.isInteger(address)||address<0||address>32768-128){$('.rv-memory-error').textContent='Choose an address between 0 and 0x7f80.';return;}$('.rv-memory-error').textContent='';let text='';for(let row=0;row<8;row++){const pos=address+row*16,bytes=Array.from({length:16},(_,i)=>cpu.read8(pos+i));text+=pos.toString(16).padStart(4,'0')+'  '+bytes.map(b=>b.toString(16).padStart(2,'0')).join(' ')+'  '+bytes.map(b=>b>=32&&b<127?String.fromCharCode(b):'.').join('')+'\n';}$('.rv-memory').textContent=text;};$('[data-memory-view]').onclick=renderMemory;$('.rv-memory-controls input').onkeydown=e=>{if(e.key==='Enter')renderMemory();};renderMemory();
 }
 function switchTab(name){stop();loadGeneration++;if(tab==='tests'&&name!=='tests')testCancel=true;tab=name;for(const b of root.querySelectorAll('[data-tab]'))b.classList.toggle('active',b.dataset.tab===name);if(name==='architecture')architecture();else if(name==='state')renderState();else if(name==='tests')renderTests();else if(current)setupProgram(current);}
 for(const b of root.querySelectorAll('[data-tab]'))b.onclick=()=>switchTab(b.dataset.tab);
 getManifest().then(m=>{if(disposed)return;manifest=m;programs=m.programs;current=programs.find(p=>p.id===m.defaultProgram)||programs.find(p=>p.language==='c')||programs.find(p=>p.language!=='test');if(!current)throw Error('No programs available');setupProgram(current);}).catch(error=>{view.innerHTML=`<div class="rv-loader"><strong>The circuit could not load.</strong><p>${h(error.message)}</p><button class="rv-button" data-retry>Try again</button></div>`;$('[data-retry]').onclick=()=>{const replacement=createComputer(api);replacementDestroy=replacement.destroy;root.replaceWith(replacement.element);};});
 return {element:root,title:'Neural Computer · Qwen3.5 RV32',destroy:()=>{disposed=true;loadGeneration++;stop();testCancel=true;replacementDestroy?.();}};
}
