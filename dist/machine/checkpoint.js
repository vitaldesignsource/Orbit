/* Exact pause/resume serialization for the sparse Qwenputer CPU circuit.
 * Recurrent matrices and convolution histories are saved, not an emulated RAM image.
 * Scratch projections are recomputed by the next token. Callers must pause execution
 * while capturing. Checksums detect corruption; they do not authenticate a state file.
 */
import {NeuralRV32} from './neural.js';

const FORMAT='orbit-neural-checkpoint-v1';
const MAX_OUTPUT=16*1024*1024;
const MAX_STATE_BYTES=128*1024*1024;
const fingerprints=new WeakMap();
const encoder=new TextEncoder();
const K=new Uint32Array([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
const rotr=(v,n)=>(v>>>n)|(v<<(32-n));
function sha256(bytes){
 const h=new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
 const size=Math.ceil((bytes.length+9)/64)*64,data=new Uint8Array(size);data.set(bytes);data[bytes.length]=128;
 const view=new DataView(data.buffer),bits=bytes.length*8;view.setUint32(size-8,Math.floor(bits/0x100000000),false);view.setUint32(size-4,bits>>>0,false);
 const w=new Uint32Array(64);
 for(let offset=0;offset<size;offset+=64){
  for(let i=0;i<16;i++)w[i]=view.getUint32(offset+4*i,false);
  for(let i=16;i<64;i++){const a=w[i-15],b=w[i-2];w[i]=(w[i-16]+(rotr(a,7)^rotr(a,18)^(a>>>3))+w[i-7]+(rotr(b,17)^rotr(b,19)^(b>>>10)))>>>0;}
  let[a,b,c,d,e,f,g,j]=h;
  for(let i=0;i<64;i++){const t1=(j+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+K[i]+w[i])>>>0,t2=((rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c)))>>>0;j=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;}
  h[0]=(h[0]+a)>>>0;h[1]=(h[1]+b)>>>0;h[2]=(h[2]+c)>>>0;h[3]=(h[3]+d)>>>0;h[4]=(h[4]+e)>>>0;h[5]=(h[5]+f)>>>0;h[6]=(h[6]+g)>>>0;h[7]=(h[7]+j)>>>0;
 }
 return Array.from(h,n=>n.toString(16).padStart(8,'0')).join('');
}
function hashJSON(value){return sha256(encoder.encode(JSON.stringify(value)));}
export function fingerprintCircuit(circuit){let hash=fingerprints.get(circuit);if(!hash){hash=hashJSON(circuit);fingerprints.set(circuit,hash);}return hash;}
function fail(message){throw new Error('Invalid neural checkpoint: '+message);}
function integer(v,min,max,label){if(!Number.isSafeInteger(v)||v<min||v>max)fail(label);return v;}
function object(value,label){if(!value||typeof value!=='object'||Array.isArray(value))fail(label);return value;}
function base64(bytes){
 if(typeof Buffer!=='undefined')return Buffer.from(bytes.buffer,bytes.byteOffset,bytes.byteLength).toString('base64');
 let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(binary);
}
const LITTLE_ENDIAN=new Uint8Array(new Uint32Array([1]).buffer)[0]===1;
function packed(values,type){
 let bytes;
 if(type==='u8')bytes=new Uint8Array(values.buffer,values.byteOffset,values.byteLength);
 else if(LITTLE_ENDIAN)bytes=new Uint8Array(values.buffer,values.byteOffset,values.byteLength);
 else{const size=type==='f64le'?8:4;bytes=new Uint8Array(values.length*size);const view=new DataView(bytes.buffer);for(let i=0;i<values.length;i++)type==='f64le'?view.setFloat64(i*size,values[i],true):view.setUint32(i*size,values[i],true);}
 return{type,length:values.length,data:base64(bytes)};
}
function checkPacked(raw,type,length,label){
 object(raw,label);if(raw.type!==type||raw.length!==length||typeof raw.data!=='string')fail(label+' shape/type');
 const bytes=length*(type==='f64le'?8:type==='u32le'?4:1),expected=Math.ceil(bytes/3)*4;
 if(raw.data.length!==expected||!(/^[A-Za-z0-9+/]*={0,2}$/).test(raw.data))fail(label+' encoding');
 const padding=bytes%3===0?'':bytes%3===1?'==':'=';if(padding&&!raw.data.endsWith(padding))fail(label+' padding');
 return bytes;
}
function unpack(raw,type,length,label){
 checkPacked(raw,type,length,label);let bytes;
 if(typeof Buffer!=='undefined')bytes=Uint8Array.from(Buffer.from(raw.data,'base64'));
 else{const binary=atob(raw.data);bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);}
 if(bytes.length!==length*(type==='f64le'?8:type==='u32le'?4:1))fail(label+' byte length');
 if(type==='u8')return bytes;
 const result=type==='f64le'?new Float64Array(length):new Uint32Array(length),view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
 for(let i=0;i<length;i++){result[i]=type==='f64le'?view.getFloat64(i*8,true):view.getUint32(i*4,true);if(type==='f64le'&&!Number.isFinite(result[i]))fail(label+' nonfinite value');}
 return result;
}
function rawHeads(circuit){
 if(!circuit||!Array.isArray(circuit.nodes)||circuit.nodes.length>50000)fail('circuit nodes');
 const dimensions=typeof circuit.dims==='number'?circuit.dims:circuit.dims?.length;integer(dimensions,1,16384,'circuit dimensions');
 let total=0;const heads=[];
 circuit.nodes.forEach((node,index)=>{if((node.type||(node.terms?'ffn':'head'))==='ffn')return;
  const columns=node.nCols??node.n_cols??128;integer(columns,1,128,'head columns');
  const vheads=node.vheads||node.vh||[node,...(node.vx||[])];if(!Array.isArray(vheads)||vheads.length<1||vheads.length>32)fail('value head count');
  const count=ch=>Array.isArray(ch)?ch.length:Object.keys(ch||{}).length;
  const history=3*(count(node.q)+count(node.k)+vheads.reduce((n,v)=>n+count(v.v),0));
  total+=vheads.length*(128*columns*8+128)+history*8;
  if(total>MAX_STATE_BYTES)fail('circuit state is too large');heads.push({index,name:node.name??'',columns,vheads:vheads.length,history});
 });
 if(heads.length>512)fail('too many heads');return{dimensions,heads};
}
function allChannels(node){return[...node.q,...node.k,...node.vheads.flatMap(v=>v.v)].map(([,ch])=>ch);}
function captureHeads(cpu){return cpu.nodes.filter(node=>node.type==='head').map(node=>({index:node.index,name:node.name??'',columns:node.nCols,heads:node.vheads.map(v=>({state:packed(v.state,'f64le'),present:packed(v.present,'u8')})),history:packed(Float64Array.from(allChannels(node).flatMap(ch=>Array.from(ch.history))),'f64le')}));}
function cleanSnapshot(snapshot,headSpecs,vocab){
 object(snapshot,'initial snapshot');const source=snapshot.states||snapshot.heads||snapshot.init||{};object(source,'initial state map');
 const byKey=new Map(headSpecs.flatMap(s=>[[String(s.index),s],[s.name,s]]));
 if(Object.keys(source).length>headSpecs.length)fail('initial snapshot head count');
 const states={};
 for(const[key,values]of Object.entries(source)){
  const spec=byKey.get(key);if(!spec||Object.hasOwn(states,spec.index)||!Array.isArray(values)||values.length>spec.vheads)fail('initial head');
  states[spec.index]=values.map(rows=>{
   object(rows,'initial rows');const entries=Object.entries(rows);if(entries.length>128)fail('initial row count');const result={};
   for(const[rowText,values]of entries){const row=Number(rowText);integer(row,0,127,'initial row');if(!Array.isArray(values)||values.length>spec.columns)fail('initial row columns');
    let pairs;if(values.length===spec.columns&&typeof values[0]==='number')pairs=values.map((v,i)=>[i,v]);else pairs=values;
    const seen=new Set();result[row]=[];
    for(const pair of pairs){if(!Array.isArray(pair)||pair.length!==2)fail('initial value');const[col,value]=pair;integer(col,0,spec.columns-1,'initial column');if(seen.has(col)||typeof value!=='number'||!Number.isFinite(value))fail('initial value');seen.add(col);if(value!==0)result[row].push([col,value]);}
   }return result;
  });
 }
 const startToken=integer(snapshot.startToken??vocab-2,0,vocab-1,'initial token'),pc=integer(snapshot.pc??0x40000,0,0xffffffff,'initial pc');
 return{format:'qwenputer-state-v1',program:typeof snapshot.program==='string'?snapshot.program.slice(0,256):'restored',states,startToken,pc};
}
function vocabulary(circuit){const families=circuit.familySizes||circuit.meta?.familySizes||[['V',32768],['F0',32768],['F1',32768],['F2',4],['J',65536],['R',65536],['BYTE',512],['FLT',8],['HEX',16],['QS',16],['QD',128],['BD',256]],controls=circuit.controls||circuit.meta?.controls||['NT','ST0','ST1','SS0','SS1','HALT','GO','SC'];return families.reduce((n,f)=>n+f[1],0)+controls.length;}

export function captureNeural(cpu){
 if(!(cpu instanceof NeuralRV32))throw new Error('A NeuralRV32 instance is required');
 const geometry=rawHeads(cpu.circuit),vocab=cpu.vocab;
 if(cpu.output.length>MAX_OUTPUT)throw new Error('CPU output is too large for a checkpoint');
 const payload={format:FORMAT,circuit:fingerprintCircuit(cpu.circuit),dimensions:cpu.dimCount,
  initial:cleanSnapshot(cpu.snapshot||{},geometry.heads,vocab),heads:captureHeads(cpu),residual:packed(cpu.residual,'f64le'),registers:packed(cpu.regs,'u32le'),
  execution:{tokens:cpu.tokens,instructions:cpu.instructions,lastToken:cpu.lastToken,nextToken:cpu.nextToken,pc:cpu.pc,state:cpu.state,exitCode:cpu.exitCode,faultCode:cpu.faultCode,output:cpu.output,outputBytes:packed(Uint8Array.from(cpu.outputBytes),'u8'),trace:Array.from(cpu.trace)}};
 return{...payload,checksum:hashJSON(payload)};
}

export function restoreNeural(circuit,checkpoint,options={}){
 object(checkpoint,'root');if(checkpoint.format!==FORMAT)fail('format/version');
 const geometry=rawHeads(circuit),vocab=vocabulary(circuit);
 if(checkpoint.circuit!==fingerprintCircuit(circuit)||checkpoint.dimensions!==geometry.dimensions)fail('circuit fingerprint/dimensions');
 if(!Array.isArray(checkpoint.heads)||checkpoint.heads.length!==geometry.heads.length)fail('head count');
 for(let i=0;i<geometry.heads.length;i++){
  const spec=geometry.heads[i],head=checkpoint.heads[i];object(head,'head');
  if(head.index!==spec.index||head.name!==spec.name||head.columns!==spec.columns||!Array.isArray(head.heads)||head.heads.length!==spec.vheads)fail('head geometry');
  checkPacked(head.history,'f64le',spec.history,'head history');
  for(const value of head.heads){object(value,'value head');checkPacked(value.state,'f64le',128*spec.columns,'recurrent matrix');checkPacked(value.present,'u8',128,'row presence');}
 }
 checkPacked(checkpoint.residual,'f64le',geometry.dimensions,'residual');checkPacked(checkpoint.registers,'u32le',32,'registers');
 const execution=object(checkpoint.execution,'execution');integer(execution.tokens,0,Number.MAX_SAFE_INTEGER,'token count');integer(execution.instructions,0,Number.MAX_SAFE_INTEGER,'instruction count');integer(execution.nextToken,0,vocab-1,'next token');if(execution.lastToken!==null)integer(execution.lastToken,0,vocab-1,'last token');integer(execution.pc,0,0xffffffff,'pc');
 if(!['ready','running','halted','fault'].includes(execution.state))fail('execution state');if(execution.exitCode!==null)integer(execution.exitCode,0,0xffffffff,'exit code');
 if(execution.faultCode!==null&&!(Number.isInteger(execution.faultCode)&&execution.faultCode>=0&&execution.faultCode<=7)&&!(typeof execution.faultCode==='string'&&execution.faultCode.length<=256))fail('fault code');
 if(typeof execution.output!=='string'||execution.output.length>MAX_OUTPUT)fail('output');checkPacked(execution.outputBytes,'u8',execution.output.length,'output bytes');
 if(!Array.isArray(execution.trace)||execution.trace.length>200)fail('trace');for(const token of execution.trace)integer(token,0,vocab-1,'trace token');
 const initial=cleanSnapshot(checkpoint.initial,geometry.heads,vocab);
 const {checksum,...payload}=checkpoint;if(typeof checksum!=='string'||checksum.length!==64||hashJSON(payload)!==checksum)fail('checksum mismatch');
 // Allocate the target CPU only after geometry, sizes, tokens, and checksum pass.
 const cpu=new NeuralRV32(circuit,options);cpu.loadSnapshot(initial);
 for(let i=0;i<geometry.heads.length;i++){
  const spec=geometry.heads[i],saved=checkpoint.heads[i],node=cpu.nodes[spec.index];
  const history=unpack(saved.history,'f64le',spec.history,'head history');let at=0;
  for(const channel of allChannels(node)){channel.history.set(history.subarray(at,at+3));at+=3;}
  saved.heads.forEach((v,j)=>{node.vheads[j].state.set(unpack(v.state,'f64le',128*spec.columns,'recurrent matrix'));const present=unpack(v.present,'u8',128,'row presence');if(present.some(n=>n>1))fail('row presence value');node.vheads[j].present.set(present);});
 }
 cpu.residual.set(unpack(checkpoint.residual,'f64le',geometry.dimensions,'residual'));cpu.regs.set(unpack(checkpoint.registers,'u32le',32,'registers'));
 for(const key of['tokens','instructions','lastToken','nextToken','pc','state','exitCode','faultCode','output'])cpu[key]=execution[key];
 cpu.outputBytes=Array.from(unpack(execution.outputBytes,'u8',execution.output.length,'output bytes'));cpu.trace=execution.trace.slice();
 for(let i=0;i<cpu.outputBytes.length;i++)if(cpu.output.charCodeAt(i)!==cpu.outputBytes[i])fail('output byte mismatch');
 return cpu;
}
