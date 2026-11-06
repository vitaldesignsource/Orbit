import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {NeuralRV32} from '../dist/machine/neural.js';
import {captureNeural,restoreNeural,fingerprintCircuit} from '../dist/machine/checkpoint.js';
const root=new URL('../dist/machine/',import.meta.url);
const json=async path=>{const bytes=await readFile(new URL(path,root));return JSON.parse(path.endsWith('.gz')?gunzipSync(bytes):bytes);};
const graph=await json('neural-circuit.json.gz');
assert.equal(fingerprintCircuit(graph),createHash('sha256').update(JSON.stringify(graph)).digest('hex'),'circuit SHA-256');
const cpu=new NeuralRV32(graph);
let restoredCount=0;
for(const [id,pauseKind]of [['c-hello',18],['rv32ui-sh','store'],['rv32um-div','division']]){
 const [snapshot,trace]=await Promise.all([json(`fixtures/${id}.snapshot.json.gz`),json(`fixtures/${id}.trace.json.gz`)]);
 cpu.loadSnapshot(snapshot);
 let pause=pauseKind;
 if(pauseKind==='store'){const end=trace.tokens.findIndex(t=>['SS0','SS1'].includes(cpu.family(t).name));assert.ok(end>3,'multi-token store exists');pause=end-2;assert.equal(cpu.family(trace.tokens[pause]).name,'V');}
 if(pauseKind==='division'){pause=trace.tokens.findIndex((t,i)=>cpu.family(t).name==='QD'&&cpu.family(trace.tokens[i+1]??-1).name==='QD');assert.ok(pause>0,'multi-digit division exists');}
 for(let i=1;i<=pause;i++)assert.equal(cpu.stepToken(),trace.tokens[i]);
 const checkpoint=captureNeural(cpu),roundtrip=JSON.parse(JSON.stringify(checkpoint));
 const before={tokens:cpu.tokens,instructions:cpu.instructions,pc:cpu.pc,output:cpu.output},expectedRegs=Array.from(cpu.regs);
 const continuation=[];while(!['halted','fault'].includes(cpu.state))continuation.push(cpu.stepToken());
 let callback='';const restored=restoreNeural(graph,roundtrip,{onOutput:s=>callback+=s});
 assert.deepEqual({tokens:restored.tokens,instructions:restored.instructions,pc:restored.pc,output:restored.output},before);
 assert.deepEqual(Array.from(restored.regs),expectedRegs);
 const resumed=[];while(!['halted','fault'].includes(restored.state))resumed.push(restored.stepToken());
 assert.deepEqual(resumed,continuation,`${id}: exact free-running continuation tokens`);
 assert.equal(restored.output,cpu.output);assert.equal(restored.exitCode,cpu.exitCode);assert.equal(restored.state,cpu.state);assert.equal(restored.instructions,cpu.instructions);assert.equal(restored.tokens,cpu.tokens);assert.deepEqual(Array.from(restored.regs),Array.from(cpu.regs));
 assert.equal(callback,restored.output.slice(before.output.length));
 restored.reset();assert.equal(restored.nextToken,trace.tokens[0]);for(let i=1;i<Math.min(8,trace.tokens.length);i++)assert.equal(restored.stepToken(),trace.tokens[i],`${id}: restored initial snapshot resets exactly`);
 if(id==='c-hello'){
  const bad=JSON.parse(JSON.stringify(roundtrip));bad.heads[0].heads[0].state.data=(bad.heads[0].heads[0].state.data[0]==='A'?'B':'A')+bad.heads[0].heads[0].state.data.slice(1);assert.throws(()=>restoreNeural(graph,bad),/checksum/);
  const oversized={...roundtrip,heads:roundtrip.heads.slice()};oversized.heads[0]={...oversized.heads[0],history:{type:'f64le',length:1000000000,data:''}};assert.throws(()=>restoreNeural(graph,oversized),/shape\/type/);
  assert.throws(()=>restoreNeural(graph,{...roundtrip,format:'orbit-neural-checkpoint-v9000'}),/format/);
  assert.throws(()=>restoreNeural(graph,{...roundtrip,circuit:'0'.repeat(64)}),/fingerprint/);
  assert.throws(()=>restoreNeural(graph,{...roundtrip,execution:{...roundtrip.execution,nextToken:-1}}),/next token/);
  const invalidInitial={...roundtrip,initial:{...roundtrip.initial,states:{[roundtrip.heads[0].index]:[{999999:[]} ]}}};assert.throws(()=>restoreNeural(graph,invalidInitial),/initial row/);
  assert.throws(()=>restoreNeural(graph,{...roundtrip,heads:[]}),/head count/);
 }
 restoredCount++;
 console.log(`PASS checkpoint ${id}: JSON round-trip, exact neural continuation (${resumed.length} tokens), reset, output/counters/registers`);
}
console.log(`Checkpoint checks passed: ${restoredCount} real neural programs, corruption and oversized/foreign states rejected.`);
