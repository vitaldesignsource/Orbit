import assert from 'node:assert/strict';
import {MemoryFilesystem,compileMemoryCircuit,DATA_START,ROW_BYTES} from '../dist/machine/memory-fs.js';
const started=performance.now();
const fs=new MemoryFilesystem({capacityBytes:65536});
assert.equal(fs.list('/').length,0);
assert.equal(fs.stat('/').type,'directory');
fs.mkdir('/documents');fs.mkdir('/sessions/chat',{recursive:true});
fs.writeFile('/documents/hello.txt','Hello from recurrent memory. 🪐');
assert.equal(fs.readText('/documents/hello.txt'),'Hello from recurrent memory. 🪐');
assert.equal(fs.stat('/documents/hello.txt').size,new TextEncoder().encode('Hello from recurrent memory. 🪐').length);
assert.deepEqual(fs.list('/').map(e=>e.path),['/documents','/sessions']);
assert.deepEqual(fs.list('/documents').map(e=>e.name),['hello.txt']);
const binary=Uint8Array.from({length:4097},(_,i)=>(i*137+3)&255);
fs.writeFile('/sessions/chat/cache.bin',binary);
assert.deepEqual(fs.readFile('/sessions/chat/cache.bin'),binary);
const copy=fs.readFile('/sessions/chat/cache.bin');copy[0]=99;assert.equal(fs.readFile('/sessions/chat/cache.bin')[0],3);
fs.writeFile('/documents/hello.txt','Updated');assert.equal(fs.readText('/documents/hello.txt'),'Updated');
fs.writeFile('/documents/empty','');assert.equal(fs.readFile('/documents/empty').length,0);
const weights=compileMemoryCircuit(65536);assert.equal(weights.format,'orbit-gdn-memory-circuit-v1');assert.equal(weights.headDimension,128);assert.ok(weights.heads.length>=5);assert.equal(weights.valueProjection[0][0][1],2**24);assert.equal(weights.outputProjection[0][2],2**21);
assert.ok(fs.neuralReads>0&&fs.neuralWrites>0);
// Path bytes and payload are present in actual recurrent-state row columns.
const raw=fs.dumpBytes();assert.ok(new TextDecoder().decode(raw).includes('/documents/hello.txt'));assert.ok(new TextDecoder().decode(raw).includes('Updated'));
const snapshot=fs.getState();const exportedBinary=fs.exportBinary();assert.equal(snapshot.encoding,'gdn-byte-rows-v1');assert.ok(exportedBinary.length<100000);
const restored=MemoryFilesystem.fromState(JSON.parse(JSON.stringify(snapshot)));
assert.equal(restored.readText('/documents/hello.txt'),'Updated');assert.deepEqual(restored.readFile('/sessions/chat/cache.bin'),binary);
assert.deepEqual(restored.dumpBytes(),fs.dumpBytes());
for(let h=0;h<fs.headCount;h++)assert.deepEqual(restored.headStates[h],fs.headStates[h]);
const restoredBinary=MemoryFilesystem.fromBinary(exportedBinary);assert.deepEqual(restoredBinary.readFile('/sessions/chat/cache.bin'),binary);
assert.deepEqual(restoredBinary.getState(),snapshot);
const fromString=MemoryFilesystem.fromState(fs.serialize());assert.equal(fromString.readText('/documents/hello.txt'),'Updated');
// Failed imports and full-disk operations preserve the current instance.
const before=restored.serialize();const corrupt=structuredClone(snapshot);corrupt.data=(corrupt.data[0]==='A'?'B':'A')+corrupt.data.slice(1);assert.throws(()=>restored.loadState(corrupt),/integrity/);assert.equal(restored.serialize(),before);
const corruptBinary=exportedBinary.slice();corruptBinary[100]^=1;assert.throws(()=>MemoryFilesystem.fromBinary(corruptBinary),/integrity/);
assert.throws(()=>restored.loadState({...snapshot,capacityBytes:64*1024*1024}),/Capacity/);
assert.throws(()=>MemoryFilesystem.fromState({...snapshot,headCount:999}),/head count/);
assert.throws(()=>MemoryFilesystem.fromState({...snapshot,headDimension:64}),/geometry/);
assert.throws(()=>MemoryFilesystem.fromState({...snapshot,data:'!!!'}),/encoding/);
assert.throws(()=>fs.writeFile('/too-big',new Uint8Array(65536)),/full/);assert.equal(fs.readText('/documents/hello.txt'),'Updated');
assert.throws(()=>fs.mkdir('/missing/deep'),/Parent/);assert.throws(()=>fs.readFile('/documents'),/not found/);assert.throws(()=>fs.writeFile('/documents','x'),/directory/);assert.throws(()=>fs.writeFile('/missing/x','x'),/Parent/);assert.throws(()=>fs.writeFile('/'+''.padStart(257,'x'),'x'),/256/);assert.throws(()=>fs.writeFile('/../../x','x'),/escapes/);
assert.throws(()=>fs.delete('/sessions'),/not empty/);fs.delete('/sessions',{recursive:true});assert.equal(fs.exists('/sessions'),false);assert.throws(()=>fs.delete('/'),/root/);assert.equal(fs.readText('/documents/hello.txt'),'Updated');
assert.equal(fs.stat('/documents/./hello.txt').path,'/documents/hello.txt');
// A non-neural shadow filesystem cannot restore files: only recurrent heads are serialized.
assert.equal(Object.hasOwn(fs,'files'),false);assert.equal(Object.hasOwn(fs,'records'),false);
const stateCopy=fs.getState();fs.format();assert.equal(fs.exists('/documents'),false);fs.loadState(stateCopy);assert.equal(fs.readText('/documents/hello.txt'),'Updated');
// Realistic language-model checkpoint + KV data workload in 4MiB of heads.
const large=new MemoryFilesystem();large.mkdir('/models');large.mkdir('/sessions');const model=Uint8Array.from({length:1042208},(_,i)=>(Math.imul(i,31)>>>3)&255),kv=Uint8Array.from({length:874000},(_,i)=>(i*17+11)&255);
const writeStart=performance.now();large.writeFile('/models/stories260K.bin',model);large.writeFile('/sessions/story-session.bin',kv);const writeMs=performance.now()-writeStart;assert.deepEqual(large.readFile('/models/stories260K.bin'),model);assert.deepEqual(large.readFile('/sessions/story-session.bin'),kv);large.writeFile('/note.txt','Files survive the state export.');assert.equal(large.readText('/note.txt'),'Files survive the state export.');
const largeBinary=large.exportBinary();const loadStart=performance.now();const largeRestored=MemoryFilesystem.fromBinary(largeBinary);const loadMs=performance.now()-loadStart;assert.deepEqual(largeRestored.readFile('/models/stories260K.bin'),model);assert.deepEqual(largeRestored.readFile('/sessions/story-session.bin'),kv);assert.equal(largeRestored.readText('/note.txt'),'Files survive the state export.');
console.log(`Memory FS passed: exact recurrent-head round trips, UTF-8/binary/directory operations, corruption rejection, 4MiB model+KV workload. Data starts${DATA_START}, row${ROW_BYTES}. Two large writes${Math.round(writeMs)}ms; import${Math.round(loadMs)}ms; packed state${largeBinary.length}bytes; total${Math.round(performance.now()-started)}ms.`);
