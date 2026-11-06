import {MemoryFilesystem} from './memory-fs.js';
import {captureNeural,restoreNeural} from './checkpoint.js';

const encoder=new TextEncoder(),decoder=new TextDecoder();
export const PATHS={model:'/models/stories260K.bin',tokenizer:'/models/tok512.bin',story:'/sessions/stories260K.json'};
let session;
export function newSession(){const fs=new MemoryFilesystem({capacityBytes:4*1024*1024});fs.mkdir('/documents');fs.mkdir('/models');fs.mkdir('/sessions');fs.writeFile('/documents/Welcome.txt','Your files live in recurrent memory head matrices.\n\nExport a state file, attach it to a chat, then import it here later.\nThe file is the saved state; a chat message by itself does not save this machine.\n');return {fs,model:null,unit:null,cpuView:null,modelBusy:false,createdAt:new Date().toISOString()};}
export function getSession(){return session??=newSession();}
export function replaceSession(value){session=value;return session;}
export async function packState(value){const bytes=encoder.encode(JSON.stringify(value));if(typeof CompressionStream==='undefined')return bytes;const stream=new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));return new Uint8Array(await new Response(stream).arrayBuffer());}
export async function unpackState(bytes){bytes=bytes instanceof Uint8Array?bytes:new Uint8Array(bytes);if(bytes.byteLength>128*1024*1024)throw Error('State file is too large.');let raw=bytes;if(bytes[0]===31&&bytes[1]===139){if(typeof DecompressionStream==='undefined')throw Error('This browser cannot decompress saved states.');const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));const reader=stream.getReader(),parts=[];let length=0;for(;;){const {value,done}=await reader.read();if(done)break;length+=value.length;if(length>160*1024*1024){await reader.cancel();throw Error('Expanded state is too large.');}parts.push(value);}raw=new Uint8Array(length);let at=0;for(const part of parts){raw.set(part,at);at+=part.length;}}return JSON.parse(decoder.decode(raw));}
export async function exportWorkspace(s=getSession()){
 if(s.modelBusy)throw Error('Pause generation before exporting the machine.');
 if(s.model)s.fs.writeFile(PATHS.story,JSON.stringify(s.model.getState()));
 const machine=s.cpuView?.getMachine?.();
 const cpu=machine?.backend==='neural'?{program:machine.program,checkpoint:captureNeural(machine.cpu)}:null;
 return packState({format:'orbit-recurrent-workspace',version:2,createdAt:s.createdAt,savedAt:new Date().toISOString(),filesystem:s.fs.exportState(),cpu,vector:{format:'orbit-weighted-vector-v1',rebuildFromModel:true},model:s.model?{checkpointPath:PATHS.story}:null});
}
export async function importWorkspace(bytes,{circuit,unit,createModel}={}){
 const data=await unpackState(bytes);if(data?.format!=='orbit-recurrent-workspace'||data.version!==2)throw Error('This is not an Orbit recurrent workspace state.');
 const fs=MemoryFilesystem.fromState(data.filesystem);let model=null,cpu=null,cpuError=null,modelError=null;
 if(data.model&&!createModel)modelError='The model runner is unavailable; files were restored.';
 if(data.model&&createModel){try{if(!fs.exists(PATHS.model)||!fs.exists(PATHS.tokenizer)||!fs.exists(PATHS.story))throw Error('Saved model files are missing.');model=await createModel(fs.readFile(PATHS.model),fs.readFile(PATHS.tokenizer),unit);model.loadState(JSON.parse(fs.readText(PATHS.story)));}catch(error){model=null;modelError=error.message;}}
 if(data.cpu&&!circuit)cpuError='The CPU circuit is unavailable; files were restored.';
 if(data.cpu&&circuit){try{cpu={cpu:restoreNeural(circuit,data.cpu.checkpoint),program:data.cpu.program,backend:'neural'};}catch(error){cpuError=error.message;}}
 const restored={fs,model,unit:unit??null,cpuView:null,modelBusy:false,createdAt:data.createdAt||new Date().toISOString(),restoredCpu:cpu,cpuError,modelError};
 replaceSession(restored);return restored;
}
export function download(bytes,name,type='application/octet-stream'){const address=URL.createObjectURL(new Blob([bytes],{type}));const a=document.createElement('a');a.href=address;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(address),3000);}
