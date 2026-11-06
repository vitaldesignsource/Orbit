/**
 * Orbit recurrent-memory filesystem.
 *
 * Each storage head is a 128x128 Gated DeltaNet state matrix. Columns 0/1 are
 * Qwenputer-style ballast, columns 2..127 carry 126 exact byte values per row.
 * Every ordinary read/write evaluates compiled sparse linear address/value
 * projections and the recurrent overwrite/read-normalization equations. The
 * host parses a binary directory in that memory; it keeps no shadow file map.
 *
 * These are additional compiled storage heads, separate from the RV32 RAM.
 * Export packing is lossless for the native matrices' known byte/ballast palette.
 * This module does not load Qwen's learned language-model weights.
 */
const HD=128, ROW_BYTES=126, ROWS=128, A_REF=2**24, OUT_SCALE=2**21;
const MAX_ENTRIES=128, ENTRY_BYTES=288, HEADER_BYTES=64;
const DIRECTORY_BYTES=MAX_ENTRIES*ENTRY_BYTES;
const DATA_START=Math.ceil((HEADER_BYTES+DIRECTORY_BYTES)/ROW_BYTES)*ROW_BYTES;
const DEFAULT_CAPACITY=4*1024*1024, MIN_CAPACITY=64*1024, MAX_CAPACITY=16*1024*1024;
const DISK_MAGIC=new TextEncoder().encode('ORBITFS2');
const BINARY_MAGIC=new TextEncoder().encode('ORGDNFS2');
const encoder=new TextEncoder();
const decoder=new TextDecoder('utf-8',{fatal:true});
const alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function fnv(bytes,seed=2166136261){let h=seed>>>0;for(let i=0;i<bytes.length;i++)h=Math.imul(h^bytes[i],16777619)>>>0;return h>>>0;}
function hex(n){return(n>>>0).toString(16).padStart(8,'0');}
function base64(bytes){const chunks=[];let s='';for(let i=0;i<bytes.length;i+=3){const a=bytes[i],b=bytes[i+1],c=bytes[i+2],n=(a<<16)|((b||0)<<8)|(c||0);s+=alphabet[n>>>18]+alphabet[(n>>>12)&63]+(i+1<bytes.length?alphabet[(n>>>6)&63]:'=')+(i+2<bytes.length?alphabet[n&63]:'=');if(s.length>=32768){chunks.push(s);s='';}}chunks.push(s);return chunks.join('');}
function unbase64(s,expected){if(typeof s!=='string'||s.length!==4*Math.ceil(expected/3)||!/^[A-Za-z0-9+/]*={0,2}$/.test(s))throw Error('Invalid packed recurrent-state encoding');const out=new Uint8Array(expected);let at=0;for(let i=0;i<s.length;i+=4){const a=alphabet.indexOf(s[i]),b=alphabet.indexOf(s[i+1]),c=s[i+2]==='='?0:alphabet.indexOf(s[i+2]),d=s[i+3]==='='?0:alphabet.indexOf(s[i+3]);if(a<0||b<0||c<0||d<0)throw Error('Invalid base64 state');const n=(a<<18)|(b<<12)|(c<<6)|d;if(at<expected)out[at++]=n>>>16;if(at<expected)out[at++]=(n>>>8)&255;if(at<expected)out[at++]=n&255;}return out;}
function asBytes(data){if(typeof data==='string')return encoder.encode(data);if(data instanceof Uint8Array)return data.slice();if(data instanceof ArrayBuffer)return new Uint8Array(data.slice(0));if(ArrayBuffer.isView(data))return new Uint8Array(data.buffer,data.byteOffset,data.byteLength).slice();throw TypeError('File data must be text or bytes');}
function capacity(n){if(!Number.isInteger(n)||n<MIN_CAPACITY||n>MAX_CAPACITY)throw RangeError(`Capacity must be ${MIN_CAPACITY}..${MAX_CAPACITY} bytes`);return n;}
function normalize(path){if(typeof path!=='string'||path.includes('\0'))throw Error('Invalid file path');const parts=[];for(const p of path.split('/')){if(!p||p==='.')continue;if(p==='..'){if(!parts.length)throw Error('Path escapes filesystem root');parts.pop();}else parts.push(p);}const result='/'+parts.join('/');if(encoder.encode(result).length>256)throw Error('UTF-8 path exceeds 256 bytes');return result;}
function parent(path){const at=path.lastIndexOf('/');return at>0?path.slice(0,at):'/';}
function basename(path){return path==='/'?'/':path.slice(path.lastIndexOf('/')+1);}
function dot(expr,x){let y=0;for(let i=0;i<expr.length;i++)y+=expr[i][1]*x[expr[i][0]];return y;}
function assert(condition,message){if(!condition)throw Error(message);}

/** Compile sparse linear/SwiGLU address selectors and Gated DeltaNet heads. */
export function compileMemoryCircuit(capacityBytes=DEFAULT_CAPACITY){
 capacityBytes=capacity(capacityBytes);const headCount=Math.ceil(capacityBytes/(ROWS*ROW_BYTES)),headBits=Math.ceil(Math.log2(headCount)),rowBase=3+headBits,dataBase=rowBase+7,selectDim=dataBase+ROW_BYTES,dims=selectDim+1;
 const heads=[];
 for(let h=0;h<headCount;h++){const select=[[0,1]];for(let b=0;b<headBits;b++){const bit=(h>>>b)&1;select.push([3+b,bit?1:-1]);if(bit)select[0][1]--;}
  heads.push({name:'filesystem.memory'+h,select});
 }
 const rowProjection=[];
 for(let r=0;r<ROWS;r++){const common=[[0,-1],[selectDim,1]];for(let b=0;b<7;b++){const bit=(r>>>b)&1;common.push([rowBase+b,bit?1:-1]);if(bit)common[0][1]--;}
  rowProjection.push({common,q:[...common,[1,1]],k:[...common,[2,1]]});
 }
 const v=[[[0,A_REF]],[[0,A_REF]],...Array.from({length:ROW_BYTES},(_,i)=>[[dataBase+i,1]])];
 return {format:'orbit-gdn-memory-circuit-v1',target:'compact Qwen3.5-style Gated DeltaNet IR',numericDomain:'byte and ballast state values are exactly representable in bf16',placement:{selectorLayers:1,storageLayers:Math.ceil(headCount/16),keyHeadsPerLayer:16},capacityBytes,headCount,headDimension:HD,rows:ROWS,rowBytes:ROW_BYTES,headBits,rowBase,dataBase,selectDim,dims,features:{one:0,read:1,write:2},heads,rowProjection,valueProjection:v,beta:[[0,1]],decay:1,gate:[[0,1]],outputProjection:Array.from({length:ROW_BYTES},(_,i)=>[i+2,i,OUT_SCALE]),description:'ReLU sparse address projections; exact one-hot DeltaNet overwrite; ballast RMS normalization and linear output projection'};
}

export class MemoryFilesystem {
 constructor(options={}){
  const n=typeof options==='number'?options:options.capacityBytes??DEFAULT_CAPACITY;
  this._configure(n);this.neuralReads=0;this.neuralWrites=0;
  if(!(typeof options==='object'&&options._empty))this.format();
 }
 _configure(n){this.capacityBytes=capacity(n);this.circuit=compileMemoryCircuit(n);this.headCount=this.circuit.headCount;this.headStates=Array.from({length:this.headCount},()=>new Float64Array(HD*ROWS));this._features=new Float64Array(this.circuit.dims);this._value=new Float64Array(HD);this._read=new Float64Array(ROW_BYTES);}
 /** One address token through compiled projections and recurrent head equations. */
 _step(row,writeData=null){
  assert(Number.isInteger(row)&&row>=0&&row<Math.ceil(this.capacityBytes/ROW_BYTES),'Memory row out of range');
  const c=this.circuit,x=this._features;x.fill(0);x[0]=1;x[writeData?2:1]=1;
  const hAddress=Math.floor(row/ROWS),rAddress=row%ROWS;
  for(let b=0;b<c.headBits;b++)x[3+b]=(hAddress>>>b)&1;
  for(let b=0;b<7;b++)x[c.rowBase+b]=(rAddress>>>b)&1;
  if(writeData){assert(writeData.length===ROW_BYTES,'A memory write requires 126 bytes');for(let b=0;b<ROW_BYTES;b++)x[c.dataBase+b]=writeData[b];}
  const value=this._value;for(let col=0;col<HD;col++)value[col]=dot(c.valueProjection[col],x);
  const output=this._read;output.fill(0);
  // All nonselected heads have zero q/k and identity recurrence. Eliding those
  // zero operations is the same compact evaluation used for the CPU circuit.
  for(let h=0;h<c.headCount;h++){
   const selected=Math.max(0,dot(c.heads[h].select,x));if(selected===0)continue;x[c.selectDim]=selected;
   const state=this.headStates[h];let ballast0=0,ballast1=0;
   for(let r=0;r<ROWS;r++){
    const common=dot(c.rowProjection[r].common,x),q=Math.max(0,common+x[1]),k=Math.max(0,common+x[2]);if(!q&&!k)continue;
    const offset=r*HD;
    if(k){const beta=dot(c.beta,x);for(let col=0;col<HD;col++){const old=state[offset+col];state[offset+col]=old+beta*k*(value[col]-k*old);}}
    if(q){ballast0+=q*state[offset];ballast1+=q*state[offset+1];for(let b=0;b<ROW_BYTES;b++)output[b]+=q*state[offset+b+2];}
   }
   const ref=ballast0*ballast0+ballast1*ballast1;
   if(ref){const norm=Math.sqrt(ref/HD),gate=Math.max(0,dot(c.gate,x));for(const [,dest,weight]of c.outputProjection)output[dest]=output[dest]/norm*gate*weight;}else output.fill(0);
  }
  if(writeData)this.neuralWrites++;else this.neuralReads++;
  return output;
 }
 _readBytes(offset,length){assert(Number.isInteger(offset)&&Number.isInteger(length)&&offset>=0&&length>=0&&offset+length<=this.capacityBytes,'Memory byte range out of bounds');const out=new Uint8Array(length);let done=0;while(done<length){const address=offset+done,row=Math.floor(address/ROW_BYTES),within=address%ROW_BYTES,count=Math.min(ROW_BYTES-within,length-done),values=this._step(row);for(let i=0;i<count;i++){const n=values[within+i];assert(Number.isInteger(n)&&n>=0&&n<=255,'Invalid recurrent byte value');out[done+i]=n;}done+=count;}return out;}
 _writeBytes(offset,bytes){assert(Number.isInteger(offset)&&offset>=0&&offset+bytes.length<=this.capacityBytes,'Memory byte range out of bounds');let done=0;while(done<bytes.length){const address=offset+done,row=Math.floor(address/ROW_BYTES),within=address%ROW_BYTES,count=Math.min(ROW_BYTES-within,bytes.length-done);const payload=within===0&&count===ROW_BYTES?bytes.subarray(done,done+count):Uint8Array.from(this._step(row));if(within||count!==ROW_BYTES)payload.set(bytes.subarray(done,done+count),within);this._step(row,payload);done+=count;}}
 _clearBytes(offset,length){const zeros=new Uint8Array(ROW_BYTES);let done=0;while(done<length){const count=Math.min(ROW_BYTES,length-done);this._writeBytes(offset+done,zeros.subarray(0,count));done+=count;}}
 _index(){
  const header=this._readBytes(0,HEADER_BYTES),v=new DataView(header.buffer);assert(DISK_MAGIC.every((b,i)=>header[i]===b),'Memory filesystem magic mismatch');assert(v.getUint32(8,true)===2,'Unsupported memory filesystem version');assert(v.getUint32(12,true)===this.capacityBytes,'Filesystem capacity mismatch');assert(v.getUint32(28,true)===DATA_START,'Filesystem data boundary mismatch');
  const raw=this._readBytes(HEADER_BYTES,DIRECTORY_BYTES);assert(fnv(raw)===v.getUint32(24,true),'Filesystem directory checksum mismatch');const records=[],paths=new Set(),view=new DataView(raw.buffer);
  for(let slot=0;slot<MAX_ENTRIES;slot++){const start=slot*ENTRY_BYTES,type=raw[start];if(!type)continue;assert(type===1||type===2,'Invalid directory record type');const n=view.getUint16(start+2,true);assert(n>0&&n<=256,'Invalid UTF-8 path length');let path;try{path=decoder.decode(raw.subarray(start+20,start+20+n));}catch{throw Error('Invalid UTF-8 path in recurrent memory');}assert(normalize(path)===path&&!paths.has(path),'Invalid or duplicate directory path');paths.add(path);const offset=view.getUint32(start+4,true),length=view.getUint32(start+8,true),allocated=view.getUint32(start+12,true),checksum=view.getUint32(start+16,true);
   if(type===1){assert(offset>=DATA_START&&offset%ROW_BYTES===0&&allocated%ROW_BYTES===0&&length<=allocated&&offset+allocated<=this.capacityBytes,'Invalid file extent');}else assert(offset===0&&length===0&&allocated===0,'Directory contains a file extent');records.push({slot,path,type:type===1?'file':'directory',offset,length,allocated,checksum});
  }
  assert(records.length===v.getUint32(20,true),'Filesystem record count mismatch');assert(records.some(r=>r.path==='/'&&r.type==='directory'),'Filesystem root directory missing');const map=new Map(records.map(r=>[r.path,r]));for(const r of records)if(r.path!=='/')assert(map.get(parent(r.path))?.type==='directory','Filesystem parent directory missing');const extents=records.filter(r=>r.allocated).sort((a,b)=>a.offset-b.offset);for(let i=1;i<extents.length;i++)assert(extents[i-1].offset+extents[i-1].allocated<=extents[i].offset,'Overlapping file extents');
  return{records,generation:v.getUint32(16,true)};
 }
 _saveIndex(records,generation){const raw=new Uint8Array(DIRECTORY_BYTES),v=new DataView(raw.buffer);for(const r of records){const start=r.slot*ENTRY_BYTES,path=encoder.encode(r.path);assert(path.length<=256,'Path is too long');raw[start]=r.type==='file'?1:2;v.setUint16(start+2,path.length,true);v.setUint32(start+4,r.offset||0,true);v.setUint32(start+8,r.length||0,true);v.setUint32(start+12,r.allocated||0,true);v.setUint32(start+16,r.checksum||0,true);raw.set(path,start+20);}const header=new Uint8Array(HEADER_BYTES),hv=new DataView(header.buffer);header.set(DISK_MAGIC);hv.setUint32(8,2,true);hv.setUint32(12,this.capacityBytes,true);hv.setUint32(16,generation>>>0,true);hv.setUint32(20,records.length,true);hv.setUint32(24,fnv(raw),true);hv.setUint32(28,DATA_START,true);this._writeBytes(HEADER_BYTES,raw);this._writeBytes(0,header);}
 _freeSlot(records){const used=new Set(records.map(r=>r.slot));for(let slot=0;slot<MAX_ENTRIES;slot++)if(!used.has(slot))return slot;throw Error('Filesystem directory is full (128 entries)');}
 _allocate(records,length,exclude=null){const size=Math.ceil(length/ROW_BYTES)*ROW_BYTES;let at=DATA_START;for(const r of records.filter(r=>r.allocated&&r!==exclude).sort((a,b)=>a.offset-b.offset)){if(at+size<=r.offset)return{offset:at,allocated:size};at=Math.max(at,r.offset+r.allocated);}if(at+size>this.capacityBytes)throw Error('Recurrent memory filesystem is full');return{offset:at,allocated:size};}
 format(){for(const state of this.headStates)state.fill(0);this._saveIndex([{slot:0,path:'/',type:'directory',offset:0,length:0,allocated:0,checksum:0}],1);return this;}
 exists(path){path=normalize(path);return this._index().records.some(r=>r.path===path);}
 stat(path){path=normalize(path);const r=this._index().records.find(r=>r.path===path);if(!r)throw Error('File or directory not found: '+path);return{path:r.path,name:basename(r.path),type:r.type,size:r.length,allocated:r.allocated};}
 list(path='/'){path=normalize(path);const {records}=this._index(),dir=records.find(r=>r.path===path);if(!dir||dir.type!=='directory')throw Error('Directory not found: '+path);return records.filter(r=>r.path!==path&&parent(r.path)===path).map(r=>({path:r.path,name:basename(r.path),type:r.type,size:r.length})).sort((a,b)=>a.type===b.type?a.name.localeCompare(b.name):a.type==='directory'?-1:1);}
 mkdir(path,options={}){path=normalize(path);const {records,generation}=this._index();if(path==='/')return this.stat('/');const segments=path.slice(1).split('/');let at='';for(let i=0;i<segments.length;i++){at+='/'+segments[i];const existing=records.find(r=>r.path===at);if(existing){if(existing.type!=='directory')throw Error('A file occupies '+at);continue;}if(i<segments.length-1&&!options.recursive)throw Error('Parent directory does not exist');records.push({slot:this._freeSlot(records),path:at,type:'directory',offset:0,length:0,allocated:0,checksum:0});}this._saveIndex(records,generation+1);return this.stat(path);}
 writeFile(path,data){path=normalize(path);if(path==='/')throw Error('Cannot write the root directory');const bytes=asBytes(data),{records,generation}=this._index(),dir=records.find(r=>r.path===parent(path));if(!dir||dir.type!=='directory')throw Error('Parent directory does not exist: '+parent(path));const existing=records.find(r=>r.path===path);if(existing?.type==='directory')throw Error('Cannot overwrite a directory');const extent=existing&&bytes.length<=existing.allocated?{offset:existing.offset,allocated:existing.allocated}:this._allocate(records,bytes.length,existing);const r={slot:existing?.slot??this._freeSlot(records),path,type:'file',...extent,length:bytes.length,checksum:fnv(bytes)};
  if(existing&&existing.offset!==r.offset)this._clearBytes(existing.offset,existing.allocated);this._writeBytes(r.offset,bytes);if(r.allocated>bytes.length)this._clearBytes(r.offset+bytes.length,r.allocated-bytes.length);if(existing)records[records.indexOf(existing)]=r;else records.push(r);this._saveIndex(records,generation+1);return this.stat(path);}
 readFile(path,options={}){path=normalize(path);const r=this._index().records.find(r=>r.path===path);if(!r||r.type!=='file')throw Error('File not found: '+path);const data=this._readBytes(r.offset,r.length);assert(fnv(data)===r.checksum,'File checksum mismatch: '+path);return options.text?new TextDecoder().decode(data):data;}
 readText(path){return this.readFile(path,{text:true});}
 delete(path,options={}){path=normalize(path);if(path==='/')throw Error('Cannot delete the filesystem root');const {records,generation}=this._index(),r=records.find(r=>r.path===path);if(!r)throw Error('File or directory not found: '+path);const children=records.filter(e=>e.path.startsWith(path+'/'));if(children.length&&!options.recursive)throw Error('Directory is not empty');const removed=[r,...children],removePaths=new Set(removed.map(e=>e.path));for(const file of removed)if(file.allocated)this._clearBytes(file.offset,file.allocated);this._saveIndex(records.filter(e=>!removePaths.has(e.path)),generation+1);return true;}
 remove(path,options={}){return this.delete(path,options);}
 getStats(){const {records,generation}=this._index(),allocated=records.reduce((n,r)=>n+r.allocated,0);return{capacityBytes:this.capacityBytes,usedBytes:DATA_START+allocated,fileBytes:records.reduce((n,r)=>n+r.length,0),freeBytes:this.capacityBytes-DATA_START-allocated,files:records.filter(r=>r.type==='file').length,directories:records.filter(r=>r.type==='directory').length,entries:records.length,maxEntries:MAX_ENTRIES,generation,headCount:this.headCount,headDimension:HD,rowBytes:ROW_BYTES,neuralReads:this.neuralReads,neuralWrites:this.neuralWrites,storage:'Gated DeltaNet recurrent matrices'};}
 stats(){return this.getStats();}
 dumpBytes(){return this._readBytes(0,this.capacityBytes);}
 exportCircuit(){return structuredClone(this.circuit);}
 _pack(){const stride=16+ROWS*ROW_BYTES,packed=new Uint8Array(this.headCount*stride);for(let h=0;h<this.headCount;h++){const state=this.headStates[h],base=h*stride;for(let r=0;r<ROWS;r++){const at=r*HD,present=state[at]===A_REF&&state[at+1]===A_REF;assert(present||state[at]===0&&state[at+1]===0,'Invalid storage ballast');if(present)packed[base+(r>>>3)]|=1<<(r&7);for(let col=0;col<ROW_BYTES;col++){const value=state[at+col+2];assert(Number.isInteger(value)&&value>=0&&value<=255,'Invalid storage state byte');assert(present||value===0,'Nonzero storage row without ballast');packed[base+16+r*ROW_BYTES+col]=value;}}}return packed;}
 getState(){const packed=this._pack();return{format:'orbit-neural-filesystem-state',version:2,encoding:'gdn-byte-rows-v1',capacityBytes:this.capacityBytes,headCount:this.headCount,headDimension:HD,rowBytes:ROW_BYTES,checksum:'fnv1a32:'+hex(fnv(packed)),data:base64(packed),circuit:'orbit-gdn-memory-circuit-v1'};}
 exportState(){return this.getState();}
 serialize(){return JSON.stringify(this.getState());}
 exportBinary(){const packed=this._pack(),bytes=new Uint8Array(32+packed.length),v=new DataView(bytes.buffer);bytes.set(BINARY_MAGIC);v.setUint32(8,2,true);v.setUint32(12,this.capacityBytes,true);v.setUint32(16,this.headCount,true);v.setUint32(20,packed.length,true);v.setUint32(24,fnv(packed),true);v.setUint32(28,1,true);bytes.set(packed,32);return bytes;}
 loadState(input){
  let state,packed;if(input instanceof ArrayBuffer||ArrayBuffer.isView(input)){const bytes=asBytes(input);assert(bytes.length>=32,'Truncated filesystem state');const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);assert(BINARY_MAGIC.every((b,i)=>bytes[i]===b)&&v.getUint32(8,true)===2&&v.getUint32(28,true)===1,'Unsupported binary filesystem state');state={capacityBytes:v.getUint32(12,true),headCount:v.getUint32(16,true),checksum:'fnv1a32:'+hex(v.getUint32(24,true))};assert(v.getUint32(20,true)===bytes.length-32,'Packed state length mismatch');packed=bytes.slice(32);
  }else{state=typeof input==='string'?JSON.parse(input):input;assert(state&&state.format==='orbit-neural-filesystem-state'&&state.version===2&&state.encoding==='gdn-byte-rows-v1'&&state.circuit==='orbit-gdn-memory-circuit-v1','Unsupported filesystem snapshot');assert(state.headDimension===HD&&state.rowBytes===ROW_BYTES,'Filesystem snapshot geometry mismatch');}
  const n=capacity(state.capacityBytes),expectedHeads=Math.ceil(n/(ROWS*ROW_BYTES)),stride=16+ROWS*ROW_BYTES;assert(state.headCount===expectedHeads,'Filesystem snapshot head count mismatch');if(!packed)packed=unbase64(state.data,expectedHeads*stride);assert(packed.length===expectedHeads*stride,'Filesystem snapshot matrix length mismatch');assert(state.checksum==='fnv1a32:'+hex(fnv(packed)),'Filesystem snapshot integrity check failed');
  const candidate=new MemoryFilesystem({capacityBytes:n,_empty:true});for(let h=0;h<expectedHeads;h++){const matrix=candidate.headStates[h],base=h*stride;for(let r=0;r<ROWS;r++){const present=(packed[base+(r>>>3)]>>>(r&7))&1,at=r*HD;matrix[at]=matrix[at+1]=present?A_REF:0;for(let col=0;col<ROW_BYTES;col++){const value=packed[base+16+r*ROW_BYTES+col];assert(present||value===0,'Snapshot row has payload without ballast');matrix[at+col+2]=value;}}}
  const index=candidate._index();for(const file of index.records)if(file.type==='file')candidate.readFile(file.path);
  this.capacityBytes=candidate.capacityBytes;this.circuit=candidate.circuit;this.headCount=candidate.headCount;this.headStates=candidate.headStates;this._features=candidate._features;this._value=candidate._value;this._read=candidate._read;this.neuralReads=candidate.neuralReads;this.neuralWrites=0;return this;
 }
 static fromState(state){const fs=new MemoryFilesystem({capacityBytes:MIN_CAPACITY,_empty:true});return fs.loadState(state);}
 static fromBinary(bytes){return MemoryFilesystem.fromState(bytes);}
}
export {normalize as normalizeMemoryPath,DEFAULT_CAPACITY,ROW_BYTES,DATA_START};
