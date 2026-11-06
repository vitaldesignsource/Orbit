/*
 * Compact Qwenputer circuit executor.
 *
 * The arithmetic here is the sparse SwiGLU / Gated DeltaNet circuit exported
 * from divagr18/qwenputer, not a RISC-V instruction interpreter. Machine state
 * lives in the heads' recurrent matrices. It executes the added CPU layers;
 * it does not load Qwen3.5-4B's original language-model layers.
 *
 * Upstream circuit: Copyright (c) 2026 Divyansh Agrawal, MIT.
 */

const DEFAULT_FAMILIES = [['V',32768],['F0',32768],['F1',32768],['F2',4],['J',65536],['R',65536],['BYTE',512],['FLT',8],['HEX',16],['QS',16],['QD',128],['BD',256]];
const DEFAULT_CONTROLS = ['NT','ST0','ST1','SS0','SS1','HALT','GO','SC'];
const FINAL = new Set(['F0','F1','F2','J','R','BYTE','NT','ST0','ST1','SS0','SS1','HALT']);

function expression(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  if (raw.terms) return expression(raw.terms);
  return Object.entries(raw).map(([dim,weight])=>[Number(dim),Number(weight)]);
}
function dot(expr, residual) {
  if(expr.kernel)return expr.kernel(residual);
  let n=0;
  for(let i=0;i<expr.length;i++) n+=residual[expr[i][0]]*expr[i][1];
  return n;
}
function entries(raw) {return Array.isArray(raw)?raw:Object.entries(raw||{}).map(([k,v])=>[Number(k),v]);}
const expressionKernels=new Map();
function compiledExpression(raw){
  const e=expression(raw);if(e.kernel)return e;
  const source=exprSource(e);
  try{let fn=expressionKernels.get(source);if(!fn){fn=new Function('m',`return ${source};`);expressionKernels.set(source,fn);}e.kernel=fn;}catch{}
  return e;
}
function compileChannel(raw) {
  if (Array.isArray(raw) && raw.length===2 && Array.isArray(raw[1]) && raw[1].length===4) raw={expr:raw[0],taps:raw[1]};
  return {expr:compiledExpression(raw?.expr??raw?.e??raw),taps:raw?.taps??[1,0,0,0],history:new Float64Array(3)};
}
function compileChannels(raw) {return entries(raw).map(([col,ch])=>[Number(col),compileChannel(ch)]);}
function conv(channel,residual) {
  const value=dot(channel.expr,residual),taps=channel.taps,h=channel.history;
  let out=value*(taps[0]||0);
  for(let lag=1;lag<4;lag++) out+=(taps[lag]||0)*h[lag-1];
  h[2]=h[1];h[1]=h[0];h[0]=value;
  return out>0?out:0;
}
function loadRows(raw,nCols) {
  const state=new Float64Array(128*nCols),present=new Uint8Array(128);
  for(const [row,values] of entries(raw)) {
    const r=Number(row); if(r<0||r>=128) throw new Error('Invalid recurrent state row');
    if(Array.isArray(values) && values.length===nCols && typeof values[0]==='number') state.set(values,r*nCols);
    else if(ArrayBuffer.isView(values)) state.set(values,r*nCols);
    else for(const [col,value] of entries(values)) state[r*nCols+Number(col)]=Number(value);
    present[r]=1;
  }
  return {state,present};
}
function compileVHead(raw,nCols,init) {
  return {v:compileChannels(raw.v),a:compiledExpression(raw.a),z:entries(raw.z).map(([col,e])=>[Number(col),compiledExpression(e)]),
    out:raw.out||[],beta:raw.beta==null?null:compiledExpression(raw.beta),into:!!raw.into,
    ...loadRows(init,nCols),vScratch:new Float64Array(nCols),oScratch:new Float64Array(nCols)};
}
function exprSource(raw) {
  const e=expression(raw);
  return e.length?'('+e.map(([d,w])=>Number(w)===1?`m[${Number(d)}]`:`m[${Number(d)}]*(${Number(w)})`).join('+')+')':'0';
}
function compileFFN(raw) {
  // This specializes only the fixed, exported circuit weights. Snapshot/program
  // bytes never enter generated source. A CSP that forbids JIT uses the same
  // arithmetic through the portable loop below.
  try {
    let source='let o=0,b;';
    for(const [w,a,b] of raw.terms)source+=`b=${exprSource(b)};if(b>0)o+=(${Number(w)})*${exprSource(a)}*b;`;
    source+=`m[${Number(raw.out)}]=${raw.into?`m[${Number(raw.out)}]+`:''}o;`;
    return new Function('m',source);
  } catch {return null;}
}
function compileNode(raw,index) {
  const type=raw.type||(raw.terms?'ffn':'head');
  if(type==='ffn') return {type,index,name:raw.name,out:raw.out,into:!!raw.into,
    terms:raw.terms.map(([weight,a,b])=>[Number(weight),expression(a),expression(b)]),kernel:compileFFN(raw)};
  const nCols=raw.nCols??raw.n_cols??128;
  const vh=raw.vheads||raw.vh||[{v:raw.v,a:raw.a,z:raw.z,out:raw.out,beta:raw.beta,into:raw.into},...(raw.vx||[])];
  return {type:'head',index,name:raw.name,nCols,q:compileChannels(raw.q),k:compileChannels(raw.k),
    init:raw.init||[],vheads:vh.map((v,j)=>compileVHead(v,nCols,raw.init?.[j])),qRows:new Int16Array(128)};
}

export class NeuralRV32 {
  constructor(circuit,options={}) {
    if(!circuit?.nodes) throw new Error('A compiled Qwenputer circuit is required');
    this.circuit=circuit;this.meta=circuit.meta||{};this.options=options;
    this.dimNames=circuit.dims||[];
    this.dimCount=typeof this.dimNames==='number'?this.dimNames:this.dimNames.length;
    this.residual=new Float64Array(this.dimCount);
    this.nodes=circuit.nodes.map(compileNode);
    this.features=(circuit.features||[]).map(f=>Array.isArray(f)?{dim:f[0],name:f[1]}:f);
    this.tables=circuit.tables||this.meta.tables||{};
    this.familySizes=circuit.familySizes||this.meta.familySizes||DEFAULT_FAMILIES;
    this.controls=circuit.controls||this.meta.controls||DEFAULT_CONTROLS;
    this.outNames=circuit.outNames||this.meta.outNames||[...this.familySizes.map(f=>f[0]),...this.controls,'Q'];
    this.familyBase=new Map();let base=0;
    for(const [name,size] of this.familySizes){this.familyBase.set(name,{base,size});base+=size;}
    this.controlBase=base;this.vocab=base+this.controls.length;
    this.nFam=this.meta.nFam??Math.ceil(Math.log2(this.outNames.length));
    this.codeDims=(circuit.codeDims||this.meta.codeDims||[]).map(Number);
    this.outFam=(circuit.outFam||this.meta.outFam||[]).map(expression);
    this.outPay=(circuit.outPay||this.meta.outPay||[]).map(expression);
    this.pcExpr=expression(circuit.pcExpr||this.meta.pcExpr);
    this.pcBits=(circuit.pcBits||this.meta.pcBits||[]).map(expression);
    this.regCols=circuit.regCols||this.meta.regCols||{};
    this.regs=new Uint32Array(32);
    this.reset();
  }
  family(token) {
    if(token<0||token>=this.vocab) return {name:'INVALID',payload:0};
    if(token>=this.controlBase)return {name:this.controls[token-this.controlBase],payload:0};
    for(const [name,{base,size}] of this.familyBase)if(token>=base&&token<base+size)return {name,payload:token-base};
    return {name:'INVALID',payload:0};
  }
  token(name,payload=0) {
    const f=this.familyBase.get(name);
    if(f){if(payload<0||payload>=f.size)return -1;return f.base+payload;}
    const index=this.controls.indexOf(name);return index<0?-1:this.controlBase+index;
  }
  tokenName(token) {const f=this.family(token);return this.controls.includes(f.name)?f.name:`${f.name}(${f.payload})`;}
  reset() {return this.loadSnapshot(this.snapshot||{states:{}});}
  loadSnapshot(snapshot={}) {
    this.snapshot=snapshot;
    const states=snapshot.states||snapshot.heads||snapshot.init||{};
    for(const node of this.nodes) {
      if(node.type==='ffn')continue;
      const init=states[node.index]??states[node.name]??node.init;
      for(let j=0;j<node.vheads.length;j++) {
        const vh=node.vheads[j],rows=loadRows(init?.[j],node.nCols);
        vh.state=rows.state;vh.present=rows.present;vh.vScratch.fill(0);vh.oScratch.fill(0);
        for(const [,channel] of vh.v)channel.history.fill(0);
      }
      for(const [,ch]of node.q)ch.history.fill(0);
      for(const [,ch]of node.k)ch.history.fill(0);
    }
    this.residual.fill(0);this.regs.fill(0);
    this.tokens=0;this.instructions=0;this.output='';this.outputBytes=[];this.lastToken=null;
    this.nextToken=snapshot.startToken??this.token('GO');this.pc=snapshot.pc??0x40000;
    this.state='ready';this.exitCode=null;this.faultCode=null;this.trace=[];
    this.refreshRegisters();return this;
  }
  embed(token) {
    const x=this.residual;x.fill(0);if(token<0)return x;if(token>=this.vocab)throw new Error('This executor accepts standalone circuit tokens only');x[0]=1;
    const {name,payload}=this.family(token),familyId=this.outNames.indexOf(name);
    for(const feature of this.features) {
      const key=feature.name;let value=0,match;
      if(key===`is${name}`)value=1;
      else if((match=/^(.+)\.b(\d+)$/.exec(key)))value=match[1]===name?(payload>>>Number(match[2]))&1:0;
      else if((match=/^digit==(\d+)$/.exec(key))) {const digit=Number(match[1]);value=Number(name==='QS'&&payload===digit||name==='QD'&&(payload>>>4)===digit-1);}
      else if((match=/^code(\d+)$/.exec(key))) {const bit=Number(match[1]);value=128*(2*(bit<this.nFam?(familyId>>>bit)&1:(payload>>>(bit-this.nFam))&1)-1);}
      else if(feature.constant!=null)value=feature.constant;
      x[feature.dim]=value;
    }
    return x;
  }
  evaluateToken(token) {
    const x=this.embed(token);
    for(const node of this.nodes) {
      if(node.type==='ffn') {
        if(node.kernel){node.kernel(x);continue;}
        let value=0;
        for(const [weight,a,b]of node.terms){const gate=dot(b,x);if(gate>0)value+=weight*dot(a,x)*gate;}
        x[node.out]=(node.into?x[node.out]:0)+value;
        continue;
      }
      let nQuery=0,keyRow=-1;
      for(const [col,ch]of node.q)if(conv(ch,x)>0)node.qRows[nQuery++]=col;
      for(const [col,ch]of node.k)if(conv(ch,x)>0)keyRow=col;
      const C=node.nCols;
      for(const vh of node.vheads) {
        const value=vh.vScratch;value.fill(0);
        for(const [col,ch]of vh.v)value[col]=conv(ch,x);
        const decay=dot(vh.a,x);
        if(decay!==0) {
          if(decay===110){for(let i=0;i<vh.state.length;i++)vh.state[i]*=.5;}
          else if(decay===238) {vh.state.fill(0);vh.present.fill(0);}
          else throw new Error(`Invalid recurrent decay selector ${decay} in ${node.name}`);
        }
        const beta=vh.beta===null?1:dot(vh.beta,x);
        if(keyRow>=0&&beta>0){vh.state.set(value,keyRow*C);vh.present[keyRow]=1;}
        const out=vh.oScratch;out.fill(0);
        for(let i=0;i<nQuery;i++) {const row=node.qRows[i];if(!vh.present[row])continue;const offset=row*C;for(let col=0;col<C;col++)out[col]+=vh.state[offset+col];}
        const ref=out[0]*out[0]+out[1]*out[1];
        if(ref>0){const inv=1/Math.sqrt(ref/C);for(let col=0;col<C;col++)out[col]*=inv;}else out.fill(0);
        // z is computed before this head writes its output dimensions.
        const gates=new Map();for(const[col,expr]of vh.z)gates.set(col,Math.max(0,dot(expr,x)));
        for(const [col,dim,weight]of vh.out){const v=out[col]*(gates.get(col)||0)*weight;x[dim]=(vh.into?x[dim]:0)+v;}
      }
    }
    if(this.pcBits.length){let word=0;for(let bit=0;bit<this.pcBits.length;bit++)word+=Math.round(dot(this.pcBits[bit],x))*2**bit;this.pc=(0x40000+4*word)>>>0;}
    else if(this.pcExpr.length)this.pc=(0x40000+4*Math.round(dot(this.pcExpr,x)))>>>0;
    this.refreshRegisters();
    return this.predictToken();
  }
  predictToken() {
    const x=this.residual;let fid=0,payload=0;
    if(this.outFam.length){for(let i=0;i<this.outFam.length;i++)fid+=Math.round(dot(this.outFam[i],x))*2**i;for(let i=0;i<this.outPay.length;i++)payload+=Math.round(dot(this.outPay[i],x))*2**i;}
    else {for(let i=0;i<this.codeDims.length;i++)if(x[this.codeDims[i]]>0){if(i<this.nFam)fid+=2**i;else payload+=2**(i-this.nFam);}}
    const name=this.outNames[fid];return name?this.token(name,payload):-1;
  }
  stepToken() {
    if(this.state==='halted'||this.state==='fault')return null;
    this.state='running';
    const input=this.nextToken,predicted=this.evaluateToken(input);this.tokens++;
    if(predicted<0){this.state='fault';this.faultCode='invalid neural output';throw new Error(`Neural circuit produced an invalid output token after ${this.tokenName(input)}`);}
    this.lastToken=predicted;this.nextToken=predicted;
    const {name,payload}=this.family(predicted);
    if(FINAL.has(name)&&name!=='R')this.instructions++;
    if(name==='BYTE'&&payload<256){this.outputBytes.push(payload);this.output+=String.fromCharCode(payload);this.options.onOutput?.(String.fromCharCode(payload));}
    if(name==='FLT'){this.faultCode=payload;}
    if(name==='HALT'){this.state=this.faultCode===null?'halted':'fault';this.exitCode=this.regs[10]>>>0;this.options.onExit?.(this.exitCode);}
    if(this.trace.length<200)this.trace.push(predicted);
    return predicted;
  }
  run(budgetTokens=1) {let n=0;while(n<budgetTokens&&this.state!=='halted'&&this.state!=='fault'){this.stepToken();n++;}return n;}
  head(table) {
    const id=typeof table==='number'?table:this.tables[table];
    if(id!=null)return this.nodes[id];
    return this.nodes.find(n=>n.type==='head'&&n.name===table);
  }
  refreshRegisters() {
    const head=this.head('regs.rs1');if(!head)return this.regs;
    const vh=head.vheads[0],C=head.nCols;this.regs[0]=0;
    for(let r=1;r<32;r++){let value=0;for(let bit=0;bit<32;bit++){const cols=this.regCols[bit];const col=Array.isArray(cols)?cols[0]:cols??bit+2;if(vh.state[r*C+col]>0)value+=2**bit;}this.regs[r]=value>>>0;}
    return this.regs;
  }
  read32(address) {
    if(address<0||address>=32768||address%4)return 0;
    const wi=address>>>2,h=(wi&15)|((wi>>>12)<<4),half=(wi>>>4)&1,row=(wi>>>5)&127;
    const node=this.head(`ram${h}`),vh=node?.vheads[half];if(!vh)return 0;let n=0;
    for(let bit=0;bit<32;bit++)if(vh.state[row*node.nCols+2+bit]>0)n+=2**bit;
    return n>>>0;
  }
  read8(address) {return (this.read32(address&~3)>>>((address&3)*8))&255;}
  get memory(){const bytes=new Uint8Array(32768);for(let i=0;i<32768;i+=4){const n=this.read32(i);bytes[i]=n&255;bytes[i+1]=(n>>>8)&255;bytes[i+2]=(n>>>16)&255;bytes[i+3]=n>>>24;}return bytes;}
  summary(){return {engine:'qwenputer-sparse-circuit',layers:11,tokens:this.tokens,instructions:this.instructions,state:this.state,pc:this.pc,registers:Array.from(this.regs),output:this.output,exitCode:this.exitCode,faultCode:this.faultCode};}
}
