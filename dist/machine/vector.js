/*
 * Orbit numerical vector circuit compiler and executor.
 *
 * Each circuit node is a fixed weighted sum of a * ReLU(b), the numerical IR
 * used by Qwenputer. Products use two signed gates. exp, reciprocal, rsqrt,
 * SiLU and softmax are compositions of those nodes; the execution path never
 * delegates nonlinear arithmetic to host Math.exp/Math.sqrt/Math.tanh.
 *
 * This is a floating-point numerical IR extension. It is not a merged Qwen
 * checkpoint and does not inherit the integer CPU's bf16 exactness proof.
 * exportSwiGLU() provides the actual up/gate/down sparse projection weights
 * for a Qwen-style SwiGLU target, with the explicit gate approximation below.
 */

const FORMAT='orbit-vector-weight-circuit-v1';
const GATE_SCALE=2**20;
const EXP_LIMIT=32;
const MAX_VARIANCE=4096;
const MIN_EPSILON=1e-8;
const MAX_VECTOR=4096;
const ONE=[[0,1]];
const variable=dim=>[[dim,1]];
const scaled=(expr,scale)=>expr.map(([dim,w])=>[dim,w*scale]);
const plus=(...exprs)=>{const terms=new Map();for(const e of exprs)for(const[d,w]of e)terms.set(d,(terms.get(d)||0)+w);return [...terms].filter(([,w])=>w!==0);};
const constant=x=>x===0?[]:[[0,x]];
const finite=(value,label='input')=>{if(!Number.isFinite(value))throw new RangeError(`${label} must be finite`);return value;};
function checkArray(array,size,label){if(!array||array.length!==size)throw new RangeError(`${label} must have ${size} elements`);for(let i=0;i<size;i++)finite(array[i],label);}
function dotExpr(expr,residual){let sum=0;for(const[d,w]of expr)sum+=residual[d]*w;return sum;}

class Builder {
 constructor(name,inputs){this.name=name;this.inputs=inputs;this.nodes=[];this.dims=inputs+1;}
 input(index){return variable(index+1);}
 node(terms,label){const out=this.dims++;this.nodes.push({out,name:label||`v${out}`,terms});return variable(out);}
 affine(expr,label){return this.node([[1,expr,ONE]],label);}
 relu(expr,label){return this.node([[1,ONE,expr]],label);}
 mul(a,b,label){return this.node([[1,a,b],[-1,a,scaled(b,-1)]],label);}
 add(a,b,label){return this.affine(plus(a,b),label);}
 max(a,b,label){return this.add(b,this.relu(plus(a,scaled(b,-1))),label);}
 clamp(expr,lo,hi){const lower=this.add(constant(lo),this.relu(plus(expr,constant(-lo))));return this.add(lower,scaled(this.relu(plus(lower,constant(-hi))),-1));}
 expNegative(expr){
  const bounded=this.clamp(expr,-EXP_LIMIT,0),y=this.affine(scaled(bounded,1/EXP_LIMIT),'exp.range');
  let factorial=1;const coeff=[1];for(let k=1;k<=14;k++){factorial*=k;coeff.push(1/factorial);}
  let p=constant(coeff[14]);for(let k=13;k>=0;k--)p=this.add(this.mul(p,y),constant(coeff[k]),`exp.horner${k}`);
  for(let k=0;k<5;k++)p=this.mul(p,p,`exp.square${k}`);
  return p;
 }
 reciprocal(expr,upper,iterations){
  const y=this.affine(scaled(expr,1/upper),'reciprocal.range');let r=constant(1);
  for(let k=0;k<iterations;k++)r=this.mul(r,plus(constant(2),scaled(this.mul(y,r),-1)),`reciprocal.newton${k}`);
  return this.affine(scaled(r,1/upper),'reciprocal.unscale');
 }
 rsqrt(expr){
  const y=this.affine(scaled(expr,1/MAX_VARIANCE),'rsqrt.range');let r=constant(1);
  for(let k=0;k<56;k++){const square=this.mul(r,r),yr2=this.mul(y,square);r=this.mul(scaled(r,.5),plus(constant(3),scaled(yr2,-1)),`rsqrt.newton${k}`);}
  return this.affine(scaled(r,1/64),'rsqrt.unscale');
 }
 finish(outputs,domain={}){return {name:this.name,inputs:this.inputs,dims:this.dims,nodes:this.nodes,outputs:outputs.map(e=>e[0]?.[0]??0),outputExpressions:outputs,domain};}
}

function exprSource(expr){return expr.length?'('+expr.map(([d,w])=>w===1?`r${d}`:`r${d}*(${w})`).join('+')+')':'0';}
function compileScalar(program,jit=true){
 let kernel;
 if(jit){try{
  let source='const r0=1;';for(let i=0;i<program.inputs;i++)source+=`const r${i+1}=a${i};`;
  for(const node of program.nodes){const parts=[];for(let i=0;i<node.terms.length;i++){const[w,a,b]=node.terms[i];const g=`g${node.out}_${i}`;source+=`const ${g}=${exprSource(b)};`;parts.push(`(${g}>0?(${w})*${exprSource(a)}*${g}:0)`);}source+=`const r${node.out}=${parts.join('+')||'0'};`;}
  source+=`return ${exprSource(program.outputExpressions[0])};`;
  kernel=new Function(...Array.from({length:program.inputs},(_,i)=>`a${i}`),source);
 }catch{}}
 if(!kernel){const residual=new Float64Array(program.dims);kernel=(...args)=>{residual.fill(0);residual[0]=1;for(let i=0;i<program.inputs;i++)residual[i+1]=args[i];for(const node of program.nodes){let sum=0;for(const[w,a,b]of node.terms){const gate=dotExpr(b,residual);if(gate>0)sum+=w*dotExpr(a,residual)*gate;}residual[node.out]=sum;}return dotExpr(program.outputExpressions[0],residual);};}
 return kernel;
}

function scalarPrograms(){
 const result={};let b;
 b=new Builder('multiply',2);result.multiply=b.finish([b.mul(b.input(0),b.input(1))]);
 b=new Builder('mac',3);result.mac=b.finish([b.node([[1,b.input(0),ONE],[1,b.input(1),b.input(2)],[-1,b.input(1),scaled(b.input(2),-1)]],'multiply-accumulate')]);
 b=new Builder('add',2);result.add=b.finish([b.add(b.input(0),b.input(1))]);
 b=new Builder('relu',1);result.relu=b.finish([b.relu(b.input(0))]);
 b=new Builder('max',2);result.max=b.finish([b.max(b.input(0),b.input(1))]);
 b=new Builder('expNegative',1);result.expNegative=b.finish([b.expNegative(b.input(0))],{input:[-EXP_LIMIT,0],clamped:true,degree:14,squarings:5});
 b=new Builder('inverseSqrt',1);result.inverseSqrt=b.finish([b.rsqrt(b.input(0))],{input:[MIN_EPSILON,MAX_VARIANCE],iterations:56});
 b=new Builder('silu',1);const x=b.input(0),positive=b.relu(x),negative=b.relu(scaled(x,-1));const absolute=b.add(positive,negative);const e=b.expNegative(scaled(absolute,-1));const inv=b.reciprocal(b.add(constant(1),e),2,8);result.silu=b.finish([b.mul(inv,plus(positive,scaled(b.mul(negative,e),-1)))],{input:[-64,64],tailClippedAt:EXP_LIMIT});
 return result;
}

function reciprocalProgram(upper,iterations){const b=new Builder(`reciprocal-${upper}-${iterations}`,1);return b.finish([b.reciprocal(b.input(0),upper,iterations)],{input:[1,upper],iterations});}

/** Emit sparse Qwen-style SwiGLU projections, with approximation metadata. */
export function exportSwiGLU(program,{gateScale=GATE_SCALE}={}){
 const layers=program.nodes.map(node=>({name:node.name,out:node.out,units:node.terms.map(([coefficient,a,b])=>({up:scaled(a,1/gateScale),gate:scaled(b,gateScale),down:[[node.out,coefficient]]}))}));
 return {format:'orbit-sparse-swiglu-v1',name:program.name,inputDims:program.inputs+1,residualDims:program.dims,layers,outputs:program.outputExpressions,precision:'float32 target; float64 numerical IR executor',gateScale,gateErrorBound:0.279/gateScale,notes:'SiLU(K*b)/K approximates ReLU(b). Per-unit absolute output error is at most |coefficient*a|*0.279/K before target rounding. This is not a bf16 exactness guarantee or a merged Hugging Face checkpoint.'};
}

export function compileVectorWeights({maxLength=512}={}){
 if(!Number.isInteger(maxLength)||maxLength<1||maxLength>MAX_VECTOR)throw new RangeError('maxLength out of range');
 const programs=scalarPrograms();const iterations=Math.ceil(Math.log2(maxLength*40));programs[`reciprocal-${maxLength}`]=reciprocalProgram(maxLength,iterations);
 return {format:FORMAT,version:1,name:'Orbit compiled vector unit',target:'Qwen-style weighted SwiGLU numerical circuit IR',precision:'float64 circuit arithmetic; float32 vector outputs',limits:{maxLength,expMinimum:-EXP_LIMIT,maxVariance:MAX_VARIANCE,minEpsilon:MIN_EPSILON,siluInput:[-64,64]},programs,swigluEmission:{gateScale:GATE_SCALE,perUnitReluAbsoluteErrorBound:0.279/GATE_SCALE,checkpointMerged:false},description:'Fixed up/gate/down coefficients implement arithmetic and nonlinear approximations. Host code schedules vector addresses and loops; scalar arithmetic is evaluated by the compiled weight circuits.'};
}

export class CompiledVectorUnit {
 constructor({jit=true,maxLength=512,weights=null}={}){
  this.weights=weights??compileVectorWeights({maxLength});if(this.weights.format!==FORMAT)throw new Error('Unsupported vector circuit format');
  this.jit=jit;this.maxLength=this.weights.limits.maxLength;this.programs=new Map(Object.entries(this.weights.programs));this.kernels=new Map();
  for(const[name,program]of this.programs)this.kernels.set(name,compileScalar(program,jit));
  this.matrixCache=new WeakMap();this.stats={scalarCircuitCalls:0,matrixProducts:0,attentionCalls:0,softmaxCalls:0,normalizationCalls:0,linearCoefficientsCompiled:0};
  this.mulKernel=this.kernels.get('multiply');this.macKernel=this.kernels.get('mac');this.addKernel=this.kernels.get('add');this.maxKernel=this.kernels.get('max');
 }
 exportWeights(){return {...this.weights,programs:Object.fromEntries(this.programs)};}
 exportSwiGLU(name){const p=this.programs.get(name);if(!p)throw new Error('Unknown vector circuit '+name);return exportSwiGLU(p);}
 summary(){return {engine:'compiled-weighted-vector-circuit',precision:this.weights.precision,fullQwenCheckpoint:false,gateTargetApproximate:true,...this.stats};}
 resetStats(){for(const k of Object.keys(this.stats))this.stats[k]=0;}
 multiplyScalar(a,b){finite(a);finite(b);this.stats.scalarCircuitCalls++;return this.mulKernel(a,b);}
 addScalar(a,b){finite(a);finite(b);this.stats.scalarCircuitCalls++;return this.addKernel(a,b);}
 dot(a,b){if(a.length!==b.length)throw new RangeError('Dot dimensions differ');let value=0;for(let i=0;i<a.length;i++){finite(a[i]);finite(b[i]);value=this.macKernel(value,a[i],b[i]);}this.stats.scalarCircuitCalls+=a.length;return value;}
 multiply(a,b){if(typeof a==='number'&&typeof b==='number')return this.multiplyScalar(a,b);if(a.length!==b.length)throw new RangeError('Vector dimensions differ');const out=new Float32Array(a.length);for(let i=0;i<a.length;i++)out[i]=this.multiplyScalar(a[i],b[i]);return out;}
 add(a,b){if(typeof a==='number'&&typeof b==='number')return this.addScalar(a,b);if(a.length!==b.length)throw new RangeError('Vector dimensions differ');const out=new Float32Array(a.length);for(let i=0;i<a.length;i++)out[i]=this.addScalar(a[i],b[i]);return out;}
 relu(x){if(typeof x==='number'){finite(x);this.stats.scalarCircuitCalls++;return this.kernels.get('relu')(x);}return Float32Array.from(x,v=>this.relu(v));}
 expNegative(x){finite(x);this.stats.scalarCircuitCalls++;return this.kernels.get('expNegative')(x);}
 reciprocalPositive(value,upper){finite(value);if(value<1||value>upper||upper>MAX_VECTOR)throw new RangeError('Reciprocal input outside compiled domain');const iterations=Math.ceil(Math.log2(upper*40)),name=`reciprocal-${upper}-${iterations}`;if(!this.kernels.has(name)){const p=reciprocalProgram(upper,iterations);this.programs.set(name,p);this.kernels.set(name,compileScalar(p,this.jit));}this.stats.scalarCircuitCalls++;return this.kernels.get(name)(value);}
 silu(x){if(typeof x==='number'){finite(x);if(Math.abs(x)>64)throw new RangeError('SiLU input outside compiled [-64,64] domain');this.stats.scalarCircuitCalls++;return this.kernels.get('silu')(x);}return Float32Array.from(x,v=>this.silu(v));}
 softmax(logits){
  const n=logits.length;if(!n||n>this.maxLength)throw new RangeError(`Softmax length must be 1..${this.maxLength}`);checkArray(logits,n,'logits');
  let maximum=logits[0];for(let i=1;i<n;i++)maximum=this.maxKernel(maximum,logits[i]);
  const exponents=new Float64Array(n);let sum=0;for(let i=0;i<n;i++){const shifted=this.addKernel(logits[i],-maximum),e=this.kernels.get('expNegative')(shifted);exponents[i]=e;sum=this.addKernel(sum,e);}
  const inv=this.reciprocalPositive(sum,n),out=new Float32Array(n);for(let i=0;i<n;i++)out[i]=this.mulKernel(exponents[i],inv);
  this.stats.scalarCircuitCalls+=n*4;this.stats.softmaxCalls++;return out;
 }
 inverseSqrt(value){finite(value);if(value<MIN_EPSILON||value>MAX_VARIANCE)throw new RangeError(`Normalization variance outside compiled [${MIN_EPSILON},${MAX_VARIANCE}] domain`);this.stats.scalarCircuitCalls++;return this.kernels.get('inverseSqrt')(value);}
 rmsnorm(x,weight,epsilon=1e-5){
  const n=x.length;if(!n||n>MAX_VECTOR)throw new RangeError('RMSNorm vector length out of range');checkArray(x,n,'normalization input');if(weight)checkArray(weight,n,'normalization weights');if(!Number.isFinite(epsilon)||epsilon<MIN_EPSILON)throw new RangeError('Normalization epsilon below compiled minimum');
  let sum=0;for(const value of x)sum=this.macKernel(sum,value,value);const variance=this.addKernel(this.mulKernel(sum,1/n),epsilon),scale=this.inverseSqrt(variance),out=new Float32Array(n);
  for(let i=0;i<n;i++)out[i]=this.mulKernel(this.mulKernel(x[i],scale),weight?weight[i]:1);this.stats.scalarCircuitCalls+=n*3+2;this.stats.normalizationCalls++;return out;
 }
 layernorm(x,weight,bias,epsilon=1e-5){
  const n=x.length;if(!n)throw new RangeError('LayerNorm requires values');checkArray(x,n,'normalization input');if(weight)checkArray(weight,n,'normalization weights');if(bias)checkArray(bias,n,'normalization bias');let mean=0;for(const value of x)mean=this.addKernel(mean,value);mean=this.mulKernel(mean,1/n);const centered=Float64Array.from(x,value=>this.addKernel(value,-mean));const result=this.rmsnorm(centered,weight,epsilon);if(bias)for(let i=0;i<n;i++)result[i]=this.addKernel(result[i],bias[i]);return result;
 }
 matmul(A,B,m,k,n){
  for(const value of[m,k,n])if(!Number.isInteger(value)||value<1)throw new RangeError('Invalid matrix dimensions');checkArray(A,m*k,'left matrix');checkArray(B,k*n,'right matrix');const out=new Float32Array(m*n);
  for(let row=0;row<m;row++)for(let col=0;col<n;col++){let value=0;for(let inner=0;inner<k;inner++)value=this.macKernel(value,A[row*k+inner],B[inner*n+col]);out[row*n+col]=value;}
  this.stats.scalarCircuitCalls+=m*k*n;this.stats.matrixProducts++;return out;
 }
 /** Compile immutable trained matrix coefficients once into affine weight rows. */
 compileMatrix(weights,outDim,inDim){
  checkArray(weights,outDim*inDim,'matrix weights');const rows=[];const kernelRows=[];
  for(let row=0;row<outDim;row++){
   const expr=[];for(let col=0;col<inDim;col++){const w=Number(weights[row*inDim+col]);if(w!==0)expr.push([col+1,w]);}
   const program={name:`matrix.row${row}`,inputs:inDim,dims:inDim+2,nodes:[{name:`row${row}.linear`,out:inDim+1,terms:[[1,expr,ONE]]}],outputs:[inDim+1],outputExpressions:[variable(inDim+1)]};rows.push(program);
   let fn;if(this.jit){try{const source=expr.map(([dim,w])=>`x[${dim-1}]*(${w})`).join('+')||'0';fn=new Function('x',`return ${source};`);}catch{}}
   if(!fn)fn=x=>{let sum=0;for(const[dim,w]of expr)sum+=x[dim-1]*w;return sum;};kernelRows.push(fn);
  }
  this.stats.linearCoefficientsCompiled+=weights.length;
  return {format:'orbit-compiled-affine-matrix-v1',outDim,inDim,rows,run:(x,out=new Float32Array(outDim))=>{for(let row=0;row<outDim;row++)out[row]=kernelRows[row](x);return out;},exportWeights:()=>({format:'orbit-compiled-affine-matrix-v1',outDim,inDim,rows})};
 }
 matvec(weights,x,outDim,inDim){
  if(!Number.isInteger(outDim)||!Number.isInteger(inDim)||outDim<1||inDim<1)throw new RangeError('Invalid matrix dimensions');checkArray(x,inDim,'matrix vector');
  let cached=this.matrixCache.get(weights);if(!cached||cached.outDim!==outDim||cached.inDim!==inDim){cached=this.compileMatrix(weights,outDim,inDim);this.matrixCache.set(weights,cached);}this.stats.scalarCircuitCalls+=outDim;this.stats.matrixProducts++;return cached.run(x);
 }
 attention(Q,K,V,seq,heads,headDim,{causal=true}={}){
  const width=heads*headDim;checkArray(Q,seq*width,'queries');checkArray(K,seq*width,'keys');checkArray(V,seq*width,'values');if(seq>this.maxLength)throw new RangeError('Attention sequence exceeds compiled softmax length');
  const out=new Float32Array(seq*width),scale=this.inverseSqrt(headDim);
  for(let head=0;head<heads;head++)for(let time=0;time<seq;time++){
   const length=causal?time+1:seq,scores=new Float64Array(length);
   for(let pos=0;pos<length;pos++){let score=0;for(let i=0;i<headDim;i++)score=this.macKernel(score,Q[time*width+head*headDim+i],K[pos*width+head*headDim+i]);scores[pos]=this.mulKernel(score,scale);}
   const probs=this.softmax(scores);for(let i=0;i<headDim;i++){let value=0;for(let pos=0;pos<length;pos++)value=this.macKernel(value,probs[pos],V[pos*width+head*headDim+i]);out[time*width+head*headDim+i]=value;}
  }
  this.stats.attentionCalls++;return out;
 }
}
