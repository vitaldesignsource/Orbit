import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {CompiledVectorUnit,compileVectorWeights,exportSwiGLU} from '../dist/machine/vector.js';
const unit=new CompiledVectorUnit(),portable=new CompiledVectorUnit({jit:false});
let assertions=0,maxSoftmaxError=0,maxNormError=0,maxSiluError=0;
function close(a,b,tolerance=1e-6,label=''){assert.ok(Number.isFinite(a),`${label}: ${a} is not finite`);assert.ok(Math.abs(a-b)<=tolerance,`${label}: ${a} != ${b} (tol ${tolerance})`);assertions++;}
let seed=0x615beede;function random(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)/4294967296;}
for(let i=0;i<1000;i++){const a=(random()-.5)*100,b=(random()-.5)*100;close(unit.multiplyScalar(a,b),a*b,1e-11,'signed product');close(portable.multiplyScalar(a,b),a*b,1e-11,'portable product');}
for(const x of[0,-1e-8,-.01,-.1,-1,-2,-8,-16,-24,-32,-100]){const expected=Math.exp(Math.max(-32,x));close(unit.expNegative(x),expected,Math.max(1e-15,expected*1e-8),'compiled exp');close(portable.expNegative(x),unit.expNegative(x),1e-14,'portable exp');}
for(const upper of[1,2,3,8,64,512,4096])for(const fraction of[0,.1,.5,.9,1]){const value=1+(upper-1)*fraction;close(unit.reciprocalPositive(value,upper),1/value,1e-12,'Newton reciprocal');}
for(let n of[1,2,8,64,512])for(let trial=0;trial<12;trial++){
 const logits=Float32Array.from({length:n},()=>60*(random()-.5)),actual=unit.softmax(logits),maximum=Math.max(...logits),es=Array.from(logits,x=>Math.exp(x-maximum)),sum=es.reduce((a,b)=>a+b,0);let actualSum=0;
 for(let i=0;i<n;i++){const expected=es[i]/sum,error=Math.abs(actual[i]-expected);maxSoftmaxError=Math.max(maxSoftmaxError,error);close(actual[i],expected,8e-8,'softmax');actualSum+=actual[i];}close(actualSum,1,1e-7,'softmax sum');
}
for(const v of[1e-8,1e-7,1e-5,.1,1,2,64,4096])close(unit.inverseSqrt(v),1/Math.sqrt(v),Math.max(1e-12,1/Math.sqrt(v)*1e-12),'Newton rsqrt');
for(let trial=0;trial<25;trial++){
 const x=Float32Array.from({length:64},()=>12*(random()-.5)),w=Float32Array.from({length:64},()=>random()*2),actual=unit.rmsnorm(x,w),scale=1/Math.sqrt(x.reduce((s,v)=>s+v*v,0)/64+1e-5);
 for(let i=0;i<64;i++){const expected=x[i]*scale*w[i],error=Math.abs(actual[i]-expected);maxNormError=Math.max(maxNormError,error);close(actual[i],expected,2e-7,'rmsnorm');}
 const ln=unit.layernorm(x,w),mean=x.reduce((s,v)=>s+v,0)/64,variance=x.reduce((s,v)=>s+(v-mean)**2,0)/64+1e-5;
 for(let i=0;i<64;i++)close(ln[i],(x[i]-mean)/Math.sqrt(variance)*w[i],2e-7,'layernorm');
}
for(let x=-64;x<=64;x+=.125){const expected=x/(1+Math.exp(-x)),actual=unit.silu(x);maxSiluError=Math.max(maxSiluError,Math.abs(actual-expected));close(actual,expected,2e-10,'SiLU');}
for(const[m,k,n]of[[2,3,4],[1,64,172],[3,8,2]]){
 const a=Float32Array.from({length:m*k},()=>random()-.5),b=Float32Array.from({length:k*n},()=>random()-.5),actual=unit.matmul(a,b,m,k,n);
 for(let row=0;row<m;row++)for(let col=0;col<n;col++){let value=0;for(let inner=0;inner<k;inner++)value+=a[row*k+inner]*b[inner*n+col];close(actual[row*n+col],Math.fround(value),0,'matmul');}
}
{
 const weights=Float32Array.from({length:172*64},()=>random()-.5),x=Float32Array.from({length:64},()=>random()-.5),actual=unit.matvec(weights,x,172,64);
 const portableActual=portable.matvec(weights,x,172,64);
 for(let row=0;row<172;row++){let value=0;for(let col=0;col<64;col++)value+=weights[row*64+col]*x[col];close(actual[row],Math.fround(value),0,'compiled model matrix');close(portableActual[row],actual[row],0,'portable model matrix');}
 const fixed=unit.compileMatrix(weights,172,64).exportWeights();assert.equal(fixed.rows.length,172);assert.equal(fixed.rows[0].nodes[0].terms[0][1].length,64);assertions+=2;
}
{
 const seq=5,heads=2,dim=4,width=heads*dim,Q=Float32Array.from({length:seq*width},()=>random()-.5),K=Float32Array.from({length:seq*width},()=>random()-.5),V=Float32Array.from({length:seq*width},()=>random()-.5),actual=unit.attention(Q,K,V,seq,heads,dim);
 for(let h=0;h<heads;h++)for(let t=0;t<seq;t++){const scores=[];for(let p=0;p<=t;p++){let v=0;for(let d=0;d<dim;d++)v+=Q[t*width+h*dim+d]*K[p*width+h*dim+d];scores.push(v/Math.sqrt(dim));}const max=Math.max(...scores),exps=scores.map(v=>Math.exp(v-max)),sum=exps.reduce((a,b)=>a+b,0);for(let d=0;d<dim;d++){let v=0;for(let p=0;p<=t;p++)v+=exps[p]/sum*V[p*width+h*dim+d];close(actual[t*width+h*dim+d],v,3e-8,'causal attention');}}
}
{
 const weights=compileVectorWeights(),exported=exportSwiGLU(weights.programs.multiply),layer=exported.layers[0],gateScale=exported.gateScale;
 function expr(e,r){return e.reduce((sum,[d,w])=>sum+r[d]*w,0);}
 const silu=x=>x>=0?x/(1+Math.exp(-x)):x*Math.exp(x)/(1+Math.exp(x));
 for(const a of[-20,-.2,0,.2,20])for(const b of[-32,-1,-1e-6,0,1e-6,1,32]){const residual=[1,a,b];let out=0;for(const u of layer.units)out+=expr(u.up,residual)*silu(expr(u.gate,residual))*u.down[0][1];close(out,a*b,Math.abs(a)*2*exported.gateErrorBound+1e-11,'emitted SwiGLU approximation');}
 assert.equal(exported.layers[0].units.length,2);assert.equal(gateScale,2**20);assertions+=2;
 const stored=JSON.parse(await readFile(new URL('../dist/machine/vector-weights.json',import.meta.url),'utf8'));assert.equal(stored.format,weights.format);assert.equal(stored.swiglu.multiply.layers[0].units.length,2);assertions+=2;
}
assert.throws(()=>unit.softmax([]),/Softmax/);assert.throws(()=>unit.softmax([NaN]),/finite/);assert.throws(()=>unit.silu(65),/domain/);assert.throws(()=>unit.rmsnorm([100],[1]),/domain/);assert.throws(()=>unit.rmsnorm([1],[1],1e-12),/epsilon/);assert.throws(()=>unit.matmul([1],[1],1,2,1),/elements/);assertions+=6;
// The production inference module must not delegate nonlinear arithmetic to host functions.
const source=await readFile(new URL('../dist/machine/vector.js',import.meta.url),'utf8');assert.ok(!/Math\.(?:exp|sqrt|tanh|sin|cos)\s*\(/.test(source));assertions++;
console.log(JSON.stringify({passed:true,assertions,maxSoftmaxError,maxNormError,maxSiluError,circuits:Object.keys(compileVectorWeights().programs),summary:unit.summary()},null,2));
