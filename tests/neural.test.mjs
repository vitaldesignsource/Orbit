import {readFile,writeFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {availableParallelism} from 'node:os';
import {NeuralRV32} from '../dist/machine/neural.js';
const machineRoot=new URL('../dist/machine/',import.meta.url);
async function json(path){const data=await readFile(new URL(path,machineRoot));return JSON.parse(path.endsWith('.gz')?gunzipSync(data).toString('utf8'):data.toString('utf8'));}
async function validate(cpu,program){
 const started=performance.now();
 try{
  const [snapshot,reference]=await Promise.all([json(program.snapshot),json(program.trace)]);
  const expected=reference.tokens;
  cpu.loadSnapshot(snapshot);
  if(cpu.nextToken!==expected[0])throw Error('Snapshot initial token differs from reference');
  let index=1;
  while(!['halted','fault'].includes(cpu.state)){
   if(index>=expected.length)throw Error('Circuit generated beyond the reference halt');
   const next=cpu.stepToken();
   if(next!==expected[index])throw Error(`Token ${index}: expected ${cpu.tokenName(expected[index])}, got ${cpu.tokenName(next)}`);
   index++;
  }
  if(index!==expected.length)throw Error(`Circuit halted after ${index} of ${expected.length} tokens`);
  if(cpu.state!=='halted')throw Error('Circuit faulted: '+cpu.faultCode);
  if(cpu.exitCode!==reference.exitCode)throw Error(`Exit ${cpu.exitCode} != reference ${reference.exitCode}`);
  if(cpu.output!==reference.output)throw Error('UART output differs from reference');
  if(cpu.exitCode!==0)throw Error('Nonzero official test exit: '+cpu.exitCode);
  return {id:program.id,passed:true,engine:'qwenputer-sparse-neural-circuit',traceMatched:true,tokens:cpu.tokens,instructions:cpu.instructions,exitCode:cpu.exitCode,output:cpu.output,timeMs:Math.round(performance.now()-started)};
 }catch(error){return{id:program.id,passed:false,engine:'qwenputer-sparse-neural-circuit',error:error.message,timeMs:Math.round(performance.now()-started)};}
}
if(!isMainThread){
 const graph=await json(workerData.circuit),cpu=new NeuralRV32(graph);
 for(const program of workerData.programs){const result=await validate(cpu,program);parentPort.postMessage(result);}
}else{
 const manifest=await json('programs.json'),all=process.argv.includes('--all');
 const programs=all?manifest.programs.filter(p=>p.language==='test'):manifest.programs.filter(p=>p.id==='hello'||p.id==='c-hello'||p.id==='rv32ui-add');
 if(!programs.length)throw Error('No neural validation programs found');
 const started=performance.now(),results=[];
 const threads=Math.min(all?4:1,availableParallelism(),programs.length),groups=Array.from({length:threads},()=>[]);
 programs.forEach((p,i)=>groups[i%threads].push(p));
 await Promise.all(groups.map(group=>new Promise((resolve,reject)=>{
  const worker=new Worker(fileURLToPath(import.meta.url),{workerData:{circuit:manifest.circuit,programs:group}});
  worker.on('message',result=>{results.push(result);console.log(`${result.passed?'PASS':'FAIL'} neural ${result.id}: ${result.tokens??0} exact trace tokens${result.error?' — '+result.error:''} [${results.length}/${programs.length}]`);});
  worker.on('error',reject);worker.on('exit',code=>code===0?resolve():reject(Error('Neural validation worker exited '+code)));
 })));
 results.sort((a,b)=>a.id.localeCompare(b.id));
 const report={engine:'qwenputer-sparse-neural-circuit',generatedAt:new Date().toISOString(),tests:results.length,passed:results.filter(r=>r.passed).length,failed:results.filter(r=>!r.passed).length,strictTraceComparison:true,scope:'Included official rv32ui, rv32um, rv32uzba user-level fixtures. No privileged, atomics, compressed or floating-point certification.',timeMs:Math.round(performance.now()-started),results};
 if(process.argv.includes('--write-report'))await writeFile(new URL('neural-results.json',machineRoot),JSON.stringify(report,null,2)+'\n');
 console.log(`Neural circuit: ${report.passed}/${report.tests} passed; ${Math.round(report.timeMs/1000)} seconds; token-by-token comparison against reference traces.`);
 if(report.failed)process.exitCode=1;
}
