import {readdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {resolve,join} from 'node:path';
import {RV32} from '../dist/machine/cpu.js';
const dir=process.argv[2]?resolve(process.argv[2]):fileURLToPath(new URL('../dist/machine/fixtures/',import.meta.url));
const reportPath=new URL('../dist/machine/reference-results.json',import.meta.url);
const files=(await readdir(dir)).filter(f=>/^rv32.*\.elf$/.test(f)&&!f.endsWith('.qemu.elf')).sort();
const results=[];
for(const file of files){let output='';const cpu=new RV32({memoryMap:'qwenputer',onOutput:s=>output+=s});let elapsed=performance.now();try{const fixture=await readFile(join(dir,file));cpu.loadELF(fixture);cpu.run(1000000);results.push({name:file.replace(/\.elf$/,''),sha256:createHash('sha256').update(fixture).digest('hex'),status:cpu.state==='halted'&&cpu.exitCode===0?'pass':'fail',exitCode:cpu.exitCode,state:cpu.state,instructions:cpu.instructions,elapsedMs:Math.round((performance.now()-elapsed)*100)/100,...(cpu.fault?{fault:cpu.fault}:{}),...(output?{output}:{})});}catch(error){results.push({name:file,status:'error',error:error.message});}}
const programs=[];
for(const name of ['c-fib','c-qsort','c-sudoku','mpy-s1','mpy-s5']){let output='';const cpu=new RV32({memoryMap:'qwenputer',onOutput:s=>output+=s});const fixture=await readFile(join(dir,`${name}.elf`));cpu.loadELF(fixture);const started=performance.now();while(cpu.instructions<10000000&&!['halted','faulted'].includes(cpu.state))cpu.run(100000);programs.push({name,sha256:createHash('sha256').update(fixture).digest('hex'),status:cpu.state==='halted'&&cpu.exitCode===0?'pass':'fail',exitCode:cpu.exitCode,state:cpu.state,instructions:cpu.instructions,elapsedMs:Math.round((performance.now()-started)*100)/100,output,...(cpu.fault?{fault:cpu.fault}:{})});}
const report={engine:'deterministic-reference-rv32im',memoryMap:'qwenputer',neuralWeights:false,scope:'Qwenputer repository supplied rv32ui, rv32um and rv32uzba ELF fixtures. These runs verify the JavaScript reference interpreter, not Qwen3.5 model execution.',source:'https://github.com/divagr18/qwenputer',generatedAt:new Date().toISOString(),passed:results.filter(r=>r.status==='pass').length,total:results.length,subsets:{rv32ui:results.filter(r=>r.name.startsWith('rv32ui')).length,rv32um:results.filter(r=>r.name.startsWith('rv32um')).length,rv32uzba:results.filter(r=>r.name.startsWith('rv32uzba')).length},results,programs};
await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,total:report.total,failures:results.filter(r=>r.status!=='pass')},null,2));
if(!report.total||report.passed!==report.total||programs.some(r=>r.status!=='pass'))process.exitCode=1;
