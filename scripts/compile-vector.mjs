#!/usr/bin/env node
import {writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {compileVectorWeights,exportSwiGLU} from '../dist/machine/vector.js';
const output=resolve(process.argv[2]||new URL('../dist/machine/vector-weights.json',import.meta.url).pathname);
const weights=compileVectorWeights({maxLength:512});
weights.swiglu=Object.fromEntries(Object.entries(weights.programs).map(([name,program])=>[name,exportSwiGLU(program)]));
await mkdir(dirname(output),{recursive:true});await writeFile(output,JSON.stringify(weights)+'\n');
console.log(`Compiled ${Object.keys(weights.programs).length} numerical weight circuits into ${output}`);
console.log('Float64 numerical IR, float32 vector outputs. Sparse SwiGLU emission is approximate; no full Qwen checkpoint is emitted.');
