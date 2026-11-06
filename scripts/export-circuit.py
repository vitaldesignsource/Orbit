"""Export upstream Qwenputer's sparse neural CPU graph and recurrent RAM snapshots.

No RISC-V semantic evaluator is embedded in these artifacts. Graph values are the
computed FFN/head weights used by the upstream circuit; snapshots are initial
Gated DeltaNet state matrices. Original Qwen3.5-4B language weights are excluded.
"""
import gzip,json,sys,shutil,subprocess,struct,argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('upstream_root',type=Path,help='Checkout of divagr18/qwenputer')
parser.add_argument('output_directory',type=Path,help='Static machine asset directory')
args=parser.parse_args()
ROOT=args.upstream_root.resolve()
sys.path.insert(0,str(ROOT/'experimental/riscv'))
from chipformer.rv import core,ref,elf,spec as S
from chipformer.rv.asm import assemble
from chipformer.q35.ir import FFN
OUT=args.output_directory.resolve()
FIX=OUT/'fixtures'; FIX.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'experimental/riscv/tests/data/rv'
ENV=ROOT/'experimental/riscv/chipformer/rv/env'

def expr(e): return [[int(d),float(w)]for d,w in e.terms.items()]
def chan(c): return {'expr':expr(c.expr),'taps':list(c.taps)}
def chans(ch): return {int(k):chan(v)for k,v in ch.items()}
def vh(v): return {'v':chans(v.v),'a':expr(v.a),'z':{int(k):expr(e)for k,e in v.z.items()},'out':v.out,'beta':None if v.beta is None else expr(v.beta),'into':v.into}
def write(path,obj):
 data=json.dumps(obj,separators=(',',':')).encode()
 temporary=path.with_name(path.name+'.tmp')
 if str(path).endswith('.gz'):
  with temporary.open('wb')as target:
   with gzip.GzipFile(filename='',fileobj=target,mode='wb',mtime=0)as f:f.write(data)
 else:temporary.write_bytes(data)
 temporary.replace(path)
 return len(data),path.stat().st_size
c=core.build();idx={id(n):i for i,n in enumerate(c.nodes)}
nodes=[]
for n in c.nodes:
 if isinstance(n,FFN): nodes.append({'type':'ffn','name':n.name,'out':n.out,'into':n.into,'terms':[[float(co),expr(a),expr(b)]for co,a,b in n.terms]})
 else:nodes.append({'type':'head','name':n.name,'q':chans(n.q),'k':chans(n.k),'vheads':[vh(v)for v in n.vheads()],'nCols':n.n_cols})
graph={'format':'qwenputer-sparse-v1','upstream':'https://github.com/divagr18/qwenputer','upstreamCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dims':[d.name for d in c.dims],'features':[{'dim':d,'name':c.dims[d].name}for d in c.token_fns],'nodes':nodes,'tables':{k:idx[id(v)]for k,v in c.tables.items()},'regCols':c.reg_cols,'pcBits':[expr(e)for e in c.probe['w']],'outFam':[expr(e)for e in c.out_fam],'outPay':[expr(e)for e in c.out_pay],'familySizes':S.FAMILY_SIZES,'controls':S.CONTROL_NAMES,'memoryMap':{'ramBase':S.RAM_BASE,'ramBytes':S.RAM_WORDS*4,'romBase':S.ROM_BASE,'romBytes':S.ROM_WORDS*4,'dromBase':S.DROM_BASE,'dromBytes':S.DROM_WORDS*4}}
print('graph',write(OUT/'neural-circuit.json.gz',graph),flush=True)
def snapshot(prog,id):
 m=ref.Machine.from_program(prog);core.snapshot(c,m);states={}
 for i,n in enumerate(c.nodes):
  if not isinstance(n,FFN)and n.init:
   states[i]=[{int(r):[[int(k),float(v)]for k,v in enumerate(row)if v]for r,row in rows.items()}for rows in n.init]
 snap={'format':'qwenputer-state-v1','program':id,'states':states,'startToken':S.tok('GO'),'pc':m.pc}
 path=FIX/(id+'.snapshot.json.gz');write(path,snap)
 return m
hello='_start: li a0, 72\nli a7, 1\necall\nli a0, 105\necall\nli a0, 10\necall\nli a0, 0\nli a7, 93\necall\n'
(FIX/'hello.s').write_text(hello)
hello_program=assemble(hello)
# A real ELF32 container for the assembled smoke program, for loading in the
# independent ISA reference machine as well as the circuit snapshot path.
code=b''.join(struct.pack('<I',w)for w in hello_program.text)
ident=b'\x7fELF'+bytes([1,1,1,0])+bytes(8)
header=ident+struct.pack('<HHIIIIIHHHHHH',2,243,1,hello_program.entry,52,0,0,52,32,1,0,0,0)
ph=struct.pack('<8I',1,0x100,S.ROM_BASE,S.ROM_BASE,len(code),len(code),5,4)
(FIX/'hello.elf').write_bytes((header+ph).ljust(0x100,b'\0')+code)
m=snapshot(hello_program,'hello');trace=m.run();write(FIX/'hello.trace.json.gz',{'tokens':trace,'output':m.out.decode(),'exitCode':m.exit_code})
programs=[{'id':'hello','name':'Hello, neural computer','language':'assembly','source':'fixtures/hello.s','snapshot':'fixtures/hello.snapshot.json.gz','elf':'fixtures/hello.elf','expectedOutput':'Hi\n','expectedExit':0,'trace':'fixtures/hello.trace.json.gz','description':'A minimal program executed by the compiled neural CPU layers.'}]
for path in sorted(DATA.glob('*.elf')):
 if '.qemu.'in path.name or path.stem in ('calc','calc-bridge','c-gmpmul'):continue
 id=path.stem;prog=elf.program(path);m=snapshot(prog,id);shutil.copyfile(path,FIX/path.name)
 test=id.startswith('rv32');language='test'if test else'python'if id.startswith('mpy-')else'c'
 entry={'id':id,'name':id,'language':language,'snapshot':f'fixtures/{id}.snapshot.json.gz','elf':f'fixtures/{path.name}'}
 if test:
  tokens=m.run();assert m.halted and m.exit_code==0,(id,m.fault,m.exit_code)
  tracepath=f'fixtures/{id}.trace.json.gz'
  write(OUT/tracepath,{'tokens':tokens,'output':m.out.decode(errors='replace'),'exitCode':m.exit_code,'instructions':m.steps})
  entry.update(expectedExit=0,suite='riscv-tests',trace=tracepath,expectedInstructions=m.steps,expectedTokens=len(tokens),description='Upstream official RISC-V ISA test, built with Qwenputer test environment.')
 elif language=='c':
  source=ENV/'progs'/(id[2:]+'.c')
  if source.exists():shutil.copyfile(source,FIX/source.name);entry['source']='fixtures/'+source.name
  entry['description']='Actual GCC-compiled RISC-V C executable.'
 else:
  source=ENV/'mpy/scripts'/(id[4:]+'.py')
  if source.exists():shutil.copyfile(source,FIX/source.name);entry['source']='fixtures/'+source.name
  entry['description']='MicroPython interpreter compiled to RISC-V, executing an embedded Python script.'
  entry['expectedExit']=0
  if id=='mpy-s1':entry['expectedOutput']='1267650600228229401496703205376\n'
 programs.append(entry)
 print('snapshot',id,flush=True)
# Preserve the locally built short C demo on a complete asset regeneration.
local_hello=FIX/'c-hello.elf'
if local_hello.exists():
 prog=elf.program(local_hello);m=snapshot(prog,'c-hello');tokens=m.run()
 assert m.halted and m.exit_code==0 and m.out.decode()=='Hello from C!\n55\n'
 write(FIX/'c-hello.trace.json.gz',{'tokens':tokens,'output':m.out.decode(),'exitCode':m.exit_code,'instructions':m.steps})
 programs.insert(0,{'id':'c-hello','name':'C · Hello + sum','language':'c','snapshot':'fixtures/c-hello.snapshot.json.gz','elf':'fixtures/c-hello.elf','source':'fixtures/hello.c','expectedOutput':m.out.decode(),'expectedExit':0,'expectedInstructions':m.steps,'expectedTokens':len(tokens),'trace':'fixtures/c-hello.trace.json.gz','description':'Locally GCC-compiled freestanding C greeting and sum demo.'})
manifest={'version':1,'defaultProgram':'c-hello'if local_hello.exists()else'hello','circuit':'neural-circuit.json.gz','execution':'sparse neural CPU circuit; original Qwen language-model layers excluded','upstream':graph['upstream'],'upstreamCommit':graph['upstreamCommit'],'programs':programs,'testCount':sum(p['language']=='test'for p in programs)}
write(OUT/'programs.json',manifest)
print('DONE',len(programs),'programs',manifest['testCount'],'official ISA tests',flush=True)
