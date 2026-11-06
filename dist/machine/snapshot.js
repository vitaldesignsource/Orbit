/* ELF32 -> recurrent-state snapshot for the compiled Qwenputer RV32 circuit.
 * Port of chipformer.rv.elf/program and rv.core.snapshot/byte_rows/rom_word.
 * Upstream: divagr18/qwenputer, Copyright 2026 Divyansh Agrawal, MIT.
 */
export const MEMORY_MAP = Object.freeze({ramBase:0,ramBytes:32768,dromBase:0x20000,dromBytes:131072,romBase:0x40000,romBytes:262144});
const BALLAST=16777216,BIT=128,WIDTH=128;
const UOPS=['ILLEGAL','LUI','AUIPC','MISA','ADDI','SLTI','SLTIU','XORI','ORI','ANDI','SLLI','SRLI','SRAI','ADD','SUB','SLL','SLT','SLTU','XOR','SRL','SRA','OR','AND','LB','LH','LW','LBU','LHU','SB','SH','SW','BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR','ECALL','EBREAK','FENCE','MUL','MULH','MULHSU','MULHU','DIV','DIVU','REM','REMU','SH1ADD','SH2ADD','SH3ADD'];
const U=Object.fromEntries(UOPS.map((n,i)=>[n,i]));Object.assign(U,{BMAC:64,BMSB:65,BLD:66,BGET:67,BPRD:68,BSCAN:69});
const sx=(n,b)=>(n<<(32-b))>>(32-b);
const inROM=t=>t%4===0&&t>=0x40000&&t<0x80000;
function pack(op,rd=0,rs1=0,rs2=0,spare=0){return ((U[op]??0)|((spare&1)<<7)|((rd<<3|((spare>>>1)&7))<<8)|((rs1<<3|((spare>>>4)&7))<<16)|((rs2<<3|((spare>>>7)&7))<<24))>>>0;}
/** Predecode static instruction formats exactly as the upstream compiled ROM. */
export function predecodeROM(w,pc){
 w>>>=0;const opc=w&127,rd=(w>>>7)&31,f3=(w>>>12)&7,rs1=(w>>>15)&31,rs2=(w>>>20)&31,f7=w>>>25;
 const ii=sx(w>>>20,12),si=sx(((w>>>25)<<5)|((w>>>7)&31),12),bi=sx(((w>>>31)<<12)|(((w>>>7)&1)<<11)|(((w>>>25)&63)<<5)|(((w>>>8)&15)<<1),13),ji=sx(((w>>>31)<<20)|(((w>>>12)&255)<<12)|(((w>>>20)&1)<<11)|(((w>>>21)&1023)<<1),21);
 let op='ILLEGAL',imm=0;
 if(opc===0x37||opc===0x17){const up=Math.floor(((opc===0x17?pc:0)+(w&0xfffff000))/4096)&0xfffff;return pack(opc===0x37?'LUI':'AUIPC',rd,up&31,(up>>>5)&31,up>>>10);}
 if(opc===0x6f){const target=(pc+ji)>>>0,wi=((target-0x40000)>>2)&65535;return pack('JAL',rd,wi&31,(wi>>>5)&31,(wi>>>10)|((!inROM(target)?1:0)<<6));}
 if(opc===0x67&&f3===0){op='JALR';imm=ii;}
 else if(opc===0x63&&f3!==2&&f3!==3){op=['BEQ','BNE','','','BLT','BGE','BLTU','BGEU'][f3];const off=(bi>>2)&2047,target=(pc+bi)>>>0;return pack(op,off&31,rs1,rs2,(off>>>5)|((!inROM(target)?1:0)<<6));}
 else if(opc===3&&[0,1,2,4,5].includes(f3)){op=['LB','LH','LW','','LBU','LHU'][f3];imm=ii;}
 else if(opc===0x23&&f3<3){const i=si&4095;return pack(['SB','SH','SW'][f3],rs2,rs1,i&31,i>>>5);}
 else if(opc===0x13){if(f3===1&&f7===0){op='SLLI';imm=rs2;}else if(f3===5&&(f7===0||f7===32)){op=f7?'SRAI':'SRLI';imm=rs2;}else if(f3!==1&&f3!==5){op=['ADDI','','SLTI','SLTIU','XORI','','ORI','ANDI'][f3];imm=ii;}}
 else if(opc===0x33){if(f7===0)op=['ADD','SLL','SLT','SLTU','XOR','SRL','OR','AND'][f3];else if(f7===32&&(f3===0||f3===5))op=f3===0?'SUB':'SRA';else if(f7===1)op=['MUL','MULH','MULHSU','MULHU','DIV','DIVU','REM','REMU'][f3];else if(f7===16&&[2,4,6].includes(f3))op={2:'SH1ADD',4:'SH2ADD',6:'SH3ADD'}[f3];}
 else if(opc===0x0b&&f3<5&&f7===0)op=['BMAC','BMSB','BLD','BGET','BSCAN'][f3];
 else if(opc===0x2b&&f3===0){imm=(w>>>20)&127;return pack('BPRD',0,0,imm&31,(imm>>>5)|((rd&7)<<7));}
 else if(w===0x73)return pack('ECALL',0,17,10);
 else if(opc===0x73&&[1,2,3,5,6,7].includes(f3)){const csr=w>>>20,pure=[2,3,6,7].includes(f3)&&rs1===0;if(!pure)return pack('ILLEGAL');if(csr===0x301)return pack('MISA',rd);return [0xc00,0xc01,0xc02,0xc80,0xc81,0xc82,0xf14].includes(csr)?pack('LUI',rd):pack('ILLEGAL');}
 else if(w===0x100073)return pack('EBREAK');
 else if(opc===0x0f)return pack('FENCE');
 if(['ADDI','SLTI','SLTIU','XORI','ORI','ANDI','SLLI','SRLI','SRAI','LB','LH','LW','LBU','LHU','JALR'].includes(op)){const i=imm&4095;return pack(op,rd,rs1,i&31,i>>>5);}
 return op==='ILLEGAL'?pack(op):pack(op,rd,rs1,rs2);
}
function asBytes(buffer){if(buffer instanceof Uint8Array)return buffer;if(ArrayBuffer.isView(buffer))return new Uint8Array(buffer.buffer,buffer.byteOffset,buffer.byteLength);if(buffer instanceof ArrayBuffer)return new Uint8Array(buffer);throw new TypeError('Expected ELF bytes or ArrayBuffer');}
/** Validate/load the static ROM-map RV32 ELF. Compressed instructions are unsupported. */
export function parseELF(buffer){
 const bytes=asBytes(buffer);if(bytes.length<52)throw Error('ELF header is truncated');const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
 if(v.getUint32(0,true)!==0x464c457f||bytes[4]!==1||bytes[5]!==1)throw Error('Only little-endian ELF32 is supported');
 if(v.getUint16(18,true)!==243)throw Error('The ELF must target RISC-V');
 const entry=v.getUint32(24,true),phoff=v.getUint32(28,true),size=v.getUint16(42,true),count=v.getUint16(44,true),flags=v.getUint32(36,true);
 if(flags&1)throw Error('Compressed RISC-V instructions are unsupported; compile with -march=rv32im_zba');
 if(!inROM(entry))throw Error('Entry point must be 4-byte aligned in ROM at 0x40000–0x7ffff');if(size<32||phoff+size*count>bytes.length)throw Error('ELF program headers are truncated');
 const buffers={ram:new Uint8Array(32768),drom:new Uint8Array(131072),rom:new Uint8Array(262144)},used={ram:0,drom:0,rom:0},segments=[];
 for(let i=0;i<count;i++){const p=phoff+i*size;if(v.getUint32(p,true)!==1)continue;const off=v.getUint32(p+4,true),addr=v.getUint32(p+8,true),filesz=v.getUint32(p+16,true),memsz=v.getUint32(p+20,true);if(!memsz)continue;if(filesz>memsz||off+filesz>bytes.length)throw Error('ELF segment is truncated');
  const region=addr+memsz<=32768?'ram':addr>=0x20000&&addr+memsz<=0x40000?'drom':addr>=0x40000&&addr+memsz<=0x80000?'rom':null;if(!region)throw Error(`Segment at 0x${addr.toString(16)} does not fit the neural RAM/DROM/ROM map`);
  const base=region==='ram'?0:region==='drom'?0x20000:0x40000,at=addr-base;buffers[region].fill(0,at,at+memsz);buffers[region].set(bytes.subarray(off,off+filesz),at);used[region]=Math.max(used[region],at+memsz);segments.push({address:addr,fileBytes:filesz,memoryBytes:memsz,region});
 }
 if(!used.rom)throw Error('ELF has no code in the neural ROM region');
 return{entry,flags,buffers,used,segments};
}
function row(ballast=true){const r=new Float64Array(WIDTH);if(ballast){r[0]=BALLAST;r[1]=BALLAST;}return r;}
function allRows(){const rows={};for(let r=0;r<128;r++)rows[r]=row();return rows;}
function counters(n=0){const r=row(false);r[2]=BIT*2**-n;return{0:r,1:row()};}
function byteTable(bytes,nHeads,predecode=false){
 const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),per=bytes.length/4/nHeads,out=[];
 for(let h=0;h<nHeads;h++){const heads=[];for(let x=0;x<2;x++){const rows=allRows();for(let wi=0;wi<per/2;wi++){const abs=h*per+x*(per/2)+wi,raw=view.getUint32(abs*4,true),word=predecode&&raw?predecodeROM(raw,0x40000+abs*4):raw,half=(wi>>>3)&1,rr=(wi>>>4)+64*half,col=2+32*half+4*(wi&7);for(let b=0;b<4;b++)rows[rr][col+b]=(word>>>(8*b))&255;}heads.push(rows);}out.push(heads);}return out;
}
/** Build runtime snapshot. RAM contents are encoded in recurrent head matrices. */
export function buildSnapshot(elfBuffer,circuit,options={}){
 const elf=parseELF(elfBuffer),tables=circuit.tables||circuit.meta?.tables,regCols=circuit.regCols||circuit.meta?.regCols;if(!tables||!regCols)throw Error('Compiled circuit table metadata is required');
 const states={};const put=(name,value)=>{const id=tables[name];if(id==null)throw Error(`Circuit table ${name} is missing`);states[id]=value;};
 const anchor=(elf.entry-0x40000)>>>2,pc=row();for(let b=0;b<16;b++)if((anchor>>>b)&1)pc[2+b]=BIT;put('pc',[{0:pc}]);for(const name of ['count','j','vcnt'])put(name,[counters()]);
 const registers={};for(let r=1;r<32;r++){const value=r===2?0x8000:0,rr=row();for(let b=0;b<32;b++)if((value>>>b)&1)rr[regCols[b][0]]=BIT;registers[r]=rr;}for(const name of ['regs.rs1','regs.rs2','regs.rd'])put(name,[registers]);for(let h=0;h<4;h++)put(`quo${h}`,[{0:row()},{0:row()}]);
 const rom=byteTable(elf.buffers.rom,32,true),drom=byteTable(elf.buffers.drom,16);rom.forEach((value,h)=>put(`rom${h}`,value));drom.forEach((value,h)=>put(`drom${h}`,value));
 for(let port=1;port<=2;port++)for(let h=0;h<4;h++)put(`bn${port}.${h}`,[allRows(),allRows()]);
 const ram=new DataView(elf.buffers.ram.buffer);for(let h=0;h<32;h++){const vheads=[{},{}];for(let wi=0;wi<8192;wi++){if(((wi&15)|((wi>>>12)<<4))!==h)continue;const value=ram.getUint32(wi*4,true);if(!value)continue;const rr=row();for(let b=0;b<32;b++)if((value>>>b)&1)rr[2+b]=BIT;vheads[(wi>>>4)&1][(wi>>>5)&127]=rr;}put(`ram${h}`,vheads);}
 const families=circuit.familySizes||circuit.meta?.familySizes||[['V',32768],['F0',32768],['F1',32768],['F2',4],['J',65536],['R',65536],['BYTE',512],['FLT',8],['HEX',16],['QS',16],['QD',128],['BD',256]],controls=circuit.controls||circuit.meta?.controls||['NT','ST0','ST1','SS0','SS1','HALT','GO','SC'];
 return{format:'qwenputer-browser-snapshot-v1',program:options.name||'Uploaded ELF',states,startToken:families.reduce((s,f)=>s+f[1],0)+controls.indexOf('GO'),pc:elf.entry,elf:{entry:elf.entry,used:elf.used,segments:elf.segments}};
}
