import assert from 'node:assert/strict';
import {RV32} from '../dist/machine/cpu.js';
const R=(f7,f3,rd=3,rs1=1,rs2=2)=>((f7<<25)|(rs2<<20)|(rs1<<15)|(f3<<12)|(rd<<7)|0x33)>>>0;
const I=(imm,f3,rd=3,rs1=1,op=0x13)=>(((imm&4095)<<20)|(rs1<<15)|(f3<<12)|(rd<<7)|op)>>>0;
const S=(imm,f3,rs1=1,rs2=2)=>((((imm&4095)>>>5)<<25)|(rs2<<20)|(rs1<<15)|(f3<<12)|((imm&31)<<7)|0x23)>>>0;
const B=(imm,f3,rs1=1,rs2=2)=>((((imm>>>12)&1)<<31)|(((imm>>>5)&63)<<25)|(rs2<<20)|(rs1<<15)|(f3<<12)|(((imm>>>1)&15)<<8)|(((imm>>>11)&1)<<7)|0x63)>>>0;
const J=(imm,rd=3)=>((((imm>>>20)&1)<<31)|(((imm>>>1)&1023)<<21)|(((imm>>>11)&1)<<20)|(((imm>>>12)&255)<<12)|(rd<<7)|0x6f)>>>0;
const csr=(address,f3,rd=3,rs1=1)=>((address<<20)|(rs1<<15)|(f3<<12)|(rd<<7)|0x73)>>>0;
const cpu=new RV32({ramSize:1024*1024});
function one(instruction,a=0,b=0){cpu.reset();cpu.regs[1]=a;cpu.regs[2]=b;cpu.write32(cpu.base,instruction);assert.equal(cpu.step(),true,cpu.fault?.message);assert.equal(cpu.regs[0],0);return cpu.regs[3];}
const pairs=[[0,0],[1,1],[0xffffffff,2],[0x80000000,0xffffffff],[0x7fffffff,0x80000000],[0xffffffff,0xffffffff],[0x12345678,0xfedcba98],[11,0],[0x80000000,0]];
const ref=(op,a,b)=>{const x=BigInt(a),y=BigInt(b),sx=BigInt(a|0),sy=BigInt(b|0);switch(op){case 0:return Number(BigInt.asUintN(32,x*y));case 1:return Number(BigInt.asUintN(64,sx*sy)>>32n);case 2:return Number(BigInt.asUintN(64,sx*y)>>32n);case 3:return Number(x*y>>32n);case 4:return b===0?0xffffffff:Number(BigInt.asUintN(32,sx/sy));case 5:return b===0?0xffffffff:Number(x/y);case 6:return b===0?a:Number(BigInt.asUintN(32,sx%sy));case 7:return b===0?a:Number(x%y);}};
let seed=0x12345678;function random(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;}
for(let n=0;n<250;n++)pairs.push([random(),random()]);
for(const[a,b]of pairs){
 assert.equal(one(R(0,0),a,b),(a+b)>>>0);assert.equal(one(R(32,0),a,b),(a-b)>>>0);
 assert.equal(one(R(0,1),a,b),(a<<(b&31))>>>0);assert.equal(one(R(0,2),a,b),(a|0)<(b|0)?1:0);assert.equal(one(R(0,3),a,b),a<b?1:0);
 assert.equal(one(R(0,4),a,b),(a^b)>>>0);assert.equal(one(R(0,5),a,b),a>>>(b&31));assert.equal(one(R(32,5),a,b),((a|0)>>(b&31))>>>0);assert.equal(one(R(0,6),a,b),(a|b)>>>0);assert.equal(one(R(0,7),a,b),(a&b)>>>0);
 for(let op=0;op<8;op++)assert.equal(one(R(1,op),a,b),ref(op,a,b),`M funct3=${op} a=${a} b=${b}`);
}
for(const imm of[-2048,-33,-1,0,1,31,2047]){assert.equal(one(I(imm,0),0xffffffff),(0xffffffff+imm)>>>0);assert.equal(one(I(imm,2),0x80000000),1);assert.equal(one(I(imm,3),0xffffffff),0);}
for(const shift of[0,1,15,31]){assert.equal(one(I(shift,1),0x80000001),(0x80000001<<shift)>>>0);assert.equal(one(I(shift,5),0x80000001),0x80000001>>>shift);assert.equal(one(I(1024+shift,5),0x80000001),(-2147483647>>shift)>>>0);}
one(I(1,0,0),123);assert.equal(cpu.regs[0],0);
cpu.reset();cpu.regs[1]=cpu.base+256;cpu.regs[2]=0x87654321;cpu.write32(cpu.base,S(-3,2));cpu.step();assert.equal(cpu.read32(cpu.base+253),0x87654321);assert.equal(cpu.read8(cpu.base+253),0x21);
for(const[f3,expected]of[[0,0xffffff87],[1,0xffff8787],[2,0x87878787],[4,0x87],[5,0x8787]]){cpu.reset();cpu.regs[1]=cpu.base+256;cpu.write32(cpu.base+256,0x87878787);cpu.write32(cpu.base,I(0,f3,3,1,0x03));cpu.step();assert.equal(cpu.regs[3],expected);}
for(const[f3,a,b,take]of[[0,4,4,true],[0,4,5,false],[1,4,5,true],[4,0xffffffff,0,true],[5,1,0xffffffff,true],[6,0xffffffff,0,false],[7,0xffffffff,0,true]]){for(const offset of[12,-16,2048,-2048]){cpu.reset();cpu.pc=cpu.base+64;cpu.regs[1]=a;cpu.regs[2]=b;cpu.write32(cpu.pc,B(offset,f3));cpu.step();assert.equal(cpu.pc,(cpu.base+64+(take?offset:4))>>>0);}}
for(const offset of[4,2048,-2048,0x7fffc,-0x7fffc]){cpu.reset();cpu.pc=cpu.base+0x80000;cpu.write32(cpu.pc,J(offset));const old=cpu.pc;cpu.step();assert.equal(cpu.pc,(old+offset)>>>0);assert.equal(cpu.regs[3],old+4);}
cpu.reset();cpu.regs[1]=cpu.base+33;cpu.write32(cpu.base,I(-1,0,3,1,0x67));cpu.step();assert.equal(cpu.pc,cpu.base+32);assert.equal(cpu.regs[3],cpu.base+4);
cpu.reset();cpu.write32(cpu.base,0xffffffff);cpu.step();assert.equal(cpu.state,'faulted');assert.equal(cpu.fault.cause,2);
cpu.reset();cpu.csrs.set(0x305,cpu.base+128);cpu.write32(cpu.base,0xffffffff);cpu.step();assert.equal(cpu.pc,cpu.base+128);assert.equal(cpu.csrs.get(0x341),cpu.base);assert.equal(cpu.csrs.get(0x342),2);cpu.write32(cpu.pc,0x30200073);cpu.step();assert.equal(cpu.pc,cpu.base);
cpu.reset();cpu.regs[1]=0x1234;cpu.write32(cpu.base,csr(0x340,1));cpu.step();assert.equal(cpu.regs[3],0);assert.equal(cpu.csrs.get(0x340),0x1234);cpu.write32(cpu.pc,csr(0xc00,2,3,0));cpu.step();assert.ok(cpu.regs[3]>=1);cpu.write32(cpu.pc,csr(0xc00,1));cpu.step();assert.equal(cpu.fault.cause,2);
let output='',exit;const uart=new RV32({onOutput:s=>output+=s,onExit:c=>exit=c});uart.enqueueInput('A\n');assert.equal(uart.read8(0x10000005)&1,1);assert.equal(uart.read8(0x10000000),65);assert.equal(uart.read8(0x10000000),10);assert.equal(uart.read8(0x10000005)&1,0);uart.write8(0x10000000,66);assert.equal(output,'B');uart.write32(0x10000004,0);assert.equal(exit,0);assert.equal(uart.state,'halted');
const image=new Uint8Array(128);const v=new DataView(image.buffer);v.setUint32(0,0x7f454c46,false);image[4]=image[5]=1;v.setUint16(18,243,true);v.setUint32(24,0x80000000,true);v.setUint32(28,52,true);v.setUint16(42,32,true);v.setUint16(44,1,true);v.setUint32(52,1,true);v.setUint32(56,96,true);v.setUint32(60,0x80000000,true);v.setUint32(64,0x80000000,true);v.setUint32(68,4,true);v.setUint32(72,16,true);v.setUint32(96,0x00100073,true);const info=cpu.loadELF(image);assert.equal(info.entry,0x80000000);assert.equal(cpu.read32(cpu.base),0x00100073);assert.equal(cpu.read32(cpu.base+4),0);cpu.run(10);assert.equal(cpu.state,'halted');assert.equal(cpu.exitReason,'breakpoint');assert.throws(()=>cpu.loadELF(new Uint8Array(12)),/Truncated/);
console.log(`RV32IM checks passed: ${pairs.length*18+100}+ arithmetic/control/memory/trap/CSR/UART/ELF assertions.`);
