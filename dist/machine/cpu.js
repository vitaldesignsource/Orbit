/** Deterministic RV32IM interpreter. RAM and registers are conventional typed arrays. */
export class RV32 {
  constructor({onOutput=()=>{},onExit=()=>{},onFault=()=>{},ramSize,memoryMap='standard'}={}) {
    if(!['standard','qwenputer'].includes(memoryMap))throw new Error('Unknown memory map');
    this.mode=memoryMap;ramSize??=memoryMap==='qwenputer'?0x80000:16*1024*1024;
    if(!Number.isInteger(ramSize)||ramSize<4096||ramSize>256*1024*1024)throw new RangeError('RAM must be 4 KiB–256 MiB');
    this.base=memoryMap==='qwenputer'?0:0x80000000;this.stackTop=memoryMap==='qwenputer'?0x8000:this.base+ramSize;this.ramSize=ramSize;this.memory=new Uint8Array(ramSize);this.regs=new Uint32Array(32);
    this.onOutput=onOutput;this.onExit=onExit;this.onFault=onFault;this.symbols={};this.reset();
  }
  reset({clearMemory=true}={}) {
    if(clearMemory)this.memory.fill(0);this.regs.fill(0);this.regs[2]=(this.stackTop-16)>>>0;
    this.pc=this.base;this.instructions=0;this.cycles=0;this.state='ready';this.exitCode=null;this.exitReason=null;this.fault=null;
    this.input=[];this.inputAt=0;this.tohost=0;this.symbols={};this.privilege=3;this.uartControl=0;
    this.csrs=new Map([[0x300,0],[0x301,0x40101100],[0x302,0],[0x303,0],[0x304,0],[0x305,0],[0x306,0],[0x340,0],[0x341,0],[0x342,0],[0x343,0],[0x344,0],[0xf11,0],[0xf12,0],[0xf13,1],[0xf14,0]]);
    return this;
  }
  enqueueInput(text) {
    const bytes=typeof text==='string'?new TextEncoder().encode(text):text;
    for(const byte of bytes)this.input.push(byte&255);
    if(this.state==='waiting')this.state='running';
  }
  _offset(address,length=1) {
    const offset=(address>>>0)-this.base;
    if(offset<0||offset+length>this.memory.length)throw new MemoryFault(address>>>0,length);
    return offset;
  }
  read8(address) {
    address>>>=0;
    if(address>=0x10000000&&address<0x10000008) {
      const reg=address-0x10000000;
      if(reg===0){if(this.uartControl&0x80)return 0;return this.inputAt<this.input.length?this.input[this.inputAt++]:0;}
      if(reg===5)return 0x60|(this.inputAt<this.input.length?1:0);
      if(reg===3)return this.uartControl;return 0;
    }
    return this.memory[this._offset(address)];
  }
  read16(address){return(this.read8(address)|(this.read8((address+1)>>>0)<<8))>>>0;}
  read32(address) {
    address>>>=0;
    const off=address-this.base;
    if(off>=0&&off+4<=this.memory.length){const m=this.memory;return(m[off]|m[off+1]<<8|m[off+2]<<16|m[off+3]<<24)>>>0;}
    return(this.read8(address)|this.read8(address+1)<<8|this.read8(address+2)<<16|this.read8(address+3)<<24)>>>0;
  }
  write8(address,value) {
    address>>>=0;value&=255;
    if(address>=0x10000000&&address<0x10000008){const reg=address-0x10000000;if(reg===0&&!(this.uartControl&0x80))this.onOutput(String.fromCharCode(value));else if(reg===3)this.uartControl=value;return;}
    this.memory[this._offset(address)]=value;
    if(this.tohost&&address===this.tohost)this._checkTohost();
  }
  write16(address,value){this.write8(address,value);this.write8((address+1)>>>0,value>>>8);}
  write32(address,value) {
    address>>>=0;value>>>=0;
    if(address===0x10000004){this._halt(value,'exit');return;}
    const off=address-this.base;
    if(off>=0&&off+4<=this.memory.length){const m=this.memory;m[off]=value&255;m[off+1]=(value>>>8)&255;m[off+2]=(value>>>16)&255;m[off+3]=value>>>24;if(this.tohost&&address===this.tohost)this._checkTohost();return;}
    this.write8(address,value);this.write8(address+1,value>>>8);this.write8(address+2,value>>>16);this.write8(address+3,value>>>24);
  }
  _checkTohost(){const value=this.read32(this.tohost);if(value&1)this._halt(value===1?0:value>>>1,'tohost');}
  _halt(code,reason){this.state='halted';this.exitCode=code;this.exitReason=reason;this.onExit(code,reason);}
  _trap(cause,value,at=this.currentPC??this.pc) {
    this.csrs.set(0x341,at>>>0);this.csrs.set(0x342,cause>>>0);this.csrs.set(0x343,value>>>0);
    const mtvec=this.csrs.get(0x305)||0;
    if(mtvec&~3){let status=this.csrs.get(0x300)||0;status=(status&~0x1888)|((status&8)<<4)|((this.privilege&3)<<11);this.csrs.set(0x300,status>>>0);this.privilege=3;this.pc=(mtvec&~3)>>>0;return;}
    this.state='faulted';this.fault={cause,pc:at>>>0,value:value>>>0,instruction:this.currentInstruction>>>0,message:`${TRAPS[cause]||'CPU trap'} at 0x${(at>>>0).toString(16).padStart(8,'0')} (0x${(value>>>0).toString(16).padStart(8,'0')})`};
    this.onFault(this.fault);
  }
  _illegal(inst){this._trap(2,inst);}
  _readCSR(csr) {
    switch(csr){case 0xc00:case 0xc01:case 0xb00:return this.cycles>>>0;case 0xc02:case 0xb02:return this.instructions>>>0;case 0xc80:case 0xc81:case 0xb80:return Math.floor(this.cycles/0x100000000)>>>0;case 0xc82:case 0xb82:return Math.floor(this.instructions/0x100000000)>>>0;case 0x100:return(this.csrs.get(0x300)||0)&0x800de133;default:return this.csrs.get(csr)??0;}
  }
  _writeCSR(csr,value) {
    value>>>=0;
    if(csr===0x301)return;
    if(csr===0xb00){this.cycles=Math.floor(this.cycles/0x100000000)*0x100000000+value;return;}
    if(csr===0xb02){this.instructions=Math.floor(this.instructions/0x100000000)*0x100000000+value;return;}
    if(csr===0xb80){this.cycles=value*0x100000000+(this.cycles>>>0);return;}
    if(csr===0xb82){this.instructions=value*0x100000000+(this.instructions>>>0);return;}
    if(csr===0x341)value&=~1;
    if(csr===0x100){this.csrs.set(0x300,((this.csrs.get(0x300)||0)&~0x800de133)|(value&0x800de133));return;}
    this.csrs.set(csr,value>>>0);
  }
  loadELF(buffer) {
    const bytes=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer);
    const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    const bounds=(start,size)=>{if(start<0||size<0||start+size>bytes.length)throw new Error('Truncated ELF image');};
    bounds(0,52);
    if(view.getUint32(0,false)!==0x7f454c46||view.getUint8(4)!==1||view.getUint8(5)!==1)throw new Error('Expected a little-endian ELF32 image');
    if(view.getUint16(18,true)!==243)throw new Error('ELF is not a RISC-V executable');
    const entry=view.getUint32(24,true),phoff=view.getUint32(28,true),shoff=view.getUint32(32,true),phsize=view.getUint16(42,true),phnum=view.getUint16(44,true),shsize=view.getUint16(46,true),shnum=view.getUint16(48,true);
    if(phnum&&phsize<32)throw new Error('Invalid ELF program header');
    const segments=[];
    for(let i=0;i<phnum;i++){const p=phoff+i*phsize;bounds(p,32);if(view.getUint32(p,true)!==1)continue;const offset=view.getUint32(p+4,true),vaddr=view.getUint32(p+8,true),paddr=view.getUint32(p+12,true),size=view.getUint32(p+16,true),memSize=view.getUint32(p+20,true);const address=paddr||vaddr;if(size>memSize)throw new Error('Invalid ELF segment sizes');bounds(offset,size);this._offset(address,memSize);segments.push({offset,address,size,memSize});}
    if(!segments.length)throw new Error('ELF has no loadable segments');
    this._offset(entry,4);this.reset();
    for(const segment of segments)this.memory.set(bytes.subarray(segment.offset,segment.offset+segment.size),segment.address-this.base);
    const symbols={};
    if(shnum&&shsize>=40){for(let i=0;i<shnum;i++){const s=shoff+i*shsize;bounds(s,40);const type=view.getUint32(s+4,true);if(type!==2&&type!==11)continue;const offset=view.getUint32(s+16,true),size=view.getUint32(s+20,true),link=view.getUint32(s+24,true),entsize=view.getUint32(s+36,true)||16;if(entsize<16||link>=shnum)continue;bounds(offset,size);const st=shoff+link*shsize;bounds(st,40);const stringOffset=view.getUint32(st+16,true),stringSize=view.getUint32(st+20,true);bounds(stringOffset,stringSize);
      for(let at=offset;at+16<=offset+size;at+=entsize){const n=view.getUint32(at,true);if(!n||n>=stringSize)continue;let end=stringOffset+n;while(end<stringOffset+stringSize&&bytes[end])end++;const name=new TextDecoder().decode(bytes.subarray(stringOffset+n,end));if(name)symbols[name]=view.getUint32(at+4,true);}
    }}
    this.pc=entry;this.entry=entry;this.symbols=symbols;this.tohost=symbols.tohost||0;this.state='ready';
    return{entry,segments,symbols,ramSize:this.ramSize,memoryMap:this.mode};
  }
  step() {
    if(this.state==='halted'||this.state==='faulted'||this.state==='waiting')return false;
    this.state='running';const r=this.regs,pc=this.pc>>>0;this.currentPC=pc;this.currentInstruction=0;this.cycles++;
    if(pc&3){this._trap(0,pc,pc);return false;}
    let inst;
    try{inst=this.read32(pc);}catch(err){if(err instanceof MemoryFault){this._trap(1,pc,pc);return false;}throw err;}
    this.currentInstruction=inst;this.pc=(pc+4)>>>0;
    const op=inst&127,rd=(inst>>>7)&31,f3=(inst>>>12)&7,rs1=(inst>>>15)&31,rs2=(inst>>>20)&31,f7=inst>>>25,a=r[rs1],b=r[rs2],imm=inst>>20;
    let retired=true;
    const target=address=>{address>>>=0;if(address&3){this._trap(0,address,pc);retired=false;}else this.pc=address;};
    try {
      switch(op){
        case 0x37:r[rd]=inst&0xfffff000;break;
        case 0x17:r[rd]=(pc+(inst&0xfffff000))>>>0;break;
        case 0x6f:{const offset=((inst&0x80000000)>>11)|(inst&0xff000)|((inst>>>9)&0x800)|((inst>>>20)&0x7fe);const address=(pc+offset)>>>0;if(address&3){target(address);}else{r[rd]=(pc+4)>>>0;this.pc=address;}break;}
        case 0x67:if(f3!==0){this._illegal(inst);retired=false;break;}else{const address=((a+imm)&~1)>>>0;if(address&3){target(address);}else{r[rd]=(pc+4)>>>0;this.pc=address;}break;}
        case 0x63:{const offset=((inst&0x80000000)>>19)|((inst<<4)&0x800)|((inst>>>20)&0x7e0)|((inst>>>7)&0x1e);let take;switch(f3){case 0:take=a===b;break;case 1:take=a!==b;break;case 4:take=(a|0)<(b|0);break;case 5:take=(a|0)>=(b|0);break;case 6:take=a<b;break;case 7:take=a>=b;break;default:this._illegal(inst);retired=false;}if(take)target(pc+offset);break;}
        case 0x03:{const address=(a+imm)>>>0;switch(f3){case 0:r[rd]=(this.read8(address)<<24)>>24;break;case 1:r[rd]=(this.read16(address)<<16)>>16;break;case 2:r[rd]=this.read32(address);break;case 4:r[rd]=this.read8(address);break;case 5:r[rd]=this.read16(address);break;default:this._illegal(inst);retired=false;}break;}
        case 0x23:{const offset=(inst>>20&~31)|((inst>>>7)&31),address=(a+offset)>>>0;switch(f3){case 0:this.write8(address,b);break;case 1:this.write16(address,b);break;case 2:this.write32(address,b);break;default:this._illegal(inst);retired=false;}break;}
        case 0x13:switch(f3){case 0:r[rd]=(a+imm)>>>0;break;case 2:r[rd]=(a|0)<imm?1:0;break;case 3:r[rd]=a<(imm>>>0)?1:0;break;case 4:r[rd]=a^imm;break;case 6:r[rd]=a|imm;break;case 7:r[rd]=a&imm;break;case 1:if(f7===0)r[rd]=a<<(rs2&31);else{this._illegal(inst);retired=false;}break;case 5:if(f7===0)r[rd]=a>>>(rs2&31);else if(f7===32)r[rd]=(a|0)>>(rs2&31);else{this._illegal(inst);retired=false;}break;}break;
        case 0x33:
          if(f7===1){switch(f3){case 0:r[rd]=Math.imul(a,b);break;case 1:r[rd]=Number(BigInt.asUintN(64,BigInt(a|0)*BigInt(b|0))>>32n);break;case 2:r[rd]=Number(BigInt.asUintN(64,BigInt(a|0)*BigInt(b))>>32n);break;case 3:r[rd]=Number((BigInt(a)*BigInt(b))>>32n);break;case 4:r[rd]=b===0?0xffffffff:((a|0)===-2147483648&&(b|0)===-1)?0x80000000:Math.trunc((a|0)/(b|0));break;case 5:r[rd]=b===0?0xffffffff:Math.floor(a/b);break;case 6:r[rd]=b===0?a:((a|0)===-2147483648&&(b|0)===-1)?0:(a|0)%(b|0);break;case 7:r[rd]=b===0?a:a%b;break;}break;}
          if(f7===16&&[2,4,6].includes(f3)){r[rd]=(b+(a<<(f3/2)))>>>0;break;}
          if(f7!==0&&!(f7===32&&(f3===0||f3===5))){this._illegal(inst);retired=false;break;}
          switch(f3){case 0:r[rd]=f7===32?a-b:a+b;break;case 1:r[rd]=a<<(b&31);break;case 2:r[rd]=(a|0)<(b|0)?1:0;break;case 3:r[rd]=a<b?1:0;break;case 4:r[rd]=a^b;break;case 5:r[rd]=f7===32?(a|0)>>(b&31):a>>>(b&31);break;case 6:r[rd]=a|b;break;case 7:r[rd]=a&b;break;}break;
        case 0x0f:if(f3!==0&&f3!==1){this._illegal(inst);retired=false;}break;
        case 0x73:
          if(f3===0){
            if(inst===0x00000073){if(!(this.csrs.get(0x305)&~3)&&r[17]===1){this.onOutput(String.fromCharCode(r[10]&255));}else if(!(this.csrs.get(0x305)&~3)&&r[17]===93)this._halt(r[10],'ecall exit');else if(!(this.csrs.get(0x305)&~3)&&r[17]===64){let text='';for(let i=0;i<Math.min(r[12],1024*1024);i++)text+=String.fromCharCode(this.read8(r[11]+i));this.onOutput(text);r[10]=r[12];}else{this._trap(8+this.privilege,0,pc);retired=false;}}
            else if(inst===0x00100073){if(this.csrs.get(0x305)&~3){this._trap(3,pc,pc);retired=false;}else this._halt(null,'breakpoint');}
            else if(inst===0x30200073){let status=this.csrs.get(0x300)||0;this.privilege=(status>>>11)&3;status=(status&~0x1888)|((status&0x80)>>>4)|0x80;this.csrs.set(0x300,status>>>0);target(this.csrs.get(0x341)||0);}
            else if(inst===0x10200073){this.privilege=(this.csrs.get(0x300)&0x100)?1:0;target(this.csrs.get(0x141)||0);}
            else if(inst===0x10500073){/* WFI is permitted to act as a no-op. */}
            else if((inst&0xfe007fff)===0x12000073){/* SFENCE.VMA: no virtual memory in this machine. */}
            else{this._illegal(inst);retired=false;}
          }else if(f3===4){this._illegal(inst);retired=false;}
          else{const csr=inst>>>20,source=f3>=5?rs1:a,old=this._readCSR(csr),kind=f3&3,write=kind===1||source!==0;
            if(write&&(csr&0xc00)===0xc00){this._illegal(inst);retired=false;break;}
            if(write)this._writeCSR(csr,kind===1?source:kind===2?old|source:old&~source);
            r[rd]=old;
          }break;
        default:this._illegal(inst);retired=false;
      }
    }catch(err){if(err instanceof MemoryFault){this._trap(op===0x23?7:5,err.address,pc);retired=false;}else throw err;}
    r[0]=0;if(retired)this.instructions++;return retired;
  }
  run(budget=100000) {budget=Math.max(0,Math.min(10000000,Math.floor(budget)));let executed=0;for(let i=0;i<budget;i++){if(this.state==='halted'||this.state==='faulted'||this.state==='waiting')break;this.step();executed++;}return executed;}
}
class MemoryFault extends Error{constructor(address,length){super(`Unmapped memory at 0x${address.toString(16)}`);this.address=address;this.length=length;}}
const TRAPS={0:'Instruction address misaligned',1:'Instruction access fault',2:'Illegal instruction',3:'Breakpoint',4:'Load address misaligned',5:'Load access fault',6:'Store address misaligned',7:'Store access fault',8:'User environment call',9:'Supervisor environment call',11:'Machine environment call'};
