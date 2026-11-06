.file "orbit-crt0.S"
/* The ELF loader initializes RAM and .bss; no libc or external runtime is used. */
.section .text.init
.globl _start
_start:
    lui sp, 0x8
    call main
    li a7, 93
    ecall
1:  j 1b
