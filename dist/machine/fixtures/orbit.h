/* Orbit's freestanding C API for the Qwenputer RV32 machine. */
#ifndef ORBIT_H
#define ORBIT_H
static inline void orbit_putchar(unsigned char value) {
    register unsigned long a0 __asm__("a0") = value;
    register unsigned long a7 __asm__("a7") = 1;
    __asm__ volatile ("ecall" : "+r" (a0) : "r" (a7) : "memory");
}
static inline __attribute__((always_inline)) void orbit_write(const char *text, unsigned count) {
    /* A short literal becomes direct putchar ecalls, ideal for circuit stepping. */
    #pragma GCC unroll 64
    for (unsigned i = 0; i < count; ++i) orbit_putchar((unsigned char)text[i]);
}
#define orbit_print(literal) orbit_write((literal), sizeof(literal) - 1)
#endif
