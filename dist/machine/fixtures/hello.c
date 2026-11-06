#include "orbit.h"

int main(void) {
    orbit_print("Hello from C!\n");
    unsigned sum = 0;
    for (unsigned i = 1; i <= 10; ++i) sum += i;
    orbit_putchar('0' + sum / 10);
    orbit_putchar('0' + sum % 10);
    orbit_putchar('\n');
    return sum != 55;
}
