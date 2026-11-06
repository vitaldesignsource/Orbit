# Third-party notices and provenance

This distribution includes software and compiled fixtures from
[divagr18/qwenputer](https://github.com/divagr18/qwenputer), commit
`cb1c969cf4e9b6d703c16678be42c9cf3a2d42fe`.

## Qwenputer

The sparse CPU circuit, recurrent-state snapshots, build environment, C examples,
and Python scripts derive from Qwenputer's `experimental/riscv` directory.
Copyright (c) 2026 Divyansh Agrawal. MIT license: [LICENSE-qwenputer.txt](LICENSE-qwenputer.txt).

The browser evaluates the exported circuit equations. The original learned
Qwen3.5-4B language-model parameters are not included. The standalone CPU circuit
uses Qwen3.5-compatible layer operations and recurrent state.

## RISC-V ISA tests

The `rv32ui-*`, `rv32um-*`, and `rv32uzba-*` executables derive from
[riscv-software-src/riscv-tests](https://github.com/riscv-software-src/riscv-tests).
Copyright (c) 2012-2015, The Regents of the University of California.
License and disclaimer: [LICENSE-riscv-tests.txt](LICENSE-riscv-tests.txt).

The supplied set is 40 RV32UI, 8 RV32UM, and 3 RV32UZBA tests. Qwenputer builds
them with a custom user-mode environment without CSRs or traps and with an exit
ecall. Its build excludes `fence_i` and `ma_data`. Passing these fixtures does not
establish privileged-architecture compliance or complete RISC-V certification.
The original riscv-tests source revision is not recorded in the supplied build
script; the distributed ELF artifacts are pinned by the Qwenputer commit and
their SHA-256 hashes in [machine/provenance.json](machine/provenance.json).

## MicroPython

`mpy-s1.elf` and `mpy-s5.elf` contain a patched MicroPython interpreter compiled to
RISC-V, executing an embedded script. They are not interactive REPL firmware.
The embedded version banner identifies `MicroPython v1.30.0-preview on 2026-10-18`.
Qwenputer supplies its port and build script; that script expects a separately
patched MicroPython source tree and does not pin its source commit.

Copyright (c) 2013-2026 Damien P. George and the other contributors identified in
[LICENSE-MicroPython.txt](LICENSE-MicroPython.txt). MIT license.

## C runtime

The upstream C and MicroPython firmware links picolibc and GCC runtime support.
Picolibc's notices are reproduced in [LICENSE-picolibc.txt](LICENSE-picolibc.txt).
The ELF producer identifies GCC 13.2.0. GCC's runtime support is provided under
GPLv3 with the [GCC Runtime Library Exception 3.1](https://www.gnu.org/licenses/gcc-exception-3.1.html).
No mini-gmp executable or source is distributed here.

## Source and build information

Associated C/Python examples are beside their ELF files in `machine/fixtures/`.
Qwenputer's startup code, linker scripts, test environment, MicroPython port,
and original build scripts are included in `machine/source/qwenputer-env/`.
The browser circuit exporter is in this project's `scripts/export-circuit.py`.
Most firmware was supplied precompiled by Qwenputer. The `c-hello` sample was
compiled locally from `machine/fixtures/hello.c` using Debian GCC12.2 and
binutils2.40, with the freestanding API and linker scripts in
`machine/source/orbit-runtime/`. It links no libc. GCC optimizes its arithmetic
loop to constants; the resulting42-instruction guest program still performs
each output ecall and exits on the neural CPU. Any in-browser C compiler has a
separate description of its supported language subset.

Artifact hashes and source provenance: [machine/provenance.json](machine/provenance.json).
