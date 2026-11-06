#!/usr/bin/env bash
# Compile actual freestanding C to the neural CPU's RV32IM ELF address map.
# Usage: RISCV_CC=/path/to/riscv64-unknown-elf-gcc scripts/build-c.sh input.c output.elf
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: RISCV_CC=riscv64-unknown-elf-gcc $0 input.c output.elf" >&2
  exit 2
fi
runtime_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/runtime" && pwd)
compiler=${RISCV_CC:-riscv64-unknown-elf-gcc}
"$compiler" -march=rv32im -mabi=ilp32 -O3 -ffreestanding -fno-builtin \
  -fno-stack-protector -fno-pic -msmall-data-limit=0 -nostdlib -nostartfiles \
  -Wl,--no-relax -Wl,--build-id=none -T "$runtime_dir/rom.ld" -I "$runtime_dir" \
  "$runtime_dir/crt0.S" "$1" -o "$2"
