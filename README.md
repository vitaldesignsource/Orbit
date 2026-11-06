# Orbit

A computer-shaped website with a real compiled neural RISC-V circuit.

Orbit combines a local desktop (windows, files, notes, terminal, calculator, and wallpapers) with a machine lab based on [Qwenputer](https://github.com/divagr18/qwenputer/tree/main/experimental/riscv).

## What actually executes

- **Neural circuit**: sparse weighted equations of Qwenputer's 11 compiled Qwen3.5-compatible Gated DeltaNet/SwiGLU CPU layers. Machine state lives in recurrent head matrices. Instructions are not interpreted by an opcode switch on this path.
- **Reference**: a conventional RV32IM + Zba JavaScript interpreter for comparison and fast execution.

The UI labels the selected execution engine. Reference results are not represented as neural results.

The browser does **not** load Qwen3.5-4B's original 32 language-model layers or the full merged checkpoint. The upstream merged model is approximately 18.7 GB in bf16 and requires a separate suitable inference environment. [Full model checkpoint](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer).

## Run the site

No build step or runtime dependencies are required. Serve `dist/` with any static web server, for example:

```sh
python -m http.server 8080 --directory dist
```

Open `http://localhost:8080`. Use an HTTP server; JavaScript modules and asset loading do not work from a `file://` page. Modern browsers support the gzip decompression API used for circuit assets.

## The neural machine

- RV32IM integer instructions plus Zba.
- 256 KiB code ROM at `0x40000`.
- 128 KiB read-only data at `0x20000`.
- 32 KiB RAM at `0x0000`; data begins at `0x100`, stack below `0x8000`.
- Output through `ecall` with `a7 = 1`, byte in `a0`.
- Exit through `ecall` with `a7 = 93`, code in `a0`.
- Bare metal: no Linux kernel, filesystem syscalls, or arbitrary native operating-system programs.

Included C and MicroPython examples are genuine precompiled RISC-V ELF binaries. The editor displays the associated source. The supplied MicroPython builds run embedded Python scripts; this is not a separate host Python interpreter masquerading as guest execution.

## Your own C and MicroPython code

For C, install a RISC-V bare-metal GCC toolchain, then compile with the provided runtime:

```sh
scripts/build-c.sh my-program.c my-program.elf
```

Set `RISCV_CC=/path/to/riscv64-unknown-elf-gcc` if the compiler is not on your PATH. Include `orbit.h` to use `orbit_print("Hello\n")` and `orbit_putchar()`. The script builds a freestanding RV32IM/ilp32 static ELF with the neural memory map. There is no C compiler inside the browser. C libraries and OS syscalls need their own compatible guest implementation.

Choose **Load ELF** in the site to load the resulting binary. The browser converts the actual ELF into recurrent neural state, including predecoded code ROM. The reference engine can load the same ELF for comparison.

MicroPython examples also expose **Apply script**, which replaces the embedded source slot in the real guest interpreter image. Custom scripts are limited to 409 UTF-8 bytes and the interpreter's 32 KiB memory environment. They are parsed and executed by MicroPython on the selected RISC-V engine, not by host Python. Neural execution is much slower than reference mode.

## Validation

```sh
npm test
# All 51 supplied ISA fixtures on the neural circuit:
npm run test:neural:all
```

CPU checks and ISA fixture checks are dependency-free Node.js programs. Snapshot validation compares every recurrent state row with exported upstream snapshots. Neural validation evaluates the weighted circuit and compares predictions with upstream reference traces; see test output and the JSON reports under `dist/machine/` for exact scope and results.

The supplied ISA fixtures cover 40 RV32UI, 8 RV32UM, and 3 Zba cases. Upstream excludes `fence_i` (self-modifying code) and `ma_data` (misaligned accesses). These fixtures do not certify all of RISC-V: privileged modes, atomics, compressed instructions, floating point, and operating-system behavior are outside this circuit's scope.

The site's ISA tests run on demand and show results from actual execution in the selected engine. No pass result is assumed before a test runs.

## Source provenance

The circuit compiler, fixture binaries, and program sources derive from Qwenputer at commit `cb1c969cf4e9b6d703c16678be42c9cf3a2d42fe` (MIT, Divyansh Agrawal). The JavaScript executor implements its documented sparse circuit equations. Retain the notices in `dist/LICENSE-qwenputer.txt` and the third-party notices alongside the fixtures.

The standalone circuit uses Qwen3.5-4B's layer capacity, not its learned language-model weights. It is a compiled digital circuit compatible with that architecture; it is not an LLM guessing instruction results.

## Rebuild the compiled circuit assets

The portable exporter requires Python and NumPy, plus the upstream compiler checkout at the pinned commit:

```sh
git clone https://github.com/divagr18/qwenputer.git /tmp/qwenputer
git -C /tmp/qwenputer checkout cb1c969cf4e9b6d703c16678be42c9cf3a2d42fe
python scripts/export-circuit.py /tmp/qwenputer dist/machine
```

It emits the actual sparse circuit weights, recurrent machine snapshots, program manifest, and reference traces. It does not download or merge Qwen's learned language-model weights. Use the upstream `chipformer.rv.merge` workflow in a suitable inference environment for the full model.

## Local desktop storage

Notes, imported text files, and wallpaper settings use browser local storage on this origin. They remain on the current browser/device. Download important files to preserve them elsewhere. The desktop terminal is a local shell for these files, separate from the neural machine lab.
