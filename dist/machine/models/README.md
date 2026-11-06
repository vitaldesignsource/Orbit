# Stories260K

These are the authentic trained checkpoint and tokenizer published by Andrej
Karpathy in `karpathy/tinyllamas`, at revision
`0bd21da7698eaf29a0d7de3992de8a46ef624add`:

- [stories260K.bin](https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/stories260K.bin)
- [tok512.bin](https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/tok512.bin)
- [Primary architecture/training documentation](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/doc/stories260K.md)
- [Original C reference](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c)

SHA-256:

```
b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696  stories260K.bin
037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312  tok512.bin
```

The model has 260,032 trained parameters: dimension 64, feedforward dimension
172, five Llama 2 layers, eight query heads, four key/value heads, a 512-token
vocabulary, and a 512-token context. The 1,056,540-byte original v0 binary has
seven little-endian int32 configuration fields, followed by little-endian
float32 tensors. The two legacy rotary tables are present but unused, matching
the C implementation. Tensors are not randomly initialized or substituted.

`tiny-model.js` reads the original data and performs autoregressive inference
using `CompiledVectorUnit` for matrix products, RMSNorm, RoPE products, residual
addition, SwiGLU, and attention softmax. RoPE sine/cosine coefficients are
precomputed constants. Numerical arithmetic is a floating-point weighted
circuit extension with documented approximations, rather than the CPU's
integer instruction circuit or an exported, merged Qwen checkpoint.

Orbit's memory filesystem can contain the checkpoint and tokenizer byte for
byte. Working float32 inference arrays are materialized from those stored
bytes. `getState()` saves used key/value cache rows, token queues, output text,
sampling settings, and RNG state as JSON; `loadState()` validates and restores
them. Store that JSON in the memory filesystem to include a paused generation
in an exported memory snapshot. Immutable weights need not be duplicated in
each generation state. A 512-position KV cache is 655,360 bytes before base64.

The fixture in `tests/fixtures/stories260k-reference.json` is produced by
`tests/stories-native-reference.c` calling the original `run.c`. It contains
the first three full logit vectors and 200 deterministic greedy token IDs.
The test compares numerical error to those independent native outputs and
verifies resumed generation. It does not claim floating-point bit identity.

The upstream code and model repository are MIT licensed. See LICENSE.
