/**
 * Karpathy's authentic Stories260K Llama 2 checkpoint, executed by Orbit's
 * compiled vector circuit. The checkpoint is trained model data; vector.js is
 * the separately compiled arithmetic circuit. No Qwen language-model fallback.
 * Binary layout / tokenizer: karpathy/llama2.c (MIT), see models/README.md.
 */
import { CompiledVectorUnit } from './vector.js';

const utf8 = new TextEncoder();
const decoder = new TextDecoder();
const f32 = Math.fround;

function bytesOf(input) {
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  throw new TypeError('Expected checkpoint/tokenizer bytes');
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function checksum(bytes) {
  let hash = 2166136261;
  for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619);
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/** Parse the original llama2.c v0 little-endian binary, never pickle/code. */
export function parseStoriesModel(input) {
  const bytes = bytesOf(input);
  assert(bytes.length >= 28, 'Truncated model header');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const names = ['dim', 'hiddenDim', 'layers', 'heads', 'kvHeads', 'vocabSize', 'context'];
  const config = Object.fromEntries(names.map((name, i) => [name, view.getInt32(i * 4, true)]));
  const sharedClassifier = config.vocabSize > 0;
  config.vocabSize = Math.abs(config.vocabSize);
  for (const name of names) assert(Number.isSafeInteger(config[name]) && config[name] > 0, `Invalid model ${name}`);
  assert(config.dim <= 4096 && config.hiddenDim <= 16384 && config.layers <= 64 && config.context <= 4096 && config.vocabSize <= 65536, 'Model dimensions exceed browser limits');
  assert(config.dim % config.heads === 0 && config.heads % config.kvHeads === 0, 'Invalid grouped-query head dimensions');
  config.headSize = config.dim / config.heads;
  config.kvDim = config.headSize * config.kvHeads;
  assert(config.headSize % 2 === 0, 'RoPE requires an even head size');
  const { dim: d, hiddenDim: h, layers: l, vocabSize: v, kvDim: k, context: s, headSize } = config;
  let offset = 28;
  let parameters = 0;
  function take(count, name, trainable = true) {
    assert(Number.isSafeInteger(count) && count >= 0 && offset + count * 4 <= bytes.length, `Truncated model tensor: ${name}`);
    // DataView also handles an unaligned input byteOffset and non-native endianness.
    const values = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      values[i] = view.getFloat32(offset + i * 4, true);
      assert(Number.isFinite(values[i]), `Non-finite model tensor: ${name}`);
    }
    offset += count * 4;
    if (trainable) parameters += count;
    return values;
  }
  const weights = {
    embedding: take(v * d, 'embedding'),
    rmsAttention: take(l * d, 'attention RMS'),
    query: take(l * d * d, 'query'),
    key: take(l * k * d, 'key'),
    value: take(l * k * d, 'value'),
    output: take(l * d * d, 'attention output'),
    rmsFeedforward: take(l * d, 'feedforward RMS'),
    gate: take(l * h * d, 'gate'),
    down: take(l * d * h, 'down'),
    up: take(l * h * d, 'up'),
    rmsFinal: take(d, 'final RMS'),
  };
  // v0 stores two obsolete RoPE tables. run.c skips them and recomputes RoPE.
  take(s * headSize / 2, 'legacy RoPE real', false);
  take(s * headSize / 2, 'legacy RoPE imaginary', false);
  weights.classifier = sharedClassifier ? weights.embedding : take(v * d, 'classifier');
  assert(offset === bytes.length, 'Unexpected trailing checkpoint bytes');
  return { config: Object.freeze(config), weights, parameters, sharedClassifier, id: `llama2c-v0-${checksum(bytes)}` };
}

export class StoriesTokenizer {
  constructor(input, vocabSize = 512) {
    const bytes = bytesOf(input);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    assert(bytes.length >= 4, 'Truncated tokenizer header');
    this.maxTokenLength = view.getUint32(0, true);
    assert(this.maxTokenLength > 0 && this.maxTokenLength <= 65536, 'Invalid tokenizer maximum token length');
    this.vocab = [];
    this.scores = new Float32Array(vocabSize);
    this.lookup = new Map();
    let offset = 4;
    for (let id = 0; id < vocabSize; id++) {
      assert(offset + 8 <= bytes.length, 'Truncated tokenizer entry');
      this.scores[id] = view.getFloat32(offset, true);
      const length = view.getInt32(offset + 4, true);
      offset += 8;
      assert(Number.isFinite(this.scores[id]) && length >= 0 && length <= this.maxTokenLength && offset + length <= bytes.length, 'Invalid tokenizer entry');
      const piece = decoder.decode(bytes.subarray(offset, offset + length));
      this.vocab.push(piece);
      this.lookup.set(piece, id);
      offset += length;
    }
    assert(offset === bytes.length, 'Unexpected trailing tokenizer bytes');
    assert(this.lookup.has(' ') && vocabSize >= 259, 'Tokenizer lacks byte fallback or dummy prefix');
    this.id = `llama2c-tokenizer-${checksum(bytes)}`;
  }

  encode(text, { bos = true, eos = false } = {}) {
    assert(typeof text === 'string' && !text.includes('\0'), 'Prompt must be text without NUL bytes');
    const tokens = bos ? [1] : [];
    if (text.length) tokens.push(this.lookup.get(' '));
    for (const point of text) {
      const id = this.lookup.get(point);
      if (id !== undefined) tokens.push(id);
      else for (const byte of utf8.encode(point)) tokens.push(byte + 3);
    }
    for (;;) {
      let bestScore = -1e10;
      let bestId = -1;
      let bestIndex = -1;
      for (let i = 0; i < tokens.length - 1; i++) {
        const id = this.lookup.get(this.vocab[tokens[i]] + this.vocab[tokens[i + 1]]);
        if (id !== undefined && this.scores[id] > bestScore) {
          bestId = id;
          bestScore = this.scores[id];
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      tokens.splice(bestIndex, 2, bestId);
    }
    if (eos) tokens.push(2);
    return tokens;
  }

  decodeBytes(previous, token) {
    assert(Number.isInteger(token) && token >= 0 && token < this.vocab.length, 'Invalid token');
    let piece = this.vocab[token];
    if (previous === 1 && piece.startsWith(' ')) piece = piece.slice(1);
    const raw = /^<0x([0-9a-fA-F]{2})>$/.exec(piece);
    return raw ? Uint8Array.of(Number.parseInt(raw[1], 16)) : utf8.encode(piece);
  }

  decode(previous, token) {
    return decoder.decode(this.decodeBytes(previous, token));
  }
}

function toBase64(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 16384) binary += String.fromCharCode(...bytes.subarray(offset, offset + 16384));
  return btoa(binary);
}

function fromBase64(encoded, count) {
  assert(typeof encoded === 'string' && encoded.length === Math.ceil(count * 4 / 3) * 4 && /^[A-Za-z0-9+/]*={0,2}$/.test(encoded), 'Invalid model cache encoding');
  const binary = atob(encoded);
  assert(binary.length === count * 4, 'Incorrect model cache size');
  const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
  const values = new Float32Array(bytes.buffer);
  assert(values.every(Number.isFinite), 'Non-finite model cache');
  return values;
}

/** Real Stories260K inference. Matrix/activation data arithmetic uses unit. */
export class Stories260K {
  constructor(modelBytes, tokenizerBytes, { unit = new CompiledVectorUnit() } = {}) {
    const model = parseStoriesModel(modelBytes);
    Object.assign(this, model);
    this.unit = unit;
    this.tokenizer = new StoriesTokenizer(tokenizerBytes, this.config.vocabSize);
    this.metadata = Object.freeze({ name: 'Karpathy Stories260K', parameters: model.parameters, architecture: 'Llama 2 / grouped-query attention / SwiGLU', arithmetic: 'compiled-vector-circuit', checkpoint: this.id, tokenizer: this.tokenizer.id });
    const { layers, context, kvDim, dim, headSize } = this.config;
    this.attentionScale = f32(unit.inverseSqrt(headSize));
    // Stable immutable views let the vector unit compile each learned affine
    // projection once. KV matrices remain dynamic circuit inputs below.
    const shapes = { rmsAttention: dim, query: dim * dim, key: kvDim * dim, value: kvDim * dim, output: dim * dim, rmsFeedforward: dim, gate: this.config.hiddenDim * dim, down: dim * this.config.hiddenDim, up: this.config.hiddenDim * dim };
    this.layerWeights = Array.from({ length: layers }, (_, layer) => Object.fromEntries(Object.entries(shapes).map(([name, stride]) => [name, this.weights[name].subarray(layer * stride, (layer + 1) * stride)])));
    this.keyCache = new Float32Array(layers * context * kvDim);
    this.valueCache = new Float32Array(layers * context * kvDim);
    // Static position coefficients, equivalent to run.c's float RoPE constants.
    this.ropeCos = new Float32Array(context * dim);
    this.ropeSin = new Float32Array(context * dim);
    for (let pos = 0; pos < context; pos++) {
      for (let i = 0; i < dim; i += 2) {
        const frequency = f32(1 / f32(Math.pow(10000, (i % headSize) / headSize)));
        const angle = f32(pos * frequency);
        const cos = f32(Math.cos(angle));
        const sin = f32(Math.sin(angle));
        this.ropeCos[pos * dim + i] = cos;
        this.ropeCos[pos * dim + i + 1] = cos;
        this.ropeSin[pos * dim + i] = -sin;
        this.ropeSin[pos * dim + i + 1] = sin;
      }
    }
    this.reset();
  }

  reset() {
    this.keyCache.fill(0);
    this.valueCache.fill(0);
    this.position = 0;
    this.nextInput = 1;
    this.pendingPrompt = [];
    this.generatedTokens = [];
    this.text = '';
    this.pendingBytes = [];
    this.seed = 42;
    this.temperature = 0;
    this.topP = 0.9;
    this.done = false;
    return this;
  }

  _rotate(vector, position) {
    const swapped = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i += 2) {
      swapped[i] = vector[i + 1];
      swapped[i + 1] = vector[i];
    }
    const start = position * this.config.dim;
    return this.unit.add(this.unit.multiply(vector, this.ropeCos.subarray(start, start + vector.length)), this.unit.multiply(swapped, this.ropeSin.subarray(start, start + vector.length)));
  }

  forward(token, position = this.position) {
    const c = this.config;
    const w = this.weights;
    const u = this.unit;
    assert(Number.isInteger(token) && token >= 0 && token < c.vocabSize, 'Token outside vocabulary');
    assert(position === this.position && position < c.context, 'Model context exhausted or nonsequential position');
    const { dim: d, hiddenDim: h, kvDim: k, headSize: hs } = c;
    let x = w.embedding.slice(token * d, (token + 1) * d);
    for (let layer = 0; layer < c.layers; layer++) {
      const lw = this.layerWeights[layer];
      const normalized = u.rmsnorm(x, lw.rmsAttention);
      const query = this._rotate(u.matvec(lw.query, normalized, d, d), position);
      const key = this._rotate(u.matvec(lw.key, normalized, k, d), position);
      const value = u.matvec(lw.value, normalized, k, d);
      const layerOffset = layer * c.context * k;
      this.keyCache.set(key, layerOffset + position * k);
      this.valueCache.set(value, layerOffset + position * k);
      const attended = new Float32Array(d);
      const length = position + 1;
      for (let head = 0; head < c.heads; head++) {
        const kvHead = Math.floor(head / (c.heads / c.kvHeads));
        const keys = new Float32Array(length * hs);
        const values = new Float32Array(length * hs);
        for (let t = 0; t < length; t++) {
          const offset = layerOffset + t * k + kvHead * hs;
          keys.set(this.keyCache.subarray(offset, offset + hs), t * hs);
          values.set(this.valueCache.subarray(offset, offset + hs), t * hs);
        }
        const scores = u.matmul(keys, query.subarray(head * hs, (head + 1) * hs), length, hs, 1);
        const scale = new Float32Array(length).fill(this.attentionScale);
        const attention = u.softmax(u.multiply(scores, scale));
        attended.set(u.matmul(attention, values, 1, length, hs), head * hs);
      }
      x = u.add(x, u.matvec(lw.output, attended, d, d));
      const ff = u.rmsnorm(x, lw.rmsFeedforward);
      const gate = u.silu(u.matvec(lw.gate, ff, h, d));
      const up = u.matvec(lw.up, ff, h, d);
      x = u.add(x, u.matvec(lw.down, u.multiply(gate, up), d, h));
    }
    const logits = u.matvec(w.classifier, u.rmsnorm(x, w.rmsFinal), c.vocabSize, d);
    assert(logits.every(Number.isFinite), 'Non-finite model logits');
    this.position++;
    return logits;
  }

  begin(prompt = '', { temperature = 0, topP = 0.9, seed = 42 } = {}) {
    assert(Number.isFinite(temperature) && temperature >= 0 && temperature <= 100, 'Temperature must be between 0 and 100');
    assert(Number.isFinite(topP) && topP > 0 && topP <= 1, 'Top-p must be in (0,1]');
    assert(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff, 'Seed must be a uint32');
    const tokens = this.tokenizer.encode(prompt);
    assert(tokens.length < this.config.context, 'Prompt fills the model context');
    this.reset();
    this.nextInput = tokens[0];
    this.pendingPrompt = tokens.slice(1);
    this.temperature = temperature;
    this.topP = topP;
    this.seed = seed || 1;
    return this;
  }

  _random() {
    let value = this.seed;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.seed = value >>> 0;
    return this.seed / 4294967296;
  }

  _sample(logits) {
    if (this.temperature === 0) {
      let best = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    const probability = this.unit.softmax(this.unit.multiply(logits, new Float32Array(logits.length).fill(1 / this.temperature)));
    const ordered = Array.from(probability, (p, token) => ({ p, token })).sort((a, b) => b.p - a.p);
    let total = 0;
    let count = 0;
    do { total += ordered[count++].p; } while (count < ordered.length && total < this.topP);
    let coin = this._random() * total;
    for (let i = 0; i < count; i++) {
      coin -= ordered[i].p;
      if (coin < 0) return ordered[i].token;
    }
    return ordered[count - 1].token;
  }

  _decode(previous, token) {
    const bytes = [...this.pendingBytes, ...this.tokenizer.decodeBytes(previous, token)];
    let end = bytes.length;
    // Preserve an incomplete UTF-8 code point across generated byte tokens and
    // snapshots, instead of serializing opaque browser TextDecoder state.
    let start = end - 1;
    while (start >= 0 && (bytes[start] & 0xc0) === 0x80) start--;
    if (start >= 0) {
      const first = bytes[start];
      const needed = first < 0x80 ? 1 : first >= 0xf0 ? 4 : first >= 0xe0 ? 3 : first >= 0xc0 ? 2 : 1;
      if (end - start < needed) end = start;
    }
    this.pendingBytes = bytes.slice(end);
    return decoder.decode(Uint8Array.from(bytes.slice(0, end))).replace(/[\x00-\x08\x0b\x0e-\x1f\x7f]/g, '');
  }

  step() {
    if (this.done) return { done: true, text: this.text, piece: '', position: this.position };
    const previous = this.nextInput;
    const logits = this.forward(previous);
    const forced = this.pendingPrompt.length > 0;
    const token = forced ? this.pendingPrompt.shift() : this._sample(logits);
    if (!forced) this.generatedTokens.push(token);
    this.nextInput = token;
    this.done = token === 1 || token === 2 || this.position >= this.config.context;
    const piece = token === 1 || token === 2 ? '' : this._decode(previous, token);
    this.text += piece;
    return { token, forced, done: this.done, text: this.text, piece, position: this.position, generated: this.generatedTokens.length };
  }

  async generate(prompt = '', { maxTokens = 80, onToken, signal, ...sampling } = {}) {
    assert(Number.isInteger(maxTokens) && maxTokens >= 1 && maxTokens <= this.config.context, 'Invalid generation token limit');
    this.begin(prompt, sampling);
    while (!this.done && this.generatedTokens.length < maxTokens) {
      if (signal?.aborted) break;
      const result = this.step();
      if (onToken) onToken(result);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    return { text: this.text, tokens: [...this.generatedTokens], position: this.position, done: this.done };
  }

  getState() {
    const { layers, context, kvDim } = this.config;
    const used = this.position * kvDim;
    const key = new Float32Array(layers * used);
    const value = new Float32Array(layers * used);
    for (let layer = 0; layer < layers; layer++) {
      const start = layer * context * kvDim;
      key.set(this.keyCache.subarray(start, start + used), layer * used);
      value.set(this.valueCache.subarray(start, start + used), layer * used);
    }
    return { format: 'orbit-stories260k-state', version: 1, model: this.id, tokenizer: this.tokenizer.id, position: this.position, nextInput: this.nextInput, pendingPrompt: [...this.pendingPrompt], generatedTokens: [...this.generatedTokens], text: this.text, pendingBytes: [...this.pendingBytes], seed: this.seed, temperature: this.temperature, topP: this.topP, done: this.done, keyCache: toBase64(key), valueCache: toBase64(value) };
  }

  loadState(state) {
    assert(state && state.format === 'orbit-stories260k-state' && state.version === 1, 'Unsupported Stories260K state');
    assert(state.model === this.id && state.tokenizer === this.tokenizer.id, 'State belongs to a different model or tokenizer');
    const { layers, context, kvDim, vocabSize } = this.config;
    const position = state.position;
    assert(Number.isInteger(position) && position >= 0 && position <= context, 'Invalid model state position');
    const validToken = token => Number.isInteger(token) && token >= 0 && token < vocabSize;
    assert(validToken(state.nextInput), 'Invalid model state input token');
    for (const name of ['pendingPrompt', 'generatedTokens']) assert(Array.isArray(state[name]) && state[name].length <= context && state[name].every(validToken), `Invalid model state ${name}`);
    assert(state.pendingPrompt.length + position <= context && state.generatedTokens.length <= position, 'Inconsistent model state token counts');
    assert(typeof state.text === 'string' && state.text.length <= context * this.tokenizer.maxTokenLength, 'Invalid model state text');
    assert(Array.isArray(state.pendingBytes) && state.pendingBytes.length <= 3 && state.pendingBytes.every(byte => Number.isInteger(byte) && byte >= 0 && byte <= 255), 'Invalid model state UTF-8 bytes');
    assert(Number.isInteger(state.seed) && state.seed > 0 && state.seed <= 0xffffffff, 'Invalid model state RNG');
    assert(Number.isFinite(state.temperature) && state.temperature >= 0 && state.temperature <= 100 && Number.isFinite(state.topP) && state.topP > 0 && state.topP <= 1, 'Invalid model state sampling');
    assert(typeof state.done === 'boolean' && (position < context || state.done), 'Invalid model state completion');
    const used = position * kvDim;
    const key = fromBase64(state.keyCache, layers * used);
    const value = fromBase64(state.valueCache, layers * used);
    // Validate the complete state before replacing a running session.
    this.reset();
    for (let layer = 0; layer < layers; layer++) {
      this.keyCache.set(key.subarray(layer * used, (layer + 1) * used), layer * context * kvDim);
      this.valueCache.set(value.subarray(layer * used, (layer + 1) * used), layer * context * kvDim);
    }
    for (const name of ['position', 'nextInput', 'text', 'seed', 'temperature', 'topP', 'done']) this[name] = state[name];
    for (const name of ['pendingPrompt', 'generatedTokens', 'pendingBytes']) this[name] = [...state[name]];
    return this;
  }
}
