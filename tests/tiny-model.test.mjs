import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { Stories260K, StoriesTokenizer, parseStoriesModel } from '../dist/machine/tiny-model.js';
import { CompiledVectorUnit } from '../dist/machine/vector.js';

const modelBytes = await fs.readFile(new URL('../dist/machine/models/stories260K.bin', import.meta.url));
const tokenizerBytes = await fs.readFile(new URL('../dist/machine/models/tok512.bin', import.meta.url));
const reference = JSON.parse(await fs.readFile(new URL('./fixtures/stories260k-reference.json', import.meta.url), 'utf8'));
assert.equal(createHash('sha256').update(modelBytes).digest('hex'), 'b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696');
assert.equal(createHash('sha256').update(tokenizerBytes).digest('hex'), '037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312');

// Authentic published layout and trained coefficient count, including GQA.
const parsed = parseStoriesModel(modelBytes);
assert.deepEqual(parsed.config, { dim: 64, hiddenDim: 172, layers: 5, heads: 8, kvHeads: 4, vocabSize: 512, context: 512, headSize: 8, kvDim: 32 });
assert.equal(parsed.parameters, 260032);
assert.equal(parsed.weights.classifier, parsed.weights.embedding);
// Input views need not begin at a four-byte aligned address.
const unaligned = new Uint8Array(modelBytes.length + 1);
unaligned.set(modelBytes, 1);
assert.equal(parseStoriesModel(unaligned.subarray(1)).id, parsed.id);
assert.throws(() => parseStoriesModel(modelBytes.subarray(0, modelBytes.length - 4)), /Truncated/);
const invalid = Buffer.from(modelBytes);
invalid.writeFloatLE(Infinity, 28);
assert.throws(() => parseStoriesModel(invalid), /Non-finite/);

// Tokenizer checks include UTF-8 code points absent from the tiny vocabulary.
const tokenizer = new StoriesTokenizer(tokenizerBytes);
const prompts = ['', 'Once upon a time', "Lily's red ball.", 'café 🌙'];
for (let i = 0; i < prompts.length; i++) assert.deepEqual(tokenizer.encode(prompts[i]), reference.promptTokens[i]);
assert.throws(() => new StoriesTokenizer(tokenizerBytes.subarray(0, 120)), /Truncated|Invalid/);

const unit = new CompiledVectorUnit();
const runner = new Stories260K(modelBytes, tokenizerBytes, { unit });
let token = 1;
let maxLogitError = 0;
for (let position = 0; position < reference.logits.length; position++) {
  const logits = runner.forward(token);
  assert.equal(logits.length, 512);
  for (let i = 0; i < logits.length; i++) maxLogitError = Math.max(maxLogitError, Math.abs(logits[i] - reference.logits[position][i]));
  token = reference.greedyTokens[position];
}
assert.ok(maxLogitError < 3e-5, `Native C logit max error ${maxLogitError}`);
assert.ok(unit.stats.matrixProducts > 0 && unit.stats.softmaxCalls > 0 && unit.stats.normalizationCalls > 0, 'Model must use the compiled vector datapath');

// Independent original C inference, all 200 official deterministic tokens.
runner.begin('');
for (let i = 0; i < reference.greedyTokens.length; i++) assert.equal(runner.step().token, reference.greedyTokens[i], `Native greedy token ${i}`);
assert.ok(runner.text.startsWith('Once upon a time, there was a little girl named Lily.'));
assert.ok(runner.text.endsWith("Lily didn't want to help her mom, so she"));

// Persist a sampled session, including KV rows and RNG, and continue identically.
runner.begin('One day', { temperature: 0.8, topP: 0.9, seed: 133742 });
for (let i = 0; i < 16; i++) runner.step();
const state = JSON.parse(JSON.stringify(runner.getState()));
const resumed = new Stories260K(modelBytes, tokenizerBytes);
resumed.loadState(state);
for (let i = 0; i < 16; i++) assert.deepEqual(resumed.step(), runner.step(), `Restored sampled step ${i}`);
assert.equal(resumed.text, runner.text);
assert.equal(resumed.seed, runner.seed);
assert.deepEqual(resumed.getState(), runner.getState());
const intact = resumed.getState();
assert.throws(() => resumed.loadState({ ...intact, position: 999999 }), /position/);
assert.throws(() => resumed.loadState({ ...intact, model: 'different' }), /different model/);
assert.throws(() => resumed.loadState({ ...intact, keyCache: 'AAAA' }), /encoding|size/);
assert.deepEqual(resumed.getState(), intact, 'Rejected states must not mutate a live session');
const blank = new Stories260K(modelBytes, tokenizerBytes);
blank.loadState(blank.getState());
assert.equal(blank.step().token, reference.greedyTokens[0]);
assert.throws(() => blank.begin('x', { temperature: NaN }), /Temperature/);
assert.throws(() => blank.begin('x', { seed: -1 }), /Seed/);

console.log(`Stories260K: authentic hashes, tokenizer, 1536 native logits (max error ${maxLogitError.toExponential(2)}), 200 native greedy tokens, sampled state restore passed`);
