const test = require('node:test');
const assert = require('node:assert/strict');
const { evaluateFaceMatch, isFaceMatch } = require('../utils/faceUtils');

test('evaluateFaceMatch accepts a stable burst of embeddings', () => {
  const sampleEmbedding = [0.2, 0.4, 0.6, 0.8];
  const sameBurst = [
    [0.2, 0.4, 0.6, 0.8],
    [0.21, 0.39, 0.61, 0.79],
    [0.19, 0.41, 0.59, 0.81],
  ];

  const result = evaluateFaceMatch(sampleEmbedding, sameBurst, 0.55);

  assert.equal(result.matched, true);
  assert.ok(result.confidence >= 0.7);
  assert.equal(result.sampleCount, 3);
});

test('isFaceMatch rejects a mismatched burst', () => {
  const storedEmbedding = [0.1, 0.2, 0.3, 0.4];
  const mismatchedBurst = [
    [0.9, 0.2, 0.1, 0.0],
    [0.8, 0.3, 0.1, 0.0],
  ];

  assert.equal(isFaceMatch(storedEmbedding, mismatchedBurst, 0.55), false);
});
