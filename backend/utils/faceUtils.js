// Checks lighting/positioning before allowing enrollment
function validateEnvironment(environmentCheck, minLighting) {
  const errors = [];

  if (!environmentCheck) {
    return { valid: false, errors: ['Environment check data is missing'] };
  }

  if (!environmentCheck.facePresent) {
    errors.push('No face detected');
  }
  if (environmentCheck.lightingScore < minLighting) {
    errors.push(`Lighting too low (minimum ${minLighting})`);
  }
  if (!environmentCheck.framePositioned) {
    errors.push('Face not properly positioned in frame');
  }

  return { valid: errors.length === 0, errors };
}

function computeDistance(embeddingA, embeddingB) {
  if (!embeddingA || !embeddingB || embeddingA.length !== embeddingB.length) {
    return null;
  }

  let sumSquares = 0;
  for (let i = 0; i < embeddingA.length; i++) {
    sumSquares += Math.pow(embeddingA[i] - embeddingB[i], 2);
  }
  return Math.sqrt(sumSquares);
}

function evaluateFaceMatch(embeddingA, embeddingBurst, threshold) {
  if (!Array.isArray(embeddingBurst) || embeddingBurst.length === 0) {
    return { matched: false, confidence: 0, sampleCount: 0, averageDistance: null };
  }

  const distances = embeddingBurst
    .map((sample) => (Array.isArray(sample) ? computeDistance(embeddingA, sample) : null))
    .filter((distance) => distance !== null);

  if (distances.length === 0) {
    return { matched: false, confidence: 0, sampleCount: 0, averageDistance: null };
  }

  const averageDistance = distances.reduce((sum, distance) => sum + distance, 0) / distances.length;
  const confidence = Math.max(0, 1 - averageDistance / 1.2);
  const matched = averageDistance <= threshold && confidence >= 0.7;

  return { matched, confidence, sampleCount: distances.length, averageDistance };
}

// Compares one embedding to another using Euclidean distance
function isFaceMatch(embeddingA, embeddingB, threshold) {
  if (Array.isArray(embeddingB) && Array.isArray(embeddingB[0])) {
    return evaluateFaceMatch(embeddingA, embeddingB, threshold).matched;
  }

  const distance = computeDistance(embeddingA, embeddingB);
  if (distance === null) {
    return false;
  }

  return distance <= threshold;
}

module.exports = { validateEnvironment, isFaceMatch, evaluateFaceMatch, computeDistance };