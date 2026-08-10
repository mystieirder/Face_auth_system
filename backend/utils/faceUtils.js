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

// Compares two face embeddings using Euclidean distance
function isFaceMatch(embeddingA, embeddingB, threshold) {
  if (!embeddingA || !embeddingB || embeddingA.length !== embeddingB.length) {
    return false;
  }

  let sumSquares = 0;
  for (let i = 0; i < embeddingA.length; i++) {
    sumSquares += Math.pow(embeddingA[i] - embeddingB[i], 2);
  }
  const distance = Math.sqrt(sumSquares);

  return distance <= threshold;
}

module.exports = { validateEnvironment, isFaceMatch };