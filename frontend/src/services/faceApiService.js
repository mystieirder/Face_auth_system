import * as faceapi from 'face-api.js';

const MODEL_URL = process.env.PUBLIC_URL + '/models';
let modelsLoaded = false;

export async function loadFaceModels() {
  if (modelsLoaded) return;
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
  modelsLoaded = true;
}

export async function checkEnvironment(videoEl) {
  const detection = await faceapi
    .detectSingleFace(videoEl, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();

  const facePresent = !!detection;
  const lightingScore = sampleBrightness(videoEl);

  let framePositioned = false;
  if (detection) {
    const { box } = detection.detection;
    const videoW = videoEl.videoWidth;
    const videoH = videoEl.videoHeight;
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;

    const withinHorizontal = centerX > videoW * 0.25 && centerX < videoW * 0.75;
    const withinVertical = centerY > videoH * 0.2 && centerY < videoH * 0.8;
    const reasonableSize = box.width > videoW * 0.2 && box.width < videoW * 0.9;

    framePositioned = withinHorizontal && withinVertical && reasonableSize;
  }

  return { facePresent, lightingScore, framePositioned, detection };
}

export async function extractFaceEmbedding(videoEl) {
  const detection = await faceapi
    .detectSingleFace(videoEl, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.4 }))
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) {
    throw new Error('No face detected — cannot extract embedding');
  }

  return Array.from(detection.descriptor);
}

const LEFT_EYE = [36, 37, 38, 39, 40, 41];
const RIGHT_EYE = [42, 43, 44, 45, 46, 47];

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function eyeAspectRatio(points, indices) {
  const [p1, p2, p3, p4, p5, p6] = indices.map((i) => points[i]);
  const vertical = dist(p2, p6) + dist(p3, p5);
  const horizontal = 2.0 * dist(p1, p4);
  return horizontal === 0 ? 0 : vertical / horizontal;
}

export async function checkLiveness(videoEl, durationMs = 6000, sampleIntervalMs = 120) {
  const earSamples = [];
  const start = Date.now();

  while (Date.now() - start < durationMs) {
    const detection = await faceapi
      .detectSingleFace(videoEl, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks();

    if (detection) {
      const points = detection.landmarks.positions;
      const ear = (eyeAspectRatio(points, LEFT_EYE) + eyeAspectRatio(points, RIGHT_EYE)) / 2;
      earSamples.push(ear);
      console.log('EAR:', ear.toFixed(3));
    } else {
      console.log('EAR: no face detected this frame');
    }

    await new Promise((resolve) => setTimeout(resolve, sampleIntervalMs));
  }

  if (earSamples.length < 5) {
    return { live: false, reason: 'Could not track your face consistently — hold still and try again' };
  }

  const maxEar = Math.max(...earSamples);
  const minEar = Math.min(...earSamples);
  const earSwing = maxEar - minEar;
  console.log(`Liveness check — min: ${minEar.toFixed(3)}, max: ${maxEar.toFixed(3)}, swing: ${earSwing.toFixed(3)}`);

  const blinkDetected = earSwing > 0.05;

  if (!blinkDetected) {
    return { live: false, reason: 'No blink detected — please look at the camera naturally and try again' };
  }

  return { live: true, reason: 'Liveness confirmed' };
}

function sampleBrightness(videoEl) {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let total = 0;
  const pixelCount = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    total += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return total / pixelCount;
}