import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  loadFaceModels,
  checkEnvironment,
  extractFaceEmbedding,
} from '../services/faceApiService';
import EnvironmentValidator from './EnvironmentValidator';

const MIN_LIGHTING_SCORE = 60;
const CHECK_INTERVAL_MS = 400;

/**
 * FR-1.2 + FR-1.3
 * Handles webcam access, continuous environmental validation, and
 * biometric enrollment capture. Calls onEnrolled(embedding, envStatus)
 * once a valid frame has been captured.
 */
export default function WebcamCapture({ onEnrolled }) {
  const videoRef = useRef(null);
  const intervalRef = useRef(null);

  const [modelsReady, setModelsReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [envStatus, setEnvStatus] = useState({
    facePresent: false,
    lightingScore: 0,
    framePositioned: false,
  });
  const [capturing, setCapturing] = useState(false);
  const [enrolled, setEnrolled] = useState(false);

  useEffect(() => {
    async function init() {
      try {
        await loadFaceModels();
        setModelsReady(true);
      } catch (err) {
        setCameraError('Failed to load face detection models.');
      }
    }
    init();
  }, []);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setCameraError('Camera access denied or unavailable.');
      }
    }
    if (modelsReady) startCamera();

    const videoNode = videoRef.current;
    return () => {
      const stream = videoNode && videoNode.srcObject;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [modelsReady]);

  useEffect(() => {
    if (!modelsReady) return undefined;

    intervalRef.current = setInterval(async () => {
      if (videoRef.current && videoRef.current.readyState === 4 && !enrolled) {
        const result = await checkEnvironment(videoRef.current);
        setEnvStatus(result);
      }
    }, CHECK_INTERVAL_MS);

    return () => clearInterval(intervalRef.current);
  }, [modelsReady, enrolled]);

  const isEnvironmentValid =
    envStatus.facePresent &&
    envStatus.lightingScore >= MIN_LIGHTING_SCORE &&
    envStatus.framePositioned;

  const handleCapture = useCallback(async () => {
    if (!isEnvironmentValid || !videoRef.current) return;
    setCapturing(true);
    try {
      const embedding = await extractFaceEmbedding(videoRef.current);
      setEnrolled(true);
      onEnrolled(embedding, {
        facePresent: envStatus.facePresent,
        lightingScore: envStatus.lightingScore,
        framePositioned: envStatus.framePositioned,
      });
    } catch (err) {
      console.error('Face capture error:', err);
      setCameraError(`Could not extract a facial embedding: ${err.message}`);
    } finally {
      setCapturing(false);
    }
  }, [isEnvironmentValid, envStatus, onEnrolled]);

  return (
    <div className="scan-panel">
      {cameraError && <p className="field-error">{cameraError}</p>}

      <div className={`scan-frame ${isEnvironmentValid ? 'is-ready' : ''} ${enrolled ? 'is-enrolled' : ''}`}>
        <video ref={videoRef} autoPlay muted playsInline width="320" height="240" />
        <span className="scan-corner scan-corner--tl" />
        <span className="scan-corner scan-corner--tr" />
        <span className="scan-corner scan-corner--bl" />
        <span className="scan-corner scan-corner--br" />
        {!modelsReady && !cameraError && (
          <div className="scan-overlay">Loading verification models…</div>
        )}
      </div>

      <div className="scan-status">
        <EnvironmentValidator status={envStatus} minLightingScore={MIN_LIGHTING_SCORE} />
      </div>

      <button
        type="button"
        className="cred-btn cred-btn--scan"
        onClick={handleCapture}
        disabled={!isEnvironmentValid || capturing || enrolled}
      >
        {enrolled ? '✓ Face captured' : capturing ? 'Capturing…' : 'Capture baseline face'}
      </button>
    </div>
  );
}