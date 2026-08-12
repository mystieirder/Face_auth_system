import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  loadFaceModels,
  checkEnvironment,
  checkLiveness,
  extractFaceEmbedding,
} from '../services/faceApiService';
import EnvironmentValidator from './EnvironmentValidator';

const MIN_LIGHTING_SCORE = 60;
const ENV_CHECK_INTERVAL_MS = 400;

/**
 * FR-2.1 + FR-2.2 + FR-2.3: webcam panel used on the login screen.
 *
 * Flow: load models -> start camera -> continuously validate the
 * environment -> on "Verify Face", run a passive liveness check
 * (blink detection) -> if live, extract the embedding and hand it to
 * the parent via onVerified(embedding, livenessVerified).
 *
 * FR-2.3 Automatic Fallback: if the camera or models fail to load,
 * onCameraUnavailable() fires so the parent can switch to the
 * password field automatically, without losing whatever the user has
 * already typed into the username field.
 */
export default function FaceLoginCapture({ onVerified, onCameraUnavailable }) {
  const videoRef = useRef(null);
  const intervalRef = useRef(null);

  const [modelsReady, setModelsReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [envStatus, setEnvStatus] = useState({
    facePresent: false,
    lightingScore: 0,
    framePositioned: false,
  });
  const [stage, setStage] = useState('idle'); // idle | checking-liveness | extracting | done
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    async function init() {
      try {
        await loadFaceModels();
        setModelsReady(true);
      } catch (err) {
        setCameraError('Failed to load face verification models.');
        onCameraUnavailable && onCameraUnavailable();
      }
    }
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setCameraError('Camera access denied or unavailable — use your password instead.');
        onCameraUnavailable && onCameraUnavailable();
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelsReady]);

  useEffect(() => {
    if (!modelsReady || stage !== 'idle') return undefined;

    intervalRef.current = setInterval(async () => {
      if (videoRef.current && videoRef.current.readyState === 4) {
        const result = await checkEnvironment(videoRef.current);
        setEnvStatus(result);
      }
    }, ENV_CHECK_INTERVAL_MS);

    return () => clearInterval(intervalRef.current);
  }, [modelsReady, stage]);

  const isEnvironmentValid =
    envStatus.facePresent &&
    envStatus.lightingScore >= MIN_LIGHTING_SCORE &&
    envStatus.framePositioned;

  const handleVerify = useCallback(async () => {
    if (!isEnvironmentValid || !videoRef.current) return;
    clearInterval(intervalRef.current);

    setStage('checking-liveness');
    setStatusMessage('Checking liveness — please look at the camera naturally…');
    const liveness = await checkLiveness(videoRef.current);

    if (!liveness.live) {
      setStatusMessage(liveness.reason);
      setStage('idle');
      return;
    }

    setStage('extracting');
    setStatusMessage('Capturing a short face sample…');
    try {
      const embeddings = await extractFaceEmbedding(videoRef.current, 3);
      setStage('done');
      setStatusMessage('Face verified.');
      onVerified(embeddings, true);
    } catch (err) {
      setStatusMessage('Could not verify — please try again.');
      setStage('idle');
    }
  }, [isEnvironmentValid, onVerified]);

  return (
    <div className="scan-panel">
      {cameraError && <p className="field-error">{cameraError}</p>}

      <div className={`scan-frame ${isEnvironmentValid ? 'is-ready' : ''} ${stage === 'done' ? 'is-enrolled' : ''}`}>
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

      {statusMessage && <p className="field-hint">{statusMessage}</p>}

      <button
        type="button"
        className="cred-btn cred-btn--scan"
        onClick={handleVerify}
        disabled={!isEnvironmentValid || stage === 'checking-liveness' || stage === 'extracting' || stage === 'done'}
      >
        {stage === 'done'
          ? '✓ Verified'
          : stage === 'checking-liveness'
          ? 'Checking liveness…'
          : stage === 'extracting'
          ? 'Verifying…'
          : 'Verify face'}
      </button>
    </div>
  );
}
