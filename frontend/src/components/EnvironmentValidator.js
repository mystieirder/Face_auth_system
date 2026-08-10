import React from 'react';

/**
 * FR-1.3: Displays live pass/fail status for face presence, lighting,
 * and frame positioning so the user gets real-time feedback before
 * registration can be confirmed.
 */
export default function EnvironmentValidator({ status, minLightingScore = 60 }) {
  const { facePresent, lightingScore, framePositioned } = status;
  const lightingOk = lightingScore >= minLightingScore;

  const items = [
    { label: 'Face detected', ok: facePresent },
    { label: `Lighting sufficient (${Math.round(lightingScore)}/${minLightingScore}+)`, ok: lightingOk },
    { label: 'Face centered in frame', ok: framePositioned },
  ];

  return (
    <ul className="env-validator">
      {items.map((item) => (
        <li key={item.label} className={item.ok ? 'ok' : 'fail'}>
          <span className="indicator">{item.ok ? '✔' : '✖'}</span> {item.label}
        </li>
      ))}
    </ul>
  );
}