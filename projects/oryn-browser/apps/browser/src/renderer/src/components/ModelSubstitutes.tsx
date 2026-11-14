import { useEffect, useState } from 'react';
import type { TrustedModelSubstitute } from '../../../shared/ipc';
import { Button } from './ui';

/** Optional compatibility work never blocks the installed inventory. */
export function ModelSubstitutes({ installationId }: { installationId: string }) {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<TrustedModelSubstitute[] | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let active = true;
    setResult(null);
    setFailed(false);
    const timeout = setTimeout(() => {
      if (active) setFailed(true);
      active = false;
    }, 30_000);
    void window.browser.getManagedModelSubstitutes(installationId).then(
      (value) => {
        if (active) setResult(value);
        clearTimeout(timeout);
      },
      () => {
        if (active) setFailed(true);
        clearTimeout(timeout);
      },
    );
    return () => {
      active = false;
      clearTimeout(timeout);
    };
  }, [installationId, attempt]);
  if (failed)
    return (
      <div role="alert">
        <p>Compatibility checks could not be completed. Your installed model is still available.</p>
        <Button size="sm" onClick={() => setAttempt((value) => value + 1)}>
          Retry compatibility checks
        </Button>
      </div>
    );
  if (result === null) return <p role="status">Checking installed substitutes…</p>;
  if (result.length === 0)
    return <p>No installed substitutes for this model's current request context.</p>;
  return (
    <ul className="substitute-list">
      {result.map((substitute) => (
        <li key={`${substitute.repository}@${substitute.exactRevision}`}>
          <strong>{substitute.repository}</strong>
          <span>
            {substitute.compatibility} ·{' '}
            {substitute.developerCertified ? 'Developer tested' : 'Not developer-tested'}
            {' · '}
            {substitute.contextOrigin
              ? `${substitute.contextOrigin} / ${substitute.contextRole}`
              : `Technical ${substitute.contextRole} context`}
          </span>
        </li>
      ))}
    </ul>
  );
}
