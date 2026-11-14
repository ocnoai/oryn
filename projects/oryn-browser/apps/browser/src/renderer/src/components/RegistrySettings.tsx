import { useEffect, useState } from 'react';
import type { RegistryStatus } from '../../../shared/ipc';
export function RegistrySettings() {
  const [status, setStatus] = useState<RegistryStatus>();
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    void window.browser
      .getRegistryStatus()
      .then((v) => {
        if (active) setStatus(v);
      })
      .catch(() => {
        if (active) setError(true);
      });
    return () => {
      active = false;
    };
  }, []);
  return (
    <section aria-label="Model recommendation catalog">
      <label>
        <span>Refresh model recommendations automatically</span>
        <input
          type="checkbox"
          disabled={!status?.configured}
          checked={status?.enabled ?? false}
          onChange={(e) =>
            void window.browser
              .setRegistryRefresh(e.target.checked)
              .then(setStatus)
              .catch(() => setError(true))
          }
        />
      </label>
      <p role="status">
        {error
          ? 'Recommendation settings unavailable.'
          : status
            ? status.status === 'recovery-required'
              ? 'Recommendation security state needs recovery. Affected selections remain blocked.'
              : `Catalog ${status.catalogVersion} · ${status.status === 'refresh-failed' ? 'Last refresh failed; using the validated local catalog' : status.configured ? (status.stale ? 'Update is stale' : 'Stored locally') : 'Bundled defaults; remote service is not configured'}. Model selection works offline.`
            : 'Loading catalog status…'}
      </p>
    </section>
  );
}
