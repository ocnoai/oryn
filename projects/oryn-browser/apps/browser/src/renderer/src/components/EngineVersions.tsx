import { useEffect, useRef, useState } from 'react';
import type { EngineVersion, EngineVersionStatus } from '../../../shared/engine-versions';
import { Badge, Button } from './ui';
import { withRequestTimeout } from '../request-timeout';

const statusLabels: Record<EngineVersionStatus, string> = {
  unchecked: 'Not checked',
  current: 'Up to date',
  update: 'Update available',
  ahead: 'Ahead of stable',
  unknown: 'Comparison unavailable',
  error: 'Check failed',
  managed: 'Updated with Oryn',
};

export function EngineVersions() {
  const [entries, setEntries] = useState<EngineVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [checking, setChecking] = useState<Set<string>>(new Set());
  const generation = useRef(0);

  const load = async () => {
    const request = ++generation.current;
    setLoading(true);
    setError(false);
    try {
      const versions = await withRequestTimeout(window.browser.getEngineVersions());
      if (request === generation.current) setEntries(versions);
    } catch {
      if (request === generation.current) setError(true);
    } finally {
      if (request === generation.current) setLoading(false);
    }
  };
  useEffect(() => {
    void load();
    return () => {
      generation.current++;
    };
  }, []);

  const check = async (id: string) => {
    const request = generation.current;
    setChecking((previous) => new Set(previous).add(id));
    try {
      const result = await withRequestTimeout(window.browser.checkEngineVersion(id), 30_000);
      if (request === generation.current)
        setEntries((previous) => previous.map((entry) => (entry.id === id ? result : entry)));
    } catch {
      if (request === generation.current)
        setEntries((previous) =>
          previous.map((entry) =>
            entry.id === id
              ? {
                  ...entry,
                  status: 'error',
                  message: 'Could not check this engine. Please try again.',
                }
              : entry,
          ),
        );
    } finally {
      if (request === generation.current)
        setChecking((previous) => {
          const next = new Set(previous);
          next.delete(id);
          return next;
        });
    }
  };

  return (
    <section className="engine-versions" aria-label="Engine versions">
      <div className="engine-versions-intro">
        <h3>Engine versions</h3>
        <p>
          Check the latest stable releases from each publisher. Engine upgrades are delivered
          through Oryn updates after compatibility testing.
        </p>
        <small>
          Checks contact the release source when you click. Results are kept for this session.
        </small>
      </div>
      {loading && <p role="status">Loading engine versions…</p>}
      {error && (
        <div role="alert">
          <p>Could not read engine versions.</p>
          <Button size="sm" onClick={() => void load()}>
            Retry loading versions
          </Button>
        </div>
      )}
      {!loading &&
        !error &&
        (['AI engines', 'Oryn runtime', 'Browser runtime'] as const).map((group) => (
          <section key={group} className="engine-version-group" aria-label={group}>
            <h4>{group}</h4>
            {entries
              .filter((entry) => entry.group === group)
              .map((entry) => {
                const busy = checking.has(entry.id);
                return (
                  <article key={entry.id} className="engine-version-row" aria-label={entry.name}>
                    <div className="engine-version-heading">
                      <strong>{entry.name}</strong>
                      <Button
                        variant="ghost"
                        size="sm"
                        icon="refresh"
                        className="engine-version-check"
                        aria-label={`Check version for ${entry.name}`}
                        disabled={busy || !entry.checkable}
                        title={
                          entry.checkable
                            ? `Check ${entry.source} for a stable release`
                            : entry.detail
                        }
                        onClick={() => void check(entry.id)}
                      >
                        {busy ? 'Checking…' : 'Check version'}
                      </Button>
                    </div>
                    <div className="engine-version-values">
                      <span>
                        {entry.versionOrigin === 'pinned'
                          ? 'Project pin'
                          : entry.versionOrigin === 'running'
                            ? 'Running'
                            : 'Bundled'}{' '}
                        <code>{entry.currentVersion ?? 'Unknown'}</code>
                      </span>
                      {entry.latestVersion && (
                        <span>
                          {entry.status === 'error' ? 'Last known' : 'Latest stable'}{' '}
                          <code>{entry.latestVersion}</code>
                        </span>
                      )}
                      <span role="status" aria-live="polite">
                        <Badge
                          tone={
                            entry.status === 'update'
                              ? 'brand'
                              : entry.status === 'current'
                                ? 'success'
                                : entry.status === 'error'
                                  ? 'warning'
                                  : 'neutral'
                          }
                        >
                          {busy ? 'Checking…' : statusLabels[entry.status]}
                        </Badge>
                      </span>
                    </div>
                    {entry.detail && <p>{entry.detail}</p>}
                    {entry.message && (
                      <p className={entry.status === 'error' ? 'engine-version-error' : undefined}>
                        {entry.message}
                      </p>
                    )}
                    <small>
                      Source: {entry.source}
                      {entry.checkedAt && (
                        <>
                          {' '}
                          · Checked{' '}
                          <time dateTime={entry.checkedAt}>
                            {new Date(entry.checkedAt).toLocaleString()}
                          </time>
                        </>
                      )}
                    </small>
                  </article>
                );
              })}
          </section>
        ))}
    </section>
  );
}
