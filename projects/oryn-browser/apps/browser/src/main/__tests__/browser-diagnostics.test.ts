import { expect, it } from 'vitest';
import { BrowserFailureDiagnostics, browserDiagnostics } from '../browser-diagnostics';

it('exports only allowed summary fields, excluding nested credentials, URLs and environment data', () => {
  const input = {
    appVersion: 'fixture',
    platform: 'darwin',
    architecture: 'arm64',
    engineState: 'unavailable',
    windows: 2,
    tabs: 5,
    downloads: 1,
    activeDownloads: false,
    versions: {
      electron: '43.4.0',
      chrome: 'fixture',
      node: 'fixture',
      extra: 'secret-version-data',
    },
    error: 'secret-error-with-path',
    url: 'https://secret.example/path',
    password: 'secret-password',
    prompt: 'secret-prompt',
    environment: { TOKEN: 'secret-token' },
  };
  const output = browserDiagnostics(input);
  expect(output.counts).toEqual({ windows: 2, tabs: 5, downloads: 1 });
  expect(JSON.stringify(output)).not.toContain('secret');
  expect(output.engineState).toBe('unavailable');
});

it('counts actual renderer failures without retaining documents or normal close events', () => {
  const failures = new BrowserFailureDiagnostics();
  for (const reason of ['clean-exit', 'memory-eviction', 'https://secret.example'])
    failures.recordRendererExit(reason);
  expect(failures.rendererCrashes).toBe(0);
  failures.recordRendererExit('crashed');
  failures.recordRendererExit('oom');
  failures.recordRendererExit('killed');
  expect(failures.rendererCrashes).toBe(3);
  expect(JSON.stringify(failures)).not.toContain('secret');
});

it('exports only closed failure facts and an opaque compiled-build digest', () => {
  const output = browserDiagnostics({
    appVersion: '0.1.0',
    versions: { electron: 'fixture' },
    platform: 'darwin',
    architecture: 'arm64',
    engineState: 'error',
    windows: 1,
    tabs: 2,
    downloads: 0,
    activeDownloads: false,
    buildFlavor: 'local-development',
    compiledMainSha256: 'a'.repeat(64),
    rendererCrashes: 2,
    engineFailures: {
      count: 1,
      last: { reason: 'unexpected-exit', exitCode: 23, secret: 'secret-stderr' },
      secret: 'secret-url',
    } as Parameters<typeof browserDiagnostics>[0]['engineFailures'],
  });
  expect(output.build).toEqual({
    flavor: 'local-development',
    compiledMainSha256: 'a'.repeat(64),
  });
  expect(output.failures).toEqual({
    rendererCrashes: 2,
    engine: { count: 1, last: { reason: 'unexpected-exit', exitCode: 23 } },
  });
  expect(JSON.stringify(output)).not.toContain('secret');
});

it('does not export unrecognized failure reasons, paths masquerading as builds or invalid counts', () => {
  const output = browserDiagnostics({
    appVersion: 'fixture',
    versions: {},
    platform: 'darwin',
    architecture: 'arm64',
    engineState: 'error',
    windows: 0,
    tabs: 0,
    downloads: 0,
    activeDownloads: false,
    compiledMainSha256: '/secret/profile/path',
    rendererCrashes: Number.POSITIVE_INFINITY,
    engineFailures: {
      count: -1,
      last: { reason: 'secret-error', exitCode: NaN },
    } as unknown as Parameters<typeof browserDiagnostics>[0]['engineFailures'],
  });
  expect(output.build).toEqual({ flavor: 'source' });
  expect(output.failures).toEqual({ rendererCrashes: 0, engine: { count: 0 } });
  expect(JSON.stringify(output)).not.toContain('secret');
});
