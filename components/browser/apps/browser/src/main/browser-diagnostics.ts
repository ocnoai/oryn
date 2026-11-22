export type EngineFailureReason = 'process-error' | 'protocol-error' | 'unexpected-exit' | 'signal';

export interface EngineFailureSummary {
  count: number;
  last?: { reason: EngineFailureReason; exitCode: number | null };
}

const engineReasons: ReadonlySet<string> = new Set([
  'process-error',
  'protocol-error',
  'unexpected-exit',
  'signal',
]);

const rendererFailures: ReadonlySet<string> = new Set([
  'abnormal-exit',
  'crashed',
  'killed',
  'oom',
  'launch-failed',
  'integrity-failure',
]);

/** Process-local aggregate only: no document identities or crash memory are retained. */
export class BrowserFailureDiagnostics {
  private crashes = 0;

  recordRendererExit(reason: string): void {
    if (rendererFailures.has(reason)) this.crashes = Math.min(this.crashes + 1, 1_000_000);
  }

  get rendererCrashes(): number {
    return this.crashes;
  }
}

function boundedCount(value: unknown): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
    ? Math.min(value, 1_000_000)
    : 0;
}

/** Explicit allowlist: never serialize BrowserData, Engine errors, URLs or caller objects. */
export function browserDiagnostics(input: {
  appVersion: string;
  versions: Record<string, string | undefined>;
  platform: string;
  architecture: string;
  engineState: string;
  windows: number;
  tabs: number;
  downloads: number;
  activeDownloads: boolean;
  buildFlavor?: 'source' | 'release' | 'local-development';
  compiledMainSha256?: string;
  rendererCrashes?: number;
  engineFailures?: EngineFailureSummary;
}) {
  const failure = input.engineFailures?.last;
  const reason = failure && engineReasons.has(failure.reason) ? failure.reason : undefined;
  const code = failure?.exitCode;
  return {
    formatVersion: 2,
    collectedAt: new Date().toISOString(),
    appVersion: input.appVersion,
    versions: {
      electron: input.versions.electron,
      chromium: input.versions.chrome,
      node: input.versions.node,
    },
    platform: input.platform,
    architecture: input.architecture,
    engineState: input.engineState,
    counts: { windows: input.windows, tabs: input.tabs, downloads: input.downloads },
    activeDownloads: input.activeDownloads,
    build: {
      flavor: ['source', 'release', 'local-development'].includes(input.buildFlavor ?? '')
        ? input.buildFlavor
        : 'source',
      ...(input.compiledMainSha256 && /^[a-f0-9]{64}$/u.test(input.compiledMainSha256)
        ? { compiledMainSha256: input.compiledMainSha256 }
        : {}),
    },
    failures: {
      rendererCrashes: boundedCount(input.rendererCrashes),
      engine: {
        count: boundedCount(input.engineFailures?.count),
        ...(reason
          ? {
              last: {
                reason,
                exitCode:
                  typeof code === 'number' &&
                  Number.isSafeInteger(code) &&
                  code >= -(2 ** 31) &&
                  code < 2 ** 31
                    ? code
                    : null,
              },
            }
          : {}),
      },
    },
  };
}
