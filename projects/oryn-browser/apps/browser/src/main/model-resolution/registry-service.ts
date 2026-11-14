import { cpus, totalmem } from 'node:os';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  bundledSnapshot,
  SnapshotCache,
  boundedJson,
  canonical,
  discover,
  recommend,
  digest,
  type Snapshot,
  type Variant,
  type Facts,
  type Decision,
  type Checkpoint,
} from '@ocno/oryn-registry';
import type { CandidateExecutionProfile, ResolvedModelRef } from '@ocno/oryn-protocol';
import { JsonFileStore, isRecord } from '../data/json-file-store';

export class RegistryService {
  readonly cache: SnapshotCache;
  private lastCheck = 0;
  private etag?: string;
  private refreshing?: Promise<void>;
  private readonly ephemeral = new Map<string, Variant>();
  private readonly discoveries = new WeakMap<Snapshot, Variant[]>();
  private readonly settings: JsonFileStore<{ enabled: boolean }>;
  private refreshAbort?: AbortController;
  private refreshFailed = false;
  constructor(
    path: string,
    private options: {
      endpoint?: string;
      keys?: Record<string, string>;
      enabled?: boolean;
      offline?: boolean;
      fetcher?: typeof fetch;
      facts?: Facts;
    } = {},
  ) {
    mkdirSync(dirname(path), { recursive: true });
    this.cache = new SnapshotCache(path, bundledSnapshot as Snapshot, options.keys ?? {});
    this.settings = new JsonFileStore(
      `${path}.settings`,
      () => ({ enabled: options.enabled ?? true }),
      (v): v is { enabled: boolean } =>
        isRecord(v) && typeof v.enabled === 'boolean' && Object.keys(v).length === 1,
    );
  }
  status() {
    const snapshot = this.capture();
    return {
      enabled: this.settings.read().enabled,
      configured: !!this.options.endpoint,
      status:
        this.cache.status === 'recovery-required'
          ? this.cache.status
          : this.refreshFailed
            ? 'refresh-failed'
            : this.cache.status,
      catalogVersion: snapshot.catalogVersion,
      publishedAt: snapshot.publishedAt,
      stale: Date.now() - Date.parse(snapshot.publishedAt) > 30 * 86400000,
    };
  }
  setEnabled(enabled: boolean) {
    this.settings.write({ enabled });
    if (!enabled) this.refreshAbort?.abort();
    else {
      this.lastCheck = 0;
      void this.refresh();
    }
    return this.status();
  }
  capture(): Snapshot {
    const snapshot = this.cache.capture();
    this.discoveries.set(snapshot, structuredClone([...this.ephemeral.values()]));
    return snapshot;
  }
  refresh(): Promise<void> {
    if (this.refreshing) return this.refreshing;
    if (
      !this.settings.read().enabled ||
      this.options.offline ||
      !this.options.endpoint ||
      Date.now() - this.lastCheck < 6 * 3600000
    )
      return Promise.resolve();
    this.lastCheck = Date.now();
    this.refreshing = this.fetchRelease()
      .then(() => {
        this.refreshFailed = false;
      })
      .catch(() => {
        this.refreshFailed = !this.refreshAbort?.signal.aborted;
        /* Keep the last complete configuration. */
      })
      .finally(() => {
        this.refreshing = undefined;
      });
    return this.refreshing;
  }
  private async fetchRelease(): Promise<void> {
    this.refreshAbort = new AbortController();
    const signal = this.refreshAbort.signal;
    const base = new URL(this.options.endpoint!);
    if (base.protocol !== 'https:' || base.username || base.password || base.search || base.hash)
      throw new Error('REGISTRY_ENDPOINT_INVALID');
    const response = await boundedJson(new URL('current.json', base), {
      limit: 32768,
      etag: this.etag,
      fetcher: this.options.fetcher,
      signal,
    });
    if (response.unchanged) return;
    // Authenticate before using any remote field to construct a path.
    const { verifyManifest } = await import('@ocno/oryn-registry');
    const manifest = verifyManifest(response.value, this.options.keys ?? {}, 0, 0);
    const snapshot = await boundedJson(new URL(`snapshots/${manifest.sha256}.json`, base), {
      limit: 4 * 1024 * 1024,
      fetcher: this.options.fetcher,
      signal,
    });
    signal.throwIfAborted();
    this.cache.accept(manifest, Buffer.from(canonical(snapshot.value) + '\n'));
    this.etag = response.etag;
  }
  variants(snapshot: Snapshot): Variant[] {
    return [...snapshot.variants, ...(this.discoveries.get(snapshot) ?? [])];
  }
  variant(model: ResolvedModelRef, snapshot: Snapshot): Variant | undefined {
    return this.variants(snapshot).find(
      (v) =>
        v.artifact.repository === model.repository &&
        v.artifact.revision === model.revision &&
        (!model.files || canonical(v.artifact.files) === canonical([...model.files].sort())),
    );
  }
  isRevoked(model: ResolvedModelRef, snapshot: Snapshot): boolean {
    if (this.cache.status === 'recovery-required') return true;
    const variants = this.variants(snapshot).filter(
      (v) => v.artifact.repository === model.repository && v.artifact.revision === model.revision,
    );
    return (
      variants.some(
        (v) =>
          (!model.files || canonical(v.artifact.files) === canonical([...model.files].sort())) &&
          (this.cache.revoked(v.artifact) ||
            snapshot.revokedArtifacts.includes(digest(v.artifact))),
      ) ||
      (!!model.files &&
        this.cache.revoked({
          repository: model.repository,
          revision: model.revision,
          files: [...model.files].sort(),
        }))
    );
  }
  decision(profile: CandidateExecutionProfile, snapshot: Snapshot): Decision {
    const variant = this.variant(profile.model, snapshot);
    const facts: Facts = this.options.facts ?? {
      os: process.platform,
      hardwareFamily:
        process.platform === 'darwin' && process.arch === 'arm64' ? 'apple-silicon' : process.arch,
      chip: cpus()[0]?.model,
      memoryBytes: totalmem(),
    };
    return recommend(
      snapshot,
      {
        ...facts,
        task: profile.technicalCapabilities.task,
        architecture: profile.technicalCapabilities.architectures?.[0],
        checkpoint: variant?.checkpoint
          ? `${variant.checkpoint.repository}@${variant.checkpoint.revision}`
          : undefined,
        representation: variant?.representation,
        quantization: variant?.quantization,
      },
      profile.routingFacts?.version === 1 ? profile.routingFacts.offers : undefined,
    );
  }
  referenceEvidence(profile: CandidateExecutionProfile, snapshot: Snapshot) {
    const variant = this.variant(profile.model, snapshot);
    const context = profile.recommendationContext;
    if (!variant || !context) return undefined;
    // Exact runtime/build, hardware/OS and workload binding. Partial matches cannot
    // turn a benchmark on a different device or workload into a ranking claim.
    return snapshot.evidence
      .filter(
        (e) =>
          e.variantId === variant.id &&
          e.runtime === context.runtime &&
          e.hardware === context.hardware &&
          e.task === profile.technicalCapabilities.task &&
          canonical(e.workload) === canonical({ fingerprint: context.workload }) &&
          ['measured', 'reported'].includes(e.kind) &&
          e.sampleCount >= 2 &&
          Date.parse(e.measuredAt) <= Date.now() &&
          Date.now() < Date.parse(e.expiresAt),
      )
      .sort((a, b) => b.measuredAt.localeCompare(a.measuredAt) || a.id.localeCompare(b.id))[0];
  }
  async search(
    model: ResolvedModelRef,
    signal: AbortSignal,
  ): Promise<{
    variants: Variant[];
    issues: Array<{ repository: string; reason: string }>;
    complete: false;
    requests: number;
  }> {
    if (this.options.offline) throw new Error('SEARCH_OFFLINE');
    const snapshot = this.capture();
    const known = this.variant(model, snapshot);
    const source: Checkpoint = known?.checkpoint ?? {
      repository: model.repository,
      revision: model.revision,
    };
    const result = await discover(source, { signal, fetcher: this.options.fetcher });
    signal.throwIfAborted();
    for (const variant of result.variants) this.ephemeral.set(variant.id, variant);
    while (this.ephemeral.size > 128) this.ephemeral.delete(this.ephemeral.keys().next().value!);
    return result;
  }
}
