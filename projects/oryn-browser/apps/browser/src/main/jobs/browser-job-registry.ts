import { exactModelKey } from '@ocno/oryn-protocol';
import { taskRequirementsFrom, validateTaskInvocation } from '@ocno/oryn-protocol';
import { isSharedTaskContract } from '@ocno/oryn-protocol';
import { validateModelWorkspaceInput } from '../model-workspace/task-contract-registry';
import { randomBytes } from 'node:crypto';
import { EventEmitter } from 'node:events';

import type {
  EngineInputArtifactBinding,
  EngineJobEvent,
  EngineJobReference,
  EngineJobState,
  EngineJobStatus,
} from '@ocno/oryn-protocol';

import { formatStorageSize as formatBytes } from '../../shared/format-storage-size';
import type { ArtifactLeaseSet } from '../artifact-manager/artifact-lease-manager';
import type { ArtifactManager } from '../artifact-manager/artifact-manager';
import { ArtifactError } from '../artifact-manager/types';
import type { BrowserCallerContext, BrowserCallerContextService } from '../browser-caller-context';
import type { AIActivityRecord, AIActivityStore } from '../data/ai-activity-store';
import type {
  BrowserModelSession,
  BrowserModelSessionRegistry,
} from '../model-resolution/browser-model-sessions';

interface EngineJobClient {
  readonly engineInstanceId?: string;
  createJob(
    jobId: string,
    sessionId: string,
    input: unknown,
    inputArtifacts: EngineInputArtifactBinding[],
  ): Promise<EngineJobReference>;
  cancelJob(
    jobId: string,
    reason: 'website' | 'user' | 'resource-policy' | 'runtime-shutdown',
  ): Promise<EngineJobStatus>;
  on(event: string, listener: (value: unknown) => void): unknown;
  removeListener(event: string, listener: (value: unknown) => void): unknown;
}

interface BrowserJobRecord {
  jobId: string;
  sessionId: string;
  owner: BrowserCallerContext;
  session: BrowserModelSession;
  state: EngineJobState;
  sequence: number;
  activityId?: string;
  inputPreview?: string;
  progressStage?: string;
  progress?: number;
  queue?: { position: number; jobsAhead: number; reason: string };
  cancellationRequested: boolean;
  cancellationReason?: 'website' | 'user';
  lease?: ArtifactLeaseSet;
  completedAt?: number;
}

export interface PublicLocalAIJobEvent {
  event: string;
  timestamp: number;
  jobId: string;
  sessionId: string;
  sequence: number;
  data: Record<string, unknown>;
}

export interface TrustedLocalAIJob {
  jobId: string;
  sessionId: string;
  origin: string;
  role: string;
  requestedModel: string;
  resolvedModel: string;
  substituted: boolean;
  state: EngineJobState;
  progressStage?: string;
  progress?: number;
  queue?: { position: number; jobsAhead: number; reason: string };
  inputPreview?: string;
  runner: string;
  backend: string;
  cancellationRequested: boolean;
}

export class BrowserJobError extends Error {
  constructor(
    readonly code:
      | 'JOB_NOT_OWNED'
      | 'JOB_INPUT_INVALID'
      | 'JOB_EVENT_INVALID'
      | 'JOB_QUEUE_FULL'
      | 'SESSION_NOT_OWNED',
  ) {
    super(code);
    this.name = 'BrowserJobError';
  }
}

export class BrowserJobRegistry extends EventEmitter {
  private static readonly MAX_LIVE_JOBS_PER_ORIGIN = 16;
  private static readonly MAX_SUBMISSIONS_PER_ORIGIN_PER_MINUTE = 60;
  private static readonly MAX_TERMINAL_JOBS = 256;
  private static readonly TERMINAL_RETENTION_MS = 15 * 60_000;
  private readonly jobs = new Map<string, BrowserJobRecord>();
  private readonly originSubmissions = new Map<string, number[]>();
  private readonly eventChains = new Map<string, Promise<void>>();
  private readonly engineEventListener = (value: unknown) => this.enqueueEngineEvent(value);
  private readonly engineStateListener = (value: unknown) => {
    if (value === 'error' || value === 'stopped') this.failAllForEngineExit();
  };

  constructor(
    private readonly engine: EngineJobClient,
    private readonly sessions: BrowserModelSessionRegistry,
    private readonly artifacts: ArtifactManager,
    private readonly callerContexts: BrowserCallerContextService,
    private readonly activity: AIActivityStore,
  ) {
    super();
    engine.on('job-event', this.engineEventListener);
    engine.on('state-changed', this.engineStateListener);
  }

  async invoke(owner: BrowserCallerContext, request: unknown): Promise<EngineJobReference> {
    this.pruneTerminalJobs();
    this.enforceOriginQuota(owner.origin);
    const invocation = validateInvocation(request);
    const session = this.sessions.requireOwned(invocation.sessionId, owner);
    validateStructuredInput(invocation.input);
    invocation.input = structuredClone(invocation.input);
    if (session.taskProfile) {
      try {
        validateTaskInvocation(
          session.taskProfile,
          invocation.input,
          taskRequirementsFrom(session.request.requirements),
        );
      } catch {
        throw new BrowserJobError('JOB_INPUT_INVALID');
      }
    }
    const task = session.capabilities.task;
    const contract =
      session.taskProfile?.contract.id ??
      (task === 'image-feature-extraction' ? 'image-embeddings-v1' : `${task}-v1`);
    if (isSharedTaskContract(contract)) {
      try {
        validateModelWorkspaceInput(contract, invocation.input);
      } catch {
        throw new BrowserJobError('JOB_INPUT_INVALID');
      }
    }
    const artifactIds = extractArtifactIds(invocation.input);
    const lease =
      artifactIds.length > 0
        ? this.artifacts.acquireEngineInputLeases(owner, artifactIds)
        : undefined;
    const jobId = `job_${randomBytes(24).toString('base64url')}`;
    const record: BrowserJobRecord = {
      jobId,
      sessionId: session.sessionId,
      owner: { ...owner },
      session,
      state: 'created',
      sequence: 0,
      inputPreview: previewInput(invocation.input, lease?.bindings ?? []),
      cancellationRequested: false,
      lease,
    };
    record.activityId = this.activity.create(activityRecord(record, invocation.input));
    this.jobs.set(jobId, record);
    this.emitActivity();
    try {
      const reference = await this.engine.createJob(
        jobId,
        session.sessionId,
        invocation.input,
        lease?.bindings ?? [],
      );
      record.state = reference.state;
      record.queue = reference.queue;
      this.emitActivity();
      return reference;
    } catch (error) {
      this.jobs.delete(jobId);
      this.emitActivity();
      await lease?.release();
      this.activity.complete(record.activityId, {
        status: 'failed',
        completedAt: Date.now(),
        errorCode: errorCode(error),
      });
      throw error;
    }
  }

  async cancelOwned(jobId: string, owner: BrowserCallerContext): Promise<void> {
    const record = this.requireOwned(jobId, owner);
    await this.cancel(record, 'website');
  }

  async cancelTrusted(jobId: string): Promise<void> {
    const record = this.jobs.get(jobId);
    if (!record) throw new BrowserJobError('JOB_NOT_OWNED');
    await this.cancel(record, 'user');
  }

  async cancelAllForOrigin(origin: string): Promise<void> {
    await Promise.all(
      [...this.jobs.values()]
        .filter((record) => record.owner.origin === origin && !isTerminal(record.state))
        .map((record) => this.cancel(record, 'user')),
    );
  }

  async cancelAllTrusted(): Promise<void> {
    await Promise.all(
      [...this.jobs.values()]
        .filter((record) => !isTerminal(record.state))
        .map((record) => this.cancel(record, 'user')),
    );
  }

  async cancelAllForModel(model: {
    provider: string;
    repository: string;
    revision: string;
  }): Promise<void> {
    const key = exactModelKey(model);
    await Promise.all(
      [...this.jobs.values()]
        .filter(
          (record) =>
            !isTerminal(record.state) &&
            `${record.session.resolution.resolvedModel.provider}:${record.session.resolution.resolvedModel.repository}@${record.session.resolution.resolvedModel.revision}` ===
              key,
        )
        .map((record) => this.cancel(record, 'user')),
    );
  }

  private async cancel(record: BrowserJobRecord, reason: 'website' | 'user'): Promise<void> {
    record.cancellationRequested = true;
    record.cancellationReason = reason;
    this.artifacts.cancelPublication(record.jobId);
    this.emitActivity();
    const status = await this.engine.cancelJob(record.jobId, reason);
    if (!isTerminal(record.state) && !isTerminal(status.state)) record.state = status.state;
    this.emitActivity();
  }

  requireOwned(jobId: string, owner: BrowserCallerContext): BrowserJobRecord {
    const record = this.jobs.get(jobId);
    if (
      !record ||
      record.owner.origin !== owner.origin ||
      record.owner.tabId !== owner.tabId ||
      record.owner.webContentsId !== owner.webContentsId ||
      record.owner.documentId !== owner.documentId ||
      record.owner.frameId !== owner.frameId
    ) {
      throw new BrowserJobError('JOB_NOT_OWNED');
    }
    return record;
  }

  snapshot(): Array<Omit<BrowserJobRecord, 'lease'>> {
    return [...this.jobs.values()].map((record) => ({
      jobId: record.jobId,
      sessionId: record.sessionId,
      owner: record.owner,
      session: record.session,
      state: record.state,
      sequence: record.sequence,
      activityId: record.activityId,
      inputPreview: record.inputPreview,
      progressStage: record.progressStage,
      progress: record.progress,
      queue: record.queue,
      cancellationRequested: record.cancellationRequested,
      completedAt: record.completedAt,
    }));
  }

  activitySnapshot(): { running: number; queued: number; jobs: TrustedLocalAIJob[] } {
    let running = 0;
    let queued = 0;
    for (const record of this.jobs.values()) {
      if (['loading', 'running', 'streaming', 'saving'].includes(record.state)) running += 1;
      if (record.state === 'created' || record.state === 'queued') queued += 1;
    }
    return {
      running,
      queued,
      jobs: [...this.jobs.values()]
        .filter((record) => !isTerminal(record.state))
        .map((record) => ({
          jobId: record.jobId,
          sessionId: record.sessionId,
          origin: record.owner.origin,
          role: record.session.role,
          requestedModel: record.session.resolution.preferredModel.repository,
          resolvedModel: record.session.resolution.resolvedModel.repository,
          substituted: record.session.resolution.substituted,
          state: record.state,
          progressStage: record.progressStage,
          progress: record.progress,
          queue: record.queue,
          inputPreview: record.inputPreview,
          runner: record.session.runtime.runner,
          backend: record.session.runtime.backend,
          cancellationRequested: record.cancellationRequested,
        })),
    };
  }

  dispose(): void {
    this.engine.removeListener('job-event', this.engineEventListener);
    this.engine.removeListener('state-changed', this.engineStateListener);
  }

  private enqueueEngineEvent(value: unknown): void {
    if (!isEngineJobEvent(value)) return;
    const previous = this.eventChains.get(value.data.jobId) ?? Promise.resolve();
    const next = previous
      .then(() => this.handleEngineEvent(value))
      .catch(() => undefined)
      .finally(() => {
        if (this.eventChains.get(value.data.jobId) === next) {
          this.eventChains.delete(value.data.jobId);
        }
      });
    this.eventChains.set(value.data.jobId, next);
  }

  private async handleEngineEvent(value: EngineJobEvent): Promise<void> {
    const record = this.jobs.get(value.data.jobId);
    if (!record || record.sessionId !== value.data.sessionId) return;
    if (isTerminal(record.state)) {
      record.completedAt = Date.now();
      console.error(`Rejected event after terminal state for ${record.jobId}`);
      return;
    }
    if (value.data.sequence <= record.sequence) {
      console.error(`Rejected non-monotonic event for ${record.jobId}`);
      return;
    }
    record.sequence = value.data.sequence;
    let transformed = value;
    try {
      if (
        record.cancellationRequested &&
        (value.event === 'job.completed' || value.event === 'job.preview')
      )
        throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      if (value.event === 'job.completed') {
        const engineInstanceId = this.engine.engineInstanceId;
        if (containsArtifactOutput(value.data.result)) {
          if (!engineInstanceId) throw new ArtifactError('ARTIFACT_INVALID');
          transformed = {
            ...value,
            data: {
              ...value.data,
              result: await this.artifacts.publishEngineResult(
                record.owner,
                engineInstanceId,
                record.jobId,
                value.data.result,
              ),
            },
          } as EngineJobEvent;
        }
      } else if (value.event === 'job.preview') {
        const engineInstanceId = this.engine.engineInstanceId;
        if (!engineInstanceId) throw new ArtifactError('ARTIFACT_INVALID');
        const published = (await this.artifacts.publishEngineResult(
          record.owner,
          engineInstanceId,
          record.jobId,
          { kind: 'artifact', artifact: value.data.output },
          true,
        )) as { kind: 'artifact'; artifact: unknown };
        transformed = {
          ...value,
          data: { ...value.data, output: published.artifact },
        } as EngineJobEvent;
      }
    } catch (error) {
      const code = error instanceof ArtifactError ? error.code : 'ARTIFACT_INVALID';
      transformed = {
        ...value,
        event: 'job.failed',
        data: {
          jobId: record.jobId,
          sessionId: record.sessionId,
          sequence: record.sequence,
          error: {
            code,
            category: 'artifact',
            message: 'The Local AI result could not be published.',
          },
        },
      } as EngineJobEvent;
    }
    // Engine exit can mark this record terminal while publication is awaiting I/O.
    if (isTerminal(record.state)) return;
    if (record.cancellationRequested && transformed.event === 'job.failed') {
      transformed = {
        ...value,
        event: 'job.cancelled',
        data: {
          jobId: record.jobId,
          sessionId: record.sessionId,
          sequence: record.sequence,
          reason: record.cancellationReason ?? 'user',
        },
      } as EngineJobEvent;
    }
    const event = publicEvent(transformed);
    const state = eventState(transformed);
    if (state) record.state = state;
    if (record.state !== 'queued') record.queue = undefined;
    if (transformed.event === 'job.queue.updated' || transformed.event === 'job.state') {
      const queue = queueStatus(transformed.data);
      if (queue) record.queue = queue;
    }
    if (transformed.event === 'job.progress') {
      record.progressStage =
        typeof transformed.data.stage === 'string'
          ? transformed.data.stage.slice(0, 128)
          : undefined;
      record.progress =
        typeof transformed.data.progress === 'number' ? transformed.data.progress : undefined;
    }
    if (isTerminal(record.state)) {
      if (this.engine.engineInstanceId) {
        await this.artifacts.cleanupEngineJob(this.engine.engineInstanceId, record.jobId);
      }
      await record.lease?.release();
      this.activity.complete(record.activityId, activityCompletion(transformed));
      this.pruneTerminalJobs();
    }
    this.emitActivity();

    if (
      this.callerContexts.hasDocumentAuthority(
        record.owner.tabId,
        record.owner.webContentsId,
        record.owner.documentId,
      )
    ) {
      setImmediate(() => this.emit('website-event', { owner: record.owner, event }));
    }
  }

  private failAllForEngineExit(): void {
    for (const record of this.jobs.values()) {
      if (isTerminal(record.state)) continue;
      this.artifacts.cancelPublication(record.jobId);
      record.state = 'failed';
      record.completedAt = Date.now();
      record.sequence += 1;
      void record.lease?.release();
      this.activity.complete(record.activityId, {
        status: 'failed',
        completedAt: Date.now(),
        errorCode: 'RUNTIME_PROTOCOL_ERROR',
      });
      this.emitActivity();
      if (
        this.callerContexts.hasDocumentAuthority(
          record.owner.tabId,
          record.owner.webContentsId,
          record.owner.documentId,
        )
      ) {
        const event: PublicLocalAIJobEvent = {
          event: 'job.failed',
          timestamp: Date.now(),
          jobId: record.jobId,
          sessionId: record.sessionId,
          sequence: record.sequence,
          data: {
            error: {
              code: 'RUNTIME_PROTOCOL_ERROR',
              message: 'The Local AI runtime stopped before the job completed.',
            },
          },
        };
        setImmediate(() => this.emit('website-event', { owner: record.owner, event }));
      }
    }
    this.pruneTerminalJobs();
  }

  private emitActivity(): void {
    this.emit('activity-changed', this.activitySnapshot());
  }

  private enforceOriginQuota(origin: string): void {
    const live = [...this.jobs.values()].filter(
      (record) => record.owner.origin === origin && !isTerminal(record.state),
    ).length;
    if (live >= BrowserJobRegistry.MAX_LIVE_JOBS_PER_ORIGIN) {
      throw new BrowserJobError('JOB_QUEUE_FULL');
    }
    const now = Date.now();
    const recent = (this.originSubmissions.get(origin) ?? []).filter(
      (timestamp) => now - timestamp < 60_000,
    );
    if (recent.length >= BrowserJobRegistry.MAX_SUBMISSIONS_PER_ORIGIN_PER_MINUTE) {
      this.originSubmissions.set(origin, recent);
      throw new BrowserJobError('JOB_QUEUE_FULL');
    }
    recent.push(now);
    this.originSubmissions.set(origin, recent);
  }

  private pruneTerminalJobs(): void {
    const now = Date.now();
    const terminal = [...this.jobs.values()]
      .filter((record) => isTerminal(record.state))
      .sort((left, right) => (left.completedAt ?? 0) - (right.completedAt ?? 0));
    const excess = Math.max(0, terminal.length - BrowserJobRegistry.MAX_TERMINAL_JOBS);
    for (const [index, record] of terminal.entries()) {
      if (
        index < excess ||
        (record.completedAt !== undefined &&
          now - record.completedAt > BrowserJobRegistry.TERMINAL_RETENTION_MS)
      ) {
        this.jobs.delete(record.jobId);
        this.eventChains.delete(record.jobId);
      }
    }
    for (const [origin, timestamps] of this.originSubmissions) {
      const recent = timestamps.filter((timestamp) => now - timestamp < 60_000);
      if (recent.length > 0) this.originSubmissions.set(origin, recent);
      else this.originSubmissions.delete(origin);
    }
  }
}

function containsArtifactOutput(value: unknown): boolean {
  if (!isPlainRecord(value)) return false;
  if (value.kind === 'artifact') return true;
  return value.kind === 'multi' && Array.isArray(value.outputs)
    ? value.outputs.some(containsArtifactOutput)
    : false;
}

function validateInvocation(value: unknown): { sessionId: string; input: unknown } {
  if (
    typeof value !== 'object' ||
    value === null ||
    Array.isArray(value) ||
    !('sessionId' in value) ||
    typeof value.sessionId !== 'string' ||
    !/^session_[A-Za-z0-9_-]{1,120}$/u.test(value.sessionId) ||
    !('input' in value)
  ) {
    throw new BrowserJobError('JOB_INPUT_INVALID');
  }
  return { sessionId: value.sessionId, input: value.input };
}

function validateStructuredInput(value: unknown): void {
  const forbidden = new Set([
    'pythonSource',
    'shellCommand',
    'entrypointPath',
    'moduleToImport',
    'pipPackages',
    'nativeLibrary',
    'filesystemPath',
    'schedulerLane',
    'priority',
    'bypassQueue',
  ]);
  const seen = new Set<object>();
  let remainingNodes = 100_000;
  let remainingBytes = 1024 * 1024;
  const charge = (bytes: number): void => {
    remainingBytes -= bytes;
    if (remainingBytes < 0) throw new BrowserJobError('JOB_INPUT_INVALID');
  };
  const chargeString = (item: string): void => {
    // UTF-16 length is a cheap lower bound before allocating its JSON encoding.
    if (item.length > remainingBytes) throw new BrowserJobError('JOB_INPUT_INVALID');
    charge(Buffer.byteLength(JSON.stringify(item), 'utf8'));
  };
  const visit = (item: unknown, depth: number): void => {
    if (depth > 32 || --remainingNodes < 0) throw new BrowserJobError('JOB_INPUT_INVALID');
    if (item === null) {
      charge(4);
      return;
    }
    if (typeof item === 'string') {
      chargeString(item);
      return;
    }
    if (typeof item === 'boolean') {
      charge(item ? 4 : 5);
      return;
    }
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) throw new BrowserJobError('JOB_INPUT_INVALID');
      charge(JSON.stringify(item).length);
      return;
    }
    if (typeof item !== 'object') throw new BrowserJobError('JOB_INPUT_INVALID');
    if (seen.has(item)) throw new BrowserJobError('JOB_INPUT_INVALID');
    seen.add(item);
    if (Array.isArray(item)) {
      if (item.length > 10_000) throw new BrowserJobError('JOB_INPUT_INVALID');
      charge(2 + Math.max(0, item.length - 1));
      for (const child of item) visit(child, depth + 1);
    } else {
      const prototype = Object.getPrototypeOf(item);
      if (prototype !== Object.prototype && prototype !== null) {
        throw new BrowserJobError('JOB_INPUT_INVALID');
      }
      const entries = Object.entries(item as Record<string, unknown>);
      if (entries.length > 1_000) throw new BrowserJobError('JOB_INPUT_INVALID');
      charge(2 + Math.max(0, entries.length - 1));
      for (const [key, child] of entries) {
        if (key.length > 256 || forbidden.has(key)) throw new BrowserJobError('JOB_INPUT_INVALID');
        chargeString(key);
        charge(1);
        visit(child, depth + 1);
      }
    }
    seen.delete(item);
  };
  visit(value, 0);
}

function extractArtifactIds(input: unknown): string[] {
  if (typeof input !== 'object' || input === null || Array.isArray(input)) return [];
  const record = input as Record<string, unknown>;
  const ids: string[] = [];
  for (const key of [
    'audioArtifactId',
    'imageArtifactId',
    'videoArtifactId',
    'documentArtifactId',
    'referenceAudioArtifactId',
    'maskVideoArtifactId',
  ]) {
    if (record[key] !== undefined) ids.push(requireArtifactId(record[key]));
  }
  if (record.inputArtifacts !== undefined) {
    if (!Array.isArray(record.inputArtifacts) || record.inputArtifacts.length > 32) {
      throw new BrowserJobError('JOB_INPUT_INVALID');
    }
    ids.push(...record.inputArtifacts.map(requireArtifactId));
  }
  if (record.referenceImageArtifactIds !== undefined) {
    if (
      !Array.isArray(record.referenceImageArtifactIds) ||
      record.referenceImageArtifactIds.length > 2
    )
      throw new BrowserJobError('JOB_INPUT_INVALID');
    ids.push(...record.referenceImageArtifactIds.map(requireArtifactId));
  }
  return [...new Set(ids)];
}

function requireArtifactId(value: unknown): string {
  if (typeof value !== 'string' || !/^artifact_[A-Za-z0-9_-]{20,80}$/u.test(value)) {
    throw new BrowserJobError('JOB_INPUT_INVALID');
  }
  return value;
}

function isEngineJobEvent(value: unknown): value is EngineJobEvent {
  return (
    typeof value === 'object' &&
    value !== null &&
    'event' in value &&
    typeof value.event === 'string' &&
    value.event.startsWith('job.') &&
    'timestamp' in value &&
    typeof value.timestamp === 'number' &&
    'data' in value &&
    typeof value.data === 'object' &&
    value.data !== null &&
    'jobId' in value.data &&
    typeof value.data.jobId === 'string' &&
    'sessionId' in value.data &&
    typeof value.data.sessionId === 'string' &&
    'sequence' in value.data &&
    typeof value.data.sequence === 'number'
  );
}

function publicEvent(event: EngineJobEvent): PublicLocalAIJobEvent {
  const { jobId, sessionId, sequence, ...data } = event.data;
  return {
    event: event.event,
    timestamp: event.timestamp,
    jobId,
    sessionId,
    sequence,
    data: sanitizePublicData(data),
  };
}

function sanitizePublicData(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return {};
  const sanitized = structuredClone(value) as Record<string, unknown>;
  if ('error' in sanitized && typeof sanitized.error === 'object' && sanitized.error !== null) {
    const error = sanitized.error as Record<string, unknown>;
    const code = typeof error.code === 'string' ? error.code : 'JOB_EXECUTION_FAILED';
    sanitized.error = {
      code,
      message: safeJobErrorMessage(code),
    };
  }
  removeInternalPaths(sanitized);
  return sanitized;
}

function safeJobErrorMessage(code: string): string {
  const messages: Record<string, string> = {
    MODEL_MEMORY_INSUFFICIENT:
      'This model is too large for the memory currently available. Quit memory-heavy apps or use a smaller or quantized variant, then try again.',
    MODEL_FORMAT_UNSUPPORTED:
      'This model uses a file layout or metadata format that Oryn cannot safely interpret. Use a compatible or quantized variant of the model.',
    MODEL_RUNTIME_INCOMPATIBLE:
      'The model is installed, but its exact architecture is not compatible with the selected local runtime. Use a compatible variant or another runtime.',
    MODEL_GENERATION_FAILED:
      'The model loaded, but generation failed. Try safer dimensions or fewer steps; if it repeats, use a compatible model variant.',
    MODEL_OUTPUT_FILTERED:
      'The model’s image safety checker withheld the result. Try a different prompt or seed.',
    MODEL_INPUT_INVALID:
      'These generation settings are not valid for the selected runtime. Check the dimensions and options, then try again.',
    RUNTIME_DEPENDENCY_UNAVAILABLE:
      'The local runtime required by this model is unavailable. Restart or update Oryn, then try again.',
    ARTIFACT_IO_ERROR:
      'The model ran, but Oryn could not save its local result. Check free disk space and try again.',
    ALIGNMENT_LANGUAGE_UNSUPPORTED:
      'The selected timestamp aligner does not support the detected language. Use a timing-capable fallback model or estimated timing.',
    ALIGNMENT_OUTPUT_INVALID:
      'The timestamp aligner returned invalid timing data. Try another timing-capable model.',
    PACK_TIMEOUT:
      'The local model did not finish within the allowed time. Try again with less work.',
    PACK_CRASHED:
      'The isolated local runtime stopped unexpectedly. Try again; if it repeats, use a smaller compatible model.',
  };
  return messages[code] ?? 'The Local AI job could not be completed.';
}

function removeInternalPaths(value: unknown): void {
  if (Array.isArray(value)) {
    for (const item of value) removeInternalPaths(item);
    return;
  }
  if (typeof value !== 'object' || value === null) return;
  for (const [key, child] of Object.entries(value)) {
    if (['internalPath', 'localPath', 'filesystemPath'].includes(key)) {
      delete (value as Record<string, unknown>)[key];
    } else {
      removeInternalPaths(child);
    }
  }
}

function eventState(event: EngineJobEvent): EngineJobState | undefined {
  if (event.event === 'job.state') return event.data.state;
  if (event.event === 'job.completed') return 'completed';
  if (event.event === 'job.failed') return 'failed';
  if (event.event === 'job.cancelled') return 'cancelled';
  return undefined;
}

function queueStatus(
  value: unknown,
): { position: number; jobsAhead: number; reason: string } | undefined {
  if (typeof value !== 'object' || value === null) return undefined;
  const direct = value as Record<string, unknown>;
  const candidate =
    typeof direct.queue === 'object' && direct.queue !== null
      ? (direct.queue as Record<string, unknown>)
      : direct;
  if (
    typeof candidate.position !== 'number' ||
    typeof candidate.jobsAhead !== 'number' ||
    typeof candidate.reason !== 'string'
  ) {
    return undefined;
  }
  return {
    position: candidate.position,
    jobsAhead: candidate.jobsAhead,
    reason: candidate.reason,
  };
}

function isTerminal(state: EngineJobState): boolean {
  return state === 'completed' || state === 'failed' || state === 'cancelled';
}

function activityRecord(
  record: BrowserJobRecord,
  input: unknown,
): Omit<AIActivityRecord, 'id' | 'createdAt'> {
  const resolution = record.session.resolution;
  return {
    origin: record.owner.origin,
    role: record.session.role,
    sessionId: record.sessionId,
    jobId: record.jobId,
    requestedModel: modelIdentity(resolution.preferredModel),
    resolvedModel: modelIdentity(resolution.resolvedModel),
    resolution: {
      compatibility: resolution.compatibility,
      substituted: resolution.substituted,
      overriddenByUser: resolution.overriddenByUser,
      reason: resolution.resolutionReason,
    },
    invocation: {
      input: structuredClone(input),
      artifacts: record.lease?.bindings.map((binding) => ({
        artifactId: binding.artifactId,
        mimeType: binding.mimeType,
        sizeBytes: binding.sizeBytes,
        ...(binding.name ? { name: binding.name } : {}),
      })),
      contentPolicy: 'metadata-only',
    },
    execution: {
      runner: record.session.runtime.runner,
      backend: record.session.runtime.backend,
      device: record.session.runtime.deviceClass,
      status: 'running',
    },
  };
}

function modelIdentity(value: { provider: 'huggingface'; repository: string; revision: string }) {
  return { provider: value.provider, repository: value.repository, revision: value.revision };
}

function activityCompletion(event: EngineJobEvent): {
  status: 'completed' | 'failed' | 'cancelled';
  completedAt: number;
  result?: AIActivityRecord['result'];
  errorCode?: string;
  execution?: Partial<AIActivityRecord['execution']>;
} {
  if (event.event === 'job.completed') {
    const execution = event.data.execution;
    return {
      status: 'completed',
      completedAt: event.timestamp,
      result: resultMetadata(event.data.result),
      ...(execution
        ? {
            execution: {
              ...(execution.runner ? { runner: execution.runner } : {}),
              ...(execution.backend ? { backend: execution.backend } : {}),
              ...(execution.durationMs !== undefined ? { durationMs: execution.durationMs } : {}),
              ...(execution.planId ? { planId: execution.planId } : {}),
              ...(execution.foundationId ? { foundationId: execution.foundationId } : {}),
              ...(execution.adapterId ? { adapterId: execution.adapterId } : {}),
              ...(execution.candidateId ? { candidateId: execution.candidateId } : {}),
              ...(execution.variantId ? { variantId: execution.variantId } : {}),
              ...(execution.precision ? { precision: execution.precision } : {}),
              ...(execution.quantization ? { quantization: execution.quantization } : {}),
              ...(execution.fallbacks ? { fallbacks: execution.fallbacks } : {}),
            },
          }
        : {}),
    };
  }
  if (event.event === 'job.cancelled') {
    return { status: 'cancelled', completedAt: event.timestamp };
  }
  const error = (event.data as unknown as Record<string, unknown>).error;
  return {
    status: 'failed',
    completedAt: event.timestamp,
    errorCode:
      typeof error === 'object' &&
      error !== null &&
      'code' in error &&
      typeof error.code === 'string'
        ? error.code
        : 'JOB_EXECUTION_FAILED',
  };
}

function resultMetadata(value: unknown): AIActivityRecord['result'] {
  if (!isPlainRecord(value)) return { kind: 'structured' };
  const kind = ['structured', 'artifact', 'multi'].includes(String(value.kind))
    ? (value.kind as 'structured' | 'artifact' | 'multi')
    : 'structured';
  const artifactIds = collectArtifactIds(value).slice(0, 64);
  let sizeBytes: number | undefined;
  try {
    sizeBytes = Buffer.byteLength(JSON.stringify(value), 'utf8');
  } catch {
    sizeBytes = undefined;
  }
  return {
    kind,
    ...(artifactIds.length > 0 ? { artifactIds } : {}),
    ...(sizeBytes !== undefined ? { sizeBytes } : {}),
  };
}

function collectArtifactIds(value: unknown): string[] {
  const ids = new Set<string>();
  const visit = (item: unknown, depth: number) => {
    if (depth > 8 || item === null || typeof item !== 'object') return;
    if (Array.isArray(item)) {
      for (const child of item.slice(0, 64)) visit(child, depth + 1);
      return;
    }
    for (const [key, child] of Object.entries(item).slice(0, 128)) {
      if (
        (key === 'artifactId' || key === 'id') &&
        typeof child === 'string' &&
        /^artifact_[A-Za-z0-9_-]{20,80}$/u.test(child)
      ) {
        ids.add(child);
      } else {
        visit(child, depth + 1);
      }
    }
  };
  visit(value, 0);
  return [...ids];
}

function previewInput(input: unknown, artifacts: EngineInputArtifactBinding[]): string | undefined {
  const firstArtifact = artifacts[0];
  if (firstArtifact) {
    return `${firstArtifact.name ?? firstArtifact.mimeType} · ${formatBytes(firstArtifact.sizeBytes)}`;
  }
  try {
    const serialized = typeof input === 'string' ? input : JSON.stringify(input);
    if (!serialized) return undefined;
    return serialized.length > 160 ? `${serialized.slice(0, 157)}…` : serialized;
  } catch {
    return undefined;
  }
}

function errorCode(value: unknown): string {
  if (value instanceof Error && 'protocolError' in value) {
    const protocolError = value.protocolError;
    if (
      typeof protocolError === 'object' &&
      protocolError !== null &&
      'code' in protocolError &&
      typeof protocolError.code === 'string'
    ) {
      return protocolError.code;
    }
  }
  return 'JOB_SUBMISSION_FAILED';
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
