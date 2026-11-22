import { randomBytes } from 'node:crypto';

import { JsonFileStore, isRecord } from './json-file-store';

export type AIActivityContentPolicy = 'full' | 'metadata-only';
export type AIActivityRetentionDays = 7 | 30 | null;
export type AIActivityStatus = 'running' | 'completed' | 'failed' | 'cancelled';

export interface AIActivityArtifactMetadata {
  artifactId: string;
  name?: string;
  mimeType: string;
  sizeBytes: number;
}

export interface AIActivityRecord {
  id: string;
  createdAt: number;
  completedAt?: number;
  origin: string;
  role: string;
  sessionId: string;
  jobId: string;
  requestedModel: { provider: 'huggingface'; repository: string; revision: string };
  resolvedModel: { provider: 'huggingface'; repository: string; revision: string };
  resolution: {
    compatibility: string;
    substituted: boolean;
    overriddenByUser: boolean;
    reason: string;
  };
  invocation: {
    input?: unknown;
    artifacts?: AIActivityArtifactMetadata[];
    contentPolicy: AIActivityContentPolicy;
  };
  execution: {
    runner: string;
    backend: string;
    device: string;
    durationMs?: number;
    status: AIActivityStatus;
    planId?: string;
    foundationId?: string;
    adapterId?: string;
    candidateId?: string;
    variantId?: string;
    precision?: string;
    quantization?: string;
    fallbacks?: Array<{
      fromCandidateId: string;
      toCandidateId: string;
      reasonCode: string;
    }>;
  };
  result?: {
    kind: 'structured' | 'artifact' | 'multi';
    artifactIds?: string[];
    sizeBytes?: number;
    chat?: {
      finishReason: 'stop' | 'length' | 'unknown';
      thinking: {
        requested: boolean | null;
        applied: boolean;
        effective: boolean | null;
        reason: string;
      };
    };
  };
  errorCode?: string;
}

export interface AIActivitySettings {
  keepActivity: boolean;
  contentPolicy: AIActivityContentPolicy;
  retentionDays: AIActivityRetentionDays;
}

interface AIActivityData {
  version: 1;
  settings: AIActivitySettings;
  records: AIActivityRecord[];
}

export class AIActivityStore {
  private readonly store: JsonFileStore<AIActivityData>;

  constructor(
    filePath: string,
    private readonly now: () => number = Date.now,
  ) {
    this.store = new JsonFileStore(filePath, defaultData, isAIActivityData);
    this.prune();
  }

  settings(): AIActivitySettings {
    return structuredClone(this.store.read().settings);
  }

  list(): AIActivityRecord[] {
    this.prune();
    return this.store
      .read()
      .records.slice()
      .sort((left, right) => right.createdAt - left.createdAt)
      .map((record) => structuredClone(record));
  }

  create(value: Omit<AIActivityRecord, 'id' | 'createdAt'>): string | undefined {
    const current = this.store.read();
    if (!current.settings.keepActivity) return undefined;
    const id = `activity_${randomBytes(24).toString('base64url')}`;
    const record: AIActivityRecord = {
      ...structuredClone(value),
      id,
      createdAt: this.now(),
      invocation: {
        ...structuredClone(value.invocation),
        input:
          current.settings.contentPolicy === 'full'
            ? structuredClone(value.invocation.input)
            : undefined,
        contentPolicy: current.settings.contentPolicy,
      },
    };
    this.store.write({
      ...current,
      records: [...this.retained(current), record].slice(-10_000),
    });
    return id;
  }

  complete(
    id: string | undefined,
    update: {
      status: Exclude<AIActivityStatus, 'running'>;
      completedAt: number;
      result?: AIActivityRecord['result'];
      errorCode?: string;
      execution?: Partial<AIActivityRecord['execution']>;
    },
  ): void {
    if (!id) return;
    this.store.update((current) => ({
      ...current,
      records: current.records.map((record) =>
        record.id === id
          ? {
              ...record,
              completedAt: update.completedAt,
              execution: {
                ...record.execution,
                ...structuredClone(update.execution ?? {}),
                status: update.status,
                durationMs: Math.max(0, update.completedAt - record.createdAt),
              },
              ...(update.result ? { result: update.result } : {}),
              ...(update.errorCode ? { errorCode: update.errorCode } : {}),
            }
          : record,
      ),
    }));
  }

  updateSettings(value: Partial<AIActivitySettings>): AIActivitySettings {
    validateSettingsUpdate(value);
    const next = this.store.update((current) => {
      const settings = { ...current.settings, ...value };
      const records =
        settings.contentPolicy === 'metadata-only'
          ? current.records.map((record) => ({
              ...record,
              invocation: {
                ...record.invocation,
                input: undefined,
                contentPolicy: 'metadata-only' as const,
              },
            }))
          : current.records;
      return { ...current, settings, records };
    }).settings;
    this.prune();
    return structuredClone(next);
  }

  clear(): void {
    this.store.update((current) => ({ ...current, records: [] }));
  }

  private prune(): void {
    const current = this.store.read();
    const retained = this.retained(current);
    if (retained.length !== current.records.length) {
      this.store.write({ ...current, records: retained });
    }
  }

  private retained(value: AIActivityData): AIActivityRecord[] {
    const days = value.settings.retentionDays;
    if (days === null) return value.records;
    const cutoff = this.now() - days * 24 * 60 * 60 * 1000;
    return value.records.filter((record) => record.createdAt >= cutoff);
  }
}

function defaultData(): AIActivityData {
  return {
    version: 1,
    settings: { keepActivity: true, contentPolicy: 'metadata-only', retentionDays: 30 },
    records: [],
  };
}

function validateSettingsUpdate(value: Partial<AIActivitySettings>): void {
  if (
    Object.keys(value).some(
      (key) => !['keepActivity', 'contentPolicy', 'retentionDays'].includes(key),
    ) ||
    (value.keepActivity !== undefined && typeof value.keepActivity !== 'boolean') ||
    (value.contentPolicy !== undefined &&
      !['full', 'metadata-only'].includes(value.contentPolicy)) ||
    (value.retentionDays !== undefined && ![7, 30, null].includes(value.retentionDays))
  ) {
    throw new Error('INVALID_AI_ACTIVITY_SETTINGS');
  }
}

function isAIActivityData(value: unknown): value is AIActivityData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    isSettings(value.settings) &&
    Array.isArray(value.records) &&
    value.records.length <= 10_000 &&
    value.records.every(isActivityRecord)
  );
}

function isSettings(value: unknown): value is AIActivitySettings {
  return (
    isRecord(value) &&
    typeof value.keepActivity === 'boolean' &&
    ['full', 'metadata-only'].includes(String(value.contentPolicy)) &&
    [7, 30, null].includes(value.retentionDays as 7 | 30 | null)
  );
}

function isActivityRecord(value: unknown): value is AIActivityRecord {
  return (
    isRecord(value) &&
    typeof value.id === 'string' &&
    typeof value.createdAt === 'number' &&
    Number.isFinite(value.createdAt) &&
    (value.completedAt === undefined ||
      (typeof value.completedAt === 'number' && Number.isFinite(value.completedAt))) &&
    typeof value.origin === 'string' &&
    typeof value.role === 'string' &&
    typeof value.sessionId === 'string' &&
    typeof value.jobId === 'string' &&
    isModel(value.requestedModel) &&
    isModel(value.resolvedModel) &&
    isResolution(value.resolution) &&
    isInvocation(value.invocation) &&
    isExecution(value.execution) &&
    (value.result === undefined || isResult(value.result)) &&
    (value.errorCode === undefined || typeof value.errorCode === 'string')
  );
}

function isModel(value: unknown): value is AIActivityRecord['resolvedModel'] {
  return (
    isRecord(value) &&
    value.provider === 'huggingface' &&
    typeof value.repository === 'string' &&
    typeof value.revision === 'string'
  );
}

function isResolution(value: unknown): value is AIActivityRecord['resolution'] {
  return (
    isRecord(value) &&
    typeof value.compatibility === 'string' &&
    typeof value.substituted === 'boolean' &&
    typeof value.overriddenByUser === 'boolean' &&
    typeof value.reason === 'string'
  );
}

function isInvocation(value: unknown): value is AIActivityRecord['invocation'] {
  return (
    isRecord(value) &&
    ['full', 'metadata-only'].includes(String(value.contentPolicy)) &&
    (value.input === undefined || isJSONValue(value.input, 0)) &&
    (value.artifacts === undefined ||
      (Array.isArray(value.artifacts) &&
        value.artifacts.length <= 32 &&
        value.artifacts.every(
          (artifact) =>
            isRecord(artifact) &&
            typeof artifact.artifactId === 'string' &&
            typeof artifact.mimeType === 'string' &&
            Number.isSafeInteger(artifact.sizeBytes) &&
            Number(artifact.sizeBytes) >= 0 &&
            (artifact.name === undefined || typeof artifact.name === 'string'),
        )))
  );
}

function isExecution(value: unknown): value is AIActivityRecord['execution'] {
  return (
    isRecord(value) &&
    typeof value.runner === 'string' &&
    typeof value.backend === 'string' &&
    typeof value.device === 'string' &&
    ['running', 'completed', 'failed', 'cancelled'].includes(String(value.status)) &&
    (value.durationMs === undefined ||
      (typeof value.durationMs === 'number' &&
        Number.isFinite(value.durationMs) &&
        value.durationMs >= 0))
  );
}

function isResult(value: unknown): value is NonNullable<AIActivityRecord['result']> {
  return (
    isRecord(value) &&
    ['structured', 'artifact', 'multi'].includes(String(value.kind)) &&
    (value.artifactIds === undefined ||
      (Array.isArray(value.artifactIds) &&
        value.artifactIds.length <= 64 &&
        value.artifactIds.every((id) => typeof id === 'string'))) &&
    (value.sizeBytes === undefined ||
      (Number.isSafeInteger(value.sizeBytes) && Number(value.sizeBytes) >= 0)) &&
    (value.chat === undefined ||
      (isRecord(value.chat) &&
        ['stop', 'length', 'unknown'].includes(String(value.chat.finishReason)) &&
        isRecord(value.chat.thinking) &&
        (value.chat.thinking.requested === null ||
          typeof value.chat.thinking.requested === 'boolean') &&
        typeof value.chat.thinking.applied === 'boolean' &&
        (value.chat.thinking.effective === null ||
          typeof value.chat.thinking.effective === 'boolean') &&
        [
          'applied',
          'already_fixed',
          'model_default',
          'unsupported_model',
          'fixed_mode',
          'unknown_support',
        ].includes(String(value.chat.thinking.reason))))
  );
}

function isJSONValue(value: unknown, depth: number): boolean {
  if (depth > 32) return false;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (Array.isArray(value)) {
    return value.length <= 10_000 && value.every((item) => isJSONValue(item, depth + 1));
  }
  if (!isRecord(value) || Object.keys(value).length > 1_000) return false;
  return Object.entries(value).every(
    ([key, item]) => key.length <= 256 && isJSONValue(item, depth + 1),
  );
}
