import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { AIActivityStore, type AIActivityRecord } from '../data/ai-activity-store';

function record(jobId: string): Omit<AIActivityRecord, 'id' | 'createdAt'> {
  const model = {
    provider: 'huggingface' as const,
    repository: 'openai/whisper-small',
    revision: 'a'.repeat(40),
  };
  return {
    origin: 'https://site.test',
    role: 'transcription',
    sessionId: 'session_one',
    jobId,
    requestedModel: model,
    resolvedModel: model,
    resolution: {
      compatibility: 'compatible',
      substituted: false,
      overriddenByUser: false,
      reason: 'preferred-model',
    },
    invocation: {
      input: { prompt: 'private prompt' },
      artifacts: [
        {
          artifactId: 'artifact_abcdefghijklmnopqrstuvwxyz',
          name: 'speech.wav',
          mimeType: 'audio/wav',
          sizeBytes: 42,
        },
      ],
      contentPolicy: 'full',
    },
    execution: {
      runner: 'transformers',
      backend: 'cpu',
      device: 'cpu',
      status: 'running',
    },
  };
}

describe('AIActivityStore', () => {
  it('defaults to metadata-only, records browser facts, and completes once', () => {
    const directory = mkdtempSync(join(tmpdir(), 'oryn-activity-'));
    let now = 1_000;
    const filePath = join(directory, 'activity.json');
    const store = new AIActivityStore(filePath, () => now);

    const id = store.create(record('job_one'));
    expect(id).toMatch(/^activity_/u);
    expect(store.list()[0]?.invocation.input).toBeUndefined();
    expect(store.list()[0]?.invocation.artifacts?.[0]).toMatchObject({ name: 'speech.wav' });

    now = 1_250;
    store.complete(id, {
      status: 'completed',
      completedAt: now,
      result: { kind: 'structured', sizeBytes: 18 },
    });
    expect(store.list()[0]).toMatchObject({
      completedAt: 1_250,
      execution: { status: 'completed', durationMs: 250 },
      result: { kind: 'structured', sizeBytes: 18 },
    });
    expect(new AIActivityStore(filePath, () => now).list()[0]?.jobId).toBe('job_one');
    rmSync(directory, { recursive: true });
  });

  it('supports full contents, immediate redaction, retention, disabling, and clear', () => {
    const directory = mkdtempSync(join(tmpdir(), 'oryn-activity-policy-'));
    let now = 10_000;
    const store = new AIActivityStore(join(directory, 'activity.json'), () => now);
    store.updateSettings({ contentPolicy: 'full', retentionDays: 7 });
    store.create(record('job_full'));
    expect(store.list()[0]?.invocation.input).toEqual({ prompt: 'private prompt' });

    store.updateSettings({ contentPolicy: 'metadata-only' });
    expect(store.list()[0]?.invocation.input).toBeUndefined();
    store.updateSettings({ keepActivity: false });
    expect(store.create(record('job_not_kept'))).toBeUndefined();
    expect(store.list()).toHaveLength(1);

    now += 8 * 24 * 60 * 60 * 1000;
    expect(store.list()).toHaveLength(0);
    store.updateSettings({ keepActivity: true, retentionDays: null });
    store.create(record('job_forever'));
    store.clear();
    expect(store.list()).toEqual([]);
    rmSync(directory, { recursive: true });
  });
});
