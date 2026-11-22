import imageReferences from './fixtures/image-references.json';
import { TASK_CONTRACTS } from '@ocno/oryn-protocol';
import { EventEmitter, once } from 'node:events';

import { describe, expect, it, vi } from 'vitest';
import type { EngineJobReference, EngineJobStatus, TaskProfile } from '@ocno/oryn-protocol';

import type { BrowserCallerContextService } from '../browser-caller-context';
import type { ArtifactManager } from '../artifact-manager/artifact-manager';
import type { BrowserModelSessionRegistry } from '../model-resolution/browser-model-sessions';
import type { AIActivityStore } from '../data/ai-activity-store';
import { BrowserJobRegistry } from '../jobs/browser-job-registry';

const owner = {
  origin: 'https://site.test',
  tabId: 'tab_one',
  webContentsId: 10,
  frameId: 1,
  frameUrl: 'https://site.test/app',
  documentId: 'document_one',
};

class FakeEngine extends EventEmitter {
  readonly engineInstanceId = 'engine_test';
  readonly createJob = vi.fn(
    async (
      jobId: string,
      sessionId: string,
      input: unknown,
      artifacts: unknown[],
    ): Promise<EngineJobReference> => {
      void input;
      void artifacts;
      return {
        jobId,
        sessionId,
        state: 'running' as const,
      };
    },
  );
  readonly cancelJob = vi.fn(async (jobId: string): Promise<EngineJobStatus> => ({
    jobId,
    sessionId: 'session_one',
    state: 'cancelled' as const,
    sequence: 2,
  }));
}

function setup(task = 'automatic-speech-recognition', taskProfile?: TaskProfile) {
  const engine = new FakeEngine();
  const release = vi.fn(async () => undefined);
  const acquireEngineInputLeases = vi.fn(() => ({
    bindings: [
      {
        artifactId: 'artifact_abcdefghijklmnopqrstuvwxyz',
        internalPath: '/controlled/input/object',
        mimeType: 'audio/wav',
        sizeBytes: 3,
        access: 'read' as const,
      },
    ],
    release,
  }));
  const publishEngineResult = vi.fn(
    async (_owner: unknown, _engineId: string, _jobId: string, value: unknown, preview = false) => {
      if (preview) {
        return {
          kind: 'artifact',
          artifact: {
            id: 'artifact_preview_abcdefghijklmnopqrstuvwxyz',
            mimeType: 'image/png',
            sizeBytes: 4,
            createdAt: 1,
          },
        };
      }
      void value;
      return {
        kind: 'artifact',
        artifact: {
          id: 'artifact_result_abcdefghijklmnopqrstuvwxyz',
          mimeType: 'video/mp4',
          sizeBytes: 64,
          createdAt: 2,
        },
      };
    },
  );
  const cleanupEngineJob = vi.fn();
  const sessions = {
    requireOwned: (sessionId: string) => ({
      sessionId,
      role: 'transcription',
      origin: owner.origin,
      tabId: owner.tabId,
      webContentsId: owner.webContentsId,
      documentId: owner.documentId,
      resolution: {
        preferredModel: {
          provider: 'huggingface',
          repository: 'openai/whisper-small',
          revision: 'a'.repeat(40),
        },
        resolvedModel: {
          provider: 'huggingface',
          repository: 'openai/whisper-small',
          revision: 'a'.repeat(40),
        },
        compatibility: 'compatible',
        executability: 'executable',
        security: 'allowed',
        developerCertified: false,
        substituted: false,
        overriddenByUser: false,
        resolutionReason: 'preferred-model',
        compatibilityReasons: [],
        downloadRequired: false,
      },
      runtime: { runner: 'test', backend: 'cpu', deviceClass: 'cpu' },
      capabilities: { task },
      taskProfile,
      request: {},
    }),
  };
  const callerContexts = {
    hasDocumentAuthority: (_tabId: string, _webContentsId: number, documentId: string) =>
      documentId.startsWith('document_'),
  };
  const activity = {
    create: vi.fn(() => 'activity_abcdefghijklmnopqrstuvwxyz'),
    complete: vi.fn(),
  };
  const registry = new BrowserJobRegistry(
    engine,
    sessions as unknown as BrowserModelSessionRegistry,
    {
      acquireEngineInputLeases,
      publishEngineResult,
      cleanupEngineJob,
      cancelPublication: vi.fn(),
    } as unknown as ArtifactManager,
    callerContexts as unknown as BrowserCallerContextService,
    activity as unknown as AIActivityStore,
  );
  return {
    engine,
    registry,
    release,
    acquireEngineInputLeases,
    publishEngineResult,
    cleanupEngineJob,
    activity,
  };
}

describe('BrowserJobRegistry', () => {
  it('uses the negotiated pose contract for broad keypoint requests before leasing artifacts', async () => {
    const profile: TaskProfile = {
      kind: 'task-profile',
      schemaVersion: 1,
      profileId: 'pose-fixture',
      contract: { id: 'pose-estimation-v1', revision: 1 },
      task: 'pose-estimation',
      inputModalities: ['image'],
      outputModalities: ['structured'],
      inputSchemaRef: 'taskInput.pose-estimation-v1',
      resultSchemaRef: 'taskResult.pose-estimation-v1',
      features: {},
      parameters: {},
      limits: {},
      streamingChannels: [],
      extensions: {},
      provenance: 'adapter-derived',
    };
    const { registry, engine, acquireEngineInputLeases } = setup('keypoint-detection', profile);
    const id = 'artifact_abcdefghijklmnopqrstuvwxyz';
    await expect(
      registry.invoke(owner, { sessionId: 'session_one', input: { imageArtifactId: id } }),
    ).rejects.toMatchObject({ code: 'JOB_INPUT_INVALID' });
    expect(acquireEngineInputLeases).not.toHaveBeenCalled();
    await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { imageArtifactId: id, boxes: [[0, 0, 1, 1]] },
    });
    expect(engine.createJob).toHaveBeenCalledOnce();
    registry.dispose();
  });

  it('leases reference audio and video masks through the same artifact authority', async () => {
    const { registry, acquireEngineInputLeases } = setup('audio-to-audio');
    const id = 'artifact_abcdefghijklmnopqrstuvwxyz';
    const reference = 'artifact_reference_abcdefghijklmnopqrstuvwxyz';
    await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { audioArtifactId: id, referenceAudioArtifactId: reference },
    });
    expect(acquireEngineInputLeases).toHaveBeenLastCalledWith(owner, [id, reference]);
    await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { videoArtifactId: id, maskVideoArtifactId: reference },
    });
    expect(acquireEngineInputLeases).toHaveBeenLastCalledWith(owner, [id, reference]);
    registry.dispose();
  });
  it('rejects forbidden new-task fields before acquiring image leases', async () => {
    const { registry, acquireEngineInputLeases, engine } = setup('image-feature-extraction');
    await expect(
      registry.invoke(owner, {
        sessionId: 'session_one',
        input: { imageArtifactId: 'artifact_abcdefghijklmnopqrstuvwxyz', backend: 'mps' },
      }),
    ).rejects.toMatchObject({ code: 'JOB_INPUT_INVALID' });
    expect(acquireEngineInputLeases).not.toHaveBeenCalled();
    expect(engine.createJob).not.toHaveBeenCalled();
    registry.dispose();
  });

  it('bounds live work submitted by one origin', async () => {
    const { registry } = setup();
    for (let index = 0; index < 16; index += 1) {
      await registry.invoke(owner, {
        sessionId: 'session_one',
        input: { prompt: `job ${index}` },
      });
    }
    await expect(
      registry.invoke(owner, { sessionId: 'session_one', input: { prompt: 'one too many' } }),
    ).rejects.toMatchObject({ code: 'JOB_QUEUE_FULL' });
    registry.dispose();
  });

  it('registers ownership before routing explicit artifact bindings', async () => {
    const { engine, registry, acquireEngineInputLeases, activity } = setup();
    const reference = await registry.invoke(owner, {
      sessionId: 'session_one',
      input: {
        audioArtifactId: 'artifact_abcdefghijklmnopqrstuvwxyz',
        note: { arbitrary: 'artifact_not_scanned_abcdefghijklmnopqrstuvwxyz' },
      },
    });

    expect(reference.state).toBe('running');
    expect(acquireEngineInputLeases).toHaveBeenCalledWith(owner, [
      'artifact_abcdefghijklmnopqrstuvwxyz',
    ]);
    expect(engine.createJob).toHaveBeenCalledWith(
      reference.jobId,
      'session_one',
      expect.anything(),
      expect.arrayContaining([expect.objectContaining({ access: 'read' })]),
    );
    expect(activity.create).toHaveBeenCalledWith(
      expect.objectContaining({
        origin: owner.origin,
        role: 'transcription',
        invocation: expect.objectContaining({
          input: expect.objectContaining({
            audioArtifactId: 'artifact_abcdefghijklmnopqrstuvwxyz',
          }),
          artifacts: [expect.objectContaining({ mimeType: 'audio/wav', sizeBytes: 3 })],
        }),
      }),
    );
    expect(activity.create.mock.invocationCallOrder[0]).toBeLessThan(
      engine.createJob.mock.invocationCallOrder[0] ?? Infinity,
    );
    registry.dispose();
  });

  it('enforces exact owner cancellation and releases leases once at terminal state', async () => {
    const { engine, registry, release, activity } = setup();
    const reference = await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { audioArtifactId: 'artifact_abcdefghijklmnopqrstuvwxyz' },
    });
    await expect(
      registry.cancelOwned(reference.jobId, { ...owner, documentId: 'document_two' }),
    ).rejects.toMatchObject({ code: 'JOB_NOT_OWNED' });

    const websiteEvent = once(registry, 'website-event');
    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.completed',
      seq: 10,
      timestamp: 123,
      data: {
        jobId: reference.jobId,
        sessionId: 'session_one',
        sequence: 1,
        result: {
          kind: 'structured',
          data: { text: 'done', internalPath: '/private/result' },
        },
      },
    });
    const [forwarded] = await websiteEvent;
    expect(forwarded.event.data).not.toHaveProperty('internalPath');
    expect(JSON.stringify(forwarded)).not.toContain('/private/result');
    expect(release).toHaveBeenCalledOnce();
    expect(activity.complete).toHaveBeenCalledWith(
      'activity_abcdefghijklmnopqrstuvwxyz',
      expect.objectContaining({ status: 'completed', completedAt: 123 }),
    );

    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.failed',
      seq: 11,
      timestamp: 124,
      data: {
        jobId: reference.jobId,
        sessionId: 'session_one',
        sequence: 2,
        error: { code: 'RAW', category: 'job', message: '/Users/private', retryable: false },
      },
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(registry.snapshot()[0]?.state).toBe('completed');
    expect(release).toHaveBeenCalledOnce();
    expect(activity.complete).toHaveBeenCalledOnce();
    registry.dispose();
  });

  it('turns safe runtime failure categories into actionable public messages', async () => {
    const { engine, registry } = setup();
    const reference = await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { prompt: 'large local image' },
    });
    const websiteEvent = once(registry, 'website-event');

    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.failed',
      seq: 10,
      timestamp: 123,
      data: {
        jobId: reference.jobId,
        sessionId: 'session_one',
        sequence: 1,
        error: {
          code: 'MODEL_MEMORY_INSUFFICIENT',
          category: 'job',
          message: '/Users/private/raw-worker-message',
        },
      },
    });

    const [forwarded] = await websiteEvent;
    expect(forwarded.event.data.error).toEqual({
      code: 'MODEL_MEMORY_INSUFFICIENT',
      message:
        'This model is too large for the memory currently available. Quit memory-heavy apps or use a smaller or quantized variant, then try again.',
    });
    expect(JSON.stringify(forwarded)).not.toContain('/Users/private');
    registry.dispose();
  });

  it('exposes an actionable language-boundary error for timestamp alignment', async () => {
    const { engine, registry } = setup();
    const reference = await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { transcript: 'local words' },
    });
    const websiteEvent = once(registry, 'website-event');

    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.failed',
      seq: 10,
      timestamp: 123,
      data: {
        jobId: reference.jobId,
        sessionId: 'session_one',
        sequence: 1,
        error: {
          code: 'ALIGNMENT_LANGUAGE_UNSUPPORTED',
          category: 'job',
          message: '/Users/private/raw-worker-message',
        },
      },
    });

    const [forwarded] = await websiteEvent;
    expect(forwarded.event.data.error).toEqual({
      code: 'ALIGNMENT_LANGUAGE_UNSUPPORTED',
      message:
        'The selected timestamp aligner does not support the detected language. Use a timing-capable fallback model or estimated timing.',
    });
    expect(JSON.stringify(forwarded)).not.toContain('/Users/private');
    registry.dispose();
  });

  it('rejects execution hooks and oversized/non-structured invocation input', async () => {
    const { registry } = setup();
    await expect(
      registry.invoke(owner, {
        sessionId: 'session_one',
        input: { shellCommand: 'whoami' },
      }),
    ).rejects.toMatchObject({ code: 'JOB_INPUT_INVALID' });
    await expect(
      registry.invoke(owner, {
        sessionId: 'session_one',
        input: { bytes: new Uint8Array(3) },
      }),
    ).rejects.toMatchObject({ code: 'JOB_INPUT_INVALID' });
    registry.dispose();
  });

  it('adopts previews and final outputs before forwarding public artifact metadata', async () => {
    const { engine, registry, publishEngineResult } = setup();
    const reference = await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { prompt: 'generate media' },
    });
    const previewEvent = once(registry, 'website-event');
    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.preview',
      seq: 20,
      timestamp: 200,
      data: {
        jobId: reference.jobId,
        sessionId: 'session_one',
        sequence: 1,
        output: {
          outputId: 'output_preview_123',
          internalPath: '/controlled/staging/preview.png',
          mimeType: 'image/png',
          sizeBytes: 4,
        },
      },
    });
    const [preview] = await previewEvent;
    expect(preview.event.data.output).toMatchObject({
      id: 'artifact_preview_abcdefghijklmnopqrstuvwxyz',
    });
    expect(JSON.stringify(preview)).not.toContain('/controlled/staging');

    const completedEvent = once(registry, 'website-event');
    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.completed',
      seq: 21,
      timestamp: 201,
      data: {
        jobId: reference.jobId,
        sessionId: 'session_one',
        sequence: 2,
        result: {
          kind: 'artifact',
          artifact: {
            outputId: 'output_video_123',
            internalPath: '/controlled/staging/video.mp4',
            mimeType: 'video/mp4',
            sizeBytes: 64,
          },
        },
      },
    });
    const [completed] = await completedEvent;
    expect(completed.event.data.result.artifact).toMatchObject({
      id: 'artifact_result_abcdefghijklmnopqrstuvwxyz',
    });
    expect(JSON.stringify(completed)).not.toContain('/controlled/staging');
    expect(publishEngineResult).toHaveBeenNthCalledWith(
      1,
      owner,
      'engine_test',
      reference.jobId,
      expect.anything(),
      true,
    );
    expect(publishEngineResult).toHaveBeenCalledTimes(2);
    registry.dispose();
  });

  it('gives trusted UI cancellation authority without website cooperation', async () => {
    const { engine, registry } = setup();
    const first = await registry.invoke(owner, {
      sessionId: 'session_one',
      input: { prompt: 'keep running' },
    });
    await registry.cancelTrusted(first.jobId);
    expect(engine.cancelJob).toHaveBeenCalledWith(first.jobId, 'user');
    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.cancelled',
      seq: 3,
      timestamp: 124,
      data: {
        jobId: first.jobId,
        sessionId: 'session_one',
        sequence: 1,
        reason: 'user',
      },
    });
    await new Promise((resolve) => setImmediate(resolve));

    const second = await registry.invoke(owner, {
      sessionId: 'session_two',
      input: { prompt: 'site work' },
    });
    expect(registry.activitySnapshot().jobs).toEqual([
      expect.objectContaining({ jobId: second.jobId, origin: owner.origin, role: 'transcription' }),
    ]);
    await registry.cancelAllForOrigin(owner.origin);
    expect(engine.cancelJob).toHaveBeenCalledWith(second.jobId, 'user');
    registry.dispose();
  });

  it('shows cross-origin queued state only through trusted aggregate records', async () => {
    const { engine, registry } = setup();
    const otherOwner = {
      ...owner,
      origin: 'https://other.test',
      tabId: 'tab_two',
      webContentsId: 11,
      frameUrl: 'https://other.test/app',
      documentId: 'document_two',
    };
    await registry.invoke(owner, { sessionId: 'session_one', input: { prompt: 'first' } });
    engine.createJob.mockImplementationOnce(async (jobId: string, sessionId: string) => ({
      jobId,
      sessionId,
      state: 'queued' as const,
      queue: {
        lane: 'generation' as const,
        position: 1,
        jobsAhead: 0,
        reason: 'generation-slot' as const,
      },
    }));
    const second = await registry.invoke(otherOwner, {
      sessionId: 'session_two',
      input: { prompt: 'second origin' },
    });
    const trusted = registry.activitySnapshot();
    expect(trusted.jobs.map((job) => job.origin)).toEqual([
      'https://site.test',
      'https://other.test',
    ]);
    expect(trusted.jobs[1]).toMatchObject({
      state: 'queued',
      resolvedModel: 'openai/whisper-small',
    });

    const forwarded = once(registry, 'website-event');
    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.queue.updated',
      seq: 1,
      timestamp: 123,
      data: {
        jobId: second.jobId,
        sessionId: 'session_two',
        sequence: 1,
        lane: 'generation',
        position: 1,
        jobsAhead: 0,
        reason: 'generation-slot',
      },
    });
    const [website] = await forwarded;
    expect(website.owner.origin).toBe('https://other.test');
    expect(JSON.stringify(website)).not.toContain('https://site.test');
    registry.dispose();
  });

  it('fails running and queued jobs once on engine crash without replaying them', async () => {
    const { engine, registry, activity } = setup();
    await registry.invoke(owner, { sessionId: 'session_one', input: { prompt: 'running' } });
    engine.createJob.mockImplementationOnce(async (jobId: string, sessionId: string) => ({
      jobId,
      sessionId,
      state: 'queued' as const,
      queue: {
        lane: 'generation' as const,
        position: 1,
        jobsAhead: 0,
        reason: 'generation-slot' as const,
      },
    }));
    await registry.invoke(
      { ...owner, origin: 'https://other.test' },
      { sessionId: 'session_two', input: { prompt: 'queued' } },
    );
    const createsBeforeCrash = engine.createJob.mock.calls.length;
    const forwarded: Array<{ event: { event: string } }> = [];
    registry.on('website-event', (value) => forwarded.push(value));

    engine.emit('state-changed', 'error');
    await new Promise((resolve) => setImmediate(resolve));
    expect(registry.snapshot().map((record) => record.state)).toEqual(['failed', 'failed']);
    expect(activity.complete).toHaveBeenCalledTimes(2);
    expect(forwarded.map((value) => value.event.event)).toEqual(['job.failed', 'job.failed']);

    engine.emit('state-changed', 'ready');
    await new Promise((resolve) => setImmediate(resolve));
    expect(engine.createJob).toHaveBeenCalledTimes(createsBeforeCrash);
    registry.dispose();
  });
});

describe('ordered editing references', () => {
  it('leases every reference for the same owner and retains input order', async () => {
    const { registry, engine, acquireEngineInputLeases, release } = setup(
      'image-to-image',
      imageReferences.optional as unknown as TaskProfile,
    );
    const ids = [...'abc'].map((c) => 'artifact_' + c.repeat(24));
    const input = {
      imageArtifactId: ids[0],
      referenceImageArtifactIds: [ids[2], ids[1]],
      prompt: 'Use all references',
    };
    await registry.invoke(owner, { sessionId: 'session_one', input });
    expect(acquireEngineInputLeases).toHaveBeenCalledWith(owner, [ids[0], ids[2], ids[1]]);
    expect(engine.createJob.mock.calls[0]![2]).toMatchObject(input);
    await registry.cancelAllTrusted();
    engine.emit('job-event', {
      v: 1,
      type: 'event',
      event: 'job.cancelled',
      seq: 3,
      timestamp: 124,
      data: {
        jobId: engine.createJob.mock.calls[0]![0],
        sessionId: 'session_one',
        sequence: 1,
        reason: 'user',
      },
    });
    await new Promise((resolve) => setImmediate(resolve));
    expect(release).toHaveBeenCalledOnce();
    registry.dispose();
  });
  it('rejects missing, duplicate, excessive references before acquiring leases', async () => {
    const { registry, engine, acquireEngineInputLeases } = setup(
      'image-to-image',
      imageReferences.pair as unknown as TaskProfile,
    );
    const ids = [...'abc'].map((c) => 'artifact_' + c.repeat(24));
    for (const refs of [undefined, [], [ids[0]], [ids[1], ids[2]]]) {
      await expect(
        registry.invoke(owner, {
          sessionId: 'session_one',
          input: {
            imageArtifactId: ids[0],
            prompt: 'Edit',
            ...(refs ? { referenceImageArtifactIds: refs } : {}),
          },
        }),
      ).rejects.toMatchObject({ code: 'JOB_INPUT_INVALID' });
    }
    expect(acquireEngineInputLeases).not.toHaveBeenCalled();
    expect(engine.createJob).not.toHaveBeenCalled();
    registry.dispose();
  });
});

it('waits for publication and never resurrects a job after Engine exit', async () => {
  const { engine, registry, publishEngineResult } = setup();
  const copied = {
    kind: 'artifact',
    artifact: {
      id: 'artifact_result_abcdefghijklmnopqrstuvwxyz',
      mimeType: 'video/mp4',
      sizeBytes: 4,
      createdAt: 1,
    },
  };
  let complete!: (value: typeof copied) => void;
  publishEngineResult.mockReturnValue(
    new Promise((resolve) => {
      complete = resolve;
    }),
  );
  const reference = await registry.invoke(owner, { sessionId: 'session_one', input: {} });
  const events: unknown[] = [];
  registry.on('website-event', (event) => events.push(event));
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.completed',
    seq: 1,
    timestamp: 1,
    data: {
      jobId: reference.jobId,
      sessionId: 'session_one',
      sequence: 1,
      result: {
        kind: 'artifact',
        artifact: {
          outputId: 'output_result_123',
          internalPath: '/controlled/staging/result',
          mimeType: 'video/mp4',
          sizeBytes: 4,
        },
      },
    },
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(events).toEqual([]);
  engine.emit('state-changed', 'error');
  complete(copied);
  await new Promise<void>((resolve) => setImmediate(resolve));
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(events).toHaveLength(1);
  expect(events[0]).toMatchObject({ event: { event: 'job.failed' } });
  expect(registry.snapshot()[0]?.state).toBe('failed');
  registry.dispose();
});

it('keeps cancellation terminal when a delayed cancellation reply arrives after publication abort', async () => {
  const { engine, registry, publishEngineResult } = setup();
  let rejectCopy!: (error: Error) => void;
  publishEngineResult.mockReturnValue(
    new Promise((_resolve, reject) => {
      rejectCopy = reject;
    }),
  );
  let reply!: (value: EngineJobStatus) => void;
  engine.cancelJob.mockReturnValue(
    new Promise((resolve) => {
      reply = resolve;
    }),
  );
  const reference = await registry.invoke(owner, { sessionId: 'session_one', input: {} });
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.completed',
    seq: 1,
    timestamp: 1,
    data: {
      jobId: reference.jobId,
      sessionId: 'session_one',
      sequence: 1,
      result: {
        kind: 'artifact',
        artifact: {
          outputId: 'output_result_123',
          internalPath: '/controlled/staging/result',
          mimeType: 'video/mp4',
          sizeBytes: 4,
        },
      },
    },
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  const cancelledEvent = once(registry, 'website-event');
  const cancel = registry.cancelOwned(reference.jobId, owner);
  rejectCopy(new Error('publication aborted'));
  const [event] = await cancelledEvent;
  expect(event.event.event).toBe('job.cancelled');
  reply({ jobId: reference.jobId, sessionId: 'session_one', state: 'queued', sequence: 1 });
  await cancel;
  expect(registry.snapshot()[0]?.state).toBe('cancelled');
  registry.dispose();
});

function thinkingProfile(): TaskProfile {
  const contract = TASK_CONTRACTS['chat-v2']!;
  return {
    kind: 'task-profile',
    schemaVersion: 1,
    profileId: 'chat-v2-unsupported-thinking',
    contract: { id: 'chat-v2', revision: 1 },
    task: 'text-generation',
    inputModalities: ['text'],
    outputModalities: ['text'],
    inputSchemaRef: 'taskInput.chat-v2',
    resultSchemaRef: 'taskResult.chat-v2',
    features: {},
    parameters: Object.fromEntries(
      contract.parameterNames.map((key) => {
        const schema = { ...contract.inputSchema.properties[key] };
        delete schema.default;
        return [
          key,
          {
            inputPointer: `/${key}`,
            valueSchema: schema,
            handling: key === 'thinking' ? 'unsupported' : 'honored',
            editable: true,
            affects: ['result'],
          },
        ];
      }),
    ),
    limits: {},
    streamingChannels: ['text', 'reasoning'],
    extensions: {},
    provenance: 'adapter-derived',
  };
}

function thinkingResult(requested = false) {
  return {
    kind: 'structured',
    value: {
      text: 'Final answer.',
      reasoning: 'Private reasoning.',
      finishReason: 'stop',
      usage: { inputTokens: null, outputTokens: null, reasoningTokens: null, answerTokens: null },
      metadata: {
        thinking: {
          requested,
          applied: false,
          effective: null,
          reason: 'unsupported_model',
          source: 'engine',
        },
      },
    },
  };
}

it('carries an unsupported thinking preference through the real broker validators and preserves response metadata', async () => {
  const { registry, engine, activity } = setup('text-generation', thinkingProfile());
  const reference = await registry.invoke(owner, {
    sessionId: 'session_one',
    input: {
      messages: [{ role: 'user', content: 'Explain.' }],
      thinking: false,
      maxTokens: 4096,
    },
  });
  expect(engine.createJob.mock.calls[0]?.[2]).toMatchObject({ thinking: false, maxTokens: 4096 });
  const result = thinkingResult();
  const delivered = once(registry, 'website-event');
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.completed',
    timestamp: 1,
    data: { jobId: reference.jobId, sessionId: 'session_one', sequence: 1, result },
  });
  const [event] = await delivered;
  expect(event.owner).toEqual(owner);
  expect(event.event.data.result).toEqual(result);
  expect(activity.complete).toHaveBeenCalledWith(
    expect.any(String),
    expect.objectContaining({
      result: expect.objectContaining({
        chat: {
          finishReason: 'stop',
          thinking: {
            requested: false,
            applied: false,
            effective: null,
            reason: 'unsupported_model',
          },
        },
      }),
    }),
  );
  expect(JSON.stringify(activity.complete.mock.calls)).not.toContain('Private reasoning.');
  registry.dispose();
});

it('rejects chat metadata for a different thinking request before delivery', async () => {
  const { registry, engine } = setup('text-generation', thinkingProfile());
  const reference = await registry.invoke(owner, {
    sessionId: 'session_one',
    input: {
      messages: [{ role: 'user', content: 'Explain.' }],
      thinking: false,
    },
  });
  const delivered = once(registry, 'website-event');
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.completed',
    timestamp: 1,
    data: {
      jobId: reference.jobId,
      sessionId: 'session_one',
      sequence: 1,
      result: thinkingResult(true),
    },
  });
  const [event] = await delivered;
  expect(event.event.event).toBe('job.failed');
  expect(JSON.stringify(event)).not.toContain('Private reasoning.');
  expect(engine.cancelJob).toHaveBeenCalledWith(reference.jobId, 'resource-policy');
  registry.dispose();
});

it('keeps reasoning as an owned channel and suppresses late deltas after cancellation', async () => {
  const { registry, engine } = setup('text-generation', thinkingProfile());
  const reference = await registry.invoke(owner, {
    sessionId: 'session_one',
    input: {
      messages: [{ role: 'user', content: 'Explain.' }],
      thinking: false,
    },
  });
  const delivered = once(registry, 'website-event');
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.output.delta',
    timestamp: 1,
    data: {
      jobId: reference.jobId,
      sessionId: 'session_one',
      sequence: 1,
      channel: 'reasoning',
      delta: 'Reasoning.',
    },
  });
  const [event] = await delivered;
  expect(event.owner).toEqual(owner);
  expect(event.event.data).toEqual({ channel: 'reasoning', delta: 'Reasoning.' });
  const late: unknown[] = [];
  registry.on('website-event', (value) => late.push(value));
  await registry.cancelOwned(reference.jobId, owner);
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.output.delta',
    timestamp: 2,
    data: {
      jobId: reference.jobId,
      sessionId: 'session_one',
      sequence: 2,
      channel: 'text',
      delta: 'Late answer.',
    },
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(late).toEqual([]);
  registry.dispose();
});

it('rejects a completed result that disagrees with the delivered reasoning stream', async () => {
  const { registry, engine } = setup('text-generation', thinkingProfile());
  const reference = await registry.invoke(owner, {
    sessionId: 'session_one',
    input: {
      messages: [{ role: 'user', content: 'Explain.' }],
      thinking: false,
    },
  });
  const delta = once(registry, 'website-event');
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.output.delta',
    timestamp: 1,
    data: {
      jobId: reference.jobId,
      sessionId: 'session_one',
      sequence: 1,
      channel: 'reasoning',
      delta: 'A different reasoning stream.',
    },
  });
  await delta;
  const final = once(registry, 'website-event');
  engine.emit('job-event', {
    v: 1,
    type: 'event',
    event: 'job.completed',
    timestamp: 2,
    data: {
      jobId: reference.jobId,
      sessionId: 'session_one',
      sequence: 2,
      result: thinkingResult(),
    },
  });
  const [event] = await final;
  expect(event.event.event).toBe('job.failed');
  expect(event.event.data.result).toBeUndefined();
  registry.dispose();
});
