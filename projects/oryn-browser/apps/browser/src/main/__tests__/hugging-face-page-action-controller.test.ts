import { EventEmitter } from 'node:events';

import { assertCandidateExecutionProfile } from '@ocno/oryn-protocol';
import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  ResolvedModelRef,
  RuntimeResourceSnapshot,
} from '@ocno/oryn-protocol';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  HuggingFacePageActionController,
  type PageActionTabIdentity,
} from '../model-page/hugging-face-page-action-controller';
import type { TrustedModelOperation } from '../../shared/ipc';
import turboMemoryBlocked from './fixtures/qwen-image-2.1-turbo-memory-blocked.json';
import unverifiedChatProfiles from './fixtures/unverified-chat-profiles.json';

function profile(repository = 'google/gemma-3-1b-it'): CandidateExecutionProfile {
  const revision = 'a'.repeat(40);
  return {
    model: { provider: 'huggingface', repository, revision },
    exactRevision: revision,
    executable: true,
    runner: 'runtime-pack',
    runtimePack: {
      packId: 'oryn.chat',
      packVersion: '1.0.0',
      runnerId: 'chat',
      taskContract: 'chat-v1',
      compatibilityLevel: 'native-declarative',
      manifestDigest: 'b'.repeat(64),
      trustedMetadata: {
        license: 'Apache-2.0',
        modelFile: 'model.safetensors',
        conformance: { status: 'passed', platforms: ['macos-arm64'] },
      },
    },
    executableBackends: ['mps'],
    requiresCustomCode: false,
    estimatedDiskBytes: 1_000,
    estimatedMemoryBytes: 2_000,
    technicalCapabilities: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true },
    },
  };
}

class FakeEngine extends EventEmitter {
  inspection = profile();
  resources: RuntimeResourceSnapshot = {
    snapshotId: 'resources_test',
    timestamp: 0,
    memoryPressure: 'normal',
    systemMemory: { availableBytes: 1_500 },
    localAI: { estimatedResidentModelBytes: 0 },
    generationLane: { capacity: 1, running: 0, queued: 0 },
  };
  inventory: EngineModelInventoryRecord[] = [];
  installed?: { operationId: string; model: ResolvedModelRef };
  cancelled?: string;
  async inspectModel() {
    return structuredClone(this.inspection);
  }
  async getModelInventory() {
    return { models: structuredClone(this.inventory) };
  }
  async getRuntimeResources() {
    return structuredClone(this.resources);
  }
  async installModel(operationId: string, model: ResolvedModelRef) {
    this.installed = { operationId, model };
  }
  async cancelModelInstall(operationId: string) {
    this.cancelled = operationId;
    return true;
  }
}

afterEach(() => vi.useRealTimers());

describe('HuggingFacePageActionController', () => {
  it('binds source inspection to the active HF document and shows reviewed-code risks', async () => {
    vi.useFakeTimers();
    const sourceProfile = structuredClone(unverifiedChatProfiles[0]) as CandidateExecutionProfile;
    sourceProfile.sourceRecognition = {
      state: 'recognized',
      match: 'structural',
      profileId: 'oryn.lfm3.source-a.v1',
      representation: 'bf16-cpu',
      admitted: true,
    };
    sourceProfile.admissionScope = 'huggingface-page';
    const inspect = vi.fn(async () => sourceProfile);
    const engine = Object.assign(new FakeEngine(), { inspectHuggingFaceModel: inspect });
    engine.inspection = {
      ...structuredClone(sourceProfile),
      sourceRecognition: undefined,
      admissionScope: undefined,
      executable: false,
      executableBackends: [],
      requiresCustomCode: true,
      blockingReasons: ['CUSTOM_CODE_REQUIRED'],
    };
    let active: PageActionTabIdentity = {
      tabId: 'tab_hf',
      documentId: 'doc_current',
      url: `https://huggingface.co/${sourceProfile.model.repository}`,
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(300);
    expect(inspect).not.toHaveBeenCalled();
    controller.retry();
    await vi.advanceTimersByTimeAsync(0);
    expect(inspect).toHaveBeenCalledWith(
      expect.objectContaining({ repository: sourceProfile.model.repository }),
      { sourceURL: active.url, tabId: active.tabId, documentId: active.documentId },
      expect.any(AbortSignal),
    );
    expect(controller.snapshot()?.implementationNotice).toContain('Repository Python is never run');
    expect(controller.snapshot()?.implementationDetails).toContain('Structural source match');
    active = { ...active, documentId: 'doc_other', url: 'https://example.com/' };
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(300);
    expect(controller.snapshot()).toBeUndefined();
    expect(inspect).toHaveBeenCalledTimes(1);
    controller.destroy();
  });

  it('aborts in-flight source inspection on document navigation and discards its late result', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile('LiquidAI/LFM3-1.7B'),
      executable: false,
      executableBackends: [],
      requiresCustomCode: true,
      blockingReasons: ['CUSTOM_CODE_REQUIRED'],
    };
    let finish!: (value: CandidateExecutionProfile) => void;
    let signal: AbortSignal | undefined;
    const inspect = vi.fn((...args: unknown[]) => {
      signal = args[2] as AbortSignal;
      return new Promise<CandidateExecutionProfile>((resolve) => {
        finish = resolve;
      });
    });
    const sourceEngine = Object.assign(engine, { inspectHuggingFaceModel: inspect });
    let active = {
      tabId: 'tab_hf',
      documentId: 'doc_hf',
      url: 'https://huggingface.co/LiquidAI/LFM3-1.7B',
    };
    const controller = new HuggingFacePageActionController(sourceEngine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(300);
    controller.retry();
    await vi.advanceTimersByTimeAsync(0);
    expect(signal?.aborted).toBe(false);
    active = { ...active, documentId: 'doc_other', url: 'https://example.com/' };
    controller.activeTabChanged();
    expect(signal?.aborted).toBe(true);
    finish(profile('LiquidAI/LFM3-1.7B'));
    await vi.advanceTimersByTimeAsync(0);
    expect(controller.snapshot()).toBeUndefined();
    controller.destroy();
  });

  it('distinguishes recognized source from unsupported quantization', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile('LiquidAI/LFM3-1.7B-MLX-4bit'),
      executable: false,
      executableBackends: [],
      sourceRecognition: {
        state: 'recognized',
        match: 'exact',
        representation: 'mlx-affine4-unsupported',
        admitted: false,
      },
      blockingReasons: ['REVIEWED_REPRESENTATION_UNSUPPORTED'],
    };
    const controller = new HuggingFacePageActionController(engine, () => ({
      tabId: 'tab_hf',
      documentId: 'doc_hf',
      url: 'https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit',
    }));
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(300);
    expect(controller.snapshot()?.status).toBe('unsupported');
    expect(controller.snapshot()?.message).toContain('weight format');
    controller.destroy();
  });

  it.each(unverifiedChatProfiles.map((profile) => [profile.model.repository, profile] as const))(
    'offers explicit unverified trust for the effective task profile of %s',
    async (repository, captured) => {
      vi.useFakeTimers();
      const engine = new FakeEngine();
      const inspection: unknown = structuredClone(captured);
      assertCandidateExecutionProfile(inspection);
      engine.inspection = inspection;
      const controller = new HuggingFacePageActionController(engine, () => ({
        tabId: 'unverified',
        documentId: 'unverified_document',
        url: `https://huggingface.co/${repository}`,
      }));
      try {
        controller.activeTabChanged();
        await vi.advanceTimersByTimeAsync(250);
        expect(controller.snapshot()).toMatchObject({
          status: 'unverified',
          trustMode: 'user-trusted',
          taskContract: 'chat-v1',
          exactRevision: inspection.exactRevision,
          blockingReasons: ['CONFORMANCE_REQUIRED'],
        });
        expect(engine.installed).toBeUndefined();
        await expect(controller.confirmInstall('oryn-verified')).rejects.toThrow('NOT_AVAILABLE');
        expect(engine.installed).toBeUndefined();
        await controller.confirmInstall('user-trusted');
        expect(engine.installed?.model).toEqual(inspection.model);
        expect(controller.snapshot()).toMatchObject({
          status: 'installed',
          trustMode: 'user-trusted',
        });
      } finally {
        controller.destroy();
      }
    },
  );

  it.each(['inspection', 'error'] as const)(
    'presents a generic resource failure without assuming it is memory (%s)',
    async (source) => {
      vi.useFakeTimers();
      const engine = new FakeEngine();
      if (source === 'inspection') {
        engine.inspection = {
          ...profile(),
          executable: false,
          executableBackends: [],
          blockingReasons: ['CONFORMANCE_REQUIRED', 'RUNTIME_RESOURCES_INSUFFICIENT'],
        };
      } else {
        engine.inspectModel = async () => {
          throw new Error('RUNTIME_RESOURCES_INSUFFICIENT');
        };
      }
      const controller = new HuggingFacePageActionController(engine, () => ({
        tabId: 'one',
        documentId: 'doc_one',
        url: 'https://huggingface.co/google/gemma-3-1b-it',
      }));
      controller.activeTabChanged();
      await vi.advanceTimersByTimeAsync(250);
      expect(controller.snapshot()).toMatchObject({
        status: 'resource-error',
        message:
          'The model exceeds the currently available runtime resources. Free resources, then retry.',
      });
      controller.destroy();
    },
  );

  it.each([false, true])(
    'shows the captured Turbo memory failure and recovers after reinspection (reverse reasons: %s)',
    async (reverseReasons) => {
      vi.useFakeTimers();
      const engine = new FakeEngine();
      const captured: unknown = structuredClone(turboMemoryBlocked);
      assertCandidateExecutionProfile(captured);
      engine.inspection = captured;
      if (reverseReasons) engine.inspection.blockingReasons?.reverse();
      engine.resources.systemMemory = { availableBytes: 32 * 1024 ** 3 };
      const controller = new HuggingFacePageActionController(engine, () => ({
        tabId: 'turbo',
        documentId: 'turbo_document',
        url: 'https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo',
      }));
      controller.activeTabChanged();
      await vi.advanceTimersByTimeAsync(250);
      expect(controller.snapshot()).toMatchObject({
        repository: 'Qwen/Qwen-Image-2.1-Turbo',
        exactRevision: 'fc13ba1986fcb905854dc8de2f52eee395f49b5b',
        status: 'resource-error',
        securityStatus: 'allowed',
        message: 'Not enough available model memory. Unload another model, then retry.',
        memoryBytes: 54_269_441_528,
        availableMemoryBytes: 32 * 1024 ** 3,
        blockingReasons: engine.inspection.blockingReasons,
      });
      await expect(controller.confirmInstall('user-trusted')).rejects.toThrow('NOT_AVAILABLE');

      engine.inspection = {
        ...engine.inspection,
        executable: true,
        executableBackends: ['mps'],
        loadDisposition: 'trust-required',
        eligibility: {
          ...engine.inspection.eligibility!,
          resources: { state: 'admissible', reasons: [] },
        },
        blockingReasons: ['CONFORMANCE_REQUIRED'],
        runtimePlanFailure: undefined,
      };
      engine.resources.systemMemory.availableBytes = 64 * 1024 ** 3;
      controller.retry();
      await vi.advanceTimersByTimeAsync(250);
      expect(controller.snapshot()).toMatchObject({
        status: 'unverified',
        trustMode: 'user-trusted',
        availableMemoryBytes: 64 * 1024 ** 3,
      });
      controller.destroy();
    },
  );

  it('keeps actual runtime incompatibility visible alongside resource and conformance reasons', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile(),
      executable: false,
      executableBackends: [],
      blockingReasons: [
        'CONFORMANCE_REQUIRED',
        'MODEL_MEMORY_INSUFFICIENT',
        'MODEL_SUBSET_INCOMPLETE',
        'RUNTIME_RESOURCES_INSUFFICIENT',
      ],
    };
    const controller = new HuggingFacePageActionController(engine, () => ({
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    }));
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);
    expect(controller.snapshot()).toMatchObject({
      status: 'unsupported',
      message: 'The repository is missing files required by its runtime pack.',
    });
    controller.destroy();
  });

  it('pins exact inspection evidence before granting install authority', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    const active: PageActionTabIdentity = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    const operations: TrustedModelOperation[] = [];
    controller.on('operation-changed', (operation) =>
      operations.push(operation as TrustedModelOperation),
    );
    controller.activeTabChanged();
    expect(controller.snapshot()?.status).toBe('inspecting');
    await vi.advanceTimersByTimeAsync(250);
    expect(controller.snapshot()).toMatchObject({
      status: 'loadable',
      repository: 'google/gemma-3-1b-it',
      exactRevision: 'a'.repeat(40),
      taskContract: 'chat-v1',
      securityStatus: 'allowed',
    });
    await controller.confirmInstall('oryn-verified');
    expect(engine.installed?.model).toEqual(engine.inspection.model);
    expect(controller.snapshot()?.status).toBe('installed');
    expect(operations.map((operation) => operation.stage)).toEqual(['downloading', 'ready']);
    expect(operations[0]).toMatchObject({
      origin: 'https://huggingface.co',
      repository: 'google/gemma-3-1b-it',
      downloadedBytes: 0,
      totalBytes: 1_000,
    });
  });

  it('never applies an inspection result to a navigated document', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    let active: PageActionTabIdentity = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    };
    let resolveInspection!: (value: CandidateExecutionProfile) => void;
    engine.inspectModel = () =>
      new Promise((resolve) => {
        resolveInspection = resolve;
      });
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);
    active = { tabId: 'one', documentId: 'doc_two', url: 'https://example.com/' };
    controller.activeTabChanged();
    resolveInspection(profile());
    await Promise.resolve();
    await Promise.resolve();
    expect(controller.snapshot()).toBeUndefined();
  });

  it('serializes inspection and coalesces stale queued documents', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    const pending: Array<(value: CandidateExecutionProfile) => void> = [];
    engine.inspectModel = () =>
      new Promise((resolve) => {
        pending.push(resolve);
      });
    let active: PageActionTabIdentity = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    vi.advanceTimersByTime(250);
    for (let index = 0; index < 3; index += 1) await Promise.resolve();
    expect(pending).toHaveLength(1);

    active = {
      tabId: 'one',
      documentId: 'doc_two',
      url: 'https://huggingface.co/google/gemma-3-2b-it',
    };
    controller.activeTabChanged();
    vi.advanceTimersByTime(250);
    expect(pending).toHaveLength(1);

    pending[0]?.(profile());
    vi.useRealTimers();
    await new Promise((resolve) => setImmediate(resolve));
    expect(pending).toHaveLength(2);
    pending[1]?.(profile('google/gemma-3-2b-it'));
    await new Promise((resolve) => setImmediate(resolve));
    expect(controller.snapshot()).toMatchObject({
      status: 'loadable',
      exactRevision: 'a'.repeat(40),
    });
  });

  it('reports unsafe/custom-code repositories as blocked without override', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile(),
      executable: false,
      requiresCustomCode: true,
      runtimePack: undefined,
      executableBackends: [],
      blockingReasons: ['CUSTOM_CODE_REQUIRED'],
    };
    const active = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);
    expect(controller.snapshot()).toMatchObject({ status: 'blocked', securityStatus: 'blocked' });
    await expect(controller.confirmInstall('user-trusted')).rejects.toThrow('NOT_AVAILABLE');
  });

  it('reports the actual runtime blocker instead of hiding it behind conformance wording', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile('RunDiffusion/Juggernaut-XI-v11'),
      executable: false,
      executableBackends: [],
      blockingReasons: ['MODEL_SUBSET_INCOMPLETE'],
      runtimePack: {
        ...profile().runtimePack!,
        trustedMetadata: {
          license: 'Repository-defined',
          conformance: { status: 'fixture-only', platforms: ['macos-arm64'] },
        },
      },
    };
    const active = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/RunDiffusion/Juggernaut-XI-v11',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);

    expect(controller.snapshot()).toMatchObject({
      status: 'unsupported',
      blockingReasons: ['MODEL_SUBSET_INCOMPLETE'],
      message: 'The repository is missing files required by its runtime pack.',
    });
  });

  it('reports memory limits for a recognized pipeline before pending conformance', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile('Qwen/Qwen-Image-2.1'),
      executable: false,
      executableBackends: [],
      blockingReasons: ['CONFORMANCE_REQUIRED', 'MODEL_MEMORY_INSUFFICIENT'],
    };
    const inspect = vi.spyOn(engine, 'inspectModel');
    const controller = new HuggingFacePageActionController(engine, () => ({
      tabId: 'one',
      documentId: 'doc_one',
      url: `https://huggingface.co/Qwen/Qwen-Image-2.1/tree/${'a'.repeat(40)}`,
    }));
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);
    expect(controller.snapshot()).toMatchObject({
      status: 'resource-error',
      message: 'Not enough available model memory. Unload another model, then retry.',
      blockingReasons: ['CONFORMANCE_REQUIRED', 'MODEL_MEMORY_INSUFFICIENT'],
    });
    await expect(controller.confirmInstall('user-trusted')).rejects.toThrow('NOT_AVAILABLE');
    expect(controller.snapshot()).toMatchObject({
      memoryBytes: 2_000,
      availableMemoryBytes: 1_500,
    });

    engine.resources.localAI = { memoryBudgetBytes: 1_200, estimatedResidentModelBytes: 800 };
    engine.emit('resources-updated', engine.resources);
    expect(controller.snapshot()?.availableMemoryBytes).toBe(400);
    engine.resources.systemMemory!.availableBytes = 200;
    engine.emit('resources-updated', engine.resources);
    expect(controller.snapshot()?.availableMemoryBytes).toBe(200);
    engine.resources.localAI.memoryBudgetBytes = 0;
    engine.emit('resources-updated', engine.resources);
    expect(controller.snapshot()?.availableMemoryBytes).toBe(0);
    engine.resources.localAI.memoryBudgetBytes = undefined;
    engine.resources.systemMemory!.availableBytes = 3_000;
    engine.inspection.executable = true;
    engine.inspection.executableBackends = ['mps'];
    engine.inspection.blockingReasons = [];
    controller.retry();
    await vi.advanceTimersByTimeAsync(250);
    expect(inspect).toHaveBeenCalledTimes(2);
    expect(controller.snapshot()).toMatchObject({
      status: 'loadable',
      availableMemoryBytes: 3_000,
    });
    controller.destroy();
    expect(engine.listenerCount('resources-updated')).toBe(0);
  });

  it('offers an exact-revision warning path for safe bundled-runner models only', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile('Qwen/Qwen3.5-4B'),
      runner: 'transformers',
      runtimePack: undefined,
    };
    const active = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/Qwen/Qwen3.5-4B',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);
    expect(controller.snapshot()).toMatchObject({
      status: 'unverified',
      trustMode: 'user-trusted',
      taskContract: 'chat-v1',
      exactRevision: 'a'.repeat(40),
    });
    await expect(controller.confirmInstall('oryn-verified')).rejects.toThrow('NOT_AVAILABLE');
    await controller.confirmInstall('user-trusted');
    expect(engine.installed?.model).toEqual(engine.inspection.model);
    expect(controller.snapshot()).toMatchObject({
      status: 'installed',
      trustMode: 'user-trusted',
    });
  });

  it('offers user trust when a signed runtime supports an untested exact revision', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.inspection = {
      ...profile('Qwen/Qwen3-ASR-1.7B-hf'),
      descriptorSchemaVersion: 'oryn-model-descriptor/1',
      capabilityRegistryDigest: 'c'.repeat(64),
      loadDisposition: 'trust-required',
      blockingReasons: ['CONFORMANCE_REQUIRED'],
      eligibility: {
        access: { state: 'available', reasons: [] },
        artifacts: { state: 'safe', reasons: [] },
        runtime: { state: 'supported', reasons: [] },
        task: { state: 'supported', reasons: [] },
        resources: { state: 'admissible', reasons: [] },
        conformance: { state: 'untested', reasons: ['CONFORMANCE_REQUIRED'] },
      },
      technicalCapabilities: {
        task: 'automatic-speech-recognition',
        inputModalities: ['audio'],
        outputModalities: ['text'],
      },
      runtimePack: {
        ...profile().runtimePack!,
        taskContract: 'transcription-v1',
      },
    };
    const active = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);

    expect(controller.snapshot()).toMatchObject({
      status: 'unverified',
      trustMode: 'user-trusted',
      taskContract: 'transcription-v1',
      blockingReasons: ['CONFORMANCE_REQUIRED'],
    });
    await controller.confirmInstall('user-trusted');
    expect(engine.installed?.model.repository).toBe('Qwen/Qwen3-ASR-1.7B-hf');
  });

  it('publishes live byte progress and cancels only through the explicit cancel command', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    let completeInstall!: () => void;
    engine.installModel = async (operationId, model) => {
      engine.installed = { operationId, model };
      await new Promise<void>((resolve) => {
        completeInstall = resolve;
      });
    };
    let active: PageActionTabIdentity = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    const operations: TrustedModelOperation[] = [];
    controller.on('operation-changed', (operation) =>
      operations.push(operation as TrustedModelOperation),
    );
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);

    const installing = controller.confirmInstall('oryn-verified');
    const operationId = engine.installed?.operationId;
    expect(operationId).toBeTruthy();
    engine.emit('model-install-progress', {
      operationId,
      downloadedBytes: 375,
      totalBytes: 1_000,
    });
    expect(controller.snapshot()).toMatchObject({
      status: 'downloading',
      downloadedBytes: 375,
      totalBytes: 1_000,
    });

    active = { tabId: 'one', documentId: 'doc_two', url: 'https://example.com/' };
    controller.activeTabChanged();
    await controller.cancelInstall(operationId);
    expect(engine.cancelled).toBe(operationId);
    expect(operations.map((operation) => operation.stage)).toEqual([
      'downloading',
      'downloading',
      'cancelled',
    ]);
    completeInstall();
    await installing;
  });

  it('keeps the authoritative download state when installation updates inventory', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    let completeInstall!: () => void;
    engine.installModel = async (operationId, model) => {
      engine.installed = { operationId, model };
      await new Promise<void>((resolve) => {
        completeInstall = resolve;
      });
    };
    const active = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);

    const installing = controller.confirmInstall('oryn-verified');
    const operationId = engine.installed?.operationId;
    engine.emit('model-inventory-updated');
    await vi.advanceTimersByTimeAsync(250);

    expect(controller.snapshot()).toMatchObject({ status: 'downloading', operationId });
    completeInstall();
    await installing;
    expect(controller.snapshot()).toMatchObject({ status: 'installed' });
  });

  it('preserves nested Engine resource codes when an installation cannot start', async () => {
    vi.useFakeTimers();
    const engine = new FakeEngine();
    engine.installModel = async () => {
      throw Object.assign(new Error('There is not enough disk space.'), {
        protocolError: {
          code: 'DISK_SPACE_INSUFFICIENT',
          details: { requiredBytes: 16_314_044_717, availableBytes: 15_032_385_536 },
        },
      });
    };
    const active = {
      tabId: 'one',
      documentId: 'doc_one',
      url: 'https://huggingface.co/google/gemma-3-1b-it',
    };
    const controller = new HuggingFacePageActionController(engine, () => active);
    const operations: TrustedModelOperation[] = [];
    controller.on('operation-changed', (operation) =>
      operations.push(operation as TrustedModelOperation),
    );
    controller.activeTabChanged();
    await vi.advanceTimersByTimeAsync(250);

    await controller.confirmInstall('oryn-verified');

    expect(controller.snapshot()).toMatchObject({
      status: 'resource-error',
      blockingReasons: ['DISK_SPACE_INSUFFICIENT'],
      requiredDiskBytes: 16_314_044_717,
      availableDiskBytes: 15_032_385_536,
      message: 'Not enough disk space. Remove models or free storage, then retry.',
    });
    expect(controller.snapshot()?.operationId).toMatch(/^hf_install_/u);
    expect(operations.at(-1)).toMatchObject({
      stage: 'failed',
      errorCode: 'DISK_SPACE_INSUFFICIENT',
    });
  });
});
