import { readFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it, vi } from 'vitest';

import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  ResolvedModelRef,
} from '@ocno/oryn-protocol';

import type { AIActivityStore } from '../data/ai-activity-store';
import type { BrowserJobRegistry } from '../jobs/browser-job-registry';
import { ModelManagerService } from '../model-manager/model-manager-service';
import type {
  BrowserModelSession,
  BrowserModelSessionRegistry,
} from '../model-resolution/browser-model-sessions';
import { ModelPreferenceStore } from '../model-resolution/model-preference-store';
import { ModelResolver } from '../model-resolution/model-resolver';

const qwen = model('qwen/chat', 'a');
const gemma = model('google/gemma-chat', 'b');

function model(repository: string, seed: string): ResolvedModelRef {
  return { provider: 'huggingface', repository, revision: seed.repeat(40) };
}

function profile(value: ResolvedModelRef): CandidateExecutionProfile {
  return {
    model: value,
    exactRevision: value.revision,
    executable: true,
    runner: 'transformers',
    executableBackends: ['cpu'],
    requiresCustomCode: false,
    estimatedDiskBytes: 100,
    estimatedMemoryBytes: 125,
    technicalCapabilities: {
      task: 'text-generation',
      architectures: [
        value.repository.includes('gemma') ? 'Gemma3ForCausalLM' : 'Qwen2ForCausalLM',
      ],
      parameterCount: value.repository.includes('gemma') ? 4_000_000_000 : 1_000_000_000,
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true, streaming: true },
    },
  };
}

function inventory(value: ResolvedModelRef, installationId: string): EngineModelInventoryRecord {
  const execution = profile(value);
  return {
    installationId,
    model: value,
    installationState: 'installed',
    residencyState: value === qwen ? 'loaded' : 'unloaded',
    logicalSizeBytes: 100,
    reclaimableSizeBytes: 80,
    estimatedResidentBytes: 125,
    activeSessions: value === qwen ? 1 : 0,
    runningJobs: 0,
    queuedJobs: 0,
    installedAt: 100,
    lastUsedAt: 200,
    runner: execution.runner,
    executableBackends: execution.executableBackends,
    technicalCapabilities: execution.technicalCapabilities,
  };
}

describe('trusted Model Manager', () => {
  it('projects authoritative state and contextual installed substitutes', async () => {
    const records = [inventory(qwen, 'install_qwen'), inventory(gemma, 'install_gemma')];
    const engine = {
      getModelInventory: vi.fn(async () => ({ models: records })),
      getRuntimePackInventory: vi.fn(async () => ({
        catalogVersion: 1,
        catalogDigest: 'a'.repeat(64),
        thirdPartyPublishersEnabled: false as const,
        automaticExecutableUpdatesEnabled: false as const,
        packs: [
          {
            packId: 'oryn.transformers-pytorch',
            version: '0.1.0',
            publisher: 'oryn.first-party',
            abi: '1',
            dependencyClosure: 'transformers-5',
            platforms: ['macos-arm64' as const],
            distribution: 'bundled' as const,
            state: 'active' as const,
            bundled: true,
            removable: false,
            sizeBytes: 0,
            minimumSafeVersion: '0.1.0',
            revoked: false,
            license: {
              summary: 'Composite runtime',
              notice: 'Review packaged notices.',
              access: 'public' as const,
              acceptanceRequired: false,
              accepted: true,
            },
          },
        ],
      })),
      cleanupRuntimePacks: vi.fn(async () => ({ removedPacks: 1, reclaimedBytes: 2048 })),
      setModelRuntimeOverride: vi.fn(async () => ({
        planId: `plan-${'1'.repeat(24)}`,
        policy: 'balanced' as const,
        reasonCodes: ['USER_OVERRIDE'],
        selected: {
          candidateId: `candidate-${'2'.repeat(24)}`,
          foundation: { id: 'oryn.runtime.pytorch', version: '2.8.0' },
          adapter: { id: 'oryn.adapter.transformers', version: '1.0.0' },
          variantId: 'oryn.compat.test',
          backend: 'cpu',
          precision: 'fp32',
          estimatedMemoryBytes: 125,
        },
        fallbacks: [],
        rejected: [],
      })),
      unloadModel: vi.fn(async () => ({
        operationId: 'unload_x',
        model: qwen,
        state: 'unloaded' as const,
      })),
      previewModelRemoval: vi.fn(async (installationId: string) => ({
        installationId,
        canRemoveNow: false,
        activeSessions: 1,
        runningJobs: 0,
        queuedJobs: 0,
        reclaimableBytes: 80,
        sharedBytesRetained: 20,
        blockers: ['active-sessions'],
      })),
      removeModel: vi.fn(async (operationId: string, installationId: string) => ({
        operationId,
        installationId,
        state: 'deferred' as const,
        reclaimedBytes: 0,
        sharedBytesRetained: 20,
      })),
    };
    const store = new ModelPreferenceStore(
      join(mkdtempSync(join(tmpdir(), 'oryn-manager-')), 'preferences.json'),
    );
    store.setGlobal('chat', gemma);
    store.setSite('https://chat.test', 'chat', qwen);
    const inspector = {
      inspectModel: vi.fn(async (value: ResolvedModelRef) => profile(value)),
    };
    const resolver = new ModelResolver(inspector, store);
    const session = {
      sessionId: 'session_qwen',
      role: 'chat',
      origin: 'https://chat.test',
      tabId: 'tab_one',
      webContentsId: 1,
      documentId: 'document_one',
      request: {
        role: 'chat',
        preferredModel: qwen,
        requirements: {
          task: 'text-generation',
          inputModalities: ['text'],
          outputModalities: ['text'],
          capabilities: { chat: true, streaming: true },
        },
        substitution: { automatic: 'compatible' },
      },
      resolution: {
        preferredModel: qwen,
        resolvedModel: qwen,
        compatibility: 'certified',
        executability: 'executable',
        security: 'allowed',
        trust: 'oryn-verified',
        developerCertified: true,
        substituted: false,
        overriddenByUser: false,
        resolutionReason: 'preferred-model',
        compatibilityReasons: [],
        downloadRequired: false,
      },
      runtime: { runner: 'transformers', backend: 'cpu', deviceClass: 'cpu' },
      capabilities: profile(qwen).technicalCapabilities,
    } as BrowserModelSession;
    const sessions = {
      snapshot: () => [session],
      releaseForModel: vi.fn(async () => undefined),
    };
    const jobs = { cancelAllForModel: vi.fn(async () => undefined) };
    const manager = new ModelManagerService(
      engine,
      resolver,
      store,
      sessions as unknown as BrowserModelSessionRegistry,
      jobs as unknown as BrowserJobRegistry,
      { list: () => [] } as unknown as AIActivityStore,
    );

    const snapshot = await manager.snapshot();
    expect(snapshot).toMatchObject({ totalLogicalBytes: 200, totalReclaimableBytes: 160 });
    expect(inspector.inspectModel).not.toHaveBeenCalled();
    expect(engine.getRuntimePackInventory).not.toHaveBeenCalled();
    expect(snapshot.models.every((model) => model.substitutes.length === 0)).toBe(true);
    const packs = await manager.runtimePacks();
    expect(packs.runtimePacks[0]).toMatchObject({
      packId: 'oryn.transformers-pytorch',
      publisher: 'oryn.first-party',
      state: 'active',
    });
    expect(await manager.cleanupRuntimePacks()).toEqual({
      removedPacks: 1,
      reclaimedBytes: 2048,
    });
    expect(snapshot.models[0]).toMatchObject({
      repository: 'qwen/chat',
      publisher: 'qwen',
      exactRevision: qwen.revision,
      task: 'text-generation',
      family: 'Qwen2',
      parameterCount: 1_000_000_000,
      activeSessions: 1,
      usedBy: ['https://chat.test'],
    });
    expect(snapshot.models[0]?.sourceUrl).toBe(
      `https://huggingface.co/qwen/chat/tree/${qwen.revision}`,
    );
    const substitutes = await manager.modelSubstitutes('install_qwen');
    expect(inspector.inspectModel).toHaveBeenCalledTimes(2);
    expect(substitutes[0]).toMatchObject({
      repository: 'google/gemma-chat',
      compatibility: 'compatible',
      developerCertified: false,
      contextOrigin: 'https://chat.test',
      contextRole: 'chat',
    });

    inspector.inspectModel.mockRejectedValueOnce(new Error('Inspection timed out'));
    await expect(manager.modelSubstitutes('install_qwen')).rejects.toThrow('Inspection timed out');
    await expect(manager.modelSubstitutes('install_missing')).rejects.toThrow(
      'MODEL_NOT_INSTALLED',
    );
    engine.getRuntimePackInventory.mockRejectedValueOnce(new Error('Runtime packs unavailable'));
    await expect(manager.runtimePacks()).rejects.toThrow('Runtime packs unavailable');
    inspector.inspectModel.mockClear();
    await expect(manager.snapshot()).resolves.toMatchObject({ totalLogicalBytes: 200 });
    expect(inspector.inspectModel).not.toHaveBeenCalled();

    await manager.unload('install_qwen');
    expect(engine.unloadModel).toHaveBeenCalledWith(
      expect.stringMatching(/^unload_/u),
      qwen,
      'when-idle',
    );
    await manager.setRuntimeOverride('install_qwen', `candidate-${'2'.repeat(24)}`);
    expect(engine.setModelRuntimeOverride).toHaveBeenCalledWith(
      qwen,
      `candidate-${'2'.repeat(24)}`,
    );
    expect(engine.unloadModel).toHaveBeenLastCalledWith(
      expect.stringMatching(/^runtime-change_/u),
      qwen,
      'when-idle',
    );
    expect(await manager.previewRemoval('install_qwen')).toMatchObject({ canRemoveNow: false });
    await manager.remove('install_qwen', 'cancel-and-delete');
    expect(jobs.cancelAllForModel).toHaveBeenCalledWith(qwen);
    expect(sessions.releaseForModel).toHaveBeenCalledWith(qwen);
    expect(engine.removeModel).toHaveBeenCalledWith(
      expect.stringMatching(/^remove_/u),
      'install_qwen',
      'when-idle',
    );
    expect(store.getGlobal('chat')).toEqual(gemma);
    expect(store.getSite('https://chat.test', 'chat')).toEqual(qwen);
    const future = await resolver.resolve(
      { ...session.request, preferredModel: gemma },
      { origin: 'https://chat.test', inventory: [records[1]!] },
    );
    expect(future.resolution).toMatchObject({
      resolvedModel: qwen,
      downloadRequired: true,
      resolutionReason: 'site-preference',
    });
  });

  it('keeps every management operation out of the website preload', () => {
    const website = readFileSync(
      join(import.meta.dirname, '../../preload/website-preload.ts'),
      'utf8',
    );
    for (const forbidden of [
      'models-get',
      'model-substitutes',
      'runtime-packs-get',
      'model-unload-trusted',
      'model-remove-preview-trusted',
      'model-remove-trusted',
      'model-source-open-trusted',
      'runtime-packs-cleanup',
      'getModelManager',
      'cleanupManagedRuntimePacks',
      'removeManagedModel',
    ]) {
      expect(website).not.toContain(forbidden);
    }
  });
});
