// Deterministic Engine boundary double. This never proves real Engine admission or byte I/O.
import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { vi } from 'vitest';
import { LocalAIPermissionStore } from '../model-resolution/local-ai-permission-store';
import { ModelPreferenceStore } from '../model-resolution/model-preference-store';
import { ModelResolver } from '../model-resolution/model-resolver';
import { PermissionPromptCoordinator } from '../model-resolution/permission-prompt-coordinator';
import { BrowserModelSessionRegistry } from '../model-resolution/browser-model-sessions';
export const owner = {
  origin: 'https://validation-a.test',
  tabId: 'tab_a',
  webContentsId: 10,
  frameId: 1,
  frameUrl: 'https://validation-a.test/app',
  documentId: 'document_a',
};
export const exact = (repository = 'validation/model', revision = 'a'.repeat(40)) => ({
  provider: 'huggingface' as const,
  repository,
  revision,
});
export function profile(model = exact()) {
  return {
    model,
    exactRevision: model.revision,
    executable: true,
    runner: 'transformers',
    executableBackends: ['cpu'],
    requiresCustomCode: false,
    estimatedDiskBytes: 64,
    estimatedMemoryBytes: 128,
    technicalCapabilities: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true },
    },
  };
}
export function request(model = exact()) {
  return {
    role: 'chat',
    preferredModel: model,
    requirements: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true },
    },
    substitution: { automatic: 'none' },
  };
}
export function fixture(initialModels = [exact()]) {
  const root = mkdtempSync(join(tmpdir(), 'oryn-validation-'));
  const models = [...initialModels];
  const engine = Object.assign(new EventEmitter(), {
    inspectModel: vi.fn(async (model) => profile(model)),
    getModelInventory: vi.fn(async () => ({
      models: models.map((model, index) => ({
        installationId: `install_${index}`,
        model,
        installationState: 'installed',
        residencyState: 'unloaded',
        logicalSizeBytes: 64,
        reclaimableSizeBytes: 64,
        estimatedResidentBytes: 128,
        activeSessions: 0,
        runningJobs: 0,
        queuedJobs: 0,
      })),
    })),
    attachModelSession: vi.fn(async (sessionId, model) => ({
      sessionId,
      model,
      runtime: { runner: 'transformers', backend: 'cpu', deviceClass: 'cpu' },
      technicalCapabilities: profile(model).technicalCapabilities,
      residencyState: 'loaded',
    })),
    releaseModelSession: vi.fn(async () => true),
    installModel: vi.fn(async (_operationId, model) => {
      models.push(model);
      return {};
    }),
    cancelModelInstall: vi.fn(async () => true),
    removeModel: vi.fn(async (operationId, installationId) => {
      models.splice(Number(installationId.split('_').at(-1)), 1);
      return {
        operationId,
        installationId,
        state: 'removed',
        reclaimedBytes: 64,
        sharedBytesRetained: 0,
      };
    }),
  });
  const preferences = new ModelPreferenceStore(join(root, 'preferences.json'));
  const permissions = new LocalAIPermissionStore(join(root, 'permissions.json'));
  const resolver = new ModelResolver(
    engine as unknown as ConstructorParameters<typeof ModelResolver>[0],
    preferences,
  );
  const prompts = new PermissionPromptCoordinator();
  const sessions = new BrowserModelSessionRegistry(
    engine as unknown as ConstructorParameters<typeof BrowserModelSessionRegistry>[0],
    resolver,
    permissions,
    preferences,
    prompts,
  );
  return {
    root,
    models,
    engine,
    preferences,
    permissions,
    resolver,
    prompts,
    sessions,
    cleanup() {
      rmSync(root, { recursive: true, force: true });
    },
  };
}
