import { readFileSync, mkdtempSync } from 'node:fs';
import { EventEmitter } from 'node:events';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it, vi } from 'vitest';
import { taskRequirementsFrom, validateTaskInvocation } from '@ocno/oryn-protocol';

import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  EngineSessionAttached,
  ResolvedModelRef,
} from '@ocno/oryn-protocol';

import type { BrowserCallerContext } from '../browser-caller-context';
import { BrowserCallerContextService } from '../browser-caller-context';
import {
  BrowserModelSessionError,
  BrowserModelSessionRegistry,
} from '../model-resolution/browser-model-sessions';
import { LocalAIPermissionStore } from '../model-resolution/local-ai-permission-store';
import { ModelPreferenceStore } from '../model-resolution/model-preference-store';
import { ModelResolver } from '../model-resolution/model-resolver';
import { PermissionPromptCoordinator } from '../model-resolution/permission-prompt-coordinator';
import unverifiedChatProfiles from './fixtures/unverified-chat-profiles.json';

function exact(repository: string): ResolvedModelRef {
  const revision = `${Buffer.from(repository).toString('hex')}${'a'.repeat(40)}`.slice(0, 40);
  return { provider: 'huggingface', repository, revision };
}

function executionProfile(repository: string): CandidateExecutionProfile {
  const model = exact(repository);
  return {
    model,
    exactRevision: model.revision,
    executable: true,
    runner: 'transformers',
    executableBackends: ['cpu'],
    requiresCustomCode: false,
    estimatedDiskBytes: 100,
    estimatedMemoryBytes: 200,
    technicalCapabilities: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true },
    },
  };
}

class FakeEngine extends EventEmitter {
  readonly released: string[] = [];
  readonly attached: Array<{ sessionId: string; model: ResolvedModelRef }> = [];

  constructor(
    readonly profiles: Record<string, CandidateExecutionProfile>,
    readonly installedRepositories: string[],
  ) {
    super();
  }

  async inspectModel(model: ResolvedModelRef): Promise<CandidateExecutionProfile> {
    return structuredClone(this.profiles[model.repository]!);
  }

  async getModelInventory(): Promise<{ models: EngineModelInventoryRecord[] }> {
    return {
      models: this.installedRepositories.map((repository, index) => ({
        installationId: `install_${index}`,
        model: exact(repository),
        installationState: 'installed',
        residencyState: 'unloaded',
        logicalSizeBytes: 100,
        reclaimableSizeBytes: 100,
        estimatedResidentBytes: 200,
        activeSessions: 0,
        runningJobs: 0,
        queuedJobs: 0,
      })),
    };
  }

  async attachModelSession(
    sessionId: string,
    model: ResolvedModelRef,
  ): Promise<EngineSessionAttached> {
    this.attached.push({ sessionId, model });
    return {
      sessionId,
      model,
      runtime: { runner: 'transformers', backend: 'cpu', deviceClass: 'cpu' },
      technicalCapabilities: this.profiles[model.repository]!.technicalCapabilities,
      ...(this.profiles[model.repository]!.taskProfile
        ? { taskProfile: structuredClone(this.profiles[model.repository]!.taskProfile!) }
        : {}),
      residencyState: 'loaded',
    };
  }

  async releaseModelSession(sessionId: string): Promise<boolean> {
    this.released.push(sessionId);
    return true;
  }

  async installModel(operationId: string, model: ResolvedModelRef): Promise<unknown> {
    this.installedRepositories.push(model.repository);
    this.emit('model-install-progress', {
      operationId,
      downloadedBytes: 100,
      totalBytes: 100,
    });
    return {};
  }

  async cancelModelInstall(): Promise<boolean> {
    return true;
  }
}

function context(overrides: Partial<BrowserCallerContext> = {}): BrowserCallerContext {
  return {
    tabId: 'tab_one',
    webContentsId: 1,
    frameId: 10,
    frameUrl: 'https://site.test/app',
    origin: 'https://site.test',
    documentId: 'document_one',
    ...overrides,
  };
}

function modelRequest(repository = 'org/preferred') {
  return {
    role: 'chat',
    preferredModel: { provider: 'huggingface', repository },
    requirements: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true },
    },
    substitution: { automatic: 'compatible' },
  };
}

function setup(repositories = ['org/preferred'], installedRepositories = repositories) {
  const root = mkdtempSync(join(tmpdir(), 'oryn-sessions-'));
  const profiles = Object.fromEntries(
    repositories.map((repository) => [repository, executionProfile(repository)]),
  );
  const engine = new FakeEngine(profiles, [...installedRepositories]);
  const preferences = new ModelPreferenceStore(join(root, 'preferences.json'));
  const permissions = new LocalAIPermissionStore(join(root, 'permissions.json'));
  const prompts = new PermissionPromptCoordinator();
  const callerContexts = new BrowserCallerContextService();
  const sessions = new BrowserModelSessionRegistry(
    engine,
    new ModelResolver(engine, preferences),
    permissions,
    preferences,
    prompts,
    callerContexts,
  );
  return { engine, preferences, permissions, prompts, sessions, callerContexts };
}

describe('Local AI permission and browser Model Sessions', () => {
  it('prepares a changed installed runtime before attachment without reporting a download', async () => {
    const value = setup();
    value.engine.profiles['org/preferred']!.runtimePreparationRequired = true;
    const prepared = vi.spyOn(value.engine, 'installModel');
    const attached = vi.spyOn(value.engine, 'attachModelSession');
    value.prompts.on('prompt', (prompt) =>
      value.prompts.respondFromTrustedUI({ promptId: prompt.promptId, action: 'allow-once' }),
    );
    const session = await value.sessions.requestSession(modelRequest(), context());
    expect(session.resolution.downloadRequired).toBe(false);
    expect(prepared).toHaveBeenCalledOnce();
    expect(prepared.mock.invocationCallOrder[0]).toBeLessThan(
      attached.mock.invocationCallOrder[0]!,
    );
  });

  it('attaches an exact workspace model while runtime evidence remains outside public inventory', async () => {
    const value = setup();
    const profile = executionProfile('org/preferred');
    profile.runtimePack = {
      packId: 'oryn.transformers-pytorch',
      packVersion: '0.1.0',
      runnerId: 'transformers.chat',
      taskContract: 'chat-v1',
      compatibilityLevel: 'native-declarative',
      manifestDigest: 'a'.repeat(64),
    };

    const session = await value.sessions.attachModelWorkspaceSession(
      profile,
      'chat-v1',
      context({ origin: 'oryn://model-workspace' }),
    );

    expect(session.resolution.resolvedModel).toEqual(profile.model);
    expect(value.engine.attached).toHaveLength(1);
  });

  it('attaches an explicitly user-trusted safe bundled-runner workspace as an override', async () => {
    const value = setup();
    const profile = executionProfile('org/preferred');

    const session = await value.sessions.attachModelWorkspaceSession(
      profile,
      'chat-v1',
      context({ origin: 'oryn://model-workspace' }),
      'user-trusted',
    );

    expect(session.resolution).toMatchObject({
      compatibility: 'uncertain',
      developerCertified: false,
      overriddenByUser: true,
      resolutionReason: 'user-override',
    });
  });

  it('pins the effective workspace contract when legacy chat capability is unknown', async () => {
    const value = setup();
    const profile = executionProfile('org/preferred');
    profile.technicalCapabilities.capabilities!.chat = 'unknown';
    profile.taskProfile = structuredClone(
      unverifiedChatProfiles[0]!.taskProfile,
    ) as CandidateExecutionProfile['taskProfile'];
    profile.loadDisposition = 'trust-required';
    profile.blockingReasons = ['CONFORMANCE_REQUIRED'];
    value.engine.profiles['org/preferred'] = profile;
    const attach = vi.spyOn(value.engine, 'attachModelSession');

    const session = await value.sessions.attachModelWorkspaceSession(
      profile,
      'chat-v1',
      context({ origin: 'oryn://model-workspace' }),
      'user-trusted',
    );

    expect(attach).toHaveBeenCalledWith(expect.any(String), profile.model, true, {
      task: profile.taskProfile!.task,
      contract: { id: 'chat-v1', revision: 1 },
    });
    expect(session.taskProfile?.contract).toEqual({ id: 'chat-v1', revision: 1 });
    expect(session.resolution).toMatchObject({
      resolvedModel: profile.model,
      trust: 'user-approved-unverified',
      developerCertified: false,
      overriddenByUser: true,
    });
  });

  it('keeps negotiated task modalities when legacy model capabilities are broader', async () => {
    const value = setup();
    const profile = executionProfile('org/preferred');
    profile.technicalCapabilities = {
      task: 'zero-shot-image-classification',
      inputModalities: ['image', 'text'],
      outputModalities: ['structured'],
      capabilities: {},
    };
    profile.taskProfile = {
      kind: 'task-profile',
      schemaVersion: 1,
      profileId: 'zero-shot-image-classification-v1',
      task: 'zero-shot-image-classification',
      contract: { id: 'zero-shot-image-classification-v1', revision: 1 },
      inputModalities: ['image'],
      outputModalities: ['structured'],
      inputSchemaRef: 'taskInput.zero-shot-image-classification-v1',
      resultSchemaRef: 'taskResult.zero-shot-image-classification-v1',
      features: {},
      parameters: {},
      limits: {},
      streamingChannels: [],
      extensions: {},
      provenance: 'adapter-derived',
    };
    value.engine.profiles['org/preferred'] = profile;
    const session = await value.sessions.attachModelWorkspaceSession(
      profile,
      'zero-shot-image-classification-v1',
      context({ origin: 'oryn://model-workspace' }),
      'user-trusted',
    );
    const requirements = taskRequirementsFrom(session.request.requirements);
    expect(requirements?.inputModalities).toEqual(['image']);
    const input = { imageArtifactId: `artifact_${'a'.repeat(24)}`, labels: ['red', 'blue'] };
    expect(() => validateTaskInvocation(session.taskProfile!, input, requirements)).not.toThrow();
    expect(() =>
      validateTaskInvocation(session.taskProfile!, { ...input, text: 'extra' }, requirements),
    ).toThrow();
  });

  it('reprepares an installed workspace before attaching its exact task contract', async () => {
    const value = setup();
    const profile = executionProfile('org/preferred');
    profile.runtimePreparationRequired = true;
    const prepared = vi.spyOn(value.engine, 'installModel');
    const attached = vi.spyOn(value.engine, 'attachModelSession');
    await value.sessions.attachModelWorkspaceSession(
      profile,
      'chat-v1',
      context({ origin: 'oryn://model-workspace' }),
      'user-trusted',
    );
    expect(prepared).toHaveBeenCalledWith(
      expect.any(String),
      profile.model,
      expect.objectContaining({ contract: { id: 'chat-v1', revision: 1 } }),
    );
    expect(attached).toHaveBeenCalledWith(
      expect.any(String),
      profile.model,
      true,
      expect.objectContaining({ contract: { id: 'chat-v1', revision: 1 } }),
    );
    expect(prepared.mock.invocationCallOrder[0]).toBeLessThan(
      attached.mock.invocationCallOrder[0]!,
    );
  });

  it('publishes candidate-specific disk and memory estimates to trusted chooser UI', async () => {
    const value = setup();
    value.prompts.on('prompt', (prompt) => {
      expect(prompt.candidates[0]).toMatchObject({
        estimatedDiskBytes: 100,
        estimatedMemoryBytes: 200,
        blockingReasons: [],
      });
      value.prompts.respondFromTrustedUI({ promptId: prompt.promptId, action: 'deny' });
    });
    await expect(value.sessions.requestSession(modelRequest(), context())).rejects.toThrow(
      'USER_DENIED',
    );
  });

  it('lets an SDK request run a safe unverified exact revision only after user approval', async () => {
    const value = setup();
    value.engine.profiles['org/preferred'] = {
      ...executionProfile('org/preferred'),
      descriptorSchemaVersion: 'oryn-model-descriptor/1',
      capabilityRegistryDigest: 'b'.repeat(64),
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
    };
    value.prompts.on('prompt', (prompt) => {
      expect(prompt.candidates[0]).toMatchObject({
        trustRequired: true,
        executable: true,
        securityAllowed: true,
        blockingReasons: ['CONFORMANCE_REQUIRED'],
      });
      value.prompts.respondFromTrustedUI({
        promptId: prompt.promptId,
        action: 'allow-once',
      });
    });

    const session = await value.sessions.requestSession(modelRequest(), context());

    expect(session.resolution).toMatchObject({
      resolvedModel: exact('org/preferred'),
      trust: 'user-approved-unverified',
      developerCertified: false,
    });
    expect(value.engine.attached).toHaveLength(1);
  });

  it('uses only trusted UI to grant, binds ownership, and rejects another origin', async () => {
    const value = setup();
    value.prompts.on('prompt', (prompt) => {
      value.prompts.respondFromTrustedUI({ promptId: prompt.promptId, action: 'allow-once' });
    });
    const session = await value.sessions.requestSession(modelRequest(), context());
    expect(value.sessions.requireOwned(session.sessionId, context())).toBe(session);
    expect(() =>
      value.sessions.requireOwned(session.sessionId, context({ origin: 'https://other.test' })),
    ).toThrowError(BrowserModelSessionError);
    expect(
      value.permissions.decision({
        origin: context().origin,
        role: 'chat',
        model: exact('org/preferred'),
      }),
    ).toBeUndefined();

    const websitePreload = readFileSync(
      join(import.meta.dirname, '../../preload/website-preload.ts'),
      'utf8',
    );
    expect(websitePreload).not.toContain('permission-respond');
    expect(websitePreload).not.toContain('always-allow');
  });

  it('persists always allow, deny, and revoked states', () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-permissions-'));
    const file = join(root, 'permissions.json');
    const store = new LocalAIPermissionStore(file);
    const key = { origin: 'https://site.test/path', role: 'chat', model: exact('org/preferred') };
    store.recordTrustedDecision(key, 'always-allow');
    expect(new LocalAIPermissionStore(file).decision(key)).toBe('always-allow');
    store.recordTrustedDecision(key, 'deny');
    expect(store.decision(key)).toBe('deny');
    store.revoke(key);
    expect(store.decision(key)).toBe('revoked');
  });

  it.each([
    { origin: 'https://site.test', installed: true },
    { origin: 'https://site.test', installed: false },
    { origin: 'http://127.0.0.1:5189', installed: true },
    { origin: 'http://127.0.0.1:5189', installed: false },
  ])(
    'prompts again after denying at $origin (installed: $installed)',
    async ({ origin, installed }) => {
      const value = setup(['org/preferred'], installed ? ['org/preferred'] : []);
      const owner = context({ origin, frameUrl: `${origin}/app` });
      const key = { origin, role: 'chat', model: exact('org/preferred') };
      let promptCount = 0;
      value.prompts.on('prompt', (prompt) => {
        promptCount += 1;
        value.prompts.respondFromTrustedUI({
          promptId: prompt.promptId,
          action: promptCount === 1 ? 'deny' : 'allow-once',
        });
      });

      await expect(value.sessions.requestSession(modelRequest(), owner)).rejects.toMatchObject({
        code: 'USER_DENIED',
        savedDecision: undefined,
      });
      expect(value.permissions.decision(key)).toBeUndefined();
      expect(value.permissions.list()).toEqual([]);
      expect(value.engine.attached).toEqual([]);
      expect(value.engine.installedRepositories).toEqual(installed ? ['org/preferred'] : []);

      const session = await value.sessions.requestSession(modelRequest(), owner);
      expect(promptCount).toBe(2);
      expect(session.origin).toBe(origin);
      expect(value.engine.attached).toHaveLength(1);
      expect(value.engine.installedRepositories).toEqual(['org/preferred']);
      expect(value.permissions.decision(key)).toBeUndefined();
    },
  );

  it.each(['deny', 'revoked'] as const)(
    'keeps a saved %s blocked until reset, then requires fresh approval',
    async (decision) => {
      const value = setup();
      const key = { origin: context().origin, role: 'chat', model: exact('org/preferred') };
      value.permissions.recordTrustedDecision(key, decision);
      let promptCount = 0;
      value.prompts.on('prompt', (prompt) => {
        promptCount += 1;
        value.prompts.respondFromTrustedUI({
          promptId: prompt.promptId,
          action: promptCount === 1 ? 'deny' : 'allow-once',
        });
      });

      await expect(value.sessions.requestSession(modelRequest(), context())).rejects.toMatchObject({
        code: 'USER_DENIED',
        savedDecision: decision,
      });
      expect(promptCount).toBe(0);
      expect(value.engine.attached).toEqual([]);

      value.permissions.resetBlockedOrigin(key.origin);
      await expect(value.sessions.requestSession(modelRequest(), context())).rejects.toThrow(
        'USER_DENIED',
      );
      expect(promptCount).toBe(1);
      expect(value.engine.attached).toEqual([]);
      await value.sessions.requestSession(modelRequest(), context());
      expect(promptCount).toBe(2);
      expect(value.engine.attached).toHaveLength(1);
    },
  );

  it('persists a blocked-origin reset without changing grants or other origins', () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-permission-reset-'));
    const file = join(root, 'permissions.json');
    const store = new LocalAIPermissionStore(file);
    const denied = { origin: 'https://site.test', role: 'chat', model: exact('org/preferred') };
    const revoked = { ...denied, role: 'transcription' };
    const allowed = { ...denied, role: 'summary' };
    const otherOrigin = { ...denied, origin: 'https://other.test' };
    store.recordTrustedDecision(denied, 'deny');
    store.recordTrustedDecision(revoked, 'revoked');
    store.recordTrustedDecision(allowed, 'always-allow');
    store.recordTrustedDecision(otherOrigin, 'deny');
    const unchanged = store
      .list()
      .filter(
        (record) => ['summary'].includes(record.role) || record.origin === otherOrigin.origin,
      );

    store.resetBlockedOrigin(denied.origin);

    const reopened = new LocalAIPermissionStore(file);
    expect(reopened.decision(denied)).toBeUndefined();
    expect(reopened.decision(revoked)).toBeUndefined();
    expect(reopened.list()).toEqual(unchanged);
  });

  it('keeps loopback Local AI grants memory-only even when always allow is requested', () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-loopback-permissions-'));
    const file = join(root, 'permissions.json');
    const key = {
      origin: 'http://localhost:54321/path',
      role: 'chat',
      model: exact('org/preferred'),
    };
    const store = new LocalAIPermissionStore(file);
    store.recordTrustedDecision(key, 'always-allow');
    expect(store.decision(key)).toBe('allow-once');
    expect(store.consumeOnce(key)).toBe(true);
    expect(new LocalAIPermissionStore(file).decision(key)).toBeUndefined();
  });

  it('releases document sessions on navigation invalidation', async () => {
    const value = setup();
    value.permissions.recordTrustedDecision(
      { origin: context().origin, role: 'chat', model: exact('org/preferred') },
      'always-allow',
    );
    const session = await value.sessions.requestSession(modelRequest(), context());
    value.callerContexts.emit('document-invalidated', {
      tabId: context().tabId,
      webContentsId: context().webContentsId,
      documentId: context().documentId,
    });
    await new Promise((resolve) => setImmediate(resolve));
    expect(value.sessions.snapshot()).toEqual([]);
    expect(value.engine.released).toContain(session.sessionId);
  });

  it('model chooser remembers site choice and pins existing session', async () => {
    const value = setup(['org/preferred', 'other/family']);
    value.prompts.on('prompt', (prompt) => {
      value.prompts.respondFromTrustedUI({
        promptId: prompt.promptId,
        action: 'always-allow',
        selectedModel: exact('other/family'),
        rememberSiteChoice: true,
      });
    });
    const session = await value.sessions.requestSession(modelRequest(), context());
    expect(session.resolution.resolvedModel.repository).toBe('other/family');
    expect(value.preferences.getSite(context().origin, 'chat')?.repository).toBe('other/family');

    value.preferences.setGlobal('chat', exact('org/preferred'));
    expect(session.resolution.resolvedModel.repository).toBe('other/family');
    expect(value.engine.attached[0]?.model.repository).toBe('other/family');
  });

  it('keeps two origins and their differently resolved models isolated', async () => {
    const value = setup(['qwen/requested', 'qwen/installed', 'google/gemma-chat']);
    value.preferences.setGlobal('chat', exact('google/gemma-chat'));
    value.preferences.setSite('https://second.test', 'chat', exact('qwen/installed'));
    const selected: string[] = [];
    value.prompts.on('prompt', (prompt) => {
      selected.push(prompt.resolvedRepository);
      value.prompts.respondFromTrustedUI({ promptId: prompt.promptId, action: 'allow-once' });
    });
    const requestWithAlternative = {
      ...modelRequest('qwen/requested'),
      certifiedModels: [{ provider: 'huggingface' as const, repository: 'qwen/installed' }],
    };
    const firstOwner = context({ origin: 'https://first.test', frameUrl: 'https://first.test/' });
    const secondOwner = context({
      tabId: 'tab_two',
      webContentsId: 2,
      frameId: 20,
      frameUrl: 'https://second.test/',
      origin: 'https://second.test',
      documentId: 'document_two',
    });

    const first = await value.sessions.requestSession(requestWithAlternative, firstOwner);
    const second = await value.sessions.requestSession(requestWithAlternative, secondOwner);

    expect(first.resolution).toMatchObject({
      resolvedModel: exact('google/gemma-chat'),
      developerCertified: false,
      compatibility: 'compatible',
      resolutionReason: 'global-user-preference',
    });
    expect(second.resolution).toMatchObject({
      resolvedModel: exact('qwen/installed'),
      developerCertified: true,
      compatibility: 'certified',
      resolutionReason: 'site-preference',
    });
    expect(selected).toEqual(['google/gemma-chat', 'qwen/installed']);
    expect(value.engine.attached.map((item) => item.model.repository)).toEqual([
      'google/gemma-chat',
      'qwen/installed',
    ]);
    expect(() => value.sessions.requireOwned(first.sessionId, secondOwner)).toThrowError(
      BrowserModelSessionError,
    );
  });

  it('reports resolving, permission, download, load, and ready progress', async () => {
    const value = setup(['org/preferred'], []);
    const stages: string[] = [];
    value.prompts.on('prompt', (prompt) => {
      value.prompts.respondFromTrustedUI({ promptId: prompt.promptId, action: 'allow-once' });
    });
    const session = await value.sessions.requestSession(modelRequest(), context(), {
      operationId: 'model_request_progress',
      onProgress: (progress) => stages.push(progress.stage),
    });
    expect(session.sessionId).toMatch(/^session_/u);
    expect(stages).toEqual(['resolving', 'awaiting-permission', 'downloading', 'loading', 'ready']);
  });

  it('aborts a pending permission request and removes its authority', async () => {
    const value = setup();
    const abort = new AbortController();
    value.prompts.once('prompt', () => abort.abort());
    await expect(
      value.sessions.requestSession(modelRequest(), context(), {
        operationId: 'model_request_abort',
        signal: abort.signal,
      }),
    ).rejects.toThrow('REQUEST_ABORTED');
    expect(value.sessions.snapshot()).toEqual([]);
  });

  it('bounds permission prompt exhaustion per origin', async () => {
    const value = setup();
    const abort = new AbortController();
    const prompted = new Promise<void>((resolve) => value.prompts.once('prompt', () => resolve()));
    const first = value.sessions
      .requestSession(modelRequest(), context(), {
        operationId: 'model_request_first_pending',
        signal: abort.signal,
      })
      .catch((error: unknown) => error);
    await prompted;

    await expect(
      value.sessions.requestSession(modelRequest(), context(), {
        operationId: 'model_request_prompt_exhaustion',
      }),
    ).rejects.toThrow('LOCAL_AI_MODEL_REQUEST_BUSY');

    abort.abort();
    expect(await first).toBeInstanceOf(Error);
  });
});
