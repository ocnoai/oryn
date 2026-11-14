import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdtempSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  ResolvedModelRef,
} from '@ocno/oryn-protocol';

import { ModelPreferenceStore } from '../model-resolution/model-preference-store';
import { ModelResolver } from '../model-resolution/model-resolver';
import {
  ModelRequestValidationError,
  validateModelRequest,
} from '../model-resolution/model-request';

const hashes = new Map<string, string>();

function exact(repository: string): ResolvedModelRef {
  if (!hashes.has(repository)) {
    const seed = Buffer.from(repository)
      .toString('hex')
      .replaceAll(/[^a-f0-9]/g, 'a');
    hashes.set(repository, `${seed}${'a'.repeat(40)}`.slice(0, 40));
  }
  return { provider: 'huggingface', repository, revision: hashes.get(repository)! };
}

function profile(
  repository: string,
  options: Partial<CandidateExecutionProfile> & {
    capabilities?: CandidateExecutionProfile['technicalCapabilities'];
  } = {},
): CandidateExecutionProfile {
  const capabilities = options.capabilities ?? {
    task: 'text-generation',
    parameterCount: 4_000_000_000,
    contextLength: 32_768,
    inputModalities: ['text'],
    outputModalities: ['text'],
    capabilities: { chat: true, streaming: true },
  };
  return {
    model: exact(repository),
    exactRevision: exact(repository).revision,
    executable: true,
    runner: 'transformers',
    executableBackends: ['cpu'],
    requiresCustomCode: false,
    estimatedDiskBytes: 100,
    estimatedMemoryBytes: 1_000,
    technicalCapabilities: capabilities,
    ...options,
  };
}

function fixtureProfile(repository: string): CandidateExecutionProfile {
  return profile(repository, {
    runner: 'runtime-pack',
    runtimePack: {
      packId: 'oryn.fixture-pack',
      packVersion: '0.1.0',
      runnerId: 'fixture.runner',
      taskContract: 'chat-v1',
      compatibilityLevel: 'native-declarative',
      manifestDigest: 'f'.repeat(64),
      trustedMetadata: {
        license: 'Apache-2.0',
        conformance: { status: 'fixture-only', platforms: ['test-cpu'] },
      },
    },
  });
}

class FakeInspector {
  constructor(
    private readonly profiles: Record<string, CandidateExecutionProfile>,
    private readonly catalog: ResolvedModelRef[] = [],
  ) {}

  async inspectModel(model: ResolvedModelRef): Promise<CandidateExecutionProfile> {
    const value = this.profiles[model.repository];
    if (!value) throw new Error(`Missing fake profile ${model.repository}`);
    return structuredClone(value);
  }

  async matchModelCatalog(): Promise<{ models: ResolvedModelRef[] }> {
    return { models: structuredClone(this.catalog) };
  }
}

function inventory(
  repository: string,
  residencyState: EngineModelInventoryRecord['residencyState'] = 'unloaded',
): EngineModelInventoryRecord {
  return {
    installationId: `install_${repository.replaceAll('/', '_')}`,
    model: exact(repository),
    installationState: 'installed',
    residencyState,
    logicalSizeBytes: 100,
    reclaimableSizeBytes: 100,
    estimatedResidentBytes: 1_000,
    activeSessions: 0,
    runningJobs: 0,
    queuedJobs: 0,
  };
}

function preferences(): ModelPreferenceStore {
  return new ModelPreferenceStore(
    join(mkdtempSync(join(tmpdir(), 'oryn-resolver-')), 'prefs.json'),
  );
}

function request(
  preferredModel: string,
  options: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    role: 'chat',
    preferredModel: { provider: 'huggingface', repository: preferredModel },
    requirements: {
      task: 'text-generation',
      minimumContextLength: 32_768,
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true },
    },
    ...options,
  };
}

describe('ModelRequest validation', () => {
  it('accepts generic modality/capability and parameter requirements', () => {
    const validated = validateModelRequest({
      ...request('org/preferred'),
      requirements: {
        inputModalities: ['text', 'image', 'audio', 'video', 'document'],
        outputModalities: ['text', 'image', 'audio', 'video', 'embedding', 'structured'],
        parameters: { minBillions: 0.5, maxBillions: 8 },
        capabilities: {
          multimodalConversation: true,
          structuredOutput: true,
          audioOutput: true,
        },
      },
    });
    expect(validated.requirements?.inputModalities).toContain('document');
    expect(validated.requirements?.capabilities?.audioOutput).toBe(true);
  });

  it('rejects unknown one-off fields and invalid ranges', () => {
    expect(() =>
      validateModelRequest({
        ...request('org/preferred'),
        requirements: { vision: true },
      }),
    ).toThrow(ModelRequestValidationError);
    expect(() =>
      validateModelRequest({
        ...request('org/preferred'),
        requirements: { parameters: { minBillions: 8, maxBillions: 2 } },
      }),
    ).toThrow('minBillions');
  });

  it('accepts only bounded capability-first requests with a task and output modality', () => {
    expect(
      validateModelRequest({
        role: 'chat',
        requirements: { task: 'text-generation', outputModalities: ['text'] },
      }),
    ).toMatchObject({ substitution: { automatic: 'compatible' } });
    expect(() =>
      validateModelRequest({ role: 'chat', requirements: { task: 'text-generation' } }),
    ).toThrow('outputModalities');
    expect(() =>
      validateModelRequest({
        role: 'chat',
        requirements: { task: 'text-generation', outputModalities: ['text'] },
        certifiedModels: [{ provider: 'huggingface', repository: 'org/model' }],
      }),
    ).toThrow('certifiedModels');
  });
});

describe('ModelResolver', () => {
  it.each([
    { admissionScope: 'huggingface-page' as const },
    { sourceRecognition: { state: 'recognized' as const, admitted: true } },
    {
      capabilities: {
        ...profile('org/reviewed').technicalCapabilities,
        architectures: ['OrynReviewedLfm3ForCausalLM'],
      },
    },
  ])('refuses Browser-only source admission even if its scope flag is missing', async (facts) => {
    const inspector = new FakeInspector({
      'org/reviewed': profile('org/reviewed', facts),
    });
    const resolver = new ModelResolver(inspector, preferences());
    await expect(
      resolver.resolve(request('org/reviewed'), {
        origin: 'https://site.test',
        inventory: [],
      }),
    ).rejects.toThrow();
  });

  it('does not inspect installed models for unrelated tasks', async () => {
    const inspector = new FakeInspector({
      'org/speech': profile('org/speech', {
        capabilities: {
          task: 'automatic-speech-recognition',
          inputModalities: ['audio'],
          outputModalities: ['text'],
          capabilities: { speechRecognition: true },
        },
      }),
    });
    // Inspecting the unrelated image model would fail the entire request.
    const resolver = new ModelResolver(inspector, preferences());
    const result = await resolver.resolve(
      request('org/speech', {
        requirements: {
          task: 'automatic-speech-recognition',
          inputModalities: ['audio'],
          outputModalities: ['text'],
          capabilities: { speechRecognition: true },
        },
      }),
      {
        origin: 'https://site.test',
        inventory: [
          {
            ...inventory('org/image'),
            technicalCapabilities: {
              task: 'text-to-image',
              inputModalities: ['text'],
              outputModalities: ['image'],
              capabilities: {},
            },
          },
        ],
      },
    );
    expect(result.resolution.resolvedModel.repository).toBe('org/speech');
    expect(result.candidates.map((item) => item.model.repository)).toEqual(['org/speech']);
  });

  it('resolves a capability-first request exclusively through the trusted catalog', async () => {
    const catalogModel = exact('org/catalog-chat');
    const resolver = new ModelResolver(
      new FakeInspector({ 'org/catalog-chat': profile('org/catalog-chat') }, [catalogModel]),
      preferences(),
    );
    const result = await resolver.resolve(
      {
        role: 'chat',
        requirements: {
          task: 'text-generation',
          inputModalities: ['text'],
          outputModalities: ['text'],
          capabilities: { chat: true },
        },
      },
      { origin: 'https://capability.example', inventory: [] },
    );

    expect(result.resolution).toMatchObject({
      preferredModel: catalogModel,
      resolvedModel: catalogModel,
      substituted: false,
      resolutionReason: 'capability-match',
    });
    expect(result.selected.sources).toContain('catalog');
  });

  it('allows repository custom-code metadata only through a signed audited adapter', async () => {
    const adapted = profile('org/audited-adapter', {
      requiresCustomCode: true,
      runner: 'runtime-pack',
      runtimePack: {
        packId: 'oryn.adapter',
        packVersion: '1.0.0',
        runnerId: 'adapter.runner',
        taskContract: 'chat-v1',
        compatibilityLevel: 'audited-adapter',
        manifestDigest: 'a'.repeat(64),
      },
    });
    const resolver = new ModelResolver(
      new FakeInspector({ 'org/audited-adapter': adapted }, [adapted.model]),
      preferences(),
    );

    const result = await resolver.resolve(
      {
        role: 'chat',
        requirements: { task: 'text-generation', outputModalities: ['text'] },
      },
      { origin: 'https://adapter.example', inventory: [] },
    );

    expect(result.resolution.security).toBe('allowed');
    expect(result.resolution.resolvedModel.repository).toBe('org/audited-adapter');
  });

  it('selects the executable preferred model', async () => {
    const resolver = new ModelResolver(
      new FakeInspector({ 'org/preferred': profile('org/preferred') }),
      preferences(),
    );
    const result = await resolver.resolve(request('org/preferred'), {
      origin: 'https://example.test/path',
      inventory: [],
    });
    expect(result.resolution.resolvedModel).toEqual(exact('org/preferred'));
    expect(result.resolution.resolutionReason).toBe('preferred-model');
  });

  it('retains certification when the preferred model is also developer-certified', async () => {
    const resolver = new ModelResolver(
      new FakeInspector({ 'org/preferred': profile('org/preferred') }),
      preferences(),
    );
    const result = await resolver.resolve(
      request('org/preferred', {
        certifiedModels: [{ provider: 'huggingface', repository: 'org/preferred' }],
      }),
      { origin: 'https://example.test', inventory: [] },
    );

    expect(result.resolution).toMatchObject({
      resolvedModel: exact('org/preferred'),
      compatibility: 'certified',
      developerCertified: true,
      resolutionReason: 'preferred-model',
    });
    expect(result.selected.sources).toEqual(['preferred', 'certified']);
  });

  it('automatically selects a certified installed substitute', async () => {
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/small': profile('org/small'),
        'org/large': profile('org/large'),
      }),
      preferences(),
    );
    const result = await resolver.resolve(
      request('org/small', {
        certifiedModels: [{ provider: 'huggingface', repository: 'org/large' }],
        substitution: { automatic: 'certified-only' },
      }),
      { origin: 'https://example.test', inventory: [inventory('org/large')] },
    );
    expect(result.resolution.resolvedModel.repository).toBe('org/large');
    expect(result.resolution.compatibility).toBe('certified');
    expect(result.resolution.resolutionReason).toBe('developer-certified-alternative');
  });

  it('reuses an installed compatible cross-family candidate', async () => {
    const resolver = new ModelResolver(
      new FakeInspector({
        'qwen/chat': profile('qwen/chat'),
        'google/gemma': profile('google/gemma'),
      }),
      preferences(),
    );
    const result = await resolver.resolve(
      request('qwen/chat', { substitution: { automatic: 'compatible' } }),
      { origin: 'https://example.test', inventory: [inventory('google/gemma', 'loaded')] },
    );
    expect(result.resolution.resolvedModel.repository).toBe('google/gemma');
    expect(result.resolution.compatibility).toBe('compatible');
    expect(result.resolution.resolutionReason).toBe('already-installed');
  });

  it('never uses a fixture-only model as an automatic production substitute', async () => {
    const unavailable = profile('org/requested', {
      executable: false,
      executableBackends: [],
      blockingReasons: ['RUNNER_UNSUPPORTED'],
    });
    const fixture = fixtureProfile('org/conformance-fixture');
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/requested': unavailable,
        'org/conformance-fixture': fixture,
      }),
      preferences(),
    );

    await expect(
      resolver.resolve(
        request('org/requested', {
          certifiedModels: [{ provider: 'huggingface', repository: 'org/conformance-fixture' }],
          substitution: { automatic: 'compatible' },
        }),
        {
          origin: 'https://production.example',
          inventory: [inventory('org/conformance-fixture', 'loaded')],
        },
      ),
    ).rejects.toMatchObject({ code: 'NO_EXECUTABLE_MODEL' });
  });

  it('allows a fixture-only model when the website requests that exact fixture', async () => {
    const fixture = fixtureProfile('org/conformance-fixture');
    const resolver = new ModelResolver(
      new FakeInspector({ 'org/conformance-fixture': fixture }),
      preferences(),
    );

    const result = await resolver.resolve(request('org/conformance-fixture'), {
      origin: 'https://conformance.example',
      inventory: [inventory('org/conformance-fixture')],
    });

    expect(result.resolution.resolvedModel).toEqual(exact('org/conformance-fixture'));
    expect(result.resolution.resolutionReason).toBe('preferred-model');
  });

  it('falls back to an uncertain production candidate after excluding a certified fixture', async () => {
    const uncertain = profile('org/production-uncertain', {
      capabilities: {
        task: 'text-generation',
        inputModalities: ['text'],
        outputModalities: ['text'],
        capabilities: { chat: 'unknown' },
      },
    });
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/requested': profile('org/requested', {
          executable: false,
          executableBackends: [],
          blockingReasons: ['RUNNER_UNSUPPORTED'],
        }),
        'org/conformance-fixture': fixtureProfile('org/conformance-fixture'),
        'org/production-uncertain': uncertain,
      }),
      preferences(),
    );

    const result = await resolver.resolve(
      request('org/requested', {
        certifiedModels: [{ provider: 'huggingface', repository: 'org/conformance-fixture' }],
        substitution: { automatic: 'compatible' },
      }),
      {
        origin: 'https://production.example',
        inventory: [
          inventory('org/conformance-fixture', 'loaded'),
          inventory('org/production-uncertain'),
        ],
      },
    );

    expect(result.resolution.resolvedModel.repository).toBe('org/production-uncertain');
    expect(result.resolution.compatibility).toBe('uncertain');
  });

  it('skips a remembered fixture preference and selects a production-compatible model', async () => {
    const store = preferences();
    store.setSite('https://production.example', 'chat', {
      provider: 'huggingface',
      repository: 'org/conformance-fixture',
    });
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/requested': profile('org/requested', {
          executable: false,
          executableBackends: [],
          blockingReasons: ['RUNNER_UNSUPPORTED'],
        }),
        'org/conformance-fixture': fixtureProfile('org/conformance-fixture'),
        'org/production-model': profile('org/production-model'),
      }),
      store,
    );

    const result = await resolver.resolve(
      request('org/requested', { substitution: { automatic: 'compatible' } }),
      {
        origin: 'https://production.example/path',
        inventory: [
          inventory('org/conformance-fixture', 'loaded'),
          inventory('org/production-model'),
        ],
      },
    );

    expect(result.resolution.resolvedModel.repository).toBe('org/production-model');
    expect(result.resolution.resolutionReason).toBe('already-installed');
  });

  it('uses an installed cross-family user preference over installed Qwen alternatives', async () => {
    const store = preferences();
    store.setGlobal('chat', { provider: 'huggingface', repository: 'google/gemma-chat' });
    const resolver = new ModelResolver(
      new FakeInspector({
        'qwen/requested': profile('qwen/requested'),
        'qwen/installed': profile('qwen/installed'),
        'google/gemma-chat': profile('google/gemma-chat'),
      }),
      store,
    );
    const result = await resolver.resolve(
      request('qwen/requested', {
        certifiedModels: [{ provider: 'huggingface', repository: 'qwen/installed' }],
        substitution: { automatic: 'compatible' },
      }),
      {
        origin: 'https://chat.example',
        inventory: [inventory('qwen/installed'), inventory('google/gemma-chat', 'loaded')],
      },
    );

    expect(result.resolution).toMatchObject({
      preferredModel: exact('qwen/requested'),
      resolvedModel: exact('google/gemma-chat'),
      compatibility: 'compatible',
      developerCertified: false,
      substituted: true,
      resolutionReason: 'global-user-preference',
      downloadRequired: false,
    });
    expect(result.selected.sources).toContain('global-preference');
  });

  it('ranks a preferred model family ahead of other compatible installed models', async () => {
    const store = preferences();
    store.setGlobalFamily('chat', 'gemma');
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/requested': profile('org/requested'),
        'qwen/installed': profile('qwen/installed'),
        'google/gemma-chat': profile('google/gemma-chat'),
      }),
      store,
    );
    const result = await resolver.resolve(
      request('org/requested', { substitution: { automatic: 'compatible' } }),
      {
        origin: 'https://chat.example',
        inventory: [inventory('qwen/installed', 'loaded'), inventory('google/gemma-chat')],
      },
    );

    expect(result.resolution.resolvedModel.repository).toBe('google/gemma-chat');
    expect(result.resolution.resolutionReason).toBe('global-user-preference');
  });

  it('does not automatically substitute when developer policy is none', async () => {
    const resolver = new ModelResolver(
      new FakeInspector({
        'qwen/chat': profile('qwen/chat'),
        'google/gemma': profile('google/gemma'),
      }),
      preferences(),
    );
    const result = await resolver.resolve(request('qwen/chat'), {
      origin: 'https://example.test',
      inventory: [inventory('google/gemma', 'loaded')],
    });
    expect(result.resolution.resolvedModel.repository).toBe('qwen/chat');
  });

  it('classifies unverifiable requirements as uncertain', async () => {
    const uncertain = profile('org/uncertain', {
      capabilities: {
        task: 'text-generation',
        inputModalities: ['text'],
        outputModalities: ['text'],
        capabilities: { chat: 'unknown' },
      },
    });
    const resolver = new ModelResolver(
      new FakeInspector({ 'org/uncertain': uncertain }),
      preferences(),
    );
    const result = await resolver.resolve(request('org/uncertain'), {
      origin: 'https://example.test',
      inventory: [],
    });
    expect(result.resolution.compatibility).toBe('uncertain');
  });

  it('allows an explicit incompatible cross-family override', async () => {
    const short = profile('google/short-context', {
      capabilities: {
        task: 'text-generation',
        contextLength: 8_192,
        inputModalities: ['text'],
        outputModalities: ['text'],
        capabilities: { chat: true },
      },
    });
    const resolver = new ModelResolver(
      new FakeInspector({ 'qwen/chat': profile('qwen/chat'), 'google/short-context': short }),
      preferences(),
    );
    const result = await resolver.resolve(request('qwen/chat'), {
      origin: 'https://example.test',
      inventory: [inventory('google/short-context')],
      userOverride: { provider: 'huggingface', repository: 'google/short-context' },
    });
    expect(result.resolution.compatibility).toBe('incompatible');
    expect(result.resolution.overriddenByUser).toBe(true);
    expect(result.resolution.resolutionReason).toBe('user-override');
  });

  it('blocks runtime-impossible and custom-code overrides', async () => {
    const unavailable = profile('org/unavailable', {
      executable: false,
      executableBackends: [],
      blockingReasons: ['RUNNER_UNSUPPORTED'],
    });
    const custom = profile('org/custom', {
      executable: false,
      executableBackends: [],
      requiresCustomCode: true,
      blockingReasons: ['CUSTOM_CODE_REQUIRED'],
    });
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/preferred': profile('org/preferred'),
        'org/unavailable': unavailable,
        'org/custom': custom,
      }),
      preferences(),
    );
    await expect(
      resolver.resolve(request('org/preferred'), {
        origin: 'https://example.test',
        inventory: [],
        userOverride: { provider: 'huggingface', repository: 'org/unavailable' },
      }),
    ).rejects.toMatchObject({ code: 'MODEL_NOT_EXECUTABLE' });
    await expect(
      resolver.resolve(request('org/preferred'), {
        origin: 'https://example.test',
        inventory: [],
        userOverride: { provider: 'huggingface', repository: 'org/custom' },
      }),
    ).rejects.toMatchObject({ code: 'MODEL_SECURITY_BLOCKED' });
  });

  it('persists site/global preferences independently from model deletion', async () => {
    const file = join(mkdtempSync(join(tmpdir(), 'oryn-preference-')), 'prefs.json');
    const store = new ModelPreferenceStore(file);
    store.setGlobal('chat', { provider: 'huggingface', repository: 'google/gemma' });
    store.setSite('https://site.test/path', 'chat', {
      provider: 'huggingface',
      repository: 'site/model',
    });
    const restored = new ModelPreferenceStore(file);
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/preferred': profile('org/preferred'),
        'google/gemma': profile('google/gemma'),
        'site/model': profile('site/model'),
      }),
      restored,
    );
    const siteResult = await resolver.resolve(request('org/preferred'), {
      origin: 'https://site.test',
      inventory: [],
    });
    expect(siteResult.resolution.resolutionReason).toBe('site-preference');
    expect(siteResult.resolution.downloadRequired).toBe(true);
    restored.clearSite('https://site.test', 'chat');
    const globalResult = await resolver.resolve(request('org/preferred'), {
      origin: 'https://site.test',
      inventory: [],
    });
    expect(globalResult.resolution.resolvedModel.repository).toBe('google/gemma');
    expect(globalResult.resolution.resolutionReason).toBe('global-user-preference');
    expect(globalResult.resolution.downloadRequired).toBe(true);
  });

  it('uses factual quality, speed, and memory ranking signals', async () => {
    const store = preferences();
    store.setRanking({ preferInstalled: false, optimization: 'quality' });
    const small = profile('org/small', {
      estimatedMemoryBytes: 500,
      capabilities: {
        task: 'text-generation',
        parameterCount: 1_000_000_000,
        contextLength: 32_768,
        inputModalities: ['text'],
        outputModalities: ['text'],
        capabilities: { chat: true },
      },
    });
    const large = profile('org/large', {
      estimatedMemoryBytes: 2_000,
      capabilities: {
        ...small.technicalCapabilities,
        parameterCount: 8_000_000_000,
      },
    });
    const resolver = new ModelResolver(
      new FakeInspector({ 'org/small': small, 'org/large': large }),
      store,
    );
    const modelRequest = request('org/small', {
      certifiedModels: [
        { provider: 'huggingface', repository: 'org/small' },
        { provider: 'huggingface', repository: 'org/large' },
      ],
      substitution: { automatic: 'compatible' },
    });
    expect(
      (await resolver.resolve(modelRequest, { origin: 'https://site.test', inventory: [] }))
        .resolution.resolvedModel.repository,
    ).toBe('org/large');
    store.setRanking({ optimization: 'memory' });
    expect(
      (await resolver.resolve(modelRequest, { origin: 'https://site.test', inventory: [] }))
        .resolution.resolvedModel.repository,
    ).toBe('org/small');
    store.setRanking({ optimization: 'speed' });
    expect(store.optimization).toBe('speed');
  });
});

describe('task contracts preserve user model choice', () => {
  const common = {
    kind: 'task-profile' as const,
    schemaVersion: 1 as const,
    profileId: 'captions.native',
    contract: { id: 'transcription-v1', revision: 1 },
    task: 'automatic-speech-recognition',
    inputModalities: ['audio' as const],
    outputModalities: ['text' as const],
    inputSchemaRef: 'taskInput.transcription-v1',
    resultSchemaRef: 'taskResult.transcription-v1',
    features: { wordTimestamps: 'supported' as const },
    parameters: {
      timestamps: {
        inputPointer: '/timestamps',
        valueSchema: { type: 'boolean' },
        default: false,
        handling: 'honored' as const,
        editable: true,
        affects: ['result' as const],
      },
      language: {
        inputPointer: '/language',
        valueSchema: { type: 'string', minLength: 1, maxLength: 32 },
        default: 'auto',
        handling: 'honored' as const,
        editable: true,
        affects: ['result' as const],
      },
    },
    limits: {},
    streamingChannels: [],
    extensions: {},
    provenance: 'adapter-derived' as const,
  };
  const captionRequest = {
    role: 'captions',
    preferredModel: { provider: 'huggingface', repository: 'org/first' },
    requirements: {
      task: 'automatic-speech-recognition',
      contract: common.contract,
      requiredFeatures: ['wordTimestamps'],
      parameterValues: { timestamps: true, language: 'sk' },
    },
    substitution: { automatic: 'none' },
  };
  function asr(repository: string, timed = true) {
    return profile(repository, {
      taskProfile: { ...common, features: { wordTimestamps: timed ? 'supported' : 'unsupported' } },
      technicalCapabilities: {
        task: 'automatic-speech-recognition',
        inputModalities: ['audio'],
        outputModalities: ['text'],
      },
    });
  }
  it('allows an explicit compatible replacement even when automatic substitution is disabled', async () => {
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/first': asr('org/first'),
        'org/replacement': asr('org/replacement'),
      }),
      preferences(),
    );
    const result = await resolver.resolve(captionRequest, {
      origin: 'https://captions.test',
      inventory: [],
      userOverride: { provider: 'huggingface', repository: 'org/replacement' },
    });
    expect(result.resolution.resolvedModel.repository).toBe('org/replacement');
    expect(result.resolution.overriddenByUser).toBe(true);
    expect(result.selected.profile.taskProfile?.contract).toEqual(common.contract);
  });
  it('rejects a same-task replacement lacking required word timestamps', async () => {
    const resolver = new ModelResolver(
      new FakeInspector({
        'org/first': asr('org/first'),
        'org/replacement': asr('org/replacement', false),
      }),
      preferences(),
    );
    await expect(
      resolver.resolve(captionRequest, {
        origin: 'https://captions.test',
        inventory: [],
        userOverride: { provider: 'huggingface', repository: 'org/replacement' },
      }),
    ).rejects.toThrow();
  });
  it('rejects contradictory timestamp requirements before inspection', () => {
    expect(() =>
      validateModelRequest({
        ...captionRequest,
        requirements: { ...captionRequest.requirements, parameterValues: { timestamps: false } },
      }),
    ).toThrow();
  });
});
