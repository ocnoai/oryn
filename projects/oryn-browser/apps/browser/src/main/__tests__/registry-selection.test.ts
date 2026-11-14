import { mkdtempSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { generateKeyPairSync } from 'node:crypto';
import { describe, it, expect, vi } from 'vitest';
import { createRelease, bundledSnapshot, type Snapshot, type Variant } from '@ocno/oryn-registry';
import type { CandidateExecutionProfile, ResolvedModelRef } from '@ocno/oryn-protocol';
import { ModelResolver } from '../model-resolution/model-resolver';
import { ModelPreferenceStore } from '../model-resolution/model-preference-store';
import { RegistryService } from '../model-resolution/registry-service';
import { validateModelRequest } from '../model-resolution/model-request';
import { OrynSettingsService } from '../settings/oryn-settings-service';
import { OrynSettingsStore } from '../settings/oryn-settings-store';
import type { EngineSupervisor } from '../engine-supervisor';

const source = {
  provider: 'huggingface' as const,
  repository: 'org/source',
  revision: 'a'.repeat(40),
};
const mlx = {
  provider: 'huggingface' as const,
  repository: 'org/variant',
  revision: 'b'.repeat(40),
  files: ['model.safetensors'],
};
function profile(model: ResolvedModelRef, pack: string): CandidateExecutionProfile {
  return {
    model,
    exactRevision: model.revision,
    executable: true,
    runner: 'runtime-pack',
    executableBackends: ['cpu'],
    requiresCustomCode: false,
    technicalCapabilities: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
    },
    estimatedMemoryBytes: 1000,
    estimatedDiskBytes: 500,
    runtimePack: {
      packId: pack,
      packVersion: '1',
      runnerId: 'runner',
      taskContract: 'chat-v1',
      compatibilityLevel: 'native-declarative',
      manifestDigest: 'f'.repeat(64),
    },
  };
}
function setup(
  lineage: Variant['lineage'] = 'verified',
  facts = { hardwareFamily: 'apple-silicon' },
) {
  const root = mkdtempSync(join(tmpdir(), 'oryn-registry-test-'));
  const keys = generateKeyPairSync('ed25519');
  const fetcher = vi.fn<typeof fetch>();
  const registry = new RegistryService(join(root, 'cache'), {
    keys: { test: keys.publicKey.export({ type: 'spki', format: 'pem' }).toString() },
    facts,
    fetcher,
  });
  const seed = structuredClone(bundledSnapshot) as Snapshot;
  const template = seed.variants[0]!;
  seed.variants = [
    {
      ...template,
      id: 'source',
      checkpoint: { repository: source.repository, revision: source.revision },
      artifact: {
        repository: source.repository,
        revision: source.revision,
        files: ['model.safetensors'],
      },
      files: [{ path: 'model.safetensors', sizeBytes: 500, digest: 'sha256:' + 'c'.repeat(64) }],
      task: 'text-generation',
      dependencies: [],
      relationship: 'original',
      lineage: 'verified',
    },
    {
      ...template,
      id: 'mlx',
      checkpoint: { repository: source.repository, revision: source.revision },
      artifact: { repository: mlx.repository, revision: mlx.revision, files: mlx.files },
      files: [{ path: 'model.safetensors', sizeBytes: 500, digest: 'sha256:' + 'd'.repeat(64) }],
      task: 'text-generation',
      dependencies: [],
      relationship: 'quantization',
      lineage,
    },
  ];
  seed.catalogVersion = 2;
  const release = createRelease(seed, 'test', keys.privateKey);
  registry.cache.accept(release.manifest, release.bytes);
  const inspector = {
    inspectModel: vi.fn(async (model: ResolvedModelRef) =>
      profile(
        model,
        model.repository === mlx.repository ? 'oryn.mlx-lm' : 'oryn.transformers-pytorch',
      ),
    ),
  };
  const preferences = new ModelPreferenceStore(join(root, 'preferences'));
  const resolver = new ModelResolver(inspector, preferences, registry);
  const request = {
    role: 'chat',
    selection: { version: 1, intent: 'source-checkpoint', sourceCheckpoint: source },
    requirements: { task: 'text-generation', outputModalities: ['text'] },
  };
  return { registry, inspector, preferences, resolver, request, fetcher, seed, keys };
}
const context = { origin: 'https://site.example', inventory: [] };
describe('registry model resolution', () => {
  it('source-only selects a verified eligible variant without any registry network request', async () => {
    const f = setup();
    const result = await f.resolver.resolve(f.request, context);
    expect(result.selected.model).toEqual(mlx);
    expect(result.resolution.selection?.kind).toBe('same-checkpoint');
    expect(result.resolution.selection?.confidence).toBe('heuristic');
    expect(f.fetcher).not.toHaveBeenCalled();
  });
  it.each(['declared', 'unknown', 'conflicting'] as const)(
    'does not silently use %s lineage',
    async (lineage) => {
      const f = setup(lineage);
      const result = await f.resolver.resolve(f.request, context);
      expect(result.selected.model.repository).toBe(source.repository);
    },
  );
  it('a preferred artifact remains selected under none, and pins override remembered preferences', async () => {
    const f = setup();
    f.preferences.setGlobal('chat', mlx);
    const pinned = { ...source, files: ['model.safetensors'] };
    const result = await f.resolver.resolve(
      {
        role: 'chat',
        preferredModel: pinned,
        selection: { version: 1, intent: 'exact-artifact' },
        substitution: { automatic: 'compatible' },
      },
      context,
    );
    expect(result.selected.model).toEqual(pinned);
    const explicit = await f.resolver.resolve(
      { role: 'chat', preferredModel: pinned, selection: { version: 1, intent: 'exact-artifact' } },
      { ...context, userOverride: mlx },
    );
    expect(explicit.selected.model).toEqual(mlx);
  });
  it('engine rejection wins over registry preference and a broken candidate does not hide usable offers', async () => {
    const f = setup();
    f.inspector.inspectModel.mockImplementation(async (model) => {
      if (model.repository === mlx.repository) throw new Error('unsupported');
      return profile(model, 'oryn.transformers-pytorch');
    });
    expect((await f.resolver.resolve(f.request, context)).selected.model.repository).toBe(
      source.repository,
    );
  });
  it('comparable local evidence can beat configuration, while missing evidence cannot', async () => {
    const f = setup();
    const evidence = {
      metric: 'decode-tokens-per-second' as const,
      value: 1,
      phase: 'warm-inference' as const,
      comparisonKey: '1'.repeat(64),
      measuredAt: Date.now() - 100,
      expiresAt: Date.now() + 100000,
      sampleCount: 3,
    };
    f.inspector.inspectModel.mockImplementation(async (model) => ({
      ...profile(
        model,
        model.repository === mlx.repository ? 'oryn.mlx-lm' : 'oryn.transformers-pytorch',
      ),
      recommendationEvidence: {
        ...evidence,
        value: model.repository === source.repository ? 100 : 1,
      },
    }));
    const result = await f.resolver.resolve(f.request, context);
    expect(result.selected.model.repository).toBe(source.repository);
    expect(result.resolution.selection?.source).toBe('local-measurements');
    f.inspector.inspectModel.mockImplementation(async (model) => ({
      ...profile(
        model,
        model.repository === mlx.repository ? 'oryn.mlx-lm' : 'oryn.transformers-pytorch',
      ),
      ...(model.repository === source.repository
        ? { recommendationEvidence: { ...evidence, value: 100 } }
        : {}),
    }));
    expect((await f.resolver.resolve(f.request, context)).selected.model).toEqual(mlx);
  });
  it('unlisted Intel hardware uses a disclosed default without gaining an MLX preference', async () => {
    const f = setup('verified', { hardwareFamily: 'x64' });
    const result = await f.resolver.resolve(f.request, context);
    expect(result.selected.recommendation?.engines).toEqual([]);
  });
  it('uses complete exact reference evidence as configuration and falls back on workload mismatch', async () => {
    const f = setup();
    const bound = { runtime: 'a'.repeat(64), hardware: 'b'.repeat(64), workload: 'c'.repeat(64) };
    const seed = { ...f.seed, catalogVersion: 3 };
    seed.evidence = seed.variants.map((v, i) => ({
      id: `reference-${i}`,
      variantId: v.id,
      runtime: bound.runtime,
      hardware: bound.hardware,
      task: 'text-generation',
      workload: { fingerprint: bound.workload },
      phase: 'warm-inference' as const,
      kind: 'reported' as const,
      metric: 'decode-tokens-per-second' as const,
      value: i ? 1 : 100,
      sampleCount: 3,
      measuredAt: new Date(Date.now() - 10000).toISOString(),
      expiresAt: new Date(Date.now() + 100000).toISOString(),
      peakMemoryBytes: 1000,
      provenance: v.provenance[0]!,
    }));
    const release = createRelease(seed, 'test', f.keys.privateKey);
    f.registry.cache.accept(release.manifest, release.bytes);
    f.inspector.inspectModel.mockImplementation(async (model) => ({
      ...profile(
        model,
        model.repository === mlx.repository ? 'oryn.mlx-lm' : 'oryn.transformers-pytorch',
      ),
      recommendationContext: bound,
    }));
    const result = await f.resolver.resolve(f.request, context);
    expect(result.selected.model.repository).toBe(source.repository);
    expect(result.resolution.selection?.source).toBe('active-configuration');
    expect(result.resolution.selection?.confidence).toBe('heuristic');
    bound.workload = 'd'.repeat(64);
    expect((await f.resolver.resolve(f.request, context)).selected.model).toEqual(mlx);
  });
  it('contradictory live parent metadata prevents catalog equivalence', async () => {
    const f = setup();
    f.inspector.inspectModel.mockImplementation(async (model) => ({
      ...profile(
        model,
        model.repository === mlx.repository ? 'oryn.mlx-lm' : 'oryn.transformers-pytorch',
      ),
      lineageDeclarations: ['different/checkpoint'],
    }));
    const result = await f.resolver.resolve(f.request, context);
    expect(result.selected.model.repository).toBe(source.repository);
    expect(result.candidates.find((c) => c.model.repository === mlx.repository)?.lineage).toBe(
      'conflicting',
    );
  });
  it('resolves a floating source checkpoint separately from a preferred variant', async () => {
    const f = setup();
    f.inspector.inspectModel.mockImplementation(async (model) =>
      profile(
        model.repository === source.repository ? { ...model, revision: source.revision } : model,
        model.repository === mlx.repository ? 'oryn.mlx-lm' : 'oryn.transformers-pytorch',
      ),
    );
    const result = await f.resolver.resolve(
      {
        role: 'chat',
        preferredModel: mlx,
        selection: {
          version: 1,
          intent: 'preferred-variant',
          sourceCheckpoint: { ...source, revision: 'main' },
        },
        substitution: { automatic: 'same-checkpoint' },
      },
      context,
    );
    expect(
      result.candidates.find((c) => c.model.repository === mlx.repository)?.sameCheckpoint,
    ).toBe(true);
  });
  it('does not infer equivalence between unlisted file subsets in the same repository', async () => {
    const f = setup();
    const unlisted = { ...source, repository: 'other/source' };
    const alternative = { ...unlisted, files: ['other-weights.safetensors'] };
    f.inspector.inspectModel.mockImplementation(async (model) => ({
      ...profile(model, 'oryn.transformers-pytorch'),
      selectedFiles: (model.files ?? ['model.safetensors']).map((name) => ({
        name,
        sizeBytes: 500,
      })),
    }));
    const result = await f.resolver.resolve(
      {
        ...f.request,
        selection: { version: 1, intent: 'source-checkpoint', sourceCheckpoint: unlisted },
      },
      {
        ...context,
        inventory: [
          { model: alternative, installationState: 'installed', residencyState: 'loaded' },
        ] as never,
      },
    );
    expect(result.selected.model.files).toEqual(['model.safetensors']);
    expect(result.candidates.find((c) => c.sources.includes('installed'))?.sameCheckpoint).toBe(
      false,
    );
  });
  it('recovery-required security state blocks even uncatalogued artifacts', async () => {
    const f = setup();
    f.registry.cache.status = 'recovery-required';
    expect(
      f.registry.isRevoked({ ...source, repository: 'unlisted/model' }, f.registry.capture()),
    ).toBe(true);
    await expect(f.resolver.resolve(f.request, context)).rejects.toThrow();
  });
  it('an explicit alternative preference retains exact files and reinspects before saving', async () => {
    const f = setup();
    const inspectModel = vi.fn(async () => profile(mlx, 'oryn.mlx-lm'));
    const settings = new OrynSettingsService(
      new OrynSettingsStore(join(mkdtempSync(join(tmpdir(), 'oryn-choice-')), 'settings.json')),
      { configureInitialRuntimePolicy: vi.fn(), inspectModel } as unknown as EngineSupervisor,
      f.preferences,
      {} as never,
      {} as never,
      {} as never,
      {} as never,
      {} as never,
    );
    await settings.updateRole({ role: 'chat', model: mlx });
    expect(inspectModel).toHaveBeenCalledWith(mlx);
    expect(f.preferences.getGlobal('chat')).toEqual(mlx);
    inspectModel.mockResolvedValue({
      ...profile(source, 'oryn.transformers-pytorch'),
      executable: false,
    });
    await expect(
      settings.updateRole({ role: 'chat', model: { ...source, files: ['model.safetensors'] } }),
    ).rejects.toThrow();
    expect(f.preferences.getGlobal('chat')).toEqual(mlx);
  });
  it('rejects website runtime commands, unsafe file paths and floating exact pins', () => {
    expect(() =>
      validateModelRequest({ role: 'chat', preferredModel: source, engine: 'mlx' }),
    ).toThrow();
    for (const patch of [{ files: ['../weights'] }, { revision: 'main', files: ['model.gguf'] }])
      expect(() =>
        validateModelRequest({
          role: 'chat',
          preferredModel: { ...source, ...patch },
          selection: { version: 1, intent: 'exact-artifact' },
        }),
      ).toThrow();
  });
});
it('background refresh activates one signed configuration, coalesces checks and can be disabled', async () => {
  const root = mkdtempSync(join(tmpdir(), 'oryn-refresh-'));
  const keys = generateKeyPairSync('ed25519');
  const snapshot = { ...structuredClone(bundledSnapshot), catalogVersion: 2 } as Snapshot;
  const release = createRelease(snapshot, 'test', keys.privateKey);
  const fetcher = vi.fn<typeof fetch>(
    async (url) =>
      new Response(
        String(url).endsWith('current.json')
          ? JSON.stringify(release.manifest)
          : release.bytes.toString(),
      ),
  );
  const registry = new RegistryService(join(root, 'cache'), {
    endpoint: 'https://registry.example/',
    keys: { test: keys.publicKey.export({ type: 'spki', format: 'pem' }).toString() },
    fetcher,
  });
  const captured = registry.capture();
  await Promise.all([registry.refresh(), registry.refresh()]);
  expect(fetcher).toHaveBeenCalledTimes(2);
  expect(captured.catalogVersion).toBe(1);
  expect(registry.capture().catalogVersion).toBe(2);
  registry.setEnabled(false);
  await registry.refresh();
  expect(fetcher).toHaveBeenCalledTimes(2);
});
