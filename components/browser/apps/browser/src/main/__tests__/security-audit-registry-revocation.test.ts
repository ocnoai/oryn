// Execute only inside the parent's verified sandbox. No live endpoint, model download or load.
// The registry implementation/signatures/cache are real. Engine profiles below are SIMULATED.
import { expect, it } from 'vitest';
import { generateKeyPairSync } from 'node:crypto';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { bundledSnapshot, createRelease, digest, type Snapshot } from '@ocno/oryn-registry';
import { RegistryService } from '../model-resolution/registry-service';
import { ModelResolver } from '../model-resolution/model-resolver';
import { ModelPreferenceStore } from '../model-resolution/model-preference-store';

const model = {
  provider: 'huggingface' as const,
  repository: 'audit-fixture/dummy-model',
  revision: 'a'.repeat(40),
};
const artifact = { repository: model.repository, revision: model.revision, files: ['dummy.gguf'] };
const pinned = { ...model, files: artifact.files };
const context = { origin: 'https://owner.invalid', inventory: [] };
function simulatedProfile(requested: typeof model & { files?: string[] }) {
  return {
    model: { ...model, ...(requested.files ? { files: [...requested.files] } : {}) },
    exactRevision: model.revision,
    executable: true,
    runner: 'runtime-pack' as const,
    executableBackends: ['cpu'],
    requiresCustomCode: false,
    selectedFiles: [{ name: 'dummy.gguf', sizeBytes: 8 }],
    estimatedDiskBytes: 8,
    estimatedMemoryBytes: 128,
    technicalCapabilities: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true },
    },
    runtimePack: {
      packId: 'oryn.llama-cpp',
      packVersion: '1',
      runnerId: 'fixture.runner',
      taskContract: 'chat-v1',
      compatibilityLevel: 'native-declarative',
      manifestDigest: 'f'.repeat(64),
    },
  };
}

it('rejects revoked selected artifacts after their catalog variant is removed', async () => {
  const root = mkdtempSync(join(tmpdir(), 'registry-fixture-'));
  const path = join(root, 'cache.json');
  const { publicKey, privateKey } = generateKeyPairSync('ed25519');
  let fetchCalls = 0;
  const options = {
    endpoint: 'https://registry.invalid/',
    keys: { fixture: publicKey.export({ format: 'pem', type: 'spki' }).toString() },
    offline: true,
    facts: { hardwareFamily: 'fixture', memoryBytes: 1024 ** 3 },
    fetcher: async () => {
      fetchCalls++;
      throw new Error('OFFLINE_FIXTURE_NO_FETCH');
    },
  };
  const registry = new RegistryService(path, options);
  const seed = structuredClone(bundledSnapshot) as unknown as Snapshot;
  seed.catalogVersion = bundledSnapshot.catalogVersion + 1;
  seed.evidence = [];
  seed.variants = [
    {
      ...seed.variants[0]!,
      id: 'dummy-fixture',
      checkpoint: { repository: model.repository, revision: model.revision },
      artifact,
      files: [{ path: 'dummy.gguf', sizeBytes: 8, digest: 'sha256:' + 'b'.repeat(64) }],
      representation: 'gguf',
      task: 'text-generation',
      dependencies: [],
      parents: [],
      relationship: 'original',
      lineage: 'verified',
    },
  ];
  seed.revokedArtifacts = [digest(artifact)];
  const first = createRelease(seed, 'fixture', privateKey);
  expect(first.bytes.length).toBeLessThan(2 * 1024 ** 2);
  registry.cache.accept(first.manifest, first.bytes);
  expect(registry.cache.revoked(artifact)).toBe(true);
  expect(registry.isRevoked(model, registry.capture())).toBe(true);
  expect(registry.isRevoked(pinned, registry.capture())).toBe(true);
  const inspector = {
    inspectModel: async (request: typeof model & { files?: string[] }) => simulatedProfile(request),
  };
  const preferences = new ModelPreferenceStore(join(root, 'preferences.json'));
  const retainedResolver = new ModelResolver(inspector as never, preferences, registry);
  const request = {
    role: 'chat',
    preferredModel: model,
    requirements: { task: 'text-generation', outputModalities: ['text'] },
  };
  await expect(retainedResolver.resolve(request, context)).rejects.toThrow();

  const removed = structuredClone(seed);
  removed.catalogVersion++;
  removed.variants = [];
  const second = createRelease(removed, 'fixture', privateKey);
  registry.cache.accept(second.manifest, second.bytes);
  const restored = new RegistryService(path, options);
  const snapshot = restored.capture();
  expect(snapshot.catalogVersion).toBe(removed.catalogVersion);
  expect(snapshot.variants).toHaveLength(0);
  expect(snapshot.revokedArtifacts).toContain(digest(artifact));
  expect(restored.cache.revoked(artifact)).toBe(true);
  expect(restored.isRevoked(model, snapshot)).toBe(false);
  expect(restored.isRevoked(pinned, snapshot)).toBe(true);
  const resolver = new ModelResolver(inspector as never, preferences, restored);
  await expect(resolver.resolve(request, context)).rejects.toThrow();
  await expect(
    resolver.resolve(
      {
        role: 'chat',
        selection: { version: 1, intent: 'source-checkpoint', sourceCheckpoint: model },
        requirements: request.requirements,
      },
      context,
    ),
  ).rejects.toThrow();
  const ordinaryInspector = {
    inspectModel: async () => ({
      ...simulatedProfile(model),
      selectedFiles: [{ name: 'ordinary.gguf', sizeBytes: 8 }],
    }),
  };
  const ordinaryResolver = new ModelResolver(ordinaryInspector as never, preferences, restored);
  expect((await ordinaryResolver.resolve(request, context)).selected.executability).toBe(
    'executable',
  );
  await expect(resolver.resolve({ ...request, preferredModel: pinned }, context)).rejects.toThrow();

  writeFileSync(path + '.security', '{damaged-fixture');
  const recovery = new RegistryService(path, options);
  expect(recovery.cache.status).toBe('recovery-required');
  expect(recovery.isRevoked(model, recovery.capture())).toBe(true);
  expect(recovery.isRevoked(pinned, recovery.capture())).toBe(true);
  expect(fetchCalls).toBe(0);
});
