import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { generateKeyPairSync } from 'node:crypto';
import { describe, it, expect, vi } from 'vitest';
import { bundledSnapshot, createRelease, type Snapshot } from '@ocno/oryn-registry';
import type { CandidateExecutionProfile, ResolvedModelRef } from '@ocno/oryn-protocol';
import { RegistryService } from '../model-resolution/registry-service';
import { ModelResolver } from '../model-resolution/model-resolver';
import { ModelPreferenceStore } from '../model-resolution/model-preference-store';

const cpu = 'oryn.transformers-pytorch';
const mlx = 'oryn.mlx-lm';
const model = {
  provider: 'huggingface' as const,
  repository: 'fixture/model',
  revision: 'a'.repeat(40),
  files: ['model.safetensors'],
};

describe('inspected recommendations reach the selection boundary', () => {
  it.each(['architecture', 'feature-version', 'foreign-version', 'unknown', 'measurement', 'gate'])(
    '%s uses one captured snapshot and at most two inspections',
    async (scenario) => {
      const root = mkdtempSync(join(tmpdir(), 'oryn-optimizer-'));
      try {
        const keys = generateKeyPairSync('ed25519');
        const registry = new RegistryService(join(root, 'cache'), {
          offline: true,
          facts: { hardwareFamily: 'apple-silicon' },
          keys: { test: keys.publicKey.export({ type: 'spki', format: 'pem' }).toString() },
        });
        const seed = structuredClone(bundledSnapshot) as Snapshot;
        seed.catalogVersion = 2;
        seed.variants = [];
        seed.rules = [
          {
            id: 'general',
            when: {},
            prefer: { engines: [cpu, mlx] },
            reason: 'Default',
            provenance: seed.rules[0]!.provenance,
          },
          {
            id: 'specific',
            when: scenario.includes('version')
              ? {
                  features: ['streaming'],
                  engineVersion: [scenario === 'foreign-version' ? 'cpu-version' : 'mlx-version'],
                }
              : { architecture: ['Qwen2ForCausalLM'] },
            prefer: { engines: [mlx] },
            reason: 'Inspected preference',
            provenance: seed.rules[0]!.provenance,
          },
        ];
        const release = createRelease(seed, 'test', keys.privateKey);
        registry.cache.accept(release.manifest, release.bytes);
        const capture = vi.spyOn(registry, 'capture');
        const hints: Array<{ snapshotDigest: string; engines: string[] }> = [];
        const inspector = {
          async inspectModel(
            ref: ResolvedModelRef,
            _requirements: unknown,
            hint: { snapshotDigest: string; engines: string[] },
          ) {
            hints.push(structuredClone(hint));
            const selected = ['measurement', 'gate'].includes(scenario) ? cpu : hint.engines[0]!;
            return {
              model: ref,
              exactRevision: ref.revision,
              executable: true,
              executableBackends: ['cpu'],
              requiresCustomCode: false,
              technicalCapabilities: {
                task: 'text-generation',
                inputModalities: ['text'],
                outputModalities: ['text'],
                ...(scenario === 'unknown' ? {} : { architectures: ['Qwen2ForCausalLM'] }),
              },
              runtimePack: {
                packId: selected,
                packVersion: selected === mlx ? 'mlx-version' : 'cpu-version',
                runnerId: 'fixture',
                taskContract: 'chat-v1',
                compatibilityLevel: 'native-declarative',
                manifestDigest: 'f'.repeat(64),
              },
              routingFacts: {
                version: 1,
                offers: [
                  { engine: cpu, engineVersion: 'cpu-version', features: ['streaming'] },
                  { engine: mlx, engineVersion: 'mlx-version', features: ['streaming'] },
                ],
              },
            } satisfies CandidateExecutionProfile;
          },
        };
        const resolver = new ModelResolver(
          inspector,
          new ModelPreferenceStore(join(root, 'preferences')),
          registry,
        );
        const result = await resolver.resolve(
          {
            role: 'chat',
            preferredModel: model,
            selection: { version: 1, intent: 'exact-artifact' },
            requirements: { task: 'text-generation' },
          },
          { origin: 'https://test.example', inventory: [] },
        );
        const preferMlx = !['foreign-version', 'unknown'].includes(scenario);
        expect(hints).toHaveLength(preferMlx ? 2 : 1);
        expect(capture).toHaveBeenCalledTimes(1);
        expect(hints.at(-1)!.engines[0]).toBe(preferMlx ? mlx : cpu);
        expect(result.selected.recommendation!.engines).toEqual(hints.at(-1)!.engines);
        expect(new Set(hints.map((h) => h.snapshotDigest)).size).toBe(1);
        expect(result.resolution.resolvedModel).toEqual(model);
        expect(result.selected.profile.runtimePack!.packId).toBe(
          preferMlx && !['measurement', 'gate'].includes(scenario) ? mlx : cpu,
        );
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    },
  );
});
