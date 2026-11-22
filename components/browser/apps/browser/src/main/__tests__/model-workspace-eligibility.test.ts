import {
  assertCandidateExecutionProfile,
  type CandidateExecutionProfile,
} from '@ocno/oryn-protocol';
import { describe, expect, it } from 'vitest';

import {
  isOrynVerified,
  isUserTrustEligible,
  taskContractForProfile,
} from '../model-workspace/model-workspace-eligibility';
import unverifiedChatProfiles from './fixtures/unverified-chat-profiles.json';

function effectiveChat(): CandidateExecutionProfile {
  const profile: unknown = structuredClone(unverifiedChatProfiles[0]);
  assertCandidateExecutionProfile(profile);
  return profile;
}

function genericChat(): CandidateExecutionProfile {
  const revision = 'a'.repeat(40);
  return {
    model: { provider: 'huggingface', repository: 'Qwen/Qwen3.5-4B', revision },
    exactRevision: revision,
    executable: true,
    runner: 'transformers',
    executableBackends: ['mps', 'cpu'],
    requiresCustomCode: false,
    estimatedDiskBytes: 10_000,
    estimatedMemoryBytes: 12_500,
    technicalCapabilities: {
      task: 'text-generation',
      inputModalities: ['text'],
      outputModalities: ['text'],
      capabilities: { chat: true, streaming: true },
    },
  };
}

describe('model workspace trust eligibility', () => {
  // Captured from frozen metadata replay, assuming bundled runtimes and sufficient resources.
  // These are inspection profiles, not model-execution or conformance evidence.
  it.each(unverifiedChatProfiles.map((profile) => [profile.model.repository, profile] as const))(
    'uses the effective task contract for the unverified model %s',
    (_repository, captured) => {
      const profile: unknown = structuredClone(captured);
      assertCandidateExecutionProfile(profile);
      expect(profile.technicalCapabilities.capabilities?.chat).toBe('unknown');
      expect(taskContractForProfile(profile)).toBe('chat-v1');
      expect(isUserTrustEligible(profile)).toBe(true);
      expect(isOrynVerified(profile)).toBe(false);
    },
  );

  it.each([
    ['unknown contract', { contract: { id: 'repository-task-v1', revision: 1 } }],
    ['unsupported revision', { contract: { id: 'chat-v1', revision: 2 } }],
    ['wrong task', { task: 'automatic-speech-recognition' }],
    ['wrong schema', { inputSchemaRef: 'taskInput.repository-task-v1' }],
    ['invalid shape', { kind: 'repository-task' }],
    [
      'widened controls',
      {
        parameters: {
          maxTokens: {
            ...effectiveChat().taskProfile!.parameters.maxTokens,
            valueSchema: { type: 'integer', minimum: 1, maximum: 100_000 },
          },
        },
      },
    ],
  ])(
    'rejects a present invalid effective profile without legacy fallback: %s',
    (_label, change) => {
      const profile = effectiveChat();
      profile.technicalCapabilities.capabilities!.chat = true;
      Object.assign(profile.taskProfile!, change);
      expect(taskContractForProfile(profile)).toBeUndefined();
      expect(isUserTrustEligible(profile)).toBe(false);
    },
  );

  it.each([
    'CUSTOM_CODE_REQUIRED',
    'UNSAFE_MODEL_FORMAT',
    'MODEL_SUBSET_INCOMPLETE',
    'MODEL_ACCESS_RESTRICTED',
    'RUNTIME_CAPABILITY_UNSUPPORTED',
    'TASK_ADAPTER_UNAVAILABLE',
    'MODEL_MEMORY_INSUFFICIENT',
    'CONFORMANCE_FAILED',
    'RUNTIME_INTEGRITY_FAILED',
  ])('keeps %s non-overridable with a valid effective profile', (reason) => {
    const profile = effectiveChat();
    profile.blockingReasons!.push(reason);
    expect(taskContractForProfile(profile)).toBe('chat-v1');
    expect(isUserTrustEligible(profile)).toBe(false);
    expect(isOrynVerified(profile)).toBe(false);
  });

  it.each([
    { requiresCustomCode: true },
    { executable: false },
    { loadDisposition: 'unsupported' as const },
  ])('keeps the sealed execution and trust checks with an effective profile: %o', (change) => {
    expect(isUserTrustEligible({ ...effectiveChat(), ...change })).toBe(false);
  });

  it('maps a safe bundled Transformers chat profile without using repository identity', () => {
    const profile = genericChat();
    expect(taskContractForProfile(profile)).toBe('chat-v1');
    expect(isUserTrustEligible(profile)).toBe(true);
    expect(isOrynVerified(profile)).toBe(false);
  });

  it('allows explicit trust for a safe untested compatibility-plane match only', () => {
    const profile: CandidateExecutionProfile = {
      ...genericChat(),
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

    expect(isUserTrustEligible(profile)).toBe(true);
    expect(isOrynVerified(profile)).toBe(false);
    expect(
      isUserTrustEligible({
        ...profile,
        blockingReasons: ['CONFORMANCE_REQUIRED', 'UNSAFE_MODEL_FORMAT'],
      }),
    ).toBe(false);
  });

  it.each([
    { requiresCustomCode: true },
    { executable: false, blockingReasons: ['UNSAFE_MODEL_FORMAT'] },
    { runner: 'runtime-pack' },
  ])('keeps hard security and unknown execution paths non-overridable: %o', (change) => {
    expect(isUserTrustEligible({ ...genericChat(), ...change })).toBe(false);
  });

  it('recognizes a non-fixture signed exact contract as Oryn verified', () => {
    const profile: CandidateExecutionProfile = {
      ...genericChat(),
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
          conformance: { status: 'passed', platforms: ['macos-arm64'] },
        },
      },
    };
    expect(isOrynVerified(profile)).toBe(true);
    expect(isUserTrustEligible(profile)).toBe(false);
    profile.taskProfile = effectiveChat().taskProfile;
    expect(taskContractForProfile(profile)).toBe('chat-v1');
    expect(isOrynVerified(profile)).toBe(true);

    profile.runtimePack!.taskContract = 'transcription-v1';
    expect(taskContractForProfile(profile)).toBeUndefined();
    expect(isOrynVerified(profile)).toBe(false);
    expect(isUserTrustEligible(profile)).toBe(false);

    profile.runtimePack!.taskContract = 'chat-v1';
    profile.taskProfile!.contract.revision = 2;
    expect(taskContractForProfile(profile)).toBeUndefined();
    expect(isOrynVerified(profile)).toBe(false);
    expect(isUserTrustEligible(profile)).toBe(false);
  });

  it('rejects unknown or missing signed contracts without granting authority', () => {
    const profile = genericChat();
    profile.runtimePack = {
      packId: 'oryn.chat',
      packVersion: '1.0.0',
      runnerId: 'chat',
      taskContract: 'repository-task-v1',
      compatibilityLevel: 'native-declarative',
      manifestDigest: 'b'.repeat(64),
    };
    expect(taskContractForProfile(profile)).toBeUndefined();
    expect(isUserTrustEligible(profile)).toBe(false);
    expect(isOrynVerified(profile)).toBe(false);

    profile.runner = 'runtime-pack';
    Reflect.deleteProperty(profile.runtimePack, 'taskContract');
    expect(taskContractForProfile(profile)).toBeUndefined();
    expect(isUserTrustEligible(profile)).toBe(false);
    expect(isOrynVerified(profile)).toBe(false);
  });
});
