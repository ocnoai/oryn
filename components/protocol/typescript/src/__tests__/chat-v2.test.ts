import { describe, expect, it } from 'vitest';
import fixtures from '../../../fixtures/task-contracts/chat-v2.json';
import { TASK_CONTRACTS, validateTaskInvocation, validateTaskProfile, validateTaskRequirements, validateTaskResult, type TaskProfile } from '../task-contracts.js';

describe('chat-v2 best-effort thinking', () => {
  for (const [name, raw] of Object.entries(fixtures.profiles)) {
    const profile = raw as unknown as TaskProfile;
    it(`accepts both thinking values for ${name}, without strict parameter constraints`, () => {
      expect(validateTaskProfile(profile).thinking).toBeDefined();
      expect(profile.parameters).not.toHaveProperty('thinking');
      for (const thinking of [true, false]) expect(() => validateTaskInvocation(profile, { messages: [{ role: 'user', content: 'Synthetic question' }], thinking, maxTokens: 4096 })).not.toThrow();
      expect(() => validateTaskInvocation(profile, { messages: [{ role: 'user', content: 'Synthetic question' }], thinking: 'false' })).toThrow();
    });
  }
  for (const entry of fixtures.cases) it(`validates ${entry.name} metadata and usage`, () => {
    expect(() => validateTaskResult(fixtures.profiles.unknown as unknown as TaskProfile, entry.result, entry.input)).not.toThrow();
  });
  it('keeps v1 frozen and rejects hard thinking filters, over-budget or contradictory results', () => {
    expect(TASK_CONTRACTS['chat-v1']!.inputSchema.properties.maxTokens!.maximum).toBe(512);
    expect(TASK_CONTRACTS['chat-v1']!.inputSchema.properties).not.toHaveProperty('thinking');
    expect(() => validateTaskRequirements({ task: 'text-generation', contract: { id: 'chat-v2', revision: 1 }, parameterValues: { thinking: false } })).toThrow();
    const profile = fixtures.profiles.switchable as unknown as TaskProfile;
    const value = fixtures.cases[0]!;
    const wrong = structuredClone(value.result); wrong.value.metadata.thinking.effective = true;
    expect(() => validateTaskResult(profile, wrong, value.input)).toThrow();
    expect(() => validateTaskResult(profile, value.result, { ...value.input, thinking: true })).toThrow();
    expect(() => validateTaskResult(profile, value.result, { ...value.input, maxTokens: 1 })).toThrow();
    expect(() => validateTaskInvocation(profile, { ...value.input, maxTokens: 8193 })).toThrow();
    expect(() => validateTaskProfile({ ...profile, contract: { id: 'chat-v2', revision: 2 } })).toThrow();
    expect(() => validateTaskProfile({ ...profile, thinking: { support: 'fixed', default: null } })).toThrow();
  });
});
