import { describe, expect, it } from 'vitest';
import { supportsTaskNegotiation, validateNegotiatedTasks } from '../task-negotiation';
describe('private task negotiation', () => {
  it('preserves old peers and negotiates only supported engine versions', () => {
    for (const version of ['0.1.0', 'garbage', '0.2'])
      expect(supportsTaskNegotiation(version)).toBe(false);
    for (const version of ['0.2.0', '0.2.0-alpha.1', '1.0.0'])
      expect(supportsTaskNegotiation(version)).toBe(true);
  });
  it('rejects unknown tasks, duplicates and extra authority', () => {
    expect(validateNegotiatedTasks({ contracts: [] })).toEqual([]);
    expect(validateNegotiatedTasks({ contracts: ['chat-v2'] })).toEqual(['chat-v2']);
    expect(validateNegotiatedTasks({ contracts: ['image-embeddings-v1'] })).toEqual([
      'image-embeddings-v1',
    ]);
    for (const value of [
      null,
      [],
      { contracts: ['future-task-v9'] },
      { contracts: ['image-embeddings-v1', 'image-embeddings-v1'] },
      { contracts: [], backend: 'mps' },
    ]) {
      expect(() => validateNegotiatedTasks(value)).toThrow();
    }
  });
});
