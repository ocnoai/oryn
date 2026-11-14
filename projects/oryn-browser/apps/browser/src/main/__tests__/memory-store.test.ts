import { describe, expect, it } from 'vitest';
import { JsonFileStore } from '../data/json-file-store';
import { ProfileDataPaths } from '../data/profile-data-paths';
import { BrowsingHistoryStore } from '../data/history-store';

describe('private memory-backed stores', () => {
  it('uses per-instance validated memory without persisting values or sharing another window', () => {
    const create = () =>
      new JsonFileStore(
        'memory:private/browser-core/test.json',
        () => ({ value: '' }),
        (value): value is { value: string } =>
          typeof value === 'object' &&
          value !== null &&
          'value' in value &&
          typeof value.value === 'string',
      );
    const first = create();
    first.write({ value: 'private-token' });
    expect(first.read()).toEqual({ value: 'private-token' });
    const copy = first.read();
    copy.value = 'mutated';
    expect(first.read().value).toBe('private-token');
    expect(create().read().value).toBe('');
    expect(() => first.write({ value: 12 } as never)).toThrow();
  });
  it('preserves memory paths through nested stores and forgets history across new instances', () => {
    const paths = new ProfileDataPaths('memory:private-session');
    expect(paths.history).toMatch(/^memory:/u);
    const history = new BrowsingHistoryStore(paths.history);
    history.recordVisit('https://private.test/token', 'Private');
    history.flush();
    expect(history.search()).toHaveLength(1);
    expect(new BrowsingHistoryStore(paths.history).search()).toHaveLength(0);
  });
});
