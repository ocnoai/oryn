import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { OrynSettingsStore } from '../settings/oryn-settings-store';

describe('OrynSettingsStore', () => {
  it('persists validated compute and diagnostics settings', () => {
    const path = join(mkdtempSync(join(tmpdir(), 'oryn-settings-')), 'settings.json');
    const store = new OrynSettingsStore(path);

    expect(store.read().compute).toEqual({
      memoryBudgetBytes: null,
      backgroundJobs: 'allow',
      batteryMode: 'normal',
    });
    store.updateCompute({
      memoryBudgetBytes: 2 * 1024 ** 3,
      backgroundJobs: 'ask',
      batteryMode: 'reduce',
    });
    store.updateDeveloper({ showDiagnostics: true });

    const restored = new OrynSettingsStore(path).read();
    expect(restored.compute.backgroundJobs).toBe('ask');
    expect(restored.compute.memoryBudgetBytes).toBe(2 * 1024 ** 3);
    expect(restored.developer.showDiagnostics).toBe(true);
  });

  it('rejects unsafe or out-of-range settings', () => {
    const store = new OrynSettingsStore(
      join(mkdtempSync(join(tmpdir(), 'oryn-settings-invalid-')), 'settings.json'),
    );
    expect(() => store.updateCompute({ memoryBudgetBytes: 1 })).toThrow('INVALID_COMPUTE_SETTINGS');
    expect(() => store.updateDeveloper({ unknown: true } as never)).toThrow(
      'INVALID_DEVELOPER_SETTINGS',
    );
  });
});
