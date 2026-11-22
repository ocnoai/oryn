import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { BrowserDataService } from '../data/browser-data-service';
import { dataForWindow, WindowRegistryStore } from '../data/window-data';
vi.mock('electron', () => ({ dialog: {}, shell: {} }));
describe('window-owned data', () => {
  it('shares profile bookmarks but keeps selected workspace, sidebar and restoration independent', () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-windows-'));
    try {
      const shared = new BrowserDataService(root);
      const registry = new WindowRegistryStore(root);
      const id = registry.add();
      const first = dataForWindow(shared, 'main');
      const second = dataForWindow(shared, id);
      const workspace = first.experience.createWorkspace('Second workspace');
      first.experience.setActiveWorkspace(workspace.workspaceId);
      expect(second.experience.read().activeWorkspaceId).not.toBe(workspace.workspaceId);
      first.experience.updateSidebar({ activeAppId: 'whatsapp' });
      expect(second.experience.read().sidebar.activeAppId).toBeNull();
      first.sessions.saveOpenTabs([{ url: 'https://one.example' }], 0);
      second.sessions.saveOpenTabs([{ url: 'https://two.example' }], 0);
      expect(first.sessions.getRestoreState().openTabs[0]?.url).toBe('https://one.example');
      expect(dataForWindow(shared, id).sessions.getRestoreState().openTabs[0]?.url).toBe(
        'https://two.example',
      );
      first.bookmarks.createBookmark('Shared', 'https://shared.example');
      expect(second.bookmarks.list()).toHaveLength(1);
      registry.remove(id);
      expect(registry.list()).toEqual(['main']);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
