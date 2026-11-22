import { EventEmitter } from 'node:events';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { BrowserWindow, WebContentsView } from 'electron';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BrowserDataService } from '../data/browser-data-service';
import { BrowserCallerContextService } from '../browser-caller-context';
import { TabManager } from '../tab-manager';
import { createOrynModelWorkspaceURL, ORYN_NEW_TAB_URL } from '../../shared/browser-urls';

vi.mock('electron', () => ({
  dialog: { showMessageBoxSync: () => 0 },
  WebContentsView: class {
    webContents = new (class extends EventEmitter {
      id = Math.random();
      url = '';
      destroyed = false;
      mainFrame = { url: '', processId: 1, routingId: 1 };
      navigationHistory = { canGoBack: () => false, canGoForward: () => false };
      session = {
        fetch: vi.fn(),
        setPermissionRequestHandler: vi.fn(),
        setPermissionCheckHandler: vi.fn(),
        on: vi.fn(),
      };
      getURL = () => this.url;
      getTitle = () => 'Fixture';
      isLoading = () => false;
      isDestroyed = () => this.destroyed;
      setZoomFactor = vi.fn();
      setWindowOpenHandler = vi.fn();
      loadURL = vi.fn(async (url: string) => {
        this.emit('did-start-navigation', {}, url, false, true);
        this.url = this.mainFrame.url = url;
        this.emit('did-navigate', {}, url);
      });
      close = vi.fn(() => {
        this.destroyed = true;
        this.emit('destroyed');
      });
    })();
    setBounds = vi.fn();
  },
}));

const directories: string[] = [];
afterEach(() => {
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
});
function setup(profilePath?: string) {
  const path = profilePath ?? mkdtempSync(join(tmpdir(), 'oryn-s60-tabs-'));
  if (!profilePath) directories.push(path);
  const data = new BrowserDataService(path);
  const contexts = new BrowserCallerContextService();
  const manager = new TabManager(
    {
      isDestroyed: () => false,
      contentView: { addChildView: vi.fn(), removeChildView: vi.fn() },
    } as unknown as BrowserWindow,
    contexts,
    data,
    { websitePreloadPath: '', workspacePreloadPath: '' },
  );
  return { manager, data, contexts, path };
}

const favicon = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg"/%3E';

describe('open-tab favicon cache', () => {
  it('restores saved icons before inactive pages load and retains them during activation', async () => {
    const first = setup();
    const active = first.manager.createTab('https://example.test/active');
    const background = first.manager.createTab('https://example.test/background', false);
    for (const tab of [active, background])
      tab.view.webContents.emit('page-favicon-updated', {}, [favicon]);
    await vi.waitFor(() => {
      expect(first.data.sessions.getRestoreState().openTabs.map((tab) => tab.favicon)).toEqual([
        favicon,
        favicon,
      ]);
    });
    // App shutdown preserves the open-tab snapshot for the next launch.
    await first.manager.closeAll();
    const restarted = setup(first.path);
    const saved = restarted.data.sessions.getRestoreState();
    restarted.manager.batch(() => {
      saved.openTabs.forEach((tab, index) =>
        restarted.manager.createTab(tab.url, index === saved.activeIndex, {
          ...tab,
          lazy: index !== saved.activeIndex,
        }),
      );
    });
    expect(restarted.manager.getTab(active.tabId).favicon).toBe(favicon);
    const restored = restarted.manager.getTab(background.tabId);
    expect(restored.favicon).toBe(favicon);
    expect(restored.view.webContents.loadURL).not.toHaveBeenCalled();
    expect(restored.view.webContents.session.fetch).not.toHaveBeenCalled();
    restarted.manager.activateTab(restored.tabId);
    expect(restarted.manager.getTab(restored.tabId).favicon).toBe(favicon);
    expect(restored.view.webContents.loadURL).toHaveBeenCalledOnce();
  });

  it('removes closed-tab images from disk without removing another open tab’s icon', async () => {
    const { manager, data } = setup();
    const first = manager.createTab('https://example.test/', true, { favicon });
    const second = manager.createTab('https://example.test/', false, { favicon, lazy: true });
    await manager.closeTab(second.tabId);
    expect(data.sessions.getRestoreState().openTabs).toEqual([
      expect.objectContaining({ tabId: first.tabId, favicon }),
    ]);
    expect(data.sessions.listRecentlyClosed().every((tab) => tab.favicon === undefined)).toBe(true);
    await manager.closeTab(first.tabId);
    expect(manager.getTabs()[0]?.url).toBe(ORYN_NEW_TAB_URL);
    expect(readFileSync(data.paths.session, 'utf8')).not.toContain('data:image/');
  });

  it('drops an old icon on navigation and persists the replacement', async () => {
    const { manager, data } = setup();
    const tab = manager.createTab('https://example.test/', true, { favicon });
    manager.navigate('https://other.test/');
    expect(manager.getTab(tab.tabId).favicon).toBeUndefined();
    expect(data.sessions.getRestoreState().openTabs[0]?.favicon).toBeUndefined();
    const replacement = 'data:image/png;base64,iVBORw0KGgo=';
    tab.view.webContents.emit('page-favicon-updated', {}, [replacement]);
    await vi.waitFor(() =>
      expect(data.sessions.getRestoreState().openTabs[0]?.favicon).toBe(replacement),
    );
  });

  it('does not restore an icon when an outstanding request finishes after closing', async () => {
    const { manager, data } = setup();
    manager.createTab();
    const tab = manager.createTab('https://example.test/', false);
    let finish!: (response: Response) => void;
    vi.mocked(tab.view.webContents.session.fetch).mockReturnValue(
      new Promise<Response>((resolve) => {
        finish = resolve;
      }),
    );
    tab.view.webContents.emit('page-favicon-updated', {}, ['https://example.test/icon.png']);
    const options = vi.mocked(tab.view.webContents.session.fetch).mock.calls[0]?.[1];
    await manager.closeTab(tab.tabId);
    expect(options?.signal?.aborted).toBe(true);
    finish(new Response('image bytes', { headers: { 'Content-Type': 'image/png' } }));
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(manager.getTabs()).toHaveLength(1);
    expect(readFileSync(data.paths.session, 'utf8')).not.toContain('data:image/');
  });
});

function detachContents(view: WebContentsView, missing: boolean) {
  const contents = view.webContents;
  vi.spyOn(contents, 'isDestroyed').mockReturnValue(true);
  for (const method of ['getURL', 'getTitle', 'isLoading'] as const)
    vi.spyOn(contents, method).mockImplementation(() => {
      throw new Error('Object has been destroyed');
    });
  if (missing) Object.defineProperty(view, 'webContents', { value: undefined });
  return contents;
}

describe('tab teardown state updates', () => {
  it.each([true, false])(
    'omits unavailable pages from UI and saved state (missing contents: %s)',
    (missing) => {
      const { manager, data } = setup();
      const active = manager.createTab('https://example.test/active');
      const closing = manager.createTab('https://example.test/closing', false);
      const contents = detachContents(closing.view, missing);

      // Another live tab can publish before the closing tab's destroyed callback runs.
      expect(() => active.view.webContents.emit('media-paused')).not.toThrow();
      expect(manager.getTabs().map((tab) => tab.tabId)).toEqual([active.tabId]);
      expect(data.sessions.getRestoreState().openTabs.map((tab) => tab.tabId)).toEqual([
        active.tabId,
      ]);
      contents.emit('destroyed');
      expect(data.sessions.listRecentlyClosed()).toHaveLength(1);
    },
  );

  it.each([true, false])(
    'ignores late state events and selects the next live tab (missing contents: %s)',
    (missing) => {
      const { manager, data } = setup();
      const remaining = manager.createTab('https://example.test/remaining');
      const closing = manager.createTab('https://example.test/closing');
      const contents = detachContents(closing.view, missing);
      expect(manager.getActiveTab()).toBeUndefined();
      for (const [event, ...args] of [
        ['media-paused'],
        ['media-started-playing'],
        ['did-start-loading'],
        ['did-stop-loading'],
        ['did-navigate', {}, 'https://example.test/late'],
        ['did-navigate-in-page', {}, 'https://example.test/late', true],
        ['page-title-updated', {}, 'Late title'],
        ['render-process-gone'],
      ] as const)
        expect(() => contents.emit(event, ...args), event).not.toThrow();
      contents.emit('destroyed');
      expect(manager.getActiveTabId()).toBe(remaining.tabId);
      expect(manager.getTabs().map((tab) => tab.tabId)).toEqual([remaining.tabId]);
      expect(data.sessions.listRecentlyClosed()[0]?.url).toBe('https://example.test/closing');
    },
  );

  it('finishes a close after Electron has cleared the view and only records it once', async () => {
    const { manager, data } = setup();
    const tab = manager.createTab('https://example.test/closing');
    const contents = detachContents(tab.view, true);
    await expect(manager.closeTab(tab.tabId)).resolves.toBe(true);
    contents.emit('destroyed');
    expect(manager.getTabs()).toHaveLength(1);
    expect(manager.getTabs()[0]?.url).toBe(ORYN_NEW_TAB_URL);
    expect(data.sessions.listRecentlyClosed()).toHaveLength(1);
  });

  it('destroys all tabs even if one view has already lost its contents', () => {
    const { manager } = setup();
    const tab = manager.createTab('https://example.test/closing');
    manager.createTab('https://example.test/other');
    detachContents(tab.view, true);
    expect(() => manager.destroy()).not.toThrow();
    expect(manager.getTabs()).toEqual([]);
  });

  it('stops a loading callback if a state observer closes its page', () => {
    const { manager } = setup();
    const tab = manager.createTab('https://example.test/closing');
    const contents = tab.view.webContents;
    manager.once('state-changed', () => {
      detachContents(tab.view, true);
      contents.emit('destroyed');
    });
    expect(() => contents.emit('did-stop-loading')).not.toThrow();
    expect(manager.getTabs()[0]?.url).toBe(ORYN_NEW_TAB_URL);
  });
});

describe('Sprint 60 organization and lifetime regressions', () => {
  it('B08 restores deleted organization into Default without consuming a failed reopen', () => {
    const { manager, data } = setup();
    data.sessions.pushRecentlyClosed({
      url: 'https://example.test/',
      workspaceId: 'deleted',
      islandId: 'deleted',
    });
    manager.freezeForRestart();
    expect(() => manager.reopenRecentlyClosed()).toThrow('BROWSER_RESTARTING');
    expect(data.sessions.listRecentlyClosed()).toHaveLength(1);
    manager.cancelRestart();
    expect(manager.reopenRecentlyClosed()).toMatchObject({
      workspaceId: 'workspace_default',
      islandId: undefined,
    });
    expect(data.sessions.listRecentlyClosed()).toHaveLength(0);
  });
  it('B09 background grouping follows the opener while another workspace remains active', () => {
    const { manager, data } = setup();
    const first = manager.createTab('https://example.test/first');
    const workspace = data.experience.createWorkspace('Second');
    manager.switchWorkspace(workspace.workspaceId);
    const active = manager.getActiveTabId();
    const child = manager.openTabFrom(first.view.webContents, 'https://example.test/child');
    expect(child.workspaceId).toBe(first.workspaceId);
    expect(manager.getTab(first.tabId).islandId).toBe(child.islandId);
    expect(manager.getActiveTabId()).toBe(active);
    expect(data.experience.read().activeWorkspaceId).toBe(workspace.workspaceId);
  });
  it('B12 retains the valid workspace for its last-tab fallback', async () => {
    const { manager, data } = setup();
    const workspace = data.experience.createWorkspace('Keep this workspace');
    const tab = manager.createTab('https://example.test/', true, {
      workspaceId: workspace.workspaceId,
    });
    await manager.closeTab(tab.tabId);
    expect(manager.getTabs()).toHaveLength(1);
    expect(manager.getTabs()[0]?.workspaceId).toBe(workspace.workspaceId);
  });
  it('B16 relative shortcuts and Close Right follow pinned, island, ungrouped order', async () => {
    const { manager } = setup();
    const ungrouped = manager.createTab('https://example.test/ungrouped');
    const grouped = manager.createTab('https://example.test/grouped');
    const pinned = manager.createTab('https://example.test/pinned');
    manager.setPinned(pinned.tabId, true);
    manager.createIsland([grouped.tabId], 'Group');
    manager.activateByIndex(0);
    expect(manager.getActiveTabId()).toBe(pinned.tabId);
    manager.activateRelative(1);
    expect(manager.getActiveTabId()).toBe(grouped.tabId);
    await manager.closeTabsToRight(grouped.tabId);
    expect(manager.getTabs().map((tab) => tab.tabId)).not.toContain(ungrouped.tabId);
    expect(manager.getTabs()).toHaveLength(2);
  });
  it('B11 opens a fresh website view when leaving a model workspace', () => {
    const { manager } = setup();
    const workspace = manager.createTab(createOrynModelWorkspaceURL('a'.repeat(32)));
    manager.navigate('https://example.test/');
    expect(manager.getActiveWebContents()).not.toBe(workspace.view.webContents);
    expect(workspace.view.webContents.getURL()).toContain('oryn:');
  });
  it('B22 rejects restart for any live page without running script or closing it', async () => {
    const { manager } = setup();
    const clean = manager.createTab();
    expect(await manager.allowsUpdateRestart()).toBe(true);
    const draft = manager.createTab('https://example.test/draft');
    expect(await manager.allowsUpdateRestart()).toBe(false);
    expect(draft.view.webContents.close).not.toHaveBeenCalled();
    expect(clean.view.webContents.close).not.toHaveBeenCalled();
    expect(manager.getTabs()).toHaveLength(2);
  });
  it('lazy inactive restore does not start a document until activation', () => {
    const { manager } = setup();
    const tab = manager.createTab('https://example.test/', false, { lazy: true });
    expect(tab.view.webContents.loadURL).not.toHaveBeenCalled();
    manager.activateTab(tab.tabId);
    expect(tab.view.webContents.loadURL).toHaveBeenCalledTimes(1);
  });
});
