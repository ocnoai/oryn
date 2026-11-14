import { EventEmitter } from 'node:events';
import { WebContentsView, type BrowserWindow, type WebContents } from 'electron';
import { describe, expect, it, vi } from 'vitest';
import type { BrowserCallerContextService } from '../browser-caller-context';
import type { BrowserDataService } from '../data/browser-data-service';
import { TabManager } from '../tab-manager';

vi.mock('electron', () => ({
  WebContentsView: class {
    webContents = Object.assign(new EventEmitter(), {
      session: {},
      navigationHistory: { canGoBack: () => false, canGoForward: () => false },
      getURL: () => '',
      getTitle: () => 'Opener',
      isLoading: () => false,
      isDestroyed: () => false,
      loadURL: vi.fn().mockResolvedValue(undefined),
      setWindowOpenHandler: vi.fn(),
    });
    constructor(options: Electron.WebContentsViewConstructorOptions = {}) {
      if ('webContents' in options) {
        if (!(options.webContents instanceof EventEmitter))
          throw new TypeError('options.webContents must be a WebContents');
        this.webContents = options.webContents as unknown as typeof this.webContents;
      }
    }
    setBounds = vi.fn();
  },
}));

function setup(autoGroupOpenerTabs = true) {
  const experience = { activeWorkspaceId: 'main', autoGroupOpenerTabs, islands: [] };
  const window = {
    isDestroyed: () => false,
    contentView: { addChildView: vi.fn(), removeChildView: vi.fn() },
  };
  const data = {
    experience: { read: () => experience, createIsland: () => ({ islandId: 'related' }) },
    sessions: { saveOpenTabs: vi.fn() },
    installChromiumSessionHandlers: vi.fn(),
  };
  const contexts = { register: () => () => {}, getDocumentId: () => 'document' };
  const manager = new TabManager(
    window as unknown as BrowserWindow,
    contexts as unknown as BrowserCallerContextService,
    data as unknown as BrowserDataService,
    { websitePreloadPath: '', workspacePreloadPath: '' },
  );
  return { manager, window };
}

async function open(
  contents: WebContents,
  disposition: string,
  options: Electron.BrowserWindowConstructorOptions & { webContents?: WebContents | null } = {},
) {
  const handler = vi.mocked(contents.setWindowOpenHandler).mock.calls[0]![0];
  const result = handler({ url: 'https://example.com/child', disposition } as Parameters<
    typeof handler
  >[0]);
  result.createWindow?.(options);
  await new Promise<void>((resolve) => setImmediate(resolve));
  return { action: result.action };
}

describe('tabs opened from a page', () => {
  it.each([undefined, null])(
    'opens a background tab when Electron supplies webContents=%s',
    async (webContents) => {
      const { manager, window } = setup();
      const opener = manager.createTab('https://example.com');
      window.contentView.addChildView.mockClear();
      expect(await open(opener.view.webContents, 'background-tab', { webContents })).toEqual({
        action: 'allow',
      });
      expect(manager.getActiveTabId()).toBe(opener.tabId);
      expect(window.contentView.addChildView).not.toHaveBeenCalled();
      expect(manager.getTab(opener.tabId).islandId).toBe('related');
      const child = manager.getTabs().find((tab) => tab.tabId !== opener.tabId)!;
      expect(child.view.webContents.loadURL).toHaveBeenCalledOnce();
      expect(child.view.webContents.loadURL).toHaveBeenCalledWith('https://example.com/child', {
        httpReferrer: undefined,
      });
    },
  );

  it('adopts an existing Chromium popup without navigating it again', async () => {
    const { manager } = setup();
    const opener = manager.createTab('https://example.com');
    const guest = new WebContentsView().webContents;
    await open(opener.view.webContents, 'foreground-tab', { webContents: guest });
    expect(manager.getTab(manager.getActiveTabId()).view.webContents).toBe(guest);
    expect(guest.loadURL).not.toHaveBeenCalled();
  });

  it('uses the existing island for context-menu tabs without switching pages', () => {
    const { manager } = setup(false);
    const opener = manager.createTab('https://example.com', true, { islandId: 'existing' });
    const child = manager.openTabFrom(opener.view.webContents, 'https://example.com/child');
    expect(child).toMatchObject({ islandId: 'existing', workspaceId: opener.workspaceId });
    expect(manager.getActiveTabId()).toBe(opener.tabId);
  });

  it('auto-groups context-menu tabs and subsequent links in the same island', () => {
    const { manager } = setup();
    const opener = manager.createTab('https://example.com');
    const first = manager.openTabFrom(opener.view.webContents, 'https://example.com/first');
    const second = manager.openTabFrom(opener.view.webContents, 'https://example.com/second');
    expect(first.islandId).toBe('related');
    expect(second.islandId).toBe(first.islandId);
  });

  it('honors foreground tab requests and the disabled auto-group preference', async () => {
    const { manager } = setup(false);
    const opener = manager.createTab('https://example.com');
    await open(opener.view.webContents, 'foreground-tab');
    expect(manager.getActiveTabId()).not.toBe(opener.tabId);
    expect(manager.getTab(manager.getActiveTabId()).islandId).toBeUndefined();
  });
});
