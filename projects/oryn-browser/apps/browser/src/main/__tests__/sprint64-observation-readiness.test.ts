import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { BrowserWindow, WebContents } from 'electron';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BrowserDataService } from '../data/browser-data-service';
import { BrowserCallerContextService } from '../browser-caller-context';
import { TabManager } from '../tab-manager';
import { SidebarWebAppController } from '../sidebar-web-app-controller';

vi.mock('electron', () => {
  const isolated = () => ({
    webRequest: { onBeforeRequest: vi.fn() },
    fetch: vi.fn(),
    setPermissionRequestHandler: vi.fn(),
    setPermissionCheckHandler: vi.fn(),
    setDisplayMediaRequestHandler: vi.fn(),
    setUserAgent: vi.fn(),
    on: vi.fn(),
  });
  return {
    app: { userAgentFallback: 'Chrome/150', getName: () => 'Oryn Browser' },
    session: { fromPartition: () => isolated() },
    dialog: { showMessageBoxSync: () => 0 },
    WebContentsView: class {
      webContents = new (class extends EventEmitter {
        id = Math.random();
        url = '';
        destroyed = false;
        mainFrame = { url: '', processId: 1, routingId: 1 };
        session = isolated();
        navigationHistory = { canGoBack: () => false, canGoForward: () => false };
        getURL = () => this.url;
        getTitle = () => 'Fixture';
        isLoading = () => false;
        isDestroyed = () => this.destroyed;
        setZoomFactor = vi.fn();
        setWindowOpenHandler = vi.fn();
        setAudioMuted = vi.fn();
        stop = vi.fn();
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
      setVisible = vi.fn();
      setBackgroundColor = vi.fn();
    },
  };
});

const directories: string[] = [];
afterEach(() => {
  vi.useRealTimers();
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
});
function setup(onWebContentsCreated: (contents: WebContents) => void | Promise<void>) {
  const path = mkdtempSync(join(tmpdir(), 'oryn-observation-ready-'));
  directories.push(path);
  const data = new BrowserDataService(path);
  const window = {
    isDestroyed: () => false,
    contentView: { addChildView: vi.fn(), removeChildView: vi.fn() },
  } as unknown as BrowserWindow;
  const manager = new TabManager(window, new BrowserCallerContextService(), data, {
    websitePreloadPath: '',
    workspacePreloadPath: '',
    beforeFirstNavigation: onWebContentsCreated,
  });
  return { data, window, manager };
}
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));
const requestedLoads = (loadURL: unknown) =>
  vi
    .mocked(loadURL as (url: string) => Promise<void>)
    .mock.calls.map(([url]) => url)
    .filter((url) => url !== 'about:blank');

describe('observed browser-initiated navigation', () => {
  it('publishes a managed active tab before loading only after observation is ready', async () => {
    const ready = deferred();
    const hook = vi.fn(() => ready.promise);
    const { manager } = setup(hook);
    const tab = manager.createTab('https://security.example/initial');
    expect(hook).toHaveBeenCalledOnce();
    expect(manager.getActiveTabId()).toBe(tab.tabId);
    expect(manager.getTabs().map((item) => item.tabId)).toContain(tab.tabId);
    expect(tab.view.webContents.loadURL).toHaveBeenCalledExactlyOnceWith('about:blank');
    ready.resolve();
    await flush();
    expect(requestedLoads(tab.view.webContents.loadURL)).toEqual([
      'https://security.example/initial',
    ]);
  });
  it('lets a newer user intent supersede the pending initial URL without duplicate loading', async () => {
    const ready = deferred();
    const { manager } = setup(() => ready.promise);
    const tab = manager.createTab('https://security.example/initial');
    manager.navigate('https://security.example/newer');
    ready.resolve();
    await flush();
    expect(requestedLoads(tab.view.webContents.loadURL)).toEqual([
      'https://security.example/newer',
    ]);
  });
  it('waits for the inert bootstrap load to finish even after observation is ready', async () => {
    const ready = deferred();
    const bootstrap = deferred();
    const { manager } = setup((contents) => {
      vi.mocked(contents.loadURL).mockImplementationOnce(() => bootstrap.promise);
      return ready.promise;
    });
    const tab = manager.createTab('https://security.example/after-bootstrap');
    expect(tab.view.webContents.loadURL).toHaveBeenCalledExactlyOnceWith('about:blank');
    ready.resolve();
    await flush();
    expect(requestedLoads(tab.view.webContents.loadURL)).toEqual([]);
    bootstrap.resolve();
    await flush();
    expect(requestedLoads(tab.view.webContents.loadURL)).toEqual([
      'https://security.example/after-bootstrap',
    ]);
  });
  it('shares an ongoing bootstrap and loads only the newest superseding intent afterward', async () => {
    const ready = deferred();
    const bootstrap = deferred();
    const hook = vi.fn((contents: WebContents) => {
      vi.mocked(contents.loadURL).mockImplementationOnce(() => bootstrap.promise);
      return ready.promise;
    });
    const { manager } = setup(hook);
    const tab = manager.createTab('https://security.example/initial');
    manager.navigate('https://security.example/newer');
    ready.resolve();
    await flush();
    expect(tab.view.webContents.loadURL).toHaveBeenCalledExactlyOnceWith('about:blank');
    manager.navigate('https://security.example/newest');
    await flush();
    expect(tab.view.webContents.loadURL).toHaveBeenCalledExactlyOnceWith('about:blank');
    expect(hook).toHaveBeenCalledOnce();
    bootstrap.resolve();
    await flush();
    expect(requestedLoads(tab.view.webContents.loadURL)).toEqual([
      'https://security.example/newest',
    ]);
  });
  it('cannot load a stopped, destroyed or reclassified target when readiness arrives late', async () => {
    const ready = deferred();
    const { manager } = setup(() => ready.promise);
    const stopped = manager.createTab('https://security.example/stopped');
    manager.stop();
    const destroyed = manager.createTab('https://security.example/destroyed');
    destroyed.view.webContents.close();
    const original = manager.createTab('https://security.example/original');
    manager.navigate('/tmp/oryn-explicit-file.html');
    const file = manager.getTab(manager.getActiveTabId()!);
    expect(file.tabId).not.toBe(original.tabId);
    ready.resolve();
    await flush();
    expect(requestedLoads(stopped.view.webContents.loadURL)).toEqual([]);
    expect(requestedLoads(destroyed.view.webContents.loadURL)).toEqual([]);
    expect(original.view.webContents.loadURL).not.toHaveBeenCalledWith(
      expect.stringMatching(/^file:/),
    );
    expect(requestedLoads(file.view.webContents.loadURL)).toEqual([
      'file:///tmp/oryn-explicit-file.html',
    ]);
  });
  it.each(['back', 'forward'] as const)(
    'a newer native %s intent invalidates a queued load',
    async (direction) => {
      const ready = deferred();
      const { manager } = setup(() => ready.promise);
      const tab = manager.createTab('https://security.example/queued');
      const native = vi.fn();
      Object.assign(
        tab.view.webContents.navigationHistory,
        direction === 'back'
          ? { canGoBack: () => true, goBack: native }
          : { canGoForward: () => true, goForward: native },
      );
      manager[direction]();
      ready.resolve();
      await flush();
      expect(native).toHaveBeenCalledOnce();
      expect(requestedLoads(tab.view.webContents.loadURL)).toEqual([]);
    },
  );
  it.each(['reject', 'timeout'])(
    'continues browsing when observation %s fails without waiting forever',
    async (failure) => {
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      const { manager } = setup(() =>
        failure === 'reject'
          ? Promise.reject(new Error('OBSERVATION_FAILED'))
          : new Promise<void>(() => undefined),
      );
      const tab = manager.createTab('https://security.example/');
      if (failure === 'timeout') await vi.advanceTimersByTimeAsync(2001);
      else await flush();
      expect(requestedLoads(tab.view.webContents.loadURL)).toEqual(['https://security.example/']);
    },
  );
  it('starts lazy/restored observation only on activation even after the initial deadline would have elapsed', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const ready = deferred();
    const hook = vi.fn(() => ready.promise);
    const { manager } = setup(hook);
    const lazy = manager.createTab('https://security.example/restored', false, { lazy: true });
    await vi.advanceTimersByTimeAsync(3000);
    expect(hook).not.toHaveBeenCalled();
    expect(lazy.view.webContents.loadURL).not.toHaveBeenCalled();
    manager.activateTab(lazy.tabId);
    expect(hook).toHaveBeenCalledOnce();
    expect(lazy.view.webContents.loadURL).toHaveBeenCalledExactlyOnceWith('about:blank');
    ready.resolve();
    await flush();
    expect(requestedLoads(lazy.view.webContents.loadURL)).toEqual([
      'https://security.example/restored',
    ]);
  });
  it('does not synthesize a navigation for a native popup while its observer initializes', async () => {
    const ready = deferred();
    const { manager } = setup(() => ready.promise);
    const popup = manager.createTab('https://security.example/popup', true, { popupOptions: {} });
    ready.resolve();
    await flush();
    expect(popup.view.webContents.loadURL).not.toHaveBeenCalled();
  });
  it('loads a registered sidebar view only once observation is ready', async () => {
    const ready = deferred();
    const { data, window } = setup(() => undefined);
    const sidebar = new SidebarWebAppController(window, data, vi.fn(), () => ready.promise);
    sidebar.open('whatsapp');
    const view = vi.mocked(window.contentView.addChildView).mock.calls.at(-1)![0] as unknown as {
      webContents: { loadURL: unknown };
    };
    expect(sidebar.hasOpenPages()).toBe(true);
    expect(requestedLoads(view.webContents.loadURL)).toEqual([]);
    ready.resolve();
    await flush();
    expect(requestedLoads(view.webContents.loadURL)).toEqual(['https://web.whatsapp.com/']);
    sidebar.destroy();
  });
  it('publishes sidebar ownership before loading and drops late loads for destroyed views', async () => {
    const ready = deferred();
    const { data, window } = setup(() => undefined);
    const sidebar = new SidebarWebAppController(window, data, vi.fn(), () => ready.promise);
    sidebar.open('whatsapp');
    const view = vi.mocked(window.contentView.addChildView).mock.calls.at(-1)![0] as unknown as {
      webContents: { loadURL: unknown };
    };
    expect(sidebar.hasOpenPages()).toBe(true);
    expect(requestedLoads(view.webContents.loadURL)).toEqual([]);
    sidebar.destroy();
    ready.resolve();
    await flush();
    expect(requestedLoads(view.webContents.loadURL)).toEqual([]);
  });
});
