import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  dialog,
  systemPreferences,
  type Session,
  type WebContents,
  type WebFrameMain,
} from 'electron';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BrowserDataService } from '../data/browser-data-service';
vi.mock('electron', () => ({
  dialog: { showMessageBox: vi.fn() },
  shell: { openExternal: vi.fn(async () => undefined) },
  systemPreferences: { getMediaAccessStatus: vi.fn(() => 'granted') },
}));
const roots: string[] = [];
afterEach(() => {
  vi.resetAllMocks();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
class FakeSession extends EventEmitter {
  setDisplayMediaRequestHandler = vi.fn();
  clearStorageData = vi.fn(async () => undefined);
  cookies = { get: vi.fn(async () => []), remove: vi.fn(async () => undefined) };
  request: Parameters<Session['setPermissionRequestHandler']>[0] = null;
  check: Parameters<Session['setPermissionCheckHandler']>[0] = null;
  setPermissionRequestHandler(handler: typeof this.request) {
    this.request = handler;
  }
  setPermissionCheckHandler(handler: typeof this.check) {
    this.check = handler;
  }
  asSession() {
    return this as unknown as Session;
  }
}
function setup() {
  const root = mkdtempSync(join(tmpdir(), 'oryn-site-policy-'));
  roots.push(root);
  const service = new BrowserDataService(root);
  const session = new FakeSession();
  service.installChromiumSessionHandlers(session.asSession());
  const page = Object.assign(new EventEmitter(), {
    id: 51,
    session: session.asSession(),
    getURL: (): string => 'https://example.com/page',
    isDestroyed: (): boolean => false,
  });
  attachFrames(page);
  return { service, session, page, contents: page as unknown as WebContents };
}
const details = { requestingUrl: 'https://example.com/page', isMainFrame: true };
describe('ordinary permission policy', () => {
  it('allows sanitized clipboard writes without a prompt or a persisted grant', () => {
    const { service, session, contents } = setup();
    const callback = vi.fn();
    session.request!(contents, 'clipboard-sanitized-write', callback, details);
    expect(callback).toHaveBeenCalledExactlyOnceWith(true);
    expect(
      session.check!(contents, 'clipboard-sanitized-write', 'https://example.com', details),
    ).toBe(true);
    expect(dialog.showMessageBox).not.toHaveBeenCalled();
    expect(service.sitePermissions.list()).toEqual([]);
  });

  it('keeps main-frame request/check agreement when native embeddingOrigin is its own origin', () => {
    const { service, session, contents } = setup();
    const nativeDetails = { ...details, embeddingOrigin: 'https://example.com/' };
    const granted = vi.fn();
    session.request!(contents, 'clipboard-sanitized-write', granted, details);
    expect(granted).toHaveBeenCalledWith(true);
    expect(
      session.check!(contents, 'clipboard-sanitized-write', 'https://example.com/', nativeDetails),
    ).toBe(true);
    service.sitePermissions.set('https://example.com', 'geolocation', 'allow');
    expect(session.check!(contents, 'geolocation', 'https://example.com/', nativeDetails)).toBe(
      true,
    );
    expect(
      session.check!(contents, 'clipboard-sanitized-write', 'https://example.com/', {
        ...nativeDetails,
        embeddingOrigin: 'https://foreign.example/',
      }),
    ).toBe(false);
  });
  it('honors clipboard write blocks within their session partition', () => {
    const { service, session, contents } = setup();
    service.sitePermissions.set('https://example.com', 'clipboard-sanitized-write', 'deny');
    const callback = vi.fn();
    session.request!(contents, 'clipboard-sanitized-write', callback, details);
    expect(callback).toHaveBeenCalledExactlyOnceWith(false);
    expect(
      session.check!(contents, 'clipboard-sanitized-write', 'https://example.com', details),
    ).toBe(false);
    const sidebar = new FakeSession();
    service.installChromiumSessionHandlers(sidebar.asSession(), 'persist:sidebar');
    const sidebarContents = Object.assign(new EventEmitter(), {
      session: sidebar.asSession(),
      getURL: () => details.requestingUrl,
      isDestroyed: () => false,
    }) as unknown as WebContents;
    attachFrames(sidebarContents);
    callback.mockClear();
    sidebar.request!(sidebarContents, 'clipboard-sanitized-write', callback, details);
    expect(callback).toHaveBeenCalledExactlyOnceWith(true);
    expect(
      sidebar.check!(sidebarContents, 'clipboard-sanitized-write', 'https://example.com', details),
    ).toBe(true);
    expect(dialog.showMessageBox).not.toHaveBeenCalled();
  });
  it('still requires permission to read the clipboard after allowing a write', async () => {
    const { service, session, contents } = setup();
    session.request!(contents, 'clipboard-sanitized-write', vi.fn(), details);
    expect(session.check!(contents, 'clipboard-read', 'https://example.com', details)).toBe(false);
    vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 0, checkboxChecked: false });
    const granted = await new Promise<boolean>((resolve) =>
      session.request!(contents, 'clipboard-read', resolve, details),
    );
    expect(granted).toBe(false);
    expect(dialog.showMessageBox).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ message: 'https://example.com requests clipboard-read' }),
    );
    expect(service.sitePermissions.list()).toEqual([]);
  });
  it.each(['subframe', 'foreign origin', 'wrong session', 'destroyed page', 'file URL'])(
    'rejects sanitized clipboard writes from a %s',
    (context) => {
      const { session, contents, page } = setup();
      const requestDetails = { ...details };
      if (context === 'subframe') requestDetails.isMainFrame = false;
      if (context === 'foreign origin') requestDetails.requestingUrl = 'https://foreign.example';
      if (context === 'wrong session') page.session = new FakeSession().asSession();
      if (context === 'destroyed page') page.isDestroyed = () => true;
      if (context === 'file URL') {
        requestDetails.requestingUrl = 'file:///tmp/clipboard.html';
        page.getURL = () => requestDetails.requestingUrl;
      }
      const callback = vi.fn();
      session.request!(contents, 'clipboard-sanitized-write', callback, requestDetails);
      expect(callback).toHaveBeenCalledExactlyOnceWith(false);
      expect(
        session.check!(
          contents,
          'clipboard-sanitized-write',
          new URL(requestDetails.requestingUrl).origin,
          requestDetails,
        ),
      ).toBe(false);
      expect(dialog.showMessageBox).not.toHaveBeenCalled();
    },
  );
  it('rejects clipboard checks without contents or with a foreign embedding origin', () => {
    const { session, contents } = setup();
    expect(session.check!(null, 'clipboard-sanitized-write', 'https://example.com', details)).toBe(
      false,
    );
    expect(
      session.check!(contents, 'clipboard-sanitized-write', 'https://example.com', {
        ...details,
        embeddingOrigin: 'https://foreign.example',
      }),
    ).toBe(false);
  });
  it('B06 rejects null, unverified and embedded contexts consistently', () => {
    const { service, session, contents } = setup();
    service.sitePermissions.set('https://example.com', 'notifications', 'allow');
    expect(session.check!(null, 'notifications', 'https://example.com', details)).toBe(false);
    const callback = vi.fn();
    session.request!(contents, 'notifications', callback, { ...details, isMainFrame: false });
    expect(callback).toHaveBeenCalledWith(false);
    expect(
      session.check!(contents, 'notifications', 'https://example.com', {
        ...details,
        isMainFrame: false,
      }),
    ).toBe(false);
    expect(session.check!(contents, 'notifications', 'https://foreign.example', details)).toBe(
      false,
    );
    session.request!(contents, 'notifications', callback, details);
    expect(callback).toHaveBeenLastCalledWith(true);
    expect(session.check!(contents, 'notifications', 'https://example.com', details)).toBe(true);
  });
  it('B05 page-lifetime grant agrees with checks and is revoked on replacement', async () => {
    const { session, contents, page } = setup();
    vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 1, checkboxChecked: false });
    const granted = new Promise<boolean>((resolve) =>
      session.request!(contents, 'notifications', resolve, details),
    );
    expect(await granted).toBe(true);
    expect(session.check!(contents, 'notifications', 'https://example.com', details)).toBe(true);
    page.emit('did-navigate');
    expect(session.check!(contents, 'notifications', 'https://example.com', details)).toBe(false);
  });

  it('preserves a page grant through same-document navigation but revokes same-URL replacement', async () => {
    const { session, contents, page } = setup();
    vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 1, checkboxChecked: false });
    expect(
      await new Promise<boolean>((done) =>
        session.request!(contents, 'notifications', done, details),
      ),
    ).toBe(true);
    const url = 'https://example.com/updated#section';
    Object.assign(contents.mainFrame, { url });
    page.getURL = () => url;
    page.emit('did-start-navigation', {}, url, true, true);
    page.emit('did-navigate-in-page', {}, url, true);
    const updated = { requestingUrl: url, isMainFrame: true };
    expect(session.check!(contents, 'notifications', 'https://example.com', updated)).toBe(true);
    // A complete replacement may retain the same frame token and URL; the epoch
    // from the committed frame-navigation event must still revoke its grant.
    page.emit('did-frame-navigate');
    expect(session.check!(contents, 'notifications', 'https://example.com', updated)).toBe(false);
  });
  it('navigation while a permission prompt is open cannot save a late grant', async () => {
    const { service, session, contents, page } = setup();
    let resolve!: (value: Electron.MessageBoxReturnValue) => void;
    vi.mocked(dialog.showMessageBox).mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    const granted = new Promise<boolean>((done) =>
      session.request!(contents, 'notifications', done, details),
    );
    page.emit('did-start-navigation');
    resolve({ response: 2, checkboxChecked: false });
    expect(await granted).toBe(false);
    expect(service.sitePermissions.list()).toHaveLength(0);
  });
  it('clearing a site revokes page-lifetime grants as well as stored decisions', async () => {
    const { service, session, contents } = setup();
    vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 1, checkboxChecked: false });
    await new Promise<boolean>((done) =>
      session.request!(contents, 'notifications', done, details),
    );
    await service.clearSiteData(session.asSession(), 'https://example.com');
    expect(session.check!(contents, 'notifications', 'https://example.com', details)).toBe(false);
  });
  it('display capture uses the system picker, denies its fallback and never persists a grant', () => {
    const { service, session, contents } = setup();
    expect(session.setDisplayMediaRequestHandler).toHaveBeenCalledWith(expect.any(Function), {
      useSystemPicker: true,
    });
    const denied = vi.fn();
    session.setDisplayMediaRequestHandler.mock.calls[0]![0]({}, denied);
    expect(denied).toHaveBeenCalledWith({});
    service.sitePermissions.set('https://example.com', 'display-capture', 'allow');
    const granted = vi.fn();
    session.request!(contents, 'display-capture', granted, details);
    expect(granted).toHaveBeenCalledWith(true);
    expect(dialog.showMessageBox).not.toHaveBeenCalled();
    const embedded = vi.fn();
    session.request!(contents, 'display-capture', embedded, { ...details, isMainFrame: false });
    expect(embedded).toHaveBeenCalledWith(false);
  });
  it('does not turn a camera grant into a microphone grant or sidebar grant', () => {
    const { service, session, contents } = setup();
    service.sitePermissions.set('https://example.com', 'camera', 'allow');
    expect(
      session.check!(contents, 'media', 'https://example.com', { ...details, mediaType: 'video' }),
    ).toBe(true);
    expect(
      session.check!(contents, 'media', 'https://example.com', { ...details, mediaType: 'audio' }),
    ).toBe(false);
    const sidebar = new FakeSession();
    service.installChromiumSessionHandlers(sidebar.asSession(), 'persist:sidebar');
    const sidebarContents = Object.assign(new EventEmitter(), {
      session: sidebar.asSession(),
      getURL: () => 'https://example.com/page',
      isDestroyed: () => false,
    }) as unknown as WebContents;
    expect(
      sidebar.check!(sidebarContents, 'media', 'https://example.com', {
        ...details,
        mediaType: 'video',
      }),
    ).toBe(false);
  });
});

function attachFrames(page: object) {
  const main = {
    url: details.requestingUrl,
    origin: 'https://example.com',
    frameToken: 'top-token',
    frameTreeNodeId: 1,
    detached: false,
    isDestroyed: () => false,
    parent: null,
  } as unknown as WebFrameMain;
  const child = {
    url: 'https://embed.example/widget',
    origin: 'https://embed.example',
    frameToken: 'child-token',
    frameTreeNodeId: 2,
    detached: false,
    isDestroyed: () => false,
    parent: main,
  } as unknown as WebFrameMain;
  Object.assign(main, { framesInSubtree: [main, child] });
  Object.assign(page, { mainFrame: main });
  return { main, child, embedded: { requestingUrl: child.url, isMainFrame: false } };
}

describe('qualified iframe consent and capture termination', () => {
  it('admits qualified fullscreen but denies camera without a qualified Chromium gate', () => {
    const { session, contents, page } = setup();
    const { embedded } = attachFrames(page);
    const callback = vi.fn();
    session.request!(contents, 'fullscreen', callback, embedded);
    expect(callback).toHaveBeenCalledWith(true);
    callback.mockClear();
    session.request!(contents, 'media', callback, { ...embedded, mediaTypes: ['video'] });
    expect(callback).toHaveBeenCalledWith(false);
    expect(dialog.showMessageBox).not.toHaveBeenCalled();
  });
  it('binds consent to a concrete frame and invalidates it on ancestor navigation', async () => {
    const { session, contents, page } = setup();
    const { embedded } = attachFrames(page);
    vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 1, checkboxChecked: false });
    expect(
      await new Promise<boolean>((done) =>
        session.request!(contents, 'geolocation', done, embedded),
      ),
    ).toBe(true);
    expect(dialog.showMessageBox).toHaveBeenCalledWith(
      expect.objectContaining({
        message: 'https://embed.example embedded in https://example.com requests geolocation',
      }),
    );
    expect(session.check!(contents, 'geolocation', 'https://embed.example', embedded)).toBe(true);
    page.emit('did-frame-navigate');
    expect(session.check!(contents, 'geolocation', 'https://embed.example', embedded)).toBe(false);
  });
  it('does not reuse a saved embedded grant through a different nesting path', async () => {
    const { service, session, contents, page } = setup();
    const { main, child, embedded } = attachFrames(page);
    vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 2, checkboxChecked: false });
    expect(
      await new Promise<boolean>((done) =>
        session.request!(contents, 'geolocation', done, embedded),
      ),
    ).toBe(true);
    expect(service.sitePermissions.get('https://embed.example', 'geolocation')).toBeUndefined();
    const foreign = {
      url: 'https://other.example/frame',
      origin: 'https://other.example',
      frameToken: 'foreign',
      frameTreeNodeId: 3,
      detached: false,
      isDestroyed: () => false,
      parent: main,
    };
    Object.assign(child, { parent: foreign });
    Object.assign(main, { framesInSubtree: [main, foreign, child] });
    expect(session.check!(contents, 'geolocation', 'https://embed.example', embedded)).toBe(false);
    await service.clearSiteData(session.asSession(), 'https://embed.example');
    expect(service.listEmbeddedPermissions('https://example.com')).toEqual([]);
  });

  it('accepts a qualified nested query with top-level native embedding origin and exact ancestor-scoped consent', () => {
    const { service, session, contents, page } = setup();
    const { main, child, embedded } = attachFrames(page);
    const middle = {
      url: 'https://middle.example/frame',
      origin: 'https://middle.example',
      frameToken: 'middle',
      frameTreeNodeId: 3,
      detached: false,
      isDestroyed: () => false,
      parent: main,
    };
    Object.assign(child, { parent: middle });
    Object.assign(main, { framesInSubtree: [main, middle, child] });
    service.sitePermissions.setEmbedded(main.origin, child.origin, 'geolocation', 'allow', [
      main.origin,
      middle.origin,
    ]);
    const nativeDetails = { ...embedded, embeddingOrigin: main.origin + '/' };
    expect(session.check!(contents, 'geolocation', child.origin + '/', nativeDetails)).toBe(true);
    expect(
      session.check!(contents, 'clipboard-sanitized-write', child.origin + '/', nativeDetails),
    ).toBe(false);
    Object.assign(middle, {
      origin: 'https://different.example',
      url: 'https://different.example/frame',
    });
    expect(session.check!(contents, 'geolocation', child.origin + '/', nativeDetails)).toBe(false);
  });
  it('does not save a late answer after a nested navigation', async () => {
    const { service, session, contents, page } = setup();
    const { embedded } = attachFrames(page);
    let answer!: (value: Electron.MessageBoxReturnValue) => void;
    vi.mocked(dialog.showMessageBox).mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    const result = new Promise<boolean>((done) =>
      session.request!(contents, 'geolocation', done, embedded),
    );
    page.emit('did-frame-navigate');
    answer({ response: 2, checkboxChecked: false });
    expect(await result).toBe(false);
    expect(service.listEmbeddedPermissions('https://example.com')).toEqual([]);
  });
  it('tracks saved checks and keeps Stop visible through canceled unload without regranting', () => {
    const { service, session, contents, page } = setup();
    service.sitePermissions.set('https://example.com', 'camera', 'allow');
    expect(
      session.check!(contents, 'media', 'https://example.com', { ...details, mediaType: 'video' }),
    ).toBe(true);
    expect(service.captureSnapshot()).toEqual([
      expect.objectContaining({ permissions: ['camera'], stopping: false }),
    ]);
    expect(service.requestStopCapture(51)).toBe(contents);
    page.emit('will-prevent-unload');
    expect(service.captureSnapshot()).toEqual([expect.objectContaining({ stopping: true })]);
    expect(
      session.check!(contents, 'media', 'https://example.com', { ...details, mediaType: 'video' }),
    ).toBe(false);
    expect(service.sitePermissions.get('https://example.com', 'camera')).toBeUndefined();
    page.emit('did-navigate');
    expect(service.captureSnapshot()).toEqual([]);
    expect(
      session.check!(contents, 'media', 'https://example.com', { ...details, mediaType: 'video' }),
    ).toBe(false);
  });
  it('explains macOS denial and opens only the matching native privacy pane', async () => {
    const { session, contents } = setup();
    vi.mocked(systemPreferences.getMediaAccessStatus).mockReturnValue('denied');
    vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 0, checkboxChecked: false });
    const callback = vi.fn();
    session.request!(contents, 'media', callback, { ...details, mediaTypes: ['audio'] });
    expect(callback).toHaveBeenCalledWith(false);
    expect(dialog.showMessageBox).toHaveBeenCalledWith(
      expect.objectContaining({
        message: 'macOS is blocking microphone access for Oryn.',
        detail: expect.stringContaining('Microphone'),
      }),
    );
    await Promise.resolve();
  });
});
