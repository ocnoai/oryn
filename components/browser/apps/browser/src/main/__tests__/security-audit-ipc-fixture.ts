import { EventEmitter } from 'node:events';
import { vi } from 'vitest';
import type { Session, WebFrameMain } from 'electron';
const electronDouble = vi.hoisted(() => {
  const handlers = new Map<string, (...args: unknown[]) => unknown>(),
    targets = new Map<number, ReturnType<typeof page>>();
  return {
    handlers,
    targets,
    app: { on: vi.fn(), removeListener: vi.fn() },
    dialog: { showMessageBox: vi.fn() },
    shell: {},
    ipcMain: {
      handle: (c: string, f: (...args: unknown[]) => unknown) => handlers.set(c, f),
      removeHandler: (c: string) => handlers.delete(c),
    },
    webContents: { fromId: (id: number) => targets.get(id) },
  };
});
export { electronDouble };
vi.mock('electron', () => electronDouble);
import { installIpcHandlers } from '../ipc-handlers';
export class FakeSession extends EventEmitter {
  request!: NonNullable<Parameters<Session['setPermissionRequestHandler']>[0]>;
  check!: NonNullable<Parameters<Session['setPermissionCheckHandler']>[0]>;
  setPermissionRequestHandler(f: FakeSession['request']) {
    this.request = f;
  }
  setPermissionCheckHandler(f: FakeSession['check']) {
    this.check = f;
  }
  setDisplayMediaRequestHandler = vi.fn();
  clearStorageData = vi.fn(async () => undefined);
  cookies = { get: vi.fn(async () => []), remove: vi.fn(async () => undefined) };
}
export function page(id: number, session: FakeSession, url = 'https://example.com/page') {
  const mainFrame = {
    url,
    origin: new URL(url).origin,
    frameToken: `frame-${id}`,
    frameTreeNodeId: id,
    processId: id * 10,
    routingId: id,
    parent: null,
    detached: false,
    isDestroyed: () => false,
  } as unknown as WebFrameMain;
  Object.assign(mainFrame, { framesInSubtree: [mainFrame] });
  return Object.assign(new EventEmitter(), {
    id,
    session,
    getURL: () => url,
    isDestroyed: () => false,
    send: vi.fn(),
    mainFrame,
  });
}
export function installFixture(
  data: unknown,
  jobs: unknown,
  prompts = new EventEmitter(),
  pages = [page(10, new FakeSession())],
) {
  const trusted = { id: 999, mainFrame: {}, send: vi.fn() };
  let active = pages[0]!;
  const callers = Object.assign(new EventEmitter(), {
    getDocumentId: (p: { id: number }) => `document_${p.id}`,
    hasDocumentAuthority: (): boolean => true,
  });
  const controller = {
    window: { isDestroyed: () => false },
    trustedWebContents: trusted,
    isTrustedUIWebContents: (p: unknown) => p === trusted,
    sendToPassiveOverlay: vi.fn(),
    get activeWebsiteWebContents() {
      return active;
    },
    currentOrigin: () => new URL(active.getURL()).origin,
    tabs: {
      getTabs: () =>
        pages.map((p) => ({
          tabId: `tab_${p.id}`,
          url: p.getURL(),
          webContentsId: p.id,
          view: { webContents: p },
        })),
      getActiveTabId: () => `tab_${active.id}`,
      getTab: () => ({ tabId: `tab_${active.id}` }),
    },
  };
  for (const p of pages) electronDouble.targets.set(p.id, p);
  const cleanup = installIpcHandlers(
    controller as never,
    {} as never,
    callers as never,
    data as Parameters<typeof installIpcHandlers>[3],
    {} as never,
    {} as never,
    prompts as never,
    {} as never,
    jobs as Parameters<typeof installIpcHandlers>[8],
    {} as never,
    {} as never,
    new EventEmitter() as never,
    {} as never,
  );
  return {
    trusted,
    callers,
    controller,
    select: (p: ReturnType<typeof page>) => {
      active = p;
    },
    invoke: (channel: string, ...args: unknown[]) =>
      electronDouble.handlers.get(channel)!(
        { sender: trusted, senderFrame: trusted.mainFrame },
        ...args,
      ),
    cleanup: () => {
      cleanup();
      electronDouble.targets.clear();
      electronDouble.handlers.clear();
    },
  };
}
