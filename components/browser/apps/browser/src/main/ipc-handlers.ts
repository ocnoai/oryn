import { browserDiagnostics } from './browser-diagnostics';
import { readBrowserBuildFlavor } from './browser-build-flavor';
import { EngineVersionService } from './engine-versions/engine-version-service';
import { open, readFile, writeFile } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { isAbsolute, join } from 'node:path';
import { homedir } from 'node:os';
import { BrowserImportService } from './data/browser-import-service';
import { requestPageClose } from './page-close';
import { exportCertificateChain } from './connection-security';
import type { BrowserImportDataType, BrowserImportOwner } from './data/browser-import-types';
import {
  parseBookmarkHTML,
  exportBookmarkHTML,
  MAX_BOOKMARK_IMPORT_BYTES,
} from './data/bookmark-html';
import type { BookmarkNode } from './data/bookmark-store';
import { AsyncLocalStorage } from 'node:async_hooks';
import { recoveredStores } from './data/json-file-store';
import {
  app,
  dialog,
  ipcMain,
  webContents,
  type IpcMainInvokeEvent,
  type WebContents,
} from 'electron';

import type { ArtifactManager } from './artifact-manager/artifact-manager';
import { ArtifactError } from './artifact-manager/types';
import type { BrowserCallerContextService } from './browser-caller-context';
import type { BrowserWindowController } from './browser-window';
import { BrowserJobError, type BrowserJobRegistry } from './jobs/browser-job-registry';
import type { ModelManagerService } from './model-manager/model-manager-service';
import type { BrowserDataService } from './data/browser-data-service';
import { isSupportedSitePermission } from './data/site-permission-store';
import type { HuggingFaceCredentialStore } from './data/hugging-face-credential-store';
import { EngineRequestError, type EngineSupervisor } from './engine-supervisor';
import { ModelResolutionError, type ModelResolver } from './model-resolution/model-resolver';
import { ModelRequestValidationError } from './model-resolution/model-request';
import {
  BrowserModelSessionError,
  type BrowserModelRequestProgress,
  type BrowserModelSession,
  type BrowserModelSessionRegistry,
} from './model-resolution/browser-model-sessions';
import type { PermissionPromptCoordinator } from './model-resolution/permission-prompt-coordinator';
import type {
  BrowserBookmarkCommand,
  BrowserChromeLayout,
  BrowserPassiveOverlayLayout,
  BrowserExperienceCommand,
  LocalAIPermissionPromptResponse,
} from '../shared/ipc';
import { IPC, type RuntimeCapabilities } from '../shared/ipc';
import type { OrynSettingsService } from './settings/oryn-settings-service';
import {
  isModelWorkspaceTaskContract,
  validateModelWorkspaceInput,
} from './model-workspace/task-contract-registry';
import type { ModelWorkspaceState } from '../shared/ipc';
import type { AppUpdateService } from './app-updates/app-update-service';
import {
  isOrynVerified,
  isUserTrustEligible,
  taskContractForProfile,
} from './model-workspace/model-workspace-eligibility';

export function installIpcHandlers(
  initialController: BrowserWindowController,
  engine: EngineSupervisor,
  callerContexts: BrowserCallerContextService,
  initialData: BrowserDataService,
  modelResolver: ModelResolver,
  modelSessions: BrowserModelSessionRegistry,
  permissionPrompts: PermissionPromptCoordinator,
  artifacts: ArtifactManager,
  jobs: BrowserJobRegistry,
  modelManager: ModelManagerService,
  orynSettings: OrynSettingsService,
  appUpdates: AppUpdateService,
  huggingFaceCredentials: HuggingFaceCredentialStore,
  windows?: {
    all(): BrowserWindowController[];
    rendererCrashes?(): number;
    data(controller: BrowserWindowController): BrowserDataService;
    create(): Promise<void>;
    createPrivate?(url?: string): Promise<void>;
    admitsActions?(): boolean;
    transfer?(owner: BrowserWindowController, tabId: string): Promise<void>;
  },
): () => void {
  let imports: BrowserImportService | undefined;
  const importService = () =>
    (imports ??= new BrowserImportService({
      bookmarks: initialData.bookmarks,
      history: initialData.history,
      appearance: initialData.appearance,
      homePath: homedir(),
      workerPath: join(__dirname, 'browser-import-worker.js'),
      temporaryRoot: join(initialData.paths.browserCoreRoot, 'import-temp'),
      nativeSandboxPath: app.isPackaged
        ? join(process.resourcesPath, 'helpers', 'browser_import_sandbox.node')
        : join(app.getAppPath(), 'build/helpers/browser_import_sandbox.node'),
    }));
  const browserImportOwners = new WeakMap<WebContents, BrowserImportOwner>();
  const importOwner = (event: IpcMainInvokeEvent): BrowserImportOwner => {
    let identity = browserImportOwners.get(event.sender);
    if (!identity) {
      identity = { webContentsId: event.sender.id, documentId: randomUUID() };
      browserImportOwners.set(event.sender, identity);
      const invalidate = () => {
        const previous = browserImportOwners.get(event.sender);
        if (previous) imports?.disposeOwner(previous);
        browserImportOwners.delete(event.sender);
      };
      event.sender.once('did-start-navigation', invalidate);
      event.sender.once('destroyed', invalidate);
    }
    return identity;
  };
  const engineVersions = new EngineVersionService({
    packaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    browserVersions: process.versions,
  });
  const scope = new AsyncLocalStorage<BrowserWindowController>();
  const selected = () => scope.getStore() ?? initialController;
  const controller = new Proxy(initialController, {
    get(_target, key) {
      const target = selected();
      const value = Reflect.get(target, key, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  const browserData = new Proxy(initialData, {
    get(_target, key) {
      const target = windows?.data(selected()) ?? initialData;
      const value = Reflect.get(target, key, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  const allWindows = () => windows?.all() ?? [initialController];
  const handle: typeof ipcMain.handle = (channel, listener) =>
    ipcMain.handle(channel, (event, ...args) => {
      if (windows?.admitsActions?.() === false && channel !== IPC.browserSetChromeLayout)
        throw new Error('BROWSER_CLOSING');
      const owner = allWindows().find(
        (candidate) =>
          candidate.isTrustedUIWebContents(event.sender) ||
          candidate.tabs.getTabs().some((tab) => tab.webContentsId === event.sender.id),
      );
      if (!owner) throw new Error('BROWSER_WINDOW_NOT_OWNED');
      if (owner.isPrivate && channel.startsWith('local-ai:'))
        throw new Error('PRIVATE_LOCAL_AI_UNAVAILABLE');
      return scope.run(owner, () => listener(event, ...args));
    });
  const promptOwners = new Map<string, BrowserWindowController>();
  const authentications = new Map<
    string,
    {
      owner: BrowserWindowController;
      finish: (credentials?: { username: string; password: string }) => void;
    }
  >();
  const login = (
    event: Electron.Event,
    contents: Electron.WebContents | null,
    details: Electron.AuthenticationResponseDetails,
    info: Electron.AuthInfo,
    callback: (username?: string, password?: string) => void,
  ) => {
    if (event.defaultPrevented || !contents) return;
    const owner = allWindows().find((candidate) =>
      candidate.tabs.getTabs().some((tab) => tab.webContentsId === contents.id),
    );
    if (!owner) return;
    let origin: string;
    try {
      const url = new URL(details.url);
      if (!['http:', 'https:'].includes(url.protocol)) return;
      origin = url.origin;
    } catch {
      return;
    }
    event.preventDefault();
    const id = randomUUID();
    let settled = false;
    const finish = (credentials?: { username: string; password: string }) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      authentications.delete(id);
      contents.removeListener('did-start-navigation', cancel);
      contents.removeListener('destroyed', cancel);
      if (credentials) callback(credentials.username, credentials.password);
      else callback();
    };
    const cancel = () => finish();
    const timeout = setTimeout(cancel, 60_000);
    contents.once('did-start-navigation', cancel);
    contents.once('destroyed', cancel);
    authentications.set(id, { owner, finish });
    owner.trustedWebContents.send(IPC.browserAuthentication, {
      id,
      target: info.isProxy ? `Sign in to proxy ${info.host}:${info.port}` : `Sign in to ${origin}`,
    });
  };
  app.on('login', login);

  const selectCertificate = (
    event: Electron.Event,
    contents: WebContents | null,
    url: string,
    certificates: Electron.Certificate[],
    callback: (certificate?: Electron.Certificate) => void,
  ) => {
    event.preventDefault();
    if (!contents) {
      callback();
      return;
    }
    const owner = allWindows().find((candidate) =>
      candidate.tabs.getTabs().some((tab) => tab.webContentsId === contents.id),
    );
    let origin: string;
    try {
      origin = new URL(url).origin;
      if (new URL(url).protocol !== 'https:') {
        callback();
        return;
      }
    } catch {
      callback();
      return;
    }
    if (!owner || certificates.length === 0 || certificates.length > 20) {
      callback();
      return;
    }
    let valid = true;
    const invalidate = () => {
      valid = false;
    };
    contents.once('did-start-navigation', invalidate);
    contents.once('destroyed', invalidate);
    void dialog
      .showMessageBox(owner.window, {
        title: 'Client certificate',
        message: `Choose a certificate for ${origin}`,
        detail: 'Only the selected certificate will be offered to this site.',
        buttons: [
          'Cancel',
          ...certificates.map((certificate) =>
            `${certificate.subjectName} — ${certificate.issuerName}`.slice(0, 180),
          ),
        ],
        defaultId: 0,
        cancelId: 0,
        noLink: true,
      })
      .then(
        ({ response }) => callback(valid ? certificates[response - 1] : undefined),
        () => callback(),
      )
      .finally(() => {
        contents.removeListener('did-start-navigation', invalidate);
        contents.removeListener('destroyed', invalidate);
      });
  };
  app.on('select-client-certificate', selectCertificate);

  const alternativeSearches = new Map<number, AbortController>();
  const pendingModelRequests = new Map<
    string,
    {
      abort: AbortController;
      owner: { tabId: string; webContentsId: number; documentId: string };
    }
  >();
  const workspaceSessions = new Map<string, BrowserModelSession>();
  const clearWorkspaceSession = (owner: { webContentsId: number; documentId: string }) => {
    workspaceSessions.delete(`${owner.webContentsId}:${owner.documentId}`);
  };
  callerContexts.on('document-invalidated', clearWorkspaceSession);
  const ensureTrustedUI = (event: IpcMainInvokeEvent) => {
    if (
      !controller.isTrustedUIWebContents(event.sender) ||
      event.senderFrame !== event.sender.mainFrame
    ) {
      throw new Error('TRUSTED_BROWSER_UI_REQUIRED');
    }
  };

  const siteContext = () => {
    const origin = controller.currentOrigin();
    if (!origin) return undefined;
    const contents = controller.activeWebsiteWebContents;
    const tab = controller.tabs.getTab(controller.tabs.getActiveTabId());
    return {
      tabId: tab.tabId,
      documentId: callerContexts.getDocumentId(contents),
      origin,
      partition: browserData.partitionFor(contents.session),
    };
  };
  const validateSiteContext = (expected: unknown) => {
    const actual = siteContext();
    if (
      !actual ||
      !expected ||
      typeof expected !== 'object' ||
      Object.entries(actual).some(
        ([key, value]) => (expected as Record<string, unknown>)[key] !== value,
      ) ||
      !callerContexts.hasDocumentAuthority(
        actual.tabId,
        controller.activeWebsiteWebContents.id,
        actual.documentId,
      )
    )
      throw new Error('SITE_CONTEXT_CHANGED');
    return actual;
  };
  const certificateContext = () => {
    const snapshot = controller.connectionSnapshot();
    const leaf = snapshot?.certificates[0];
    if (!snapshot || !leaf) return undefined;
    return {
      tabId: controller.tabs.getActiveTabId(),
      documentId: callerContexts.getDocumentId(controller.activeWebsiteWebContents),
      url: snapshot.url,
      fingerprint256: leaf.fingerprint256,
    };
  };
  handle(IPC.browserDiagnosticsExport, async (event) => {
    ensureTrustedUI(event);
    const compiledMainSha256 = await readFile(__filename)
      .then((bytes) => createHash('sha256').update(bytes).digest('hex'))
      .catch(() => undefined);
    const summary = browserDiagnostics({
      appVersion: app.getVersion(),
      versions: process.versions,
      platform: process.platform,
      architecture: process.arch,
      engineState: engine.state,
      windows: allWindows().length,
      tabs: allWindows().reduce((sum, owner) => sum + owner.tabs.getTabs().length, 0),
      downloads: browserData.downloads.list().length,
      activeDownloads: browserData.hasActiveDownloads(),
      buildFlavor: app.isPackaged ? readBrowserBuildFlavor(app.getAppPath()) : 'source',
      compiledMainSha256,
      rendererCrashes: windows?.rendererCrashes?.(),
      engineFailures: engine.failureSummary,
    });
    const target = await dialog.showSaveDialog(controller.window, {
      title: 'Export browser diagnostics',
      defaultPath: 'Oryn diagnostics.json',
      filters: [{ name: 'JSON', extensions: ['json'] }],
    });
    if (!target.canceled && target.filePath)
      await writeFile(target.filePath, `${JSON.stringify(summary, null, 2)}\n`, { mode: 0o600 });
  });
  handle(IPC.browserWindowAction, (event, action: unknown) => {
    ensureTrustedUI(event);
    const owner = selected();
    if (action === 'close') setImmediate(() => owner.window.close());
    else if (action === 'quit') setImmediate(() => app.quit());
    else throw new Error('INVALID_WINDOW_ACTION');
  });
  handle(IPC.browserDownloadDirectory, async (event) => {
    ensureTrustedUI(event);
    const selected = await dialog.showOpenDialog(controller.window, {
      title: 'Default download folder',
      properties: ['openDirectory', 'createDirectory'],
    });
    if (!selected.canceled && selected.filePaths[0]) {
      browserData.downloadPreferences.write({ directory: selected.filePaths[0] });
      controller.activeWebsiteWebContents.session.setDownloadPath(selected.filePaths[0]);
    }
    return browserData.downloadPreferences.read().directory;
  });
  handle(IPC.browserDownloadControl, async (event, id: unknown, action: unknown) => {
    ensureTrustedUI(event);
    requireIdentifier(id, 'download');
    if (typeof action !== 'string') throw new Error('INVALID_DOWNLOAD_ACTION');
    await browserData.controlDownload(id, action);
  });
  handle(IPC.browserAuthenticationRespond, (event, id: unknown, credentials: unknown) => {
    ensureTrustedUI(event);
    if (typeof id !== 'string') throw new Error('AUTHENTICATION_NOT_FOUND');
    const pending = authentications.get(id);
    if (!pending || pending.owner !== selected()) throw new Error('AUTHENTICATION_NOT_FOUND');
    if (credentials === null) {
      pending.finish();
      return;
    }
    if (!credentials || typeof credentials !== 'object') throw new Error('INVALID_CREDENTIALS');
    const value = credentials as Record<string, unknown>;
    if (
      typeof value.username !== 'string' ||
      typeof value.password !== 'string' ||
      value.username.length > 2048 ||
      value.password.length > 2048
    )
      throw new Error('INVALID_CREDENTIALS');
    pending.finish({ username: value.username, password: value.password });
  });
  handle(IPC.browserSetDefault, (event) => {
    ensureTrustedUI(event);
    if (!app.isPackaged) throw new Error('DEFAULT_BROWSER_REQUIRES_PACKAGED_APP');
    return app.setAsDefaultProtocolClient('http') && app.setAsDefaultProtocolClient('https');
  });
  handle(IPC.browserNewWindow, async (event) => {
    ensureTrustedUI(event);
    await windows?.create();
  });
  handle(IPC.browserNewPrivateWindow, async (event, value: unknown) => {
    ensureTrustedUI(event);
    if (
      value !== undefined &&
      (typeof value !== 'string' || value.length > 8192 || !/^https?:/u.test(value))
    )
      throw new Error('INVALID_PRIVATE_URL');
    await windows?.createPrivate?.(value as string | undefined);
  });
  handle(IPC.browserOpenFile, async (event) => {
    ensureTrustedUI(event);
    const result = await dialog.showOpenDialog(controller.window, {
      properties: ['openFile'],
      filters: [{ name: 'HTML documents', extensions: ['html', 'htm', 'xhtml', 'shtml'] }],
    });
    if (!result.canceled && result.filePaths[0])
      controller.navigate(pathToFileURL(result.filePaths[0]).toString());
  });
  handle('browser:native-file-drop', (event, values: unknown) => {
    const frame = event.senderFrame;
    const main = event.sender.mainFrame;
    const owner = allWindows().find(
      (candidate) =>
        candidate.isTrustedUIWebContents(event.sender) ||
        candidate.tabs.getTabs().some((tab) => tab.webContentsId === event.sender.id),
    );
    if (
      !owner ||
      !frame ||
      frame.processId !== main.processId ||
      frame.routingId !== main.routingId ||
      !Array.isArray(values) ||
      values.length > 16 ||
      values.some(
        (value) =>
          typeof value !== 'string' ||
          value.length > 8192 ||
          !isAbsolute(value) ||
          !/\.(html?|xhtml|shtml)$/iu.test(value),
      )
    )
      throw new Error('NATIVE_FILE_DROP_INVALID');
    for (const value of values) owner.createTab(pathToFileURL(value).toString(), true);
  });
  handle(IPC.browserGetState, (event) => {
    ensureTrustedUI(event);
    return controller.getState();
  });
  handle(IPC.browserImportDiscover, (event) => {
    ensureTrustedUI(event);
    if (controller.isPrivate) throw new Error('PRIVATE_IMPORT_UNAVAILABLE');
    importOwner(event);
    return importService().discover();
  });
  handle(IPC.browserImportPreview, async (event, value: unknown) => {
    ensureTrustedUI(event);
    if (controller.isPrivate || !value || typeof value !== 'object')
      throw new Error('IMPORT_REQUEST_INVALID');
    const request = value as { sourceId?: unknown; dataTypes?: unknown };
    if (
      Object.keys(value).some((key) => !['sourceId', 'dataTypes'].includes(key)) ||
      typeof request.sourceId !== 'string' ||
      !Array.isArray(request.dataTypes) ||
      request.dataTypes.some((type) => !['bookmarks', 'history', 'searchEngine'].includes(type))
    )
      throw new Error('IMPORT_REQUEST_INVALID');
    const owner = importOwner(event);
    const sources = await importService().discover();
    const source = sources.find((candidate) => candidate.sourceId === request.sourceId);
    if (!source) throw new Error('IMPORT_SOURCE_INVALID');
    let safariArchivePath: string | undefined;
    if (source.browser === 'safari') {
      const picked = await dialog.showOpenDialog(controller.window, {
        properties: ['openFile'],
        title: 'Choose Safari browsing-data export',
        filters: [{ name: 'Safari export', extensions: ['zip'] }],
      });
      if (picked.canceled) return undefined;
      safariArchivePath = picked.filePaths[0];
    }
    if (browserImportOwners.get(event.sender) !== owner) throw new Error('IMPORT_PREVIEW_EXPIRED');
    return importService().preview(owner, {
      sourceId: request.sourceId,
      dataTypes: request.dataTypes as BrowserImportDataType[],
      ...(safariArchivePath ? { safariArchivePath } : {}),
    });
  });
  handle(IPC.browserImportCommit, (event, token: unknown) => {
    ensureTrustedUI(event);
    if (controller.isPrivate || typeof token !== 'string' || token.length > 128)
      throw new Error('IMPORT_REQUEST_INVALID');
    return importService().commit(importOwner(event), token);
  });
  handle(IPC.browserImportCancel, (event, token: unknown) => {
    ensureTrustedUI(event);
    if (token !== undefined && (typeof token !== 'string' || token.length > 128))
      throw new Error('IMPORT_REQUEST_INVALID');
    importService().cancel(importOwner(event), token as string | undefined);
  });
  handle(IPC.browserExperienceCommand, async (event, value: unknown) => {
    ensureTrustedUI(event);
    if (
      !value ||
      typeof value !== 'object' ||
      typeof (value as { type?: unknown }).type !== 'string'
    ) {
      throw new Error('INVALID_BROWSER_EXPERIENCE_COMMAND');
    }
    const command = value as BrowserExperienceCommand;
    if (
      controller.isPrivate &&
      (command.type.startsWith('workspace-') ||
        command.type.startsWith('sidebar-') ||
        ['tab-new-window', 'tab-move-workspace', 'island-move-workspace'].includes(command.type))
    )
      throw new Error('PRIVATE_FEATURE_UNAVAILABLE');
    if (command.type === 'tab-new-window') {
      await windows?.transfer?.(selected(), command.tabId);
      return;
    }
    if (command.type === 'workspace-delete') {
      if (command.workspaceId === 'workspace_default')
        throw new Error('DEFAULT_WORKSPACE_REQUIRED');
      if (command.tabs !== 'close' && command.tabs !== 'move-to-default')
        throw new Error('INVALID_WORKSPACE_ACTION');
      if (command.tabs === 'close')
        for (const owner of allWindows()) {
          const ids = owner.tabs
            .getTabs()
            .filter((tab) => tab.workspaceId === command.workspaceId)
            .map((tab) => tab.tabId);
          if (!(await owner.tabs.closeTabs(ids))) {
            for (const window of allWindows()) window.tabs.recoverOrganization();
            return;
          }
        }
      browserData.experience.deleteWorkspace(command.workspaceId);
      for (const owner of allWindows()) owner.tabs.recoverOrganization();
      return;
    }
    if (command.type === 'island-move-workspace') {
      for (const owner of allWindows())
        owner.tabs.moveIslandToWorkspace(command.islandId, command.workspaceId);
      return;
    }
    await controller.updateBrowserExperience(command);
    for (const owner of allWindows()) if (owner !== selected()) owner.tabs.recoverOrganization();
  });
  handle(IPC.browserBookmarkCommand, (event, value: unknown) => {
    ensureTrustedUI(event);
    if (
      !value ||
      typeof value !== 'object' ||
      typeof (value as { type?: unknown }).type !== 'string'
    ) {
      throw new Error('INVALID_BOOKMARK_COMMAND');
    }
    const command = value as BrowserBookmarkCommand;
    switch (command.type) {
      case 'create-folder':
        browserData.bookmarks.createFolder(command.title, command.parentId ?? null);
        break;
      case 'edit':
        browserData.bookmarks.edit(command.bookmarkId, command.title, command.url);
        break;
      case 'folder-from-items':
        browserData.bookmarks.createFolderFromItems(command.bookmarkIds, command.title);
        break;
      case 'move':
        browserData.bookmarks.move(command.bookmarkIds, command.parentId, command.copy);
        break;
      case 'trash':
        browserData.bookmarks.trash(command.bookmarkIds);
        break;
      case 'restore':
        browserData.bookmarks.restore(command.bookmarkIds);
        break;
      case 'delete-permanently':
        browserData.bookmarks.deletePermanently(command.bookmarkIds);
        break;
      case 'favorite':
        browserData.bookmarks.setFavorite(command.bookmarkId, command.favorite);
        break;
      case 'thumbnail':
        browserData.bookmarks.setThumbnail(command.bookmarkId, command.thumbnail);
        break;
      case 'reorder':
        browserData.bookmarks.reorder(command.bookmarkId, command.order);
        break;
      case 'save-all-tabs': {
        const folder = browserData.bookmarks.createFolder(command.title ?? 'Saved tabs');
        for (const tab of controller.getState().tabs) {
          browserData.bookmarks.createBookmark(tab.title || tab.url, tab.url, folder.bookmarkId);
        }
        break;
      }
    }
  });
  handle(IPC.browserNewTab, (event, value: unknown) => {
    ensureTrustedUI(event);
    if (value !== undefined && (typeof value !== 'string' || value.length > 8192)) {
      throw new Error('INVALID_NAVIGATION_INPUT');
    }
    return controller.createTab(value);
  });
  handle(IPC.browserActivateTab, (event, tabId: unknown) => {
    ensureTrustedUI(event);
    if (typeof tabId !== 'string') {
      throw new Error('INVALID_TAB_ID');
    }
    controller.activateTab(tabId);
  });
  handle(IPC.browserCloseTab, (event, tabId: unknown) => {
    ensureTrustedUI(event);
    if (tabId !== undefined && typeof tabId !== 'string') {
      throw new Error('INVALID_TAB_ID');
    }
    return controller.closeTab(tabId);
  });
  handle(IPC.browserReopenClosedTab, (event) => {
    ensureTrustedUI(event);
    return controller.reopenRecentlyClosed();
  });
  handle(IPC.browserNavigate, (event, value: unknown) => {
    ensureTrustedUI(event);
    if (typeof value !== 'string' || value.length > 8192) {
      throw new Error('INVALID_NAVIGATION_INPUT');
    }
    controller.navigate(value);
  });
  handle(IPC.browserBack, (event) => {
    ensureTrustedUI(event);
    controller.back();
  });
  handle(IPC.browserForward, (event) => {
    ensureTrustedUI(event);
    controller.forward();
  });
  handle(IPC.browserReload, (event) => {
    ensureTrustedUI(event);
    controller.reload();
  });
  handle(IPC.browserStop, (event) => {
    ensureTrustedUI(event);
    controller.stop();
  });
  handle(IPC.browserOpenDevTools, (event) => {
    ensureTrustedUI(event);
    controller.openDevTools();
  });
  handle(IPC.browserGetData, (event, query: unknown) => {
    ensureTrustedUI(event);
    if (query !== undefined && (typeof query !== 'string' || query.length > 512)) {
      throw new Error('INVALID_HISTORY_QUERY');
    }
    const currentOrigin = controller.currentOrigin();
    return {
      history: browserData.history.search(query),
      bookmarks: browserData.bookmarks.list(),
      downloads: browserData.downloads.list(),
      downloadDirectory: browserData.downloadPreferences.read().directory,
      sitePermissions: currentOrigin
        ? browserData.sitePermissions.list(currentOrigin, siteContext()?.partition)
        : [],
      bookmarkBarVisible: browserData.bookmarks.isBarVisible(),
      currentOrigin,
      siteContext: siteContext(),
      recoveryNotices: [...recoveredStores],
      appearance: browserData.appearance.read(),
      connection: controller.connectionSnapshot(),
      certificateContext: certificateContext(),
      captureAccess: [
        ...new Map(
          allWindows()
            .flatMap((owner) => (windows?.data(owner) ?? initialData).captureSnapshot())
            .map((entry) => [entry.generation, entry]),
        ).values(),
      ],
      embeddedPermissions: currentOrigin
        ? browserData.listEmbeddedPermissions(currentOrigin, siteContext()?.partition)
        : [],
    };
  });
  handle(IPC.browserCaptureStop, async (event, id: unknown, generation: unknown) => {
    ensureTrustedUI(event);
    if (
      typeof id !== 'number' ||
      !Number.isSafeInteger(id) ||
      typeof generation !== 'string' ||
      generation.length > 128
    )
      throw new Error('CAPTURE_TARGET_INVALID');
    const targetData = [
      ...new Set(allWindows().map((owner) => windows?.data(owner) ?? initialData)),
    ].find((data) =>
      data
        .captureSnapshot()
        .some((entry) => entry.webContentsId === id && entry.generation === generation),
    );
    const targetContents = webContents.fromId(id);
    if (!targetData || !targetContents || targetContents.isDestroyed())
      throw new Error('CAPTURE_TARGET_CHANGED');
    let current = true;
    const invalidate = () => {
      current = false;
    };
    targetContents.on('did-start-navigation', invalidate);
    targetContents.once('destroyed', invalidate);
    try {
      const answer = await dialog.showMessageBox(controller.window, {
        type: 'warning',
        message: 'Stop capture and close this page?',
        detail:
          'Closing the page ends its camera, microphone and screen access. Unsaved work may be lost.',
        buttons: ['Keep page open', 'Stop and close page'],
        defaultId: 0,
        cancelId: 0,
      });
      if (answer.response !== 1) return false;
      if (targetContents.isDestroyed()) return true;
      if (
        !current ||
        !targetData
          .captureSnapshot()
          .some((entry) => entry.webContentsId === id && entry.generation === generation)
      )
        throw new Error('CAPTURE_TARGET_CHANGED');
      const contents = targetData.requestStopCapture(id);
      return contents ? requestPageClose(contents, controller.window) : true;
    } finally {
      targetContents.removeListener('did-start-navigation', invalidate);
      targetContents.removeListener('destroyed', invalidate);
    }
  });
  handle(IPC.browserCertificateExport, async (event, expected: unknown) => {
    ensureTrustedUI(event);
    const actual = certificateContext();
    if (
      !actual ||
      !expected ||
      typeof expected !== 'object' ||
      Object.entries(actual).some(
        ([key, value]) => (expected as Record<string, unknown>)[key] !== value,
      )
    )
      throw new Error('CERTIFICATE_DOCUMENT_CHANGED');
    const contents = controller.activeWebsiteWebContents;
    const document = callerContexts.getDocumentId(contents);
    const snapshot = controller.connectionSnapshot();
    if (!snapshot?.certificates.length) throw new Error('CERTIFICATE_UNAVAILABLE');
    const selectedFile = await dialog.showSaveDialog(controller.window, {
      defaultPath: 'certificate-chain.pem',
      filters: [{ name: 'PEM certificate chain', extensions: ['pem'] }],
    });
    if (selectedFile.canceled || !selectedFile.filePath) return;
    if (
      contents.isDestroyed() ||
      controller.activeWebsiteWebContents !== contents ||
      callerContexts.getDocumentId(contents) !== document
    )
      throw new Error('CERTIFICATE_DOCUMENT_CHANGED');
    await writeFile(selectedFile.filePath, exportCertificateChain(snapshot.certificates), {
      mode: 0o600,
    });
  });
  handle(
    IPC.browserEmbeddedPermissionRevoke,
    (event, embeddedOrigin: unknown, permission: unknown, expected: unknown) => {
      ensureTrustedUI(event);
      const context = validateSiteContext(expected);
      if (
        typeof embeddedOrigin !== 'string' ||
        typeof permission !== 'string' ||
        !isSupportedSitePermission(permission)
      )
        throw new Error('EMBEDDED_PERMISSION_INVALID');
      browserData.revokeEmbeddedPermission(
        context.origin,
        embeddedOrigin,
        permission,
        context.partition,
      );
    },
  );
  handle(IPC.browserAppearanceUpdate, (event, value: unknown) => {
    ensureTrustedUI(event);
    browserData.appearance.update(value as Parameters<typeof browserData.appearance.update>[0]);
  });
  handle(IPC.appUpdateGet, (event) => {
    ensureTrustedUI(event);
    return appUpdates.snapshot();
  });
  handle(IPC.appUpdateCheck, (event) => {
    ensureTrustedUI(event);
    return appUpdates.check(true);
  });
  handle(
    IPC.appUpdateDownload,
    (event, version: unknown, allowSkipped: unknown = false, approvalId: unknown) => {
      ensureTrustedUI(event);
      requireUpdateVersion(version);
      if (
        typeof allowSkipped !== 'boolean' ||
        typeof approvalId !== 'string' ||
        approvalId.length > 128
      )
        throw new Error('INVALID_UPDATE_DOWNLOAD');
      return appUpdates.download(version, allowSkipped, approvalId);
    },
  );
  handle(IPC.appUpdateCancelDownload, (event) => {
    ensureTrustedUI(event);
    return appUpdates.cancelDownload();
  });
  handle(IPC.appUpdateDefer, (event, version: unknown) => {
    ensureTrustedUI(event);
    requireUpdateVersion(version);
    return appUpdates.defer(version);
  });
  handle(IPC.appUpdateSkip, (event, version: unknown) => {
    ensureTrustedUI(event);
    requireUpdateVersion(version);
    return appUpdates.skip(version);
  });
  handle(IPC.appUpdateClearSkip, (event) => {
    ensureTrustedUI(event);
    return appUpdates.clearSkip();
  });
  handle(IPC.appUpdateRestart, (event, version: unknown, approvalId: unknown) => {
    ensureTrustedUI(event);
    requireUpdateVersion(version);
    if (typeof approvalId !== 'string' || approvalId.length > 128)
      throw new Error('INVALID_UPDATE_RESTART');
    return appUpdates.restart(version, approvalId);
  });
  handle(IPC.appUpdateSettingsUpdate, (event, value: unknown) => {
    ensureTrustedUI(event);
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('INVALID_APP_UPDATE_SETTINGS');
    }
    return appUpdates.updateSettings(value as Parameters<AppUpdateService['updateSettings']>[0]);
  });
  handle(IPC.huggingFacePageActionConfirm, async (event, trustMode: unknown) => {
    ensureTrustedUI(event);
    if (trustMode !== 'oryn-verified' && trustMode !== 'user-trusted') {
      throw new Error('HUGGING_FACE_PAGE_ACTION_TRUST_MODE_INVALID');
    }
    await controller.huggingFacePageAction.confirmInstall(trustMode);
  });
  handle(IPC.huggingFacePageActionCancel, async (event) => {
    ensureTrustedUI(event);
    await controller.huggingFacePageAction.cancelInstall();
  });
  handle(IPC.huggingFacePageActionOpen, async (event) => {
    ensureTrustedUI(event);
    return controller.openHuggingFaceModelWorkspace();
  });
  handle(IPC.huggingFacePageActionRetry, (event) => {
    ensureTrustedUI(event);
    controller.huggingFacePageAction.retry();
  });
  handle(IPC.huggingFacePageActionDetails, (event) => {
    ensureTrustedUI(event);
    controller.sendToPassiveOverlay(IPC.huggingFacePageActionDetails, controller.getState());
  });
  handle(IPC.huggingFaceCredentialGet, (event) => {
    ensureTrustedUI(event);
    return huggingFaceCredentials.status();
  });
  handle(IPC.huggingFaceCredentialSet, async (event, token: unknown) => {
    ensureTrustedUI(event);
    if (typeof token !== 'string' || token.length > 515) {
      throw new Error('HUGGING_FACE_TOKEN_INVALID');
    }
    const status = huggingFaceCredentials.save(token);
    await engine.updateHuggingFaceCredential(huggingFaceCredentials.token());
    controller.huggingFacePageAction.credentialsChanged();
    return status;
  });
  handle(IPC.huggingFaceCredentialClear, async (event) => {
    ensureTrustedUI(event);
    const status = huggingFaceCredentials.clear();
    await engine.updateHuggingFaceCredential(undefined);
    controller.huggingFacePageAction.credentialsChanged();
    return status;
  });
  const workspaceContext = async (event: IpcMainInvokeEvent) => {
    const owner = callerContexts.deriveModelWorkspace(event);
    const binding = browserData.modelWorkspaces.get(owner.reference);
    if (!binding) throw new Error('MODEL_WORKSPACE_REFERENCE_UNKNOWN');
    const profile = await engine.inspectModel(binding.model, binding.requirements);
    const trustMode = binding.trustMode ?? 'oryn-verified';
    const taskContract = taskContractForProfile(profile);
    const selectedFiles =
      profile.selectedFiles?.map((file) => file.name).sort() ?? profile.model.files;
    const filesMatch =
      !binding.selectedFiles ||
      JSON.stringify([...binding.selectedFiles].sort()) === JSON.stringify(selectedFiles);
    const packMatches = binding.runtimePack
      ? profile.runtimePack?.packId === binding.runtimePack.packId &&
        profile.runtimePack.packVersion === binding.runtimePack.packVersion
      : true;
    const trustEligible =
      trustMode === 'oryn-verified'
        ? isOrynVerified(profile)
        : isUserTrustEligible(profile) || isOrynVerified(profile);
    if (
      profile.model.repository !== binding.model.repository ||
      profile.exactRevision !== binding.model.revision ||
      !packMatches ||
      !filesMatch ||
      !trustEligible ||
      taskContract !== binding.taskContract ||
      !isModelWorkspaceTaskContract(binding.taskContract)
    ) {
      throw new Error('MODEL_WORKSPACE_REVALIDATION_FAILED');
    }
    const inventory = await engine.getModelInventory();
    if (
      !inventory.models.some(
        (item) =>
          item.installationState === 'installed' &&
          item.model.repository === binding.model.repository &&
          item.model.revision === binding.model.revision,
      )
    )
      throw new Error('MODEL_WORKSPACE_MODEL_REMOVED');
    return { owner, binding, profile, taskContract, trustMode };
  };
  handle(IPC.workspaceGetState, async (event): Promise<ModelWorkspaceState> => {
    try {
      const { binding, profile, taskContract, trustMode } = await workspaceContext(event);
      return {
        status: 'ready',
        repository: binding.model.repository,
        exactRevision: binding.model.revision,
        sourceURL: `https://huggingface.co/${binding.model.repository}/tree/${binding.model.revision}`,
        taskContract,
        task: profile.technicalCapabilities.task,
        ...(profile.taskProfile ? { taskProfile: structuredClone(profile.taskProfile) } : {}),
        trustMode,
        capabilities: {
          ...structuredClone(profile.technicalCapabilities.capabilities ?? {}),
          imageInput: profile.technicalCapabilities.inputModalities?.includes('image') === true,
        },
        ...(profile.generationProfile
          ? { generationProfile: structuredClone(profile.generationProfile) }
          : {}),
        ...(profile.runtimePlan
          ? {
              runtimeSelection: {
                candidateId: profile.runtimePlan.selected.candidateId,
                automatic: !profile.runtimePlan.reasonCodes.includes('USER_OVERRIDE'),
                policy: profile.runtimePlan.policy,
                foundationId: profile.runtimePlan.selected.foundation.id,
                foundationVersion: profile.runtimePlan.selected.foundation.version,
                adapterId: profile.runtimePlan.selected.adapter.id,
                adapterVersion: profile.runtimePlan.selected.adapter.version,
                backend: profile.runtimePlan.selected.backend,
                precision: profile.runtimePlan.selected.precision,
                ...(profile.runtimePlan.selected.quantization
                  ? { quantization: profile.runtimePlan.selected.quantization }
                  : {}),
                estimatedMemoryBytes: profile.runtimePlan.selected.estimatedMemoryBytes,
                fallbackCount: profile.runtimePlan.fallbacks.length,
                reasonCodes: profile.runtimePlan.reasonCodes,
                alternatives: profile.runtimePlan.fallbacks.map((candidate) => ({
                  candidateId: candidate.candidateId,
                  foundationId: candidate.foundation.id,
                  foundationVersion: candidate.foundation.version,
                  adapterId: candidate.adapter.id,
                  adapterVersion: candidate.adapter.version,
                  backend: candidate.backend,
                  precision: candidate.precision,
                  ...(candidate.quantization ? { quantization: candidate.quantization } : {}),
                  estimatedMemoryBytes: candidate.estimatedMemoryBytes,
                })),
              },
            }
          : {}),
        appearance: browserData.appearance.read(),
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : '';
      if (
        message.includes('MODEL_WORKSPACE_MODEL_REMOVED') ||
        message.includes('MODEL_NOT_INSTALLED')
      ) {
        return {
          status: 'model-removed',
          message:
            'This exact model is no longer installed. Return to its Hugging Face page to load it again.',
        };
      }
      if (message.includes('REVALIDATION') || message.includes('RUNTIME_DEPENDENCY')) {
        return {
          status: 'runtime-revoked',
          message:
            'The signed runtime for this model is no longer active. Update Oryn or choose another supported model.',
        };
      }
      if (message.includes('REFERENCE_UNKNOWN'))
        return {
          status: 'error',
          message: 'This model workspace reference is no longer available.',
        };
      return { status: 'error', message: 'Oryn could not revalidate this model workspace.' };
    }
  });
  handle(IPC.workspaceInvoke, async (event, input: unknown) => {
    try {
      const { owner, profile, taskContract, trustMode } = await workspaceContext(event);
      const validated = validateModelWorkspaceInput(
        taskContract,
        input,
        profile.technicalCapabilities.capabilities,
      );
      const sessionKey = `${owner.webContentsId}:${owner.documentId}`;
      let modelSession = workspaceSessions.get(sessionKey);
      if (!modelSession) {
        modelSession = await modelSessions.attachModelWorkspaceSession(
          profile,
          taskContract,
          owner,
          trustMode,
        );
        workspaceSessions.set(sessionKey, modelSession);
      }
      return {
        ok: true as const,
        value: await jobs.invoke(owner, { sessionId: modelSession.sessionId, input: validated }),
      };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceCancel, async (event, jobId: unknown) => {
    try {
      const owner = callerContexts.deriveModelWorkspace(event);
      if (typeof jobId !== 'string') throw new BrowserJobError('JOB_NOT_OWNED');
      await jobs.cancelOwned(jobId, owner);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceArtifactCreateBegin, async (event, metadata: unknown) => {
    try {
      const owner = callerContexts.deriveModelWorkspace(event);
      return {
        ok: true as const,
        value: await artifacts.beginCreate(
          owner,
          metadata as Parameters<ArtifactManager['beginCreate']>[1],
        ),
      };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceArtifactCreateChunk, async (event, uploadId: unknown, chunk: unknown) => {
    try {
      const owner = callerContexts.deriveModelWorkspace(event);
      if (typeof uploadId !== 'string') throw new ArtifactError('ARTIFACT_INVALID');
      await artifacts.appendChunk(owner, uploadId, chunk);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceArtifactCreateFinish, async (event, uploadId: unknown) => {
    try {
      const owner = callerContexts.deriveModelWorkspace(event);
      if (typeof uploadId !== 'string') throw new ArtifactError('ARTIFACT_INVALID');
      return { ok: true as const, value: await artifacts.finishCreate(owner, uploadId) };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceArtifactCreateAbort, async (event, uploadId: unknown) => {
    const owner = callerContexts.deriveModelWorkspace(event);
    if (typeof uploadId === 'string') await artifacts.abortCreate(owner, uploadId);
  });
  handle(IPC.workspaceArtifactGetUrl, (event, artifactId: unknown) => {
    try {
      const owner = callerContexts.deriveModelWorkspace(event);
      if (typeof artifactId !== 'string') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      return { ok: true as const, value: artifacts.getUrl(owner, artifactId) };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceArtifactSaveAs, async (event, artifactId: unknown, options: unknown) => {
    try {
      const owner = callerContexts.deriveModelWorkspace(event);
      if (typeof artifactId !== 'string') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      const metadata = artifacts.get(owner, artifactId);
      const selected = await dialog.showSaveDialog(controller.window, {
        title: 'Save Oryn result',
        defaultPath: validateSaveOptions(options, metadata.name),
      });
      if (!selected.canceled && selected.filePath)
        await artifacts.saveTo(owner, artifactId, selected.filePath);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceArtifactRelease, async (event, artifactId: unknown) => {
    try {
      const owner = callerContexts.deriveModelWorkspace(event);
      if (typeof artifactId !== 'string') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      await artifacts.release(owner, artifactId);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: workspaceError(error) };
    }
  });
  handle(IPC.workspaceOpenSource, async (event) => {
    const { binding } = await workspaceContext(event);
    controller.createTab(
      `https://huggingface.co/${binding.model.repository}/tree/${binding.model.revision}`,
      true,
    );
  });
  handle(IPC.browserHistoryDelete, (event, historyId: unknown) => {
    ensureTrustedUI(event);
    requireIdentifier(historyId, 'history');
    browserData.history.delete(historyId);
  });
  handle(IPC.browserClearBrowsingData, async (event, category: unknown) => {
    ensureTrustedUI(event);
    const session = controller.activeWebsiteWebContents.session;
    if (browserData.partitionFor(session) !== 'default')
      throw new Error('ORDINARY_PARTITION_REQUIRED');
    if (category === 'cookies') await session.clearStorageData();
    else if (category === 'cache') await session.clearCache();
    else if (category === 'permissions') {
      browserData.sitePermissions.clear(undefined, 'default');
      for (const owner of allWindows())
        for (const tab of owner.tabs.getTabs())
          for (const permission of [
            'camera',
            'microphone',
            'media',
            'geolocation',
            'notifications',
            'clipboard-read',
          ])
            browserData.clearPagePermission(tab.view.webContents, permission);
    } else throw new Error('INVALID_BROWSING_DATA_CATEGORY');
  });
  handle(IPC.browserHistoryClear, (event, range: unknown = 'all') => {
    ensureTrustedUI(event);
    if (!['hour', 'day', 'all'].includes(String(range))) throw new Error('INVALID_HISTORY_RANGE');
    browserData.history.clear(
      range === 'hour' ? Date.now() - 3600000 : range === 'day' ? Date.now() - 86400000 : 0,
    );
  });
  const pendingImports = new Map<
    number,
    { token: string; nodes: BookmarkNode[]; expires: number }
  >();
  const importOwners = new WeakSet<WebContents>();
  handle(IPC.browserBookmarkImport, async (event, token?: unknown) => {
    ensureTrustedUI(event);
    if (token !== undefined) {
      const pending = pendingImports.get(event.sender.id);
      pendingImports.delete(event.sender.id);
      if (!pending || pending.token !== token || pending.expires < Date.now())
        throw new Error('BOOKMARK_IMPORT_EXPIRED');
      browserData.bookmarks.importNodes(pending.nodes);
      return undefined;
    }
    pendingImports.delete(event.sender.id);
    const selected = await dialog.showOpenDialog(controller.window, {
      title: 'Import bookmarks',
      properties: ['openFile'],
      filters: [{ name: 'Bookmark HTML', extensions: ['html', 'htm'] }],
    });
    if (selected.canceled || !selected.filePaths[0]) return undefined;
    const file = await open(selected.filePaths[0], 'r');
    let html: string;
    try {
      const buffer = Buffer.alloc(MAX_BOOKMARK_IMPORT_BYTES + 1);
      const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
      if (bytesRead > MAX_BOOKMARK_IMPORT_BYTES) throw new Error('BOOKMARK_IMPORT_TOO_LARGE');
      html = buffer.subarray(0, bytesRead).toString('utf8');
    } finally {
      await file.close();
    }
    const nodes = parseBookmarkHTML(html);
    const value = { token: randomUUID(), nodes, expires: Date.now() + 300_000 };
    pendingImports.set(event.sender.id, value);
    if (!importOwners.has(event.sender)) {
      importOwners.add(event.sender);
      event.sender.once('destroyed', () => pendingImports.delete(event.sender.id));
    }
    return { token: value.token, count: nodes.length };
  });
  handle(IPC.browserBookmarkExport, async (event) => {
    ensureTrustedUI(event);
    const html = exportBookmarkHTML(browserData.bookmarks.list());
    const selected = await dialog.showSaveDialog(controller.window, {
      title: 'Export bookmarks',
      defaultPath: 'Oryn bookmarks.html',
      filters: [{ name: 'Bookmark HTML', extensions: ['html'] }],
    });
    if (!selected.canceled && selected.filePath)
      await writeFile(selected.filePath, html, { mode: 0o600 });
  });
  handle(IPC.browserBookmarkAdd, (event, title: unknown, url: unknown) => {
    ensureTrustedUI(event);
    if (typeof title !== 'string' || typeof url !== 'string' || title.length > 512) {
      throw new Error('INVALID_BOOKMARK');
    }
    browserData.bookmarks.createBookmark(title, url);
  });
  handle(IPC.browserBookmarkDelete, (event, bookmarkId: unknown) => {
    ensureTrustedUI(event);
    requireIdentifier(bookmarkId, 'bookmark');
    browserData.bookmarks.delete(bookmarkId);
  });
  handle(IPC.browserBookmarkBarVisible, (event, visible: unknown) => {
    ensureTrustedUI(event);
    if (typeof visible !== 'boolean') {
      throw new Error('INVALID_BOOKMARK_BAR_STATE');
    }
    browserData.bookmarks.setBarVisible(visible);
  });
  handle(
    IPC.browserPermissionSet,
    (event, permission: unknown, decision: unknown, expected: unknown) => {
      ensureTrustedUI(event);
      const { origin, partition } = validateSiteContext(expected);
      if (
        !origin ||
        typeof permission !== 'string' ||
        !isSupportedSitePermission(permission) ||
        !['allow', 'deny', 'ask'].includes(String(decision))
      ) {
        throw new Error('INVALID_SITE_PERMISSION');
      }
      if (permission === 'sound') {
        for (const owner of allWindows())
          for (const tab of owner.tabs.getTabs()) {
            if (tab.view.webContents.session !== controller.activeWebsiteWebContents.session)
              continue;
            try {
              if (new URL(tab.url).origin === origin)
                tab.view.webContents.setAudioMuted(decision === 'deny');
            } catch {
              /* Internal tabs have no site setting. */
            }
          }
      }
      browserData.clearSitePagePermissions(origin, partition, permission);
      if (decision === 'allow' && ['camera', 'microphone', 'display-capture'].includes(permission))
        browserData.allowFreshCaptureConsent(controller.activeWebsiteWebContents);
      if (decision === 'ask') browserData.sitePermissions.clear(origin, partition, permission);
      else
        browserData.sitePermissions.set(
          origin,
          permission,
          decision as 'allow' | 'deny',
          partition,
        );
    },
  );
  handle(IPC.browserClearSiteData, async (event, expected: unknown) => {
    ensureTrustedUI(event);
    const { origin } = validateSiteContext(expected);
    if (!origin) {
      throw new Error('SITE_ORIGIN_UNAVAILABLE');
    }
    await browserData.clearSiteData(controller.activeWebsiteWebContents.session, origin);
  });
  handle(IPC.browserFind, (event, query: unknown, forward: unknown) => {
    ensureTrustedUI(event);
    if (typeof query !== 'string' || query.length > 2048) {
      throw new Error('INVALID_FIND_QUERY');
    }
    controller.findInPage(query, forward !== false);
  });
  handle(IPC.browserStopFind, (event) => {
    ensureTrustedUI(event);
    controller.stopFindInPage();
  });
  handle(IPC.browserShowFind, (event) => {
    ensureTrustedUI(event);
    controller.showFindInPage();
  });
  handle(IPC.browserSetZoom, (event, action: unknown) => {
    ensureTrustedUI(event);
    if (action !== 'in' && action !== 'out' && action !== 'reset') {
      throw new Error('INVALID_ZOOM_ACTION');
    }
    controller.setZoom(action);
  });
  handle(IPC.browserPrint, async (event) => {
    ensureTrustedUI(event);
    await controller.printPage();
  });
  handle(IPC.browserDismissPanelAndClickPage, (event, x: unknown, y: unknown, button: unknown) => {
    ensureTrustedUI(event);
    if (event.sender !== controller.trustedWebContents) throw new Error('UNTRUSTED_UI');
    if (
      typeof x !== 'number' ||
      !Number.isFinite(x) ||
      typeof y !== 'number' ||
      !Number.isFinite(y) ||
      (button !== 'left' && button !== 'middle' && button !== 'right')
    ) {
      throw new Error('INVALID_PAGE_CLICK');
    }
    controller.dismissPanelAndClickPage(x, y, button);
  });
  handle(IPC.browserSetChromeLayout, (event, value: unknown) => {
    ensureTrustedUI(event);
    if (!isBrowserChromeLayout(value)) {
      throw new Error('INVALID_CHROME_LAYOUT');
    }
    controller.setChromeLayout(value);
  });
  handle(IPC.browserSetPassiveOverlayLayout, (event, value: unknown) => {
    ensureTrustedUI(event);
    if (!isBrowserPassiveOverlayLayout(value)) {
      throw new Error('INVALID_PASSIVE_OVERLAY_LAYOUT');
    }
    controller.setPassiveOverlayLayout(value);
  });
  handle(IPC.localAIGetCapabilities, async (event): Promise<RuntimeCapabilities> => {
    callerContexts.derive(event, controller.activeWebsiteWebContents);
    try {
      const capabilities = await engine.getRuntimeCapabilities();
      const generationLane = capabilities.scheduler.lanes.find((lane) => lane.id === 'generation');
      return {
        runtimeAvailable: true,
        runtimeState: 'ready',
        sdkVersion: '0.1.0',
        browserApiVersion: '1',
        supportedModelProviders: capabilities.providers,
        ...(capabilities.taskContracts ? { taskContracts: capabilities.taskContracts } : {}),
        features: {
          modelSelection: capabilities.features.modelInspection === true,
          taskProfiles: capabilities.features.taskProfiles === true,
          modelRequests: capabilities.features.modelInspection,
          modelSubstitution: Boolean(modelResolver && modelSessions),
          streaming: capabilities.features.streaming,
          artifacts: true,
          artifactUrls: true,
          cancellation: true,
          queueing: true,
          multipleSubmittedJobs: true,
        },
        scheduler: generationLane
          ? {
              generation: {
                globalAcrossOrigins: generationLane.global,
                maxConcurrentRunning: generationLane.capacity,
              },
            }
          : undefined,
      };
    } catch {
      return unavailableCapabilities(engine.state);
    }
  });
  handle(IPC.localAIModelRequest, async (event, operationId: unknown, request: unknown) => {
    let trustedOperationId: string | undefined;
    let trustedOrigin: string | undefined;
    try {
      const owner = callerContexts.derive(event, controller.activeWebsiteWebContents);
      requireModelOperationId(operationId);
      trustedOperationId = operationId;
      trustedOrigin = owner.origin;
      const key = `${owner.webContentsId}:${operationId}`;
      if (pendingModelRequests.has(key)) throw new Error('MODEL_REQUEST_DUPLICATE');
      const abort = new AbortController();
      pendingModelRequests.set(key, { abort, owner });
      try {
        const session = await modelSessions.requestSession(request, owner, {
          operationId,
          signal: abort.signal,
          onProgress: (progress) => {
            if (!controller.window.isDestroyed()) {
              controller.publishModelOperation({
                operationId,
                origin: owner.origin,
                stage: progress.stage,
                ...('model' in progress
                  ? {
                      repository: progress.model.repository,
                      revision: progress.model.revision,
                    }
                  : {}),
                ...('downloadedBytes' in progress
                  ? {
                      downloadedBytes: progress.downloadedBytes,
                      totalBytes: progress.totalBytes,
                    }
                  : {}),
                ...('progress' in progress ? { progress: progress.progress } : {}),
              });
            }
            if (!event.sender.isDestroyed()) {
              event.sender.send(IPC.localAIModelRequestProgress, {
                operationId,
                progress: publicProgress(progress),
              });
            }
          },
        });
        return { ok: true as const, value: publicSession(session) };
      } finally {
        pendingModelRequests.delete(key);
      }
    } catch (error) {
      const publicError = websiteError(error);
      if (trustedOperationId && trustedOrigin && !controller.window.isDestroyed()) {
        controller.publishModelOperation({
          operationId: trustedOperationId,
          origin: trustedOrigin,
          stage: publicError.code === 'REQUEST_ABORTED' ? 'cancelled' : 'failed',
          errorCode: publicError.code,
          ...trustedResourceErrorDetails(error),
        });
      }
      return { ok: false as const, error: publicError };
    }
  });
  handle(IPC.localAIModelRequestCancel, (event, operationId: unknown) => {
    const owner = callerContexts.derive(event, controller.activeWebsiteWebContents);
    requireModelOperationId(operationId);
    const pending = pendingModelRequests.get(`${owner.webContentsId}:${operationId}`);
    if (
      pending &&
      pending.owner.tabId === owner.tabId &&
      pending.owner.documentId === owner.documentId
    ) {
      pending.abort.abort();
    }
  });
  handle(IPC.localAIModelOperationCancelTrusted, (event, operationId: unknown) => {
    ensureTrustedUI(event);
    requireModelOperationId(operationId);
    for (const [key, pending] of pendingModelRequests) {
      if (key.endsWith(`:${operationId}`)) pending.abort.abort();
    }
    void controller.huggingFacePageAction.cancelInstall(operationId);
  });
  handle(IPC.localAIModelSessionRelease, async (event, sessionId: unknown) => {
    try {
      const owner = callerContexts.deriveManaged(event);
      if (typeof sessionId !== 'string' || !/^session_[a-zA-Z0-9_-]{1,120}$/u.test(sessionId)) {
        throw new BrowserModelSessionError('SESSION_NOT_OWNED');
      }
      await modelSessions.releaseOwned(sessionId, owner);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIArtifactCreateBegin, async (event, metadata: unknown) => {
    try {
      const owner = callerContexts.derive(event, controller.activeWebsiteWebContents);
      return {
        ok: true as const,
        value: await artifacts.beginCreate(
          owner,
          metadata as Parameters<ArtifactManager['beginCreate']>[1],
        ),
      };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIArtifactCreateChunk, async (event, uploadId: unknown, chunk: unknown) => {
    try {
      const owner = callerContexts.derive(event, controller.activeWebsiteWebContents);
      if (typeof uploadId !== 'string') throw new ArtifactError('ARTIFACT_INVALID');
      await artifacts.appendChunk(owner, uploadId, chunk);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIArtifactCreateFinish, async (event, uploadId: unknown) => {
    try {
      const owner = callerContexts.derive(event, controller.activeWebsiteWebContents);
      if (typeof uploadId !== 'string') throw new ArtifactError('ARTIFACT_INVALID');
      return { ok: true as const, value: await artifacts.finishCreate(owner, uploadId) };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIArtifactCreateAbort, async (event, uploadId: unknown) => {
    const owner = callerContexts.derive(event, controller.activeWebsiteWebContents);
    if (typeof uploadId === 'string') await artifacts.abortCreate(owner, uploadId);
  });
  handle(IPC.localAIArtifactGet, (event, artifactId: unknown) => {
    try {
      const owner = callerContexts.deriveManaged(event);
      if (typeof artifactId !== 'string') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      return { ok: true as const, value: artifacts.get(owner, artifactId) };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIArtifactGetUrl, (event, artifactId: unknown) => {
    try {
      const owner = callerContexts.deriveManaged(event);
      if (typeof artifactId !== 'string') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      return { ok: true as const, value: artifacts.getUrl(owner, artifactId) };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIArtifactSaveAs, async (event, artifactId: unknown, options: unknown) => {
    try {
      const owner = callerContexts.derive(event, controller.activeWebsiteWebContents);
      if (typeof artifactId !== 'string') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      const metadata = artifacts.get(owner, artifactId);
      const suggested = validateSaveOptions(options, metadata.name);
      const selected = await dialog.showSaveDialog(controller.window, {
        title: 'Save Local AI artifact',
        defaultPath: suggested,
      });
      if (!selected.canceled && selected.filePath) {
        await artifacts.saveTo(owner, artifactId, selected.filePath);
      }
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIArtifactRelease, async (event, artifactId: unknown) => {
    try {
      const owner = callerContexts.deriveManaged(event);
      if (typeof artifactId !== 'string') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      await artifacts.release(owner, artifactId);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIJobInvoke, async (event, request: unknown) => {
    try {
      const owner = callerContexts.deriveManaged(event);
      orynSettings.assertBackgroundAllowed(owner.tabId === controller.activeTabId);
      return { ok: true as const, value: await jobs.invoke(owner, request) };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });
  handle(IPC.localAIJobCancel, async (event, jobId: unknown) => {
    try {
      const owner = callerContexts.deriveManaged(event);
      if (typeof jobId !== 'string') throw new BrowserJobError('JOB_NOT_OWNED');
      await jobs.cancelOwned(jobId, owner);
      return { ok: true as const, value: undefined };
    } catch (error) {
      return { ok: false as const, error: websiteError(error) };
    }
  });

  const forwardJobEvent = (value: {
    owner: { tabId: string; webContentsId: number; documentId: string; origin?: string };
    event: unknown;
  }) => {
    const target = webContents.fromId(value.owner.webContentsId);
    if (
      target &&
      !target.isDestroyed() &&
      callerContexts.hasDocumentAuthority(
        value.owner.tabId,
        value.owner.webContentsId,
        value.owner.documentId,
      )
    ) {
      target.send(
        value.owner.origin === 'oryn://model-workspace'
          ? IPC.workspaceJobEvent
          : IPC.localAIJobEvent,
        value.event,
      );
    }
  };
  jobs.on('website-event', forwardJobEvent);
  handle(IPC.localAIJobActivityGet, (event) => {
    ensureTrustedUI(event);
    return jobs.activitySnapshot();
  });
  const forwardJobActivity = (activity: { running: number; queued: number }) => {
    for (const owner of allWindows())
      if (!owner.isPrivate && !owner.window.isDestroyed()) {
        owner.trustedWebContents.send(IPC.localAIJobActivityChanged, activity);
        owner.sendToPassiveOverlay(IPC.localAIJobActivityChanged, activity);
      }
  };
  jobs.on('activity-changed', forwardJobActivity);
  handle(IPC.localAIJobCancelTrusted, async (event, jobId: unknown) => {
    ensureTrustedUI(event);
    if (typeof jobId !== 'string') throw new Error('INVALID_JOB_ID');
    await jobs.cancelTrusted(jobId);
  });
  handle(IPC.localAIJobCancelSite, async (event, origin: unknown) => {
    ensureTrustedUI(event);
    if (typeof origin !== 'string' || new URL(origin).origin !== origin) {
      throw new Error('INVALID_ORIGIN');
    }
    await jobs.cancelAllForOrigin(origin);
  });
  handle(IPC.localAIJobCancelAll, async (event) => {
    ensureTrustedUI(event);
    await jobs.cancelAllTrusted();
  });
  handle(IPC.localAIActivityGet, (event) => {
    ensureTrustedUI(event);
    return { settings: browserData.aiActivity.settings(), records: browserData.aiActivity.list() };
  });
  handle(IPC.localAIActivitySettingsUpdate, (event, settings: unknown) => {
    ensureTrustedUI(event);
    if (typeof settings !== 'object' || settings === null || Array.isArray(settings)) {
      throw new Error('INVALID_AI_ACTIVITY_SETTINGS');
    }
    return browserData.aiActivity.updateSettings(
      settings as Parameters<typeof browserData.aiActivity.updateSettings>[0],
    );
  });
  handle(IPC.localAIActivityClear, (event) => {
    ensureTrustedUI(event);
    browserData.aiActivity.clear();
  });
  handle(IPC.localAIArtifactSaveTrusted, async (event, artifactId: unknown) => {
    ensureTrustedUI(event);
    if (typeof artifactId !== 'string' || !/^artifact_[A-Za-z0-9_-]{20,80}$/u.test(artifactId)) {
      throw new Error('INVALID_ARTIFACT_ID');
    }
    const record = artifacts.registry.get(artifactId);
    if (!record || record.state !== 'ready') throw new Error('ARTIFACT_NOT_FOUND');
    const selected = await dialog.showSaveDialog(controller.window, {
      title: 'Save Local AI result',
      defaultPath: validateSaveOptions(undefined, record.metadata.name),
    });
    if (!selected.canceled && selected.filePath) {
      await artifacts.saveTrustedTo(artifactId, selected.filePath);
    }
  });
  handle(IPC.localAIModelsGet, (event) => {
    ensureTrustedUI(event);
    return modelManager.snapshot();
  });
  handle(IPC.localAIRegistryStatus, (event) => {
    ensureTrustedUI(event);
    return modelManager.registryStatus();
  });
  handle(IPC.localAIRegistryRefresh, (event, enabled: unknown) => {
    ensureTrustedUI(event);
    if (typeof enabled !== 'boolean') throw new Error('INVALID_REGISTRY_SETTING');
    return modelManager.setRegistryRefresh(enabled);
  });
  handle(IPC.localAIAlternativesSearch, async (event, installationId: unknown) => {
    ensureTrustedUI(event);
    requireInstallationId(installationId);
    const id = event.sender.id;
    alternativeSearches.get(id)?.abort();
    const abort = new AbortController();
    alternativeSearches.set(id, abort);
    const stop = () => abort.abort();
    event.sender.once('destroyed', stop);
    try {
      return await modelManager.checkBetterAlternatives(installationId, abort.signal);
    } finally {
      event.sender.removeListener('destroyed', stop);
      if (alternativeSearches.get(id) === abort) alternativeSearches.delete(id);
    }
  });
  handle(IPC.localAIAlternativesCancel, (event) => {
    ensureTrustedUI(event);
    alternativeSearches.get(event.sender.id)?.abort();
  });
  handle(IPC.localAIModelSubstitutes, (event, installationId: unknown) => {
    ensureTrustedUI(event);
    requireInstallationId(installationId);
    return modelManager.modelSubstitutes(installationId);
  });
  handle(IPC.localAIEngineVersionsGet, (event) => {
    ensureTrustedUI(event);
    return engineVersions.list();
  });
  handle(IPC.localAIEngineVersionCheck, (event, id: unknown) => {
    ensureTrustedUI(event);
    return engineVersions.check(id);
  });
  handle(IPC.localAIRuntimePacksGet, (event) => {
    ensureTrustedUI(event);
    return modelManager.runtimePacks();
  });
  handle(IPC.localAIRuntimePacksCleanup, (event) => {
    ensureTrustedUI(event);
    return modelManager.cleanupRuntimePacks();
  });
  handle(IPC.localAIModelUnloadTrusted, async (event, installationId: unknown) => {
    ensureTrustedUI(event);
    requireInstallationId(installationId);
    await modelManager.unload(installationId);
  });
  handle(
    IPC.localAIModelRuntimeOverrideTrusted,
    async (event, installationId: unknown, candidateId: unknown) => {
      ensureTrustedUI(event);
      requireInstallationId(installationId);
      if (
        candidateId !== undefined &&
        (typeof candidateId !== 'string' || !/^candidate-[0-9a-f]{24}$/u.test(candidateId))
      ) {
        throw new Error('INVALID_RUNTIME_CANDIDATE_ID');
      }
      await modelManager.setRuntimeOverride(installationId, candidateId);
    },
  );
  handle(IPC.localAIModelRemovePreviewTrusted, (event, installationId: unknown) => {
    ensureTrustedUI(event);
    requireInstallationId(installationId);
    return modelManager.previewRemoval(installationId);
  });
  handle(IPC.localAIModelRemoveTrusted, async (event, installationId: unknown, action: unknown) => {
    ensureTrustedUI(event);
    requireInstallationId(installationId);
    if (!['delete', 'when-idle', 'cancel-and-delete'].includes(String(action))) {
      throw new Error('INVALID_MODEL_REMOVE_ACTION');
    }
    const result = await modelManager.remove(
      installationId,
      action as 'delete' | 'when-idle' | 'cancel-and-delete',
    );
    return result.state;
  });
  handle(IPC.localAIModelSourceOpenTrusted, async (event, installationId: unknown) => {
    ensureTrustedUI(event);
    requireInstallationId(installationId);
    controller.createTab(await modelManager.sourceUrl(installationId), true);
  });
  handle(
    IPC.localAIModelReferenceSourceOpenTrusted,
    async (event, repository: unknown, revision: unknown) => {
      ensureTrustedUI(event);
      if (
        typeof repository !== 'string' ||
        !/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/u.test(repository) ||
        (revision !== undefined &&
          (typeof revision !== 'string' || !/^[0-9a-f]{40,64}$/u.test(revision)))
      ) {
        throw new Error('INVALID_MODEL_REFERENCE');
      }
      const suffix = revision ? `/tree/${revision}` : '';
      controller.createTab(`https://huggingface.co/${repository}${suffix}`, true);
    },
  );
  handle(IPC.localAISettingsGet, (event) => {
    ensureTrustedUI(event);
    return orynSettings.snapshot();
  });
  handle(IPC.localAISettingsRankingUpdate, async (event, value: unknown) => {
    ensureTrustedUI(event);
    await orynSettings.updateRanking(value as Parameters<OrynSettingsService['updateRanking']>[0]);
  });
  handle(IPC.localAISettingsRoleUpdate, async (event, value: unknown) => {
    ensureTrustedUI(event);
    await orynSettings.updateRole(value as Parameters<OrynSettingsService['updateRole']>[0]);
  });
  handle(IPC.localAISettingsComputeUpdate, async (event, value: unknown) => {
    ensureTrustedUI(event);
    await orynSettings.updateCompute(value as Parameters<OrynSettingsService['updateCompute']>[0]);
  });
  handle(IPC.localAISettingsSiteReset, (event, origin: unknown) => {
    ensureTrustedUI(event);
    if (typeof origin !== 'string') throw new Error('INVALID_ORIGIN');
    orynSettings.resetSiteChoices(origin);
  });
  handle(IPC.localAISettingsSiteAskAgain, (event, origin: unknown) => {
    ensureTrustedUI(event);
    if (typeof origin !== 'string') throw new Error('INVALID_ORIGIN');
    orynSettings.askSiteAgain(origin);
  });
  handle(IPC.localAISettingsSiteRevoke, async (event, origin: unknown, cancelActive: unknown) => {
    ensureTrustedUI(event);
    if (typeof origin !== 'string' || typeof cancelActive !== 'boolean') {
      throw new Error('INVALID_SITE_REVOCATION');
    }
    await orynSettings.revokeSite(origin, cancelActive);
  });
  handle(IPC.localAISettingsStorageCleanup, async (event, target: unknown) => {
    ensureTrustedUI(event);
    if (target !== 'artifacts' && target !== 'temporary') {
      throw new Error('INVALID_STORAGE_CLEANUP');
    }
    await orynSettings.cleanupStorage(target);
  });
  handle(IPC.localAISettingsDeveloperUpdate, (event, value: unknown) => {
    ensureTrustedUI(event);
    orynSettings.updateDeveloper(value as Parameters<OrynSettingsService['updateDeveloper']>[0]);
  });

  const forwardPermissionPrompt = (prompt: { promptId: string }) => {
    const owner = selected();
    if (!owner.window.isDestroyed()) {
      promptOwners.set(prompt.promptId, owner);
      owner.trustedWebContents.send(IPC.localAIPermissionPrompt, prompt);
    }
  };
  const forwardAppUpdate = (state: unknown) => {
    for (const owner of allWindows())
      if (!owner.window.isDestroyed())
        owner.trustedWebContents.send(IPC.appUpdateStateChanged, state);
  };
  appUpdates.on('changed', forwardAppUpdate);
  permissionPrompts.on('prompt', forwardPermissionPrompt);
  const forwardPermissionCancellation = (promptId: string) => {
    const owner = promptOwners.get(promptId);
    if (owner && !owner.window.isDestroyed())
      owner.trustedWebContents.send(IPC.localAIPermissionPromptCancelled, promptId);
    promptOwners.delete(promptId);
  };
  permissionPrompts.on('cancelled', forwardPermissionCancellation);
  handle(IPC.localAIPermissionRespond, (event, response: unknown) => {
    ensureTrustedUI(event);
    if (!isPermissionResponse(response)) throw new Error('INVALID_PERMISSION_RESPONSE');
    if (promptOwners.get(response.promptId) !== selected())
      throw new Error('PERMISSION_WINDOW_MISMATCH');
    promptOwners.delete(response.promptId);
    return permissionPrompts.respondFromTrustedUI(response);
  });
  handle(IPC.localAIPermissionChooserOpen, (event, promptId: unknown) => {
    ensureTrustedUI(event);
    if (typeof promptId !== 'string' || !/^permission_[A-Za-z0-9_-]{1,120}$/u.test(promptId)) {
      throw new Error('INVALID_PERMISSION_PROMPT');
    }
    if (promptOwners.get(promptId) !== selected()) throw new Error('PERMISSION_WINDOW_MISMATCH');
    return permissionPrompts.requestChoiceFromTrustedUI(promptId);
  });

  return () => {
    imports?.dispose();
    app.removeListener('login', login);
    app.removeListener('select-client-certificate', selectCertificate);
    for (const pending of authentications.values()) pending.finish();
    for (const search of alternativeSearches.values()) search.abort();
    alternativeSearches.clear();
    for (const pending of pendingModelRequests.values()) pending.abort.abort();
    pendingModelRequests.clear();
    workspaceSessions.clear();
    callerContexts.removeListener('document-invalidated', clearWorkspaceSession);
    permissionPrompts.removeListener('prompt', forwardPermissionPrompt);
    permissionPrompts.removeListener('cancelled', forwardPermissionCancellation);
    appUpdates.removeListener('changed', forwardAppUpdate);
    jobs.removeListener('website-event', forwardJobEvent);
    jobs.removeListener('activity-changed', forwardJobActivity);
    for (const channel of Object.values(IPC)) {
      if (channel !== IPC.browserStateChanged) {
        ipcMain.removeHandler(channel);
      }
    }
  };
}

function requireUpdateVersion(value: unknown): asserts value is string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 128) {
    throw new Error('INVALID_UPDATE_VERSION');
  }
}

function requireModelOperationId(value: unknown): asserts value is string {
  if (typeof value !== 'string' || !/^model_request_[a-zA-Z0-9_-]{1,100}$/u.test(value)) {
    throw new Error('MODEL_REQUEST_ID_INVALID');
  }
}

function publicModel(model: { provider: 'huggingface'; repository: string; revision: string }) {
  return {
    ...model,
    sourceUrl: `https://huggingface.co/${model.repository}/tree/${model.revision}`,
  };
}

function publicProgress(progress: BrowserModelRequestProgress): unknown {
  if ('model' in progress) return { ...progress, model: publicModel(progress.model) };
  return progress;
}

function publicSession(session: BrowserModelSession) {
  return {
    sessionId: session.sessionId,
    role: session.role,
    origin: session.origin,
    resolution: {
      preferredModel: publicModel(session.resolution.preferredModel),
      resolvedModel: publicModel(session.resolution.resolvedModel),
      compatibility: session.resolution.compatibility,
      executability: session.resolution.executability,
      security: session.resolution.security,
      trust: session.resolution.trust,
      developerCertified: session.resolution.developerCertified,
      substituted: session.resolution.substituted,
      overriddenByUser: session.resolution.overriddenByUser,
      resolutionReason: session.resolution.resolutionReason,
      ...(session.resolution.selection
        ? {
            selection: {
              ...session.resolution.selection,
              ruleIds: [],
              reasons: [
                session.resolution.selection.kind === 'same-checkpoint'
                  ? 'Verified source lineage; conversion can affect quality.'
                  : session.resolution.selection.kind === 'different-model'
                    ? 'A different task-compatible model was selected.'
                    : 'The requested artifact was selected.',
                session.resolution.selection.source === 'local-measurements'
                  ? 'Comparable local performance evidence informed this choice.'
                  : 'Active local recommendation configuration informed this choice.',
              ],
            },
          }
        : {}),
      compatibilityReasons: session.resolution.compatibilityReasons.map((item) => ({
        code: item.requirement.replaceAll(/[^a-z0-9]+/giu, '-').toLowerCase(),
        status:
          item.outcome === 'not-satisfied'
            ? ('failed' as const)
            : item.outcome === 'satisfied'
              ? ('satisfied' as const)
              : ('unknown' as const),
        message: item.detail,
      })),
    },
    capabilities: session.capabilities,
    ...(session.taskProfile ? { taskProfile: structuredClone(session.taskProfile) } : {}),
  };
}

function websiteError(error: unknown): { code: string; message: string } {
  // Keep diagnostic details in the test process; websites still receive sanitized errors.
  if (process.env.LOCAL_AI_SMOKE_TEST === '1') {
    console.error('ORYN_SMOKE_INTERNAL_ERROR', error);
  }
  if (error instanceof ArtifactError) {
    return { code: error.code, message: safeArtifactErrorMessage(error.code) };
  }
  if (error instanceof BrowserJobError) {
    return {
      code: error.code === 'JOB_NOT_OWNED' ? 'JOB_ACCESS_DENIED' : error.code,
      message:
        error.code === 'JOB_INPUT_INVALID'
          ? 'The job input is invalid.'
          : error.code === 'JOB_QUEUE_FULL'
            ? 'This site has submitted too much Local AI work. Try again shortly.'
            : 'The job is not available to this document.',
    };
  }
  if (error instanceof BrowserModelSessionError) {
    const code = error.code === 'SESSION_NOT_OWNED' ? 'SESSION_ACCESS_DENIED' : error.code;
    return {
      code,
      message:
        code === 'USER_DENIED'
          ? error.savedDecision
            ? 'Local AI access is blocked for this site. Open Local AI → Settings → Sites, choose Ask again, then retry.'
            : 'This Local AI request was denied. Retry when you are ready to allow it.'
          : code === 'DOWNLOAD_REQUIRED'
            ? 'The selected model must be downloaded.'
            : 'The Model Session is not available to this document.',
    };
  }
  if (error instanceof ModelResolutionError) {
    return { code: error.code, message: safeModelErrorMessage(error.code) };
  }
  if (error instanceof ModelRequestValidationError) {
    return { code: 'MODEL_REFERENCE_INVALID', message: 'The Model Request is invalid.' };
  }
  if (error instanceof EngineRequestError) {
    return {
      code: error.protocolError.code,
      message: safeModelErrorMessage(error.protocolError.code),
    };
  }
  if (error instanceof Error && error.message === 'REQUEST_ABORTED') {
    return { code: 'REQUEST_ABORTED', message: 'The model request was aborted.' };
  }
  if (error instanceof Error && error.message === 'LOCAL_AI_FRAME_NOT_ALLOWED') {
    return { code: 'UNAUTHORIZED_FRAME', message: 'This frame cannot use Local AI.' };
  }
  if (error instanceof Error && error.message === 'LOCAL_AI_MODEL_REQUEST_BUSY') {
    return {
      code: 'MODEL_REQUEST_BUSY',
      message: 'Another Local AI model request is in progress. Try again when it finishes.',
    };
  }
  if (error instanceof Error && error.message === 'LOCAL_AI_PERMISSION_PROMPT_BUSY') {
    return {
      code: 'PERMISSION_REQUEST_BUSY',
      message: 'Another Local AI permission request from this site needs attention first.',
    };
  }
  if (
    error instanceof Error &&
    ['BACKGROUND_COMPUTE_DISABLED', 'BACKGROUND_COMPUTE_REQUIRES_ATTENTION'].includes(error.message)
  ) {
    return {
      code: error.message,
      message:
        error.message === 'BACKGROUND_COMPUTE_DISABLED'
          ? 'Background Local AI work is disabled.'
          : 'Return to this tab to start Local AI work.',
    };
  }
  if (
    error instanceof Error &&
    ['LOCAL_AI_CALLER_NOT_ACTIVE', 'LOCAL_AI_SECURE_CONTEXT_REQUIRED'].includes(error.message)
  ) {
    return { code: 'UNAUTHORIZED_ORIGIN', message: 'This origin cannot use Local AI.' };
  }
  return {
    code: 'RUNTIME_PROTOCOL_ERROR',
    message: 'The Local AI request could not be completed.',
  };
}

function workspaceError(error: unknown): { code: string; message: string } {
  if (
    error instanceof ArtifactError ||
    error instanceof BrowserJobError ||
    error instanceof BrowserModelSessionError ||
    error instanceof ModelResolutionError ||
    error instanceof ModelRequestValidationError ||
    error instanceof EngineRequestError
  ) {
    return websiteError(error);
  }
  if (error instanceof Error && error.message === 'MODEL_WORKSPACE_INPUT_INVALID') {
    return { code: error.message, message: 'The workspace input is invalid for this model task.' };
  }
  if (error instanceof Error && error.message.startsWith('MODEL_WORKSPACE_')) {
    return {
      code: error.message,
      message: 'This model workspace is no longer available. Reload it to revalidate the model.',
    };
  }
  return {
    code: 'MODEL_WORKSPACE_ERROR',
    message: 'The model workspace request could not be completed.',
  };
}

function trustedResourceErrorDetails(error: unknown): {
  requiredMemoryBytes?: number;
  availableMemoryBytes?: number;
  memoryBudgetBytes?: number;
  memoryPressure?: 'normal' | 'high' | 'critical';
} {
  if (!(error instanceof EngineRequestError) || error.protocolError.category !== 'resource') {
    return {};
  }
  const details = error.protocolError.details;
  if (typeof details !== 'object' || details === null || Array.isArray(details)) return {};
  const value = details as Record<string, unknown>;
  const nonNegativeInteger = (item: unknown): number | undefined =>
    typeof item === 'number' && Number.isSafeInteger(item) && item >= 0 ? item : undefined;
  const pressure = value.memoryPressure;
  return {
    ...(nonNegativeInteger(value.requiredBytes) !== undefined
      ? { requiredMemoryBytes: nonNegativeInteger(value.requiredBytes) }
      : {}),
    ...(nonNegativeInteger(value.availableForModelBytes) !== undefined
      ? { availableMemoryBytes: nonNegativeInteger(value.availableForModelBytes) }
      : {}),
    ...(nonNegativeInteger(value.memoryBudgetBytes) !== undefined
      ? { memoryBudgetBytes: nonNegativeInteger(value.memoryBudgetBytes) }
      : {}),
    ...(pressure === 'normal' || pressure === 'high' || pressure === 'critical'
      ? { memoryPressure: pressure }
      : {}),
  };
}

function safeArtifactErrorMessage(code: string): string {
  const messages: Record<string, string> = {
    ARTIFACT_NOT_FOUND: 'The artifact is not available.',
    ARTIFACT_ACCESS_DENIED: 'The artifact is not available to this document.',
    ARTIFACT_EXPIRED: 'The artifact has expired.',
    ARTIFACT_INVALID: 'The artifact request is invalid.',
    ARTIFACT_IO_ERROR: 'The artifact could not be stored.',
    ARTIFACT_PATH_INVALID: 'The artifact is invalid.',
    RESOURCE_LIMIT_EXCEEDED: 'The Local AI artifact storage limit was exceeded.',
    DISK_SPACE_INSUFFICIENT: 'There is not enough disk space for this artifact.',
  };
  return messages[code] ?? 'The artifact request could not be completed.';
}

function validateSaveOptions(value: unknown, fallback?: string): string {
  if (
    value !== undefined &&
    (typeof value !== 'object' || value === null || Array.isArray(value))
  ) {
    throw new ArtifactError('ARTIFACT_INVALID');
  }
  const suggested =
    value && typeof value === 'object' && 'suggestedFilename' in value
      ? (value as { suggestedFilename?: unknown }).suggestedFilename
      : fallback;
  if (suggested !== undefined && (typeof suggested !== 'string' || suggested.length > 255)) {
    throw new ArtifactError('ARTIFACT_INVALID');
  }
  const sanitized = (suggested || fallback || 'artifact.bin')
    .replaceAll(/[\\/]/gu, '_')
    .split('')
    .map((character) => {
      const code = character.charCodeAt(0);
      return code < 32 || code === 127 ? '_' : character;
    })
    .join('')
    .replaceAll(/\.\.+/gu, '_')
    .trim();
  return sanitized.slice(0, 255) || 'artifact.bin';
}

function safeModelErrorMessage(code: string): string {
  const messages: Record<string, string> = {
    MODEL_REFERENCE_INVALID: 'The model reference is invalid.',
    MODEL_NOT_FOUND: 'The requested model was not found.',
    MODEL_METADATA_INVALID: 'The model metadata is invalid.',
    MODEL_NOT_EXECUTABLE: 'The selected model cannot run on this device.',
    MODEL_SECURITY_BLOCKED: 'The selected model is blocked by Local AI security policy.',
    CUSTOM_CODE_REQUIRED: 'The selected model requires prohibited custom code.',
    UNSAFE_MODEL_FORMAT: 'The selected model uses an unsafe file format.',
    MODEL_MEMORY_INSUFFICIENT: 'There is not enough safely available memory to load this model.',
    MODEL_LOAD_FAILED:
      'The installed model could not be loaded into its local runtime. Try again; if it continues, restart Oryn and retry.',
    MODEL_RESIDENCY_BUSY: 'This model is already changing state. Wait a moment and try again.',
    RUNTIME_DEPENDENCY_UNAVAILABLE:
      'The required local runtime is unavailable. Restart or update Oryn, then try again.',
    CHAT_INPUT_ENCODING_FAILED:
      'This model could not encode the conversation with its installed tokenizer.',
    CHAT_GENERATION_FAILED:
      'The model loaded, but local text generation failed. Try again or use a shorter message.',
    MODEL_ACCESS_RESTRICTED:
      'The selected model requires approval or authentication with its model host.',
    NO_EXECUTABLE_MODEL: 'No executable model satisfies this request.',
    DISK_SPACE_INSUFFICIENT: 'There is not enough disk space for the selected model.',
  };
  return messages[code] ?? 'The Local AI request could not be completed.';
}

function isPermissionResponse(value: unknown): value is LocalAIPermissionPromptResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    'promptId' in value &&
    typeof value.promptId === 'string' &&
    'action' in value &&
    ['allow-once', 'always-allow', 'deny'].includes(String(value.action))
  );
}

function isBrowserChromeLayout(value: unknown): value is BrowserChromeLayout {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const candidate = value as Partial<BrowserChromeLayout>;
  const keys = Object.keys(value);
  return (
    keys.length === 4 &&
    keys.every((key) =>
      ['contentHeight', 'overlayHeight', 'coverWindow', 'sidebarInteractive'].includes(key),
    ) &&
    typeof candidate.contentHeight === 'number' &&
    Number.isFinite(candidate.contentHeight) &&
    candidate.contentHeight >= 64 &&
    candidate.contentHeight <= 4096 &&
    typeof candidate.overlayHeight === 'number' &&
    Number.isFinite(candidate.overlayHeight) &&
    candidate.overlayHeight >= candidate.contentHeight &&
    candidate.overlayHeight <= 4096 &&
    typeof candidate.coverWindow === 'boolean' &&
    typeof candidate.sidebarInteractive === 'boolean'
  );
}

function isBrowserPassiveOverlayLayout(value: unknown): value is BrowserPassiveOverlayLayout {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const candidate = value as Partial<BrowserPassiveOverlayLayout>;
  const keys = Object.keys(value);
  return (
    keys.length === 3 &&
    keys.every((key) => ['visible', 'width', 'height'].includes(key)) &&
    typeof candidate.visible === 'boolean' &&
    typeof candidate.width === 'number' &&
    Number.isFinite(candidate.width) &&
    candidate.width >= 280 &&
    candidate.width <= 640 &&
    typeof candidate.height === 'number' &&
    Number.isFinite(candidate.height) &&
    candidate.height >= 0 &&
    candidate.height <= 4096
  );
}

function requireIdentifier(value: unknown, prefix: string): asserts value is string {
  if (typeof value !== 'string' || !value.startsWith(`${prefix}_`) || value.length > 128) {
    throw new Error('INVALID_IDENTIFIER');
  }
}

function requireInstallationId(value: unknown): asserts value is string {
  if (typeof value !== 'string' || !/^install_[A-Za-z0-9_-]{1,120}$/u.test(value)) {
    throw new Error('INVALID_INSTALLATION_ID');
  }
}

function unavailableCapabilities(state: EngineSupervisor['state']): RuntimeCapabilities {
  return {
    runtimeAvailable: false,
    runtimeState: state === 'starting' ? 'starting' : state === 'error' ? 'error' : 'unavailable',
    sdkVersion: '0.1.0',
    browserApiVersion: '1',
    supportedModelProviders: [],
    features: {
      modelSelection: false,
      modelRequests: false,
      modelSubstitution: false,
      streaming: false,
      artifacts: false,
      artifactUrls: false,
      cancellation: false,
      queueing: false,
      multipleSubmittedJobs: false,
    },
  };
}
