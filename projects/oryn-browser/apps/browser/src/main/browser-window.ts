import { join } from 'node:path';

import {
  app,
  clipboard,
  dialog,
  BrowserWindow,
  Menu,
  screen,
  shell,
  WebContentsView,
  type Input,
  type WebContents,
} from 'electron';

import type { BrowserCallerContextService } from './browser-caller-context';
import type { BrowserDataService } from './data/browser-data-service';
import {
  DEFAULT_WINDOW_SIZE,
  fitWindowBoundsToWorkArea,
  MINIMUM_WINDOW_SIZE,
} from './data/window-placement-store';
import type { EngineSupervisor, EngineSupervisorState } from './engine-supervisor';
import {
  DEFAULT_START_URL,
  TabManager,
  normalizeNavigationInput,
  searchNavigationInput,
} from './tab-manager';
import { migrateLegacyDevelopmentStartURL } from '../shared/browser-urls';
import { createOrynModelWorkspaceURL } from '../shared/browser-urls';
import {
  IPC,
  TAB_SEARCH_SHORTCUT,
  clampSidebarPanelWidth,
  type BrowserChromeLayout,
  type BrowserExperienceCommand,
  type BrowserPassiveOverlayLayout,
  type BrowserWindowState,
  type TrustedModelOperation,
} from '../shared/ipc';
import { HuggingFacePageActionController } from './model-page/hugging-face-page-action-controller';
import { SidebarWebAppController } from './sidebar-web-app-controller';
import { ConnectionSecurityObserver } from './connection-security';
import { TRUSTED_BROWSER_UI_URL, registerTrustedBrowserUI } from './trusted-browser-ui';
import type { CaptureAccessSnapshot } from './data/capture-access-registry';
import { ExternalApplicationHandoff } from './external-application-handoff';
import { browserContextMenu } from './browser-context-menu';
import { BROWSER_PAGE_SAVE_FORMAT, saveBrowserPage } from './save-browser-page';
import {
  isOrynVerified,
  isUserTrustEligible,
  taskContractForProfile,
} from './model-workspace/model-workspace-eligibility';

const DEFAULT_CHROME_HEIGHT = 112;
const WINDOW_PLACEMENT_SAVE_DELAY_MS = 250;

function withSurface(base: string, surface: string): string {
  const url = new URL(base);
  url.search = '';
  url.hash = '';
  url.searchParams.set('surface', surface);
  return url.toString();
}

function normalizeTrustedUIURL(value: string): string {
  const url = new URL(value);
  url.hash = '';
  return url.toString();
}

export interface BrowserWindowControllerOptions {
  captureAccess?: () => CaptureAccessSnapshot[];
  loadTrustedUI?: boolean;
  openWindow?: (url?: string) => void;
  canClose?: () => Promise<boolean>;
  privateMode?: boolean;
  session?: Electron.Session;
  openPrivateWindow?: (url?: string) => void;
}

export class BrowserWindowController {
  get isPrivate(): boolean {
    return this.options.privateMode === true;
  }
  private readonly connectionObservers = new Map<number, ConnectionSecurityObserver>();
  readonly window: BrowserWindow;
  readonly trustedView: WebContentsView;
  readonly passiveOverlayView: WebContentsView;
  readonly sidebarView: WebContentsView;
  readonly sidebarResizeView: WebContentsView;
  readonly tabs: TabManager;
  readonly sidebarApps: SidebarWebAppController;
  private engineState: EngineSupervisorState;
  private closePrepared = false;
  private chromeContentHeight = DEFAULT_CHROME_HEIGHT;
  private chromeCoversWindow = false;
  private chromeAllowsSidebarInteraction = false;
  private focusFindWhenVisible = false;
  private sidebarPanelWidthPreview: number | undefined;
  private passiveOverlayLayout: BrowserPassiveOverlayLayout = {
    visible: false,
    width: 456,
    height: 0,
  };
  private windowPlacementSaveTimer: ReturnType<typeof setTimeout> | undefined;
  private browserMemorySampleTimer: ReturnType<typeof setInterval> | undefined;
  private browserPrivateBytes: number | undefined;
  readonly huggingFacePageAction: HuggingFacePageActionController;
  private readonly trustedUIURLs = new Map<number, string>();
  private readonly externalApplications: ExternalApplicationHandoff;
  private pageSavePending = false;

  constructor(
    private readonly engine: EngineSupervisor,
    private readonly callerContexts: BrowserCallerContextService,
    private readonly browserData: BrowserDataService,
    private readonly options: BrowserWindowControllerOptions = {},
  ) {
    this.engineState = engine.state;
    const appearance = browserData.appearance.read();
    this.chromeContentHeight = browserData.bookmarks.isBarVisible() ? 112 : 80;
    this.chromeCoversWindow = !appearance.onboardingComplete;
    const savedPlacement = browserData.windowPlacement.read();
    const initialBounds = savedPlacement.bounds
      ? fitWindowBoundsToWorkArea(
          savedPlacement.bounds,
          screen.getDisplayMatching(savedPlacement.bounds).workArea,
        )
      : DEFAULT_WINDOW_SIZE;
    this.window = new BrowserWindow({
      ...initialBounds,
      minWidth: MINIMUM_WINDOW_SIZE.width,
      minHeight: MINIMUM_WINDOW_SIZE.height,
      title: this.isPrivate ? `${app.getName()} — Private` : app.getName(),
      show: false,
      backgroundColor: '#101315',
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
      },
    });
    this.externalApplications = new ExternalApplicationHandoff(
      {
        resolveHandler: (url) => app.getApplicationInfoForProtocol(url).catch(() => null),
        confirm: async ({ origin, application, signal }) => {
          if (this.window.isDestroyed()) return false;
          const result = await dialog.showMessageBox(this.window, {
            type: 'question',
            title: 'Open application',
            message: `Open ${application}?`,
            detail: `${origin} requested this link.`,
            buttons: ['Cancel', `Open ${application}`],
            defaultId: 0,
            cancelId: 0,
            signal,
          });
          return result.response === 1;
        },
        notify: async (reason) => {
          if (this.window.isDestroyed()) return;
          await dialog.showMessageBox(this.window, {
            type: 'info',
            title: 'Open application',
            message:
              reason === 'missing-handler'
                ? 'No installed application can open this link.'
                : 'The application could not be opened.',
            buttons: ['OK'],
          });
        },
        openExternal: (url) => shell.openExternal(url),
      },
      !this.isPrivate,
    );
    if (savedPlacement.maximized) {
      this.window.maximize();
    }

    this.trustedView = new WebContentsView({
      webPreferences: {
        preload: join(__dirname, '../preload/trusted.cjs'),
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        transparent: true,
      },
    });
    this.trustedView.setBackgroundColor('#00000000');
    if (this.options.loadTrustedUI === false) this.trustedView.setVisible(false);
    this.window.contentView.addChildView(this.trustedView);

    this.passiveOverlayView = new WebContentsView({
      webPreferences: {
        preload: join(__dirname, '../preload/trusted.cjs'),
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        transparent: true,
      },
    });
    this.passiveOverlayView.setBackgroundColor('#00000000');
    this.passiveOverlayView.setVisible(false);
    this.window.contentView.addChildView(this.passiveOverlayView);

    this.sidebarView = new WebContentsView({
      webPreferences: {
        preload: join(__dirname, '../preload/trusted.cjs'),
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        transparent: false,
      },
    });
    this.sidebarView.setBackgroundColor('#171a20');
    if (this.options.loadTrustedUI === false) this.sidebarView.setVisible(false);
    this.window.contentView.addChildView(this.sidebarView);

    this.sidebarResizeView = new WebContentsView({
      webPreferences: {
        preload: join(__dirname, '../preload/trusted.cjs'),
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        transparent: true,
      },
    });
    this.sidebarResizeView.setBackgroundColor('#00000000');
    this.sidebarResizeView.setVisible(false);
    this.window.contentView.addChildView(this.sidebarResizeView);

    const trustedUIBaseURL =
      !app.isPackaged && process.env.ELECTRON_RENDERER_URL
        ? process.env.ELECTRON_RENDERER_URL
        : TRUSTED_BROWSER_UI_URL;
    const trustedTargets = [
      [this.trustedWebContents, trustedUIBaseURL],
      [this.passiveOverlayView.webContents, withSurface(trustedUIBaseURL, 'passive-overlay')],
      [this.sidebarView.webContents, withSurface(trustedUIBaseURL, 'sidebar')],
      [this.sidebarResizeView.webContents, withSurface(trustedUIBaseURL, 'sidebar-resize')],
    ] as const;
    for (const [contents, target] of trustedTargets) {
      registerTrustedBrowserUI(contents);
      this.installTrustedUINavigationPolicy(contents, target);
    }

    this.tabs = new TabManager(this.window, callerContexts, browserData, {
      websitePreloadPath: join(__dirname, '../preload/website.cjs'),
      workspacePreloadPath: join(__dirname, '../preload/workspace.cjs'),
      privatePreloadPath: join(__dirname, '../preload/private.cjs'),
      privateMode: this.isPrivate,
      externalApplications: this.externalApplications,
      session: this.options.session,
      beforeFirstNavigation: (contents) => this.observeConnection(contents),
      onWebContentsCreated: (contents) => {
        this.installKeyboardHandler(contents);
        this.installContextMenu(contents);
        contents.on('found-in-page', (_event, result) => {
          if (contents === this.activeWebsiteWebContents)
            this.sendToPassiveOverlay(IPC.browserFindResult, {
              matches: result.matches,
              activeMatchOrdinal: result.activeMatchOrdinal,
            });
        });
      },
      onActiveViewChanged: () => this.raiseTrustedView(),
    });
    this.huggingFacePageAction = new HuggingFacePageActionController(engine, () => {
      const tab = this.tabs.getActiveTab();
      return {
        tabId: tab?.tabId ?? '',
        documentId: tab?.activeDocumentId ?? '',
        url: this.isPrivate ? '' : (tab?.url ?? ''),
      };
    });
    this.sidebarApps = new SidebarWebAppController(
      this.window,
      browserData,
      (url) => {
        this.createTab(url, true);
      },
      (contents) => this.observeConnection(contents),
      this.externalApplications,
    );
    this.sidebarApps.on('changed', () => {
      this.layoutViews();
      this.publishState();
    });
    this.tabs.on('state-changed', () => {
      this.huggingFacePageAction.activeTabChanged();
      this.publishState();
    });
    this.huggingFacePageAction.on('changed', () => this.publishState());
    this.huggingFacePageAction.on('operation-changed', (operation: TrustedModelOperation) => {
      this.publishModelOperation(operation);
    });
    this.installKeyboardHandler(this.trustedWebContents);
    this.installKeyboardHandler(this.passiveOverlayView.webContents);
    this.installKeyboardHandler(this.sidebarView.webContents);
    this.installKeyboardHandler(this.sidebarResizeView.webContents);
    this.trustedWebContents.on('preload-error', (_event, preloadPath, error) => {
      console.error(`Trusted UI preload failed (${preloadPath}):`, error);
    });
    this.installEventHandlers();
    void this.sampleBrowserMemory();
    this.browserMemorySampleTimer = setInterval(() => void this.sampleBrowserMemory(), 5_000);
    this.browserMemorySampleTimer.unref();
    this.layoutViews();

    if (this.options.loadTrustedUI !== false) {
      for (const [contents, target] of trustedTargets) void contents.loadURL(target);
    }

    this.tabs.batch(() => this.restoreTabs());
    const activeSidebarApp = browserData.experience.read().sidebar.activeAppId;
    if (activeSidebarApp && !this.isPrivate) this.sidebarApps.open(activeSidebarApp);
    this.huggingFacePageAction.activeTabChanged();
  }

  get activeWebsiteWebContents(): WebContents {
    return this.tabs.getActiveWebContents();
  }

  connectionSnapshot() {
    return this.connectionObservers.get(this.activeWebsiteWebContents.id)?.snapshot();
  }

  refreshCaptureState(): void {
    if (this.window.isDestroyed()) return;
    this.publishState();
    if (!this.trustedWebContents.isDestroyed())
      this.trustedWebContents.send(IPC.browserDataChanged);
  }

  recordCertificateError(
    contents: WebContents,
    url: string,
    certificate: Electron.Certificate,
  ): void {
    this.connectionObservers.get(contents.id)?.recordCertificateError(url, certificate);
  }

  private observeConnection(contents: WebContents): void | Promise<void> {
    if (!contents.debugger) return;
    const existing = this.connectionObservers.get(contents.id);
    if (existing) return existing.ready;
    const observer = new ConnectionSecurityObserver(contents, () => {
      this.publishState();
      if (!this.trustedWebContents.isDestroyed())
        this.trustedWebContents.send(IPC.browserDataChanged);
    });
    this.connectionObservers.set(contents.id, observer);
    contents.once('destroyed', () => {
      observer.dispose();
      this.connectionObservers.delete(contents.id);
    });
    return observer.ready;
  }

  get trustedWebContents(): WebContents {
    return this.trustedView.webContents;
  }

  isTrustedUIWebContents(contents: WebContents): boolean {
    const expectedURL = this.trustedUIURLs.get(contents.id);
    if (!expectedURL) return false;
    if (this.options.loadTrustedUI === false) return true;
    return normalizeTrustedUIURL(contents.getURL()) === expectedURL;
  }

  sendToPassiveOverlay(channel: string, ...args: unknown[]): void {
    if (!this.passiveOverlayView.webContents.isDestroyed()) {
      this.passiveOverlayView.webContents.send(channel, ...args);
    }
  }

  publishModelOperation(operation: TrustedModelOperation): void {
    if (!this.trustedWebContents.isDestroyed()) {
      this.trustedWebContents.send(IPC.localAIModelOperationChanged, operation);
    }
    this.sendToPassiveOverlay(IPC.localAIModelOperationChanged, operation);
  }

  get activeTabId(): string {
    return this.tabs.getActiveTabId();
  }

  showWhenReady(): void {
    const show = () => {
      this.trustedWebContents.removeListener('did-finish-load', show);
      this.trustedWebContents.removeListener('did-fail-load', show);
      if (!this.window.isDestroyed()) this.window.show();
    };
    if (!this.trustedWebContents.isLoading() && this.trustedWebContents.getURL()) {
      show();
      return;
    }
    this.trustedWebContents.once('did-finish-load', show);
    this.trustedWebContents.once('did-fail-load', show);
  }

  getState(): BrowserWindowState {
    const tabs = this.tabs.getTabs();
    const activeTab = this.tabs.getActiveTab();
    const engineResources = this.engine.lastResources;
    const runtimeWorkerResidentBytes = engineResources?.workers?.reduce(
      (sum, worker) => sum + (worker.residentBytes ?? 0),
      0,
    );
    return {
      privateMode: this.isPrivate,
      activeTabId: activeTab?.tabId ?? '',
      captureAccess: this.options.captureAccess?.() ?? this.browserData.captureSnapshot(),
      tabs: tabs.map((tab) => ({
        tabId: tab.tabId,
        webContentsId: tab.webContentsId,
        url: tab.url,
        title: tab.title,
        favicon: tab.favicon,
        crashed: tab.crashed,
        audible: tab.audible,
        muted: tab.muted,
        isLoading: tab.isLoading,
        canGoBack: tab.canGoBack,
        canGoForward: tab.canGoForward,
        workspaceId: tab.workspaceId,
        islandId: tab.islandId,
        pinned: tab.pinned,
        captureAccess: [
          ...new Set(
            this.browserData
              .captureSnapshot()
              .filter((entry) => entry.webContentsId === tab.webContentsId)
              .flatMap((entry) => entry.permissions),
          ),
        ],
        connection: this.connectionObservers.get(tab.webContentsId)?.snapshot(),
      })),
      engineState: this.engineState,
      zoomFactor: activeTab ? activeTab.view.webContents.getZoomFactor() : 1,
      memoryPressure: engineResources?.memoryPressure ?? 'normal',
      localAIResources: engineResources
        ? {
            processResidentBytes:
              engineResources.supervisorMemory?.residentBytes ??
              engineResources.processMemory?.residentBytes,
            browserPrivateBytes: this.browserPrivateBytes,
            workerResidentBytes:
              engineResources.workers === undefined ? undefined : runtimeWorkerResidentBytes,
            systemTotalBytes: engineResources.systemMemory?.totalBytes,
            systemAvailableBytes: engineResources.systemMemory?.availableBytes,
            memoryBudgetBytes: engineResources.localAI.memoryBudgetBytes,
            estimatedResidentModelBytes: engineResources.localAI.estimatedResidentModelBytes,
            acceleratorBackend: engineResources.accelerator?.backend,
            acceleratorAllocatedBytes: engineResources.accelerator?.allocatedBytes,
            acceleratorReservedBytes: engineResources.accelerator?.reservedBytes,
            acceleratorReliable: engineResources.accelerator?.allocatedBytes !== undefined,
            cpuPercent: process.getCPUUsage().percentCPUUsage,
            engineCPUPercent: engineResources.processMemory?.cpuPercent,
            sampleWindowMs: engineResources.processMemory?.sampleWindowMs,
            activeExecution: engineResources.runningJob,
            throughput: engineResources.throughput,
            generationRunning: engineResources.generationLane.running,
            generationQueued: engineResources.generationLane.queued,
          }
        : undefined,
      huggingFacePageAction: this.isPrivate ? undefined : this.huggingFacePageAction.snapshot(),
      experience: this.experienceState(),
    };
  }

  async updateBrowserExperience(command: BrowserExperienceCommand): Promise<void> {
    if (
      this.isPrivate &&
      (command.type.startsWith('workspace-') ||
        command.type.startsWith('sidebar-') ||
        command.type === 'tab-move-workspace' ||
        command.type === 'island-move-workspace')
    )
      throw new Error('PRIVATE_FEATURE_UNAVAILABLE');
    const experience = this.browserData.experience;
    let shouldPublish = true;
    switch (command.type) {
      case 'workspace-create': {
        const created = experience.createWorkspace(command.name);
        this.tabs.switchWorkspace(created.workspaceId);
        break;
      }
      case 'workspace-rename':
        experience.renameWorkspace(command.workspaceId, command.name);
        break;
      case 'workspace-delete':
        await this.tabs.deleteWorkspace(command.workspaceId, command.tabs);
        break;
      case 'workspace-switch':
        this.tabs.switchWorkspace(command.workspaceId);
        break;
      case 'tab-move-workspace':
        this.tabs.moveTabToWorkspace(command.tabId, command.workspaceId);
        break;
      case 'tab-mute':
        this.tabs.mute(command.tabId, command.muted);
        break;
      case 'tab-reorder':
        this.tabs.reorder(command.tabId, command.beforeTabId);
        break;
      case 'tab-pin':
        this.tabs.setPinned(command.tabId, command.pinned);
        break;
      case 'tab-duplicate':
        this.tabs.duplicateTab(command.tabId);
        break;
      case 'tab-close-others':
        await this.tabs.closeOtherTabs(command.tabId);
        break;
      case 'tab-close-right':
        await this.tabs.closeTabsToRight(command.tabId);
        break;
      case 'tab-close-duplicates':
        await this.tabs.closeDuplicateTabs();
        break;
      case 'tab-auto-group':
        experience.updateTabPreferences({ autoGroupOpenerTabs: command.enabled });
        break;
      case 'tab-search-shortcut':
        experience.updateTabPreferences({ tabSearchShortcut: command.shortcut });
        break;
      case 'island-create':
        this.tabs.createIsland(command.tabIds, command.name);
        break;
      case 'island-update':
        experience.updateIsland(command.islandId, {
          ...(command.name === undefined ? {} : { name: command.name }),
          ...(command.accent === undefined ? {} : { accent: command.accent }),
          ...(command.collapsed === undefined ? {} : { collapsed: command.collapsed }),
        });
        break;
      case 'island-delete':
        this.tabs.deleteIsland(command.islandId);
        break;
      case 'island-move-workspace':
        this.tabs.moveIslandToWorkspace(command.islandId, command.workspaceId);
        break;
      case 'tab-move-island':
        this.tabs.moveTabToIsland(command.tabId, command.islandId);
        break;
      case 'sidebar-toggle':
        experience.updateSidebar({ visible: command.visible });
        this.layoutViews();
        break;
      case 'sidebar-tool-open':
        if (experience.read().sidebar.activeAppId) this.sidebarApps.open();
        if (command.toolId === 'tab-search') {
          this.trustedWebContents.send(IPC.browserOpenTabSearch);
        } else {
          this.trustedWebContents.send(
            IPC.browserOpenPanel,
            command.toolId === 'resources' ? 'local-ai' : command.toolId,
          );
        }
        break;
      case 'sidebar-tool-enable':
        experience.updateSidebarTool(command.toolId, { enabled: command.enabled });
        break;
      case 'sidebar-tool-order':
        {
          const tools = experience
            .read()
            .sidebar.tools.slice()
            .sort((a, b) => a.order - b.order);
          const currentIndex = tools.findIndex((tool) => tool.toolId === command.toolId);
          const nextIndex = Math.min(tools.length - 1, Math.max(0, command.order));
          const displaced = tools[nextIndex];
          if (currentIndex >= 0 && displaced) {
            experience.updateSidebarTool(displaced.toolId, { order: currentIndex });
            experience.updateSidebarTool(command.toolId, { order: nextIndex });
          }
        }
        break;
      case 'sidebar-app-open':
        this.sidebarPanelWidthPreview = undefined;
        this.trustedWebContents.send(IPC.browserOpenPanel, null);
        this.sidebarApps.open(command.appId);
        break;
      case 'sidebar-mode':
        experience.updateSidebar({ panelMode: command.mode });
        this.layoutViews();
        break;
      case 'sidebar-resize':
        if (command.persist === false) {
          this.sidebarPanelWidthPreview = clampSidebarPanelWidth(command.width);
          shouldPublish = false;
        } else {
          this.sidebarPanelWidthPreview = undefined;
          experience.updateSidebar({ panelWidth: command.width });
        }
        this.layoutViews();
        break;
      case 'sidebar-app-mute':
        this.sidebarApps.setMuted(command.appId, command.muted);
        break;
      case 'sidebar-app-enable':
        experience.updateSidebarApp(command.appId, { enabled: command.enabled });
        if (!command.enabled && experience.read().sidebar.activeAppId === command.appId) {
          this.sidebarApps.open();
        }
        break;
      case 'sidebar-app-clear-data':
        void this.sidebarApps.clearData(command.appId).catch(() => undefined);
        break;
      case 'sidebar-app-order':
        {
          const apps = experience
            .read()
            .sidebar.apps.slice()
            .sort((a, b) => a.order - b.order);
          const currentIndex = apps.findIndex((app) => app.appId === command.appId);
          const nextIndex = Math.min(apps.length - 1, Math.max(0, command.order));
          const displaced = apps[nextIndex];
          if (currentIndex >= 0 && displaced) {
            experience.updateSidebarApp(displaced.appId, { order: currentIndex });
            experience.updateSidebarApp(command.appId, { order: nextIndex });
          }
        }
        break;
    }
    if (shouldPublish) this.publishState();
  }

  async openHuggingFaceModelWorkspace(): Promise<string> {
    if (this.isPrivate) throw new Error('PRIVATE_LOCAL_AI_UNAVAILABLE');
    const action = this.huggingFacePageAction.requireInstalled();
    if (!action.exactRevision || !action.taskContract || !action.runner || !action.trustMode) {
      throw new Error('MODEL_WORKSPACE_BINDING_INCOMPLETE');
    }
    const profile = await this.engine.inspectModel({
      provider: 'huggingface',
      repository: action.repository,
      revision: action.exactRevision,
    });
    const taskContract = taskContractForProfile(profile);
    const eligible =
      action.trustMode === 'oryn-verified' ? isOrynVerified(profile) : isUserTrustEligible(profile);
    if (!eligible || taskContract !== action.taskContract) {
      throw new Error('MODEL_WORKSPACE_REVALIDATION_FAILED');
    }
    const binding = this.browserData.modelWorkspaces.create({
      model: profile.model,
      selectedFiles: profile.selectedFiles?.map((file) => file.name).sort() ?? profile.model.files,
      taskContract,
      requirements: {
        task: profile.technicalCapabilities.task,
        contract: profile.taskProfile?.contract ?? { id: taskContract, revision: 1 },
      },
      trustMode: action.trustMode,
      ...(profile.runtimePack
        ? {
            runtimePack: {
              packId: profile.runtimePack.packId,
              packVersion: profile.runtimePack.packVersion,
            },
          }
        : {}),
    });
    return this.createTab(createOrynModelWorkspaceURL(binding.reference), true);
  }

  createTab(url?: string, activate = true): string {
    return this.tabs.createTab(url, activate).tabId;
  }

  activateTab(tabId: string): void {
    this.tabs.activateTab(tabId);
  }

  async closeTab(tabId?: string): Promise<void> {
    await this.tabs.closeTab(tabId);
  }

  reopenRecentlyClosed(): string | undefined {
    return this.tabs.reopenRecentlyClosed()?.tabId;
  }

  navigate(input: string): void {
    this.tabs.navigate(input);
  }

  back(): void {
    this.tabs.back();
  }

  forward(): void {
    this.tabs.forward();
  }

  reload(): void {
    this.tabs.reload();
  }

  stop(): void {
    this.tabs.stop();
  }

  openDevTools(): void {
    this.tabs.openDevTools();
  }

  async prepareClose(): Promise<boolean> {
    if (this.closePrepared) return true;
    if (!(await this.sidebarApps.requestClose())) return false;
    this.closePrepared = await this.tabs.closeAll();
    if (this.closePrepared) this.browserData.history.flush();
    return this.closePrepared;
  }

  allowPreparedClose(): void {
    this.closePrepared = true;
  }

  cancelPreparedClose(): void {
    this.closePrepared = false;
    if (!this.window.isDestroyed()) this.tabs.recoverOrganization();
  }

  async allowsUpdateRestart(): Promise<boolean> {
    return !this.sidebarApps.hasOpenPages() && (await this.tabs.allowsUpdateRestart());
  }

  currentOrigin(): string | undefined {
    try {
      const url = new URL(this.activeWebsiteWebContents.getURL());
      return ['https:', 'http:'].includes(url.protocol) ? url.origin : undefined;
    } catch {
      return undefined;
    }
  }

  findInPage(query: string, forward = true): void {
    if (!query) {
      this.activeWebsiteWebContents.stopFindInPage('clearSelection');
      this.sendToPassiveOverlay(IPC.browserFindResult, { matches: 0, activeMatchOrdinal: 0 });
    } else {
      this.activeWebsiteWebContents.findInPage(query, { forward, findNext: true });
    }
  }

  stopFindInPage(): void {
    this.focusFindWhenVisible = false;
    this.activeWebsiteWebContents.stopFindInPage('clearSelection');
    this.activeWebsiteWebContents.focus();
  }

  showFindInPage(): void {
    this.focusFindWhenVisible = true;
    this.sendToPassiveOverlay(IPC.browserOpenFind);
    this.focusFindOverlayIfReady();
  }

  setZoom(action: 'in' | 'out' | 'reset'): void {
    const current = this.activeWebsiteWebContents.getZoomFactor();
    const next = action === 'reset' ? 1 : action === 'in' ? current + 0.1 : current - 0.1;
    const factor = Math.min(2, Math.max(0.5, next));
    this.activeWebsiteWebContents.setZoomFactor(factor);
    this.browserData.zoom.set(this.activeWebsiteWebContents.getURL(), factor);
    this.publishState();
  }

  async savePageAs(contents = this.activeWebsiteWebContents): Promise<void> {
    if (this.pageSavePending) return;
    this.pageSavePending = true;
    try {
      await saveBrowserPage(contents, {
        owned: () => this.tabs.getTabs().some((tab) => tab.view.webContents === contents),
        choosePath: async (defaultFilename) => {
          if (this.window.isDestroyed()) return undefined;
          const target = await dialog.showSaveDialog(this.window, {
            title: `Save Page As — ${BROWSER_PAGE_SAVE_FORMAT.label}`,
            defaultPath: defaultFilename,
            filters: [
              {
                name: BROWSER_PAGE_SAVE_FORMAT.label,
                extensions: [BROWSER_PAGE_SAVE_FORMAT.extension],
              },
            ],
          });
          return target.canceled ? undefined : target.filePath;
        },
        failed: () => this.pageActionFailed('The page could not be saved.'),
      });
    } finally {
      this.pageSavePending = false;
    }
  }

  printPage(contents = this.activeWebsiteWebContents): Promise<void> {
    return new Promise((resolve, reject) => {
      contents.print({}, (success, reason) => {
        if (success) {
          resolve();
        } else {
          reject(new Error(reason || 'PRINT_FAILED'));
        }
      });
    });
  }

  setChromeLayout(layout: BrowserChromeLayout): void {
    const [, contentHeight = 520] = this.window.getContentSize();
    this.chromeContentHeight = Math.min(
      contentHeight,
      Math.max(80, Math.round(layout.contentHeight)),
    );
    this.chromeCoversWindow = layout.coverWindow;
    this.chromeAllowsSidebarInteraction = layout.sidebarInteractive;
    this.layoutViews();
    this.focusFindOverlayIfReady();
  }

  dismissPanelAndClickPage(x: number, y: number, button: 'left' | 'middle' | 'right'): void {
    if (!this.chromeCoversWindow || !this.chromeAllowsSidebarInteraction) return;
    const tab = this.tabs.getTab(this.tabs.getActiveTabId());
    const bounds = tab.view.getBounds();
    const chromeBounds = this.trustedView.getBounds();
    const pageX = Math.round(x + chromeBounds.x - bounds.x);
    const pageY = Math.round(y - bounds.y);
    if (pageX < 0 || pageY < 0 || pageX >= bounds.width || pageY >= bounds.height) return;
    this.chromeCoversWindow = false;
    this.chromeAllowsSidebarInteraction = false;
    this.trustedWebContents.send(IPC.browserOpenPanel, null);
    this.layoutViews();
    const contents = tab.view.webContents;
    contents.focus();
    contents.sendInputEvent({ type: 'mouseDown', x: pageX, y: pageY, button, clickCount: 1 });
    contents.sendInputEvent({ type: 'mouseUp', x: pageX, y: pageY, button, clickCount: 1 });
  }

  setPassiveOverlayLayout(layout: BrowserPassiveOverlayLayout): void {
    this.passiveOverlayLayout = {
      visible: layout.visible,
      width: Math.round(layout.width),
      height: Math.round(layout.height),
    };
    this.layoutViews();
    this.focusFindOverlayIfReady();
  }

  private focusFindOverlayIfReady(): void {
    if (this.focusFindWhenVisible && this.passiveOverlayView.getVisible()) {
      this.passiveOverlayView.webContents.focus();
      this.focusFindWhenVisible = false;
    }
  }

  isTrustedChromeCoveringWindow(): boolean {
    return this.chromeCoversWindow;
  }

  private installEventHandlers(): void {
    this.window.on('resize', () => {
      this.layoutViews();
      this.scheduleWindowPlacementSave();
    });
    this.window.on('move', () => this.scheduleWindowPlacementSave());
    let acceptedClose = false;
    let pendingClose = false;
    this.window.on('close', (event) => {
      this.saveWindowPlacement();
      if (acceptedClose || this.closePrepared) return;
      event.preventDefault();
      if (pendingClose) return;
      pendingClose = true;
      void (async () => {
        if (this.options.canClose && !(await this.options.canClose())) return false;
        return this.prepareClose();
      })()
        .then((accepted) => {
          pendingClose = false;
          if (accepted) {
            acceptedClose = true;
            this.window.close();
          }
        })
        .catch((error) => {
          pendingClose = false;
          console.error('Window close could not be completed.', error);
        });
    });
    this.window.on('closed', () => {
      for (const observer of this.connectionObservers.values()) observer.dispose();
      this.connectionObservers.clear();
      if (this.windowPlacementSaveTimer) clearTimeout(this.windowPlacementSaveTimer);
      if (this.browserMemorySampleTimer) clearInterval(this.browserMemorySampleTimer);
      this.tabs.destroy();
      this.sidebarApps.destroy();
      this.huggingFacePageAction.destroy();
    });

    const stateChanged = (state: EngineSupervisorState) => {
      this.engineState = state;
      this.publishState();
    };
    const resourcesChanged = () => this.publishState();
    const dataChanged = () => {
      if (!this.trustedWebContents.isDestroyed())
        this.trustedWebContents.send(IPC.browserDataChanged);
    };
    const removeCaptureListener = this.browserData.subscribeCaptureAccess(() => {
      this.publishState();
      dataChanged();
    });
    this.engine.on('state-changed', stateChanged);
    this.engine.on('resources-updated', resourcesChanged);
    this.browserData.downloads.on('changed', dataChanged);
    this.window.once('closed', () => {
      removeCaptureListener();
      this.engine.removeListener('state-changed', stateChanged);
      this.engine.removeListener('resources-updated', resourcesChanged);
      this.browserData.downloads.removeListener('changed', dataChanged);
      for (const view of [
        this.trustedView,
        this.passiveOverlayView,
        this.sidebarView,
        this.sidebarResizeView,
      ]) {
        const contents = view.webContents;
        if (contents && !contents.isDestroyed()) contents.close();
      }
    });
  }

  private installTrustedUINavigationPolicy(contents: WebContents, target: string): void {
    const expectedURL = normalizeTrustedUIURL(target);
    this.trustedUIURLs.set(contents.id, expectedURL);
    contents.setWindowOpenHandler(() => ({ action: 'deny' }));
    contents.on('will-attach-webview', (event) => event.preventDefault());
    contents.on('will-navigate', (event, url) => {
      if (normalizeTrustedUIURL(url) !== expectedURL) event.preventDefault();
    });
  }

  private scheduleWindowPlacementSave(): void {
    if (this.windowPlacementSaveTimer) clearTimeout(this.windowPlacementSaveTimer);
    this.windowPlacementSaveTimer = setTimeout(
      () => this.saveWindowPlacement(),
      WINDOW_PLACEMENT_SAVE_DELAY_MS,
    );
  }

  private async sampleBrowserMemory(): Promise<void> {
    try {
      const memory = await process.getProcessMemoryInfo();
      const next = memory.private * 1024;
      const previous = this.browserPrivateBytes;
      this.browserPrivateBytes = next;
      if (
        previous === undefined ||
        Math.abs(next - previous) >= Math.max(8 * 1024 ** 2, previous * 0.05)
      ) {
        this.publishState();
      }
    } catch {
      this.browserPrivateBytes = undefined;
    }
  }

  private saveWindowPlacement(): void {
    if (this.window.isDestroyed()) return;
    if (this.windowPlacementSaveTimer) {
      clearTimeout(this.windowPlacementSaveTimer);
      this.windowPlacementSaveTimer = undefined;
    }
    this.browserData.windowPlacement.write(
      this.window.getNormalBounds(),
      this.window.isMaximized(),
    );
  }

  private restoreTabs(): void {
    const configuredStartURL = process.env.ORYN_START_URL ?? process.env.LOCAL_AI_START_URL;
    if (configuredStartURL) {
      this.tabs.createTab(configuredStartURL);
      return;
    }

    const restore = this.browserData.sessions.getRestoreState();
    if (restore.openTabs.length === 0) {
      this.tabs.createTab(DEFAULT_START_URL);
      return;
    }

    const tabIds = restore.openTabs.map(
      (tab, index) =>
        this.tabs.createTab(
          migrateLegacyDevelopmentStartURL(tab.url),
          index === Math.min(restore.activeIndex, restore.openTabs.length - 1),
          {
            tabId: tab.tabId,
            favicon: tab.favicon,
            workspaceId: tab.workspaceId,
            islandId: tab.islandId,
            pinned: tab.pinned,
            lazy: index !== Math.min(restore.activeIndex, restore.openTabs.length - 1),
          },
        ).tabId,
    );
    const activeTabId = tabIds[Math.min(restore.activeIndex, tabIds.length - 1)];
    if (activeTabId) {
      this.tabs.activateTab(activeTabId);
    }
  }

  private installKeyboardHandler(contents: WebContents): void {
    contents.on('before-input-event', (event, input) => {
      if (input.type !== 'keyDown') {
        return;
      }
      if (this.handleKeyboardInput(input)) {
        event.preventDefault();
      }
    });
  }

  private installContextMenu(contents: WebContents): void {
    contents.on('context-menu', (_event, params) => {
      const menu = browserContextMenu(contents, params, {
        owned: () => this.tabs.getTabs().some((tab) => tab.view.webContents === contents),
        openTab: (url, activate) => {
          this.tabs.openTabFrom(contents, url, activate);
        },
        openWindow: (url) => this.options.openWindow?.(url),
        openPrivateWindow: (url) => this.options.openPrivateWindow?.(url),
        copyText: (text) => clipboard.writeText(text),
        searchURL: (text) =>
          searchNavigationInput(text, this.browserData.appearance.read().searchProvider),
        savePage: () => this.savePageAs(contents),
        printPage: () => this.printPage(contents),
        failed: () => this.pageActionFailed('The page action could not be completed.'),
        externalApplications: this.externalApplications,
      });
      Menu.buildFromTemplate(menu.template).popup({ window: this.window, callback: menu.dispose });
    });
  }

  private async pageActionFailed(message: string): Promise<void> {
    if (this.window.isDestroyed()) return;
    await dialog.showMessageBox(this.window, {
      type: 'error',
      title: 'Browser action',
      message,
      buttons: ['OK'],
    });
  }

  private handleKeyboardInput(input: Input): boolean {
    const primary = process.platform === 'darwin' ? input.meta : input.control;
    const key = input.key.toLowerCase();
    if (primary && input.shift && key === 'n') {
      this.options.openPrivateWindow?.();
      return true;
    }
    if (primary && key === 'n') {
      this.options.openWindow?.();
      return true;
    }
    if (primary && input.shift && key === 't') {
      this.reopenRecentlyClosed();
      return true;
    }
    if (primary && key === 't') {
      this.createTab();
      return true;
    }
    if (primary && key === 'w') {
      this.closeTab();
      return true;
    }
    if (primary && key === 'd') {
      const contents = this.activeWebsiteWebContents;
      this.browserData.bookmarks.createBookmark(contents.getTitle(), contents.getURL());
      this.trustedWebContents.send(IPC.browserDataChanged);
      return true;
    }
    if (primary && key === 'l') {
      this.trustedWebContents.send(IPC.browserFocusOmnibox);
      return true;
    }
    if (primary && key === 'r') {
      this.reload();
      return true;
    }
    if (primary && key === 'f') {
      this.showFindInPage();
      return true;
    }
    const tabSearchShortcut = this.browserData.experience.read().tabSearchShortcut;
    if (tabSearchShortcut === TAB_SEARCH_SHORTCUT && primary && input.shift && key === 'a') {
      this.trustedWebContents.send(IPC.browserOpenTabSearch);
      return true;
    }
    if (primary && key === 'p') {
      void this.printPage();
      return true;
    }
    if (primary && key === 's') {
      void this.savePageAs();
      return true;
    }
    if (primary && (key === '+' || key === '=')) {
      this.setZoom('in');
      return true;
    }
    if (primary && key === '-') {
      this.setZoom('out');
      return true;
    }
    if (primary && key === '0') {
      this.setZoom('reset');
      return true;
    }
    if (input.control && key === 'tab') {
      this.tabs.activateRelative(input.shift ? -1 : 1);
      return true;
    }
    if (primary && /^[1-9]$/u.test(key)) {
      this.tabs.activateByIndex(key === '9' ? -1 : Number(key) - 1);
      return true;
    }
    if (key === 'escape' && this.activeWebsiteWebContents.isLoading()) {
      this.stop();
      return true;
    }
    return false;
  }

  private layoutViews(): void {
    const [width = 0, height = 0] = this.window.getContentSize();
    const contentHeight = Math.min(height, this.chromeContentHeight);
    const sidebar = this.browserData.experience.read().sidebar;
    const railWidth = sidebar.visible ? sidebar.width : 0;
    const panelOpen = Boolean(sidebar.activeAppId);
    const panelWidth = panelOpen
      ? Math.min(
          this.sidebarPanelWidthPreview ?? sidebar.panelWidth,
          Math.max(0, width - railWidth),
        )
      : 0;
    const pinnedPanelWidth = panelOpen && sidebar.panelMode === 'pinned' ? panelWidth : 0;
    this.tabs.layout({
      x: railWidth + pinnedPanelWidth,
      y: contentHeight,
      width: Math.max(0, width - railWidth - pinnedPanelWidth),
      height: Math.max(0, height - contentHeight),
    });
    this.trustedView.setBounds({
      x: railWidth,
      y: 0,
      width: Math.max(0, width - railWidth),
      height: this.chromeCoversWindow ? height : contentHeight,
    });
    const websiteLeft = railWidth + pinnedPanelWidth;
    const passiveWidth = Math.min(
      this.passiveOverlayLayout.width,
      Math.max(0, width - websiteLeft - 24),
    );
    const passiveHeight = Math.min(
      this.passiveOverlayLayout.height,
      Math.max(0, height - contentHeight - 12),
    );
    const passiveVisible =
      this.passiveOverlayLayout.visible &&
      !this.chromeCoversWindow &&
      passiveWidth > 0 &&
      passiveHeight > 0;
    this.passiveOverlayView.setVisible(passiveVisible);
    if (passiveWidth > 0) {
      this.passiveOverlayView.setBounds({
        x: Math.max(websiteLeft + 12, width - passiveWidth - 12),
        y: contentHeight + 12,
        width: passiveWidth,
        height: passiveVisible ? passiveHeight : Math.max(1, height - contentHeight - 12),
      });
    }
    this.sidebarView.setVisible(
      sidebar.visible && (!this.chromeCoversWindow || this.chromeAllowsSidebarInteraction),
    );
    this.sidebarView.setBounds({ x: 0, y: 0, width: railWidth, height });
    const panelVisible = panelOpen && !this.chromeCoversWindow && panelWidth > 0;
    this.sidebarApps.setPanelVisible(panelVisible);
    this.sidebarResizeView.setVisible(panelVisible);
    if (panelVisible) {
      this.sidebarApps.layout({
        x: railWidth,
        y: contentHeight,
        width: panelWidth,
        height: Math.max(0, height - contentHeight),
      });
      const resizeHandleWidth = 8;
      this.sidebarResizeView.setBounds({
        x: Math.max(
          railWidth,
          Math.min(width - resizeHandleWidth, railWidth + panelWidth - resizeHandleWidth / 2),
        ),
        y: contentHeight,
        width: resizeHandleWidth,
        height: Math.max(0, height - contentHeight),
      });
    }
    this.raiseTrustedView();
  }

  private raiseTrustedView(): void {
    if (!this.window.isDestroyed()) {
      // Re-adding an existing child is Electron's supported topmost-view operation.
      this.window.contentView.addChildView(this.passiveOverlayView);
      this.window.contentView.addChildView(this.trustedView);
      this.sidebarApps.raise();
      this.window.contentView.addChildView(this.sidebarResizeView);
      this.window.contentView.addChildView(this.sidebarView);
    }
  }

  private publishState(): void {
    if (!this.window.isDestroyed() && this.tabs.getTabs().length > 0) {
      const state = this.getState();
      this.trustedWebContents.send(IPC.browserStateChanged, state);
      this.sendToPassiveOverlay(IPC.browserStateChanged, state);
      this.sidebarView.webContents.send(IPC.browserStateChanged, state);
      this.sidebarResizeView.webContents.send(IPC.browserStateChanged, state);
    }
  }

  private experienceState(): BrowserWindowState['experience'] {
    const value = this.browserData.experience.read();
    return {
      activeWorkspaceId: value.activeWorkspaceId,
      workspaces: value.workspaces.map(({ workspaceId, name, accent, order }) => ({
        workspaceId,
        name,
        accent,
        order,
      })),
      islands: value.islands.map(({ islandId, workspaceId, name, accent, collapsed, order }) => ({
        islandId,
        workspaceId,
        name,
        accent,
        collapsed,
        order,
      })),
      recentlyClosed: this.browserData.sessions.listRecentlyClosed(),
      sidebar: {
        visible: value.sidebar.visible,
        width: value.sidebar.width,
        panelWidth: value.sidebar.panelWidth,
        panelMode: value.sidebar.panelMode,
        activeAppId: value.sidebar.activeAppId,
        tools: value.sidebar.tools.map(({ toolId, name, enabled, order }) => ({
          toolId,
          name,
          enabled,
          order,
        })),
        apps: value.sidebar.apps.map(({ appId, name, enabled, order, muted }) => ({
          appId,
          name,
          enabled,
          order,
          muted,
        })),
      },
      tabSearchShortcut: value.tabSearchShortcut,
      autoGroupOpenerTabs: value.autoGroupOpenerTabs,
    };
  }
}

export { normalizeNavigationInput };
