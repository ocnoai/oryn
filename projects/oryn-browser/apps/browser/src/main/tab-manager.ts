import { isFaviconDataURL, resolveFaviconDataURL } from './favicon';
import { requestPageClose } from './page-close';
import { boundedObservationReady } from './observation-readiness';
import { randomUUID } from 'node:crypto';
import { isAbsolute } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  consumeInsecureNavigationCancellation,
  registerPopupNavigationSource,
} from './insecure-form-guard';
import { EventEmitter } from 'node:events';

import { WebContentsView, type BrowserWindow, type WebContents } from 'electron';

import type { BrowserCallerContextService } from './browser-caller-context';
import type { BrowserDataService } from './data/browser-data-service';
import {
  isExternalApplicationNavigation,
  type ExternalApplicationHandoff,
} from './external-application-handoff';
import type { BrowserTabState } from '../shared/ipc';
import { DEFAULT_WORKSPACE_ID } from './data/browser-experience-store';
import {
  createOrynNetworkErrorURL,
  ORYN_NEW_TAB_URL,
  readOrynNetworkErrorURL,
  readOrynModelWorkspaceReference,
} from '../shared/browser-urls';

export const DEFAULT_START_URL = ORYN_NEW_TAB_URL;

export interface BrowserTab extends BrowserTabState {
  view: WebContentsView;
  activeDocumentId: string;
}

interface ManagedTab {
  tabId: string;
  view: WebContentsView;
  unregisterCallerContext: () => void;
  currentURL: string;
  failedNavigationURL?: string;
  lastRecordedDocumentId?: string;
  favicon?: string;
  faviconPending: boolean;
  faviconRequestId: number;
  faviconAbort?: AbortController;
  closing?: Promise<boolean>;
  crashed?: boolean;
  lastHistoryURL?: string;
  unwire?: () => void;
  pendingURL?: string;
  observationReady?: Promise<void>;
  observationStarted?: boolean;
  loadGeneration?: number;
  rendererBootstrapped?: boolean;
  rendererBootstrapReady?: Promise<void>;
  securityClass: 'website' | 'model-workspace' | 'local-file';
  workspaceId: string;
  islandId?: string;
  pinned: boolean;
}

export interface TabCreationOptions {
  tabId?: string;
  favicon?: string;
  workspaceId?: string;
  islandId?: string;
  pinned?: boolean;
  popupOptions?: Electron.BrowserWindowConstructorOptions;
  popupView?: WebContentsView;
  lazy?: boolean;
}

export interface TabManagerOptions {
  externalApplications?: ExternalApplicationHandoff;
  websitePreloadPath: string;
  workspacePreloadPath: string;
  privatePreloadPath?: string;
  privateMode?: boolean;
  session?: Electron.Session;
  onWebContentsCreated?: (webContents: WebContents) => void;
  beforeFirstNavigation?: (webContents: WebContents) => void | Promise<void>;
  onActiveViewChanged?: () => void;
}

export class TabManager extends EventEmitter {
  private readonly tabs = new Map<string, ManagedTab>();
  private activeTabId?: string;
  private bounds = { x: 0, y: 0, width: 0, height: 0 };
  private destroying = false;
  private batching = false;
  private restartFrozen = false;

  constructor(
    private readonly window: BrowserWindow,
    private readonly callerContexts: BrowserCallerContextService,
    private readonly browserData: BrowserDataService,
    private readonly options: TabManagerOptions,
  ) {
    super();
  }

  batch(operation: () => void): void {
    this.batching = true;
    try {
      operation();
    } finally {
      this.batching = false;
      this.publishState();
    }
  }

  createTab(
    url = DEFAULT_START_URL,
    activate = true,
    organization: TabCreationOptions = {},
  ): BrowserTab {
    if (this.restartFrozen) throw new Error('BROWSER_RESTARTING');
    const experience = this.browserData.experience.read();
    if (
      experience.workspaces &&
      organization.workspaceId &&
      !experience.workspaces.some((item) => item.workspaceId === organization.workspaceId)
    ) {
      organization = { ...organization, workspaceId: DEFAULT_WORKSPACE_ID, islandId: undefined };
    }
    if (
      organization.islandId &&
      experience.workspaces &&
      !experience.islands.some(
        (item) =>
          item.islandId === organization.islandId &&
          item.workspaceId === (organization.workspaceId ?? experience.activeWorkspaceId),
      )
    )
      organization = { ...organization, islandId: undefined };
    const normalizedURL = normalizeNavigationInput(url);
    if (this.options.privateMode && readOrynModelWorkspaceReference(normalizedURL))
      throw new Error('PRIVATE_LOCAL_AI_UNAVAILABLE');
    const securityClass = normalizedURL.startsWith('file:')
      ? 'local-file'
      : readOrynModelWorkspaceReference(normalizedURL)
        ? 'model-workspace'
        : 'website';
    const tabId = organization.tabId ?? `tab_${randomUUID().replaceAll('-', '')}`;
    if (this.tabs.has(tabId)) throw new Error('TAB_ID_DUPLICATE');
    const view =
      organization.popupView ??
      new WebContentsView({
        webPreferences: {
          ...organization.popupOptions?.webPreferences,
          session: this.options.session,
          preload:
            securityClass === 'local-file'
              ? undefined
              : this.options.privateMode
                ? this.options.privatePreloadPath
                : securityClass === 'model-workspace'
                  ? this.options.workspacePreloadPath
                  : this.options.websitePreloadPath,
          nodeIntegration: false,
          contextIsolation: true,
          sandbox: true,
          webSecurity: true,
          spellcheck: true,
        },
      });
    const managed: ManagedTab = {
      tabId,
      view,
      unregisterCallerContext: this.callerContexts.register(
        tabId,
        view.webContents,
        securityClass === 'local-file'
          ? 'local-file'
          : this.options.privateMode
            ? 'private'
            : securityClass,
      ),
      currentURL: normalizedURL,
      pendingURL: organization.lazy ? normalizedURL : undefined,
      favicon: isFaviconDataURL(organization.favicon) ? organization.favicon : undefined,
      faviconPending: false,
      faviconRequestId: 0,
      securityClass,
      rendererBootstrapped: Boolean(organization.popupOptions || organization.popupView),
      workspaceId: organization.workspaceId ?? this.browserData.experience.read().activeWorkspaceId,
      ...(organization.islandId ? { islandId: organization.islandId } : {}),
      pinned: organization.pinned ?? false,
    };
    this.tabs.set(tabId, managed);
    this.wireTab(managed);

    if (activate || (!this.activeTabId && !organization.lazy)) {
      this.activateTab(tabId);
    } else {
      this.publishState();
    }
    if (!organization.popupOptions && !organization.lazy)
      this.loadWhenObserved(managed, normalizedURL);
    return this.getTab(tabId);
  }

  private wireTab(tab: ManagedTab): void {
    const contents = tab.view.webContents;
    const before = new Map(contents.eventNames().map((name) => [name, contents.listeners(name)]));
    const unregisterExternal = this.options.externalApplications?.register(
      contents,
      () =>
        !this.options.privateMode &&
        this.tabs.get(tab.tabId) === tab &&
        tab.securityClass === 'website',
    );
    this.installNavigationPolicy(tab);
    this.installStateTracking(tab);
    this.browserData.installChromiumSessionHandlers(contents.session);
    this.options.onWebContentsCreated?.(contents);
    tab.observationStarted = false;
    tab.observationReady = undefined;
    if (tab.rendererBootstrapped) this.beginObservation(tab);
    const owned = contents.eventNames().flatMap((name) =>
      contents
        .listeners(name)
        .filter((listener) => !before.get(name)?.includes(listener))
        .map((listener) => ({ name, listener })),
    );
    tab.unwire = () => {
      unregisterExternal?.();
      for (const { name, listener } of owned)
        EventEmitter.prototype.removeListener.call(contents, name, listener);
    };
  }

  transferTabTo(tabId: string, destination: TabManager): void {
    if (destination === this) return;
    if (this.options.privateMode || destination.options.privateMode)
      throw new Error('PRIVATE_TAB_MOVE_UNAVAILABLE');
    const tab = this.requireTab(tabId);
    if (tab.closing || destination.tabs.has(tabId)) throw new Error('TAB_MOVE_UNAVAILABLE');
    if (this.activeTabId === tabId) {
      this.window.contentView.removeChildView(tab.view);
      this.activeTabId = undefined;
    }
    tab.unwire?.();
    this.tabs.delete(tabId);
    destination.tabs.set(tabId, tab);
    destination.wireTab(tab);
    destination.activateTab(tabId);
    this.recoverOrganization();
    this.publishState();
  }

  async closeTabs(ids: string[]): Promise<boolean> {
    this.destroying = true;
    try {
      for (const id of ids) if (!(await this.closeTab(id))) return false;
      return true;
    } finally {
      this.destroying = false;
      if (!this.activeTabId && this.tabs.size) this.activateTab(this.tabs.keys().next().value!);
    }
  }

  recoverOrganization(): void {
    const experience = this.browserData.experience.read();
    for (const tab of this.tabs.values()) {
      if (!experience.workspaces.some((item) => item.workspaceId === tab.workspaceId)) {
        tab.workspaceId = DEFAULT_WORKSPACE_ID;
        tab.islandId = undefined;
      }
      if (
        tab.islandId &&
        !experience.islands.some(
          (item) => item.islandId === tab.islandId && item.workspaceId === tab.workspaceId,
        )
      )
        tab.islandId = undefined;
    }
    if (!this.tabs.size) this.createTab(DEFAULT_START_URL);
    else if (!this.activeTabId) this.activateTab(this.tabs.keys().next().value!);
    this.publishState();
  }

  openTabFrom(
    contents: WebContents,
    url: string,
    activate = false,
    popupOptions?: Electron.BrowserWindowConstructorOptions,
    popupView?: WebContentsView,
  ): BrowserTab {
    if (!/^(https?:|about:blank$)/u.test(url)) throw new Error('UNTRUSTED_LINK_URL_BLOCKED');
    const opener = [...this.tabs.values()].find((tab) => tab.view.webContents === contents);
    const created = this.createTab(url, activate, {
      popupOptions,
      popupView,
      workspaceId: opener?.workspaceId,
      islandId: opener?.islandId,
    });
    if (opener && !opener.islandId && this.browserData.experience.read().autoGroupOpenerTabs) {
      this.createIsland(
        [opener.tabId, created.tabId],
        contents.getTitle() || 'Related tabs',
        opener.workspaceId,
      );
    }
    return this.getTab(created.tabId);
  }

  activateTab(tabId: string): void {
    const next = this.requireTab(tabId);
    if (next.workspaceId !== this.browserData.experience.read().activeWorkspaceId) {
      this.browserData.experience.setActiveWorkspace(next.workspaceId);
    }
    if (this.activeTabId === tabId) {
      return;
    }

    const current = this.activeTabId ? this.tabs.get(this.activeTabId) : undefined;
    const currentContents = current?.view.webContents;
    if (current && currentContents && !currentContents.isDestroyed()) {
      this.window.contentView.removeChildView(current.view);
    }

    this.activeTabId = tabId;
    this.window.contentView.addChildView(next.view);
    next.view.setBounds(this.bounds);
    if (next.pendingURL) {
      const url = next.pendingURL;
      this.loadWhenObserved(next, url);
    }
    this.options.onActiveViewChanged?.();
    this.publishState();
  }

  async closeTab(tabId = this.activeTabId): Promise<boolean> {
    if (!tabId || !this.tabs.has(tabId)) return true;
    const tab = this.requireTab(tabId);
    if (tab.closing) return tab.closing;
    const contents = tab.view.webContents;
    if (!contents || contents.isDestroyed()) {
      this.removeClosedTab(tab);
      return true;
    }
    const title = contents.getTitle();
    const url = contents.getURL();
    tab.closing = requestPageClose(contents, this.window);
    try {
      const accepted = await tab.closing;
      if (accepted) this.removeClosedTab(tab, url, title);
      return accepted;
    } finally {
      tab.closing = undefined;
    }
  }

  private removeClosedTab(tab: ManagedTab, url = tab.currentURL, title = ''): void {
    if (this.tabs.get(tab.tabId) !== tab) return;
    const tabId = tab.tabId;
    const wasActive = this.activeTabId === tabId;
    const tabIds = this.visibleTabIds(tab.workspaceId);
    const closedIndex = tabIds.indexOf(tabId);
    // Cleanup can synchronously notify other state observers. Remove the tab before
    // those callbacks can try to read a WebContents that Electron has already cleared.
    this.tabs.delete(tabId);
    if (wasActive) this.activeTabId = undefined;
    if (url) {
      this.browserData.sessions.pushRecentlyClosed({
        url,
        title,
        workspaceId: tab.workspaceId,
        islandId: tab.islandId,
        pinned: tab.pinned,
        closedAt: new Date().toISOString(),
      });
    }

    if (wasActive && !this.window.isDestroyed()) {
      this.window.contentView.removeChildView(tab.view);
    }
    tab.unwire?.();
    tab.unregisterCallerContext();
    tab.faviconAbort?.abort();
    tab.favicon = undefined;
    tab.faviconPending = false;
    tab.faviconRequestId += 1;
    const remainingContents = tab.view.webContents;
    if (remainingContents && !remainingContents.isDestroyed()) remainingContents.close();

    if (this.destroying) {
      return;
    }
    if (this.tabs.size === 0) {
      this.createTab(DEFAULT_START_URL, true, { workspaceId: tab.workspaceId });
      return;
    }
    if (wasActive) {
      const remaining = this.visibleTabIds(tab.workspaceId);
      if (remaining.length === 0) {
        const fallbackWorkspace = this.browserData.experience
          .read()
          .workspaces.find((workspace) =>
            [...this.tabs.values()].some((item) => item.workspaceId === workspace.workspaceId),
          );
        if (fallbackWorkspace) {
          this.switchWorkspace(fallbackWorkspace.workspaceId);
          return;
        }
      }
      const nextIndex = Math.min(Math.max(0, closedIndex), remaining.length - 1);
      const next = remaining[nextIndex];
      if (next) this.activateTab(next);
    } else {
      this.publishState();
    }
  }

  reopenRecentlyClosed(): BrowserTab | undefined {
    const closed = this.browserData.sessions.listRecentlyClosed()[0];
    if (!closed) return undefined;
    const restored = this.createTab(closed.url, true, {
      workspaceId: closed.workspaceId,
      islandId: closed.islandId,
      pinned: closed.pinned,
    });
    this.browserData.sessions.popRecentlyClosed();
    return restored;
  }

  activateRelative(offset: number): void {
    const ids = this.visibleTabIds();
    if (ids.length < 2 || !this.activeTabId) {
      return;
    }
    const activeIndex = ids.indexOf(this.activeTabId);
    const nextIndex = (activeIndex + offset + ids.length) % ids.length;
    this.activateTab(ids[nextIndex] as string);
  }

  activateByIndex(index: number): void {
    const ids = this.visibleTabIds();
    const resolvedIndex = index === -1 ? ids.length - 1 : index;
    const tabId = ids[resolvedIndex];
    if (tabId) {
      this.activateTab(tabId);
    }
  }

  getActiveWebContents(): WebContents {
    return this.requireActiveTab().view.webContents;
  }

  getActiveTabId(): string {
    return this.requireActiveTab().tabId;
  }

  getTabs(): BrowserTab[] {
    return [...this.tabs.values()]
      .map((tab) => this.snapshot(tab))
      .filter((tab) => tab !== undefined);
  }

  getActiveTab(): BrowserTab | undefined {
    const tab = this.activeTabId ? this.tabs.get(this.activeTabId) : undefined;
    return tab ? this.snapshot(tab) : undefined;
  }

  getTab(tabId: string): BrowserTab {
    const snapshot = this.snapshot(this.requireTab(tabId));
    if (!snapshot) throw new Error(`Browser tab is closed: ${tabId}`);
    return snapshot;
  }

  switchWorkspace(workspaceId: string): void {
    this.browserData.experience.setActiveWorkspace(workspaceId);
    const nextId = this.visibleTabIds(workspaceId)[0];
    if (nextId) {
      this.activateTab(nextId);
    } else {
      this.createTab(DEFAULT_START_URL, true, { workspaceId });
    }
  }

  moveTabToWorkspace(tabId: string, workspaceId: string): void {
    const tab = this.requireTab(tabId);
    if (
      !this.browserData.experience
        .read()
        .workspaces.some((item) => item.workspaceId === workspaceId)
    ) {
      throw new Error('WORKSPACE_NOT_FOUND');
    }
    tab.workspaceId = workspaceId;
    tab.islandId = undefined;
    if (this.activeTabId === tabId) {
      const remaining = this.visibleTabIds(this.browserData.experience.read().activeWorkspaceId);
      if (remaining.length > 0) this.activateTab(remaining[0] as string);
      else this.switchWorkspace(workspaceId);
    } else {
      this.publishState();
    }
  }

  async deleteWorkspace(workspaceId: string, action: 'move-to-default' | 'close'): Promise<void> {
    if (workspaceId === DEFAULT_WORKSPACE_ID) throw new Error('DEFAULT_WORKSPACE_REQUIRED');
    const affected = [...this.tabs.values()].filter((tab) => tab.workspaceId === workspaceId);
    if (action === 'close') {
      for (const tab of affected) if (!(await this.closeTab(tab.tabId))) return;
    } else {
      for (const tab of affected) {
        tab.workspaceId = DEFAULT_WORKSPACE_ID;
        tab.islandId = undefined;
      }
    }
    this.browserData.experience.deleteWorkspace(workspaceId);
    if (this.browserData.experience.read().activeWorkspaceId === DEFAULT_WORKSPACE_ID) {
      const next = this.visibleTabIds(DEFAULT_WORKSPACE_ID)[0];
      if (next) this.activateTab(next);
      else this.createTab(DEFAULT_START_URL, true, { workspaceId: DEFAULT_WORKSPACE_ID });
    }
    this.publishState();
  }

  reorder(tabId: string, beforeTabId?: string): void {
    const tab = this.requireTab(tabId);
    const target = beforeTabId ? this.requireTab(beforeTabId) : undefined;
    if (target && target.workspaceId !== tab.workspaceId) throw new Error('TAB_WORKSPACE_MISMATCH');
    if (target?.tabId === tabId) return;
    tab.pinned = target?.pinned ?? false;
    tab.islandId = target?.islandId;
    const ordered = [...this.tabs.values()].filter((item) => item !== tab);
    const index = target ? ordered.indexOf(target) : ordered.length;
    ordered.splice(index, 0, tab);
    this.tabs.clear();
    for (const item of ordered) this.tabs.set(item.tabId, item);
    this.publishState();
  }

  mute(tabId: string, muted: boolean): void {
    this.requireTab(tabId).view.webContents.setAudioMuted(muted);
    this.publishState();
  }

  setPinned(tabId: string, pinned: boolean): void {
    this.requireTab(tabId).pinned = pinned;
    this.publishState();
  }

  duplicateTab(tabId: string): BrowserTab {
    const tab = this.requireTab(tabId);
    return this.createTab(tab.view.webContents.getURL() || tab.currentURL, true, {
      workspaceId: tab.workspaceId,
      islandId: tab.islandId,
      pinned: tab.pinned,
    });
  }

  async closeOtherTabs(tabId: string): Promise<void> {
    const tab = this.requireTab(tabId);
    for (const candidate of [...this.tabs.values()]) {
      if (
        candidate.workspaceId === tab.workspaceId &&
        candidate.tabId !== tabId &&
        !candidate.pinned
      ) {
        if (!(await this.closeTab(candidate.tabId))) break;
      }
    }
    this.activateTab(tabId);
  }

  async closeTabsToRight(tabId: string): Promise<void> {
    const tab = this.requireTab(tabId);
    const visible = this.visibleTabIds(tab.workspaceId);
    const index = visible.indexOf(tabId);
    for (const candidateId of visible.slice(index + 1)) {
      if (!this.requireTab(candidateId).pinned && !(await this.closeTab(candidateId))) break;
    }
  }

  async closeDuplicateTabs(): Promise<void> {
    const workspaceId = this.browserData.experience.read().activeWorkspaceId;
    const active = this.activeTabId;
    const byURL = new Map<string, string>();
    for (const tabId of this.visibleTabIds(workspaceId)) {
      const tab = this.requireTab(tabId);
      const canonical = canonicalTabURL(tab.view.webContents.getURL() || tab.currentURL);
      const kept = byURL.get(canonical);
      if (!kept || tabId === active) {
        if (
          kept &&
          kept !== active &&
          !this.requireTab(kept).pinned &&
          !(await this.closeTab(kept))
        )
          break;
        byURL.set(canonical, tabId);
      } else if (tabId !== active && !tab.pinned) {
        if (!(await this.closeTab(tabId))) break;
      }
    }
  }

  createIsland(
    tabIds: string[],
    name?: string,
    workspaceId = this.browserData.experience.read().activeWorkspaceId,
  ): void {
    const valid = tabIds.map((id) => this.requireTab(id));
    if (valid.some((tab) => tab.workspaceId !== workspaceId))
      throw new Error('TAB_ISLAND_WORKSPACE_MISMATCH');
    if (valid.length === 0) throw new Error('TAB_ISLAND_REQUIRES_TABS');
    const island = this.browserData.experience.createIsland(
      workspaceId,
      name ?? valid[0]?.view.webContents.getTitle() ?? 'Tab island',
      this.browserData.experience.read().islands.filter((item) => item.workspaceId === workspaceId)
        .length,
    );
    for (const tab of valid) tab.islandId = island.islandId;
    this.publishState();
  }

  moveTabToIsland(tabId: string, islandId?: string): void {
    const tab = this.requireTab(tabId);
    if (islandId) {
      const island = this.browserData.experience
        .read()
        .islands.find((item) => item.islandId === islandId);
      if (!island || island.workspaceId !== tab.workspaceId)
        throw new Error('TAB_ISLAND_NOT_FOUND');
    }
    tab.islandId = islandId;
    this.publishState();
  }

  deleteIsland(islandId: string): void {
    for (const tab of this.tabs.values()) if (tab.islandId === islandId) tab.islandId = undefined;
    this.browserData.experience.deleteIsland(islandId);
    this.publishState();
  }

  moveIslandToWorkspace(islandId: string, workspaceId: string): void {
    const experience = this.browserData.experience.read();
    const island = experience.islands.find((item) => item.islandId === islandId);
    if (!island || !experience.workspaces.some((item) => item.workspaceId === workspaceId)) {
      throw new Error('TAB_ISLAND_NOT_FOUND');
    }
    for (const tab of this.tabs.values()) {
      if (tab.islandId === islandId) tab.workspaceId = workspaceId;
    }
    this.browserData.experience.updateIsland(islandId, { workspaceId });
    if (
      this.activeTabId &&
      this.tabs.get(this.activeTabId)?.islandId === islandId &&
      this.browserData.experience.read().activeWorkspaceId !== workspaceId
    ) {
      const next = this.visibleTabIds(this.browserData.experience.read().activeWorkspaceId)[0];
      if (next) this.activateTab(next);
      else this.switchWorkspace(workspaceId);
    }
    this.publishState();
  }

  navigate(input: string): void {
    if (this.restartFrozen) throw new Error('BROWSER_RESTARTING');
    const tab = this.requireActiveTab();
    tab.failedNavigationURL = undefined;
    const target = normalizeNavigationInput(
      input,
      this.browserData.appearance.read().searchProvider,
    );
    const securityClass = target.startsWith('file:')
      ? 'local-file'
      : readOrynModelWorkspaceReference(target)
        ? 'model-workspace'
        : 'website';
    if (securityClass !== tab.securityClass) {
      if (securityClass === 'model-workspace') throw new Error('OPEN_MODEL_FROM_LIBRARY');
      this.createTab(target, true, { workspaceId: tab.workspaceId });
      return;
    }
    this.loadWhenObserved(tab, target);
  }

  back(): void {
    if (this.restartFrozen) return;
    const history = this.getActiveWebContents().navigationHistory;
    if (history.canGoBack()) {
      const tab = this.requireActiveTab();
      tab.loadGeneration = (tab.loadGeneration ?? 0) + 1;
      tab.pendingURL = undefined;
      history.goBack();
    }
  }

  forward(): void {
    if (this.restartFrozen) return;
    const history = this.getActiveWebContents().navigationHistory;
    if (history.canGoForward()) {
      const tab = this.requireActiveTab();
      tab.loadGeneration = (tab.loadGeneration ?? 0) + 1;
      tab.pendingURL = undefined;
      history.goForward();
    }
  }

  reload(): void {
    if (this.restartFrozen) return;
    const tab = this.requireActiveTab();
    tab.crashed = false;
    if (tab.failedNavigationURL) {
      const target = tab.failedNavigationURL;
      tab.failedNavigationURL = undefined;
      this.loadWhenObserved(tab, target);
      return;
    }
    if (tab.pendingURL) {
      this.loadWhenObserved(tab, tab.pendingURL);
      return;
    }
    tab.view.webContents.reload();
  }

  stop(): void {
    const tab = this.requireActiveTab();
    tab.loadGeneration = (tab.loadGeneration ?? 0) + 1;
    tab.pendingURL = undefined;
    tab.view.webContents.stop();
  }

  private loadWhenObserved(tab: ManagedTab, url: string): void {
    const generation = (tab.loadGeneration = (tab.loadGeneration ?? 0) + 1);
    const contents = tab.view.webContents;
    const securityClass = tab.securityClass;
    const smoke = process.env.LOCAL_AI_SMOKE_TEST === '1';
    const location = (value: unknown) => {
      if (typeof value !== 'string' || !value) return undefined;
      try {
        const parsed = new URL(value);
        return { scheme: parsed.protocol, origin: parsed.origin, blank: value === 'about:blank' };
      } catch {
        return undefined;
      }
    };
    const trace = (phase: string, details: Record<string, unknown> = {}) => {
      if (!smoke) return;
      try {
        const destroyed = contents.isDestroyed();
        console.info(
          `ORYN_NAVIGATION_SMOKE_TRACE ${JSON.stringify({
            phase,
            tabId: tab.tabId,
            generation,
            currentGeneration: tab.loadGeneration,
            securityClass,
            target: location(url),
            current: destroyed || !contents.getURL() ? undefined : location(contents.getURL()),
            loading: destroyed ? undefined : contents.isLoading(),
            ...details,
          })}`,
        );
      } catch {
        // Diagnostic native getters must never change navigation or teardown.
      }
    };
    const failure = (error: unknown) => {
      const code = error && typeof error === 'object' && 'code' in error ? error.code : undefined;
      return {
        code: typeof code === 'string' && /^ERR_[A-Z0-9_]+$/u.test(code) ? code : 'unknown',
      };
    };
    tab.pendingURL = url;
    this.beginObservation(tab);
    trace('requested', {
      hasObserverReady: Boolean(tab.observationReady),
      rendererBootstrapped: tab.rendererBootstrapped,
    });
    if (smoke && tab.observationReady)
      void tab.observationReady.then(
        () => trace('observation-ready'),
        (error) => trace('observation-rejected', failure(error)),
      );
    // A new view needs an inert renderer before Page/Network setup can finish.
    // Native popup guests already own their navigation and are never primed here.
    if (tab.observationReady && !tab.rendererBootstrapped) {
      tab.rendererBootstrapped = true;
      if (smoke) {
        const documentRequests = new Set<string>();
        const cdp = (_event: Electron.Event, method: string, raw: unknown, sessionId?: string) => {
          const object = (value: unknown): Record<string, unknown> =>
            value && typeof value === 'object' && !Array.isArray(value)
              ? (value as Record<string, unknown>)
              : {};
          const params = object(raw);
          if (method === 'Target.attachedToTarget' || method === 'Target.detachedFromTarget') {
            const target = object(params.targetInfo);
            const type = [
              'iframe',
              'page',
              'tab',
              'worker',
              'service_worker',
              'shared_worker',
            ].includes(String(target.type))
              ? target.type
              : 'unknown';
            trace('cdp-target', {
              attached: method === 'Target.attachedToTarget',
              type,
              waitingForDebugger: params.waitingForDebugger === true,
              hasParentSession: Boolean(sessionId),
              hasChildSession: typeof params.sessionId === 'string',
              targetLocation: location(target.url),
            });
            return;
          }
          const requestKey = `${sessionId ?? ''}:${String(params.requestId ?? '')}`;
          if (
            (method === 'Network.requestWillBeSent' || method === 'Network.responseReceived') &&
            params.type === 'Document'
          ) {
            if (documentRequests.size < 128) documentRequests.add(requestKey);
            const payload = object(
              method === 'Network.requestWillBeSent' ? params.request : params.response,
            );
            trace('cdp-document', {
              response: method === 'Network.responseReceived',
              hasParentSession: Boolean(sessionId),
              documentLocation: location(payload.url),
              status: typeof payload.status === 'number' ? payload.status : undefined,
            });
          } else if (method === 'Network.loadingFailed' && documentRequests.delete(requestKey)) {
            trace('cdp-document-failed', {
              ...failure({
                code:
                  typeof params.errorText === 'string'
                    ? params.errorText.replace(/^net::/u, '')
                    : undefined,
              }),
              cancelled: params.canceled === true,
              hasParentSession: Boolean(sessionId),
            });
          } else if (method === 'Network.loadingFinished' && documentRequests.delete(requestKey)) {
            trace('cdp-document-finished', { hasParentSession: Boolean(sessionId) });
          }
        };
        const started = (
          _event: Electron.Event,
          destination: string,
          sameDocument: boolean,
          mainFrame: boolean,
        ) => {
          if (mainFrame)
            trace('native-navigation-started', {
              destination: location(destination),
              sameDocument,
            });
        };
        const committed = (_event: Electron.Event, destination: string) =>
          trace('native-navigation-committed', { destination: location(destination) });
        const finished = () => trace('native-load-finished');
        const stopped = () => trace('native-load-stopped');
        const failed = (
          _event: Electron.Event,
          code: number,
          _description: string,
          destination: string,
          mainFrame: boolean,
        ) =>
          trace('native-load-failed', {
            nativeCode: Number.isInteger(code) ? code : undefined,
            destination: location(destination),
            mainFrame,
          });
        const debuggerChannel = contents.debugger;
        debuggerChannel?.on('message', cdp);
        contents.on('did-start-navigation', started);
        contents.on('did-navigate', committed);
        contents.on('did-finish-load', finished);
        contents.on('did-stop-loading', stopped);
        contents.on('did-fail-load', failed);
        contents.once('destroyed', () => {
          try {
            debuggerChannel?.removeListener('message', cdp);
          } catch {
            // A destroyed native debugger must not interrupt subsequent cleanup.
          }
          try {
            contents.removeListener('did-start-navigation', started);
            contents.removeListener('did-navigate', committed);
            contents.removeListener('did-finish-load', finished);
            contents.removeListener('did-stop-loading', stopped);
            contents.removeListener('did-fail-load', failed);
          } catch {
            // Native teardown is best effort for this smoke-only subscription.
          }
          documentRequests.clear();
        });
      }
      if (!contents.getURL()) {
        trace('bootstrap-started');
        tab.rendererBootstrapReady = contents.loadURL('about:blank').then(
          () => trace('bootstrap-resolved'),
          (error) => trace('bootstrap-rejected', failure(error)),
        );
      }
    }
    const load = () => {
      if (
        this.destroying ||
        this.restartFrozen ||
        this.window.isDestroyed() ||
        this.tabs.get(tab.tabId) !== tab ||
        tab.view.webContents !== contents ||
        contents.isDestroyed() ||
        tab.closing ||
        tab.loadGeneration !== generation ||
        tab.pendingURL !== url ||
        tab.securityClass !== securityClass
      ) {
        if (smoke)
          trace('load-skipped', {
            destroying: this.destroying,
            restartFrozen: this.restartFrozen,
            windowDestroyed: this.window.isDestroyed(),
            sameTab: this.tabs.get(tab.tabId) === tab,
            sameContents: tab.view.webContents === contents,
            contentsDestroyed: contents.isDestroyed(),
            closing: Boolean(tab.closing),
            sameGeneration: tab.loadGeneration === generation,
            samePendingTarget: tab.pendingURL === url,
            sameSecurityClass: tab.securityClass === securityClass,
          });
        return;
      }
      tab.pendingURL = undefined;
      trace('target-started');
      void contents.loadURL(url).then(
        () => trace('target-resolved'),
        (error) => trace('target-rejected', failure(error)),
      );
    };
    if (tab.observationReady || tab.rendererBootstrapReady)
      void Promise.all([tab.observationReady, tab.rendererBootstrapReady]).then(load);
    else load();
  }

  private beginObservation(tab: ManagedTab): void {
    if (tab.observationStarted) return;
    tab.observationStarted = true;
    tab.observationReady = boundedObservationReady(
      this.options.beforeFirstNavigation?.(tab.view.webContents),
    );
  }

  openDevTools(): void {
    this.getActiveWebContents().openDevTools({ mode: 'detach' });
  }

  async allowsUpdateRestart(): Promise<boolean> {
    // Electron has no non-destructive native beforeunload preflight. Never partially
    // close a set of documents then call a canceled update harmless.
    return [...this.tabs.values()].every(
      (tab) =>
        !tab.pendingURL &&
        !tab.view.webContents.isLoading() &&
        tab.view.webContents.getURL() === DEFAULT_START_URL,
    );
  }

  freezeForRestart(): void {
    this.restartFrozen = true;
  }
  cancelRestart(): void {
    this.restartFrozen = false;
  }

  async closeAll(): Promise<boolean> {
    this.destroying = true;
    try {
      for (const id of [...this.tabs.keys()]) if (!(await this.closeTab(id))) return false;
      return true;
    } finally {
      this.destroying = false;
      if (!this.activeTabId && this.tabs.size) this.activateTab(this.tabs.keys().next().value!);
    }
  }

  layout(bounds: { x: number; y: number; width: number; height: number }): void {
    this.bounds = bounds;
    const active = this.activeTabId ? this.tabs.get(this.activeTabId) : undefined;
    active?.view.setBounds(bounds);
  }

  destroy(): void {
    this.destroying = true;
    for (const tabId of [...this.tabs.keys()]) {
      const tab = this.requireTab(tabId);
      const contents = tab.view.webContents;
      if (contents && !contents.isDestroyed()) contents.close();
      if (this.tabs.has(tabId)) this.removeClosedTab(tab);
    }
    this.activeTabId = undefined;
    this.destroying = false;
  }

  private installNavigationPolicy(tab: ManagedTab): void {
    const contents = tab.view.webContents;
    contents.on('will-frame-navigate', (details) => {
      if (!isExternalApplicationNavigation(details.url)) return;
      details.preventDefault();
      if (!this.restartFrozen)
        void this.options.externalApplications?.requestNavigation(contents, details);
    });
    contents.setWindowOpenHandler(({ url, disposition, referrer, postBody }) => {
      if (this.restartFrozen) return { action: 'deny' };
      try {
        if (
          this.browserData.sitePermissions?.get(new URL(contents.getURL()).origin, 'popups') ===
          'deny'
        )
          return { action: 'deny' };
      } catch {
        /* Uncommitted popup pages are governed by the opener relationship below. */
      }
      if (tab.securityClass !== 'website' || !/^(https?:|about:blank$)/u.test(url))
        // Electron's popup HandlerDetails contains no initiating frame. Referrer/target
        // identity is not proof that a top-level document (rather than an iframe) initiated it.
        return { action: 'deny' };
      return {
        action: 'allow',
        createWindow: (options) => {
          // Link gestures can include an empty guest; WebContentsView rejects a present
          // webContents property unless it contains an actual Chromium WebContents.
          const guest = (
            options as Electron.BrowserWindowConstructorOptions & {
              webContents?: WebContents | null;
            }
          ).webContents;
          const view = new WebContentsView({
            ...(guest ? { webContents: guest } : {}),
            webPreferences: {
              ...options.webPreferences,
              session: this.options.session,
              preload: this.options.privateMode
                ? this.options.privatePreloadPath
                : this.options.websitePreloadPath,
              nodeIntegration: false,
              contextIsolation: true,
              sandbox: true,
              webSecurity: true,
            },
          });
          registerPopupNavigationSource(view.webContents, contents);
          // Chromium completes child construction only after this callback returns.
          // Synchronous document queries here can deadlock the opener's popup request.
          setImmediate(() => {
            const child = view.webContents;
            if (!child || child.isDestroyed()) return;
            if (!this.window.isDestroyed() && !this.destroying && !contents.isDestroyed()) {
              this.openTabFrom(contents, url, disposition !== 'background-tab', options, view);
              if (!guest)
                void child
                  .loadURL(url, {
                    httpReferrer: referrer,
                    ...(postBody
                      ? {
                          postData: postBody.data,
                          extraHeaders: `Content-Type: ${postBody.contentType}\r\n`,
                        }
                      : {}),
                  })
                  .catch(() => undefined);
            } else child.close();
          });
          return view.webContents;
        },
      };
    });
    const guardDestination = (event: Electron.Event, url: string) => {
      // Trusted loadURL calls do not trigger will-navigate; website-controlled navigations
      // never acquire a file or privileged workspace principal.
      if (url.startsWith('file:') || isExternalApplicationNavigation(url)) {
        event.preventDefault();
        return;
      }
      if (tab.securityClass === 'local-file' && /^https?:/u.test(url)) {
        event.preventDefault();
        this.createTab(url, true, { workspaceId: tab.workspaceId });
        return;
      }
      if (
        this.restartFrozen ||
        (tab.securityClass === 'model-workspace'
          ? url !== tab.currentURL
          : Boolean(readOrynModelWorkspaceReference(url)))
      )
        event.preventDefault();
    };
    contents.on('will-navigate', guardDestination);
    contents.on('will-redirect', guardDestination);
    contents.on('will-attach-webview', (event) => event.preventDefault());
    if (tab.securityClass === 'model-workspace') {
      contents.on('will-navigate', (event, url) => {
        if (url !== tab.currentURL) event.preventDefault();
      });
    }
    contents.on('preload-error', (_event, preloadPath, error) => {
      console.error(`Website preload failed (${preloadPath}):`, error);
    });
  }

  private installStateTracking(tab: ManagedTab): void {
    const contents = tab.view.webContents;
    const isCurrent = () =>
      this.tabs.get(tab.tabId) === tab &&
      tab.view.webContents === contents &&
      !contents.isDestroyed();
    const publish = () => {
      if (isCurrent()) this.publishState();
    };
    contents.once('destroyed', () => this.removeClosedTab(tab));
    contents.on('media-started-playing', publish);
    contents.on('media-paused', publish);
    contents.on('did-start-loading', publish);
    contents.on('did-stop-loading', () => {
      if (!isCurrent()) return;
      publish();
      if (!isCurrent()) return;
      if (readOrynNetworkErrorURL(contents.getURL())) return;
      if (!tab.favicon && !tab.faviconPending) {
        try {
          const current = new URL(contents.getURL());
          if (['http:', 'https:'].includes(current.protocol)) {
            void this.updateFavicon(tab, [new URL('/favicon.ico', current).toString()]);
          }
        } catch {
          // Pages without a conventional favicon keep the trusted fallback icon.
        }
      }
      const documentId = this.callerContexts.getDocumentId(contents);
      if (tab.lastRecordedDocumentId !== documentId) {
        tab.lastRecordedDocumentId = documentId;
        tab.lastHistoryURL = contents.getURL();
        this.browserData.history.recordVisit(contents.getURL(), contents.getTitle());
      }
    });
    contents.on('did-navigate', (_event, url) => {
      if (!isCurrent()) return;
      if (readOrynNetworkErrorURL(url)) {
        publish();
        return;
      }
      tab.crashed = false;
      tab.failedNavigationURL = undefined;
      // Keep the restored image while its original page loads or reloads. A new
      // destination must not inherit an unrelated page's icon.
      if (url !== tab.currentURL) tab.favicon = undefined;
      tab.currentURL = url;
      contents.setZoomFactor(this.browserData.zoom.get(url));
      contents.setAudioMuted?.(
        this.browserData.sitePermissions?.get(new URL(url).origin, 'sound') === 'deny',
      );
      tab.faviconAbort?.abort();
      tab.faviconPending = false;
      tab.faviconRequestId += 1;
      publish();
    });
    contents.on('did-fail-load', (_event, code, description, validatedURL, isMainFrame) => {
      if (!isCurrent()) return;
      if (!isMainFrame || code === -3 || readOrynNetworkErrorURL(validatedURL)) return;
      if (consumeInsecureNavigationCancellation(contents, validatedURL, code)) {
        tab.failedNavigationURL = undefined;
        tab.currentURL = contents.getURL() || DEFAULT_START_URL;
        if (!contents.getURL()) void contents.loadURL(DEFAULT_START_URL).catch(() => undefined);
        publish();
        return;
      }
      try {
        const failedURL = new URL(validatedURL);
        if (!['http:', 'https:', 'file:'].includes(failedURL.protocol)) return;
        if (failedURL.toString() !== tab.currentURL) tab.favicon = undefined;
        tab.failedNavigationURL = failedURL.toString();
        tab.currentURL = failedURL.toString();
        publish();
        if (!isCurrent()) return;
        void contents
          .loadURL(
            createOrynNetworkErrorURL({
              url: failedURL.toString(),
              code,
              description,
            }),
          )
          .catch(() => undefined);
      } catch {
        // Chromium can report an empty or malformed URL for provisional failures.
      }
    });
    contents.on('did-navigate-in-page', (_event, url, isMainFrame) => {
      if (!isCurrent()) return;
      if (isMainFrame && tab.lastHistoryURL !== url) {
        tab.currentURL = url;
        tab.lastHistoryURL = url;
        this.browserData.history.recordVisit(url, contents.getTitle());
      }
      publish();
    });
    contents.on('page-title-updated', (_event, title) => {
      if (!isCurrent()) return;
      this.browserData.history.updateTitle(contents.getURL(), title);
      publish();
    });
    contents.on('page-favicon-updated', (_event, favicons) => {
      void this.updateFavicon(tab, favicons);
    });
    contents.on('render-process-gone', () => {
      if (!isCurrent()) return;
      tab.crashed = true;
      tab.failedNavigationURL = tab.currentURL;
      publish();
    });
  }

  private async updateFavicon(tab: ManagedTab, candidates: string[]): Promise<void> {
    const contents = tab.view.webContents as WebContents | undefined;
    if (!contents || contents.isDestroyed() || this.tabs.get(tab.tabId) !== tab) return;
    tab.faviconAbort?.abort();
    tab.faviconAbort = new AbortController();
    const requestId = ++tab.faviconRequestId;
    tab.faviconPending = true;
    let favicon: string | undefined;
    try {
      favicon = await resolveFaviconDataURL(contents, candidates, tab.faviconAbort.signal);
    } catch {
      // A tab may navigate or close while its browser-owned favicon request is in flight.
    }
    if (
      this.tabs.get(tab.tabId) !== tab ||
      requestId !== tab.faviconRequestId ||
      contents.isDestroyed()
    ) {
      return;
    }
    tab.faviconPending = false;
    if (!favicon) return;
    tab.favicon = favicon;
    this.publishState();
  }

  private snapshot(tab: ManagedTab): BrowserTab | undefined {
    const contents = tab.view.webContents;
    // The native view may lose its contents before our destroyed listener runs.
    if (!contents || contents.isDestroyed()) return undefined;
    return {
      tabId: tab.tabId,
      crashed: tab.crashed,
      audible: contents.isCurrentlyAudible?.() ?? false,
      muted: contents.isAudioMuted?.() ?? false,
      webContentsId: contents.id,
      url: tab.failedNavigationURL ?? (contents.getURL() || tab.currentURL),
      title: contents.getTitle(),
      favicon: tab.favicon,
      isLoading: contents.isLoading(),
      canGoBack: contents.navigationHistory.canGoBack(),
      canGoForward: contents.navigationHistory.canGoForward(),
      activeDocumentId: this.callerContexts.getDocumentId(contents),
      view: tab.view,
      workspaceId: tab.workspaceId,
      islandId: tab.islandId,
      pinned: tab.pinned,
    };
  }

  private requireActiveTab(): ManagedTab {
    if (!this.activeTabId) {
      throw new Error('No active browser tab.');
    }
    return this.requireTab(this.activeTabId);
  }

  private requireTab(tabId: string): ManagedTab {
    const tab = this.tabs.get(tabId);
    if (!tab) {
      throw new Error(`Unknown browser tab: ${tabId}`);
    }
    return tab;
  }

  private publishState(): void {
    if (this.batching || this.destroying) return;
    const snapshots = this.getTabs();
    // Let close cleanup choose the successor before notifying active-tab consumers.
    if (!snapshots.some((tab) => tab.tabId === this.activeTabId)) return;
    const activeIndex = Math.max(
      0,
      snapshots.findIndex((tab) => tab.tabId === this.activeTabId),
    );
    this.browserData.sessions.saveOpenTabs(
      snapshots.map((tab) => ({
        url: tab.url,
        tabId: tab.tabId,
        title: tab.title,
        favicon: tab.favicon,
        workspaceId: tab.workspaceId,
        islandId: tab.islandId,
        pinned: tab.pinned,
      })),
      activeIndex,
    );
    this.emit('state-changed');
  }

  private visibleTabIds(
    workspaceId = this.browserData.experience.read().activeWorkspaceId,
  ): string[] {
    return [...this.tabs.values()]
      .filter((tab) => tab.workspaceId === workspaceId)
      .sort((left, right) => {
        if (left.pinned || right.pinned) return Number(right.pinned) - Number(left.pinned);
        const islands = this.browserData.experience
          .read()
          .islands.filter((island) => island.workspaceId === workspaceId)
          .sort((a, b) => a.order - b.order);
        const rank = (tab: ManagedTab) => {
          const index = islands.findIndex((island) => island.islandId === tab.islandId);
          return index < 0 ? islands.length : index;
        };
        return rank(left) - rank(right);
      })
      .map((tab) => tab.tabId);
  }
}

export function canonicalTabURL(value: string): string {
  try {
    return new URL(value).toString();
  } catch {
    return value;
  }
}

export function normalizeNavigationInput(
  input: string,
  provider: 'google' | 'duckduckgo' | 'bing' = 'google',
): string {
  const search = (query: string) => searchNavigationInput(query, provider);
  const value = input.trim();
  if (!value) {
    return DEFAULT_START_URL;
  }
  if (isAbsolute(value)) return pathToFileURL(value).toString();
  if (/^file:/iu.test(value)) {
    const file = new URL(value);
    if (file.hostname && file.hostname !== 'localhost') throw new Error('REMOTE_FILE_URL_DENIED');
    return file.toString();
  }

  const isLoopbackAddress = /^(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:[/?#].*)?$/iu.test(
    value,
  );
  const isHostname =
    /^(?:(?:[\p{L}\p{N}_-]+\.)+[\p{L}\p{N}_-]+|\[[0-9a-f:]+\])(?::\d+)?(?:[/?#].*)?$/iu.test(value);
  try {
    if (isLoopbackAddress) return new URL(`http://${value}`).toString();
    if (isHostname) return new URL(`https://${value}`).toString();
    if (/^(https?:|oryn:|about:blank$)/iu.test(value)) return new URL(value).toString();
    return search(value);
  } catch {
    return search(value);
  }
}

export function searchNavigationInput(
  query: string,
  provider: 'google' | 'duckduckgo' | 'bing' = 'google',
): string {
  return `${provider === 'duckduckgo' ? 'https://duckduckgo.com/?q=' : provider === 'bing' ? 'https://www.bing.com/search?q=' : 'https://www.google.com/search?q='}${encodeURIComponent(query)}`;
}
