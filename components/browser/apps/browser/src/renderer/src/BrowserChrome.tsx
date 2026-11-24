import { modelIdentityKey } from '../../shared/model-identity';
import {
  BrowserDialog,
  requestCredentials,
  requestText,
  requestChoice,
  useBrowserDialogOpen,
} from './components/BrowserDialog';
import { RegistrySettings } from './components/RegistrySettings';
import { EngineVersions } from './components/EngineVersions';
import { AddressBar } from './components/AddressBar';
import { BrowserImportSettings } from './components/BrowserImportSettings';
import { ModelAlternatives } from './components/ModelAlternatives';
import { TabStrip } from './components/TabStrip';
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from 'react';

import type {
  AIActivitySettings,
  AIActivitySnapshot,
  AppUpdateSnapshot,
  BrowserCoreDataSnapshot,
  BrowserWindowState,
  HuggingFaceCredentialStatus,
  HuggingFacePageActionState,
  LocalAIJobActivity,
  LocalAIPermissionPrompt,
  LocalAIPermissionPromptResponse,
  ModelManagerSnapshot,
  OrynSettingsSnapshot,
  TrustedInstalledModel,
  TrustedModelRemovalPreview,
  TrustedModelOperation,
} from '../../shared/ipc';
import { withRequestTimeout } from './request-timeout';
import { ModelSubstitutes } from './components/ModelSubstitutes';
import { TAB_SEARCH_SHORTCUT } from '../../shared/ipc';
import { isOrynNewTabURL } from '../../shared/browser-urls';
import { resolveBrowserChromeLayout } from '../../shared/chrome-layout';
import { formatStorageSize as formatBytes } from '../../shared/format-storage-size';
import { TAB_ISLAND_COLORS } from '../../shared/tab-island-colors';
import {
  Badge,
  Button,
  EmptyState as UIEmptyState,
  Icon,
  IconButton,
  MenuItem,
  ProgressBar,
  SegmentedTabs,
  Sheet,
  Stat,
  type IconName,
} from './components/ui';
import { nextWorkspaceName } from './sidebar-presentation';
import { ModelMemoryRequirements } from './components/ModelMemoryRequirements';

type Panel = 'history' | 'bookmarks' | 'downloads' | 'site' | 'settings' | 'local-ai' | null;

type TabMenuState = {
  tabId: string;
  x: number;
  y: number;
};

type TabPointerDrag = {
  tabId: string;
  pointerId: number;
  startX: number;
  startY: number;
  dragging: boolean;
};

const initialState: BrowserWindowState = {
  activeTabId: '',
  tabs: [],
  engineState: 'starting',
  zoomFactor: 1,
  memoryPressure: 'normal',
  localAIResources: undefined,
  experience: {
    activeWorkspaceId: 'workspace_default',
    workspaces: [],
    islands: [],
    recentlyClosed: [],
    sidebar: {
      visible: true,
      width: 52,
      panelWidth: 390,
      panelMode: 'overlay',
      activeAppId: null,
      tools: [],
      apps: [],
    },
    tabSearchShortcut: TAB_SEARCH_SHORTCUT,
    autoGroupOpenerTabs: true,
  },
};

const emptyData: BrowserCoreDataSnapshot = {
  history: [],
  bookmarks: [],
  downloads: [],
  sitePermissions: [],
  bookmarkBarVisible: true,
  appearance: { theme: 'system', density: 'default', onboardingComplete: true },
};

const emptyAIActivity: AIActivitySnapshot = {
  settings: { keepActivity: true, contentPolicy: 'metadata-only', retentionDays: 30 },
  records: [],
};

const emptyModelManager: ModelManagerSnapshot = {
  totalLogicalBytes: 0,
  totalReclaimableBytes: 0,
  models: [],
  runtimePacks: [],
  runtimePackPolicy: {
    catalogVersion: 0,
    catalogDigest: '',
    thirdPartyPublishersEnabled: false,
    automaticExecutableUpdatesEnabled: false,
  },
};

const emptyOrynSettings: OrynSettingsSnapshot = {
  preferences: {
    preferInstalled: true,
    optimization: 'balanced',
    globalModels: [],
    globalFamilies: [],
    siteChoices: [],
    availableModels: [],
  },
  compute: {
    memoryBudgetBytes: null,
    backgroundJobs: 'allow',
    batteryMode: 'normal',
    generationConcurrency: 1,
  },
  sites: [],
  storage: {
    modelBytes: 0,
    modelReclaimableBytes: 0,
    modelSharedBytes: 0,
    artifactBytes: 0,
    temporaryArtifactBytes: 0,
    readyArtifacts: 0,
    reclaimableArtifacts: 0,
  },
  developer: {
    showDiagnostics: false,
    showResolutionDetails: false,
    showProtocolDiagnostics: false,
    experimentalRunnersAllowed: false,
  },
};

const emptyAppUpdate: AppUpdateSnapshot = {
  status: 'unavailable',
  enabled: false,
  currentVersion: '',
  channel: 'stable',
  automaticChecks: true,
  unavailableReason: 'Updates are unavailable in this build.',
};

export function BrowserChrome(): React.JSX.Element {
  const browserDialogOpen = useBrowserDialogOpen();
  const [state, setState] = useState(initialState);
  const privateModeRef = useRef<boolean | undefined>(undefined);
  const [data, setData] = useState(emptyData);
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [panel, setPanel] = useState<Panel>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [historyQuery, setHistoryQuery] = useState('');
  const [permissionPrompt, setPermissionPrompt] = useState<LocalAIPermissionPrompt | null>(null);
  const [chooserOpen, setChooserOpen] = useState(false);
  const [selectedModelKey, setSelectedModelKey] = useState('');
  const [rememberSiteChoice, setRememberSiteChoice] = useState(false);
  const [jobActivity, setJobActivity] = useState<LocalAIJobActivity>({
    running: 0,
    queued: 0,
    jobs: [],
  });
  const [modelOperations, setModelOperations] = useState<TrustedModelOperation[]>([]);
  const [aiActivity, setAIActivity] = useState<AIActivitySnapshot>(emptyAIActivity);
  const [appUpdate, setAppUpdate] = useState<AppUpdateSnapshot>(emptyAppUpdate);
  const [dismissedReadyVersion, setDismissedReadyVersion] = useState<string | null>(null);
  const [pageActionConfirmation, setPageActionConfirmation] = useState(false);
  const [persistentChromeHeight, setPersistentChromeHeight] = useState(112);
  const [tabSearchOpen, setTabSearchOpen] = useState(false);
  const [tabSearchQuery, setTabSearchQuery] = useState('');
  const [tabMenu, setTabMenu] = useState<TabMenuState | null>(null);
  const [draggedTabId, setDraggedTabId] = useState<string | null>(null);
  const [dragOverIslandId, setDragOverIslandId] = useState<string | null>(null);
  const [ungroupDropActive, setUngroupDropActive] = useState(false);
  const [islandColorPickerId, setIslandColorPickerId] = useState<string | null>(null);
  const [workspaceMenuOpen, setWorkspaceMenuOpen] = useState(false);
  const onboardingDialogRef = useRef<HTMLElement>(null);
  const pageActionDialogRef = useRef<HTMLElement>(null);
  const islandColorDialogRef = useRef<HTMLElement>(null);
  const appUpdateDialogRef = useRef<HTMLElement>(null);
  const omniboxRef = useRef<HTMLInputElement>(null);
  const persistentChromeRef = useRef<HTMLDivElement>(null);
  const tabPointerDragRef = useRef<TabPointerDrag | null>(null);
  const suppressTabActivationRef = useRef<string | null>(null);
  const chromeCaptureRef = useRef(false);
  const showAppUpdateDialog =
    appUpdate.status === 'available' ||
    (appUpdate.status === 'ready' && appUpdate.candidate?.version !== dismissedReadyVersion);
  chromeCaptureRef.current =
    browserDialogOpen ||
    Boolean(state.tabs.find((tab) => tab.tabId === state.activeTabId)?.crashed) ||
    menuOpen ||
    Boolean(panel) ||
    !data.appearance.onboardingComplete ||
    Boolean(permissionPrompt) ||
    pageActionConfirmation ||
    tabSearchOpen ||
    Boolean(tabMenu) ||
    Boolean(islandColorPickerId) ||
    workspaceMenuOpen ||
    suggestionsOpen ||
    showAppUpdateDialog;
  const activeTab = useMemo(
    () => state.tabs.find((tab) => tab.tabId === state.activeTabId),
    [state],
  );
  const rootBookmarks = data.bookmarks.filter(
    (bookmark) =>
      bookmark.kind === 'bookmark' &&
      bookmark.parentId === null &&
      bookmark.favorite &&
      !bookmark.trashedAt,
  );
  const activeBookmark = data.bookmarks.find(
    (bookmark) =>
      bookmark.kind === 'bookmark' && !bookmark.trashedAt && bookmark.url === activeTab?.url,
  );
  const visibleTabs = useMemo(
    () => state.tabs.filter((tab) => tab.workspaceId === state.experience.activeWorkspaceId),
    [state.tabs, state.experience.activeWorkspaceId],
  );
  const visibleIslands = useMemo(
    () =>
      state.experience.islands
        .filter((island) => island.workspaceId === state.experience.activeWorkspaceId)
        .sort((left, right) => left.order - right.order)
        .map((island) => ({
          ...island,
          tabs: visibleTabs.filter((tab) => !tab.pinned && tab.islandId === island.islandId),
        }))
        .filter((island) => island.tabs.length > 0),
    [state.experience, visibleTabs],
  );
  const pinnedTabs = visibleTabs.filter((tab) => tab.pinned);
  const standaloneTabs = visibleTabs.filter((tab) => !tab.pinned && !tab.islandId);
  const expandedTabCount = visibleIslands.reduce(
    (count, island) => count + (island.collapsed ? 0 : island.tabs.length),
    standaloneTabs.length,
  );
  const visibleModelOperations = modelOperations.filter(isVisibleModelOperation);
  const activeCollapsedIslandId = visibleIslands.find(
    (island) => island.collapsed && island.islandId === activeTab?.islandId,
  )?.islandId;
  const engineStatusLabel = state.privateMode
    ? 'Private'
    : [
        jobActivity.running > 0 || jobActivity.queued > 0
          ? `Oryn ● ${jobActivity.running} running · ${jobActivity.queued} queued`
          : visibleModelOperations.length > 0
            ? `Local AI · ${visibleModelOperations.length} model ${visibleModelOperations.length === 1 ? 'operation' : 'operations'}`
            : 'Local AI',
        state.engineState === 'ready' ? '' : state.engineState,
        state.memoryPressure === 'normal' ? '' : `${state.memoryPressure} memory pressure`,
      ]
        .filter(Boolean)
        .join(' · ');
  const openTabMenu = (tabId: string, event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    setTabMenu({ tabId, x: event.clientX, y: event.clientY });
  };
  const tabDropTargetAtPoint = (x: number, y: number) => {
    const element = document.elementFromPoint(x, y);
    return {
      tabId: element?.closest<HTMLElement>('[data-tab-id]')?.dataset.tabId ?? null,
      islandId: element?.closest<HTMLElement>('[data-island-id]')?.dataset.islandId ?? null,
      inTabStrip: Boolean(element?.closest('.tab-strip')),
    };
  };
  const startTabDrag = (tabId: string, event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0) return;
    setTabMenu(null);
    tabPointerDragRef.current = {
      tabId,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      dragging: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const moveTabDrag = (event: ReactPointerEvent<HTMLElement>) => {
    const drag = tabPointerDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!drag.dragging && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 6)
      return;
    if (!drag.dragging) {
      drag.dragging = true;
      setDraggedTabId(drag.tabId);
    }
    event.preventDefault();
    const target = tabDropTargetAtPoint(event.clientX, event.clientY);
    const sourceTab = state.tabs.find((tab) => tab.tabId === drag.tabId);
    setDragOverIslandId(target.islandId);
    setUngroupDropActive(Boolean(sourceTab?.islandId && target.inTabStrip && !target.islandId));
  };
  const finishTabDrag = (event: ReactPointerEvent<HTMLElement>) => {
    const drag = tabPointerDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    tabPointerDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (drag.dragging) {
      event.preventDefault();
      suppressTabActivationRef.current = drag.tabId;
      window.setTimeout(() => {
        if (suppressTabActivationRef.current === drag.tabId)
          suppressTabActivationRef.current = null;
      }, 0);
      const target = tabDropTargetAtPoint(event.clientX, event.clientY);
      const sourceTab = state.tabs.find((tab) => tab.tabId === drag.tabId);
      if (target.tabId && target.tabId !== drag.tabId) {
        void window.browser.updateBrowserExperience({
          type: 'tab-reorder',
          tabId: drag.tabId,
          beforeTabId: target.tabId,
        });
      } else if (target.islandId && sourceTab?.islandId !== target.islandId) {
        void window.browser.updateBrowserExperience({
          type: 'tab-move-island',
          tabId: drag.tabId,
          islandId: target.islandId,
        });
      } else if (target.inTabStrip && sourceTab?.islandId && !target.islandId) {
        void window.browser.updateBrowserExperience({ type: 'tab-move-island', tabId: drag.tabId });
      }
    } else {
      // Pointer capture keeps drag tracking reliable. Activate at pointer release so
      // the primary-pointer path does not depend on a later synthesized click; the
      // button onClick remains for keyboard and synthetic activation.
      activateTab(drag.tabId);
    }
    setDraggedTabId(null);
    setDragOverIslandId(null);
    setUngroupDropActive(false);
  };
  const cancelTabDrag = (event: ReactPointerEvent<HTMLElement>) => {
    if (tabPointerDragRef.current?.pointerId !== event.pointerId) return;
    tabPointerDragRef.current = null;
    setDraggedTabId(null);
    setDragOverIslandId(null);
    setUngroupDropActive(false);
  };
  const activateTab = (tabId: string) => {
    if (suppressTabActivationRef.current === tabId) {
      suppressTabActivationRef.current = null;
      return;
    }
    setWorkspaceMenuOpen(false);
    setTabMenu(null);
    void window.browser.activateTab(tabId);
  };
  useFocusTrap(!data.appearance.onboardingComplete, onboardingDialogRef);
  useFocusTrap(pageActionConfirmation, pageActionDialogRef, () => setPageActionConfirmation(false));
  useFocusTrap(Boolean(islandColorPickerId), islandColorDialogRef, () =>
    setIslandColorPickerId(null),
  );
  useFocusTrap(showAppUpdateDialog, appUpdateDialogRef);

  const refreshData = () => window.browser.getBrowserData().then(setData);

  useEffect(() => {
    void window.browser.getState().then((next) => {
      privateModeRef.current = next.privateMode === true;
      setState(next);
      if (!next.privateMode) {
        void window.browser.getLocalAIJobActivity().then(setJobActivity);
        void window.browser.getAIActivity().then(setAIActivity);
      }
    });
    void refreshData();
    void window.browser.getAppUpdate().then(setAppUpdate);
    const stopAuthentication = window.browser.onAuthenticationRequest((request) => {
      void requestCredentials(request.target)
        .then((credentials) => window.browser.respondAuthentication(request.id, credentials))
        .catch(() => undefined);
    });
    const stopStateListener = window.browser.onStateChanged((next) => {
      privateModeRef.current = next.privateMode === true;
      setState(next);
    });
    const stopDataListener = window.browser.onBrowserDataChanged(() => {
      void refreshData();
    });
    const stopFocusListener = window.browser.onFocusOmnibox(() => {
      omniboxRef.current?.focus();
      omniboxRef.current?.select();
    });
    const stopTabSearchListener = window.browser.onOpenTabSearch(() => {
      setPanel(null);
      setTabSearchOpen(true);
      setMenuOpen(false);
    });
    const stopPanelListener = window.browser.onOpenPanel((nextPanel) => {
      if (privateModeRef.current && nextPanel === 'local-ai') return;
      setPanel((current) => (nextPanel !== null && current === nextPanel ? null : nextPanel));
      setTabSearchOpen(false);
      setMenuOpen(false);
    });
    const stopPermissionListener = window.browser.onLocalAIPermissionPrompt((prompt) => {
      if (privateModeRef.current !== false) return;
      setPermissionPrompt(prompt);
      setSelectedModelKey(modelIdentityKey(prompt.resolvedModel));
      setChooserOpen(false);
      setRememberSiteChoice(false);
    });
    const stopPermissionCancellationListener = window.browser.onLocalAIPermissionPromptCancelled(
      (promptId) => {
        setPermissionPrompt((current) => (current?.promptId === promptId ? null : current));
      },
    );
    const stopJobActivityListener = window.browser.onLocalAIJobActivityChanged((activity) => {
      if (privateModeRef.current !== false) return;
      setJobActivity(activity);
      void window.browser.getAIActivity().then(setAIActivity);
    });
    const stopModelOperations = window.browser.onModelOperationChanged((operation) => {
      if (privateModeRef.current !== false) return;
      setModelOperations((current) => mergeModelOperation(current, operation));
    });
    const stopAppUpdates = window.browser.onAppUpdateStateChanged(setAppUpdate);
    return () => {
      stopAuthentication();
      stopStateListener();
      stopDataListener();
      stopFocusListener();
      stopTabSearchListener();
      stopPanelListener();
      stopPermissionListener();
      stopPermissionCancellationListener();
      stopJobActivityListener();
      stopModelOperations();
      stopAppUpdates();
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = data.appearance.theme;
    if (data.appearance.density === 'default') {
      delete document.documentElement.dataset.density;
    } else {
      document.documentElement.dataset.density = data.appearance.density;
    }
  }, [data.appearance]);

  useEffect(() => {
    void refreshData();
    setPageActionConfirmation(false);
  }, [state.activeTabId, activeTab?.url]);

  useLayoutEffect(() => {
    const measure = () => {
      const contentHeight = Math.min(
        160,
        Math.max(80, Math.ceil(persistentChromeRef.current?.getBoundingClientRect().height ?? 0)),
      );
      const overlayBottom = Math.max(
        contentHeight,
        ...Array.from(document.querySelectorAll<HTMLElement>('[data-trusted-overlay="true"]')).map(
          (element) => Math.ceil(element.getBoundingClientRect().bottom),
        ),
      );
      setPersistentChromeHeight((current) => (current === contentHeight ? current : contentHeight));
      void window.browser.setChromeLayout(
        resolveBrowserChromeLayout({
          contentHeight,
          overlayBottom,
          captureWindow: chromeCaptureRef.current,
          sidebarInteractive:
            Boolean(panel) &&
            !(
              browserDialogOpen ||
              menuOpen ||
              permissionPrompt ||
              pageActionConfirmation ||
              tabSearchOpen ||
              tabMenu ||
              islandColorPickerId ||
              workspaceMenuOpen ||
              suggestionsOpen ||
              showAppUpdateDialog ||
              !data.appearance.onboardingComplete
            ),
        }),
      );
    };
    // Native overlay bounds must be published during the commit. Occluded views
    // can pause animation/resize callbacks until after those bounds expand.
    measure();
    const observer = new ResizeObserver(measure);
    if (persistentChromeRef.current) observer.observe(persistentChromeRef.current);
    for (const element of document.querySelectorAll<HTMLElement>('[data-trusted-overlay="true"]')) {
      observer.observe(element);
    }
    const frame = requestAnimationFrame(measure);
    window.addEventListener('resize', measure);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', measure);
      observer.disconnect();
    };
  }, [
    browserDialogOpen,
    activeTab?.crashed,
    data.appearance.density,
    data.appearance.onboardingComplete,
    data.bookmarkBarVisible,
    menuOpen,
    panel,
    permissionPrompt,
    pageActionConfirmation,
    tabSearchOpen,
    tabMenu,
    islandColorPickerId,
    workspaceMenuOpen,
    suggestionsOpen,
    appUpdate.status,
    showAppUpdateDialog,
  ]);

  useEffect(() => {
    if (panel) {
      void refreshData();
      if (panel === 'local-ai') void window.browser.getAIActivity().then(setAIActivity);
    }
  }, [panel]);

  useEffect(() => {
    if (!menuOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMenuOpen(false);
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [menuOpen]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      const primaryModifier = isMacOS() ? event.metaKey : event.ctrlKey;
      if (primaryModifier && event.shiftKey && event.code === 'KeyA') {
        event.preventDefault();
        setTabSearchOpen(true);
      }
      if (event.key === 'Escape') {
        setTabMenu(null);
        setIslandColorPickerId(null);
        setWorkspaceMenuOpen(false);
        setTabSearchOpen(false);
      }
    };
    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, []);

  const selectPanel = (next: Exclude<Panel, null>) => {
    if (state.privateMode && next === 'local-ai') return;
    setPanel((current) => (current === next ? null : next));
    setMenuOpen(false);
  };

  const toggleBookmark = async () => {
    if (!activeTab?.url) return;
    if (activeBookmark) {
      await window.browser.deleteBookmark(activeBookmark.bookmarkId);
    } else {
      await window.browser.addBookmark(activeTab.title || activeTab.url, activeTab.url);
    }
    await refreshData();
  };

  return (
    <header
      className={`browser-chrome${state.privateMode ? ' private-mode' : ''}`}
      aria-label="Browser controls"
      style={
        {
          '--browser-content-height': `${persistentChromeHeight}px`,
        } as CSSProperties
      }
    >
      <BrowserDialog />
      {activeTab?.crashed && (
        <section className="browser-input-dialog" role="alert" data-trusted-overlay="true">
          <h2>This page stopped responding</h2>
          <p>{activeTab.url}</p>
          <p>Reloading starts a new document. Unsaved page changes may be lost.</p>
          <button onClick={() => void window.browser.reload()}>Reload page</button>
        </section>
      )}
      {data.recoveryNotices?.length ? (
        <p role="status">
          Recovered damaged browser data: {data.recoveryNotices.join(', ')}. Diagnostic copies were
          preserved.
        </p>
      ) : null}
      <div ref={persistentChromeRef} className="persistent-chrome">
        <div className="tab-row">
          <button
            type="button"
            className="workspace-chip"
            aria-label="Workspaces"
            disabled={state.privateMode}
            aria-haspopup="menu"
            aria-expanded={workspaceMenuOpen}
            onClick={() => setWorkspaceMenuOpen((open) => !open)}
          >
            <span
              className="workspace-dot"
              style={{
                background:
                  state.experience.workspaces.find(
                    (item) => item.workspaceId === state.experience.activeWorkspaceId,
                  )?.accent ?? '#8b8cff',
              }}
            />
            {state.experience.workspaces.find(
              (item) => item.workspaceId === state.experience.activeWorkspaceId,
            )?.name ?? 'Main'}
          </button>
          {workspaceMenuOpen && (
            <>
              <PopoverDismissLayer onDismiss={() => setWorkspaceMenuOpen(false)} />
              <WorkspaceMenu state={state} close={() => setWorkspaceMenuOpen(false)} />
            </>
          )}
          <TabStrip
            activeTabId={state.activeTabId}
            activeCollapsedIslandId={activeCollapsedIslandId}
            ungroupDropActive={ungroupDropActive}
            unpinnedTabCount={expandedTabCount}
            pinnedTabCount={pinnedTabs.length}
            islandCount={visibleIslands.length}
          >
            {pinnedTabs.map((tab) => (
              <div
                key={tab.tabId}
                className={`tab pinned ${tab.tabId === state.activeTabId ? 'active' : ''}`}
                onContextMenu={(event) => openTabMenu(tab.tabId, event)}
              >
                <button
                  className="tab-activate"
                  role="tab"
                  aria-label={tabAccessibleName(tab)}
                  aria-selected={tab.tabId === state.activeTabId}
                  title={tab.title || tab.url}
                  onPointerDown={(event) => startTabDrag(tab.tabId, event)}
                  onPointerMove={moveTabDrag}
                  onPointerUp={finishTabDrag}
                  onPointerCancel={cancelTabDrag}
                  onClick={() => activateTab(tab.tabId)}
                >
                  <TabFavicon tab={tab} />
                  <span className="tab-title">
                    {tab.title || 'New tab'}
                    {tab.audible ? ' · Audio' : ''}
                    {tab.muted ? ' · Muted' : ''}
                  </span>
                </button>
                <IconButton
                  className="tab-close"
                  icon="close"
                  label={`Close ${tab.title || 'tab'}`}
                  onClick={() => void window.browser.closeTab(tab.tabId)}
                />
              </div>
            ))}
            {visibleIslands.map((island) => {
              const islandTabs = island.tabs;
              return (
                <div
                  className={`tab-island ${island.collapsed ? 'collapsed' : ''} ${dragOverIslandId === island.islandId ? 'drag-over' : ''}`}
                  key={island.islandId}
                  data-island-id={island.islandId}
                  data-active-island={activeCollapsedIslandId === island.islandId || undefined}
                  style={
                    {
                      '--island-accent': island.accent,
                      '--island-tab-count': islandTabs.length,
                    } as CSSProperties
                  }
                >
                  <button
                    type="button"
                    className="tab-island-label"
                    aria-expanded={!island.collapsed}
                    aria-haspopup="dialog"
                    title={`${island.collapsed ? 'Expand' : 'Collapse'} ${island.name} · Right-click to change color`}
                    onContextMenu={(event) => {
                      event.preventDefault();
                      setTabMenu(null);
                      setIslandColorPickerId(island.islandId);
                    }}
                    onClick={() =>
                      void window.browser.updateBrowserExperience({
                        type: 'island-update',
                        islandId: island.islandId,
                        collapsed: !island.collapsed,
                      })
                    }
                  >
                    <span>{island.name}</span>
                    <small>{islandTabs.length}</small>
                  </button>
                  <button
                    type="button"
                    className="tab-island-more"
                    aria-label={`Manage ${island.name} island`}
                    title="Rename, move, or dissolve island"
                    onClick={async () => {
                      const action = await requestChoice(
                        'Manage island',
                        [
                          {
                            label: 'Rename',
                            value: 'rename',
                            description: 'Give this group of tabs a new name.',
                            icon: 'edit',
                          },
                          {
                            label: 'Move',
                            value: 'move',
                            description: state.experience.workspaces.some(
                              (workspace) => workspace.workspaceId !== island.workspaceId,
                            )
                              ? 'Take this island to another workspace.'
                              : 'Create another workspace to move this island.',
                            icon: 'arrow-right',
                            disabled: !state.experience.workspaces.some(
                              (workspace) => workspace.workspaceId !== island.workspaceId,
                            ),
                          },
                          {
                            label: 'Dissolve',
                            value: 'dissolve',
                            description: 'Ungroup the tabs. All tabs stay open.',
                            icon: 'ungroup',
                            tone: 'danger',
                          },
                        ],
                        {
                          description: `${island.name} · ${islandTabs.length} ${islandTabs.length === 1 ? 'tab' : 'tabs'}`,
                          icon: 'models',
                        },
                      );
                      if (action === 'rename') {
                        const name = await requestText('Island name', island.name);
                        if (name)
                          void window.browser.updateBrowserExperience({
                            type: 'island-update',
                            islandId: island.islandId,
                            name,
                          });
                      }
                      if (action === 'move') {
                        const destination = await requestChoice(
                          'Move island to workspace',
                          state.experience.workspaces
                            .filter((workspace) => workspace.workspaceId !== island.workspaceId)
                            .map((workspace) => ({
                              label: workspace.name,
                              value: workspace.workspaceId,
                            })),
                        );
                        const workspace = state.experience.workspaces.find(
                          (item) => item.workspaceId === destination,
                        );
                        if (workspace)
                          void window.browser.updateBrowserExperience({
                            type: 'island-move-workspace',
                            islandId: island.islandId,
                            workspaceId: workspace.workspaceId,
                          });
                      }
                      if (action === 'dissolve')
                        void window.browser.updateBrowserExperience({
                          type: 'island-delete',
                          islandId: island.islandId,
                        });
                    }}
                  >
                    •••
                  </button>
                  {!island.collapsed &&
                    islandTabs.map((tab) => (
                      <TabButton
                        key={tab.tabId}
                        tab={tab}
                        active={tab.tabId === state.activeTabId}
                        dragging={draggedTabId === tab.tabId}
                        openMenu={(event) => openTabMenu(tab.tabId, event)}
                        startDrag={(event) => startTabDrag(tab.tabId, event)}
                        moveDrag={moveTabDrag}
                        finishDrag={finishTabDrag}
                        cancelDrag={cancelTabDrag}
                        activate={() => activateTab(tab.tabId)}
                      />
                    ))}
                </div>
              );
            })}
            {standaloneTabs.map((tab) => (
              <TabButton
                key={tab.tabId}
                tab={tab}
                active={tab.tabId === state.activeTabId}
                dragging={draggedTabId === tab.tabId}
                openMenu={(event) => openTabMenu(tab.tabId, event)}
                startDrag={(event) => startTabDrag(tab.tabId, event)}
                moveDrag={moveTabDrag}
                finishDrag={finishTabDrag}
                cancelDrag={cancelTabDrag}
                activate={() => activateTab(tab.tabId)}
              />
            ))}
          </TabStrip>
          <div className="tab-row-actions">
            <IconButton
              className="new-tab-button"
              icon="plus"
              label="New tab"
              onClick={() => void window.browser.newTab()}
            />
            <IconButton
              className="tab-search-button"
              icon="search"
              label="Search tabs"
              onClick={() => setTabSearchOpen(true)}
            />
          </div>
          <button
            type="button"
            className={`engine-status ${state.privateMode ? 'private' : `${state.engineState} pressure-${state.memoryPressure}`}`}
            aria-live="polite"
            aria-label={state.privateMode ? 'Private browsing' : 'Local AI activity'}
            aria-expanded={panel === 'local-ai'}
            title={engineStatusLabel}
            onClick={() => {
              if (!state.privateMode) selectPanel('local-ai');
            }}
          >
            {!state.privateMode && <span className="status-dot" aria-hidden="true" />}
            <Icon name="sparkles" />
            <span className="engine-status-label">{engineStatusLabel}</span>
          </button>
        </div>

        <div className="toolbar">
          <IconButton
            className="toolbar-button"
            icon="arrow-left"
            label="Back"
            disabled={!activeTab?.canGoBack}
            onClick={() => void window.browser.back()}
          />
          <IconButton
            className="toolbar-button"
            icon="arrow-right"
            label="Forward"
            disabled={!activeTab?.canGoForward}
            onClick={() => void window.browser.forward()}
          />
          <IconButton
            className="toolbar-button"
            icon={activeTab?.isLoading ? 'close' : 'refresh'}
            label={activeTab?.isLoading ? 'Stop' : 'Reload'}
            onClick={() =>
              void (activeTab?.isLoading ? window.browser.stop() : window.browser.reload())
            }
          />
          <AddressBar
            activeTabId={state.activeTabId}
            currentURL={activeTab?.url ?? ''}
            inputRef={omniboxRef}
            suggestionsEnabled={data.appearance.localSuggestions ?? true}
            tabs={state.tabs}
            bookmarks={data.bookmarks}
            history={data.history}
            onNavigate={(value) => void window.browser.navigate(value)}
            onSuggestionsOpenChange={setSuggestionsOpen}
            leading={
              <>
                <IconButton
                  type="button"
                  className="site-control"
                  icon="globe"
                  label="Site controls"
                  onClick={() => selectPanel('site')}
                />
                <span className="connection-label" title={activeTab?.connection?.reason}>
                  {connectionStateLabel(activeTab?.connection?.state)}
                </span>
                {Boolean(activeTab?.captureAccess?.length) && (
                  <button
                    type="button"
                    className="capture-access-indicator"
                    aria-label={`This page can use ${activeTab!.captureAccess!.join(', ')}`}
                    onClick={() => selectPanel('site')}
                  >
                    ●
                  </button>
                )}
              </>
            }
            trailing={
              <IconButton
                type="button"
                className="bookmark-button"
                icon="star"
                label={activeBookmark ? 'Remove bookmark' : 'Bookmark this page'}
                aria-pressed={Boolean(activeBookmark)}
                onClick={() => void toggleBookmark()}
              />
            }
          />
          <IconButton
            className="toolbar-button"
            icon="download"
            label="Downloads"
            onClick={() => selectPanel('downloads')}
          />
          {state.huggingFacePageAction && (
            <Button
              className={`hf-page-action hf-${state.huggingFacePageAction.status}`}
              size="sm"
              variant={
                ['loadable', 'installed'].includes(state.huggingFacePageAction.status)
                  ? 'primary'
                  : 'secondary'
              }
              disabled={state.huggingFacePageAction.status === 'inspecting'}
              onClick={() => {
                const action = state.huggingFacePageAction;
                if (!action) return;
                if (['loadable', 'unverified'].includes(action.status))
                  setPageActionConfirmation(true);
                if (action.status === 'installed')
                  void window.browser.openHuggingFaceModelWorkspace();
                if (
                  [
                    'downloading',
                    'access-required',
                    'unsupported',
                    'blocked',
                    'resource-error',
                    'failed',
                  ].includes(action.status)
                ) {
                  void window.browser.showHuggingFacePageActionDetails();
                }
              }}
              title={state.huggingFacePageAction.message}
            >
              <Icon
                name={state.huggingFacePageAction.status === 'installed' ? 'sparkles' : 'download'}
              />
              {huggingFacePageActionLabel(state.huggingFacePageAction)}
            </Button>
          )}
          <IconButton
            className="toolbar-button"
            icon="menu"
            label="Browser menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          />
          {menuOpen && (
            <>
              <PopoverDismissLayer onDismiss={() => setMenuOpen(false)} />
              <BrowserMenu
                state={state}
                onClose={() => setMenuOpen(false)}
                onSelect={selectPanel}
                onFind={() => {
                  setMenuOpen(false);
                  void window.browser.showFindInPage();
                }}
                onCheckUpdates={() => {
                  setPanel('settings');
                  setMenuOpen(false);
                  void window.browser.checkForAppUpdate().then(setAppUpdate);
                }}
              />
            </>
          )}
        </div>

        {data.bookmarkBarVisible && (
          <nav className="bookmarks-bar" aria-label="Bookmarks bar">
            {rootBookmarks.map(
              (bookmark) =>
                bookmark.kind === 'bookmark' && (
                  <button
                    key={bookmark.bookmarkId}
                    title={bookmark.url}
                    onClick={() => void window.browser.navigate(bookmark.url)}
                  >
                    {bookmark.title}
                  </button>
                ),
            )}
            {rootBookmarks.length === 0 && <span>Bookmark pages from the address bar</span>}
          </nav>
        )}
      </div>

      {tabMenu && (
        <>
          <PopoverDismissLayer onDismiss={() => setTabMenu(null)} />
          <TabContextMenu menu={tabMenu} state={state} close={() => setTabMenu(null)} />
        </>
      )}

      {islandColorPickerId && (
        <IslandColorDialog
          ref={islandColorDialogRef}
          island={state.experience.islands.find((item) => item.islandId === islandColorPickerId)}
          close={() => setIslandColorPickerId(null)}
        />
      )}

      {tabSearchOpen && (
        <TabSearchOverlay
          state={state}
          query={tabSearchQuery}
          setQuery={setTabSearchQuery}
          close={() => setTabSearchOpen(false)}
        />
      )}

      {pageActionConfirmation && state.huggingFacePageAction && (
        <div
          className="local-ai-dialog-backdrop hf-confirmation-backdrop"
          data-trusted-overlay="true"
        >
          <section
            ref={pageActionDialogRef}
            className="hf-install-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="hf-install-title"
          >
            <header>
              <div>
                <span className="ui-eyebrow">Hugging Face model</span>
                <h2 id="hf-install-title">
                  {state.huggingFacePageAction.trustMode === 'user-trusted'
                    ? 'Trust and load this model?'
                    : 'Load into Oryn?'}
                </h2>
              </div>
              <IconButton
                icon="close"
                label="Close"
                onClick={() => setPageActionConfirmation(false)}
              />
            </header>
            <div className="hf-model-reference">
              <p className="hf-model-identity">{state.huggingFacePageAction.repository}</p>
              <code>{state.huggingFacePageAction.exactRevision}</code>
            </div>
            {state.huggingFacePageAction.implementationNotice && (
              <p>{state.huggingFacePageAction.implementationNotice}</p>
            )}
            <dl className="hf-install-facts">
              <div>
                <dt>Task</dt>
                <dd>
                  {state.huggingFacePageAction.taskContract ??
                    state.huggingFacePageAction.task ??
                    'Model-defined'}
                </dd>
              </div>
              <div>
                <dt>Compatibility</dt>
                <dd>
                  {state.huggingFacePageAction.compatibility ?? 'Signed exact-model evidence'}
                </dd>
              </div>
              <div>
                <dt>Security</dt>
                <dd>
                  {state.huggingFacePageAction.securityStatus === 'allowed'
                    ? 'Oryn runtime only · no repository code'
                    : 'Blocked'}
                </dd>
              </div>
              <div>
                <dt>License</dt>
                <dd>{state.huggingFacePageAction.license ?? 'See source repository'}</dd>
              </div>
              <div>
                <dt>Download</dt>
                <dd>
                  {state.huggingFacePageAction.diskBytes === undefined
                    ? 'Calculating…'
                    : formatBytes(state.huggingFacePageAction.diskBytes)}
                </dd>
              </div>
              <div>
                <dt>Peak local use</dt>
                <dd>
                  {state.huggingFacePageAction.memoryBytes === undefined
                    ? 'Unknown'
                    : `~${formatBytes(state.huggingFacePageAction.memoryBytes)}`}
                </dd>
              </div>
            </dl>
            <div className="hf-selected-files">
              <strong>Selected file subset</strong>
              <ul>
                {state.huggingFacePageAction.selectedFiles.map((file) => (
                  <li key={file}>{file}</li>
                ))}
              </ul>
            </div>
            {state.huggingFacePageAction.trustMode === 'user-trusted' ? (
              <div className="hf-trust-warning" role="alert">
                <strong>This exact model has not been verified by Oryn.</strong>
                <p>
                  Oryn can attempt it using a sealed bundled runtime and will not execute repository
                  code. The model may still fail, behave unexpectedly, or use the local disk and
                  memory shown above. Your trust applies only to this pinned revision.
                </p>
              </div>
            ) : (
              <p className="hf-security-note">
                Oryn downloads the pinned revision and verifies it for the selected signed runtime
                pack. Repository scripts are never installed or executed.
              </p>
            )}
            <footer>
              <Button
                onClick={() =>
                  void window.browser.openModelSource(
                    state.huggingFacePageAction!.repository,
                    state.huggingFacePageAction!.exactRevision,
                  )
                }
              >
                View source
              </Button>
              <span className="hf-dialog-spacer" />
              <Button onClick={() => setPageActionConfirmation(false)}>Cancel</Button>
              <Button
                variant="primary"
                onClick={() => {
                  setPageActionConfirmation(false);
                  void window.browser.confirmHuggingFacePageAction(
                    state.huggingFacePageAction!.trustMode ?? 'oryn-verified',
                  );
                }}
              >
                {state.huggingFacePageAction.trustMode === 'user-trusted'
                  ? 'Trust and download exact model'
                  : 'Download exact model'}
              </Button>
            </footer>
          </section>
        </div>
      )}

      {panel && panel !== 'local-ai' && (
        <BrowserPanel
          panel={panel}
          state={state}
          data={data}
          historyQuery={historyQuery}
          setHistoryQuery={setHistoryQuery}
          refreshData={refreshData}
          appUpdate={appUpdate}
          setAppUpdate={setAppUpdate}
          close={() => setPanel(null)}
        />
      )}
      {showAppUpdateDialog && appUpdate.candidate && (
        <AppUpdateDialog
          dialogRef={appUpdateDialogRef}
          update={appUpdate}
          setUpdate={setAppUpdate}
          later={() => setDismissedReadyVersion(appUpdate.candidate!.version)}
          openSettings={() => setPanel('settings')}
        />
      )}
      {panel === 'local-ai' && (
        <LocalAIPanel
          resources={state.localAIResources}
          pressure={state.memoryPressure}
          engineState={state.engineState}
          activity={jobActivity}
          modelOperations={visibleModelOperations}
          dismissModelOperation={(operationId) =>
            setModelOperations((current) =>
              current.filter((operation) => operation.operationId !== operationId),
            )
          }
          audit={aiActivity}
          refreshAudit={() => window.browser.getAIActivity().then(setAIActivity)}
          showOnboarding={() =>
            setData((current) => ({
              ...current,
              appearance: { ...current.appearance, onboardingComplete: false },
            }))
          }
          close={() => setPanel(null)}
        />
      )}
      {permissionPrompt && (
        <LocalAIPermissionDialog
          prompt={permissionPrompt}
          chooserOpen={chooserOpen}
          selectedModelKey={selectedModelKey}
          rememberSiteChoice={rememberSiteChoice}
          setChooserOpen={setChooserOpen}
          setSelectedModelKey={setSelectedModelKey}
          setRememberSiteChoice={setRememberSiteChoice}
          respond={(action) => {
            const selected = permissionPrompt.candidates.find(
              (candidate) => modelIdentityKey(candidate.model) === selectedModelKey,
            );
            if (
              action !== 'deny' &&
              (!selected || !selected.executable || !selected.securityAllowed)
            )
              return;
            const response: LocalAIPermissionPromptResponse = {
              promptId: permissionPrompt.promptId,
              action,
              ...(selectedModelKey === modelIdentityKey(permissionPrompt.resolvedModel)
                ? {}
                : { selectedModel: selected?.model }),
              rememberSiteChoice,
            };
            void window.browser.respondToLocalAIPermission(response).then(() => {
              setPermissionPrompt(null);
            });
          }}
        />
      )}
      {!data.appearance.onboardingComplete && (
        <div className="onboarding-backdrop">
          <section
            ref={onboardingDialogRef}
            className="onboarding-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="oryn-onboarding-title"
            aria-describedby="oryn-onboarding-description"
          >
            <span className="onboarding-mark">
              <Icon name="sparkles" />
            </span>
            <span className="trusted-label">Welcome to Oryn Browser</span>
            <h1 id="oryn-onboarding-title">Run AI models on your computer</h1>
            <p id="oryn-onboarding-description">
              Websites can request compatible AI models. You choose what runs. Models are downloaded
              locally, and Oryn always shows when AI is using your hardware.
            </p>
            <p>
              You can change models, cancel work, inspect AI Activity, and delete downloaded models
              at any time.
            </p>
            <Button
              autoFocus
              variant="primary"
              icon="arrow-right"
              onClick={() =>
                void window.browser
                  .updateAppearance({ onboardingComplete: true })
                  .then(() => refreshData())
              }
            >
              Start browsing
            </Button>
          </section>
        </div>
      )}
    </header>
  );
}

function LocalAIPermissionDialog({
  prompt,
  chooserOpen,
  selectedModelKey,
  rememberSiteChoice,
  setChooserOpen,
  setSelectedModelKey,
  setRememberSiteChoice,
  respond,
}: {
  prompt: LocalAIPermissionPrompt;
  chooserOpen: boolean;
  selectedModelKey: string;
  rememberSiteChoice: boolean;
  setChooserOpen: (value: boolean) => void;
  setSelectedModelKey: (value: string) => void;
  setRememberSiteChoice: (value: boolean) => void;
  respond: (action: LocalAIPermissionPromptResponse['action']) => void;
}) {
  const selected = prompt.candidates.find(
    (candidate) => modelIdentityKey(candidate.model) === selectedModelKey,
  );
  const selectedRepository = selected?.model.repository ?? prompt.resolvedRepository;
  const dialogRef = useRef<HTMLElement>(null);
  const compatibility = selected?.compatibility ?? prompt.compatibility;
  const trustRequired = selected?.trustRequired ?? false;
  useFocusTrap(true, dialogRef, () => respond('deny'));
  return (
    <div className="local-ai-dialog-backdrop" role="presentation">
      <section
        ref={dialogRef}
        className="local-ai-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Local AI permission"
      >
        <header className="permission-header">
          <span className="permission-mark">
            <Icon name="shield" />
          </span>
          <div>
            <span className="trusted-label">Local execution request</span>
            <h2>{prompt.origin} wants to use AI</h2>
            <p>Review the model and choose how long this site may use it.</p>
          </div>
        </header>
        <div className="permission-model-summary">
          <div>
            <span className="permission-summary-label">Selected model</span>
            <strong>{selectedRepository}</strong>
          </div>
          <Badge tone={compatibility === 'incompatible' || trustRequired ? 'warning' : 'success'}>
            {trustRequired
              ? 'Not Oryn verified · approval required'
              : compatibilityLabel(compatibility, Boolean(selected?.developerCertified))}
          </Badge>
        </div>
        <dl className="permission-facts">
          <div>
            <dt>Purpose</dt>
            <dd>{prompt.role}</dd>
          </div>
          <div>
            <dt>Developer requested</dt>
            <dd>{prompt.requestedRepository}</dd>
          </div>
          <div>
            <dt>Source</dt>
            <dd>
              Hugging Face · {selectedRepository}{' '}
              <Button
                variant="ghost"
                size="sm"
                icon="external-link"
                className="source-link"
                onClick={() =>
                  void window.browser.openModelSource(
                    selectedRepository,
                    selected?.model.revision ?? prompt.exactRevision,
                  )
                }
              >
                Model page
              </Button>
            </dd>
          </div>
          <div>
            <dt>Exact revision</dt>
            <dd className="revision-value">{selected?.model.revision ?? prompt.exactRevision}</dd>
          </div>
          <div>
            <dt>Download</dt>
            <dd>
              {selected?.installed
                ? 'Already installed'
                : formatBytes(selected?.estimatedDiskBytes ?? prompt.downloadBytes ?? 0)}
            </dd>
          </div>
          <div>
            <dt>Expected local use</dt>
            <dd>
              ~{formatBytes(selected?.estimatedMemoryBytes ?? prompt.estimatedMemoryBytes ?? 0)}{' '}
              memory
            </dd>
          </div>
        </dl>
        {prompt.selection && (
          <p className="permission-note">
            {prompt.selection.kind === 'different-model'
              ? 'Different model. '
              : prompt.selection.kind === 'same-checkpoint'
                ? 'Same checkpoint; outputs may differ after conversion. '
                : ''}
            {prompt.selection.reasons.join(' ')} · {prompt.selection.confidence} · Catalog{' '}
            {prompt.selection.catalogVersion}
          </p>
        )}
        {prompt.substituted && (
          <p className="permission-note">
            <Icon name="sparkles" /> Substituted: {prompt.resolutionReason}
          </p>
        )}
        {compatibility === 'incompatible' && selected?.executable && selected.securityAllowed && (
          <div className="compatibility-warning" role="alert">
            <strong>
              <Icon name="alert" /> This model does not meet the website's requirements.
            </strong>
            <p>
              The feature may malfunction, truncate content, or fail. Choose another model or use it
              anyway.
            </p>
          </div>
        )}
        {trustRequired && selected?.executable && selected.securityAllowed && (
          <div className="compatibility-warning" role="alert">
            <strong>
              <Icon name="alert" /> This exact model revision has not passed Oryn conformance.
            </strong>
            <p>
              Oryn can attempt it inside the sealed runtime shown above. Repository code will not
              run, but the model may fail or produce unexpected results. Only you can grant this
              trust; the website cannot.
            </p>
          </div>
        )}
        {chooserOpen && (
          <div className="model-chooser" aria-label="Choose model">
            {prompt.candidates.map((candidate) => (
              <label
                className={selectedModelKey === modelIdentityKey(candidate.model) ? 'selected' : ''}
                key={modelIdentityKey(candidate.model)}
              >
                <input
                  type="radio"
                  name="model-choice"
                  value={modelIdentityKey(candidate.model)}
                  checked={selectedModelKey === modelIdentityKey(candidate.model)}
                  disabled={!candidate.executable || !candidate.securityAllowed}
                  onChange={() => setSelectedModelKey(modelIdentityKey(candidate.model))}
                />
                <span>
                  <strong>{candidate.model.repository}</strong>
                  <small className="candidate-status">
                    {candidateAvailabilityLabel(candidate)}
                    {' · '}
                    {candidate.installed ? 'Installed' : 'Download required'}
                  </small>
                </span>
              </label>
            ))}
            <label className="remember-choice">
              <input
                type="checkbox"
                checked={rememberSiteChoice}
                onChange={(event) => setRememberSiteChoice(event.target.checked)}
              />
              Remember for {prompt.origin} / {prompt.role}
            </label>
          </div>
        )}
        <p className="local-execution-note">
          <Icon name="cpu" /> This model runs on your computer.
        </p>
        <div className="permission-actions">
          <Button
            variant="ghost"
            icon="models"
            autoFocus
            onClick={() => {
              const next = !chooserOpen;
              setChooserOpen(next);
              if (next) void window.browser.openLocalAIModelChooser(prompt.promptId);
            }}
          >
            Choose model
          </Button>
          <Button onClick={() => respond('deny')}>Deny once</Button>
          <Button
            disabled={!selected?.executable || !selected.securityAllowed}
            onClick={() => respond('allow-once')}
          >
            {trustRequired
              ? 'Trust and allow once'
              : compatibility === 'incompatible'
                ? 'Use anyway once'
                : 'Allow once'}
          </Button>
          {!isLoopbackOrigin(prompt.origin) && (
            <Button
              variant="primary"
              disabled={!selected?.executable || !selected.securityAllowed}
              onClick={() => respond('always-allow')}
            >
              {trustRequired ? 'Trust for this site' : 'Always allow'}
            </Button>
          )}
        </div>
      </section>
    </div>
  );
}

function isLoopbackOrigin(origin: string): boolean {
  try {
    const hostname = new URL(origin).hostname.toLowerCase();
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]';
  } catch {
    return false;
  }
}

function isVisibleModelOperation(operation: TrustedModelOperation): boolean {
  return !['ready', 'cancelled'].includes(operation.stage);
}

function mergeModelOperation(
  current: TrustedModelOperation[],
  operation: TrustedModelOperation,
): TrustedModelOperation[] {
  const withoutOperation = current.filter((item) => item.operationId !== operation.operationId);
  if (!isVisibleModelOperation(operation)) return withoutOperation;
  const previous = current.find((item) => item.operationId === operation.operationId);
  return [...withoutOperation, { ...previous, ...operation }];
}

function ModelOperationList({
  operations,
  dismiss,
}: {
  operations: TrustedModelOperation[];
  dismiss: (operationId: string) => void;
}): React.JSX.Element | null {
  if (operations.length === 0) return null;
  return (
    <section className="model-operation-strip" aria-label="Model preparation" aria-live="polite">
      <header>
        <span className="operation-mark">
          <Icon name="download" />
        </span>
        <div>
          <span className="ui-eyebrow">Local AI</span>
          <strong>Preparing models</strong>
        </div>
        <Badge tone="brand">{operations.length}</Badge>
      </header>
      {operations.map((operation) => (
        <article key={operation.operationId}>
          <div>
            <strong>{modelOperationTitle(operation)}</strong>
            <span>
              {operation.origin}
              {operation.repository ? ` · Hugging Face · ${operation.repository}` : ''}
            </span>
            {operation.stage === 'downloading' && operation.totalBytes !== undefined && (
              <ProgressBar
                label={`Download progress for ${operation.repository ?? 'model'}`}
                value={operation.downloadedBytes ?? 0}
                max={operation.totalBytes}
              />
            )}
            {operation.stage === 'failed' && (
              <span role="alert">{modelOperationError(operation)}</span>
            )}
          </div>
          {operation.stage !== 'failed' ? (
            <Button
              size="sm"
              onClick={() => void window.browser.cancelModelOperation(operation.operationId)}
            >
              Cancel
            </Button>
          ) : (
            <Button size="sm" onClick={() => dismiss(operation.operationId)}>
              Dismiss
            </Button>
          )}
        </article>
      ))}
    </section>
  );
}

function HuggingFacePageActionDetails({
  action,
  close,
}: {
  action: HuggingFacePageActionState;
  close: () => void;
}) {
  const [huggingFaceCredential, setHuggingFaceCredential] =
    useState<HuggingFaceCredentialStatus | null>(null);
  const [huggingFaceToken, setHuggingFaceToken] = useState('');
  const [huggingFaceCredentialBusy, setHuggingFaceCredentialBusy] = useState(false);
  const [huggingFaceCredentialError, setHuggingFaceCredentialError] = useState<string | null>(null);

  useEffect(() => {
    if (action.status === 'access-required') {
      void window.browser.getHuggingFaceCredentialStatus().then(setHuggingFaceCredential);
    }
  }, [action.status]);

  return (
    <aside
      className="hf-page-action-details"
      data-trusted-overlay="true"
      role="dialog"
      aria-label="Oryn model support details"
    >
      <header>
        <Badge
          tone={
            action.status === 'blocked'
              ? 'danger'
              : action.status === 'downloading'
                ? 'brand'
                : 'warning'
          }
        >
          {huggingFacePageActionLabel(action)}
        </Badge>
        <IconButton icon="close" label="Close details" onClick={() => close()} />
      </header>
      <strong>{action.repository}</strong>
      <p>{action.message}</p>
      {action.blockingReasons?.includes('CUSTOM_CODE_REQUIRED') &&
        action.status !== 'inspecting' && (
          <section>
            <p>
              Oryn can compare the repository source with its reviewed built-in implementations
              without running repository Python. A match does not verify the model’s outputs.
            </p>
            <Button
              size="sm"
              onClick={() => {
                close();
                void window.browser.retryHuggingFacePageAction();
              }}
            >
              Check reviewed implementation
            </Button>
          </section>
        )}
      {action.implementationNotice && <p>{action.implementationNotice}</p>}
      {action.implementationDetails && (
        <details>
          <summary>Implementation details</summary>
          <p>{action.implementationDetails}</p>
        </details>
      )}
      {action.status === 'resource-error' &&
        action.blockingReasons?.some((reason) =>
          ['MODEL_MEMORY_INSUFFICIENT', 'MODEL_MEMORY_BUDGET_EXCEEDED'].includes(reason),
        ) && (
          <ModelMemoryRequirements
            requiredBytes={action.memoryBytes}
            availableBytes={action.availableMemoryBytes}
          />
        )}
      {action.exactRevision && <code>{action.exactRevision}</code>}
      {action.status === 'downloading' && <DownloadDetails action={action} />}
      {action.status === 'resource-error' &&
        action.requiredDiskBytes !== undefined &&
        action.availableDiskBytes !== undefined && (
          <section className="hf-download-details" aria-label="Model storage requirements">
            <strong>Storage required before download</strong>
            <dl>
              <div>
                <dt>Required</dt>
                <dd>{formatBytes(action.requiredDiskBytes)}</dd>
              </div>
              <div>
                <dt>Available</dt>
                <dd>{formatBytes(action.availableDiskBytes)}</dd>
              </div>
            </dl>
            <p>Remove an unused model in Model Manager or free disk space, then retry.</p>
          </section>
        )}
      {action.status === 'access-required' && (
        <section className="hf-access-flow" aria-label="Hugging Face access setup">
          <div className="hf-access-step">
            <Badge tone="brand">1</Badge>
            <div>
              <strong>Accept the model terms in this Oryn tab</strong>
              <p>
                Hugging Face grants access to your account. If the page still shows the form,
                complete it here first.
              </p>
            </div>
          </div>
          <div className="hf-access-step">
            <Badge tone="brand">2</Badge>
            <div>
              <strong>Connect the Engine with a read token</strong>
              <p>
                Oryn keeps website cookies isolated. The token is encrypted by macOS and is never
                exposed to websites or model code.
              </p>
            </div>
          </div>
          {huggingFaceCredential?.configured ? (
            <div className="hf-credential-status" role="status">
              <Icon name="check" /> Hugging Face token configured
            </div>
          ) : (
            <label className="field hf-token-field">
              <span>Hugging Face read token</span>
              <input
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder="hf_…"
                value={huggingFaceToken}
                disabled={huggingFaceCredentialBusy}
                onChange={(event) => setHuggingFaceToken(event.target.value)}
              />
            </label>
          )}
          {huggingFaceCredential && !huggingFaceCredential.secureStorageAvailable && (
            <p className="ui-error" role="alert">
              Secure credential storage is unavailable on this Mac. Oryn will not save a token
              without it.
            </p>
          )}
          {huggingFaceCredentialError && (
            <p className="ui-error" role="alert">
              {huggingFaceCredentialError}
            </p>
          )}
          <div className="hf-access-actions">
            <Button
              size="sm"
              onClick={() => void window.browser.newTab('https://huggingface.co/settings/tokens')}
            >
              Create a read token
            </Button>
            {huggingFaceCredential?.configured ? (
              <>
                <Button
                  size="sm"
                  variant="primary"
                  disabled={huggingFaceCredentialBusy}
                  onClick={() => {
                    setHuggingFaceCredentialError(null);
                    void window.browser.retryHuggingFacePageAction();
                  }}
                >
                  Check access again
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  disabled={huggingFaceCredentialBusy}
                  onClick={() => {
                    setHuggingFaceCredentialBusy(true);
                    setHuggingFaceCredentialError(null);
                    void window.browser
                      .clearHuggingFaceCredential()
                      .then(setHuggingFaceCredential)
                      .catch(() =>
                        setHuggingFaceCredentialError('The saved token could not be removed.'),
                      )
                      .finally(() => setHuggingFaceCredentialBusy(false));
                  }}
                >
                  Remove token
                </Button>
              </>
            ) : (
              <Button
                size="sm"
                variant="primary"
                disabled={
                  huggingFaceCredentialBusy ||
                  !huggingFaceCredential?.secureStorageAvailable ||
                  !huggingFaceToken
                }
                onClick={() => {
                  setHuggingFaceCredentialBusy(true);
                  setHuggingFaceCredentialError(null);
                  void window.browser
                    .setHuggingFaceCredential(huggingFaceToken)
                    .then((status) => {
                      setHuggingFaceCredential(status);
                      setHuggingFaceToken('');
                    })
                    .catch(() =>
                      setHuggingFaceCredentialError(
                        'Use a valid Hugging Face token beginning with hf_.',
                      ),
                    )
                    .finally(() => setHuggingFaceCredentialBusy(false));
                }}
              >
                Save token and check
              </Button>
            )}
          </div>
        </section>
      )}
      <footer>
        {action.status === 'resource-error' && (
          <Button size="sm" onClick={() => void window.browser.retryHuggingFacePageAction()}>
            Check again
          </Button>
        )}
        <Button
          size="sm"
          onClick={() =>
            void window.browser.openModelSource(action.repository, action.exactRevision)
          }
        >
          Open source in Oryn
        </Button>
        {action.status === 'downloading' && (
          <Button
            size="sm"
            variant="danger"
            onClick={() => {
              close();
              void window.browser.cancelHuggingFacePageAction();
            }}
          >
            Cancel download
          </Button>
        )}
      </footer>
    </aside>
  );
}

export function BrowserPassiveOverlay(): React.JSX.Element {
  // Informational cards must stay in this bounded native view. A transparent
  // full-window chrome view would still intercept website input outside the card.
  const [modelDetailsState, setModelDetailsState] = useState<BrowserWindowState | null>(null);
  const privateModeRef = useRef<boolean | undefined>(undefined);
  useEffect(() => {
    const stopOpen = window.browser.onHuggingFacePageActionDetails((next) => {
      if (!next.privateMode) setModelDetailsState(next);
    });
    const stopState = window.browser.onStateChanged((next) => {
      privateModeRef.current = next.privateMode === true;
      setModelDetailsState((current) => {
        if (next.privateMode || !current || current.activeTabId !== next.activeTabId) return null;
        const previousURL = current.tabs.find((tab) => tab.tabId === current.activeTabId)?.url;
        const nextURL = next.tabs.find((tab) => tab.tabId === next.activeTabId)?.url;
        return previousURL === nextURL && next.huggingFacePageAction ? next : null;
      });
    });
    return () => {
      stopOpen();
      stopState();
    };
  }, []);
  const modelDetails = modelDetailsState?.huggingFacePageAction;

  const [modelOperations, setModelOperations] = useState<TrustedModelOperation[]>([]);
  const [notification, setNotification] = useState<string | null>(null);
  const [findOpen, setFindOpen] = useState(false);
  const [findQuery, setFindQuery] = useState('');
  const [findResult, setFindResult] = useState({ matches: 0, activeMatchOrdinal: 0 });
  useEffect(() => window.browser.onFindResult(setFindResult), []);
  const previousJobs = useRef(new Set<string>());
  const surfaceRef = useRef<HTMLElement>(null);
  const findInputRef = useRef<HTMLInputElement>(null);
  const visibleModelOperations = modelOperations.filter(isVisibleModelOperation);

  useEffect(() => {
    const stopFindListener = window.browser.onOpenFind(() => {
      setFindOpen(true);
      requestAnimationFrame(() => {
        findInputRef.current?.focus();
        findInputRef.current?.select();
      });
    });
    return stopFindListener;
  }, []);

  const closeFind = () => {
    setFindOpen(false);
    void window.browser.stopFindInPage();
  };

  useEffect(() => {
    void window.browser
      .getState()
      .then((next) => {
        privateModeRef.current = next.privateMode === true;
        if (!next.privateMode)
          return window.browser.getLocalAIJobActivity().then((activity) => {
            previousJobs.current = new Set(activity.jobs.map((job) => job.jobId));
          });
      })
      .catch(() => undefined);
    const stopJobs = window.browser.onLocalAIJobActivityChanged((activity) => {
      if (privateModeRef.current !== false) return;
      const completed = [...previousJobs.current].filter(
        (jobId) => !activity.jobs.some((job) => job.jobId === jobId),
      );
      if (completed.length > 0) {
        setNotification('Local AI work finished. Open AI Activity for details.');
      }
      previousJobs.current = new Set(activity.jobs.map((job) => job.jobId));
    });
    const stopOperations = window.browser.onModelOperationChanged((operation) => {
      if (privateModeRef.current !== false) return;
      setModelOperations((current) => mergeModelOperation(current, operation));
    });
    return () => {
      stopJobs();
      stopOperations();
    };
  }, []);

  useLayoutEffect(() => {
    const visible =
      findOpen ||
      visibleModelOperations.length > 0 ||
      Boolean(notification) ||
      Boolean(modelDetails);
    const measure = () => {
      const surface = surfaceRef.current;
      if (!surface) return;
      const style = getComputedStyle(surface);
      // Keep cards at their existing width while reserving native space for the shadow.
      const width = 448 + parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
      const height = visible ? Math.min(4096, Math.ceil(surface.scrollHeight)) : 0;
      void window.browser.setPassiveOverlayLayout({ visible, width, height });
    };
    measure();
    const frame = requestAnimationFrame(measure);
    const observer = new ResizeObserver(measure);
    if (surfaceRef.current) observer.observe(surfaceRef.current);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [findOpen, notification, modelOperations, modelDetails]);

  return (
    <main className="passive-overlay-surface" ref={surfaceRef}>
      {modelDetails && (
        <HuggingFacePageActionDetails
          key={`${modelDetailsState.activeTabId}:${modelDetails.repository}`}
          action={modelDetails}
          close={() => setModelDetailsState(null)}
        />
      )}
      {findOpen && (
        <form
          className="find-bar"
          role="search"
          onSubmit={(event) => {
            event.preventDefault();
            void window.browser.findInPage(findQuery);
          }}
        >
          <input
            ref={findInputRef}
            autoFocus
            aria-label="Find in page"
            value={findQuery}
            onKeyDown={(event) => {
              if (event.key === 'Escape') closeFind();
            }}
            onChange={(event) => {
              setFindQuery(event.target.value);
              void window.browser.findInPage(event.target.value);
            }}
          />
          <IconButton
            type="button"
            icon="arrow-left"
            label="Previous match"
            onClick={() => void window.browser.findInPage(findQuery, false)}
          />
          <IconButton
            type="button"
            icon="arrow-right"
            label="Next match"
            onClick={() => void window.browser.findInPage(findQuery, true)}
          />
          <span role="status">
            {findQuery
              ? findResult.matches
                ? `${findResult.activeMatchOrdinal} of ${findResult.matches}`
                : 'No matches'
              : 'Type to find'}
          </span>
          <IconButton type="button" icon="close" label="Close find" onClick={closeFind} />
        </form>
      )}
      <ModelOperationList
        operations={visibleModelOperations}
        dismiss={(operationId) =>
          setModelOperations((current) =>
            current.filter((operation) => operation.operationId !== operationId),
          )
        }
      />

      {notification && (
        <div className="browser-notification" role="status">
          <span>{notification}</span>
          <IconButton
            icon="close"
            label="Dismiss notification"
            variant="surface"
            onClick={() => setNotification(null)}
          />
        </div>
      )}
    </main>
  );
}

function BrowserMenu({
  state,
  onClose,
  onSelect,
  onFind,
  onCheckUpdates,
}: {
  state: BrowserWindowState;
  onClose: () => void;
  onSelect: (panel: Exclude<Panel, null>) => void;
  onFind: () => void;
  onCheckUpdates: () => void;
}) {
  const runAndClose = (action: () => unknown) => {
    onClose();
    void action();
  };
  return (
    <div className="browser-menu" data-trusted-overlay="true" role="menu">
      <MenuItem icon="plus" onClick={() => runAndClose(() => window.browser.newWindow())}>
        New window
      </MenuItem>
      <MenuItem icon="plus" onClick={() => runAndClose(() => window.browser.newPrivateWindow())}>
        New private window
      </MenuItem>
      <MenuItem icon="box" onClick={() => runAndClose(() => window.browser.openFile())}>
        Open file…
      </MenuItem>
      <MenuItem icon="close" onClick={() => runAndClose(() => window.browser.closeWindow())}>
        Close window
      </MenuItem>
      <MenuItem icon="close" onClick={() => runAndClose(() => window.browser.quit())}>
        Quit Oryn
      </MenuItem>

      <MenuItem
        icon="plus"
        shortcut="⌘T"
        onClick={() => runAndClose(() => window.browser.newTab())}
      >
        New tab
      </MenuItem>
      <MenuItem icon="history" onClick={() => runAndClose(() => window.browser.reopenClosedTab())}>
        Reopen closed tab
      </MenuItem>
      <MenuItem icon="history" onClick={() => onSelect('history')}>
        History
      </MenuItem>
      <MenuItem icon="bookmark" onClick={() => onSelect('bookmarks')}>
        Bookmarks
      </MenuItem>
      <MenuItem icon="download" onClick={() => onSelect('downloads')}>
        Downloads
      </MenuItem>
      <MenuItem icon="search" shortcut="⌘F" onClick={onFind}>
        Find in page
      </MenuItem>
      <div className="menu-row">
        <span>Zoom {Math.round(state.zoomFactor * 100)}%</span>
        <button aria-label="Zoom out" onClick={() => void window.browser.setZoom('out')}>
          <Icon name="minus" />
        </button>
        <button onClick={() => void window.browser.setZoom('reset')}>Reset</button>
        <button aria-label="Zoom in" onClick={() => void window.browser.setZoom('in')}>
          <Icon name="plus" />
        </button>
      </div>
      <MenuItem icon="print" onClick={() => runAndClose(() => window.browser.printPage())}>
        Print…
      </MenuItem>
      <MenuItem icon="settings" onClick={() => onSelect('settings')}>
        Browser settings
      </MenuItem>
      <MenuItem icon="refresh" onClick={onCheckUpdates}>
        Check for updates…
      </MenuItem>
      <MenuItem icon="cpu" onClick={() => runAndClose(() => window.browser.openDevTools())}>
        Developer tools
      </MenuItem>
    </div>
  );
}

function LocalAIPanel({
  activity,
  modelOperations,
  dismissModelOperation,
  audit,
  resources,
  pressure,
  engineState,
  refreshAudit,
  showOnboarding,
  close,
}: {
  activity: LocalAIJobActivity;
  modelOperations: TrustedModelOperation[];
  dismissModelOperation: (operationId: string) => void;
  audit: AIActivitySnapshot;
  resources: BrowserWindowState['localAIResources'];
  pressure: BrowserWindowState['memoryPressure'];
  engineState: BrowserWindowState['engineState'];
  refreshAudit: () => Promise<void>;
  showOnboarding: () => void;
  close: () => void;
}) {
  const [view, setView] = useState<'running' | 'activity' | 'models' | 'engines' | 'settings'>(
    'running',
  );
  const [query, setQuery] = useState('');
  const [confirmClear, setConfirmClear] = useState(false);
  const [models, setModels] = useState<ModelManagerSnapshot>(emptyModelManager);
  const [modelLoadState, setModelLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [hasModelSnapshot, setHasModelSnapshot] = useState(false);
  const [runtimeLoadState, setRuntimeLoadState] = useState<'loading' | 'ready' | 'error'>(
    'loading',
  );
  const [expandedModels, setExpandedModels] = useState<Record<string, boolean>>({});
  const [compatibilityRevision, setCompatibilityRevision] = useState(0);
  const modelRefresh = useRef(0);
  const runtimeRefresh = useRef(0);
  useEffect(
    () => () => {
      modelRefresh.current++;
      runtimeRefresh.current++;
    },
    [],
  );
  const [removal, setRemoval] = useState<{
    model: TrustedInstalledModel;
    preview: TrustedModelRemovalPreview;
  } | null>(null);
  const [modelActionStatus, setModelActionStatus] = useState('');
  const [settings, setOrynSettings] = useState<OrynSettingsSnapshot>(emptyOrynSettings);
  const [settingsSection, setSettingsSection] = useState<
    'models' | 'preferences' | 'compute' | 'sites' | 'activity' | 'storage' | 'developer'
  >('models');
  const [cleanupConfirmation, setCleanupConfirmation] = useState<'artifacts' | 'temporary' | null>(
    null,
  );
  const filtered = audit.records.filter((record) => {
    const needle = query.trim().toLowerCase();
    if (!needle) return true;
    return JSON.stringify({
      origin: record.origin,
      role: record.role,
      requested: record.requestedModel.repository,
      resolved: record.resolvedModel.repository,
      status: record.execution.status,
      date: new Date(record.createdAt).toLocaleString(),
      invocation: record.invocation.input,
    })
      .toLowerCase()
      .includes(needle);
  });
  const updateSettings = async (settings: Partial<AIActivitySettings>) => {
    await window.browser.updateAIActivitySettings(settings);
    await refreshAudit();
  };
  const refreshRuntimePacks = async () => {
    const request = ++runtimeRefresh.current;
    setRuntimeLoadState('loading');
    try {
      const result = await withRequestTimeout(window.browser.getManagedRuntimePacks());
      if (request !== runtimeRefresh.current) return;
      setModels((previous) => ({ ...previous, ...result }));
      setRuntimeLoadState('ready');
    } catch {
      if (request === runtimeRefresh.current) setRuntimeLoadState('error');
    }
  };
  const refreshModels = async () => {
    const request = ++modelRefresh.current;
    setModelLoadState('loading');
    void refreshRuntimePacks();
    try {
      const result = await withRequestTimeout(window.browser.getModelManager());
      if (request !== modelRefresh.current) return;
      setModels((previous) => ({
        ...previous,
        models: result.models,
        totalLogicalBytes: result.totalLogicalBytes,
        totalReclaimableBytes: result.totalReclaimableBytes,
      }));
      setHasModelSnapshot(true);
      setModelLoadState('ready');
      setCompatibilityRevision((value) => value + 1);
    } catch {
      if (request === modelRefresh.current) setModelLoadState('error');
    }
  };
  const refreshOrynSettings = async () => setOrynSettings(await window.browser.getOrynSettings());
  useEffect(() => {
    if (view === 'models') void refreshModels();
    if (view === 'settings') void refreshOrynSettings();
  }, [view]);
  const unloadModel = async (model: TrustedInstalledModel) => {
    setModelActionStatus(`Unloading ${model.displayName}…`);
    try {
      await window.browser.unloadManagedModel(model.installationId);
      setModelActionStatus(`${model.displayName} is unloaded or will unload when idle.`);
      await refreshModels();
    } catch {
      setModelActionStatus(`${model.displayName} could not be unloaded.`);
    }
  };
  const setRuntimeOverride = async (model: TrustedInstalledModel, candidateId: string) => {
    setModelActionStatus(`Updating the runtime for ${model.displayName}…`);
    try {
      await window.browser.setManagedModelRuntimeOverride(
        model.installationId,
        candidateId || undefined,
      );
      setModelActionStatus(
        candidateId
          ? `${model.displayName} will use the selected safe runtime.`
          : `${model.displayName} is back on automatic runtime selection.`,
      );
      await refreshModels();
    } catch {
      setModelActionStatus('That runtime is no longer available for this installed model.');
    }
  };
  const previewRemoval = async (model: TrustedInstalledModel) => {
    setModelActionStatus('');
    setRemoval({
      model,
      preview: await window.browser.previewManagedModelRemoval(model.installationId),
    });
  };
  const removeModel = async (action: 'delete' | 'when-idle' | 'cancel-and-delete') => {
    if (!removal) return;
    const name = removal.model.displayName;
    setModelActionStatus(`Removing ${name}…`);
    try {
      const state = await window.browser.removeManagedModel(removal.model.installationId, action);
      setRemoval(null);
      setModelActionStatus(
        state === 'removed' ? `${name} was removed.` : `${name} will be removed when idle.`,
      );
      await refreshModels();
    } catch {
      setModelActionStatus(`${name} could not be removed because it is in use.`);
    }
  };
  const cleanupRuntimePacks = async () => {
    setModelActionStatus('Cleaning inactive runtime-pack versions…');
    try {
      const result = await window.browser.cleanupManagedRuntimePacks();
      setModelActionStatus(
        result.removedPacks === 0
          ? 'Runtime packs are already clean.'
          : `Removed ${result.removedPacks} inactive runtime-pack ${result.removedPacks === 1 ? 'version' : 'versions'} and reclaimed ${formatModelBytes(result.reclaimedBytes)}.`,
      );
      await refreshModels();
    } catch {
      setModelActionStatus('Inactive runtime packs could not be cleaned up.');
    }
  };
  return (
    <Sheet title="Local AI" icon="sparkles" wide close={close} ariaLabel="Oryn Local AI">
      <div className="local-ai-panel">
        <SegmentedTabs
          label="Local AI views"
          value={view}
          onChange={setView}
          items={[
            {
              value: 'running',
              label: 'Running',
              icon: 'activity',
              count: activity.running + activity.queued + modelOperations.length,
            },
            { value: 'activity', label: 'Activity', icon: 'history' },
            { value: 'models', label: 'Models', icon: 'models' },
            { value: 'engines', label: 'Engines', icon: 'cpu' },
            { value: 'settings', label: 'Settings', icon: 'settings' },
          ]}
        />
        <div className="local-ai-content">
          {view === 'engines' && <EngineVersions />}
          {view === 'running' && (
            <div>
              <ModelOperationList operations={modelOperations} dismiss={dismissModelOperation} />
              {engineState !== 'ready' && (
                <div className="runtime-error" role="alert">
                  <strong>Local AI runtime unavailable</strong>
                  <p>
                    Oryn cannot start new model work right now. Jobs are never replayed after a
                    crash; wait for the runtime to restart or restart Oryn Browser.
                  </p>
                </div>
              )}
              <div className="running-overview">
                <Stat label="Running" value={activity.running} />
                <Stat label="Queued" value={activity.queued} />
                {resources && (
                  <Stat
                    label="Compute"
                    value={(resources.acceleratorBackend ?? 'cpu').toUpperCase()}
                  />
                )}
                {activity.jobs.length > 0 && (
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => void window.browser.cancelAllLocalAIJobs()}
                  >
                    Cancel all
                  </Button>
                )}
              </div>
              <ResourceControlCard
                resources={resources}
                pressure={pressure}
                activeJob={activity.jobs.find((job) => job.state === 'running')}
                queued={activity.queued}
              />
              <div className="activity-job-list">
                {activity.jobs.map((job) => (
                  <article key={job.jobId}>
                    <div>
                      <strong>{job.origin}</strong>
                      <span>{job.resolvedModel}</span>
                      <small>
                        {titleCase(job.role)} ·{' '}
                        {job.cancellationRequested ? 'Cancelling…' : job.state}
                        {' · '}
                        {job.backend}
                        {job.progressStage ? ` · ${job.progressStage}` : ''}
                        {job.progress !== undefined ? ` · ${Math.round(job.progress * 100)}%` : ''}
                        {job.queue
                          ? ` · queue ${job.queue.position} (${job.queue.jobsAhead} ahead)`
                          : ''}
                      </small>
                      {job.substituted && <small>Requested: {job.requestedModel}</small>}
                      {job.inputPreview && <p>“{job.inputPreview}”</p>}
                    </div>
                    <div className="job-actions">
                      <Button
                        size="sm"
                        disabled={job.cancellationRequested}
                        onClick={() => void window.browser.cancelLocalAIJob(job.jobId)}
                      >
                        {job.cancellationRequested ? 'Cancelling…' : 'Cancel'}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => void window.browser.cancelLocalAIJobsForSite(job.origin)}
                      >
                        Cancel site
                      </Button>
                    </div>
                  </article>
                ))}
                {activity.jobs.length === 0 && <EmptyState text="No Local AI jobs running" />}
              </div>
            </div>
          )}
          {view === 'activity' && (
            <div>
              <div className="panel-actions">
                <input
                  aria-label="Search AI Activity"
                  placeholder="Filter by site, model, role, status, or prompt"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </div>
              <div className="ai-audit-list">
                {filtered.map((record) => (
                  <details key={record.id}>
                    <summary>
                      <span>{new Date(record.createdAt).toLocaleString()}</span>
                      <strong>
                        {record.origin} · {titleCase(record.role)}
                      </strong>
                      <span>{record.resolvedModel.repository}</span>
                      <span>
                        {titleCase(record.execution.status)}
                        {record.execution.durationMs !== undefined
                          ? ` · ${(record.execution.durationMs / 1000).toFixed(1)}s`
                          : ''}
                      </span>
                    </summary>
                    <dl className="activity-detail">
                      <div>
                        <dt>Developer requested</dt>
                        <dd>
                          {record.requestedModel.repository}@
                          {shortRevision(record.requestedModel.revision)}
                        </dd>
                      </div>
                      <div>
                        <dt>Executed</dt>
                        <dd>
                          {record.resolvedModel.repository}@
                          {shortRevision(record.resolvedModel.revision)}
                        </dd>
                      </div>
                      <div>
                        <dt>Why</dt>
                        <dd>
                          {record.resolution.reason} · {record.resolution.compatibility}
                        </dd>
                      </div>
                      <div>
                        <dt>Runtime</dt>
                        <dd>
                          {record.execution.runner} · {record.execution.backend}
                          {record.execution.foundationId
                            ? ` · ${record.execution.foundationId.replace('oryn.runtime.', '')}`
                            : ''}
                          {record.execution.adapterId
                            ? ` · ${record.execution.adapterId.replace('oryn.adapter.', '')}`
                            : ''}
                          {(record.execution.quantization ?? record.execution.precision)
                            ? ` · ${record.execution.quantization ?? record.execution.precision}`
                            : ''}
                          {record.execution.fallbacks?.length
                            ? ` · ${record.execution.fallbacks.length} runtime fallback${record.execution.fallbacks.length === 1 ? '' : 's'}`
                            : ''}
                        </dd>
                      </div>
                      <div>
                        <dt>Invocation</dt>
                        <dd>
                          <pre>
                            {record.invocation.input === undefined
                              ? 'Metadata only'
                              : JSON.stringify(record.invocation.input, null, 2)}
                          </pre>
                        </dd>
                      </div>
                      <div>
                        <dt>Artifacts</dt>
                        <dd>
                          {record.invocation.artifacts
                            ?.map(
                              (item) =>
                                `${item.name ?? item.mimeType} · ${formatBytes(item.sizeBytes)}`,
                            )
                            .join(', ') || 'None'}
                          {record.invocation.artifacts?.length ? (
                            <small className="upload-transparency">
                              These files stay local unless this website explicitly uploads them
                              over HTTPS.
                            </small>
                          ) : null}
                        </dd>
                      </div>
                      <div>
                        <dt>Result</dt>
                        <dd>
                          {record.result?.kind ?? 'None'}
                          {record.errorCode ? ` · ${record.errorCode}` : ''}
                          {record.result?.chat && (
                            <p>
                              {record.result.chat.finishReason === 'length'
                                ? 'Output limit reached'
                                : record.result.chat.finishReason === 'unknown'
                                  ? 'Completion unknown'
                                  : 'Answer completed'}
                              . Thinking requested:{' '}
                              {record.result.chat.thinking.requested === null
                                ? 'Model default'
                                : record.result.chat.thinking.requested
                                  ? 'On'
                                  : 'Off'}
                              ; effective:{' '}
                              {record.result.chat.thinking.effective === null
                                ? 'unknown'
                                : record.result.chat.thinking.effective
                                  ? 'On'
                                  : 'Off'}
                              .
                              {record.result.chat.thinking.requested !== null &&
                                !record.result.chat.thinking.applied &&
                                ' Preference was not applied.'}
                            </p>
                          )}
                          {record.result?.artifactIds?.map((artifactId) => (
                            <Button
                              key={artifactId}
                              variant="ghost"
                              size="sm"
                              icon="download"
                              className="source-link"
                              onClick={() => void window.browser.saveLocalAIArtifact(artifactId)}
                            >
                              Save result…
                            </Button>
                          ))}
                        </dd>
                      </div>
                    </dl>
                  </details>
                ))}
                {filtered.length === 0 && (
                  <EmptyState
                    text={
                      query.trim()
                        ? 'No AI Activity matches this filter.'
                        : 'No Oryn activity yet. When websites run models locally, their activity will appear here.'
                    }
                  />
                )}
              </div>
            </div>
          )}
          {view === 'models' && (
            <div className="model-manager" aria-label="Model Manager">
              <div className="model-manager-summary">
                <Stat label="Installed" value={hasModelSnapshot ? models.models.length : '—'} />
                <Stat
                  label="On disk"
                  value={hasModelSnapshot ? formatModelBytes(models.totalLogicalBytes) : '—'}
                />
                <Stat
                  label="Reclaimable"
                  value={hasModelSnapshot ? formatModelBytes(models.totalReclaimableBytes) : '—'}
                />
                <IconButton
                  icon="refresh"
                  label="Refresh models"
                  variant="surface"
                  onClick={() => void refreshModels()}
                />
              </div>
              {modelLoadState === 'loading' && <p role="status">Loading installed models…</p>}
              {modelLoadState === 'error' && (
                <p role="alert">
                  Could not refresh installed models.{' '}
                  {hasModelSnapshot ? 'Showing the last successful list. ' : ''}Use Refresh models
                  to try again.
                </p>
              )}
              {modelActionStatus && <p role="status">{modelActionStatus}</p>}
              <div className="model-list">
                {models.models.map((model) => (
                  <article key={model.installationId} className="model-card">
                    <div className="model-card-heading">
                      <div>
                        <strong>{model.displayName}</strong>
                        <span>Hugging Face · {model.repository}</span>
                      </div>
                      <div className="model-actions">
                        <Button
                          size="sm"
                          variant="ghost"
                          icon="external-link"
                          onClick={() =>
                            void window.browser.openManagedModelSource(model.installationId)
                          }
                        >
                          View source
                        </Button>
                        <Button
                          size="sm"
                          disabled={model.residencyState === 'unloaded'}
                          onClick={() => void unloadModel(model)}
                        >
                          Unload
                        </Button>
                        <Button
                          variant="danger"
                          size="sm"
                          icon="trash"
                          onClick={() => void previewRemoval(model)}
                        >
                          Delete
                        </Button>
                      </div>
                    </div>
                    <div className="model-state-grid">
                      <span>Installation: {titleCase(model.installationState)}</span>
                      <span>Residency: {titleCase(model.residencyState)}</span>
                      <span>Disk: {formatModelBytes(model.logicalSizeBytes)}</span>
                      <span>
                        Estimated memory:{' '}
                        {model.estimatedResidentBytes === undefined
                          ? 'Unavailable'
                          : formatModelBytes(model.estimatedResidentBytes)}
                      </span>
                      <span>Sessions: {model.activeSessions}</span>
                      <span>Running jobs: {model.runningJobs}</span>
                      <span>Queued jobs: {model.queuedJobs}</span>
                      <span>
                        Last used:{' '}
                        {model.lastUsedAt ? new Date(model.lastUsedAt).toLocaleString() : 'Never'}
                      </span>
                    </div>
                    <details
                      onToggle={(event) => {
                        const open = event.currentTarget.open;
                        setExpandedModels((previous) => ({
                          ...previous,
                          [model.installationId]: open,
                        }));
                      }}
                    >
                      <summary>Model details and substitutes</summary>
                      <ModelAlternatives
                        installationId={model.installationId}
                        availableBytes={
                          resources?.systemAvailableBytes === undefined
                            ? undefined
                            : Math.min(
                                resources.systemAvailableBytes,
                                resources.memoryBudgetBytes === undefined
                                  ? resources.systemAvailableBytes
                                  : Math.max(
                                      0,
                                      resources.memoryBudgetBytes -
                                        resources.estimatedResidentModelBytes,
                                    ),
                              )
                        }
                      />
                      <dl className="activity-detail">
                        <div>
                          <dt>Exact revision</dt>
                          <dd>{model.exactRevision}</dd>
                        </div>
                        <div>
                          <dt>Task and modalities</dt>
                          <dd>
                            {[model.task, ...model.modalities].filter(Boolean).join(' · ') ||
                              'Unavailable'}
                          </dd>
                        </div>
                        <div>
                          <dt>Family</dt>
                          <dd>{model.family ?? 'Unavailable'}</dd>
                        </div>
                        <div>
                          <dt>Parameters</dt>
                          <dd>{formatParameters(model.parameterCount)}</dd>
                        </div>
                        <div>
                          <dt>Precision / variant</dt>
                          <dd>{model.precision ?? 'Unavailable'}</dd>
                        </div>
                        <div>
                          <dt>Runner / backends</dt>
                          <dd>
                            {[model.runner, ...model.executableBackends]
                              .filter(Boolean)
                              .join(' · ') || 'Unavailable'}
                          </dd>
                        </div>
                        {model.runtimePack && (
                          <>
                            <div>
                              <dt>Trusted runtime pack</dt>
                              <dd>
                                {model.runtimePack.packId} {model.runtimePack.packVersion} ·{' '}
                                {model.runtimePack.runnerId} · {model.runtimePack.taskContract} ·{' '}
                                {titleCase(model.runtimePack.compatibilityLevel)}
                              </dd>
                            </div>
                            {model.runtimePack.trustedMetadata && (
                              <div>
                                <dt>License / conformance</dt>
                                <dd>
                                  {model.runtimePack.trustedMetadata.license} ·{' '}
                                  {titleCase(model.runtimePack.trustedMetadata.conformance.status)}
                                  {model.runtimePack.trustedMetadata.languages?.length
                                    ? ` · ${model.runtimePack.trustedMetadata.languages.join(', ')}`
                                    : ''}
                                  {model.runtimePack.trustedMetadata.voices?.length
                                    ? ` · ${model.runtimePack.trustedMetadata.voices.length} voices`
                                    : ''}
                                  {model.runtimePack.trustedMetadata.quantization
                                    ? ` · ${model.runtimePack.trustedMetadata.quantization}`
                                    : ''}
                                  {model.runtimePack.trustedMetadata.contextLength
                                    ? ` · ${model.runtimePack.trustedMetadata.contextLength.toLocaleString()} context`
                                    : ''}
                                </dd>
                              </div>
                            )}
                          </>
                        )}
                        {model.runtimeSelection && (
                          <>
                            <div>
                              <dt>
                                {model.runtimeSelection.automatic
                                  ? 'Automatic engine'
                                  : 'Selected engine'}
                              </dt>
                              <dd>
                                {model.runtimeSelection.foundationId.replace('oryn.runtime.', '')}{' '}
                                {model.runtimeSelection.foundationVersion} ·{' '}
                                {model.runtimeSelection.backend} ·{' '}
                                {model.runtimeSelection.precision}
                                {model.runtimeSelection.quantization
                                  ? ` · ${model.runtimeSelection.quantization}`
                                  : ''}
                              </dd>
                            </div>
                            <div>
                              <dt>Architecture adapter</dt>
                              <dd>
                                {model.runtimeSelection.adapterId.replace('oryn.adapter.', '')}{' '}
                                {model.runtimeSelection.adapterVersion} ·{' '}
                                {titleCase(model.runtimeSelection.policy)}
                                {model.runtimeSelection.fallbackCount
                                  ? ` · ${model.runtimeSelection.fallbackCount} safe fallback${model.runtimeSelection.fallbackCount === 1 ? '' : 's'}`
                                  : ''}
                              </dd>
                            </div>
                            <div>
                              <dt>Why Oryn selected it</dt>
                              <dd>
                                {model.runtimeSelection.reasonCodes.map(runtimeReason).join(' · ')}{' '}
                                · ~{formatModelBytes(model.runtimeSelection.estimatedMemoryBytes)}{' '}
                                peak local use
                              </dd>
                            </div>
                            {model.runtimeSelection.alternatives.length > 0 && (
                              <div>
                                <dt>Runtime preference</dt>
                                <dd>
                                  <select
                                    aria-label={`Runtime preference for ${model.displayName}`}
                                    value={
                                      model.runtimeSelection.automatic
                                        ? ''
                                        : model.runtimeSelection.candidateId
                                    }
                                    onChange={(event) =>
                                      void setRuntimeOverride(model, event.currentTarget.value)
                                    }
                                  >
                                    <option value="">Automatic (recommended)</option>
                                    {[
                                      {
                                        candidateId: model.runtimeSelection.candidateId,
                                        foundationId: model.runtimeSelection.foundationId,
                                        backend: model.runtimeSelection.backend,
                                        precision: model.runtimeSelection.precision,
                                        quantization: model.runtimeSelection.quantization,
                                      },
                                      ...model.runtimeSelection.alternatives,
                                    ].map((candidate) => (
                                      <option
                                        key={candidate.candidateId}
                                        value={candidate.candidateId}
                                      >
                                        {candidate.foundationId.replace('oryn.runtime.', '')} ·{' '}
                                        {candidate.backend} ·{' '}
                                        {candidate.quantization ?? candidate.precision}
                                      </option>
                                    ))}
                                  </select>
                                </dd>
                              </div>
                            )}
                          </>
                        )}
                        <div>
                          <dt>Used by</dt>
                          <dd>{model.usedBy.join(', ') || 'No associated sites'}</dd>
                        </div>
                        <div>
                          <dt>Preferences</dt>
                          <dd>{model.preferredFor.join(', ') || 'Not preferred'}</dd>
                        </div>
                      </dl>
                      <h3>Possible installed substitutes</h3>
                      {expandedModels[model.installationId] && (
                        <ModelSubstitutes
                          key={`${model.installationId}:${compatibilityRevision}`}
                          installationId={model.installationId}
                        />
                      )}
                    </details>
                  </article>
                ))}
                {modelLoadState === 'ready' && models.models.length === 0 && (
                  <UIEmptyState
                    icon="models"
                    title="No local AI models installed"
                    description="Models will appear here after you approve a website request."
                    action={<Button onClick={showOnboarding}>Learn how Oryn works</Button>}
                  />
                )}
              </div>
              <section className="runtime-pack-manager" aria-label="Trusted runtime packs">
                <div className="runtime-pack-heading">
                  <div>
                    <h3>Trusted runtime packs</h3>
                    <p>
                      Oryn-signed, first-party execution components. Third-party publishers and
                      automatic executable updates are off.
                    </p>
                  </div>
                  <Button size="sm" variant="secondary" onClick={() => void cleanupRuntimePacks()}>
                    Clean inactive versions
                  </Button>
                </div>
                {runtimeLoadState === 'loading' && <p role="status">Loading runtime packs…</p>}
                {runtimeLoadState === 'error' && (
                  <div role="alert">
                    <p>Could not refresh runtime packs. Your installed models are shown above.</p>
                    <Button size="sm" onClick={() => void refreshRuntimePacks()}>
                      Retry runtime packs
                    </Button>
                  </div>
                )}
                <div className="runtime-pack-list">
                  {models.runtimePacks.map((pack) => (
                    <details key={`${pack.packId}@${pack.version}`} className="runtime-pack-card">
                      <summary>
                        <span>
                          <strong>{pack.packId}</strong>
                          <small>
                            {pack.version} · {titleCase(pack.state)} ·{' '}
                            {pack.bundled ? 'Bundled with Oryn' : formatModelBytes(pack.sizeBytes)}
                          </small>
                        </span>
                        <Badge tone={pack.revoked ? 'danger' : 'success'}>
                          <Icon name={pack.revoked ? 'alert' : 'shield'} />
                          <span>{pack.revoked ? 'Revoked' : 'Verified'}</span>
                        </Badge>
                      </summary>
                      <dl className="activity-detail">
                        <div>
                          <dt>Publisher</dt>
                          <dd>{pack.publisher}</dd>
                        </div>
                        <div>
                          <dt>Runtime contract</dt>
                          <dd>
                            {pack.abi} · {pack.dependencyClosure}
                          </dd>
                        </div>
                        <div>
                          <dt>Platforms</dt>
                          <dd>{pack.platforms.join(', ')}</dd>
                        </div>
                        <div>
                          <dt>Safety floor</dt>
                          <dd>Minimum safe version {pack.minimumSafeVersion}</dd>
                        </div>
                        <div>
                          <dt>License and access</dt>
                          <dd>
                            {pack.license.summary} · {titleCase(pack.license.access)} access
                          </dd>
                        </div>
                        <div>
                          <dt>Notice</dt>
                          <dd>{pack.license.notice}</dd>
                        </div>
                      </dl>
                    </details>
                  ))}
                </div>
                <small className="runtime-pack-catalog-id">
                  Catalog v{models.runtimePackPolicy.catalogVersion}
                  {models.runtimePackPolicy.catalogDigest
                    ? ` · ${models.runtimePackPolicy.catalogDigest.slice(0, 12)}…`
                    : ''}
                </small>
              </section>
              {removal && (
                <div className="model-delete-dialog" role="alertdialog" aria-modal="true">
                  <h3>Delete {removal.model.displayName}?</h3>
                  <p>
                    This can reclaim {formatModelBytes(removal.preview.reclaimableBytes)}.{' '}
                    {removal.preview.sharedBytesRetained > 0 &&
                      `${formatModelBytes(removal.preview.sharedBytesRetained)} is shared and will be kept. `}
                    Saved model preferences will not change.
                  </p>
                  {!removal.preview.canRemoveNow && (
                    <p>
                      In use: {removal.preview.activeSessions} sessions,{' '}
                      {removal.preview.runningJobs} running jobs, {removal.preview.queuedJobs}{' '}
                      queued jobs.
                    </p>
                  )}
                  <div className="model-actions">
                    {removal.preview.canRemoveNow ? (
                      <Button
                        variant="danger"
                        icon="trash"
                        onClick={() => void removeModel('delete')}
                      >
                        Delete model
                      </Button>
                    ) : (
                      <>
                        <Button
                          variant="danger"
                          icon="trash"
                          onClick={() => void removeModel('cancel-and-delete')}
                        >
                          Cancel affected work and delete
                        </Button>
                        <Button onClick={() => void removeModel('when-idle')}>
                          Delete when idle
                        </Button>
                      </>
                    )}
                    <Button variant="ghost" onClick={() => setRemoval(null)}>
                      Keep model
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
          {view === 'settings' && (
            <div className="oryn-settings" aria-label="Local AI Settings">
              <nav className="settings-section-tabs" aria-label="Local AI Settings sections">
                {(
                  [
                    ['models', 'Models'],
                    ['preferences', 'Preferences'],
                    ['compute', 'Compute'],
                    ['sites', 'Sites'],
                    ['activity', 'AI Activity'],
                    ['storage', 'Storage'],
                    ['developer', 'Developer'],
                  ] as const
                ).map(([section, label]) => (
                  <button
                    key={section}
                    aria-pressed={settingsSection === section}
                    onClick={() => setSettingsSection(section)}
                  >
                    {label}
                  </button>
                ))}
              </nav>
              {settingsSection === 'models' && (
                <section>
                  <h3>Models</h3>
                  <p>
                    {settings.preferences.availableModels.length} installed ·{' '}
                    {formatModelBytes(settings.storage.modelBytes)} on disk
                  </p>
                  <Button icon="models" onClick={() => setView('models')}>
                    Open Model Manager
                  </Button>
                </section>
              )}
              {settingsSection === 'preferences' && (
                <section className="ai-activity-settings">
                  <h3>Preferences</h3>
                  <RegistrySettings />
                  <label>
                    <span>Prefer models already installed</span>
                    <input
                      type="checkbox"
                      checked={settings.preferences.preferInstalled}
                      onChange={(event) =>
                        void window.browser
                          .updateOrynRanking({ preferInstalled: event.target.checked })
                          .then(refreshOrynSettings)
                      }
                    />
                  </label>
                  <label>
                    <span>Optimize model selection for</span>
                    <select
                      value={settings.preferences.optimization}
                      onChange={(event) =>
                        void window.browser
                          .updateOrynRanking({
                            optimization: event.target.value as
                              'quality' | 'balanced' | 'speed' | 'memory',
                          })
                          .then(refreshOrynSettings)
                      }
                    >
                      <option value="balanced">Balanced</option>
                      <option value="quality">Quality</option>
                      <option value="speed">Performance</option>
                      <option value="memory">Memory Saver</option>
                    </select>
                  </label>
                  {['chat', 'coding', 'image-generation', 'transcription', 'song-generation'].map(
                    (role) => {
                      const exact = settings.preferences.globalModels.find(
                        (item) => item.role === role,
                      )?.model;
                      const family = settings.preferences.globalFamilies.find(
                        (item) => item.role === role,
                      )?.family;
                      return (
                        <fieldset key={role}>
                          <legend>{titleCase(role)}</legend>
                          <label>
                            <span>Exact model</span>
                            <select
                              value={exact ? `${exact.repository}@${exact.revision}` : ''}
                              onChange={(event) => {
                                const model = settings.preferences.availableModels.find(
                                  (item) =>
                                    `${item.repository}@${item.revision}` === event.target.value,
                                );
                                void window.browser
                                  .updateOrynRolePreference({ role, model, family })
                                  .then(refreshOrynSettings);
                              }}
                            >
                              <option value="">Automatic</option>
                              {settings.preferences.availableModels.map((model) => (
                                <option
                                  key={`${model.repository}@${model.revision}`}
                                  value={`${model.repository}@${model.revision}`}
                                >
                                  {model.repository}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            <span>Preferred family</span>
                            <input
                              defaultValue={family ?? ''}
                              placeholder="Optional model family"
                              onBlur={(event) =>
                                void window.browser
                                  .updateOrynRolePreference({
                                    role,
                                    model: exact,
                                    family: event.target.value,
                                  })
                                  .then(refreshOrynSettings)
                              }
                            />
                          </label>
                        </fieldset>
                      );
                    },
                  )}
                </section>
              )}
              {settingsSection === 'compute' && (
                <section className="ai-activity-settings">
                  <h3>Compute</h3>
                  <label>
                    <span>Local AI memory budget (GiB, blank for automatic)</span>
                    <input
                      type="number"
                      min="0.5"
                      step="0.5"
                      defaultValue={
                        settings.compute.memoryBudgetBytes === null
                          ? ''
                          : settings.compute.memoryBudgetBytes / 1024 ** 3
                      }
                      onBlur={(event) =>
                        void window.browser
                          .updateOrynCompute({
                            memoryBudgetBytes: event.target.value
                              ? Math.round(Number(event.target.value) * 1024 ** 3)
                              : null,
                          })
                          .then(refreshOrynSettings)
                      }
                    />
                  </label>
                  <label>
                    <span>Background jobs</span>
                    <select
                      value={settings.compute.backgroundJobs}
                      onChange={(event) =>
                        void window.browser
                          .updateOrynCompute({
                            backgroundJobs: event.target.value as 'allow' | 'ask' | 'disable',
                          })
                          .then(refreshOrynSettings)
                      }
                    >
                      <option value="allow">Allow</option>
                      <option value="ask">Require active tab</option>
                      <option value="disable">Disable</option>
                    </select>
                  </label>
                  <label>
                    <span>On battery</span>
                    <select
                      value={settings.compute.batteryMode}
                      onChange={(event) =>
                        void window.browser
                          .updateOrynCompute({
                            batteryMode: event.target.value as 'normal' | 'reduce' | 'disable',
                          })
                          .then(refreshOrynSettings)
                      }
                    >
                      <option value="normal">Normal</option>
                      <option value="reduce">Reduce background preparation</option>
                      <option value="disable">Disable new generation</option>
                    </select>
                  </label>
                  <p>Generation concurrency: {settings.compute.generationConcurrency} (fixed)</p>
                </section>
              )}
              {settingsSection === 'sites' && (
                <section>
                  <h3>Sites</h3>
                  {settings.sites.map((site) => (
                    <article key={site.origin} className="model-card">
                      <strong>{site.origin}</strong>
                      <p>
                        {site.permissions.length} permissions · {site.modelChoices.length} saved
                        model choices · {site.activityCount} activity records
                      </p>
                      <div className="model-actions">
                        {site.permissions.some(
                          (permission) =>
                            permission.decision === 'deny' || permission.decision === 'revoked',
                        ) && (
                          <Button
                            size="sm"
                            title="Clear saved blocks. The next request will ask for your approval."
                            onClick={() =>
                              void window.browser
                                .askOrynSiteAgain(site.origin)
                                .then(refreshOrynSettings)
                            }
                          >
                            Ask again
                          </Button>
                        )}
                        <Button
                          size="sm"
                          onClick={() =>
                            void window.browser
                              .resetOrynSiteChoices(site.origin)
                              .then(refreshOrynSettings)
                          }
                        >
                          Reset model choices
                        </Button>
                        <Button
                          size="sm"
                          onClick={() =>
                            void window.browser
                              .revokeOrynSite(site.origin, false)
                              .then(refreshOrynSettings)
                          }
                        >
                          Revoke future access
                        </Button>
                        <Button
                          size="sm"
                          variant="danger"
                          onClick={() =>
                            void window.browser
                              .revokeOrynSite(site.origin, true)
                              .then(refreshOrynSettings)
                          }
                        >
                          Revoke and cancel active work
                        </Button>
                      </div>
                    </article>
                  ))}
                  {settings.sites.length === 0 && (
                    <EmptyState text="No sites have used Local AI." />
                  )}
                </section>
              )}
              {settingsSection === 'activity' && (
                <section className="ai-activity-settings">
                  <h3>AI Activity</h3>
                  <label>
                    <span>Keep AI Activity</span>
                    <input
                      type="checkbox"
                      checked={audit.settings.keepActivity}
                      onChange={(event) =>
                        void updateSettings({ keepActivity: event.target.checked })
                      }
                    />
                  </label>
                  <fieldset>
                    <legend>Store invocation contents</legend>
                    {(['full', 'metadata-only'] as const).map((policy) => (
                      <label key={policy}>
                        <input
                          type="radio"
                          name="activity-content"
                          checked={audit.settings.contentPolicy === policy}
                          onChange={() => void updateSettings({ contentPolicy: policy })}
                        />
                        {policy === 'full' ? 'Full prompts/messages' : 'Metadata only'}
                      </label>
                    ))}
                  </fieldset>
                  <label>
                    <span>Retention</span>
                    <select
                      value={audit.settings.retentionDays ?? 'forever'}
                      onChange={(event) =>
                        void updateSettings({
                          retentionDays:
                            event.target.value === 'forever'
                              ? null
                              : (Number(event.target.value) as 7 | 30),
                        })
                      }
                    >
                      <option value="7">7 days</option>
                      <option value="30">30 days</option>
                      <option value="forever">Forever</option>
                    </select>
                  </label>
                  <Button
                    variant={confirmClear ? 'danger' : 'secondary'}
                    onClick={() => {
                      if (!confirmClear) return setConfirmClear(true);
                      void window.browser.clearAIActivity().then(async () => {
                        setConfirmClear(false);
                        await refreshAudit();
                      });
                    }}
                  >
                    {confirmClear ? 'Confirm clear AI Activity' : 'Clear AI Activity'}
                  </Button>
                </section>
              )}
              {settingsSection === 'storage' && (
                <section>
                  <h3>Storage</h3>
                  <p>Models: {formatBytes(settings.storage.modelBytes)}</p>
                  <p>
                    Shared model data retained: {formatBytes(settings.storage.modelSharedBytes)}
                  </p>
                  <p>
                    Artifacts: {formatBytes(settings.storage.artifactBytes)} (
                    {settings.storage.readyArtifacts} ready)
                  </p>
                  <p>Temporary files: {formatBytes(settings.storage.temporaryArtifactBytes)}</p>
                  <div className="model-actions">
                    <Button icon="models" onClick={() => setView('models')}>
                      Manage models
                    </Button>
                    {(['artifacts', 'temporary'] as const).map((target) => (
                      <Button
                        key={target}
                        variant={cleanupConfirmation === target ? 'danger' : 'secondary'}
                        onClick={() => {
                          if (cleanupConfirmation !== target) return setCleanupConfirmation(target);
                          void window.browser.cleanupOrynStorage(target).then(async () => {
                            setCleanupConfirmation(null);
                            await refreshOrynSettings();
                          });
                        }}
                      >
                        {cleanupConfirmation === target
                          ? `Confirm clear ${target}`
                          : `Clear ${target}`}
                      </Button>
                    ))}
                  </div>
                </section>
              )}
              {settingsSection === 'developer' && (
                <section className="ai-activity-settings">
                  <h3>Developer</h3>
                  {(
                    [
                      ['showDiagnostics', 'Show runtime diagnostics'],
                      ['showResolutionDetails', 'Show model resolution details'],
                      ['showProtocolDiagnostics', 'Show protocol diagnostics'],
                    ] as const
                  ).map(([key, label]) => (
                    <label key={key}>
                      <span>{label}</span>
                      <input
                        type="checkbox"
                        checked={settings.developer[key]}
                        onChange={(event) =>
                          void window.browser
                            .updateOrynDeveloper({ [key]: event.target.checked })
                            .then(refreshOrynSettings)
                        }
                      />
                    </label>
                  ))}
                  <p>
                    Experimental model runners: Off. Diagnostics never enable unsafe model code.
                  </p>
                </section>
              )}
            </div>
          )}
        </div>
      </div>
    </Sheet>
  );
}

function ResourceControlCard({
  resources,
  pressure,
  activeJob,
  queued,
}: {
  resources: BrowserWindowState['localAIResources'];
  pressure: BrowserWindowState['memoryPressure'];
  activeJob?: LocalAIJobActivity['jobs'][number];
  queued: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const runtimeResidentBytes = measuredSum(
    resources?.processResidentBytes,
    resources?.workerResidentBytes,
  );
  const pressureTone =
    pressure === 'critical' ? 'danger' : pressure === 'high' ? 'warning' : 'success';
  return (
    <section className="resource-control-card" aria-label="Local AI resources">
      <header>
        <div>
          <span className="ui-eyebrow">Resources</span>
          <strong>System memory for Local AI</strong>
        </div>
        <Badge tone={resources ? pressureTone : 'neutral'}>
          {resources ? `${titleCase(pressure)} pressure` : 'Unavailable'}
        </Badge>
        <Button size="sm" variant="ghost" onClick={() => setExpanded((value) => !value)}>
          {expanded ? 'Less' : 'Details'}
        </Button>
      </header>
      <div className="resource-summary-grid">
        <ResourceValue label="System memory" value={resources?.systemTotalBytes} />
        <ResourceValue label="Available now" value={resources?.systemAvailableBytes} />
        <ResourceValue label="Browser core" value={resources?.browserPrivateBytes} />
        <ResourceValue label="Oryn AI runtime" value={runtimeResidentBytes} />
      </div>
      {expanded && (
        <div className="resource-details-grid">
          <ResourceValue
            label="Resident models"
            value={resources?.estimatedResidentModelBytes}
            provenance="Estimated peak residency"
          />
          <ResourceValue
            label="Engine supervisor"
            value={resources?.processResidentBytes}
            provenance="Measured"
          />
          <ResourceValue
            label="Runtime workers"
            value={resources?.workerResidentBytes}
            provenance="Measured"
          />
          <ResourceValue
            label="Memory budget"
            value={resources?.memoryBudgetBytes}
            provenance="Configured"
          />
          <div className="resource-value">
            <span>Tabs and helpers</span>
            <strong>Unavailable</strong>
            <small>No reliable non-overlapping macOS total</small>
          </div>
          <div className="resource-value">
            <span>Queued jobs</span>
            <strong>{queued}</strong>
            <small>Measured</small>
          </div>
          <div className="resource-value">
            <span>Browser CPU</span>
            <strong>
              {resources?.cpuPercent === undefined
                ? 'Unavailable'
                : `${resources.cpuPercent.toFixed(1)}%`}
            </strong>
            <small>
              {resources?.cpuPercent === undefined ? 'Unavailable' : 'Measured · current sample'}
            </small>
          </div>
          <div className="resource-value">
            <span>Engine CPU</span>
            <strong>
              {resources?.engineCPUPercent === undefined
                ? 'Unavailable'
                : `${resources.engineCPUPercent.toFixed(1)}%`}
            </strong>
            <small>
              {resources?.engineCPUPercent === undefined
                ? 'Unavailable'
                : `Measured · ${resources.sampleWindowMs ?? 'unknown'} ms sample`}
            </small>
          </div>
          <div className="resource-value">
            <span>Accelerator</span>
            <strong>{resources?.acceleratorBackend?.toUpperCase() ?? 'Unavailable'}</strong>
            <small>
              {resources?.acceleratorReliable
                ? 'Measured allocation available'
                : 'Utilization unavailable'}
            </small>
          </div>
          <div className="resource-value wide">
            <span>Active execution</span>
            <strong>
              {resources?.activeExecution?.model ?? activeJob?.resolvedModel ?? 'None'}
            </strong>
            <small>
              {resources?.activeExecution
                ? `${resources.activeExecution.task ?? 'task'} · Engine-reported`
                : activeJob
                  ? `${activeJob.role} · ${activeJob.backend} · measured job state`
                  : 'No running job'}
            </small>
          </div>
          <div className="resource-value">
            <span>Throughput</span>
            <strong>
              {resources?.throughput
                ? `${resources.throughput.value.toFixed(1)} ${resources.throughput.unit}`
                : 'Unavailable'}
            </strong>
            <small>
              {resources?.throughput
                ? `Measured · ${resources.throughput.sampleWindowMs} ms sample`
                : 'This runtime did not report a task-native sample'}
            </small>
          </div>
        </div>
      )}
    </section>
  );
}

function measuredSum(...values: Array<number | undefined>): number | undefined {
  const measured = values.filter((value): value is number => value !== undefined);
  return measured.length === 0 ? undefined : measured.reduce((sum, value) => sum + value, 0);
}

function isMacOS(): boolean {
  return typeof navigator !== 'undefined' && navigator.platform.toLowerCase().includes('mac');
}

function tabSearchShortcutLabel(): string {
  return isMacOS() ? '⌘ Shift A' : 'Ctrl Shift A';
}

function PopoverDismissLayer({ onDismiss }: { onDismiss: () => void }) {
  return <div className="popover-dismiss-layer" aria-hidden="true" onPointerDown={onDismiss} />;
}

function ResourceValue({
  label,
  value,
  provenance,
}: {
  label: string;
  value?: number;
  provenance?: string;
}) {
  return (
    <div className="resource-value">
      <span>{label}</span>
      <strong>{value === undefined ? 'Unavailable' : formatResourceMemory(value)}</strong>
      {provenance && <small>{value === undefined ? 'Unavailable' : provenance}</small>}
    </div>
  );
}

function tabAccessibleName(tab: BrowserWindowState['tabs'][number]) {
  return [
    tab.title || 'New tab',
    tab.audible ? 'Audio' : '',
    tab.muted ? 'Muted' : '',
    tab.captureAccess?.length ? `Capture access: ${tab.captureAccess.join(', ')}` : '',
  ]
    .filter(Boolean)
    .join(' · ');
}

function TabButton({
  tab,
  active,
  dragging,
  openMenu,
  startDrag,
  moveDrag,
  finishDrag,
  cancelDrag,
  activate,
}: {
  tab: BrowserWindowState['tabs'][number];
  active: boolean;
  dragging: boolean;
  openMenu: (event: MouseEvent<HTMLElement>) => void;
  startDrag: (event: ReactPointerEvent<HTMLElement>) => void;
  moveDrag: (event: ReactPointerEvent<HTMLElement>) => void;
  finishDrag: (event: ReactPointerEvent<HTMLElement>) => void;
  cancelDrag: (event: ReactPointerEvent<HTMLElement>) => void;
  activate: () => void;
}) {
  return (
    <div
      className={`tab ${tab.islandId ? '' : 'standalone'} ${active ? 'active' : ''} ${dragging ? 'dragging' : ''}`}
      data-tab-id={tab.tabId}
      onContextMenu={openMenu}
    >
      <button
        className="tab-activate"
        role="tab"
        aria-label={tabAccessibleName(tab)}
        aria-selected={active}
        title={tab.title || tab.url}
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={finishDrag}
        onPointerCancel={cancelDrag}
        onClick={activate}
      >
        <TabFavicon tab={tab} />
        {Boolean(tab.captureAccess?.length) && (
          <span
            className="capture-access-indicator"
            aria-label={`Capture access: ${tab.captureAccess!.join(', ')}`}
            title={`This page can use ${tab.captureAccess!.join(', ')}`}
          >
            ●
          </span>
        )}
        <span className="tab-title">
          {tab.title || 'New tab'}
          {tab.audible ? ' · Audio' : ''}
          {tab.muted ? ' · Muted' : ''}
        </span>
      </button>
      <IconButton
        className="tab-close"
        icon="close"
        label={`Close ${tab.title || 'tab'}`}
        onClick={() => void window.browser.closeTab(tab.tabId)}
      />
    </div>
  );
}

function TabFavicon({ tab }: { tab: BrowserWindowState['tabs'][number] }) {
  if (tab.favicon) return <img className="tab-favicon-image" src={tab.favicon} alt="" />;
  return (
    <Icon className="tab-favicon-fallback" name={isOrynNewTabURL(tab.url) ? 'grid' : 'globe'} />
  );
}

function IslandColorDialog({
  ref,
  island,
  close,
}: {
  ref: RefObject<HTMLElement | null>;
  island: BrowserWindowState['experience']['islands'][number] | undefined;
  close: () => void;
}) {
  if (!island) return null;
  const currentColor = island.accent.toLowerCase();
  return (
    <div
      className="island-color-backdrop"
      data-trusted-overlay="true"
      role="presentation"
      onPointerDown={close}
    >
      <section
        ref={ref}
        className="island-color-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="island-color-title"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header>
          <span
            className="island-color-preview"
            style={{ background: island.accent, color: island.accent }}
          />
          <div>
            <h2 id="island-color-title">Island color</h2>
            <p>{island.name}</p>
          </div>
          <IconButton icon="close" label="Close color picker" onClick={close} />
        </header>
        <div className="island-color-grid" role="radiogroup" aria-label="Available colors">
          {TAB_ISLAND_COLORS.map((color) => {
            const selected = color.value === currentColor;
            return (
              <button
                key={color.value}
                type="button"
                className={selected ? 'selected' : ''}
                role="radio"
                aria-checked={selected}
                autoFocus={selected}
                onClick={() => {
                  void window.browser.updateBrowserExperience({
                    type: 'island-update',
                    islandId: island.islandId,
                    accent: color.value,
                  });
                  close();
                }}
              >
                <span className="island-color-swatch" style={{ background: color.value }} />
                <span>{color.name}</span>
                {selected && <Icon name="check" />}
              </button>
            );
          })}
        </div>
      </section>
    </div>
  );
}

function WorkspaceMenu({ state, close }: { state: BrowserWindowState; close: () => void }) {
  const activeWorkspace = state.experience.workspaces.find(
    (item) => item.workspaceId === state.experience.activeWorkspaceId,
  );
  return (
    <div className="workspace-menu" data-trusted-overlay="true" role="menu">
      <strong>Workspaces</strong>
      {state.experience.workspaces.map((workspace) => (
        <button
          key={workspace.workspaceId}
          type="button"
          role="menuitemradio"
          aria-checked={workspace.workspaceId === state.experience.activeWorkspaceId}
          onClick={() => {
            void window.browser.updateBrowserExperience({
              type: 'workspace-switch',
              workspaceId: workspace.workspaceId,
            });
            close();
          }}
        >
          <span className="workspace-dot" style={{ background: workspace.accent }} />
          <span>{workspace.name}</span>
          <small>
            {state.tabs.filter((tab) => tab.workspaceId === workspace.workspaceId).length}
          </small>
        </button>
      ))}
      <div className="workspace-menu-actions">
        <button
          type="button"
          onClick={() => {
            void window.browser.updateBrowserExperience({
              type: 'workspace-create',
              name: nextWorkspaceName(state.experience.workspaces),
            });
            close();
          }}
        >
          + New
        </button>
        <button
          type="button"
          disabled={!activeWorkspace}
          onClick={async () => {
            if (!activeWorkspace) return;
            const name = await requestText('Rename workspace', activeWorkspace.name);
            if (name)
              void window.browser.updateBrowserExperience({
                type: 'workspace-rename',
                workspaceId: activeWorkspace.workspaceId,
                name,
              });
            close();
          }}
        >
          Rename
        </button>
        <button
          type="button"
          disabled={!activeWorkspace || activeWorkspace.workspaceId === 'workspace_default'}
          onClick={async () => {
            if (!activeWorkspace || activeWorkspace.workspaceId === 'workspace_default') return;
            const action = await requestChoice('Delete workspace?', [
              { label: 'Move tabs to Main and delete', value: 'move-to-default' },
              { label: 'Close tabs and delete', value: 'close' },
            ]);
            if (!action) return;
            void window.browser.updateBrowserExperience({
              type: 'workspace-delete',
              workspaceId: activeWorkspace.workspaceId,
              tabs: action as 'close' | 'move-to-default',
            });
            close();
          }}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

function TabContextMenu({
  menu,
  state,
  close,
}: {
  menu: TabMenuState;
  state: BrowserWindowState;
  close: () => void;
}) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ left: menu.x, top: menu.y });
  const { tabId } = menu;
  const tab = state.tabs.find((item) => item.tabId === tabId);
  useLayoutEffect(() => {
    const element = menuRef.current;
    if (!element) return;
    const gutter = 8;
    const bounds = element.getBoundingClientRect();
    setPosition({
      left: Math.max(gutter, Math.min(menu.x, window.innerWidth - bounds.width - gutter)),
      top: Math.max(gutter, Math.min(menu.y, window.innerHeight - bounds.height - gutter)),
    });
  }, [menu.x, menu.y]);
  if (!tab) return null;
  const run = (command: Parameters<typeof window.browser.updateBrowserExperience>[0]) => {
    void window.browser.updateBrowserExperience(command);
    close();
  };
  return (
    <div
      ref={menuRef}
      className="tab-context-menu"
      data-trusted-overlay="true"
      role="menu"
      style={{ left: position.left, top: position.top }}
    >
      <button disabled={state.privateMode} onClick={() => run({ type: 'tab-new-window', tabId })}>
        Move tab to new window
      </button>
      {/^https?:/u.test(tab.url) && (
        <button
          onClick={() => {
            void window.browser.newPrivateWindow(tab.url);
            close();
          }}
        >
          Open page in new private window
        </button>
      )}
      <button onClick={() => run({ type: 'tab-mute', tabId, muted: !tab.muted })}>
        {tab.muted ? 'Unmute tab' : 'Mute tab'}
      </button>
      <button onClick={() => run({ type: 'tab-reorder', tabId })}>Move tab to end</button>
      <label>
        Move tab before
        <select
          aria-label="Move tab before"
          defaultValue=""
          onChange={(event) => {
            if (event.target.value)
              run({ type: 'tab-reorder', tabId, beforeTabId: event.target.value });
          }}
        >
          <option value="">Choose a tab</option>
          {state.tabs
            .filter((item) => item.workspaceId === tab.workspaceId && item.tabId !== tabId)
            .map((item) => (
              <option key={item.tabId} value={item.tabId}>
                {item.title || item.url}
              </option>
            ))}
        </select>
      </label>
      <button onClick={() => run({ type: 'tab-pin', tabId, pinned: !tab.pinned })}>
        {tab.pinned ? 'Unpin tab' : 'Pin tab'}
      </button>
      <button onClick={() => run({ type: 'tab-duplicate', tabId })}>Duplicate tab</button>
      <button onClick={() => run({ type: 'island-create', tabIds: [tabId] })}>
        Create tab island
      </button>
      <button onClick={() => run({ type: 'tab-close-others', tabId })}>Close other tabs</button>
      <button onClick={() => run({ type: 'tab-close-right', tabId })}>
        Close tabs to the right
      </button>
      <button onClick={() => run({ type: 'tab-close-duplicates' })}>Close duplicate tabs</button>
      <span className="menu-label">Move to workspace</span>
      {state.experience.workspaces
        .filter((workspace) => workspace.workspaceId !== tab.workspaceId)
        .map((workspace) => (
          <button
            key={workspace.workspaceId}
            onClick={() =>
              run({ type: 'tab-move-workspace', tabId, workspaceId: workspace.workspaceId })
            }
          >
            {workspace.name}
          </button>
        ))}
    </div>
  );
}

function TabSearchOverlay({
  state,
  query,
  setQuery,
  close,
}: {
  state: BrowserWindowState;
  query: string;
  setQuery: (value: string) => void;
  close: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  useFocusTrap(true, dialogRef, close);
  const needle = query.trim().toLowerCase();
  const matches = state.tabs.filter(
    (tab) => !needle || `${tab.title} ${tab.url}`.toLowerCase().includes(needle),
  );
  const recentlyClosed = state.experience.recentlyClosed.filter(
    (tab) => !needle || `${tab.title ?? ''} ${tab.url}`.toLowerCase().includes(needle),
  );
  const workspaceName = (workspaceId?: string) =>
    state.experience.workspaces.find((item) => item.workspaceId === workspaceId)?.name ?? 'Main';
  const islandName = (islandId?: string) =>
    state.experience.islands.find((item) => item.islandId === islandId)?.name;
  return (
    <div className="tab-search-backdrop" data-trusted-overlay="true" onPointerDown={close}>
      <section
        ref={dialogRef}
        className="tab-search-overlay"
        role="dialog"
        aria-modal="true"
        aria-label="Search tabs"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <div className="tab-search-input">
          <Icon name="search" />
          <input
            value={query}
            placeholder="Search open and recently closed tabs"
            aria-label="Search tabs"
            onChange={(event) => setQuery(event.target.value)}
          />
          <kbd>{tabSearchShortcutLabel()}</kbd>
        </div>
        <div className="tab-search-results">
          <h3>Open tabs</h3>
          {matches.map((tab) => (
            <button
              key={tab.tabId}
              onClick={() => {
                void window.browser.activateTab(tab.tabId);
                close();
              }}
            >
              <span>
                <strong>{tab.title || 'New tab'}</strong>
                <small>{tab.url}</small>
              </span>
              <em>
                {workspaceName(tab.workspaceId)}
                {islandName(tab.islandId) ? ` · ${islandName(tab.islandId)}` : ''}
              </em>
            </button>
          ))}
          <h3>Recently closed</h3>
          {recentlyClosed.map((tab, index) => (
            <button
              key={`${tab.url}-${index}`}
              onClick={() => {
                void window.browser.newTab(tab.url);
                close();
              }}
            >
              <span>
                <strong>{tab.title || tab.url}</strong>
                <small>{tab.url}</small>
              </span>
              <em>{workspaceName(tab.workspaceId)}</em>
            </button>
          ))}
          {matches.length + recentlyClosed.length === 0 && <EmptyState text="No matching tabs" />}
        </div>
      </section>
    </div>
  );
}

function BrowserPanel({
  panel,
  state,
  data,
  historyQuery,
  setHistoryQuery,
  refreshData,
  appUpdate,
  setAppUpdate,
  close,
}: {
  panel: Exclude<Panel, null | 'local-ai'>;
  state: BrowserWindowState;
  data: BrowserCoreDataSnapshot;
  historyQuery: string;
  setHistoryQuery: (value: string) => void;
  refreshData: (query?: string) => Promise<void>;
  appUpdate: AppUpdateSnapshot;
  setAppUpdate: (value: AppUpdateSnapshot) => void;
  close: () => void;
}) {
  const title = {
    history: 'Browsing History',
    bookmarks: 'Bookmarks',
    downloads: 'Downloads',
    site: 'Site controls',
    settings: 'Browser settings',
  }[panel];
  const icon: Record<typeof panel, IconName> = {
    history: 'history',
    bookmarks: 'bookmark',
    downloads: 'download',
    site: 'shield',
    settings: 'settings',
  };
  const normalizedHistoryQuery = historyQuery.trim().toLocaleLowerCase();
  const visibleHistory = data.history.filter(
    (item) =>
      item.url.toLocaleLowerCase().includes(normalizedHistoryQuery) ||
      item.title.toLocaleLowerCase().includes(normalizedHistoryQuery),
  );
  return (
    <Sheet title={title} icon={icon[panel]} close={close}>
      <div className="browser-panel">
        {panel === 'history' && state.privateMode && (
          <p>
            Private browsing does not save browsing history. Existing normal-window history may
            appear in address-bar suggestions.
          </p>
        )}
        {panel === 'history' && !state.privateMode && (
          <>
            <div className="panel-actions">
              <input
                aria-label="Search history"
                placeholder="Search history"
                value={historyQuery}
                onChange={(event) => setHistoryQuery(event.target.value)}
              />
              <Button
                variant="danger"
                size="sm"
                onClick={() => {
                  void requestChoice('Clear Browsing History', [
                    { label: 'Last hour', value: 'hour' },
                    { label: 'Last day', value: 'day' },
                    { label: 'All time', value: 'all' },
                  ]).then(async (range) => {
                    if (range) {
                      await window.browser.clearHistory(range as 'hour' | 'day' | 'all');
                      await refreshData();
                    }
                  });
                }}
              >
                Clear history
              </Button>
            </div>
            <div className="item-list">
              {visibleHistory.map((item) => (
                <article key={item.historyId}>
                  <button
                    className="item-main"
                    onClick={() => void window.browser.navigate(item.url)}
                  >
                    <strong>{item.title}</strong>
                    <span>{item.url}</span>
                  </button>
                  <time>{new Date(item.visitedAt).toLocaleString()}</time>
                  <IconButton
                    icon="trash"
                    label={`Delete ${item.title}`}
                    onClick={() =>
                      void window.browser.deleteHistory(item.historyId).then(() => refreshData())
                    }
                  />
                </article>
              ))}
            </div>
          </>
        )}
        {panel === 'bookmarks' && <BookmarksManager data={data} refreshData={refreshData} />}
        {panel === 'downloads' && <DownloadsPanel data={data} />}
        {panel === 'site' && (
          <SiteControls key={data.siteContext?.documentId} data={data} refreshData={refreshData} />
        )}
        {panel === 'settings' && (
          <div className="settings-grid">
            <h3>Appearance</h3>
            <label>
              Theme
              <select
                aria-label="Browser theme"
                value={data.appearance.theme}
                onChange={(event) =>
                  void window.browser
                    .updateAppearance({
                      theme: event.target.value as 'light' | 'dark' | 'system',
                    })
                    .then(() => refreshData())
                }
              >
                <option value="system">System</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </label>
            <label>
              Density
              <select
                aria-label="Browser density"
                value={data.appearance.density}
                onChange={(event) =>
                  void window.browser
                    .updateAppearance({
                      density: event.target.value as 'default' | 'compact' | 'comfortable',
                    })
                    .then(() => refreshData())
                }
              >
                <option value="default">Default</option>
                <option value="compact">Compact</option>
                <option value="comfortable">Comfortable</option>
              </select>
            </label>
            <label>
              <input
                type="checkbox"
                checked={data.bookmarkBarVisible}
                onChange={(event) =>
                  void window.browser
                    .setBookmarkBarVisible(event.target.checked)
                    .then(() => refreshData())
                }
              />{' '}
              Show bookmarks bar
            </label>
            <h3>Diagnostics</h3>
            <button onClick={() => void window.browser.exportDiagnostics()}>
              Export browser diagnostics
            </button>
            <p>
              Exports build identity, versions, Engine state and coarse failure codes and counts.
              Recent failure information stays in memory. URLs, prompts, credentials, page content
              and profile paths are excluded. Review the file before sharing.
            </p>
            <h3>Default browser</h3>
            <button onClick={() => void window.browser.setDefaultBrowser()}>
              Make Oryn the default browser
            </button>
            <p>Available in the packaged macOS app; macOS asks you to confirm.</p>
            <h3>Search</h3>
            <label>
              Search provider
              <select
                value={data.appearance.searchProvider ?? 'google'}
                onChange={(event) =>
                  void window.browser
                    .updateAppearance({
                      searchProvider: event.target.value as 'google' | 'duckduckgo' | 'bing',
                    })
                    .then(() => refreshData())
                }
              >
                <option value="google">Google</option>
                <option value="duckduckgo">DuckDuckGo</option>
                <option value="bing">Bing</option>
              </select>
            </label>
            <label>
              <input
                type="checkbox"
                checked={data.appearance.localSuggestions ?? true}
                onChange={(event) =>
                  void window.browser
                    .updateAppearance({ localSuggestions: event.target.checked })
                    .then(() => refreshData())
                }
              />
              Suggest local history, bookmarks and open tabs
            </label>
            <p>
              Suggestions stay on this device. Searches are sent to your chosen provider when
              submitted.
            </p>
            {!state.privateMode && <BrowserImportSettings onImported={() => refreshData()} />}
            <h3>Privacy</h3>
            <p>Browsing History and Local AI Activity are stored separately.</p>
            <Button
              variant="danger"
              onClick={async () => {
                const category = await requestChoice(
                  'Clear ordinary browsing data for all sites, for all time. Sidebar apps and Local AI files are separate.',
                  [
                    { label: 'Cookies and site storage', value: 'cookies' },
                    { label: 'Cached files', value: 'cache' },
                    { label: 'Site permission decisions', value: 'permissions' },
                  ],
                );
                if (category) {
                  await window.browser.clearBrowsingData(
                    category as 'cookies' | 'cache' | 'permissions',
                  );
                  await refreshData();
                }
              }}
            >
              Clear browsing data…
            </Button>
            <Button
              variant="danger"
              disabled={!data.siteContext}
              onClick={() =>
                void window.browser
                  .clearCurrentSiteData(data.siteContext!)
                  .then(() => refreshData())
              }
            >
              Clear current site data
            </Button>
            <AppUpdateSettings update={appUpdate} setUpdate={setAppUpdate} />
            <h3>Sidebar</h3>
            <label>
              <input
                type="checkbox"
                checked={state.experience.sidebar.visible}
                onChange={(event) =>
                  void window.browser.updateBrowserExperience({
                    type: 'sidebar-toggle',
                    visible: event.target.checked,
                  })
                }
              />{' '}
              Show Oryn sidebar
            </label>
            <h3>Tabs</h3>
            <label>
              <input
                type="checkbox"
                checked={state.experience.autoGroupOpenerTabs}
                onChange={(event) =>
                  void window.browser.updateBrowserExperience({
                    type: 'tab-auto-group',
                    enabled: event.target.checked,
                  })
                }
              />{' '}
              Automatically group tabs opened by another tab
            </label>
            <p>
              Search Tabs shortcut <kbd>{tabSearchShortcutLabel()}</kbd>
            </p>
            {state.experience.sidebar.tools
              .slice()
              .sort((left, right) => left.order - right.order)
              .map((tool, index) => (
                <div className="sidebar-app-setting" key={tool.toolId}>
                  <label>
                    <input
                      type="checkbox"
                      checked={tool.enabled}
                      onChange={(event) =>
                        void window.browser.updateBrowserExperience({
                          type: 'sidebar-tool-enable',
                          toolId: tool.toolId,
                          enabled: event.target.checked,
                        })
                      }
                    />{' '}
                    {tool.name}
                  </label>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={index === 0}
                    onClick={() =>
                      void window.browser.updateBrowserExperience({
                        type: 'sidebar-tool-order',
                        toolId: tool.toolId,
                        order: index - 1,
                      })
                    }
                  >
                    Move up
                  </Button>
                </div>
              ))}
            {state.experience.sidebar.apps
              .slice()
              .sort((left, right) => left.order - right.order)
              .map((app, index) => (
                <div className="sidebar-app-setting" key={app.appId}>
                  <label>
                    <input
                      type="checkbox"
                      checked={app.enabled}
                      onChange={(event) =>
                        void window.browser.updateBrowserExperience({
                          type: 'sidebar-app-enable',
                          appId: app.appId,
                          enabled: event.target.checked,
                        })
                      }
                    />{' '}
                    {app.name}
                  </label>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={index === 0}
                    onClick={() =>
                      void window.browser.updateBrowserExperience({
                        type: 'sidebar-app-order',
                        appId: app.appId,
                        order: index - 1,
                      })
                    }
                  >
                    Move up
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      if (
                        window.confirm(
                          `Clear ${app.name} cookies, cache, and local storage from its isolated sidebar profile?`,
                        )
                      ) {
                        void window.browser.updateBrowserExperience({
                          type: 'sidebar-app-clear-data',
                          appId: app.appId,
                        });
                      }
                    }}
                  >
                    Clear data
                  </Button>
                </div>
              ))}
            <h3>Developer</h3>
            <Button icon="cpu" onClick={() => void window.browser.openDevTools()}>
              Open Developer Tools
            </Button>
          </div>
        )}
      </div>
    </Sheet>
  );
}

function AppUpdateSettings({
  update,
  setUpdate,
}: {
  update: AppUpdateSnapshot;
  setUpdate: (value: AppUpdateSnapshot) => void;
}) {
  const candidate = update.candidate;
  const apply = (operation: Promise<AppUpdateSnapshot>) =>
    operation.then(setUpdate).catch((error: unknown) => {
      window.alert(error instanceof Error ? error.message : 'The update action failed.');
    });
  return (
    <section className="app-update-settings" aria-label="Application updates">
      <h3>Application updates</h3>
      <p>
        Installed version {update.currentVersion || 'development build'} · {update.channel} channel
      </p>
      <label>
        Update channel
        <select
          aria-label="Application update channel"
          value={update.channel}
          disabled={!update.enabled || ['downloading', 'installing'].includes(update.status)}
          onChange={(event) =>
            void apply(
              window.browser.updateAppUpdateSettings({
                channel: event.target.value as 'stable' | 'beta',
              }),
            )
          }
        >
          <option value="stable">Stable — tested releases</option>
          <option value="beta">Beta — newest builds immediately</option>
        </select>
      </label>
      <label>
        <input
          type="checkbox"
          checked={update.automaticChecks}
          disabled={!update.enabled || update.status === 'installing'}
          onChange={(event) =>
            void apply(
              window.browser.updateAppUpdateSettings({ automaticChecks: event.target.checked }),
            )
          }
        />{' '}
        Check automatically (downloads still require approval)
      </label>
      {update.lastCheckAt && <p>Last checked {new Date(update.lastCheckAt).toLocaleString()}</p>}
      {!update.enabled && <p>{update.unavailableReason}</p>}
      {candidate && (
        <div className="app-update-card">
          <strong>
            {candidate.releaseName || `Oryn ${candidate.version}`} ·{' '}
            {formatBytes(candidate.sizeBytes)}
          </strong>
          <span>Temporary free space required: {formatBytes(candidate.requiredFreeBytes)}</span>
          {candidate.releaseDate && (
            <span>{new Date(candidate.releaseDate).toLocaleDateString()}</span>
          )}
          {candidate.releaseNotes && <p className="app-update-notes">{candidate.releaseNotes}</p>}
          {update.progress && (
            <>
              <ProgressBar
                label="Application update download"
                value={update.progress.percent}
                max={100}
              />
              <p>
                {formatBytes(update.progress.transferredBytes)} /{' '}
                {formatBytes(update.progress.totalBytes)} ({Math.round(update.progress.percent)}%)
              </p>
            </>
          )}
          {update.restartBlockingReason && (
            <p role="alert">Restart paused: {update.restartBlockingReason}</p>
          )}
          <div className="app-update-actions">
            {['available', 'deferred', 'cancelled', 'error'].includes(update.status) &&
              candidate.sizeBytes > 0 &&
              candidate.unpackedSizeBytes > 0 && (
                <Button
                  disabled={!candidate.approvalId}
                  variant="primary"
                  onClick={() =>
                    void apply(
                      window.browser.downloadAppUpdate(
                        candidate.version,
                        false,
                        candidate.approvalId,
                      ),
                    )
                  }
                >
                  {['cancelled', 'error'].includes(update.status)
                    ? 'Retry download'
                    : 'Download update'}
                </Button>
              )}
            {update.status === 'skipped' && (
              <>
                <Button
                  disabled={!candidate.approvalId}
                  variant="primary"
                  onClick={() =>
                    void apply(
                      window.browser.downloadAppUpdate(
                        candidate.version,
                        true,
                        candidate.approvalId,
                      ),
                    )
                  }
                >
                  Download anyway
                </Button>
                <Button onClick={() => void apply(window.browser.clearAppUpdateSkip())}>
                  Clear skip
                </Button>
              </>
            )}
            {update.status === 'downloading' && (
              <Button onClick={() => void apply(window.browser.cancelAppUpdateDownload())}>
                Cancel download
              </Button>
            )}
            {update.status === 'ready' && (
              <Button
                disabled={!candidate.approvalId}
                variant="primary"
                onClick={() =>
                  void apply(
                    window.browser.restartToInstallAppUpdate(
                      candidate.version,
                      candidate.approvalId,
                    ),
                  )
                }
              >
                Restart and install
              </Button>
            )}
          </div>
        </div>
      )}
      {update.error && (
        <p role="alert" className="app-update-error">
          {update.error.message}
          {update.error.requiredDiskBytes !== undefined &&
          update.error.availableDiskBytes !== undefined
            ? ` Required ${formatBytes(update.error.requiredDiskBytes)}; available ${formatBytes(update.error.availableDiskBytes)}.`
            : ''}
        </p>
      )}
      {update.status === 'installing' && (
        <p role="status">Preparing and verifying the update before restarting…</p>
      )}
      {update.status === 'downloading' && (
        <p role="status">
          Downloading and checking the update. Large releases may take a few minutes to check.
        </p>
      )}
      {update.channel === 'stable' && update.currentVersion.includes('-beta.') && (
        <p>
          You will stay on this version until a newer stable release is available. Switching
          channels never downgrades Oryn.
        </p>
      )}
      <Button
        disabled={
          !update.enabled ||
          ['checking', 'downloading', 'ready', 'installing'].includes(update.status)
        }
        onClick={() => void apply(window.browser.checkForAppUpdate())}
      >
        {update.status === 'checking' ? 'Checking…' : 'Check for updates'}
      </Button>
      <p className="app-update-privacy">
        Oryn uses a random identifier stored only on this device to determine stable rollout
        eligibility. It is never sent to the update server. Checking reveals the normal network
        information required for an HTTPS request, such as your IP address.
      </p>
    </section>
  );
}

function AppUpdateDialog({
  dialogRef,
  update,
  setUpdate,
  later,
  openSettings,
}: {
  dialogRef: RefObject<HTMLElement | null>;
  update: AppUpdateSnapshot;
  setUpdate: (value: AppUpdateSnapshot) => void;
  later: () => void;
  openSettings: () => void;
}) {
  const candidate = update.candidate!;
  const apply = (operation: Promise<AppUpdateSnapshot>) =>
    operation.then(setUpdate).catch((error: unknown) => {
      window.alert(error instanceof Error ? error.message : 'The update action failed.');
    });
  const ready = update.status === 'ready';
  return (
    <div className="local-ai-dialog-backdrop" data-trusted-overlay="true">
      <section
        ref={dialogRef}
        className="hf-install-dialog app-update-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={ready ? 'Application update ready' : 'Application update available'}
        tabIndex={-1}
      >
        <header>
          <div>
            <span className="ui-eyebrow">Application update</span>
            <h2>{ready ? 'Ready to install' : 'A new Oryn version is available'}</h2>
          </div>
        </header>
        <dl className="hf-install-facts">
          <div>
            <dt>Version</dt>
            <dd>{candidate.releaseName || candidate.version}</dd>
          </div>
          {candidate.bundleVersion && (
            <div>
              <dt>Build</dt>
              <dd>{candidate.bundleVersion}</dd>
            </div>
          )}
          <div>
            <dt>Download size</dt>
            <dd>{formatBytes(candidate.sizeBytes)}</dd>
          </div>
          <div>
            <dt>Temporary free space</dt>
            <dd>{formatBytes(candidate.requiredFreeBytes)}</dd>
          </div>
        </dl>
        {candidate.releaseNotes && <p className="app-update-notes">{candidate.releaseNotes}</p>}
        {update.restartBlockingReason && <p role="alert">{update.restartBlockingReason}</p>}
        <p className="hf-security-note">
          {ready
            ? 'Oryn will restart only when you choose. Active Local AI work, downloads, unsaved artifacts, or a page that blocks closing will postpone installation.'
            : 'Nothing is downloaded until you approve this version. You can be reminded later or skip only this version.'}
        </p>
        <footer>
          <Button
            variant="ghost"
            onClick={() => {
              openSettings();
              if (ready) later();
              else void apply(window.browser.deferAppUpdate(candidate.version));
            }}
          >
            Update settings
          </Button>
          <span className="hf-dialog-spacer" />
          {ready ? (
            <>
              <Button onClick={later}>Later</Button>
              <Button
                disabled={!candidate.approvalId}
                variant="primary"
                onClick={() =>
                  void apply(
                    window.browser.restartToInstallAppUpdate(
                      candidate.version,
                      candidate.approvalId,
                    ),
                  )
                }
              >
                Restart and install
              </Button>
            </>
          ) : (
            <>
              <Button onClick={() => void apply(window.browser.skipAppUpdate(candidate.version))}>
                Skip this version
              </Button>
              <Button onClick={() => void apply(window.browser.deferAppUpdate(candidate.version))}>
                Remind me later
              </Button>
              <Button
                disabled={!candidate.approvalId}
                variant="primary"
                onClick={() =>
                  void apply(
                    window.browser.downloadAppUpdate(
                      candidate.version,
                      false,
                      candidate.approvalId,
                    ),
                  )
                }
              >
                Download update
              </Button>
            </>
          )}
        </footer>
      </section>
    </div>
  );
}

function BookmarksManager({
  data,
  refreshData,
}: {
  data: BrowserCoreDataSnapshot;
  refreshData: () => Promise<void>;
}) {
  const [query, setQuery] = useState('');
  const [showTrash, setShowTrash] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [currentFolderId, setCurrentFolderId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');
  const visible = data.bookmarks.filter((item) => {
    if (showTrash !== Boolean(item.trashedAt)) return false;
    const needle = query.trim().toLowerCase();
    if (needle)
      return `${item.title} ${item.kind === 'bookmark' ? item.url : ''}`
        .toLowerCase()
        .includes(needle);
    return showTrash || item.parentId === currentFolderId;
  });
  const folders = data.bookmarks.filter((item) => item.kind === 'folder' && !item.trashedAt);
  const mutate = async (command: Parameters<typeof window.browser.updateBookmarks>[0]) => {
    await window.browser.updateBookmarks(command);
    setSelected([]);
    await refreshData();
  };
  const replaceThumbnail = async (bookmarkId: string, file?: File) => {
    if (!file) return;
    try {
      const thumbnail = await createSafeBookmarkThumbnail(file);
      await mutate({ type: 'thumbnail', bookmarkId, thumbnail });
    } catch (error) {
      window.alert(error instanceof Error ? error.message : 'This image could not be used.');
    }
  };
  return (
    <div className="bookmarks-manager">
      <div className="panel-actions bookmark-manager-toolbar">
        <input
          aria-label="Search bookmarks"
          placeholder="Search bookmarks"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <Button
          size="sm"
          variant={showTrash ? 'secondary' : 'ghost'}
          onClick={() => setShowTrash((value) => !value)}
        >
          {showTrash ? 'Back to bookmarks' : 'Trash'}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={async () => {
            const title = await requestText('Folder name', 'New folder');
            if (title) void mutate({ type: 'create-folder', title });
          }}
        >
          New folder
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={async () => {
            try {
              const preview = await window.browser.importBookmarks();
              if (!preview) return;
              const choice = await requestChoice(
                `Import ${preview.count} bookmarks and folders? Existing items are kept; duplicates are added.`,
                [{ label: 'Import', value: 'import' }],
              );
              if (choice) {
                await window.browser.importBookmarks(preview.token);
                await refreshData();
              }
            } catch {
              await requestChoice(
                'The bookmark file could not be imported. No items were changed.',
                [{ label: 'Close', value: 'close' }],
              );
            }
          }}
        >
          Import HTML
        </Button>
        <Button size="sm" variant="ghost" onClick={() => void window.browser.exportBookmarks()}>
          Export HTML
        </Button>
        <Button size="sm" variant="ghost" onClick={() => void mutate({ type: 'save-all-tabs' })}>
          Save all tabs
        </Button>
        <Button
          size="sm"
          variant={viewMode === 'list' ? 'secondary' : 'ghost'}
          onClick={() => setViewMode('list')}
        >
          List
        </Button>
        <Button
          size="sm"
          variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
          onClick={() => setViewMode('grid')}
        >
          Grid
        </Button>
      </div>
      {!showTrash && !query && currentFolderId && (
        <div className="bookmark-breadcrumbs">
          <Button
            size="sm"
            variant="ghost"
            icon="arrow-left"
            onClick={() => {
              const folder = data.bookmarks.find((item) => item.bookmarkId === currentFolderId);
              setCurrentFolderId(folder?.parentId ?? null);
            }}
          >
            Back
          </Button>
          <span>
            {data.bookmarks.find((item) => item.bookmarkId === currentFolderId)?.title ??
              'Bookmarks'}
          </span>
        </div>
      )}
      {selected.length > 0 && (
        <div className="bookmark-selection-bar">
          <strong>{selected.length} selected</strong>
          {!showTrash && (
            <>
              <select
                aria-label="Move selected bookmarks"
                defaultValue=""
                onChange={(event) => {
                  if (event.target.value)
                    void mutate({
                      type: 'move',
                      bookmarkIds: selected,
                      parentId: event.target.value === 'root' ? null : event.target.value,
                    });
                }}
              >
                <option value="">Move to…</option>
                <option value="root">Bookmarks root</option>
                {folders.map((folder) => (
                  <option key={folder.bookmarkId} value={folder.bookmarkId}>
                    {folder.title}
                  </option>
                ))}
              </select>
              <select
                aria-label="Copy selected bookmarks"
                defaultValue=""
                onChange={(event) => {
                  if (event.target.value)
                    void mutate({
                      type: 'move',
                      bookmarkIds: selected,
                      parentId: event.target.value === 'root' ? null : event.target.value,
                      copy: true,
                    });
                }}
              >
                <option value="">Copy to…</option>
                <option value="root">Bookmarks root</option>
                {folders.map((folder) => (
                  <option key={folder.bookmarkId} value={folder.bookmarkId}>
                    {folder.title}
                  </option>
                ))}
              </select>
              {selected.length > 1 && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={async () => {
                    const title = await requestText('Folder name', 'New folder');
                    if (title)
                      void mutate({ type: 'folder-from-items', bookmarkIds: selected, title });
                  }}
                >
                  Group in folder
                </Button>
              )}
              <Button
                size="sm"
                variant="danger"
                onClick={() => void mutate({ type: 'trash', bookmarkIds: selected })}
              >
                Move to trash
              </Button>
            </>
          )}
          {showTrash && (
            <>
              <Button
                size="sm"
                onClick={() => void mutate({ type: 'restore', bookmarkIds: selected })}
              >
                Restore
              </Button>
              <Button
                size="sm"
                variant="danger"
                onClick={() => {
                  if (window.confirm(`Permanently delete ${selected.length} selected item(s)?`)) {
                    void mutate({ type: 'delete-permanently', bookmarkIds: selected });
                  }
                }}
              >
                Delete permanently
              </Button>
            </>
          )}
        </div>
      )}
      <div className={`item-list bookmark-tree ${viewMode}`}>
        {visible.map((item) => (
          <article
            key={item.bookmarkId}
            draggable={!showTrash}
            onDragStart={(event) =>
              event.dataTransfer.setData('text/oryn-bookmark-id', item.bookmarkId)
            }
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              const dragged = event.dataTransfer.getData('text/oryn-bookmark-id');
              if (!dragged || dragged === item.bookmarkId) return;
              if (item.kind === 'folder') {
                void mutate({ type: 'move', bookmarkIds: [dragged], parentId: item.bookmarkId });
              } else {
                void mutate({ type: 'folder-from-items', bookmarkIds: [dragged, item.bookmarkId] });
              }
            }}
          >
            <input
              type="checkbox"
              aria-label={`Select ${item.title}`}
              checked={selected.includes(item.bookmarkId)}
              onChange={(event) =>
                setSelected((current) =>
                  event.target.checked
                    ? [...current, item.bookmarkId]
                    : current.filter((id) => id !== item.bookmarkId),
                )
              }
            />
            {item.kind === 'bookmark' && (
              <span
                className={`bookmark-artwork ${item.thumbnail ? 'has-image' : ''}`}
                style={item.thumbnail ? { backgroundImage: `url(${item.thumbnail})` } : undefined}
                aria-hidden="true"
              >
                {!item.thumbnail && item.title.slice(0, 1).toUpperCase()}
              </span>
            )}
            <button
              className="item-main"
              onClick={async () => {
                if (item.kind === 'bookmark') void window.browser.navigate(item.url);
                else {
                  setCurrentFolderId(item.bookmarkId);
                  setSelected([]);
                }
              }}
            >
              <strong>
                {item.kind === 'folder' ? 'Folder · ' : ''}
                {item.title}
              </strong>
              {item.kind === 'bookmark' && <span>{item.url}</span>}
            </button>
            {!showTrash && (
              <button
                aria-label={`Edit ${item.title}`}
                onClick={async () => {
                  const title = await requestText('Bookmark title', item.title);
                  if (!title) return;
                  const url =
                    item.kind === 'bookmark'
                      ? await requestText('Bookmark URL', item.url)
                      : undefined;
                  if (item.kind === 'folder' || url)
                    void mutate({
                      type: 'edit',
                      bookmarkId: item.bookmarkId,
                      title,
                      ...(url ? { url } : {}),
                    });
                }}
              >
                Edit
              </button>
            )}
            {!showTrash && item.kind === 'bookmark' && (
              <div className="bookmark-item-actions">
                <label className="bookmark-image-button">
                  {item.thumbnail ? 'Replace image' : 'Add image'}
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={(event) => {
                      void replaceThumbnail(item.bookmarkId, event.target.files?.[0]);
                      event.currentTarget.value = '';
                    }}
                  />
                </label>
                {item.thumbnail && (
                  <button
                    className="bookmark-image-button"
                    onClick={() => void mutate({ type: 'thumbnail', bookmarkId: item.bookmarkId })}
                  >
                    Remove image
                  </button>
                )}
                <button
                  className={`favorite-toggle ${item.favorite ? 'active' : ''}`}
                  aria-label={`${item.favorite ? 'Remove' : 'Add'} ${item.title} ${item.favorite ? 'from' : 'to'} Favorites`}
                  onClick={() =>
                    void mutate({
                      type: 'favorite',
                      bookmarkId: item.bookmarkId,
                      favorite: !item.favorite,
                    })
                  }
                >
                  <Icon name="star" fill={item.favorite ? 'currentColor' : 'none'} />
                </button>
              </div>
            )}
            {!showTrash && (
              <div className="bookmark-order-actions" aria-label={`Reorder ${item.title}`}>
                <button
                  aria-label={`Move ${item.title} up`}
                  onClick={() =>
                    void mutate({
                      type: 'reorder',
                      bookmarkId: item.bookmarkId,
                      order: item.order - 1,
                    })
                  }
                >
                  <Icon name="arrow-up" />
                </button>
                <button
                  aria-label={`Move ${item.title} down`}
                  onClick={() =>
                    void mutate({
                      type: 'reorder',
                      bookmarkId: item.bookmarkId,
                      order: item.order + 1,
                    })
                  }
                >
                  <Icon name="arrow-down" />
                </button>
              </div>
            )}
          </article>
        ))}
        {visible.length === 0 && (
          <EmptyState text={showTrash ? 'Trash is empty' : 'No bookmarks yet'} />
        )}
      </div>
    </div>
  );
}

async function createSafeBookmarkThumbnail(file: File): Promise<string> {
  if (
    !['image/png', 'image/jpeg', 'image/webp'].includes(file.type) ||
    file.size > 8 * 1024 * 1024
  ) {
    throw new Error('Choose a PNG, JPEG, or WebP image smaller than 8 MB.');
  }
  const bitmap = await createImageBitmap(file);
  try {
    if (bitmap.width > 8192 || bitmap.height > 8192) {
      throw new Error('The image dimensions are too large.');
    }
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('The image could not be processed.');
    const sourceSize = Math.min(bitmap.width, bitmap.height);
    const sourceX = (bitmap.width - sourceSize) / 2;
    const sourceY = (bitmap.height - sourceSize) / 2;
    context.drawImage(bitmap, sourceX, sourceY, sourceSize, sourceSize, 0, 0, 256, 256);
    return canvas.toDataURL('image/webp', 0.84);
  } finally {
    bitmap.close();
  }
}

function SiteControls({
  data,
  refreshData,
}: {
  data: BrowserCoreDataSnapshot;
  refreshData: () => Promise<void>;
}) {
  const [error, setError] = useState('');
  const permissions = [
    'camera',
    'microphone',
    'geolocation',
    'notifications',
    'clipboard-read',
    'fullscreen',
    'pointerLock',
    'popups',
    'sound',
  ];
  return (
    <div className="settings-grid">
      <p>
        Controls for <strong>{data.currentOrigin ?? 'this page'}</strong>
      </p>
      <p>
        These settings apply to ordinary tabs for this site. Sidebar apps have separate permissions
        and storage.
      </p>
      <ConnectionDetails connection={data.connection} context={data.certificateContext} />
      {Boolean(data.captureAccess?.length) && (
        <section aria-label="Pages with capture access">
          <h3>Camera, microphone and screen access</h3>
          <p>These pages have access. This does not mean a device is currently recording.</p>
          {data.captureAccess!.map((entry) => (
            <div key={entry.generation}>
              <p>
                {entry.topOrigin}
                {entry.origin !== entry.topOrigin ? ` embeds ${entry.origin}` : ''}:{' '}
                {entry.permissions.join(', ')}
                {entry.stopping ? ' — still has access until the page closes' : ''}
              </p>
              <Button
                variant="danger"
                onClick={() =>
                  void window.browser
                    .stopCapture(entry.webContentsId, entry.generation)
                    .then(() => refreshData())
                    .catch(() =>
                      setError('Capture could not be stopped. Close the page to end its access.'),
                    )
                }
              >
                Stop and close page
              </Button>
            </div>
          ))}
        </section>
      )}
      {Boolean(data.embeddedPermissions?.length) && (
        <section aria-label="Embedded sites">
          <h3>Embedded sites</h3>
          {data.embeddedPermissions!.map((entry, index) => (
            <p key={`${entry.embeddedOrigin}:${entry.permission}:${index}`}>
              {entry.embeddedOrigin}: {entry.permission} — {entry.decision}{' '}
              <Button
                variant="secondary"
                onClick={() =>
                  void window.browser
                    .revokeEmbeddedPermission(
                      entry.embeddedOrigin!,
                      entry.permission,
                      data.siteContext!,
                    )
                    .then(() => refreshData())
                }
              >
                Revoke
              </Button>
            </p>
          ))}
        </section>
      )}
      <p>
        JavaScript and cookies use the browser defaults; individual blocking controls are
        unavailable. Screen sharing requires the system source picker on macOS 15 or later. Camera
        and microphone access also depend on macOS permissions.
      </p>
      {error && <p role="alert">{error}</p>}
      {permissions.map((permission) => {
        const decision =
          data.sitePermissions.find((item) => item.permission === permission)?.decision ??
          (['popups', 'sound'].includes(permission) ? 'allow' : 'ask');
        return (
          <label key={permission}>
            {permission}
            <select
              value={decision}
              disabled={!data.siteContext}
              onChange={(event) =>
                void window.browser
                  .setSitePermission(
                    permission,
                    event.target.value as 'allow' | 'deny' | 'ask',
                    data.siteContext!,
                  )
                  .then(() => refreshData())
                  .catch(() =>
                    setError('This page changed. Reopen Site controls to review its permissions.'),
                  )
              }
            >
              {!['popups', 'sound'].includes(permission) && <option value="ask">Ask</option>}
              <option value="deny">Block</option>
              <option value="allow">Allow</option>
            </select>
          </label>
        );
      })}
      <Button
        variant="danger"
        disabled={!data.siteContext}
        onClick={() =>
          void window.browser
            .clearCurrentSiteData(data.siteContext!)
            .then(() => refreshData())
            .catch(() =>
              setError('Site data could not be cleared. Reopen Site controls and retry.'),
            )
        }
      >
        Clear cookies and site data
      </Button>
    </div>
  );
}

function connectionStateLabel(state: string | undefined): string {
  return (
    (
      {
        secure: 'Secure',
        'not-fully-secure': 'Not fully secure',
        'not-secure': 'Not secure',
        local: 'Local',
        'local-file': 'Local file',
        'oryn-page': 'Oryn page',
        unavailable: 'Unavailable',
      } as Record<string, string>
    )[state ?? 'unavailable'] ?? 'Unavailable'
  );
}

function ConnectionDetails({
  connection,
  context,
}: {
  connection: BrowserCoreDataSnapshot['connection'];
  context: BrowserCoreDataSnapshot['certificateContext'];
}) {
  const [showCertificates, setShowCertificates] = useState(false);
  return (
    <section aria-label="Connection details">
      <h3>Connection: {connectionStateLabel(connection?.state)}</h3>
      {connection?.reason && <p>{connection.reason}</p>}
      {connection?.tls && (
        <p>
          {connection.tls.protocol} · {connection.tls.cipher} ·{' '}
          {connection.tls.keyExchange ?? 'Key exchange unavailable'} · Certificate Transparency:{' '}
          {connection.tls.certificateTransparency ?? 'unavailable'}
        </p>
      )}
      {connection?.issues.map((issue, index) => (
        <p key={`${issue.url}:${index}`}>
          {issue.resolution === 'blocked'
            ? 'Some content on this page was blocked'
            : issue.resolution === 'upgraded'
              ? 'Content was upgraded to HTTPS'
              : 'Insecure content was reported'}
          : {issue.resourceType} — {issue.url}
        </p>
      ))}
      {Boolean(connection?.certificates.length) && (
        <>
          <Button variant="secondary" onClick={() => setShowCertificates((value) => !value)}>
            {showCertificates ? 'Hide certificates' : 'View certificates'}
          </Button>
          {showCertificates && (
            <div role="region" aria-label="Certificate viewer">
              <p>
                Read-only certificate details.{' '}
                {connection!.chainIncludesRoot
                  ? 'The chain includes its root.'
                  : 'The server did not supply a complete root chain.'}
              </p>
              {connection!.certificates.map((certificate, index) => (
                <details key={certificate.fingerprint256} open={index === 0}>
                  <summary>
                    {index === 0 ? 'Website certificate' : `Issuer ${index}`}: {certificate.subject}
                  </summary>
                  <dl>
                    {Object.entries(certificate)
                      .filter(([key]) => key !== 'pem')
                      .map(([key, value]) => (
                        <div key={key}>
                          <dt>{key.replace(/([A-Z])/g, ' $1')}</dt>
                          <dd>
                            {Array.isArray(value)
                              ? value.join(', ')
                              : String(value ?? 'Unavailable')}
                          </dd>
                        </div>
                      ))}
                  </dl>
                </details>
              ))}
              {connection!.tls?.signedCertificateTimestamps.map((stamp, index) => (
                <p key={index}>
                  Certificate timestamp: {stamp.log} · {new Date(stamp.timestamp).toISOString()} ·{' '}
                  {stamp.signatureAlgorithm}
                </p>
              ))}
              <Button
                variant="secondary"
                disabled={!context}
                onClick={() => {
                  if (context) void window.browser.exportCertificate(context);
                }}
              >
                Export certificate chain…
              </Button>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function EmptyState({ text }: { text: string }) {
  return <UIEmptyState title={text} />;
}

export function huggingFacePageActionLabel(
  action: NonNullable<BrowserWindowState['huggingFacePageAction']>,
): string {
  const labels = {
    inspecting: 'Inspecting…',
    loadable: 'Load into Oryn',
    unverified: 'Trust required',
    installed: 'Installed · Open in Oryn',
    'access-required': 'Access required',
    unsupported: 'Unsupported',
    blocked: 'Blocked by security policy',
    'resource-error': 'Resources unavailable',
    failed: action.operationId ? 'Download failed' : 'Inspection failed',
  } as const;
  if (action.status === 'downloading') {
    const percentage = downloadPercentage(action.downloadedBytes, action.totalBytes);
    if (percentage !== undefined) return `Downloading · ${percentage}%`;
    if ((action.downloadedBytes ?? 0) > 0)
      return `Downloading · ${formatBytes(action.downloadedBytes ?? 0)}`;
    return 'Starting download…';
  }
  return labels[action.status];
}

function DownloadDetails({
  action,
}: {
  action: NonNullable<BrowserWindowState['huggingFacePageAction']>;
}) {
  const downloaded = Math.max(0, action.downloadedBytes ?? 0);
  const total = action.totalBytes;
  const percentage = downloadPercentage(downloaded, total);
  const remaining = total === undefined ? undefined : Math.max(0, total - downloaded);
  return (
    <section className="hf-download-details" aria-label="Model download progress">
      <div className="hf-download-progress-heading" aria-live="polite">
        <strong>{percentage === undefined ? 'Downloading…' : `${percentage}% complete`}</strong>
        <span>
          {total === undefined
            ? `${formatBytes(downloaded)} downloaded`
            : `${formatBytes(downloaded)} of ${formatBytes(total)}`}
        </span>
      </div>
      {total !== undefined && total > 0 && (
        <ProgressBar
          label={`Download progress for ${action.repository}`}
          value={downloaded}
          max={total}
        />
      )}
      <dl>
        <div>
          <dt>Downloaded</dt>
          <dd>{formatBytes(downloaded)}</dd>
        </div>
        <div>
          <dt>Remaining</dt>
          <dd>{remaining === undefined ? 'Calculating…' : formatBytes(remaining)}</dd>
        </div>
        <div>
          <dt>Total download</dt>
          <dd>{total === undefined ? 'Calculating…' : formatBytes(total)}</dd>
        </div>
        <div>
          <dt>Expected local use</dt>
          <dd>
            {action.memoryBytes === undefined ? 'Unknown' : `~${formatBytes(action.memoryBytes)}`}
          </dd>
        </div>
        <div>
          <dt>Selected files</dt>
          <dd>{action.selectedFiles.length}</dd>
        </div>
      </dl>
      <p className="hf-download-cancel-note">
        You can close this panel or switch tabs while the download continues. Use Cancel download
        below only if you want to stop it.
      </p>
    </section>
  );
}

export function downloadPercentage(
  downloadedBytes: number | undefined,
  totalBytes: number | undefined,
): number | undefined {
  if (totalBytes === undefined || totalBytes <= 0) return undefined;
  return Math.min(100, Math.max(0, Math.floor(((downloadedBytes ?? 0) / totalBytes) * 100)));
}

export function formatResourceMemory(bytes: number): string {
  return formatBytes(bytes);
}

export function formatModelBytes(bytes: number): string {
  return formatResourceMemory(bytes);
}

function formatParameters(value: number | undefined): string {
  if (value === undefined) return 'Unavailable';
  if (value >= 1_000_000_000) return `~${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `~${(value / 1_000_000).toFixed(1)}M`;
  return value.toLocaleString();
}

function titleCase(value: string): string {
  return value
    .split('-')
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(' ');
}

function runtimeReason(value: string): string {
  const labels: Record<string, string> = {
    POLICY_BALANCED: 'Balanced for this device',
    POLICY_PERFORMANCE: 'Best predicted performance',
    POLICY_MEMORY_SAVER: 'Best memory margin',
    POLICY_QUALITY: 'Highest quality compatible variant',
    ALREADY_RESIDENT: 'Already loaded',
    ALREADY_INSTALLED: 'Already installed',
    CONFORMANCE_ACCEPTED: 'Compatibility evidence accepted',
  };
  return labels[value] ?? value.replace(/^BACKEND_/u, 'Accelerated by ').replaceAll('_', ' ');
}

function shortRevision(value: string): string {
  return value.length > 12 ? value.slice(0, 12) : value;
}

function modelOperationTitle(operation: TrustedModelOperation): string {
  const labels: Record<TrustedModelOperation['stage'], string> = {
    resolving: 'Resolving a compatible model…',
    'awaiting-permission': 'Waiting for your permission',
    'awaiting-model-choice': 'Waiting for a model choice',
    downloading: `Downloading model${
      operation.totalBytes === undefined
        ? '…'
        : ` · ${formatBytes(operation.downloadedBytes ?? 0)} / ${formatBytes(operation.totalBytes)}`
    }`,
    loading: 'Loading model into memory…',
    ready: 'Model ready',
    failed: 'Model preparation failed',
    cancelled: 'Model preparation cancelled',
  };
  return labels[operation.stage];
}

function modelOperationError(operation: TrustedModelOperation): string {
  const code = operation.errorCode;
  const errors: Record<string, string> = {
    MODEL_NOT_EXECUTABLE: 'This model cannot run on this device. Choose another model.',
    MODEL_SECURITY_BLOCKED:
      'This model is blocked because normal Local AI does not run arbitrary model code.',
    CUSTOM_CODE_REQUIRED:
      'This model requires custom executable code. Choose a model supported in normal mode.',
    DISK_SPACE_INSUFFICIENT: 'The model download needs more available disk space.',
    MODEL_ACCESS_RESTRICTED:
      'This model requires host approval and an Engine read token. Complete access from its Hugging Face page in Oryn, then connect the token there.',
    REQUEST_ABORTED: 'The model request was cancelled.',
    USER_DENIED:
      'The request was denied. Retry on the website. If no permission window appears, open Local AI → Settings → Sites and choose Ask again.',
  };
  if (code === 'MODEL_MEMORY_INSUFFICIENT') {
    const required = operation.requiredMemoryBytes;
    const available = operation.availableMemoryBytes;
    const model = operation.repository ? `${operation.repository} ` : 'This model ';
    if (required !== undefined && available !== undefined) {
      return `${model}needs about ${formatBytes(required)} of working memory; ${formatBytes(available)} is safely available right now. Close memory-heavy apps or choose a smaller compatible model, then retry on the website.`;
    }
    return `${model}does not fit safely in memory right now. Close memory-heavy apps or choose a smaller compatible model, then retry on the website.`;
  }
  return (
    errors[code ?? ''] ??
    `The model could not be prepared${code ? ` (${code})` : ''}. Return to the website to retry.`
  );
}

function candidateAvailabilityLabel(
  candidate: LocalAIPermissionPrompt['candidates'][number],
): string {
  if (candidate.blockingReasons.includes('MODEL_ACCESS_RESTRICTED')) {
    return 'Access restricted by model host · Set up access from its model page';
  }
  if (!candidate.securityAllowed) return 'Blocked by Local AI security policy · No override';
  if (!candidate.executable) return 'Cannot run on this device · No override';
  if (candidate.trustRequired) return 'Not Oryn verified · User approval required';
  return compatibilityLabel(candidate.compatibility, candidate.developerCertified);
}

function compatibilityLabel(value: string, developerCertified: boolean): string {
  if (developerCertified) return 'Developer tested';
  const labels: Record<string, string> = {
    certified: 'Developer tested',
    compatible: 'Compatible · Not developer-tested',
    uncertain: 'Compatibility uncertain · Not developer-tested',
    incompatible: 'Does not meet website requirements',
  };
  return labels[value] ?? value;
}

function useFocusTrap(
  active: boolean,
  ref: RefObject<HTMLElement | null>,
  onEscape?: () => void,
): void {
  const escape = useRef(onEscape);
  escape.current = onEscape;
  useEffect(() => {
    if (!active) return;
    const previous = document.activeElement as HTMLElement | null;
    const focusable = () =>
      Array.from(
        ref.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], summary, [contenteditable="true"], [tabindex]:not([tabindex="-1"])',
        ) ?? [],
      ).filter((element) => element.getClientRects().length > 0 && !element.closest('[inert]'));
    const frame = requestAnimationFrame(() => {
      if (!ref.current?.contains(document.activeElement)) focusable()[0]?.focus();
    });
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && escape.current) {
        event.preventDefault();
        event.stopPropagation();
        escape.current();
        return;
      }
      if (event.key !== 'Tab') return;
      const elements = focusable();
      const first = elements[0];
      const last = elements.at(-1);
      if (!first || !last) return;
      if (!ref.current?.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKey, true);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown', handleKey, true);
      if (previous?.isConnected) previous.focus();
    };
  }, [active, ref]);
}

function DownloadsPanel({ data }: { data: BrowserCoreDataSnapshot }) {
  const [directory, setDirectory] = useState(data.downloadDirectory);
  const [error, setError] = useState('');
  const act = (id: string, action: string) => {
    setError('');
    void window.browser
      .controlDownload(id, action)
      .catch((error: Error) => setError(error.message));
  };
  return (
    <div className="item-list" aria-label="Downloads">
      <p>Default folder: {directory ?? 'System Downloads folder'}</p>
      <button
        onClick={() =>
          void window.browser
            .chooseDownloadDirectory()
            .then(setDirectory)
            .catch((error: Error) => setError(error.message))
        }
      >
        Choose download folder
      </button>
      {error && <p role="alert">{error}</p>}
      {data.downloads.map((item) => (
        <article key={item.downloadId}>
          <div className="item-main">
            <strong>{item.filename}</strong>
            <span role="status">
              {item.state} · {formatBytes(item.receivedBytes)}
              {item.totalBytes > 0 ? ` / ${formatBytes(item.totalBytes)}` : ' · Size unknown'}
            </span>
            {item.state === 'completed' && item.openState !== 'ready' && (
              <span role={item.openState === 'error' ? 'alert' : 'status'}>
                {item.openState === 'pending'
                  ? 'Checking macOS quarantine…'
                  : item.openError === 'file-changed'
                    ? 'The downloaded file changed. Check it again before opening.'
                    : item.openError === 'file-unavailable'
                      ? 'The downloaded file is unavailable.'
                      : item.openError === 'open-failed'
                        ? 'The file could not be opened.'
                        : item.openError === 'quarantine'
                          ? 'macOS quarantine could not be verified. Retry the file check.'
                          : 'Check the downloaded file before opening.'}
              </span>
            )}
            {item.totalBytes > 0 && (
              <progress
                aria-label={`${item.filename} progress`}
                max={item.totalBytes}
                value={item.receivedBytes}
              />
            )}
            <div className="browser-dialog-actions">
              {['progressing', 'paused'].includes(item.state) ? (
                <>
                  <button
                    onClick={() =>
                      act(item.downloadId, item.state === 'paused' ? 'resume' : 'pause')
                    }
                  >
                    {item.state === 'paused' ? 'Resume' : 'Pause'}
                  </button>
                  <button onClick={() => act(item.downloadId, 'cancel')}>Cancel download</button>
                </>
              ) : (
                <>
                  {item.state === 'completed' ? (
                    <>
                      <button
                        disabled={item.openState !== 'ready'}
                        onClick={() => act(item.downloadId, 'open')}
                      >
                        Open file
                      </button>
                      {item.openState !== 'ready' && item.openState !== 'pending' && (
                        <button onClick={() => act(item.downloadId, 'check')}>
                          Retry file check
                        </button>
                      )}
                      <button onClick={() => act(item.downloadId, 'reveal')}>Show in Finder</button>
                    </>
                  ) : (
                    <button onClick={() => act(item.downloadId, 'retry')}>Retry download</button>
                  )}
                  <button onClick={() => act(item.downloadId, 'remove')}>Remove record</button>
                </>
              )}
            </div>
          </div>
        </article>
      ))}
      {!data.downloads.length && <EmptyState text="No downloads yet" />}
      <p>
        Removing a record keeps the downloaded file. Interrupted downloads may require a new
        transfer.
      </p>
    </div>
  );
}
