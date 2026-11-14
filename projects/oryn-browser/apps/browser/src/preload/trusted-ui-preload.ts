import type { SiteContext, CertificateReviewContext } from '../shared/ipc';
import type { BrowserImportDataType } from '../main/data/browser-import-types';
import { contextBridge, ipcRenderer, webUtils } from 'electron';

import type {
  BrowserCoreDataSnapshot,
  BrowserWindowState,
  LocalAIJobActivity,
  LocalAIPermissionPrompt,
  LocalAIPermissionPromptResponse,
  AIActivitySettings,
  AIActivitySnapshot,
  ModelManagerSnapshot,
  TrustedModelRemovalPreview,
  TrustedModelSubstitute,
  TrustedBrowserAPI,
  OrynComputeSettings,
  OrynDeveloperSettings,
  OrynSettingsSnapshot,
  BrowserAppearance,
  BrowserChromeLayout,
  BrowserPassiveOverlayLayout,
  BrowserExperienceCommand,
  BrowserBookmarkCommand,
  TrustedModelOperation,
  AppUpdateSnapshot,
} from '../shared/ipc';

const channels = {
  browserGetState: 'browser:get-state',
  browserStateChanged: 'browser:state-changed',
  browserFocusOmnibox: 'browser:focus-omnibox',
  browserOpenFind: 'browser:open-find',
  browserOpenTabSearch: 'browser:open-tab-search',
  browserOpenPanel: 'browser:open-panel',
  browserAuthentication: 'browser:authentication',
  browserAuthenticationRespond: 'browser:authentication-respond',
  browserSetDefault: 'browser:set-default',
  browserNewWindow: 'browser:new-window',
  browserNewPrivateWindow: 'browser:new-private-window',
  browserOpenFile: 'browser:open-file',
  browserNewTab: 'browser:new-tab',
  browserActivateTab: 'browser:activate-tab',
  browserCloseTab: 'browser:close-tab',
  browserReopenClosedTab: 'browser:reopen-closed-tab',
  browserNavigate: 'browser:navigate',
  browserBack: 'browser:back',
  browserForward: 'browser:forward',
  browserReload: 'browser:reload',
  browserStop: 'browser:stop',
  browserOpenDevTools: 'browser:open-dev-tools',
  browserGetData: 'browser:get-data',
  browserDataChanged: 'browser:data-changed',
  browserDownloadControl: 'browser:download-control',
  browserDownloadDirectory: 'browser:download-directory',
  browserWindowAction: 'browser:window-action',
  browserDiagnosticsExport: 'browser:diagnostics-export',
  browserHistoryDelete: 'browser:history-delete',
  browserClearBrowsingData: 'browser:clear-browsing-data',
  browserHistoryClear: 'browser:history-clear',
  browserBookmarkImport: 'browser:bookmark-import',
  browserImportDiscover: 'browser:import-discover',
  browserImportPreview: 'browser:import-preview',
  browserImportCommit: 'browser:import-commit',
  browserImportCancel: 'browser:import-cancel',
  browserBookmarkExport: 'browser:bookmark-export',
  browserBookmarkAdd: 'browser:bookmark-add',
  browserBookmarkDelete: 'browser:bookmark-delete',
  browserBookmarkBarVisible: 'browser:bookmark-bar-visible',
  browserPermissionSet: 'browser:permission-set',
  browserCaptureStop: 'browser:capture-stop',
  browserCertificateExport: 'browser:certificate-export',
  browserEmbeddedPermissionRevoke: 'browser:embedded-permission-revoke',
  browserClearSiteData: 'browser:clear-site-data',
  browserFindResult: 'browser:find-result',
  browserFind: 'browser:find',
  browserStopFind: 'browser:stop-find',
  browserShowFind: 'browser:show-find',
  browserSetZoom: 'browser:set-zoom',
  browserPrint: 'browser:print',
  browserSetChromeLayout: 'browser:set-chrome-layout',
  browserDismissPanelAndClickPage: 'browser:dismiss-panel-and-click-page',
  browserSetPassiveOverlayLayout: 'browser:set-passive-overlay-layout',
  browserAppearanceUpdate: 'browser:appearance-update',
  browserExperienceCommand: 'browser:experience-command',
  browserBookmarkCommand: 'browser:bookmark-command',
  appUpdateGet: 'app-update:get',
  appUpdateCheck: 'app-update:check',
  appUpdateDownload: 'app-update:download',
  appUpdateCancelDownload: 'app-update:cancel-download',
  appUpdateDefer: 'app-update:defer',
  appUpdateSkip: 'app-update:skip',
  appUpdateClearSkip: 'app-update:clear-skip',
  appUpdateRestart: 'app-update:restart',
  appUpdateSettingsUpdate: 'app-update:settings-update',
  appUpdateStateChanged: 'app-update:state-changed',
  huggingFacePageActionConfirm: 'hugging-face-page-action:confirm',
  huggingFacePageActionCancel: 'hugging-face-page-action:cancel',
  huggingFacePageActionOpen: 'hugging-face-page-action:open',
  huggingFacePageActionRetry: 'hugging-face-page-action:retry',
  huggingFacePageActionDetails: 'hugging-face-page-action:details',
  huggingFaceCredentialGet: 'hugging-face-credential:get',
  huggingFaceCredentialSet: 'hugging-face-credential:set',
  huggingFaceCredentialClear: 'hugging-face-credential:clear',
  localAIPermissionPrompt: 'local-ai:permission-prompt',
  localAIPermissionRespond: 'local-ai:permission-respond',
  localAIPermissionChooserOpen: 'local-ai:permission-chooser-open',
  localAIPermissionPromptCancelled: 'local-ai:permission-prompt-cancelled',
  localAIJobActivityGet: 'local-ai:job-activity-get',
  localAIJobActivityChanged: 'local-ai:job-activity-changed',
  localAIJobCancelTrusted: 'local-ai:job-cancel-trusted',
  localAIJobCancelSite: 'local-ai:job-cancel-site',
  localAIJobCancelAll: 'local-ai:job-cancel-all',
  localAIModelOperationChanged: 'local-ai:model-operation-changed',
  localAIModelOperationCancelTrusted: 'local-ai:model-operation-cancel-trusted',
  localAIActivityGet: 'local-ai:activity-get',
  localAIActivitySettingsUpdate: 'local-ai:activity-settings-update',
  localAIActivityClear: 'local-ai:activity-clear',
  localAIArtifactSaveTrusted: 'local-ai:artifact-save-trusted',
  localAIModelsGet: 'local-ai:models-get',
  localAIModelSubstitutes: 'local-ai:model-substitutes',
  localAIRegistryStatus: 'local-ai:registry-status',
  localAIRegistryRefresh: 'local-ai:registry-refresh',
  localAIAlternativesSearch: 'local-ai:alternatives-search',
  localAIAlternativesCancel: 'local-ai:alternatives-cancel',
  localAIEngineVersionsGet: 'local-ai:engine-versions-get',
  localAIEngineVersionCheck: 'local-ai:engine-version-check',
  localAIRuntimePacksGet: 'local-ai:runtime-packs-get',
  localAIRuntimePacksCleanup: 'local-ai:runtime-packs-cleanup',
  localAIModelUnloadTrusted: 'local-ai:model-unload-trusted',
  localAIModelRuntimeOverrideTrusted: 'local-ai:model-runtime-override-trusted',
  localAIModelRemovePreviewTrusted: 'local-ai:model-remove-preview-trusted',
  localAIModelRemoveTrusted: 'local-ai:model-remove-trusted',
  localAIModelSourceOpenTrusted: 'local-ai:model-source-open-trusted',
  localAIModelReferenceSourceOpenTrusted: 'local-ai:model-reference-source-open-trusted',
  localAISettingsGet: 'local-ai:settings-get',
  localAISettingsRankingUpdate: 'local-ai:settings-ranking-update',
  localAISettingsRoleUpdate: 'local-ai:settings-role-update',
  localAISettingsComputeUpdate: 'local-ai:settings-compute-update',
  localAISettingsSiteReset: 'local-ai:settings-site-reset',
  localAISettingsSiteAskAgain: 'local-ai:settings-site-ask-again',
  localAISettingsSiteRevoke: 'local-ai:settings-site-revoke',
  localAISettingsStorageCleanup: 'local-ai:settings-storage-cleanup',
  localAISettingsDeveloperUpdate: 'local-ai:settings-developer-update',
} as const;

const browser: TrustedBrowserAPI = Object.freeze({
  getState: () => ipcRenderer.invoke(channels.browserGetState) as Promise<BrowserWindowState>,
  onAuthenticationRequest: (callback: (request: { id: string; target: string }) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, request: { id: string; target: string }) =>
      callback(request);
    ipcRenderer.on(channels.browserAuthentication, listener);
    return () => ipcRenderer.removeListener(channels.browserAuthentication, listener);
  },
  respondAuthentication: (id: string, credentials: { username: string; password: string } | null) =>
    ipcRenderer.invoke(channels.browserAuthenticationRespond, id, credentials) as Promise<void>,
  setDefaultBrowser: () => ipcRenderer.invoke(channels.browserSetDefault) as Promise<boolean>,
  exportDiagnostics: () => ipcRenderer.invoke(channels.browserDiagnosticsExport),
  closeWindow: () => ipcRenderer.invoke(channels.browserWindowAction, 'close'),
  quit: () => ipcRenderer.invoke(channels.browserWindowAction, 'quit'),
  newWindow: () => ipcRenderer.invoke(channels.browserNewWindow) as Promise<void>,
  newPrivateWindow: (url?: string) =>
    ipcRenderer.invoke(channels.browserNewPrivateWindow, url) as Promise<void>,
  openFile: () => ipcRenderer.invoke(channels.browserOpenFile) as Promise<void>,
  stopCapture: (id: number, generation: string) =>
    ipcRenderer.invoke(channels.browserCaptureStop, id, generation) as Promise<boolean>,
  exportCertificate: (context: CertificateReviewContext) =>
    ipcRenderer.invoke(channels.browserCertificateExport, context) as Promise<void>,
  revokeEmbeddedPermission: (origin: string, permission: string, context: SiteContext) =>
    ipcRenderer.invoke(
      channels.browserEmbeddedPermissionRevoke,
      origin,
      permission,
      context,
    ) as Promise<void>,
  newTab: (url?: string) => ipcRenderer.invoke(channels.browserNewTab, url) as Promise<string>,
  activateTab: (tabId: string) =>
    ipcRenderer.invoke(channels.browserActivateTab, tabId) as Promise<void>,
  closeTab: (tabId?: string) =>
    ipcRenderer.invoke(channels.browserCloseTab, tabId) as Promise<void>,
  reopenClosedTab: () =>
    ipcRenderer.invoke(channels.browserReopenClosedTab) as Promise<string | undefined>,
  navigate: (value: string) => ipcRenderer.invoke(channels.browserNavigate, value) as Promise<void>,
  back: () => ipcRenderer.invoke(channels.browserBack) as Promise<void>,
  forward: () => ipcRenderer.invoke(channels.browserForward) as Promise<void>,
  reload: () => ipcRenderer.invoke(channels.browserReload) as Promise<void>,
  stop: () => ipcRenderer.invoke(channels.browserStop) as Promise<void>,
  openDevTools: () => ipcRenderer.invoke(channels.browserOpenDevTools) as Promise<void>,
  getBrowserData: (query?: string) =>
    ipcRenderer.invoke(channels.browserGetData, query) as Promise<BrowserCoreDataSnapshot>,
  deleteHistory: (historyId: string) =>
    ipcRenderer.invoke(channels.browserHistoryDelete, historyId) as Promise<void>,
  clearBrowsingData: (category: 'cookies' | 'cache' | 'permissions') =>
    ipcRenderer.invoke(channels.browserClearBrowsingData, category) as Promise<void>,
  clearHistory: (range?: 'hour' | 'day' | 'all') =>
    ipcRenderer.invoke(channels.browserHistoryClear, range) as Promise<void>,
  importBookmarks: (token?: string) =>
    ipcRenderer.invoke(channels.browserBookmarkImport, token) as Promise<
      { token: string; count: number } | undefined
    >,
  discoverBrowserImports: () => ipcRenderer.invoke(channels.browserImportDiscover),
  previewBrowserImport: (request: { sourceId: string; dataTypes: BrowserImportDataType[] }) =>
    ipcRenderer.invoke(channels.browserImportPreview, request),
  commitBrowserImport: (token: string) => ipcRenderer.invoke(channels.browserImportCommit, token),
  cancelBrowserImport: (token?: string) => ipcRenderer.invoke(channels.browserImportCancel, token),
  exportBookmarks: () => ipcRenderer.invoke(channels.browserBookmarkExport) as Promise<void>,
  addBookmark: (title: string, url: string) =>
    ipcRenderer.invoke(channels.browserBookmarkAdd, title, url) as Promise<void>,
  deleteBookmark: (bookmarkId: string) =>
    ipcRenderer.invoke(channels.browserBookmarkDelete, bookmarkId) as Promise<void>,
  setBookmarkBarVisible: (visible: boolean) =>
    ipcRenderer.invoke(channels.browserBookmarkBarVisible, visible) as Promise<void>,
  setSitePermission: (
    permission: string,
    decision: 'allow' | 'deny' | 'ask',
    context: SiteContext,
  ) =>
    ipcRenderer.invoke(
      channels.browserPermissionSet,
      permission,
      decision,
      context,
    ) as Promise<void>,
  clearCurrentSiteData: (context: SiteContext) =>
    ipcRenderer.invoke(channels.browserClearSiteData, context) as Promise<void>,
  chooseDownloadDirectory: () => ipcRenderer.invoke(channels.browserDownloadDirectory),
  controlDownload: (id: string, action: string) =>
    ipcRenderer.invoke(channels.browserDownloadControl, id, action) as Promise<void>,
  onBrowserDataChanged: (callback: () => void) => {
    const listener = () => callback();
    ipcRenderer.on(channels.browserDataChanged, listener);
    return () => ipcRenderer.removeListener(channels.browserDataChanged, listener);
  },
  onFindResult: (callback: (result: { matches: number; activeMatchOrdinal: number }) => void) => {
    const listener = (
      _event: Electron.IpcRendererEvent,
      result: { matches: number; activeMatchOrdinal: number },
    ) => callback(result);
    ipcRenderer.on(channels.browserFindResult, listener);
    return () => ipcRenderer.removeListener(channels.browserFindResult, listener);
  },
  findInPage: (query: string, forward = true) =>
    ipcRenderer.invoke(channels.browserFind, query, forward) as Promise<void>,
  stopFindInPage: () => ipcRenderer.invoke(channels.browserStopFind) as Promise<void>,
  showFindInPage: () => ipcRenderer.invoke(channels.browserShowFind) as Promise<void>,
  setZoom: (action: 'in' | 'out' | 'reset') =>
    ipcRenderer.invoke(channels.browserSetZoom, action) as Promise<void>,
  printPage: () => ipcRenderer.invoke(channels.browserPrint) as Promise<void>,
  dismissPanelAndClickPage: (x: number, y: number, button: 'left' | 'middle' | 'right') =>
    ipcRenderer.invoke(channels.browserDismissPanelAndClickPage, x, y, button),
  setChromeLayout: (layout: BrowserChromeLayout) =>
    ipcRenderer.invoke(channels.browserSetChromeLayout, layout) as Promise<void>,
  setPassiveOverlayLayout: (layout: BrowserPassiveOverlayLayout) =>
    ipcRenderer.invoke(channels.browserSetPassiveOverlayLayout, layout) as Promise<void>,
  updateAppearance: (value: Partial<BrowserAppearance>) =>
    ipcRenderer.invoke(channels.browserAppearanceUpdate, value) as Promise<void>,
  updateBrowserExperience: (command: BrowserExperienceCommand) =>
    ipcRenderer.invoke(channels.browserExperienceCommand, command) as Promise<void>,
  updateBookmarks: (command: BrowserBookmarkCommand) =>
    ipcRenderer.invoke(channels.browserBookmarkCommand, command) as Promise<void>,
  getAppUpdate: () => ipcRenderer.invoke(channels.appUpdateGet) as Promise<AppUpdateSnapshot>,
  checkForAppUpdate: () =>
    ipcRenderer.invoke(channels.appUpdateCheck) as Promise<AppUpdateSnapshot>,
  downloadAppUpdate: (version: string, allowSkipped = false, approvalId?: string) =>
    ipcRenderer.invoke(
      channels.appUpdateDownload,
      version,
      allowSkipped,
      approvalId,
    ) as Promise<AppUpdateSnapshot>,
  cancelAppUpdateDownload: () =>
    ipcRenderer.invoke(channels.appUpdateCancelDownload) as Promise<AppUpdateSnapshot>,
  deferAppUpdate: (version: string) =>
    ipcRenderer.invoke(channels.appUpdateDefer, version) as Promise<AppUpdateSnapshot>,
  skipAppUpdate: (version: string) =>
    ipcRenderer.invoke(channels.appUpdateSkip, version) as Promise<AppUpdateSnapshot>,
  clearAppUpdateSkip: () =>
    ipcRenderer.invoke(channels.appUpdateClearSkip) as Promise<AppUpdateSnapshot>,
  restartToInstallAppUpdate: (version: string, approvalId?: string) =>
    ipcRenderer.invoke(
      channels.appUpdateRestart,
      version,
      approvalId,
    ) as Promise<AppUpdateSnapshot>,
  updateAppUpdateSettings: (value: Parameters<TrustedBrowserAPI['updateAppUpdateSettings']>[0]) =>
    ipcRenderer.invoke(channels.appUpdateSettingsUpdate, value) as Promise<AppUpdateSnapshot>,
  onAppUpdateStateChanged(listener: (state: AppUpdateSnapshot) => void) {
    const handler = (_event: Electron.IpcRendererEvent, state: AppUpdateSnapshot) =>
      listener(state);
    ipcRenderer.on(channels.appUpdateStateChanged, handler);
    return () => ipcRenderer.removeListener(channels.appUpdateStateChanged, handler);
  },
  confirmHuggingFacePageAction: (trustMode: 'oryn-verified' | 'user-trusted') =>
    ipcRenderer.invoke(channels.huggingFacePageActionConfirm, trustMode) as Promise<void>,
  cancelHuggingFacePageAction: () =>
    ipcRenderer.invoke(channels.huggingFacePageActionCancel) as Promise<void>,
  openHuggingFaceModelWorkspace: () =>
    ipcRenderer.invoke(channels.huggingFacePageActionOpen) as Promise<string>,
  retryHuggingFacePageAction: () =>
    ipcRenderer.invoke(channels.huggingFacePageActionRetry) as Promise<void>,
  showHuggingFacePageActionDetails: () =>
    ipcRenderer.invoke(channels.huggingFacePageActionDetails) as Promise<void>,
  onHuggingFacePageActionDetails(listener: (state: BrowserWindowState) => void) {
    const handler = (_event: Electron.IpcRendererEvent, state: BrowserWindowState) =>
      listener(state);
    ipcRenderer.on(channels.huggingFacePageActionDetails, handler);
    return () => ipcRenderer.removeListener(channels.huggingFacePageActionDetails, handler);
  },
  getHuggingFaceCredentialStatus: () =>
    ipcRenderer.invoke(channels.huggingFaceCredentialGet) as ReturnType<
      TrustedBrowserAPI['getHuggingFaceCredentialStatus']
    >,
  setHuggingFaceCredential: (token: string) =>
    ipcRenderer.invoke(channels.huggingFaceCredentialSet, token) as ReturnType<
      TrustedBrowserAPI['setHuggingFaceCredential']
    >,
  clearHuggingFaceCredential: () =>
    ipcRenderer.invoke(channels.huggingFaceCredentialClear) as ReturnType<
      TrustedBrowserAPI['clearHuggingFaceCredential']
    >,
  onStateChanged(listener: (state: BrowserWindowState) => void) {
    const handler = (_event: Electron.IpcRendererEvent, state: BrowserWindowState) =>
      listener(state);
    ipcRenderer.on(channels.browserStateChanged, handler);
    return () => ipcRenderer.removeListener(channels.browserStateChanged, handler);
  },
  onFocusOmnibox(listener: () => void) {
    const handler = () => listener();
    ipcRenderer.on(channels.browserFocusOmnibox, handler);
    return () => ipcRenderer.removeListener(channels.browserFocusOmnibox, handler);
  },
  onOpenFind(listener: () => void) {
    const handler = () => listener();
    ipcRenderer.on(channels.browserOpenFind, handler);
    return () => ipcRenderer.removeListener(channels.browserOpenFind, handler);
  },
  onOpenTabSearch(listener: () => void) {
    const handler = () => listener();
    ipcRenderer.on(channels.browserOpenTabSearch, handler);
    return () => ipcRenderer.removeListener(channels.browserOpenTabSearch, handler);
  },
  onOpenPanel(
    listener: (
      panel: 'bookmarks' | 'history' | 'downloads' | 'settings' | 'local-ai' | null,
    ) => void,
  ) {
    const handler = (
      _event: Electron.IpcRendererEvent,
      panel: 'bookmarks' | 'history' | 'downloads' | 'settings' | 'local-ai' | null,
    ) => listener(panel);
    ipcRenderer.on(channels.browserOpenPanel, handler);
    return () => ipcRenderer.removeListener(channels.browserOpenPanel, handler);
  },
  onLocalAIPermissionPrompt(listener: (prompt: LocalAIPermissionPrompt) => void) {
    const handler = (_event: Electron.IpcRendererEvent, prompt: LocalAIPermissionPrompt) =>
      listener(prompt);
    ipcRenderer.on(channels.localAIPermissionPrompt, handler);
    return () => ipcRenderer.removeListener(channels.localAIPermissionPrompt, handler);
  },
  respondToLocalAIPermission(response: LocalAIPermissionPromptResponse) {
    return ipcRenderer.invoke(channels.localAIPermissionRespond, response) as Promise<boolean>;
  },
  openLocalAIModelChooser: (promptId: string) =>
    ipcRenderer.invoke(channels.localAIPermissionChooserOpen, promptId) as Promise<boolean>,
  onLocalAIPermissionPromptCancelled(listener: (promptId: string) => void) {
    const handler = (_event: Electron.IpcRendererEvent, promptId: string) => listener(promptId);
    ipcRenderer.on(channels.localAIPermissionPromptCancelled, handler);
    return () => ipcRenderer.removeListener(channels.localAIPermissionPromptCancelled, handler);
  },
  getLocalAIJobActivity: () =>
    ipcRenderer.invoke(channels.localAIJobActivityGet) as Promise<LocalAIJobActivity>,
  onLocalAIJobActivityChanged(listener: (activity: LocalAIJobActivity) => void) {
    const handler = (_event: Electron.IpcRendererEvent, activity: LocalAIJobActivity) =>
      listener(activity);
    ipcRenderer.on(channels.localAIJobActivityChanged, handler);
    return () => ipcRenderer.removeListener(channels.localAIJobActivityChanged, handler);
  },
  onModelOperationChanged(listener: (operation: TrustedModelOperation) => void) {
    const handler = (_event: Electron.IpcRendererEvent, operation: TrustedModelOperation) =>
      listener(operation);
    ipcRenderer.on(channels.localAIModelOperationChanged, handler);
    return () => ipcRenderer.removeListener(channels.localAIModelOperationChanged, handler);
  },
  cancelModelOperation: (operationId: string) =>
    ipcRenderer.invoke(channels.localAIModelOperationCancelTrusted, operationId) as Promise<void>,
  cancelLocalAIJob: (jobId: string) =>
    ipcRenderer.invoke(channels.localAIJobCancelTrusted, jobId) as Promise<void>,
  cancelLocalAIJobsForSite: (origin: string) =>
    ipcRenderer.invoke(channels.localAIJobCancelSite, origin) as Promise<void>,
  cancelAllLocalAIJobs: () => ipcRenderer.invoke(channels.localAIJobCancelAll) as Promise<void>,
  getAIActivity: () =>
    ipcRenderer.invoke(channels.localAIActivityGet) as Promise<AIActivitySnapshot>,
  updateAIActivitySettings: (settings: Partial<AIActivitySettings>) =>
    ipcRenderer.invoke(
      channels.localAIActivitySettingsUpdate,
      settings,
    ) as Promise<AIActivitySettings>,
  clearAIActivity: () => ipcRenderer.invoke(channels.localAIActivityClear) as Promise<void>,
  saveLocalAIArtifact: (artifactId: string) =>
    ipcRenderer.invoke(channels.localAIArtifactSaveTrusted, artifactId) as Promise<void>,
  getModelManager: () =>
    ipcRenderer.invoke(channels.localAIModelsGet) as Promise<ModelManagerSnapshot>,
  getRegistryStatus: () => ipcRenderer.invoke(channels.localAIRegistryStatus),
  setRegistryRefresh: (enabled: boolean) =>
    ipcRenderer.invoke(channels.localAIRegistryRefresh, enabled),
  checkBetterAlternatives: (installationId: string) =>
    ipcRenderer.invoke(channels.localAIAlternativesSearch, installationId),
  cancelBetterAlternatives: () => ipcRenderer.invoke(channels.localAIAlternativesCancel),
  getManagedModelSubstitutes: (installationId: string) =>
    ipcRenderer.invoke(channels.localAIModelSubstitutes, installationId) as Promise<
      TrustedModelSubstitute[]
    >,
  getEngineVersions: () =>
    ipcRenderer.invoke(channels.localAIEngineVersionsGet) as ReturnType<
      TrustedBrowserAPI['getEngineVersions']
    >,
  checkEngineVersion: (id: string) =>
    ipcRenderer.invoke(channels.localAIEngineVersionCheck, id) as ReturnType<
      TrustedBrowserAPI['checkEngineVersion']
    >,
  getManagedRuntimePacks: () =>
    ipcRenderer.invoke(channels.localAIRuntimePacksGet) as Promise<
      Pick<ModelManagerSnapshot, 'runtimePacks' | 'runtimePackPolicy'>
    >,
  cleanupManagedRuntimePacks: () =>
    ipcRenderer.invoke(channels.localAIRuntimePacksCleanup) as Promise<{
      removedPacks: number;
      reclaimedBytes: number;
    }>,
  unloadManagedModel: (installationId: string) =>
    ipcRenderer.invoke(channels.localAIModelUnloadTrusted, installationId) as Promise<void>,
  setManagedModelRuntimeOverride: (installationId: string, candidateId?: string) =>
    ipcRenderer.invoke(
      channels.localAIModelRuntimeOverrideTrusted,
      installationId,
      candidateId,
    ) as Promise<void>,
  previewManagedModelRemoval: (installationId: string) =>
    ipcRenderer.invoke(
      channels.localAIModelRemovePreviewTrusted,
      installationId,
    ) as Promise<TrustedModelRemovalPreview>,
  removeManagedModel: (
    installationId: string,
    action: 'delete' | 'when-idle' | 'cancel-and-delete',
  ) =>
    ipcRenderer.invoke(channels.localAIModelRemoveTrusted, installationId, action) as Promise<
      'removed' | 'deferred'
    >,
  openManagedModelSource: (installationId: string) =>
    ipcRenderer.invoke(channels.localAIModelSourceOpenTrusted, installationId) as Promise<void>,
  openModelSource: (repository: string, revision?: string) =>
    ipcRenderer.invoke(
      channels.localAIModelReferenceSourceOpenTrusted,
      repository,
      revision,
    ) as Promise<void>,
  getOrynSettings: () =>
    ipcRenderer.invoke(channels.localAISettingsGet) as Promise<OrynSettingsSnapshot>,
  updateOrynRanking: (value: Parameters<TrustedBrowserAPI['updateOrynRanking']>[0]) =>
    ipcRenderer.invoke(channels.localAISettingsRankingUpdate, value) as Promise<void>,
  updateOrynRolePreference: (value: Parameters<TrustedBrowserAPI['updateOrynRolePreference']>[0]) =>
    ipcRenderer.invoke(channels.localAISettingsRoleUpdate, value) as Promise<void>,
  updateOrynCompute: (value: Partial<OrynComputeSettings>) =>
    ipcRenderer.invoke(channels.localAISettingsComputeUpdate, value) as Promise<void>,
  resetOrynSiteChoices: (origin: string) =>
    ipcRenderer.invoke(channels.localAISettingsSiteReset, origin) as Promise<void>,
  askOrynSiteAgain: (origin: string) =>
    ipcRenderer.invoke(channels.localAISettingsSiteAskAgain, origin) as Promise<void>,
  revokeOrynSite: (origin: string, cancelActive: boolean) =>
    ipcRenderer.invoke(channels.localAISettingsSiteRevoke, origin, cancelActive) as Promise<void>,
  cleanupOrynStorage: (target: 'artifacts' | 'temporary') =>
    ipcRenderer.invoke(channels.localAISettingsStorageCleanup, target) as Promise<void>,
  updateOrynDeveloper: (value: Partial<OrynDeveloperSettings>) =>
    ipcRenderer.invoke(channels.localAISettingsDeveloperUpdate, value) as Promise<void>,
});

contextBridge.exposeInMainWorld('browser', browser);

window.addEventListener('drop', (event) => {
  if (!event.isTrusted || event.defaultPrevented || !process.isMainFrame) return;
  const paths = [...(event.dataTransfer?.files ?? [])]
    .filter((file) => /\.(html?|xhtml|shtml)$/iu.test(file.name))
    .map((file) => webUtils.getPathForFile(file))
    .filter(Boolean)
    .slice(0, 16);
  if (!paths.length) return;
  event.preventDefault();
  void ipcRenderer.invoke('browser:native-file-drop', paths);
});
