import type { ConnectionSecuritySnapshot } from '../main/connection-security';
import type { CaptureAccessSnapshot } from '../main/data/capture-access-registry';
import type { SitePermissionRecord } from '../main/data/site-permission-store';
import type {
  BrowserImportSource,
  BrowserImportDataType,
  BrowserImportPreview,
  BrowserImportCommitResult,
} from '../main/data/browser-import-types';
import type {
  ModelGenerationProfile,
  SelectionExplanation,
  SharedTaskContract,
  TaskProfile,
  TrustedModelMetadata,
} from '@ocno/oryn-protocol';
import type { EngineVersion } from './engine-versions';

export const SIDEBAR_PANEL_MIN_WIDTH = 300;
export const SIDEBAR_PANEL_MAX_WIDTH = 720;
export const TAB_SEARCH_SHORTCUT = 'CommandOrControl+Shift+A' as const;

export function clampSidebarPanelWidth(width: number): number {
  return Math.min(SIDEBAR_PANEL_MAX_WIDTH, Math.max(SIDEBAR_PANEL_MIN_WIDTH, Math.round(width)));
}

export const IPC = {
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
  workspaceGetState: 'model-workspace:get-state',
  workspaceInvoke: 'model-workspace:invoke',
  workspaceCancel: 'model-workspace:cancel',
  workspaceJobEvent: 'model-workspace:job-event',
  workspaceArtifactCreateBegin: 'model-workspace:artifact-create-begin',
  workspaceArtifactCreateChunk: 'model-workspace:artifact-create-chunk',
  workspaceArtifactCreateFinish: 'model-workspace:artifact-create-finish',
  workspaceArtifactCreateAbort: 'model-workspace:artifact-create-abort',
  workspaceArtifactGetUrl: 'model-workspace:artifact-get-url',
  workspaceArtifactSaveAs: 'model-workspace:artifact-save-as',
  workspaceArtifactRelease: 'model-workspace:artifact-release',
  workspaceOpenSource: 'model-workspace:open-source',
  localAIGetCapabilities: 'local-ai:get-capabilities',
  localAIPermissionPrompt: 'local-ai:permission-prompt',
  localAIPermissionRespond: 'local-ai:permission-respond',
  localAIPermissionChooserOpen: 'local-ai:permission-chooser-open',
  localAIPermissionPromptCancelled: 'local-ai:permission-prompt-cancelled',
  localAIModelRequest: 'local-ai:model-request',
  localAIModelRequestCancel: 'local-ai:model-request-cancel',
  localAIModelRequestProgress: 'local-ai:model-request-progress',
  localAIModelOperationChanged: 'local-ai:model-operation-changed',
  localAIModelOperationCancelTrusted: 'local-ai:model-operation-cancel-trusted',
  localAIModelSessionRelease: 'local-ai:model-session-release',
  localAIArtifactCreateBegin: 'local-ai:artifact-create-begin',
  localAIArtifactCreateChunk: 'local-ai:artifact-create-chunk',
  localAIArtifactCreateFinish: 'local-ai:artifact-create-finish',
  localAIArtifactCreateAbort: 'local-ai:artifact-create-abort',
  localAIArtifactGet: 'local-ai:artifact-get',
  localAIArtifactGetUrl: 'local-ai:artifact-get-url',
  localAIArtifactSaveAs: 'local-ai:artifact-save-as',
  localAIArtifactRelease: 'local-ai:artifact-release',
  localAIJobInvoke: 'local-ai:job-invoke',
  localAIJobCancel: 'local-ai:job-cancel',
  localAIJobEvent: 'local-ai:job-event',
  localAIJobActivityGet: 'local-ai:job-activity-get',
  localAIJobActivityChanged: 'local-ai:job-activity-changed',
  localAIJobCancelTrusted: 'local-ai:job-cancel-trusted',
  localAIJobCancelSite: 'local-ai:job-cancel-site',
  localAIJobCancelAll: 'local-ai:job-cancel-all',
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

export interface RuntimeCapabilities {
  runtimeAvailable: boolean;
  runtimeState: 'ready' | 'starting' | 'unavailable' | 'error';
  sdkVersion: string;
  browserApiVersion?: string;
  supportedModelProviders: Array<'huggingface'>;
  features: {
    modelSelection?: boolean;
    taskProfiles?: boolean;
    modelRequests: boolean;
    modelSubstitution: boolean;
    streaming: boolean;
    artifacts: boolean;
    artifactUrls: boolean;
    cancellation: boolean;
    queueing: boolean;
    multipleSubmittedJobs: boolean;
  };
  scheduler?: {
    generation: {
      globalAcrossOrigins: boolean;
      maxConcurrentRunning: number;
    };
  };
}

export interface InjectedLocalAI {
  readonly sdkVersion: string;
  getCapabilities(): Promise<RuntimeCapabilities>;
  requestModel(operationId: string, request: unknown): Promise<WebsiteBridgeResult<unknown>>;
  cancelModelRequest(operationId: string): Promise<void>;
  releaseModelSession(sessionId: string): Promise<WebsiteBridgeResult<void>>;
  onModelRequestProgress(operationId: string, listener: (progress: unknown) => void): () => void;
  beginArtifactCreate(metadata: unknown): Promise<WebsiteBridgeResult<{ uploadId: string }>>;
  appendArtifactChunk(uploadId: string, chunk: ArrayBuffer): Promise<WebsiteBridgeResult<void>>;
  finishArtifactCreate(uploadId: string): Promise<WebsiteBridgeResult<unknown>>;
  abortArtifactCreate(uploadId: string): Promise<void>;
  getArtifact(artifactId: string): Promise<WebsiteBridgeResult<unknown>>;
  getArtifactUrl(artifactId: string): Promise<WebsiteBridgeResult<string>>;
  saveArtifactAs(
    artifactId: string,
    options?: { suggestedFilename?: string },
  ): Promise<WebsiteBridgeResult<void>>;
  releaseArtifact(artifactId: string): Promise<WebsiteBridgeResult<void>>;
  invokeJob(request: unknown): Promise<WebsiteBridgeResult<unknown>>;
  cancelJob(jobId: string): Promise<WebsiteBridgeResult<void>>;
  onJobEvent(listener: (event: unknown) => void): () => void;
}

export type WebsiteBridgeResult<T> =
  | { ok: true; value: T }
  | {
      ok: false;
      error: { code: string; message: string; details?: Record<string, unknown> };
    };

export type ModelWorkspaceTaskContract =
  | SharedTaskContract
  | 'chat-v1'
  | 'text-to-image-v1'
  | 'image-to-image-v1'
  | 'text-to-music-v1'
  | 'speech-synthesis-v1'
  | 'transcription-v1'
  | 'media-audio-extract-v1'
  | 'audio-source-separation-v1'
  | 'forced-alignment-v1'
  | 'subtitle-export-v1'
  | 'embeddings-v1'
  | 'reranking-v1'
  | 'text-to-video-v1'
  | 'image-to-video-v1'
  | 'text-classification-v1'
  | 'token-classification-v1'
  | 'question-answering-v1'
  | 'fill-mask-v1'
  | 'translation-v1'
  | 'summarization-v1'
  | 'image-classification-v1'
  | 'audio-classification-v1'
  | 'object-detection-v1'
  | 'depth-estimation-v1'
  | 'zero-shot-image-classification-v1'
  | 'zero-shot-classification-v1'
  | 'image-embeddings-v1'
  | 'image-segmentation-v1'
  | 'mask-generation-v1'
  | 'zero-shot-object-detection-v1'
  | 'image-editing-v1'
  | 'image-layers-v1'
  | 'image-to-text-v1'
  | 'visual-question-answering-v1';

export interface ModelWorkspaceState {
  status: 'ready' | 'model-removed' | 'runtime-revoked' | 'unsupported' | 'error';
  repository?: string;
  exactRevision?: string;
  sourceURL?: string;
  taskContract?: ModelWorkspaceTaskContract;
  task?: string;
  trustMode?: 'oryn-verified' | 'user-trusted';
  capabilities?: Record<string, unknown>;
  generationProfile?: ModelGenerationProfile;
  taskProfile?: TaskProfile;
  runtimeSelection?: TrustedRuntimeSelection;
  message?: string;
  appearance?: BrowserAppearance;
}

export interface TrustedRuntimeSelection {
  candidateId: string;
  automatic: boolean;
  policy: 'balanced' | 'performance' | 'memory-saver' | 'quality';
  foundationId: string;
  foundationVersion: string;
  adapterId: string;
  adapterVersion: string;
  backend: string;
  precision: string;
  quantization?: string;
  estimatedMemoryBytes: number;
  fallbackCount: number;
  reasonCodes: string[];
  alternatives: TrustedRuntimeCandidate[];
}

export interface TrustedRuntimeCandidate {
  candidateId: string;
  foundationId: string;
  foundationVersion: string;
  adapterId: string;
  adapterVersion: string;
  backend: string;
  precision: string;
  quantization?: string;
  estimatedMemoryBytes: number;
}

export interface ModelWorkspaceAPI {
  getState(): Promise<ModelWorkspaceState>;
  invoke(
    input: unknown,
  ): Promise<WebsiteBridgeResult<{ jobId: string; sessionId: string; state: string }>>;
  cancel(jobId: string): Promise<WebsiteBridgeResult<void>>;
  onJobEvent(listener: (event: unknown) => void): () => void;
  beginArtifactCreate(metadata: {
    mimeType: string;
    sizeBytes: number;
    name?: string;
  }): Promise<WebsiteBridgeResult<{ uploadId: string }>>;
  appendArtifactChunk(uploadId: string, chunk: ArrayBuffer): Promise<WebsiteBridgeResult<void>>;
  finishArtifactCreate(uploadId: string): Promise<WebsiteBridgeResult<unknown>>;
  abortArtifactCreate(uploadId: string): Promise<void>;
  getArtifactUrl(artifactId: string): Promise<WebsiteBridgeResult<string>>;
  saveArtifactAs(
    artifactId: string,
    suggestedFilename?: string,
  ): Promise<WebsiteBridgeResult<void>>;
  releaseArtifact(artifactId: string): Promise<WebsiteBridgeResult<void>>;
  openSource(): Promise<void>;
}

export interface BrowserTabState {
  connection?: ConnectionSecuritySnapshot;
  captureAccess?: Array<'camera' | 'microphone' | 'screen'>;
  crashed?: boolean;
  audible?: boolean;
  muted?: boolean;
  tabId: string;
  webContentsId: number;
  title: string;
  url: string;
  favicon?: string;
  canGoBack: boolean;
  canGoForward: boolean;
  isLoading: boolean;
  workspaceId: string;
  islandId?: string;
  pinned: boolean;
}

export interface BrowserWorkspaceState {
  workspaceId: string;
  name: string;
  accent: string;
  order: number;
}

export interface BrowserTabIslandState {
  islandId: string;
  workspaceId: string;
  name: string;
  accent: string;
  collapsed: boolean;
  order: number;
}

export interface BrowserSidebarAppState {
  appId: string;
  name: string;
  enabled: boolean;
  order: number;
  muted: boolean;
}

export interface BrowserSidebarToolState {
  toolId: 'tab-search' | 'bookmarks' | 'history' | 'downloads' | 'resources' | 'settings';
  name: string;
  enabled: boolean;
  order: number;
}

export interface BrowserExperienceState {
  activeWorkspaceId: string;
  workspaces: BrowserWorkspaceState[];
  islands: BrowserTabIslandState[];
  recentlyClosed: Array<{ url: string; title?: string; closedAt?: string; workspaceId?: string }>;
  sidebar: {
    visible: boolean;
    width: number;
    panelWidth: number;
    panelMode: 'overlay' | 'pinned';
    activeAppId: string | null;
    tools: BrowserSidebarToolState[];
    apps: BrowserSidebarAppState[];
  };
  tabSearchShortcut: typeof TAB_SEARCH_SHORTCUT;
  autoGroupOpenerTabs: boolean;
}

export type BrowserExperienceCommand =
  | { type: 'workspace-create'; name: string }
  | { type: 'workspace-rename'; workspaceId: string; name: string }
  | { type: 'workspace-delete'; workspaceId: string; tabs: 'move-to-default' | 'close' }
  | { type: 'workspace-switch'; workspaceId: string }
  | { type: 'tab-move-workspace'; tabId: string; workspaceId: string }
  | { type: 'tab-new-window'; tabId: string }
  | { type: 'tab-mute'; tabId: string; muted: boolean }
  | { type: 'tab-reorder'; tabId: string; beforeTabId?: string }
  | { type: 'tab-pin'; tabId: string; pinned: boolean }
  | { type: 'tab-duplicate'; tabId: string }
  | { type: 'tab-close-others'; tabId: string }
  | { type: 'tab-close-right'; tabId: string }
  | { type: 'tab-close-duplicates' }
  | { type: 'tab-auto-group'; enabled: boolean }
  | { type: 'tab-search-shortcut'; shortcut: BrowserExperienceState['tabSearchShortcut'] }
  | { type: 'island-create'; tabIds: string[]; name?: string }
  | { type: 'island-update'; islandId: string; name?: string; accent?: string; collapsed?: boolean }
  | { type: 'island-delete'; islandId: string }
  | { type: 'island-move-workspace'; islandId: string; workspaceId: string }
  | { type: 'tab-move-island'; tabId: string; islandId?: string }
  | { type: 'sidebar-toggle'; visible: boolean }
  | { type: 'sidebar-tool-open'; toolId: BrowserSidebarToolState['toolId'] }
  | { type: 'sidebar-tool-enable'; toolId: BrowserSidebarToolState['toolId']; enabled: boolean }
  | { type: 'sidebar-tool-order'; toolId: BrowserSidebarToolState['toolId']; order: number }
  | { type: 'sidebar-app-open'; appId?: string }
  | { type: 'sidebar-mode'; mode: 'overlay' | 'pinned' }
  | { type: 'sidebar-resize'; width: number; persist?: boolean }
  | { type: 'sidebar-app-mute'; appId: string; muted: boolean }
  | { type: 'sidebar-app-enable'; appId: string; enabled: boolean }
  | { type: 'sidebar-app-clear-data'; appId: string }
  | { type: 'sidebar-app-order'; appId: string; order: number };

export type BrowserBookmarkCommand =
  | { type: 'create-folder'; title: string; parentId?: string | null }
  | { type: 'edit'; bookmarkId: string; title: string; url?: string }
  | { type: 'folder-from-items'; bookmarkIds: string[]; title?: string }
  | { type: 'move'; bookmarkIds: string[]; parentId: string | null; copy?: boolean }
  | { type: 'trash'; bookmarkIds: string[] }
  | { type: 'restore'; bookmarkIds: string[] }
  | { type: 'delete-permanently'; bookmarkIds: string[] }
  | { type: 'favorite'; bookmarkId: string; favorite: boolean }
  | { type: 'thumbnail'; bookmarkId: string; thumbnail?: string }
  | { type: 'reorder'; bookmarkId: string; order: number }
  | { type: 'save-all-tabs'; title?: string };

export interface BrowserWindowState {
  privateMode?: boolean;
  captureAccess?: CaptureAccessSnapshot[];
  activeTabId: string;
  tabs: BrowserTabState[];
  engineState: 'starting' | 'ready' | 'error' | 'stopped';
  zoomFactor: number;
  memoryPressure: 'normal' | 'high' | 'critical';
  localAIResources?: {
    processResidentBytes?: number;
    browserPrivateBytes?: number;
    workerResidentBytes?: number;
    systemTotalBytes?: number;
    systemAvailableBytes?: number;
    memoryBudgetBytes?: number;
    estimatedResidentModelBytes: number;
    acceleratorBackend?: string;
    acceleratorAllocatedBytes?: number;
    acceleratorReservedBytes?: number;
    acceleratorReliable: boolean;
    cpuPercent?: number;
    engineCPUPercent?: number;
    sampleWindowMs?: number;
    activeExecution?: { jobId?: string; model?: string; task?: string; estimatedBytes?: number };
    throughput?: { value: number; unit: string; sampleWindowMs: number };
    generationRunning: number;
    generationQueued: number;
  };
  huggingFacePageAction?: HuggingFacePageActionState;
  experience: BrowserExperienceState;
}

export type AppUpdateStatus =
  | 'unavailable'
  | 'idle'
  | 'checking'
  | 'available'
  | 'deferred'
  | 'skipped'
  | 'cancelled'
  | 'downloading'
  | 'ready'
  | 'installing'
  | 'error';

export interface AppUpdateSnapshot {
  status: AppUpdateStatus;
  enabled: boolean;
  currentVersion: string;
  channel: 'stable' | 'beta';
  automaticChecks: boolean;
  lastCheckAt?: number;
  unavailableReason?: string;
  candidate?: {
    version: string;
    bundleVersion?: string;
    /** Ephemeral approval for the exact main-owned candidate; never sent to the feed. */
    approvalId?: string;
    releaseName?: string;
    releaseNotes?: string;
    releaseDate?: string;
    sizeBytes: number;
    unpackedSizeBytes: number;
    requiredFreeBytes: number;
  };
  progress?: {
    transferredBytes: number;
    totalBytes: number;
    percent: number;
    bytesPerSecond?: number;
  };
  error?: {
    code: string;
    message: string;
    requiredDiskBytes?: number;
    availableDiskBytes?: number;
  };
  restartBlockingReason?: string;
}

export type HuggingFacePageActionStatus =
  | 'inspecting'
  | 'loadable'
  | 'unverified'
  | 'downloading'
  | 'installed'
  | 'access-required'
  | 'unsupported'
  | 'blocked'
  | 'resource-error'
  | 'failed';

export interface HuggingFacePageActionState {
  implementationNotice?: string;
  implementationDetails?: string;
  status: HuggingFacePageActionStatus;
  repository: string;
  sourceURL: string;
  exactRevision?: string;
  task?: string;
  taskContract?: string;
  runner?: string;
  compatibility?: string;
  securityStatus?: 'allowed' | 'blocked';
  trustMode?: 'oryn-verified' | 'user-trusted';
  license?: string;
  selectedFiles: string[];
  diskBytes?: number;
  memoryBytes?: number;
  availableMemoryBytes?: number;
  downloadedBytes?: number;
  totalBytes?: number;
  requiredDiskBytes?: number;
  availableDiskBytes?: number;
  blockingReasons?: string[];
  message?: string;
  operationId?: string;
}

export interface HuggingFaceCredentialStatus {
  configured: boolean;
  secureStorageAvailable: boolean;
}

export interface BrowserHistoryItem {
  historyId: string;
  url: string;
  title: string;
  visitedAt: string;
}

export type BrowserBookmarkItem =
  | {
      kind: 'bookmark';
      bookmarkId: string;
      parentId: string | null;
      title: string;
      url: string;
      createdAt: string;
      order: number;
      favorite: boolean;
      trashedAt?: string;
      thumbnail?: string;
    }
  | {
      kind: 'folder';
      bookmarkId: string;
      parentId: string | null;
      title: string;
      createdAt: string;
      order: number;
      favorite: boolean;
      trashedAt?: string;
    };

export interface BrowserDownloadItem {
  downloadId: string;
  url: string;
  filename: string;
  receivedBytes: number;
  totalBytes: number;
  state: string;
  openState?: 'pending' | 'ready' | 'error';
  openError?: 'quarantine' | 'file-unavailable' | 'file-changed' | 'open-failed';
  startedAt: string;
  updatedAt: string;
}

export interface BrowserSitePermissionItem {
  origin: string;
  permission: string;
  decision: 'allow' | 'deny';
  updatedAt: string;
}

export interface SiteContext {
  tabId: string;
  documentId: string;
  origin: string;
  partition: string;
}

export interface CertificateReviewContext {
  tabId: string;
  documentId: string;
  url: string;
  fingerprint256: string;
}

export interface BrowserCoreDataSnapshot {
  certificateContext?: CertificateReviewContext;
  connection?: ConnectionSecuritySnapshot;
  captureAccess?: CaptureAccessSnapshot[];
  embeddedPermissions?: SitePermissionRecord[];
  siteContext?: SiteContext;
  recoveryNotices?: string[];
  history: BrowserHistoryItem[];
  bookmarks: BrowserBookmarkItem[];
  downloads: BrowserDownloadItem[];
  downloadDirectory?: string;
  sitePermissions: BrowserSitePermissionItem[];
  bookmarkBarVisible: boolean;
  currentOrigin?: string;
  appearance: BrowserAppearance;
}

export interface BrowserAppearance {
  searchProvider?: 'google' | 'duckduckgo' | 'bing';
  localSuggestions?: boolean;
  theme: 'light' | 'dark' | 'system';
  density: 'default' | 'compact' | 'comfortable';
  onboardingComplete: boolean;
}

export interface BrowserChromeLayout {
  /** Persistent chrome that owns website layout space. */
  contentHeight: number;
  /** Trusted view height used to paint transient overlays above the website. */
  overlayHeight: number;
  /** Makes a blocking trusted dialog cover the complete browser content area. */
  coverWindow: boolean;
  /** Keeps the separate sidebar rail available while a switchable sheet is open. */
  sidebarInteractive: boolean;
}

export interface BrowserPassiveOverlayLayout {
  visible: boolean;
  width: number;
  height: number;
}

export interface TrustedBrowserAPI {
  getState(): Promise<BrowserWindowState>;
  onAuthenticationRequest(callback: (request: { id: string; target: string }) => void): () => void;
  respondAuthentication(
    id: string,
    credentials: { username: string; password: string } | null,
  ): Promise<void>;
  setDefaultBrowser(): Promise<boolean>;
  newWindow(): Promise<void>;
  newPrivateWindow(url?: string): Promise<void>;
  openFile(): Promise<void>;
  stopCapture(webContentsId: number, generation: string): Promise<boolean>;
  exportCertificate(context: CertificateReviewContext): Promise<void>;
  revokeEmbeddedPermission(origin: string, permission: string, context: SiteContext): Promise<void>;
  closeWindow(): Promise<void>;
  quit(): Promise<void>;
  exportDiagnostics(): Promise<void>;
  newTab(url?: string): Promise<string>;
  activateTab(tabId: string): Promise<void>;
  closeTab(tabId?: string): Promise<void>;
  reopenClosedTab(): Promise<string | undefined>;
  navigate(value: string): Promise<void>;
  back(): Promise<void>;
  forward(): Promise<void>;
  reload(): Promise<void>;
  stop(): Promise<void>;
  openDevTools(): Promise<void>;
  getBrowserData(query?: string): Promise<BrowserCoreDataSnapshot>;
  deleteHistory(historyId: string): Promise<void>;
  clearHistory(range?: 'hour' | 'day' | 'all'): Promise<void>;
  clearBrowsingData(category: 'cookies' | 'cache' | 'permissions'): Promise<void>;
  importBookmarks(confirmToken?: string): Promise<{ token: string; count: number } | undefined>;
  discoverBrowserImports(): Promise<BrowserImportSource[]>;
  previewBrowserImport(request: {
    sourceId: string;
    dataTypes: BrowserImportDataType[];
  }): Promise<BrowserImportPreview | undefined>;
  commitBrowserImport(token: string): Promise<BrowserImportCommitResult>;
  cancelBrowserImport(token?: string): Promise<void>;
  exportBookmarks(): Promise<void>;
  addBookmark(title: string, url: string): Promise<void>;
  deleteBookmark(bookmarkId: string): Promise<void>;
  setBookmarkBarVisible(visible: boolean): Promise<void>;
  setSitePermission(
    permission: string,
    decision: 'allow' | 'deny' | 'ask',
    context: SiteContext,
  ): Promise<void>;
  clearCurrentSiteData(context: SiteContext): Promise<void>;
  controlDownload(id: string, action: string): Promise<void>;
  chooseDownloadDirectory(): Promise<string | undefined>;
  onBrowserDataChanged(callback: () => void): () => void;
  onFindResult(
    callback: (result: { matches: number; activeMatchOrdinal: number }) => void,
  ): () => void;
  findInPage(query: string, forward?: boolean): Promise<void>;
  stopFindInPage(): Promise<void>;
  showFindInPage(): Promise<void>;
  setZoom(action: 'in' | 'out' | 'reset'): Promise<void>;
  printPage(): Promise<void>;
  setChromeLayout(layout: BrowserChromeLayout): Promise<void>;
  dismissPanelAndClickPage(
    x: number,
    y: number,
    button: 'left' | 'middle' | 'right',
  ): Promise<void>;
  setPassiveOverlayLayout(layout: BrowserPassiveOverlayLayout): Promise<void>;
  updateAppearance(value: Partial<BrowserAppearance>): Promise<void>;
  updateBrowserExperience(command: BrowserExperienceCommand): Promise<void>;
  updateBookmarks(command: BrowserBookmarkCommand): Promise<void>;
  getAppUpdate(): Promise<AppUpdateSnapshot>;
  checkForAppUpdate(): Promise<AppUpdateSnapshot>;
  downloadAppUpdate(
    version: string,
    allowSkipped?: boolean,
    approvalId?: string,
  ): Promise<AppUpdateSnapshot>;
  cancelAppUpdateDownload(): Promise<AppUpdateSnapshot>;
  deferAppUpdate(version: string): Promise<AppUpdateSnapshot>;
  skipAppUpdate(version: string): Promise<AppUpdateSnapshot>;
  clearAppUpdateSkip(): Promise<AppUpdateSnapshot>;
  restartToInstallAppUpdate(version: string, approvalId?: string): Promise<AppUpdateSnapshot>;
  updateAppUpdateSettings(value: {
    automaticChecks?: boolean;
    channel?: 'stable' | 'beta';
  }): Promise<AppUpdateSnapshot>;
  onAppUpdateStateChanged(listener: (state: AppUpdateSnapshot) => void): () => void;
  confirmHuggingFacePageAction(trustMode: 'oryn-verified' | 'user-trusted'): Promise<void>;
  cancelHuggingFacePageAction(): Promise<void>;
  openHuggingFaceModelWorkspace(): Promise<string>;
  retryHuggingFacePageAction(): Promise<void>;
  showHuggingFacePageActionDetails(): Promise<void>;
  onHuggingFacePageActionDetails(listener: (state: BrowserWindowState) => void): () => void;
  getHuggingFaceCredentialStatus(): Promise<HuggingFaceCredentialStatus>;
  setHuggingFaceCredential(token: string): Promise<HuggingFaceCredentialStatus>;
  clearHuggingFaceCredential(): Promise<HuggingFaceCredentialStatus>;
  onStateChanged(listener: (state: BrowserWindowState) => void): () => void;
  onFocusOmnibox(listener: () => void): () => void;
  onOpenFind(listener: () => void): () => void;
  onOpenTabSearch(listener: () => void): () => void;
  onOpenPanel(
    listener: (
      panel: 'bookmarks' | 'history' | 'downloads' | 'settings' | 'local-ai' | null,
    ) => void,
  ): () => void;
  onLocalAIPermissionPrompt(listener: (prompt: LocalAIPermissionPrompt) => void): () => void;
  onLocalAIPermissionPromptCancelled(listener: (promptId: string) => void): () => void;
  respondToLocalAIPermission(response: LocalAIPermissionPromptResponse): Promise<boolean>;
  openLocalAIModelChooser(promptId: string): Promise<boolean>;
  getLocalAIJobActivity(): Promise<LocalAIJobActivity>;
  onLocalAIJobActivityChanged(listener: (activity: LocalAIJobActivity) => void): () => void;
  onModelOperationChanged(listener: (operation: TrustedModelOperation) => void): () => void;
  cancelModelOperation(operationId: string): Promise<void>;
  cancelLocalAIJob(jobId: string): Promise<void>;
  cancelLocalAIJobsForSite(origin: string): Promise<void>;
  cancelAllLocalAIJobs(): Promise<void>;
  getAIActivity(): Promise<AIActivitySnapshot>;
  updateAIActivitySettings(settings: Partial<AIActivitySettings>): Promise<AIActivitySettings>;
  clearAIActivity(): Promise<void>;
  saveLocalAIArtifact(artifactId: string): Promise<void>;
  getModelManager(): Promise<ModelManagerSnapshot>;
  getManagedModelSubstitutes(installationId: string): Promise<TrustedModelSubstitute[]>;
  getRegistryStatus(): Promise<RegistryStatus>;
  setRegistryRefresh(enabled: boolean): Promise<RegistryStatus>;
  checkBetterAlternatives(installationId: string): Promise<TrustedAlternativeSearch>;
  cancelBetterAlternatives(): Promise<void>;
  getEngineVersions(): Promise<EngineVersion[]>;
  checkEngineVersion(id: string): Promise<EngineVersion>;
  getManagedRuntimePacks(): Promise<
    Pick<ModelManagerSnapshot, 'runtimePacks' | 'runtimePackPolicy'>
  >;
  cleanupManagedRuntimePacks(): Promise<{ removedPacks: number; reclaimedBytes: number }>;
  unloadManagedModel(installationId: string): Promise<void>;
  setManagedModelRuntimeOverride(installationId: string, candidateId?: string): Promise<void>;
  previewManagedModelRemoval(installationId: string): Promise<TrustedModelRemovalPreview>;
  removeManagedModel(
    installationId: string,
    action: 'delete' | 'when-idle' | 'cancel-and-delete',
  ): Promise<'removed' | 'deferred'>;
  openManagedModelSource(installationId: string): Promise<void>;
  openModelSource(repository: string, revision?: string): Promise<void>;
  getOrynSettings(): Promise<OrynSettingsSnapshot>;
  updateOrynRanking(value: {
    preferInstalled?: boolean;
    optimization?: 'quality' | 'balanced' | 'speed' | 'memory';
  }): Promise<void>;
  updateOrynRolePreference(value: {
    role: string;
    model?: { provider: 'huggingface'; repository: string; revision?: string; files?: string[] };
    family?: string;
  }): Promise<void>;
  updateOrynCompute(value: Partial<OrynComputeSettings>): Promise<void>;
  resetOrynSiteChoices(origin: string): Promise<void>;
  askOrynSiteAgain(origin: string): Promise<void>;
  revokeOrynSite(origin: string, cancelActive: boolean): Promise<void>;
  cleanupOrynStorage(target: 'artifacts' | 'temporary'): Promise<void>;
  updateOrynDeveloper(value: Partial<OrynDeveloperSettings>): Promise<void>;
}

export interface TrustedModelOperation {
  operationId: string;
  origin: string;
  stage:
    | 'resolving'
    | 'awaiting-permission'
    | 'awaiting-model-choice'
    | 'downloading'
    | 'loading'
    | 'ready'
    | 'failed'
    | 'cancelled';
  repository?: string;
  revision?: string;
  downloadedBytes?: number;
  totalBytes?: number;
  progress?: number;
  errorCode?: string;
  requiredMemoryBytes?: number;
  availableMemoryBytes?: number;
  memoryBudgetBytes?: number;
  memoryPressure?: 'normal' | 'high' | 'critical';
}

export interface OrynComputeSettings {
  memoryBudgetBytes: number | null;
  backgroundJobs: 'allow' | 'ask' | 'disable';
  batteryMode: 'normal' | 'reduce' | 'disable';
  generationConcurrency: 1;
}

export interface OrynDeveloperSettings {
  showDiagnostics: boolean;
  showResolutionDetails: boolean;
  showProtocolDiagnostics: boolean;
  experimentalRunnersAllowed: false;
}

export interface OrynSiteSettings {
  origin: string;
  permissions: Array<{
    role: string;
    repository: string;
    decision: 'always-allow' | 'deny' | 'revoked';
    updatedAt: string;
  }>;
  modelChoices: Array<{ role: string; repository: string }>;
  activityCount: number;
  lastActivityAt?: number;
}

export interface OrynSettingsSnapshot {
  preferences: {
    preferInstalled: boolean;
    optimization: 'quality' | 'balanced' | 'speed' | 'memory';
    globalModels: Array<{
      role: string;
      model: { provider: 'huggingface'; repository: string; revision?: string; files?: string[] };
    }>;
    globalFamilies: Array<{ role: string; family: string }>;
    siteChoices: Array<{ origin: string; role: string; repository: string }>;
    availableModels: Array<{
      provider: 'huggingface';
      repository: string;
      revision: string;
      files?: string[];
    }>;
  };
  compute: OrynComputeSettings;
  sites: OrynSiteSettings[];
  storage: {
    modelBytes: number;
    modelReclaimableBytes: number;
    modelSharedBytes: number;
    artifactBytes: number;
    temporaryArtifactBytes: number;
    readyArtifacts: number;
    reclaimableArtifacts: number;
  };
  developer: OrynDeveloperSettings;
}

export interface RegistryStatus {
  enabled: boolean;
  configured: boolean;
  status: string;
  catalogVersion: number;
  publishedAt: string;
  stale: boolean;
}
export interface TrustedAlternativeSearch {
  suggestedRole: string;
  status: 'results' | 'no-results' | 'no-better-option';
  variants: Array<{
    model: { provider: 'huggingface'; repository: string; revision: string; files?: string[] };
    executable: boolean;
    lineage: string;
    reasons: string[];
    requiredMemoryBytes?: number;
    downloadBytes?: number;
  }>;
}
export interface TrustedModelSubstitute {
  repository: string;
  exactRevision: string;
  compatibility: 'certified' | 'compatible' | 'uncertain' | 'incompatible';
  developerCertified: boolean;
  reasons: string[];
  contextOrigin?: string;
  contextRole: string;
}

export interface TrustedInstalledModel {
  installationId: string;
  displayName: string;
  publisher: string;
  repository: string;
  exactRevision: string;
  sourceUrl: string;
  task?: string;
  modalities: string[];
  family?: string;
  parameterCount?: number;
  precision?: string;
  runner?: string;
  runtimePack?: {
    packId: string;
    packVersion: string;
    runnerId: string;
    taskContract: string;
    compatibilityLevel: 'native-declarative' | 'audited-adapter' | 'converted-artifact';
    trustedMetadata?: {
      license: string;
      source?: { provider: 'huggingface'; repository: string; revision: string };
      quantization?: string;
      contextLength?: number;
      chatTemplate?: TrustedModelMetadata['chatTemplate'];
      modelFile?: string;
      languages?: string[];
      voices?: Array<{ id: string; language: string; name: string }>;
      conformance: { status: 'passed' | 'fixture-only'; platforms: string[] };
    };
  };
  runtimeSelection?: TrustedRuntimeSelection;
  executableBackends: string[];
  logicalSizeBytes: number;
  reclaimableSizeBytes: number;
  estimatedResidentBytes?: number;
  installationState: 'installed' | 'downloading' | 'removing' | 'error';
  residencyState: 'unloaded' | 'loading' | 'loaded' | 'unloading' | 'error';
  activeSessions: number;
  runningJobs: number;
  queuedJobs: number;
  installedAt?: number;
  lastUsedAt?: number;
  usedBy: string[];
  preferredFor: string[];
  substitutes: TrustedModelSubstitute[];
}

export interface ModelManagerSnapshot {
  totalLogicalBytes: number;
  totalReclaimableBytes: number;
  models: TrustedInstalledModel[];
  runtimePacks: TrustedRuntimePack[];
  runtimePackPolicy: {
    catalogVersion: number;
    catalogDigest: string;
    thirdPartyPublishersEnabled: false;
    automaticExecutableUpdatesEnabled: false;
  };
}

export interface TrustedRuntimePack {
  packId: string;
  version: string;
  publisher: string;
  abi: string;
  dependencyClosure: string;
  platforms: Array<'macos-arm64' | 'windows-amd64' | 'linux-amd64'>;
  distribution: 'bundled' | 'download';
  state: 'available' | 'staged' | 'active' | 'revoked';
  bundled: boolean;
  removable: boolean;
  sizeBytes: number;
  minimumSafeVersion: string;
  revoked: boolean;
  license: {
    summary: string;
    notice: string;
    access: 'public' | 'gated';
    acceptanceRequired: boolean;
    accepted: boolean;
  };
}

export interface TrustedModelRemovalPreview {
  installationId: string;
  canRemoveNow: boolean;
  activeSessions: number;
  runningJobs: number;
  queuedJobs: number;
  reclaimableBytes: number;
  sharedBytesRetained: number;
  blockers: string[];
}

export interface LocalAIJobActivity {
  running: number;
  queued: number;
  jobs: TrustedLocalAIJob[];
}

export interface TrustedLocalAIJob {
  jobId: string;
  sessionId: string;
  origin: string;
  role: string;
  requestedModel: string;
  resolvedModel: string;
  substituted: boolean;
  state: string;
  progressStage?: string;
  progress?: number;
  queue?: { position: number; jobsAhead: number; reason: string };
  inputPreview?: string;
  runner: string;
  backend: string;
  cancellationRequested: boolean;
}

export interface AIActivitySettings {
  keepActivity: boolean;
  contentPolicy: 'full' | 'metadata-only';
  retentionDays: 7 | 30 | null;
}

export interface AIActivityRecord {
  id: string;
  createdAt: number;
  completedAt?: number;
  origin: string;
  role: string;
  sessionId: string;
  jobId: string;
  requestedModel: { provider: 'huggingface'; repository: string; revision: string };
  resolvedModel: { provider: 'huggingface'; repository: string; revision: string };
  resolution: {
    compatibility: string;
    substituted: boolean;
    overriddenByUser: boolean;
    reason: string;
  };
  invocation: {
    input?: unknown;
    artifacts?: Array<{
      artifactId: string;
      name?: string;
      mimeType: string;
      sizeBytes: number;
    }>;
    contentPolicy: 'full' | 'metadata-only';
  };
  execution: {
    runner: string;
    backend: string;
    device: string;
    durationMs?: number;
    status: 'running' | 'completed' | 'failed' | 'cancelled';
    planId?: string;
    foundationId?: string;
    adapterId?: string;
    candidateId?: string;
    variantId?: string;
    precision?: string;
    quantization?: string;
    fallbacks?: Array<{
      fromCandidateId: string;
      toCandidateId: string;
      reasonCode: string;
    }>;
  };
  result?: {
    kind: 'structured' | 'artifact' | 'multi';
    artifactIds?: string[];
    sizeBytes?: number;
  };
  errorCode?: string;
}

export interface AIActivitySnapshot {
  settings: AIActivitySettings;
  records: AIActivityRecord[];
}

export interface LocalAIPermissionPrompt {
  resolvedModel: {
    provider: 'huggingface';
    repository: string;
    revision: string;
    files?: string[];
  };
  selection?: SelectionExplanation;
  promptId: string;
  origin: string;
  role: string;
  requestedRepository: string;
  resolvedRepository: string;
  resolutionReason: string;
  compatibility: string;
  developerCertified: boolean;
  substituted: boolean;
  provider: 'huggingface';
  exactRevision: string;
  installed: boolean;
  downloadBytes?: number;
  estimatedMemoryBytes?: number;
  candidates: Array<{
    model: { provider: 'huggingface'; repository: string; revision?: string; files?: string[] };
    compatibility: string;
    developerCertified: boolean;
    installed: boolean;
    executable: boolean;
    securityAllowed: boolean;
    trustRequired: boolean;
    blockingReasons: string[];
    estimatedDiskBytes?: number;
    estimatedMemoryBytes?: number;
  }>;
}

export interface LocalAIPermissionPromptResponse {
  promptId: string;
  action: 'allow-once' | 'always-allow' | 'deny';
  selectedModel?: {
    provider: 'huggingface';
    repository: string;
    revision?: string;
    files?: string[];
  };
  rememberSiteChoice?: boolean;
}
