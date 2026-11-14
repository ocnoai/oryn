import { artifactIoRuntime } from './paths';
import { dataForWindow, WindowRegistryStore } from './data/window-data';
import { dataForPrivateWindow } from './data/private-window-data';
import { PrivateCleanupQueue } from './data/private-cleanup-queue';
import { cleanupImportSnapshots } from './data/browser-import-snapshot';
import { installBrowserRequestGate } from './insecure-form-guard';
import { randomUUID } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import registryReleaseConfig from './model-resolution/registry-release-config.json';
import { RegistryService } from './model-resolution/registry-service';
import {
  closeSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  statfsSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

import { Readable } from 'node:stream';

import type { CandidateExecutionProfile } from '@ocno/oryn-protocol';
import {
  app,
  BrowserWindow,
  dialog,
  Menu,
  powerMonitor,
  protocol,
  safeStorage,
  session,
  type WebContents,
  type MenuItemConstructorOptions,
} from 'electron';

import { BrowserCallerContextService, type BrowserCallerContext } from './browser-caller-context';
import { ArtifactManager } from './artifact-manager/artifact-manager';
import {
  ARTIFACT_SCHEME,
  ArtifactCapabilityService,
} from './artifact-manager/artifact-capability-service';
import { ArtifactProtocolHandler } from './artifact-manager/artifact-protocol-handler';
import { BrowserWindowController } from './browser-window';
import { BrowserDataService } from './data/browser-data-service';
import { HuggingFaceCredentialStore } from './data/hugging-face-credential-store';
import { EngineSupervisor } from './engine-supervisor';
import { installIpcHandlers } from './ipc-handlers';
import { BrowserJobRegistry } from './jobs/browser-job-registry';
import { runWorkspaceAcceptance } from './model-workspace/model-workspace-acceptance';
import { runModelManagerSmoke } from './model-manager/model-manager-smoke';
import { ModelManagerService } from './model-manager/model-manager-service';
import { ModelPreferenceStore } from './model-resolution/model-preference-store';
import { ModelResolver } from './model-resolution/model-resolver';
import { LocalAIPermissionStore } from './model-resolution/local-ai-permission-store';
import { PermissionPromptCoordinator } from './model-resolution/permission-prompt-coordinator';
import { BrowserModelSessionRegistry } from './model-resolution/browser-model-sessions';
import { OrynSettingsService } from './settings/oryn-settings-service';
import { packagedProfileRoot } from './paths';
import { browserApplicationName, readBrowserBuildFlavor } from './browser-build-flavor';
import { installNewTabProtocol } from './new-tab-protocol';
import { startReferenceDemoServers, type ReferenceDemoServer } from './reference-demo-server';
import { createOrynModelWorkspaceURL, ORYN_SCHEME } from '../shared/browser-urls';
import { IPC } from '../shared/ipc';
import { readAppUpdateBuildConfiguration } from './app-updates/app-update-build-config';
import { AppUpdateService } from './app-updates/app-update-service';
import { ElectronUpdaterAdapter } from './app-updates/electron-updater-adapter';
import { waitForSmokePage } from './smoke-page-readiness';
import { BrowserFailureDiagnostics } from './browser-diagnostics';
import {
  keychainPersistenceSmokeIdentity,
  runKeychainPersistenceSmoke,
} from './smoke-keychain-persistence';

let removeIpcHandlers: (() => void) | undefined;
let referenceDemoServers: ReferenceDemoServer[] = [];
let engineStartupMs = 0;
const browserFailureDiagnostics = new BrowserFailureDiagnostics();
app.on('render-process-gone', (_event, _contents, details) => {
  browserFailureDiagnostics.recordRendererExit(details.reason);
});
protocol.registerSchemesAsPrivileged([
  {
    scheme: ORYN_SCHEME,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
    },
  },
  {
    scheme: ARTIFACT_SCHEME,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      corsEnabled: true,
    },
  },
]);
const callerContexts = new BrowserCallerContextService();
const buildFlavor = readBrowserBuildFlavor(app.getAppPath());
const keychainSmokeIdentity = keychainPersistenceSmokeIdentity(
  browserApplicationName(buildFlavor),
  process.env,
);
// Electron 45 loads this main script before PostCreateMainMessageLoop derives
// its Keychain service/account from Browser::GetName(). Keep this before sessions.
app.setName(keychainSmokeIdentity.applicationName);
if (
  process.env.LOCAL_AI_SMOKE_TEST === '1' &&
  process.env.ORYN_KEYCHAIN_PERSISTENCE_SMOKE === '1'
) {
  console.log(`ORYN_KEYCHAIN_PERSISTENCE_SMOKE_SCOPE ${JSON.stringify(keychainSmokeIdentity)}`);
}
if (process.env.LOCAL_AI_SMOKE_TEST === '1') {
  const configuredSmokeProfile = process.env.ORYN_SMOKE_PROFILE_ROOT;
  const smokeProfile = configuredSmokeProfile
    ? resolve(configuredSmokeProfile)
    : mkdtempSync(join(tmpdir(), 'oryn-browser-smoke-'));
  mkdirSync(smokeProfile, { recursive: true, mode: 0o700 });
  app.setPath('userData', smokeProfile);
  if (process.env.ORYN_KEYCHAIN_PERSISTENCE_SMOKE === '1') {
    app.setPath('sessionData', smokeProfile);
  }
} else if (process.env.ORYN_PROFILE_ROOT) {
  const profile = resolve(process.env.ORYN_PROFILE_ROOT);
  mkdirSync(profile, { recursive: true, mode: 0o700 });
  app.setPath('userData', profile);
} else if (app.isPackaged) {
  app.setPath('userData', packagedProfileRoot(app.getPath('appData'), buildFlavor));
}
const browserData = new BrowserDataService(app.getPath('userData'));
const privateCleanup = new PrivateCleanupQueue(app.getPath('userData'));
let privateCleanupReady = false;
const huggingFaceCredentials = new HuggingFaceCredentialStore(
  browserData.paths.huggingFaceCredential,
  {
    isAvailable: () => safeStorage.isEncryptionAvailable(),
    encrypt: (value) => safeStorage.encryptString(value),
    decrypt: (value) => safeStorage.decryptString(value),
  },
);
const artifacts = new ArtifactManager(
  browserData.paths.artifactRegistry,
  join(browserData.paths.localAIRoot, 'artifacts'),
  {
    artifactIo: artifactIoRuntime(),
    hasDocumentAuthority: (owner) =>
      callerContexts.hasDocumentAuthority(owner.tabId, owner.webContentsId, owner.documentId),
  },
);
callerContexts.on('document-invalidated', (owner) => {
  void artifacts.invalidateDocument(owner).catch(() => {
    console.error(
      'Artifact cleanup failed after document revocation; startup reconciliation will retry.',
    );
  });
});
const engine = new EngineSupervisor({
  modelRoot: join(browserData.paths.localAIRoot, 'models'),
  runtimePackRoot: join(browserData.paths.localAIRoot, 'runtime-packs'),
  artifactInputRoot: artifacts.storage.inputRoot,
  artifactOutputRoot: artifacts.storage.engineStagingRoot,
  runtimeScratchRoot: join(browserData.paths.localAIRoot, 'runtime-temp'),
  huggingFaceToken: () => huggingFaceCredentials.token(),
});
const modelPreferences = new ModelPreferenceStore(browserData.paths.localAIPreferences);
const recommendationRegistry = new RegistryService(
  join(browserData.paths.localAIRoot, 'recommendation-catalog.json'),
  { endpoint: registryReleaseConfig.endpoint ?? undefined, keys: registryReleaseConfig.keys },
);
const modelResolver = new ModelResolver(engine, modelPreferences, recommendationRegistry);
const localAIPermissions = new LocalAIPermissionStore(browserData.paths.localAIPermissions);
const permissionPrompts = new PermissionPromptCoordinator();
const modelSessions = new BrowserModelSessionRegistry(
  engine,
  modelResolver,
  localAIPermissions,
  modelPreferences,
  permissionPrompts,
  callerContexts,
);
const jobs = new BrowserJobRegistry(
  engine,
  modelSessions,
  artifacts,
  callerContexts,
  browserData.aiActivity,
);
const modelManager = new ModelManagerService(
  engine,
  modelResolver,
  modelPreferences,
  modelSessions,
  jobs,
  browserData.aiActivity,
  recommendationRegistry,
  engine,
);
const orynSettings = new OrynSettingsService(
  browserData.orynSettings,
  engine,
  modelPreferences,
  localAIPermissions,
  modelSessions,
  jobs,
  browserData.aiActivity,
  artifacts,
);
const browserControllers = new Set<BrowserWindowController>();
const windowData = new Map<BrowserWindowController, BrowserDataService>();
const privateCleanupTasks = new Set<Promise<void>>();

async function finishPreparedWindows(privateOnly = false): Promise<void> {
  await Promise.all(
    [...browserControllers]
      .filter((owner) => !privateOnly || owner.isPrivate)
      .map((owner) => {
        if (owner.window.isDestroyed()) return;
        return new Promise<void>((resolveClosed) => {
          owner.window.once('closed', resolveClosed);
          owner.window.close();
        });
      }),
  );
  while (privateCleanupTasks.size) await Promise.allSettled([...privateCleanupTasks]);
}
const windowRegistry = new WindowRegistryStore(browserData.paths.browserCoreRoot);
const publishCaptureState = () => {
  for (const owner of browserControllers) owner.refreshCaptureState();
};
browserData.subscribeCaptureAccess(publishCaptureState);
let sharedServicesStarted = false;
let updateBuildConfiguration: ReturnType<typeof readAppUpdateBuildConfiguration> = {
  enabled: false,
};
try {
  updateBuildConfiguration = readAppUpdateBuildConfiguration({
    packaged: app.isPackaged,
    appPath: app.getAppPath(),
  });
  if (updateBuildConfiguration.enabled && updateBuildConfiguration.channel) {
    browserData.appUpdates.initializeChannel(updateBuildConfiguration.channel);
  }
} catch (error) {
  updateBuildConfiguration = { enabled: false };
  console.error('The packaged app update configuration is invalid.', error);
}
const updateVerificationRoot = join(browserData.paths.browserCoreRoot, 'update-verification');
let appUpdateAdapter: ElectronUpdaterAdapter | undefined;
try {
  if (
    updateBuildConfiguration.enabled &&
    updateBuildConfiguration.baseUrl &&
    updateBuildConfiguration.expectedTeamId
  ) {
    mkdirSync(updateVerificationRoot, { recursive: true, mode: 0o700 });
    appUpdateAdapter = new ElectronUpdaterAdapter(updateBuildConfiguration.baseUrl, {
      stagedRolloutQualified: updateBuildConfiguration.stagedRolloutQualified === true,
      expectedTeamId: updateBuildConfiguration.expectedTeamId,
      expectedAppId: 'ai.ocno.oryn.browser',
      currentVersion: app.getVersion(),
      currentSystemVersion: process.getSystemVersion(),
      verificationRoot: updateVerificationRoot,
    });
  }
} catch {
  updateBuildConfiguration = { enabled: false };
  console.error(
    'Application updates are unavailable because verification storage could not be initialized.',
  );
}
app.on('login', (event, _webContents, details, _authenticationInfo, callback) => {
  if (!updateBuildConfiguration.baseUrl) return;
  try {
    if (new URL(details.url).origin !== new URL(updateBuildConfiguration.baseUrl).origin) return;
  } catch {
    return;
  }
  event.preventDefault();
  callback();
});
const appUpdates = new AppUpdateService({
  enabled: Boolean(appUpdateAdapter),
  currentVersion: app.getVersion(),
  settings: browserData.appUpdates,
  adapter: appUpdateAdapter,
  availableDiskBytes: () =>
    // electron-updater uses ~/Library/Caches, not the Browser profile. Include
    // the real cache volume as well as native staging and the install volume.
    [
      join(homedir(), 'Library', 'Caches'),
      updateVerificationRoot,
      tmpdir(),
      dirname(app.getPath('exe')),
    ].map((path) => {
      const stats = statfsSync(path);
      return stats.bavail * stats.bsize;
    }),
  prepareInstall: prepareAppUpdateInstall,
  recoverInstall: recoverAppUpdateInstall,
});

async function createBrowser(
  windowId?: string,
  openingURL?: string,
  privateMode = false,
): Promise<BrowserWindowController> {
  if (privateMode && !privateCleanupReady) throw new Error('PRIVATE_CLEANUP_UNAVAILABLE');
  if (process.env.LOCAL_AI_SMOKE_TEST === '1') {
    const smokeDownloads = join(app.getPath('userData'), 'downloads');
    mkdirSync(smokeDownloads, { recursive: true, mode: 0o700 });
    session.defaultSession.setDownloadPath(smokeDownloads);
    session.defaultSession.on('will-download', (_event, item) => {
      item.setSavePath(join(smokeDownloads, item.getFilename()));
    });
  }
  if (shutdownStarted) throw new Error('BROWSER_CLOSING');
  windowId ??= privateMode ? `private_${randomUUID()}` : windowRegistry.add();
  const scopedData = privateMode
    ? dataForPrivateWindow(browserData)
    : dataForWindow(browserData, windowId);
  const privateNamespace = privateMode ? privateCleanup.register() : undefined;
  const privateSession = privateMode
    ? session.fromPartition(`oryn-private-${randomUUID()}`, { cache: false })
    : undefined;
  if (privateSession) {
    installBrowserRequestGate(privateSession);
    installNewTabProtocol(privateSession, { privateMode: true });
  }
  const controller = new BrowserWindowController(engine, callerContexts, scopedData, {
    captureAccess: () => [
      ...new Map(
        [...windowData.values(), scopedData]
          .flatMap((data) => data.captureSnapshot())
          .map((entry) => [entry.generation, entry]),
      ).values(),
    ],
    privateMode,
    session: privateSession,
    openWindow: (url) => {
      void createBrowser(undefined, url, privateMode);
    },
    openPrivateWindow: (url) => {
      void createBrowser(undefined, url, true);
    },
    canClose: async () => {
      if (privateMode) {
        if (!scopedData.hasActiveDownloads()) return true;
        const answer = await dialog.showMessageBox({
          type: 'question',
          message: 'Downloads are still active',
          detail:
            'Closing this private window cancels its active downloads. Files you explicitly saved remain.',
          buttons: ['Keep window open', 'Cancel downloads and close'],
          defaultId: 0,
          cancelId: 0,
        });
        if (answer.response !== 1) return false;
        for (const item of scopedData.downloads.list())
          if (['progressing', 'paused'].includes(item.state))
            await scopedData.controlDownload(item.downloadId, 'cancel');
        return true;
      }
      const activity = jobs.activitySnapshot();
      if (
        !activity.running &&
        !activity.queued &&
        !artifacts.hasPendingWrites() &&
        ![...windowData.values()].some((data) => data.hasActiveDownloads()) &&
        ![...browserControllers].some(
          (owner) => owner.getState().huggingFacePageAction?.status === 'downloading',
        )
      )
        return true;
      await dialog.showMessageBox({
        message: 'Work is still active',
        detail: 'Finish or cancel Local AI work, exports and downloads before closing the window.',
        buttons: ['Keep window open'],
      });
      return false;
    },
  });
  windowData.set(controller, scopedData);
  if (openingURL) controller.navigate(openingURL);
  browserControllers.add(controller);
  const stopPrivateCaptureUpdates = privateMode
    ? scopedData.subscribeCaptureAccess(publishCaptureState)
    : undefined;
  controller.window.once('closed', () => {
    stopPrivateCaptureUpdates?.();
    browserControllers.delete(controller);
    windowData.delete(controller);
    if (!shutdownStarted && !privateMode) windowRegistry.remove(windowId!);
    if (privateSession && privateNamespace) {
      const cleanup = (async () => {
        try {
          await privateSession.closeAllConnections();
          await privateSession.clearStorageData();
          await privateSession.clearCache();
          await privateSession.clearAuthCache();
          if (!privateCleanup.cleanup(privateNamespace.id))
            throw new Error('PRIVATE_CLEANUP_PENDING');
        } catch {
          await dialog.showMessageBox({
            type: 'warning',
            message: 'Private temporary-file cleanup is pending',
            detail:
              'Oryn will retry cleanup the next time it starts. Files you explicitly saved are kept.',
            buttons: ['OK'],
          });
        }
      })();
      privateCleanupTasks.add(cleanup);
      void cleanup.finally(() => privateCleanupTasks.delete(cleanup)).catch(() => undefined);
    }
  });
  if (!removeIpcHandlers)
    removeIpcHandlers = installIpcHandlers(
      controller,
      engine,
      callerContexts,
      browserData,
      modelResolver,
      modelSessions,
      permissionPrompts,
      artifacts,
      jobs,
      modelManager,
      orynSettings,
      appUpdates,
      huggingFaceCredentials,
      {
        all: () => [...browserControllers],
        rendererCrashes: () => browserFailureDiagnostics.rendererCrashes,
        admitsActions: () => !shutdownStarted,
        data: (owner) => windowData.get(owner)!,
        create: async () => {
          await createBrowser();
        },
        createPrivate: async (url) => {
          await createBrowser(undefined, url, true);
        },
        transfer: async (owner, tabId) => {
          if (owner.isPrivate) throw new Error('PRIVATE_TAB_MOVE_UNAVAILABLE');
          const target = await createBrowser();
          owner.tabs.transferTabTo(tabId, target.tabs);
        },
      },
    );
  controller.showWhenReady();
  if (sharedServicesStarted) return controller;
  sharedServicesStarted = true;
  appUpdates.start();
  void recommendationRegistry.refresh();
  setInterval(() => void recommendationRegistry.refresh(), 6 * 3600000).unref();

  const engineStartedAt = performance.now();
  void engine
    .start()
    .then(() => {
      engineStartupMs = performance.now() - engineStartedAt;
    })
    .catch((error) => {
      console.error('Oryn Engine failed to start:', error);
    });

  if (process.env.LOCAL_AI_SMOKE_TEST === '1') {
    controller.activeWebsiteWebContents.once('did-finish-load', () => {
      void runSmokeTest(controller);
    });
  }
  return controller;
}

async function runSmokeTest(controller: BrowserWindowController): Promise<void> {
  try {
    controller.trustedWebContents.setBackgroundThrottling(false);
    await waitForSmokePage(controller.activeWebsiteWebContents, {
      expectedURL:
        process.env.ORYN_NETWORK_ERROR_SMOKE === '1' ? undefined : process.env.ORYN_START_URL,
      label: 'initial smoke document',
      ignoreLoadFailures: process.env.ORYN_NETWORK_ERROR_SMOKE === '1',
    });
    if (
      process.env.ORYN_KEYCHAIN_PERSISTENCE_SMOKE === '1' &&
      !process.env.ORYN_SMOKE_PROFILE_ROOT
    ) {
      throw new Error('Keychain persistence smoke requires a retained ORYN_SMOKE_PROFILE_ROOT.');
    }
    const keychainPersistence =
      process.env.ORYN_KEYCHAIN_PERSISTENCE_SMOKE === '1'
        ? await runKeychainPersistenceSmoke({
            identity: keychainSmokeIdentity,
            profileRoot: app.getPath('userData'),
            fixtureURL: process.env.ORYN_START_URL ?? 'http://127.0.0.1/',
            mockKeychain: app.commandLine.hasSwitch('use-mock-keychain'),
            protection: safeStorage,
            websiteSession: controller.activeWebsiteWebContents.session,
          })
        : undefined;
    if (keychainPersistence) {
      console.log(`ORYN_KEYCHAIN_PERSISTENCE_SMOKE_RESULT ${JSON.stringify(keychainPersistence)}`);
    }
    // Automation opens overlays while the app may be occluded. Keep this test
    // renderer painting so native resize updates reach its viewport.
    const initialTabId = controller.activeTabId;
    const initialDocument = (await controller.activeWebsiteWebContents.executeJavaScript(`({
      title: document.title,
      bodyText: document.body?.innerText ?? '',
      styleSheetCount: document.styleSheets.length,
      bodyWidth: document.body?.getBoundingClientRect().width ?? 0,
      bodyHeight: document.body?.getBoundingClientRect().height ?? 0
    })`)) as {
      title: string;
      bodyText: string;
      styleSheetCount: number;
      bodyWidth: number;
      bodyHeight: number;
    };
    const expectedInitialTitle = process.env.ORYN_EXPECT_START_TITLE;
    const initialDocumentRendered =
      (!expectedInitialTitle || initialDocument.title === expectedInitialTitle) &&
      initialDocument.bodyText.trim().length > 0 &&
      initialDocument.styleSheetCount > 0 &&
      initialDocument.bodyWidth > 0 &&
      initialDocument.bodyHeight > 0;
    if (process.env.ORYN_FIND_OVERLAY_SMOKE === '1') {
      await waitUntilLoaded(controller.trustedWebContents);
      await waitUntilLoaded(controller.passiveOverlayView.webContents);
      await waitForCondition(
        () =>
          controller.passiveOverlayView.webContents
            .executeJavaScript("Boolean(document.querySelector('.passive-overlay-surface'))")
            .then(Boolean),
        'passive overlay renderer',
      );
      await waitForCondition(
        () =>
          controller.trustedWebContents
            .executeJavaScript(
              `Boolean(document.querySelector('[aria-labelledby="oryn-onboarding-title"] button'))`,
            )
            .then(Boolean),
        'browser onboarding controls',
      );
      await controller.trustedWebContents.executeJavaScript(`
        Array.from(document.querySelector('[aria-labelledby="oryn-onboarding-title"]')
          ?.querySelectorAll('button') ?? [])
          .find((button) => button.textContent?.includes('Start browsing'))?.click()
      `);
      await waitForCondition(
        () => !controller.isTrustedChromeCoveringWindow(),
        'browser onboarding dismissal',
      );
      const websiteBoundsBefore = controller.tabs.getTab(controller.activeTabId).view.getBounds();
      controller.showFindInPage();
      await waitForCondition(async () => {
        const ready = await controller.passiveOverlayView.webContents.executeJavaScript(
          `window.innerHeight > 0 && document.activeElement?.getAttribute('aria-label') === 'Find in page'`,
        );
        return Boolean(ready) && controller.passiveOverlayView.getVisible();
      }, 'find overlay focus and layout');
      const findBounds = controller.passiveOverlayView.getBounds();
      const trustedBounds = controller.trustedView.getBounds();
      const websiteBoundsDuring = controller.tabs.getTab(controller.activeTabId).view.getBounds();
      const findGeometry = (await controller.passiveOverlayView.webContents.executeJavaScript(`
        (() => {
          const bar = document.querySelector('.find-bar');
          const surface = document.querySelector('.passive-overlay-surface');
          if (!bar || !surface) return null;
          const bounds = bar.getBoundingClientRect();
          return {
            top: bounds.top,
            bottom: bounds.bottom,
            surfaceHeight: surface.getBoundingClientRect().height,
            surfaceScrollHeight: surface.scrollHeight,
            viewportHeight: window.innerHeight,
          };
        })()
      `)) as {
        top: number;
        bottom: number;
        surfaceHeight: number;
        surfaceScrollHeight: number;
        viewportHeight: number;
      } | null;
      await controller.passiveOverlayView.webContents.executeJavaScript(
        `document.querySelector('[aria-label="Close find"]')?.click()`,
      );
      await waitForCondition(
        () =>
          controller.passiveOverlayView.webContents
            .executeJavaScript("!document.querySelector('.find-bar')")
            .then(Boolean),
        'find overlay dismissal',
      );
      await waitForCondition(
        () => !controller.passiveOverlayView.getVisible(),
        'find overlay view dismissal',
      );
      const passed =
        initialDocumentRendered &&
        JSON.stringify(websiteBoundsBefore) === JSON.stringify(websiteBoundsDuring) &&
        findBounds.y >= websiteBoundsBefore.y &&
        findBounds.x >= websiteBoundsBefore.x &&
        findBounds.x + findBounds.width <= websiteBoundsBefore.x + websiteBoundsBefore.width &&
        trustedBounds.height <= websiteBoundsBefore.y &&
        findGeometry !== null &&
        findGeometry.top >= 0 &&
        findGeometry.bottom <= findGeometry.viewportHeight;
      console.log(
        `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({ passed, websiteBoundsBefore, findBounds, trustedBounds, findGeometry })}`,
      );
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    if (process.env.ORYN_NETWORK_ERROR_SMOKE === '1') {
      const failedURL = process.env.ORYN_START_URL;
      const errorPage = (await controller.activeWebsiteWebContents.executeJavaScript(`({
        retryURL: document.querySelector('a')?.href ?? '',
        bodyText: document.body?.innerText ?? ''
      })`)) as { retryURL: string; bodyText: string };
      const displayedURL = controller.getState().tabs[0]?.url;
      const passed =
        initialDocumentRendered &&
        initialDocument.title === 'Page unavailable' &&
        errorPage.bodyText.includes('This page could not be reached') &&
        errorPage.bodyText.includes('local address') &&
        errorPage.retryURL === failedURL &&
        displayedURL === failedURL;
      console.log(
        `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({
          passed,
          initialDocument,
          initialDocumentRendered,
          errorPage,
          displayedURL,
        })}`,
      );
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    if (process.env.ORYN_QWEN35_CHOOSER_SMOKE === '1') {
      const chooser = await runQwen35ChooserSmoke(controller);
      const passed = initialDocumentRendered && chooser.passed;
      console.log(
        `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({
          passed,
          initialDocument,
          initialDocumentRendered,
          chooser,
        })}`,
      );
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    if (process.env.ORYN_MODEL_MANAGER_SMOKE === '1') {
      const manager = await runModelManagerSmoke(controller, modelManager);
      const passed = initialDocumentRendered && manager.passed;
      console.log(`LOCAL_AI_SMOKE_RESULT ${JSON.stringify({ passed, manager })}`);
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    if (
      process.env.ORYN_WORKSPACE_ACCEPTANCE_CASES &&
      process.env.ORYN_WORKSPACE_ACCEPTANCE_REPORT
    ) {
      const acceptance = await runWorkspaceAcceptance(
        controller,
        engine,
        browserData.modelWorkspaces,
        modelResolver,
        process.env.ORYN_WORKSPACE_ACCEPTANCE_CASES,
        process.env.ORYN_WORKSPACE_ACCEPTANCE_REPORT,
      );
      console.log(`LOCAL_AI_SMOKE_RESULT ${JSON.stringify(acceptance)}`);
      finishSmokeTest(acceptance.passed ? 0 : 1);
      return;
    }
    if (process.env.ORYN_MODEL_WORKSPACE_SMOKE === '1') {
      const workspace = await runModelWorkspaceSmoke(controller);
      const passed = initialDocumentRendered && workspace.passed;
      console.log(
        `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({ passed, initialDocument, initialDocumentRendered, keychainPersistence, workspace })}`,
      );
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    if (process.env.ORYN_HF_USER_TRUST_SMOKE === '1') {
      const trust = await runHuggingFaceUserTrustSmoke(controller);
      const passed = initialDocumentRendered && trust.passed;
      console.log(
        `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({ passed, initialDocument, initialDocumentRendered, trust })}`,
      );
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    if (process.env.ORYN_HF_GATED_ACCESS_SMOKE === '1') {
      const access = await runHuggingFaceGatedAccessSmoke(controller);
      const passed = initialDocumentRendered && access.passed;
      console.log(
        `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({ passed, initialDocument, initialDocumentRendered, access })}`,
      );
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    const secondInitialUrl = process.env.ORYN_SECOND_SITE_URL ?? 'https://example.org';
    const secondTabId = controller.createTab(secondInitialUrl, true);
    await waitUntilLoaded(
      controller.activeWebsiteWebContents,
      secondInitialUrl,
      'second smoke tab',
    );
    const thirdInitialUrl = process.env.ORYN_THIRD_SITE_URL ?? 'https://example.net';
    const thirdTabId = controller.createTab(thirdInitialUrl, true);
    await waitUntilLoaded(controller.activeWebsiteWebContents, thirdInitialUrl, 'third smoke tab');
    const originalTabIds = [initialTabId, secondTabId, thirdTabId];

    const perTab = [] as Array<{
      tabId: string;
      nodeType: string;
      requireType: string;
      electronType: string;
      capabilities: {
        runtimeAvailable: boolean;
        runtimeState: string;
        features: { modelRequests: boolean; taskProfiles?: boolean };
      };
      marker: string;
      publicModelAPI: boolean;
      globalInventoryHidden: boolean;
      documentId: string;
      origin: string;
    }>;

    for (const tabId of originalTabIds) {
      controller.activateTab(tabId);
      const contents = controller.activeWebsiteWebContents;
      await waitUntilLoaded(contents);
      const result = (await contents.executeJavaScript(`
        (async () => {
          globalThis.__orynTabMarker = ${JSON.stringify(tabId)};
          return {
            nodeType: typeof globalThis.process,
            requireType: typeof globalThis.require,
            electronType: typeof globalThis.electron,
            capabilities: await globalThis.localAI.getCapabilities(),
            marker: globalThis.__orynTabMarker,
            publicModelAPI:
              typeof globalThis.localAI.requestModel === 'function' &&
              typeof globalThis.localAI.cancelModelRequest === 'function' &&
              typeof globalThis.localAI.releaseModelSession === 'function' &&
              typeof globalThis.localAI.beginArtifactCreate === 'function' &&
              typeof globalThis.localAI.appendArtifactChunk === 'function' &&
              typeof globalThis.localAI.finishArtifactCreate === 'function' &&
              typeof globalThis.localAI.getArtifact === 'function' &&
              typeof globalThis.localAI.getArtifactUrl === 'function' &&
              typeof globalThis.localAI.saveArtifactAs === 'function' &&
              typeof globalThis.localAI.releaseArtifact === 'function' &&
              typeof globalThis.localAI.invokeJob === 'function' &&
              typeof globalThis.localAI.cancelJob === 'function' &&
              typeof globalThis.localAI.onJobEvent === 'function',
            globalInventoryHidden:
              typeof globalThis.browser === 'undefined' &&
              typeof globalThis.ipcRenderer === 'undefined' &&
              Object.isFrozen(globalThis.localAI) &&
              !('listInstalled' in globalThis.localAI) &&
              !('settings' in globalThis.localAI) &&
              !('activity' in globalThis.localAI) &&
              !('getAIActivity' in globalThis.localAI) &&
              !('clearAIActivity' in globalThis.localAI) &&
              !('askOrynSiteAgain' in globalThis.localAI) &&
              !('cancelAllLocalAIJobs' in globalThis.localAI) &&
              !('getModelManager' in globalThis.localAI) &&
              !('unloadManagedModel' in globalThis.localAI) &&
              !('removeManagedModel' in globalThis.localAI)
          };
        })()
      `)) as {
        nodeType: string;
        requireType: string;
        electronType: string;
        capabilities: {
          runtimeAvailable: boolean;
          runtimeState: string;
          features: { modelRequests: boolean };
        };
        marker: string;
        publicModelAPI: boolean;
        globalInventoryHidden: boolean;
      };
      const tab = controller.tabs.getTab(tabId);
      perTab.push({
        tabId,
        ...result,
        documentId: tab.activeDocumentId,
        origin: new URL(tab.url).origin,
      });
    }

    const distinctOrigins = new Set(perTab.map((tab) => tab.origin)).size === 3;
    const isolatedEverywhere = perTab.every(
      (tab) =>
        tab.nodeType === 'undefined' &&
        tab.requireType === 'undefined' &&
        tab.electronType === 'undefined',
    );
    const capabilitiesEverywhere = perTab.every(
      (tab) => tab.capabilities.runtimeAvailable && tab.capabilities.runtimeState === 'ready',
    );
    const modelInspectionAdvertised = perTab.every(
      (tab) => tab.capabilities.features.modelRequests,
    );
    const publicModelAPIEverywhere = perTab.every(
      (tab) => tab.publicModelAPI && tab.globalInventoryHidden,
    );
    const transcriptionFoundation = await runTranscriptionFoundation(controller, initialTabId);
    const chatFoundation = await runChatFoundation(controller, initialTabId);
    if (process.env.ORYN_SHARED_TASK_SMOKE === '1') {
      await runSharedTaskSmoke(controller, initialTabId);
    }
    const songFoundation = await runSongFoundation(controller, initialTabId, secondTabId);
    if (process.env.ORYN_TASK_CONTRACT_SMOKE === '1') {
      const requested = [
        process.env.ORYN_TRANSCRIPTION_SMOKE,
        process.env.ORYN_CHAT_SMOKE,
        process.env.ORYN_SONG_SMOKE,
      ].some((value) => value === '1');
      const taskProfilesAdvertised = perTab.every((tab) => tab.capabilities.features.taskProfiles);
      const passed =
        requested &&
        taskProfilesAdvertised &&
        isolatedEverywhere &&
        publicModelAPIEverywhere &&
        transcriptionFoundation.passed &&
        chatFoundation.passed &&
        songFoundation.passed;
      console.log(
        `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({
          passed,
          scope: 'task-contracts',
          keychainPersistence,
          taskProfilesAdvertised,
          isolatedEverywhere,
          publicModelAPIEverywhere,
          transcriptionFoundation,
          chatFoundation,
          songFoundation,
        })}`,
      );
      finishSmokeTest(passed ? 0 : 1);
      return;
    }
    const alphaFoundation = await runDeveloperAlphaFoundation(
      controller,
      initialTabId,
      secondTabId,
    );
    const artifactOutputBoundary = await runArtifactOutputSmoke(
      controller,
      initialTabId,
      thirdTabId,
    );
    controller.activateTab(thirdTabId);
    const websiteBridgeBoundary = (await controller.activeWebsiteWebContents.executeJavaScript(`
      (async () => {
        const progress = [];
        const operationId = 'model_request_smoke_boundary';
        const stop = globalThis.localAI.onModelRequestProgress(operationId, (value) => progress.push(value.stage));
        const result = await globalThis.localAI.requestModel(operationId, {
          role: 'chat',
          preferredModel: { provider: 'huggingface', repository: '../invalid' }
        });
        await new Promise((resolve) => setTimeout(resolve, 50));
        stop();
        const release = await globalThis.localAI.releaseModelSession('session_unknown');
        const spoofedOrigin = await globalThis.localAI.requestModel('model_request_spoofed_origin', {
          role: 'chat',
          origin: 'https://spoofed.example',
          preferredModel: { provider: 'huggingface', repository: 'Qwen/Qwen3.5-0.8B' }
        });
        const serialized = JSON.stringify({ result, release, spoofedOrigin });
        return {
          progress: progress.includes('resolving'),
          invalidRequestSanitized:
            result.ok === false &&
            result.error.code === 'MODEL_REFERENCE_INVALID' &&
            !serialized.includes('/Users/') &&
            !serialized.includes('Traceback'),
          releaseOwnershipEnforced:
            release.ok === false && release.error.code === 'SESSION_ACCESS_DENIED',
          originSpoofRejected:
            spoofedOrigin.ok === false && spoofedOrigin.error.code === 'MODEL_REFERENCE_INVALID'
        };
      })()
    `)) as {
      progress: boolean;
      invalidRequestSanitized: boolean;
      releaseOwnershipEnforced: boolean;
      originSpoofRejected: boolean;
    };
    const websiteModelBoundary = Object.values(websiteBridgeBoundary).every(Boolean);

    controller.activateTab(thirdTabId);
    const createdArtifact = (await controller.activeWebsiteWebContents.executeJavaScript(`
      (async () => {
        const total = 700000;
        const begin = await globalThis.localAI.beginArtifactCreate({
          mimeType: 'audio/wav', sizeBytes: total, name: 'smoke.wav', lifetime: 'session'
        });
        if (!begin.ok) return { ok: false, begin };
        const sizes = [262144, 262144, total - 524288];
        for (const size of sizes) {
          const appended = await globalThis.localAI.appendArtifactChunk(
            begin.value.uploadId,
            new Uint8Array(size).buffer
          );
          if (!appended.ok) return { ok: false, appended };
        }
        const finished = await globalThis.localAI.finishArtifactCreate(begin.value.uploadId);
        if (!finished.ok) return { ok: false, finished };
        const metadata = await globalThis.localAI.getArtifact(finished.value.id);
        return { ok: metadata.ok, artifact: finished.value, metadata };
      })()
    `)) as {
      ok: boolean;
      artifact?: { id: string; sizeBytes: number; internalPath?: string };
      metadata?: { ok: boolean; value?: { internalPath?: string } };
    };
    controller.activateTab(initialTabId);
    const crossOriginArtifact = (await controller.activeWebsiteWebContents.executeJavaScript(`
      globalThis.localAI.getArtifact(${JSON.stringify(createdArtifact.artifact?.id ?? '')})
    `)) as { ok: boolean; error?: { code?: string } };
    controller.activateTab(thirdTabId);
    const releasedArtifact = (await controller.activeWebsiteWebContents.executeJavaScript(`
      (async () => {
        const id = ${JSON.stringify(createdArtifact.artifact?.id ?? '')};
        const release = await globalThis.localAI.releaseArtifact(id);
        const after = await globalThis.localAI.getArtifact(id);
        return { release, after };
      })()
    `)) as {
      release: { ok: boolean };
      after: { ok: boolean; error?: { code?: string } };
    };
    const artifactInputBoundary =
      createdArtifact.ok &&
      createdArtifact.artifact?.sizeBytes === 700_000 &&
      createdArtifact.artifact.internalPath === undefined &&
      createdArtifact.metadata?.value?.internalPath === undefined &&
      !crossOriginArtifact.ok &&
      crossOriginArtifact.error?.code === 'ARTIFACT_ACCESS_DENIED' &&
      releasedArtifact.release.ok &&
      !releasedArtifact.after.ok &&
      releasedArtifact.after.error?.code === 'ARTIFACT_ACCESS_DENIED';
    const websiteJobBoundary = (await controller.activeWebsiteWebContents.executeJavaScript(`
      (async () => {
        const events = [];
        const unsubscribe = globalThis.localAI.onJobEvent((event) => events.push(event));
        const invalid = await globalThis.localAI.invokeJob({
          sessionId: '', input: { shellCommand: 'whoami' }
        });
        // Input contents are validated only after session ownership is established.
        const unownedInvalidInput = await globalThis.localAI.invokeJob({
          sessionId: 'session_unknown', input: { shellCommand: 'whoami' }
        });
        const missingSession = await globalThis.localAI.invokeJob({
          sessionId: 'session_unknown', input: { prompt: 'hello' }
        });
        const unauthorizedCancel = await globalThis.localAI.cancelJob('job_abcdefghijklmnopqrstuvwxyz');
        unsubscribe();
        const serialized = JSON.stringify({ invalid, unownedInvalidInput, missingSession, unauthorizedCancel });
        return {
          invalidInput: invalid.ok === false && invalid.error.code === 'JOB_INPUT_INVALID',
          sessionOwnership:
            missingSession.ok === false && missingSession.error.code === 'SESSION_ACCESS_DENIED' &&
            unownedInvalidInput.ok === false && unownedInvalidInput.error.code === 'SESSION_ACCESS_DENIED',
          cancelOwnership:
            unauthorizedCancel.ok === false && unauthorizedCancel.error.code === 'JOB_ACCESS_DENIED',
          sanitized: !serialized.includes('/Users/') && !serialized.includes('Traceback'),
          noForeignEvents: events.length === 0
        };
      })()
    `)) as Record<string, boolean>;
    const websiteJobModel = Object.values(websiteJobBoundary).every(Boolean);

    const markersSurvivedSwitch = await verifyMarkers(controller, perTab);
    const secondBeforeNavigation = controller.tabs.getTab(secondTabId);
    const otherUrlsBeforeNavigation = [
      controller.tabs.getTab(initialTabId).url,
      controller.tabs.getTab(thirdTabId).url,
    ];
    controller.activateTab(secondTabId);
    const crossOriginNavigationURL = new URL(
      '?oryn=cross-origin',
      process.env.ORYN_START_URL ?? 'https://example.com',
    ).href;
    await navigateAndWait(controller.activeWebsiteWebContents, () =>
      controller.navigate(crossOriginNavigationURL),
    );
    const secondAfterNavigation = controller.tabs.getTab(secondTabId);
    const crossOriginRotatedDocument =
      secondAfterNavigation.activeDocumentId !== secondBeforeNavigation.activeDocumentId &&
      !callerContexts.hasDocumentAuthority(
        secondTabId,
        secondAfterNavigation.webContentsId,
        secondBeforeNavigation.activeDocumentId,
      );
    const crossTabReuseRejected = !callerContexts.hasDocumentAuthority(
      secondTabId,
      secondAfterNavigation.webContentsId,
      perTab[0]?.documentId ?? '',
    );
    const independentNavigation =
      controller.tabs.getTab(initialTabId).url === otherUrlsBeforeNavigation[0] &&
      controller.tabs.getTab(thirdTabId).url === otherUrlsBeforeNavigation[1];

    const backWait = waitForNavigation(controller.activeWebsiteWebContents);
    controller.back();
    await backWait;
    const wentBack =
      new URL(controller.tabs.getTab(secondTabId).url).origin === new URL(secondInitialUrl).origin;
    const forwardWait = waitForNavigation(controller.activeWebsiteWebContents);
    controller.forward();
    await forwardWait;
    const wentForward = controller.tabs.getTab(secondTabId).url === crossOriginNavigationURL;

    await controller.closeTab(thirdTabId);
    const closePreservedOthers =
      controller.getState().tabs.length === 2 &&
      controller.getState().tabs.some((tab) => tab.tabId === initialTabId) &&
      controller.getState().tabs.some((tab) => tab.tabId === secondTabId);
    const reopenedTabId = controller.reopenRecentlyClosed();
    if (!reopenedTabId) {
      throw new Error('Recently closed tab was not restored.');
    }
    await waitUntilLoaded(
      controller.activeWebsiteWebContents,
      thirdInitialUrl,
      'reopened smoke tab',
    );
    const reopened = controller.tabs.getTab(reopenedTabId);
    const reopenedRecentlyClosed =
      controller.getState().tabs.length === 3 &&
      new URL(reopened.url).origin === new URL(thirdInitialUrl).origin;

    const expectedHistoryOrigins = [
      new URL(crossOriginNavigationURL).origin,
      new URL(secondInitialUrl).origin,
      new URL(thirdInitialUrl).origin,
    ];
    // Navigation commits before did-stop-loading persists the history entry.
    await waitForCondition(() => {
      const origins = new Set(
        browserData.history.search().map((entry) => new URL(entry.url).origin),
      );
      return expectedHistoryOrigins.every((origin) => origins.has(origin));
    }, 'history entries after navigation');
    const historyOrigins = new Set(
      browserData.history.search().map((entry) => new URL(entry.url).origin),
    );
    const historyRecordedMultiTabNavigation =
      historyOrigins.has(new URL(crossOriginNavigationURL).origin) &&
      historyOrigins.has(new URL(secondInitialUrl).origin) &&
      historyOrigins.has(new URL(thirdInitialUrl).origin);

    const persistedBeforeRestart = browserData.sessions.getRestoreState();
    const configuredSmokeStartURL = process.env.ORYN_START_URL;
    delete process.env.ORYN_START_URL;
    const restoredController = new BrowserWindowController(engine, callerContexts, browserData, {
      loadTrustedUI: false,
    });
    if (configuredSmokeStartURL) process.env.ORYN_START_URL = configuredSmokeStartURL;
    const restoredState = restoredController.getState();
    const expectedActiveURL =
      persistedBeforeRestart.openTabs[persistedBeforeRestart.activeIndex]?.url;
    const restoredActiveURL = restoredState.tabs.find(
      (tab) => tab.tabId === restoredState.activeTabId,
    )?.url;
    const sessionRestoreReconstructed =
      restoredState.tabs.length === persistedBeforeRestart.openTabs.length &&
      restoredState.tabs.map((tab) => tab.url).join('\n') ===
        persistedBeforeRestart.openTabs.map((tab) => tab.url).join('\n') &&
      restoredActiveURL === expectedActiveURL;
    restoredController.window.destroy();

    await waitUntilLoaded(controller.trustedWebContents);
    const onboardingWasPreviouslyComplete = browserData.appearance.read().onboardingComplete;
    const browserCoreUI = (await controller.trustedWebContents.executeJavaScript(`
      (async () => {
        const onboarding = document.querySelector('[aria-labelledby="oryn-onboarding-title"]');
        const onboardingShown =
          (onboarding?.textContent ?? '').includes('Run AI models on your computer') &&
          (onboarding?.textContent ?? '').includes('You choose what runs');
        Array.from(onboarding?.querySelectorAll('button') ?? [])
          .find((button) => button.textContent?.includes('Start browsing'))?.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const requiredControls = [
          'Address and search',
          'Back',
          'Forward',
          'Downloads',
          'Site controls',
          'Browser menu',
          'New tab'
        ];
        const visibleControls = requiredControls.every((label) =>
          document.querySelector('[aria-label="' + label + '"]')
        );
        document.querySelector('[aria-label="Browser menu"]')?.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const menuItems = Array.from(document.querySelectorAll('[role="menuitem"]'))
          .map((element) => element.textContent ?? '');
        const menuCoverage = ['History', 'Bookmarks', 'Downloads', 'Find in page', 'Print', 'Browser settings', 'Developer tools']
          .every((label) => menuItems.some((item) => item.includes(label)));
        await globalThis.browser.addBookmark('Oryn smoke bookmark', ${JSON.stringify(
          'https://example.net',
        )});
        await globalThis.browser.setSitePermission('notifications', 'allow', (await globalThis.browser.getBrowserData()).siteContext);
        await globalThis.browser.setZoom('in');
        await globalThis.browser.updateAppearance({ theme: 'dark', density: 'compact' });
        const data = await globalThis.browser.getBrowserData();
        const windowState = await globalThis.browser.getState();
        return {
          visibleControls,
          onboardingShown,
          onboardingHandled: onboardingShown || ${String(onboardingWasPreviouslyComplete)},
          appearanceManaged:
            data.appearance.onboardingComplete &&
            data.appearance.theme === 'dark' &&
            data.appearance.density === 'compact',
          menuCoverage,
          bookmarkManaged: data.bookmarks.some((item) => item.kind === 'bookmark' && item.title === 'Oryn smoke bookmark'),
          sitePermissionManaged: data.sitePermissions.some((item) => item.permission === 'notifications' && item.decision === 'allow'),
          zoomManaged: windowState.zoomFactor > 1
        };
      })()
    `)) as {
      visibleControls: boolean;
      onboardingShown: boolean;
      onboardingHandled: boolean;
      appearanceManaged: boolean;
      menuCoverage: boolean;
      bookmarkManaged: boolean;
      sitePermissionManaged: boolean;
      zoomManaged: boolean;
    };
    const ordinaryBrowserControls =
      browserCoreUI.visibleControls &&
      browserCoreUI.onboardingHandled &&
      browserCoreUI.appearanceManaged &&
      browserCoreUI.menuCoverage &&
      browserCoreUI.bookmarkManaged &&
      browserCoreUI.sitePermissionManaged &&
      browserCoreUI.zoomManaged;
    await controller.trustedWebContents.executeJavaScript(
      `document.querySelector('.popover-dismiss-layer')?.dispatchEvent(
        new PointerEvent('pointerdown', { bubbles: true })
      )`,
    );
    await waitForCondition(
      () =>
        controller.trustedWebContents
          .executeJavaScript("!document.querySelector('.browser-menu')")
          .then(Boolean),
      'closed browser menu before compositor check',
    );
    const websiteBoundsBeforeMenu = controller.tabs.getTab(controller.activeTabId).view.getBounds();
    await controller.trustedWebContents.executeJavaScript(
      `document.querySelector('[aria-label="Browser menu"]')?.click()`,
    );
    await waitForCondition(async () => {
      const menuVisible = await controller.trustedWebContents
        .executeJavaScript("Boolean(document.querySelector('.browser-menu'))")
        .then(Boolean);
      if (!menuVisible) {
        await controller.trustedWebContents.executeJavaScript(
          `document.querySelector('[aria-label="Browser menu"]')?.click()`,
        );
        return false;
      }
      return menuVisible && controller.isTrustedChromeCoveringWindow();
    }, 'trusted browser menu compositor');
    const websiteBoundsDuringMenu = controller.tabs.getTab(controller.activeTabId).view.getBounds();
    const trustedBoundsDuringMenu = controller.trustedView.getBounds();
    await controller.trustedWebContents.executeJavaScript(
      `document.querySelector('.popover-dismiss-layer')?.dispatchEvent(
        new PointerEvent('pointerdown', { bubbles: true })
      )`,
    );
    await waitForCondition(
      () =>
        controller.trustedWebContents
          .executeJavaScript("!document.querySelector('.browser-menu')")
          .then(Boolean),
      'trusted browser menu dismissal',
    );
    const websiteBoundsAfterMenu = controller.tabs.getTab(controller.activeTabId).view.getBounds();
    const browserMenuOverlayStable =
      JSON.stringify(websiteBoundsBeforeMenu) === JSON.stringify(websiteBoundsDuringMenu) &&
      JSON.stringify(websiteBoundsBeforeMenu) === JSON.stringify(websiteBoundsAfterMenu) &&
      trustedBoundsDuringMenu.height > websiteBoundsBeforeMenu.y;
    const primaryClickOriginTabId = controller.activeTabId;
    const primaryClickTargetTabId = controller
      .getState()
      .tabs.find((tab) => tab.tabId !== primaryClickOriginTabId)?.tabId;
    let primaryTabClickWorks = false;
    if (primaryClickTargetTabId) {
      await controller.trustedWebContents.executeJavaScript(
        `document.querySelector('.workspace-chip')?.click()`,
      );
      await waitForCondition(
        () =>
          controller.trustedWebContents
            .executeJavaScript("Boolean(document.querySelector('.workspace-menu'))")
            .then(Boolean),
        'workspace menu before primary tab click',
      );
      const point = (await controller.trustedWebContents.executeJavaScript(`(() => {
        const tab = document.querySelector(
          '[data-tab-id="${primaryClickTargetTabId}"] .tab-activate'
        );
        const bounds = tab?.getBoundingClientRect();
        return bounds
          ? { x: Math.round(bounds.left + bounds.width / 2), y: Math.round(bounds.top + bounds.height / 2) }
          : undefined;
      })()`)) as { x: number; y: number } | undefined;
      if (point) {
        controller.trustedWebContents.sendInputEvent({
          type: 'mouseDown',
          x: point.x,
          y: point.y,
          button: 'left',
          clickCount: 1,
        });
        controller.trustedWebContents.sendInputEvent({
          type: 'mouseUp',
          x: point.x,
          y: point.y,
          button: 'left',
          clickCount: 1,
        });
        await waitForCondition(
          async () =>
            controller.activeTabId === primaryClickTargetTabId &&
            (await controller.trustedWebContents
              .executeJavaScript("!document.querySelector('.workspace-menu')")
              .then(Boolean)),
          'primary tab click through open workspace menu',
        );
        primaryTabClickWorks = true;
        controller.activateTab(primaryClickOriginTabId);
      }
    }
    const expectedSongRepository =
      process.env.ORYN_EXPECT_SONG_MODEL ??
      'hf-tiny-v2/tiny-random-MusicgenForConditionalGeneration';
    const permissionRecoveryKey = {
      origin: 'https://permission-recovery.test',
      role: 'permission-recovery',
      model: {
        provider: 'huggingface' as const,
        repository: 'oryn/smoke',
        revision: 'a'.repeat(40),
      },
    };
    localAIPermissions.recordTrustedDecision(permissionRecoveryKey, 'deny');
    await controller.trustedWebContents.executeJavaScript(
      `document.querySelector('[aria-label="Local AI activity"]')?.click()`,
    );
    await waitForCondition(async () => {
      const panelVisible = await controller.trustedWebContents
        .executeJavaScript('Boolean(document.querySelector(\'[aria-label="Oryn Local AI"]\'))')
        .then(Boolean);
      if (!panelVisible) {
        await controller.trustedWebContents.executeJavaScript(
          `document.querySelector('[aria-label="Local AI activity"]')?.click()`,
        );
        return false;
      }
      if (!controller.isTrustedChromeCoveringWindow()) return false;
      // Native bounds update before the renderer receives its new viewport.
      // Wait for the actual sheet geometry before taking the assertions below.
      return controller.trustedWebContents
        .executeJavaScript(
          `(() => {
          const bounds = document.querySelector('[aria-label="Oryn Local AI"]')?.getBoundingClientRect();
          return Boolean(bounds && bounds.width <= 560
            && (innerWidth < 760 ? bounds.left >= 0 && bounds.right <= innerWidth : bounds.left > innerWidth / 4)
            && bounds.height > innerHeight / 2);
        })()`,
        )
        .then(Boolean);
    }, 'trusted Local AI sheet compositor');
    const trustedAIActivity = (await controller.trustedWebContents.executeJavaScript(`
      (async () => {
        const panel = document.querySelector('[aria-label="Oryn Local AI"]');
        const panelBounds = panel?.getBoundingClientRect();
        const sideSheetLayout = Boolean(
          panelBounds &&
          panelBounds.width <= 560 &&
          (window.innerWidth < 760
            ? panelBounds.left >= 0 && panelBounds.right <= window.innerWidth
            : panelBounds.left > window.innerWidth / 4) &&
          panelBounds.height > window.innerHeight / 2
        );
        const tabs = ['Running', 'Activity', 'Models', 'Engines', 'Settings'].every((label) =>
          Array.from(panel?.querySelectorAll('button') ?? []).some((button) => button.textContent?.includes(label))
        );
        Array.from(panel?.querySelectorAll('button') ?? []).find((button) => button.textContent?.trim() === 'Activity')?.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const activityText = panel?.textContent ?? '';
        Array.from(panel?.querySelectorAll('button') ?? []).find((button) => button.textContent?.includes('Models'))?.click();
        await new Promise((resolve) => setTimeout(resolve, 2000));
        const modelsText = panel?.textContent ?? '';
        Array.from(panel?.querySelectorAll('button') ?? []).find((button) => button.textContent?.includes('Settings'))?.click();
        await new Promise((resolve) => setTimeout(resolve, 100));
        const settingsNavigationText = panel?.textContent ?? '';
        Array.from(panel?.querySelectorAll('[aria-label="Local AI Settings sections"] button') ?? [])
          .find((button) => button.textContent?.trim() === 'AI Activity')?.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const settingsText = panel?.textContent ?? '';
        const settingsSnapshot = await globalThis.browser.getOrynSettings();
        Array.from(panel?.querySelectorAll('[aria-label="Local AI Settings sections"] button') ?? [])
          .find((button) => button.textContent?.trim() === 'Sites')?.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const recoverySite = Array.from(panel?.querySelectorAll('article') ?? [])
          .find((article) => article.querySelector('strong')?.textContent === 'https://permission-recovery.test');
        const askAgainButton = Array.from(recoverySite?.querySelectorAll('button') ?? [])
          .find((button) => button.textContent?.trim() === 'Ask again');
        askAgainButton?.click();
        let permissionRecovery = false;
        for (let attempt = 0; askAgainButton && attempt < 50; attempt += 1) {
          const snapshot = await globalThis.browser.getOrynSettings();
          const permissions = snapshot.sites.find((site) => site.origin === 'https://permission-recovery.test')?.permissions ?? [];
          if (permissions.length === 0) {
            permissionRecovery = true;
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 20));
        }
        const trustedMethods =
          typeof globalThis.browser.cancelLocalAIJob === 'function' &&
          typeof globalThis.browser.cancelLocalAIJobsForSite === 'function' &&
          typeof globalThis.browser.cancelAllLocalAIJobs === 'function' &&
          typeof globalThis.browser.clearAIActivity === 'function' &&
          typeof globalThis.browser.getModelManager === 'function' &&
          typeof globalThis.browser.unloadManagedModel === 'function' &&
          typeof globalThis.browser.removeManagedModel === 'function' &&
          typeof globalThis.browser.updateOrynRanking === 'function' &&
          typeof globalThis.browser.updateOrynCompute === 'function' &&
          typeof globalThis.browser.askOrynSiteAgain === 'function' &&
          typeof globalThis.browser.revokeOrynSite === 'function' &&
          typeof globalThis.browser.cleanupOrynStorage === 'function';
        document.querySelector('[aria-label="Close Local AI"]')?.click();
        return {
          tabs,
          sideSheetLayout,
          permissionRecovery,
          recordVisible:
            ${process.env.ORYN_TRANSCRIPTION_SMOKE === '1'}
              ? activityText.includes('Transcription') &&
                activityText.includes('Xenova/tiny-random-WhisperForConditionalGeneration') &&
                activityText.includes('Completed')
              : ${process.env.ORYN_CHAT_SMOKE === '1'}
                ? activityText.includes('Chat') &&
                  activityText.includes('trl-internal-testing/tiny-Qwen2ForCausalLM-2.5') &&
                  activityText.includes('Completed')
                : ${process.env.ORYN_SONG_SMOKE === '1'}
                  ? activityText.includes('Song Generation') &&
                    activityText.includes(${JSON.stringify(expectedSongRepository)}) &&
                    activityText.includes('Completed')
                  : ${process.env.ORYN_ALPHA_INTEGRATION_SMOKE === '1'}
                    ? activityText.includes('Chat') &&
                      activityText.includes('trl-internal-testing/tiny-Qwen2ForCausalLM-2.5') &&
                      activityText.includes('Transcription') &&
                      activityText.includes('Xenova/tiny-random-WhisperForConditionalGeneration') &&
                      activityText.includes('Completed')
                  : activityText.includes('No Oryn activity yet'),
          settingsVisible:
            ['Models', 'Preferences', 'Compute', 'Sites', 'AI Activity', 'Storage', 'Developer']
              .every((label) => settingsNavigationText.includes(label)) &&
            settingsText.includes('Keep AI Activity') &&
            settingsText.includes('Metadata only') &&
            settingsText.includes('30 days') &&
            settingsText.includes('Clear AI Activity') &&
            settingsSnapshot.compute.generationConcurrency === 1 &&
            settingsSnapshot.developer.experimentalRunnersAllowed === false,
          modelManagerVisible:
            ${process.env.ORYN_TRANSCRIPTION_SMOKE === '1'}
              ? modelsText.includes('Xenova/tiny-random-WhisperForConditionalGeneration') &&
                modelsText.includes('Installation: Installed') &&
                modelsText.includes('Residency: Loaded') &&
                modelsText.includes('View source') &&
                modelsText.includes('Unload') &&
                modelsText.includes('Delete') &&
                modelsText.includes('Possible installed substitutes')
              : ${process.env.ORYN_CHAT_SMOKE === '1'}
                ? modelsText.includes('trl-internal-testing/tiny-Qwen2ForCausalLM-2.5') &&
                  modelsText.includes('Installation: Installed') &&
                  modelsText.includes('View source') &&
                  modelsText.includes('Unload') &&
                  modelsText.includes('Delete') &&
                  modelsText.includes('Possible installed substitutes')
                : ${process.env.ORYN_SONG_SMOKE === '1'}
                  ? modelsText.includes(${JSON.stringify(expectedSongRepository)}) &&
                    modelsText.includes('Installation: Installed') &&
                    modelsText.includes('View source') &&
                    modelsText.includes('Unload') &&
                    modelsText.includes('Delete') &&
                    modelsText.includes('Possible installed substitutes')
                  : ${process.env.ORYN_ALPHA_INTEGRATION_SMOKE === '1'}
                    ? modelsText.includes('trl-internal-testing/tiny-Qwen2ForCausalLM-2.5') &&
                      modelsText.includes('Xenova/tiny-random-WhisperForConditionalGeneration') &&
                      modelsText.includes('Installation: Installed') &&
                      modelsText.includes('View source') &&
                      modelsText.includes('Unload') &&
                      modelsText.includes('Delete') &&
                      modelsText.includes('Possible installed substitutes')
                  : modelsText.includes('No local AI models installed'),
          trustedMethods
        };
      })()
    `)) as Record<string, boolean>;
    trustedAIActivity.permissionRecovery &&=
      localAIPermissions.decision(permissionRecoveryKey) === undefined;
    const trustedAIActivityUX = Object.values(trustedAIActivity).every(Boolean);

    const smokeModel = {
      provider: 'huggingface' as const,
      repository: 'Qwen/Qwen3.5-0.8B',
      revision: 'a'.repeat(40),
    };
    const smokeProfile: CandidateExecutionProfile = {
      model: smokeModel,
      exactRevision: smokeModel.revision,
      executable: true,
      runner: 'transformers',
      executableBackends: ['cpu'],
      requiresCustomCode: false,
      estimatedDiskBytes: 820_000_000,
      estimatedMemoryBytes: 2_000_000_000,
      descriptorSchemaVersion: 'oryn-model-descriptor/1',
      capabilityRegistryDigest: 'b'.repeat(64),
      loadDisposition: 'trust-required' as const,
      blockingReasons: ['CONFORMANCE_REQUIRED'],
      eligibility: {
        access: { state: 'available', reasons: [] },
        artifacts: { state: 'safe', reasons: [] },
        runtime: { state: 'supported', reasons: [] },
        task: { state: 'supported', reasons: [] },
        resources: { state: 'admissible', reasons: [] },
        conformance: { state: 'untested', reasons: ['CONFORMANCE_REQUIRED'] },
      },
      technicalCapabilities: {
        task: 'text-generation',
        inputModalities: ['text' as const],
        outputModalities: ['text' as const],
        capabilities: { chat: true },
      },
    };
    const permissionResponsePromise = permissionPrompts.request('https://example.net', 'chat', {
      resolution: {
        preferredModel: smokeModel,
        resolvedModel: smokeModel,
        compatibility: 'compatible',
        executability: 'executable',
        security: 'allowed',
        trust: 'user-approved-unverified',
        developerCertified: false,
        substituted: false,
        overriddenByUser: false,
        resolutionReason: 'preferred-model',
        compatibilityReasons: [],
        downloadRequired: true,
      },
      selected: {
        model: smokeModel,
        profile: smokeProfile,
        compatibility: 'compatible',
        executability: 'executable',
        security: 'allowed',
        developerCertified: false,
        installed: false,
        loaded: false,
        sources: ['preferred', 'certified'],
        compatibilityReasons: [],
      },
      candidates: [
        {
          model: smokeModel,
          profile: smokeProfile,
          compatibility: 'compatible',
          executability: 'executable',
          security: 'allowed',
          developerCertified: false,
          installed: false,
          loaded: false,
          sources: ['preferred', 'certified'],
          compatibilityReasons: [],
        },
      ],
    });
    await waitForCondition(
      () =>
        controller.trustedWebContents
          .executeJavaScript(
            'Boolean(document.querySelector(\'[aria-label="Local AI permission"]\'))',
          )
          .then(Boolean),
      'trusted Local AI permission dialog',
    );
    const trustedLocalAIPermissionUI = (await controller.trustedWebContents.executeJavaScript(`
      (() => {
        const dialog = document.querySelector('[aria-label="Local AI permission"]');
        const text = dialog?.textContent ?? '';
        const complete = ['https://example.net', 'Chat', 'Qwen/Qwen3.5-0.8B', 'Not Oryn verified', 'approval required', 'Hugging Face', 'Exact revision', 'Download', 'Expected local use', 'runs on your computer', 'has not passed Oryn conformance', 'Only you can grant this trust', 'Choose model', 'Deny once', 'Trust and allow once', 'Trust for this site']
          .every((value) => text.toLowerCase().includes(value.toLowerCase()));
        Array.from(dialog?.querySelectorAll('button') ?? []).find((button) => button.textContent === 'Deny once')?.click();
        return complete;
      })()
    `)) as boolean;
    const permissionResponse = await permissionResponsePromise;
    const trustedPermissionResponse = permissionResponse.action === 'deny';

    const engineBeforeForcedExit = engine.engineInstanceId;
    engine.forceExitForSmokeTest();
    await waitForCondition(
      () =>
        engine.state === 'ready' &&
        Boolean(engine.engineInstanceId) &&
        engine.engineInstanceId !== engineBeforeForcedExit,
      'engine restart handshake after forced exit',
    );
    const forcedExitRestarted = await engine
      .getRuntimeCapabilities()
      .then((capabilities) => capabilities.engineInstanceId === engine.engineInstanceId);

    const engineBeforeGracefulRestart = engine.engineInstanceId;
    await engine.restart();
    const gracefulShutdownRestarted =
      engine.state === 'ready' && engine.engineInstanceId !== engineBeforeGracefulRestart;
    const [runtimeResources, resourceCapabilities] = await Promise.all([
      engine.getRuntimeResources(),
      engine.getRuntimeCapabilities(),
      engine.ping(),
    ]);
    const policyResult = await engine.updateRuntimePolicy({
      memoryBudgetBytes: Math.max(
        512 * 1024 * 1024,
        Math.floor((runtimeResources.systemMemory?.totalBytes ?? 1024 * 1024 * 1024) / 2),
      ),
      batteryMode: 'normal',
      allowSpeculativePreload: false,
    });
    const runtimeResourceAuthority =
      resourceCapabilities.features.memoryPressureMonitoring &&
      resourceCapabilities.backends.length > 0 &&
      ['normal', 'high', 'critical'].includes(runtimeResources.memoryPressure) &&
      runtimeResources.generationLane.capacity === 1 &&
      policyResult.appliedRevision === 2 &&
      policyResult.resources.localAI.memoryBudgetBytes !== undefined;
    const reconciledModels = (await engine.getModelInventory()).models;
    const requiredRepositories = [
      ...(process.env.ORYN_TRANSCRIPTION_SMOKE === '1'
        ? ['Xenova/tiny-random-WhisperForConditionalGeneration']
        : []),
      ...(process.env.ORYN_CHAT_SMOKE === '1'
        ? ['trl-internal-testing/tiny-Qwen2ForCausalLM-2.5']
        : []),
      ...(process.env.ORYN_SONG_SMOKE === '1' ? [expectedSongRepository] : []),
      ...(process.env.ORYN_ALPHA_INTEGRATION_SMOKE === '1'
        ? [
            'trl-internal-testing/tiny-Qwen2ForCausalLM-2.5',
            'Xenova/tiny-random-WhisperForConditionalGeneration',
          ]
        : []),
      ...(process.env.ORYN_EXPECT_PERSISTED_MODEL ? [process.env.ORYN_EXPECT_PERSISTED_MODEL] : []),
    ];
    const modelInventoryReconciled =
      requiredRepositories.every((repository) =>
        reconciledModels.some(
          (item) => item.model.repository === repository && item.installationState === 'installed',
        ),
      ) &&
      reconciledModels.every(
        (item) => item.installationState === 'installed' && item.residencyState === 'unloaded',
      );

    const passed =
      initialDocumentRendered &&
      distinctOrigins &&
      isolatedEverywhere &&
      capabilitiesEverywhere &&
      modelInspectionAdvertised &&
      publicModelAPIEverywhere &&
      transcriptionFoundation.passed &&
      chatFoundation.passed &&
      songFoundation.passed &&
      alphaFoundation.passed &&
      artifactOutputBoundary.passed &&
      websiteModelBoundary &&
      artifactInputBoundary &&
      websiteJobModel &&
      markersSurvivedSwitch &&
      crossOriginRotatedDocument &&
      crossTabReuseRejected &&
      independentNavigation &&
      wentBack &&
      wentForward &&
      closePreservedOthers &&
      reopenedRecentlyClosed &&
      historyRecordedMultiTabNavigation &&
      sessionRestoreReconstructed &&
      ordinaryBrowserControls &&
      browserMenuOverlayStable &&
      primaryTabClickWorks &&
      trustedAIActivityUX &&
      trustedLocalAIPermissionUI &&
      trustedPermissionResponse &&
      forcedExitRestarted &&
      gracefulShutdownRestarted &&
      runtimeResourceAuthority &&
      modelInventoryReconciled;

    console.log(
      `LOCAL_AI_SMOKE_RESULT ${JSON.stringify({
        passed,
        initialDocument,
        initialDocumentRendered,
        keychainPersistence,
        tabCount: controller.getState().tabs.length,
        distinctOrigins,
        isolatedEverywhere,
        capabilitiesEverywhere,
        modelInspectionAdvertised,
        publicModelAPIEverywhere,
        transcriptionFoundation,
        chatFoundation,
        songFoundation,
        alphaFoundation,
        artifactOutputBoundary,
        websiteModelBoundary,
        websiteBridgeBoundary,
        artifactInputBoundary,
        websiteJobModel,
        websiteJobBoundary,
        markersSurvivedSwitch,
        crossOriginRotatedDocument,
        crossTabReuseRejected,
        independentNavigation,
        wentBack,
        wentForward,
        closePreservedOthers,
        reopenedRecentlyClosed,
        historyRecordedMultiTabNavigation,
        sessionRestoreReconstructed,
        ordinaryBrowserControls,
        browserMenuOverlayStable,
        primaryTabClickWorks,
        trustedAIActivityUX,
        trustedAIActivity,
        trustedLocalAIPermissionUI,
        trustedPermissionResponse,
        browserCoreUI,
        forcedExitRestarted,
        gracefulShutdownRestarted,
        runtimeResourceAuthority,
        modelInventoryReconciled,
      })}`,
    );
    finishSmokeTest(passed ? 0 : 1);
  } catch (error) {
    console.error('LOCAL_AI_SMOKE_ERROR', error);
    console.error('LOCAL_AI_SMOKE_CHROME', {
      nativeBounds: controller.trustedView.getBounds(),
      coversWindow: controller.isTrustedChromeCoveringWindow(),
      renderer: await controller.trustedWebContents
        .executeJavaScript(
          `(() => ({
          width: innerWidth, height: innerHeight,
          panel: document.querySelector('[aria-label="Oryn Local AI"]')?.getBoundingClientRect().toJSON(),
          menu: document.querySelector('.browser-menu')?.getBoundingClientRect().toJSON(),
        }))()`,
        )
        .catch(() => null),
    });
    finishSmokeTest(1);
  }
}

async function runModelWorkspaceSmoke(controller: BrowserWindowController): Promise<{
  passed: boolean;
  opaqueURL: boolean;
  isolatedBridge: boolean;
  ready: boolean;
  invoked: boolean;
  reopenedOffline: boolean;
  userTrusted: boolean;
  unverifiedNotPromoted: boolean;
}> {
  await engine.start();
  const expected = process.env.ORYN_EXPECT_PERSISTED_MODEL;
  const inventory = await engine.getModelInventory();
  const record = inventory.models.find(
    (item) =>
      item.installationState === 'installed' && (!expected || item.model.repository === expected),
  );
  if (!record) {
    return {
      passed: false,
      opaqueURL: false,
      isolatedBridge: false,
      ready: false,
      invoked: false,
      reopenedOffline: false,
      userTrusted: false,
      unverifiedNotPromoted: false,
    };
  }
  const profile = await engine.inspectModel(record.model);
  if (!profile.runtimePack || profile.runtimePack.taskContract !== 'chat-v1') {
    return {
      passed: false,
      opaqueURL: false,
      isolatedBridge: false,
      ready: false,
      invoked: false,
      reopenedOffline: false,
      userTrusted: false,
      unverifiedNotPromoted: false,
    };
  }
  const expectedTrustMode =
    profile.loadDisposition === 'trust-required' ? 'user-trusted' : 'oryn-verified';
  const binding = browserData.modelWorkspaces.create({
    model: profile.model,
    taskContract: profile.runtimePack.taskContract,
    trustMode: expectedTrustMode,
    runtimePack: {
      packId: profile.runtimePack.packId,
      packVersion: profile.runtimePack.packVersion,
    },
  });
  const workspaceURL = createOrynModelWorkspaceURL(binding.reference);
  controller.createTab(workspaceURL, true);
  await waitUntilLoaded(controller.activeWebsiteWebContents);
  const result = await invokeActiveModelWorkspace(controller, expectedTrustMode);
  controller.reload();
  await waitUntilLoaded(controller.activeWebsiteWebContents);
  const reopened = (await controller.activeWebsiteWebContents.executeJavaScript(
    `globalThis.orynWorkspace.getState()`,
  )) as { status: string; taskContract?: string; trustMode?: string };
  const opaqueURL =
    workspaceURL === controller.activeWebsiteWebContents.getURL() &&
    !workspaceURL.includes(record.model.repository) &&
    !workspaceURL.includes(record.model.revision);
  const reopenedOffline =
    reopened.status === 'ready' &&
    reopened.taskContract === 'chat-v1' &&
    reopened.trustMode === expectedTrustMode;

  let unverifiedNotPromoted = true;
  if (expectedTrustMode === 'user-trusted') {
    const forged = browserData.modelWorkspaces.create({
      model: profile.model,
      taskContract: 'chat-v1',
      trustMode: 'oryn-verified',
      runtimePack: {
        packId: profile.runtimePack.packId,
        packVersion: profile.runtimePack.packVersion,
      },
    });
    controller.createTab(createOrynModelWorkspaceURL(forged.reference), true);
    await waitUntilLoaded(controller.activeWebsiteWebContents);
    const rejected = (await controller.activeWebsiteWebContents.executeJavaScript(
      'globalThis.orynWorkspace.getState()',
    )) as { status: string };
    unverifiedNotPromoted = rejected.status !== 'ready';
  }

  const userBinding = browserData.modelWorkspaces.create({
    model: profile.model,
    taskContract: 'chat-v1',
    trustMode: 'user-trusted',
  });
  controller.createTab(createOrynModelWorkspaceURL(userBinding.reference), true);
  await waitUntilLoaded(controller.activeWebsiteWebContents);
  const userResult = await invokeActiveModelWorkspace(controller, 'user-trusted');
  const userTrusted = userResult.ready && userResult.invoked;
  return {
    passed:
      opaqueURL &&
      result.isolatedBridge &&
      result.ready &&
      result.invoked &&
      reopenedOffline &&
      userTrusted &&
      unverifiedNotPromoted,
    opaqueURL,
    isolatedBridge: result.isolatedBridge,
    ready: result.ready,
    invoked: result.invoked,
    reopenedOffline,
    userTrusted,
    unverifiedNotPromoted,
  };
}

async function runHuggingFaceUserTrustSmoke(controller: BrowserWindowController): Promise<{
  passed: boolean;
  exactRevision: boolean;
  userTrustOnly: boolean;
  warningVisible: boolean;
  viewportStable: boolean;
  layoutBalanced: boolean;
}> {
  await waitForCondition(
    async () => controller.getState().huggingFacePageAction?.status === 'unverified',
    'Hugging Face user-trust page action',
  );
  const action = controller.getState().huggingFacePageAction;
  const widthBefore = await controller.activeWebsiteWebContents.executeJavaScript(
    `document.documentElement.clientWidth`,
  );
  await waitUntilLoaded(controller.trustedWebContents);
  const dialog = (await controller.trustedWebContents.executeJavaScript(`
    (async () => {
      document.querySelector('.hf-page-action')?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
      const panel = document.querySelector('.hf-install-dialog');
      const panelRect = panel?.getBoundingClientRect();
      const selectedFiles = panel?.querySelector('.hf-selected-files ul');
      const leftInset = (selector) => {
        const rect = panel?.querySelector(selector)?.getBoundingClientRect();
        return rect && panelRect ? rect.left - panelRect.left : 0;
      };
      return {
        text: panel?.textContent ?? '',
        paddingLeft: panel ? Number.parseFloat(getComputedStyle(panel).paddingLeft) : 0,
        headerInset: leftInset(':scope > header'),
        factsInset: leftInset(':scope > .hf-install-facts'),
        footerInset: leftInset(':scope > footer'),
        selectedFilesMaxHeight: selectedFiles
          ? Number.parseFloat(getComputedStyle(selectedFiles).maxHeight)
          : 0
      };
    })()
  `)) as {
    text: string;
    paddingLeft: number;
    headerInset: number;
    factsInset: number;
    footerInset: number;
    selectedFilesMaxHeight: number;
  };
  const widthAfter = await controller.activeWebsiteWebContents.executeJavaScript(
    `document.documentElement.clientWidth`,
  );
  const exactRevision =
    action?.repository === 'Qwen/Qwen3.5-4B' &&
    action.exactRevision === '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a';
  const userTrustOnly =
    action?.status === 'unverified' &&
    action.trustMode === 'user-trusted' &&
    action.taskContract === 'chat-v1';
  const warningVisible =
    dialog.text.includes('Trust and load this model?') &&
    dialog.text.includes('This exact model has not been verified by Oryn.') &&
    dialog.text.includes('Trust and download exact model') &&
    dialog.text.includes('will not execute repository code');
  const viewportStable = widthBefore === widthAfter;
  const layoutBalanced =
    dialog.paddingLeft >= 20 &&
    dialog.headerInset >= 20 &&
    dialog.factsInset >= 20 &&
    dialog.footerInset >= 20 &&
    dialog.selectedFilesMaxHeight > 0 &&
    dialog.selectedFilesMaxHeight <= 224;
  return {
    passed: exactRevision && userTrustOnly && warningVisible && viewportStable && layoutBalanced,
    exactRevision,
    userTrustOnly,
    warningVisible,
    viewportStable,
    layoutBalanced,
  };
}

async function runHuggingFaceGatedAccessSmoke(controller: BrowserWindowController): Promise<{
  passed: boolean;
  accessRequired: boolean;
  accessInstructionsVisible: boolean;
  protectedTokenInput: boolean;
  sourceStayedInOryn: boolean;
}> {
  await waitForCondition(
    async () => controller.getState().huggingFacePageAction?.status === 'access-required',
    'Hugging Face gated-access page action',
  );
  const action = controller.getState().huggingFacePageAction;
  const accessRequired =
    action?.repository === 'krea/Krea-2-Turbo' &&
    action.status === 'access-required' &&
    /^[0-9a-f]{40,64}$/u.test(action.exactRevision ?? '');
  await waitUntilLoaded(controller.trustedWebContents);
  const dialog = (await controller.trustedWebContents.executeJavaScript(`
    (async () => {
      document.querySelector('.hf-page-action')?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
      const panel = document.querySelector('.hf-page-action-details');
      return {
        text: panel?.textContent ?? '',
        protectedTokenInput: panel?.querySelector('input[type="password"]') !== null
      };
    })()
  `)) as { text: string; protectedTokenInput: boolean };
  const accessInstructionsVisible =
    dialog.text.includes('Accept the model terms in this Oryn tab') &&
    dialog.text.includes('Connect the Engine with a read token') &&
    dialog.text.includes('Create a read token') &&
    dialog.text.includes('Save token and check') &&
    dialog.text.includes('Open source in Oryn');
  const tabCountBefore = controller.getState().tabs.length;
  await controller.trustedWebContents.executeJavaScript(`
    [...document.querySelectorAll('.hf-page-action-details button')]
      .find((button) => button.textContent?.trim() === 'Open source in Oryn')
      ?.click()
  `);
  await waitForCondition(
    async () => controller.getState().tabs.length === tabCountBefore + 1,
    'gated source tab inside Oryn',
  );
  const sourceTab = controller.getState().tabs.find((tab) => tab.tabId === controller.activeTabId);
  const sourceStayedInOryn =
    sourceTab?.url.startsWith(
      `https://huggingface.co/krea/Krea-2-Turbo/tree/${action?.exactRevision}`,
    ) === true;
  return {
    passed:
      accessRequired &&
      accessInstructionsVisible &&
      dialog.protectedTokenInput &&
      sourceStayedInOryn,
    accessRequired,
    accessInstructionsVisible,
    protectedTokenInput: dialog.protectedTokenInput,
    sourceStayedInOryn,
  };
}

async function invokeActiveModelWorkspace(
  controller: BrowserWindowController,
  expectedTrustMode: 'oryn-verified' | 'user-trusted',
): Promise<{ isolatedBridge: boolean; ready: boolean; invoked: boolean }> {
  return (await controller.activeWebsiteWebContents.executeJavaScript(`
    (async () => {
      const api = globalThis.orynWorkspace;
      const isolatedBridge =
        typeof globalThis.process === 'undefined' &&
        typeof globalThis.require === 'undefined' &&
        typeof globalThis.localAI === 'undefined' &&
        typeof globalThis.browser === 'undefined' &&
        typeof api?.getState === 'function' &&
        typeof api?.invoke === 'function' &&
        typeof api?.cancel === 'function';
      const state = await api.getState();
      const terminal = new Promise((resolve) => {
        const timeout = setTimeout(() => resolve({ event: 'timeout' }), 120000);
        const stop = api.onJobEvent((event) => {
          if (event.event === 'job.completed' || event.event === 'job.failed' || event.event === 'job.cancelled') {
            clearTimeout(timeout);
            stop();
            resolve(event);
          }
        });
      });
      const invocation = await api.invoke({
        messages: [{ role: 'user', content: 'Reply with one short greeting.' }],
        maxTokens: 12,
        temperature: 0,
        topP: 1
      });
      const completed = invocation.ok ? await terminal : { event: 'invoke-failed' };
      return {
        isolatedBridge,
        ready:
          state.status === 'ready' &&
          state.taskContract === 'chat-v1' &&
          state.trustMode === ${JSON.stringify(expectedTrustMode)},
        invoked: invocation.ok === true && completed.event === 'job.completed'
      };
    })()
  `)) as { isolatedBridge: boolean; ready: boolean; invoked: boolean };
}

async function runQwen35ChooserSmoke(controller: BrowserWindowController): Promise<{
  passed: boolean;
  disabled: boolean;
  checked: boolean;
  rowText: string;
  selectedModel: string;
  initialDownload: string;
  selectedDownload: string;
  selectedMemory: string;
}> {
  const website = controller.activeWebsiteWebContents;
  await waitForCondition(
    async () =>
      (
        (await website.executeJavaScript(
          `document.querySelector('#status')?.textContent ?? ''`,
        )) as string
      ).startsWith('Ready.'),
    'chat demo readiness',
  );
  await website.executeJavaScript(`
    document.querySelector('#chat-form')?.dispatchEvent(
      new Event('submit', { bubbles: true, cancelable: true })
    )
  `);
  await waitForCondition(
    async () =>
      (await controller.trustedWebContents.executeJavaScript(
        `document.body.textContent?.includes('Qwen/Qwen3.5-0.8B') ?? false`,
      )) as boolean,
    'Qwen 3.5 permission prompt',
  );
  await controller.trustedWebContents.executeJavaScript(`
    [...document.querySelectorAll('button')]
      .find((button) => button.textContent?.trim() === 'Choose model')
      ?.click()
  `);
  await waitForCondition(
    async () =>
      (await controller.trustedWebContents.executeJavaScript(
        `Boolean(document.querySelector('input[value="Qwen/Qwen3.5-0.8B"]'))`,
      )) as boolean,
    'Qwen 3.5 chooser row',
  );
  const initialDownload = (await controller.trustedWebContents.executeJavaScript(`
    [...document.querySelectorAll('.permission-facts > div')]
      .find((item) => item.querySelector('dt')?.textContent === 'Download')
      ?.querySelector('dd')?.textContent?.trim() ?? ''
  `)) as string;
  const before = (await controller.trustedWebContents.executeJavaScript(`(() => {
    const input = document.querySelector('input[value="Qwen/Qwen3.5-0.8B"]');
    const label = input?.closest('label');
    input?.click();
    return {
      disabled: input?.disabled ?? true,
      rowText: label?.textContent?.replace(/\\s+/g, ' ').trim() ?? ''
    };
  })()`)) as { disabled: boolean; rowText: string };
  await waitForCondition(
    async () =>
      (await controller.trustedWebContents.executeJavaScript(
        `document.querySelector('input[value="Qwen/Qwen3.5-0.8B"]')?.checked ?? false`,
      )) as boolean,
    'Qwen 3.5 chooser selection',
  );
  const after = (await controller.trustedWebContents.executeJavaScript(`(() => {
    const facts = [...document.querySelectorAll('.permission-facts > div')];
    const value = (name) => facts
      .find((item) => item.querySelector('dt')?.textContent === name)
      ?.querySelector('dd')?.textContent?.trim() ?? '';
    return {
      checked: document.querySelector('input[value="Qwen/Qwen3.5-0.8B"]')?.checked ?? false,
      selectedModel: value('Selected model'),
      selectedDownload: value('Download'),
      selectedMemory: value('Expected local use')
    };
  })()`)) as {
    checked: boolean;
    selectedModel: string;
    selectedDownload: string;
    selectedMemory: string;
  };
  await controller.trustedWebContents.executeJavaScript(`
    [...document.querySelectorAll('button')]
      .find((button) => button.textContent?.trim() === 'Deny once')
      ?.click()
  `);
  return {
    passed:
      !before.disabled &&
      after.checked &&
      !before.rowText.includes('Cannot run on this device') &&
      !before.rowText.includes('Does not meet website requirements') &&
      before.rowText.includes('Compatible') &&
      after.selectedModel === 'Qwen/Qwen3.5-0.8B' &&
      after.selectedDownload !== initialDownload &&
      after.selectedDownload.endsWith('MB') &&
      after.selectedMemory.endsWith('MB memory'),
    ...before,
    ...after,
    initialDownload,
  };
}

async function runArtifactOutputSmoke(
  controller: BrowserWindowController,
  ownerTabId: string,
  crossOriginTabId: string,
): Promise<Record<string, boolean> & { passed: boolean }> {
  const engineInstanceId = engine.engineInstanceId;
  if (!engineInstanceId) throw new Error('Engine identity unavailable for artifact output smoke.');
  controller.activateTab(ownerTabId);
  const ownerContents = controller.activeWebsiteWebContents;
  await waitUntilLoaded(ownerContents);
  const owner = smokeCallerContext(controller, ownerTabId);
  const jobId = 'job_artifact_smoke';
  const outputRoot = join(artifacts.storage.engineStagingRoot, engineInstanceId, jobId);
  mkdirSync(outputRoot, { recursive: true, mode: 0o700 });
  const image = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    'base64',
  );
  const audio = smokeWaveBytes();
  const video = Buffer.from(
    'GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAI8EU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHWTbuMU6uEElTDZ1OsggEjTbuMU6uEHFO7a1OsggIm7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmsCrXsYMPQkBNgIxMYXZmNjIuMy4xMDBXQYxMYXZmNjIuMy4xMDBEiYhAaQAAAAAAABZUrmvIrgEAAAAAAAA/14EBc8WIqQWwMsmlRracgQAitZyDdW5kiIEAhoVWX1ZQOYOBASPjg4QCYloA4JCwgRC6gRCagQJVsIRVuYEBElTDZ0B/c3OfY8CAZ8iZRaOHRU5DT0RFUkSHjExhdmY2Mi4zLjEwMHNz2mPAi2PFiKkFsDLJpUa2Z8ilRaOHRU5DT0RFUkSHmExhdmM2Mi4xMS4xMDAgbGlidnB4LXZwOWfIoUWjiERVUkFUSU9ORIeTMDA6MDA6MDAuMjAwMDAwMDAwAB9DtnX554EAo6CBAACAgkmDQgAA8AD2ADgkHBhKAAAwYAAAEL///UiMAKOTgQAoAIYAQJKcAFAAAANgAABCQKOTgQBQAIYAQJKcAE7gAANgAABCQKOTgQB4AIYAQJKcAFAAAANgAABCQKOTgQCgAIYAQJKcAE1AAANgAABCQKOTgQC3iveBAfGCAajwgQM=',
    'base64',
  );
  const staged = [
    ['output_smoke_image', 'image.png', 'image/png', image],
    ['output_smoke_audio', 'audio.wav', 'audio/wav', audio],
    ['output_smoke_video', 'video.webm', 'video/webm', video],
  ] as const;
  for (const [, name, , bytes] of staged) writeFileSync(join(outputRoot, name), bytes);
  const published = (await artifacts.publishEngineResult(owner, engineInstanceId, jobId, {
    kind: 'multi',
    outputs: staged.map(([outputId, name, mimeType, bytes]) => ({
      kind: 'artifact',
      artifact: {
        outputId,
        internalPath: join(outputRoot, name),
        mimeType,
        sizeBytes: bytes.byteLength,
        name,
      },
    })),
  })) as {
    outputs: Array<{ artifact: { id: string; internalPath?: string } }>;
  };
  const ids = published.outputs.map((output) => output.artifact.id);
  const media = (await ownerContents.executeJavaScript(`
    (async () => {
      const ids = ${JSON.stringify(ids)};
      const urls = [];
      for (const id of ids) {
        const result = await globalThis.localAI.getArtifactUrl(id);
        if (!result.ok) throw new Error(result.error.code);
        urls.push(result.value);
      }
      const waitFor = (target, event) => new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Timed out waiting for ' + event)), 10000);
        target.addEventListener(event, () => { clearTimeout(timeout); resolve(); }, { once: true });
        target.addEventListener('error', () => { clearTimeout(timeout); reject(new Error('Media error')); }, { once: true });
      });
      const image = new Image();
      image.src = urls[0];
      await waitFor(image, 'load');
      const audio = new Audio(urls[1]);
      audio.load();
      await waitFor(audio, 'loadedmetadata');
      const video = document.createElement('video');
      video.src = urls[2];
      video.load();
      await waitFor(video, 'loadedmetadata');
      video.currentTime = Math.min(0.1, video.duration / 2);
      await waitFor(video, 'seeked');
      const range = await fetch(urls[2], { headers: { Range: 'bytes=8-15' } });
      const blob = await fetch(urls[1], { cache: 'no-store' }).then((response) => response.blob());
      return {
        urls,
        imageRendered: image.naturalWidth === 1 && image.naturalHeight === 1,
        audioPlayable: Number.isFinite(audio.duration) && audio.duration > 0,
        videoSeeked: video.currentTime > 0 && Number.isFinite(video.duration),
        rangeServed:
          range.status === 206 &&
          range.headers.get('content-range') === 'bytes 8-15/${video.byteLength}' &&
          (await range.arrayBuffer()).byteLength === 8,
        blobRead: blob.size === ${audio.byteLength},
        pathsHidden: !JSON.stringify({ ids, urls }).includes('/Users/')
      };
    })()
  `)) as Record<string, unknown> & { urls: string[] };

  controller.activateTab(crossOriginTabId);
  await waitUntilLoaded(controller.activeWebsiteWebContents);
  const crossOriginRejected = (await controller.activeWebsiteWebContents.executeJavaScript(`
    fetch(${JSON.stringify(media.urls[2])}).then((response) => !response.ok, () => true)
  `)) as boolean;

  const navigationInitialURL = ownerContents.getURL();
  const navigationTabId = controller.createTab(navigationInitialURL, true);
  const navigationContents = controller.activeWebsiteWebContents;
  await waitUntilLoaded(
    navigationContents,
    navigationInitialURL,
    'artifact navigation smoke document',
  );
  const navigationOwner = smokeCallerContext(controller, navigationTabId);
  const navigationJobId = 'job_artifact_navigation';
  const navigationRoot = join(
    artifacts.storage.engineStagingRoot,
    engineInstanceId,
    navigationJobId,
  );
  mkdirSync(navigationRoot, { recursive: true, mode: 0o700 });
  writeFileSync(join(navigationRoot, 'image.png'), image);
  const navigationResult = (await artifacts.publishEngineResult(
    navigationOwner,
    engineInstanceId,
    navigationJobId,
    {
      kind: 'artifact',
      artifact: {
        outputId: 'output_navigation_image',
        internalPath: join(navigationRoot, 'image.png'),
        mimeType: 'image/png',
        sizeBytes: image.byteLength,
      },
    },
  )) as { artifact: { id: string } };
  const navigationUrlResult = (await navigationContents.executeJavaScript(`
    globalThis.localAI.getArtifactUrl(${JSON.stringify(navigationResult.artifact.id)})
  `)) as { ok: boolean; value: string };
  const navigatedUrl = new URL(navigationContents.getURL());
  navigatedUrl.searchParams.set('artifact-navigation', '1');
  await navigateAndWait(
    navigationContents,
    () => void navigationContents.loadURL(navigatedUrl.href),
  );
  const navigationRejected = (await navigationContents.executeJavaScript(`
    fetch(${JSON.stringify(navigationUrlResult.value)}).then((response) => !response.ok, () => true)
  `)) as boolean;
  await controller.closeTab(navigationTabId);

  const checks = {
    imageRendered: media.imageRendered === true,
    audioPlayable: media.audioPlayable === true,
    videoSeeked: media.videoSeeked === true,
    rangeServed: media.rangeServed === true,
    blobRead: media.blobRead === true,
    pathsHidden:
      media.pathsHidden === true && JSON.stringify(published).includes('/Users/') === false,
    crossOriginRejected,
    navigationRejected,
  };
  return { ...checks, passed: Object.values(checks).every(Boolean) };
}

function smokeCallerContext(
  controller: BrowserWindowController,
  tabId: string,
): BrowserCallerContext {
  const tab = controller.tabs.getTab(tabId);
  const contents = controller.activeWebsiteWebContents;
  return {
    tabId,
    webContentsId: contents.id,
    frameId: contents.mainFrame.routingId,
    frameUrl: contents.getURL(),
    origin: new URL(contents.getURL()).origin,
    documentId: tab.activeDocumentId,
  };
}

function smokeWaveBytes(): Buffer {
  const frames = 3_200;
  const result = Buffer.alloc(44 + frames * 2);
  result.write('RIFF', 0);
  result.writeUInt32LE(36 + frames * 2, 4);
  result.write('WAVEfmt ', 8);
  result.writeUInt32LE(16, 16);
  result.writeUInt16LE(1, 20);
  result.writeUInt16LE(1, 22);
  result.writeUInt32LE(16_000, 24);
  result.writeUInt32LE(32_000, 28);
  result.writeUInt16LE(2, 32);
  result.writeUInt16LE(16, 34);
  result.write('data', 36);
  result.writeUInt32LE(frames * 2, 40);
  return result;
}

async function runDeveloperAlphaFoundation(
  controller: BrowserWindowController,
  chatTabId: string,
  transcriptionTabId: string,
): Promise<{
  passed: boolean;
  status: string;
  downloadCompleted: boolean;
  queueObserved: boolean;
  trustedUIShowsBoth: boolean;
  websiteIsolation: boolean;
  bothResident: boolean;
  performance: {
    engineStartupMs: number;
    inspectMs: number;
    downloadMs: number;
    loadMs: number;
    modelInspectMs: number;
    modelPrepareMs: number;
    queueWaitMs: number;
    generationMs: number;
    browserFrameMs: number;
    cancellationLatencyMs: number;
    artifactThroughputMiBps: number;
    mainProcessRssDeltaBytes: number;
  };
}> {
  if (process.env.ORYN_ALPHA_INTEGRATION_SMOKE !== '1') {
    return {
      passed: true,
      status: 'not-requested',
      downloadCompleted: false,
      queueObserved: false,
      trustedUIShowsBoth: false,
      websiteIsolation: false,
      bothResident: false,
      performance: {
        engineStartupMs: 0,
        inspectMs: 0,
        downloadMs: 0,
        loadMs: 0,
        modelInspectMs: 0,
        modelPrepareMs: 0,
        queueWaitMs: 0,
        generationMs: 0,
        browserFrameMs: 0,
        cancellationLatencyMs: 0,
        artifactThroughputMiBps: 0,
        mainProcessRssDeltaBytes: 0,
      },
    };
  }

  const chatRepository = 'trl-internal-testing/tiny-Qwen2ForCausalLM-2.5';
  const transcriptionRepository = 'Xenova/tiny-random-WhisperForConditionalGeneration';
  controller.activateTab(chatTabId);
  const chatContents = controller.activeWebsiteWebContents;
  await waitUntilLoaded(chatContents);
  controller.activateTab(transcriptionTabId);
  const transcriptionContents = controller.activeWebsiteWebContents;
  await waitUntilLoaded(transcriptionContents);
  const chatOrigin = new URL(chatContents.getURL()).origin;
  const transcriptionOrigin = new URL(transcriptionContents.getURL()).origin;
  modelPreferences.setSite(chatOrigin, 'chat', {
    provider: 'huggingface',
    repository: chatRepository,
    revision: 'f56c2fbf7fabc03ffff3494d3c2f8d06c774493d',
  });
  modelPreferences.setSite(transcriptionOrigin, 'transcription', {
    provider: 'huggingface',
    repository: transcriptionRepository,
    revision: 'e1a9e01a339a6c8ba2aadb92a271624e46f86d05',
  });

  const chatModel = {
    provider: 'huggingface' as const,
    repository: chatRepository,
    revision: 'f56c2fbf7fabc03ffff3494d3c2f8d06c774493d',
  };
  const transcriptionModel = {
    provider: 'huggingface' as const,
    repository: transcriptionRepository,
    revision: 'e1a9e01a339a6c8ba2aadb92a271624e46f86d05',
  };
  const inspectStartedAt = performance.now();
  await Promise.all([engine.inspectModel(chatModel), engine.inspectModel(transcriptionModel)]);
  const inspectMs = performance.now() - inspectStartedAt;
  const downloadStartedAt = performance.now();
  await engine.installModel('operation_alpha_install_chat', chatModel);
  await engine.installModel('operation_alpha_install_transcription', transcriptionModel);
  const downloadMs = performance.now() - downloadStartedAt;
  const loadStartedAt = performance.now();
  await engine.loadModel('operation_alpha_load_chat', chatModel);
  await engine.loadModel('operation_alpha_load_transcription', transcriptionModel);
  const loadMs = performance.now() - loadStartedAt;

  controller.activateTab(chatTabId);
  chatContents.downloadURL(`${process.env.ORYN_START_URL}/__oryn_alpha_download`);
  await waitForCondition(
    () => browserData.downloads.list().some((item) => item.state === 'completed'),
    'normal browser download',
  );
  const downloadCompleted = browserData.downloads
    .list()
    .some(
      (item) =>
        item.state === 'completed' &&
        item.filename === 'oryn-alpha.txt' &&
        item.receivedBytes === item.totalBytes,
    );

  const modelRequestStarted = performance.now();
  await chatContents.executeJavaScript(`
    (() => {
      const prompt = document.querySelector('#prompt');
      const maxTokens = document.querySelector('#max-tokens');
      if (!(prompt instanceof HTMLTextAreaElement) || !(maxTokens instanceof HTMLSelectElement)) {
        throw new Error('Chat controls missing.');
      }
      prompt.value = 'Write a detailed explanation of local inference scheduling and keep going.';
      maxTokens.value = '96';
      document.querySelector('#chat-form')?.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true })
      );
    })()
  `);
  await waitForSmokePermission(controller, 'chat');
  const modelInspectMs = performance.now() - modelRequestStarted;
  const permissionAcceptedAt = performance.now();
  await acceptSmokePermission(controller, 'chat');
  await waitForCondition(
    () =>
      jobs
        .activitySnapshot()
        .jobs.some((item) => item.origin === chatOrigin && item.state === 'streaming'),
    'running alpha chat job',
    180_000,
  );
  const generationStartedAt = performance.now();
  const modelPrepareMs = generationStartedAt - permissionAcceptedAt;

  await chatContents.executeJavaScript(`
    (async () => {
      const prompt = document.querySelector('#prompt');
      if (!(prompt instanceof HTMLTextAreaElement)) throw new Error('Chat prompt missing.');
      for (let index = 1; index <= 5; index += 1) {
        prompt.value = 'Queued alpha workload ' + index + ': explain one scheduler invariant.';
        document.querySelector('#chat-form')?.dispatchEvent(
          new Event('submit', { bubbles: true, cancelable: true })
        );
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
    })()
  `);
  await waitForCondition(() => jobs.activitySnapshot().queued >= 3, 'alpha chat backlog', 30_000);
  const cancellable = jobs
    .activitySnapshot()
    .jobs.find((item) => item.origin === chatOrigin && item.state === 'queued');
  if (!cancellable) throw new Error('Queued alpha cancellation target is unavailable.');
  const cancellationStartedAt = performance.now();
  await jobs.cancelTrusted(cancellable.jobId);
  await waitForCondition(
    () => !jobs.activitySnapshot().jobs.some((item) => item.jobId === cancellable.jobId),
    'trusted queued-job cancellation',
  );
  const cancellationLatencyMs = performance.now() - cancellationStartedAt;

  const frameStartedAt = performance.now();
  await controller.trustedWebContents.executeJavaScript(
    'new Promise((resolve) => requestAnimationFrame(() => resolve(true)))',
  );
  const browserFrameMs = performance.now() - frameStartedAt;

  controller.activateTab(transcriptionTabId);
  await transcriptionContents.executeJavaScript(`
    (() => {
      const sampleRate = 16000;
      const frames = sampleRate * 8;
      const buffer = new ArrayBuffer(44 + frames * 2);
      const view = new DataView(buffer);
      const write = (offset, value) => [...value].forEach((char, index) =>
        view.setUint8(offset + index, char.charCodeAt(0))
      );
      write(0, 'RIFF'); view.setUint32(4, 36 + frames * 2, true); write(8, 'WAVE');
      write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
      view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
      view.setUint16(34, 16, true); write(36, 'data'); view.setUint32(40, frames * 2, true);
      for (let index = 0; index < frames; index += 1) {
        view.setInt16(44 + index * 2, Math.sin(index / sampleRate * Math.PI * 440) * 5000, true);
      }
      const transfer = new DataTransfer();
      transfer.items.add(new File([buffer], 'alpha.wav', { type: 'audio/wav' }));
      const input = document.querySelector('#audio-file');
      if (!(input instanceof HTMLInputElement)) throw new Error('Transcription control missing.');
      input.files = transfer.files;
      document.querySelector('#transcription-form')?.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true })
      );
    })()
  `);
  await waitForSmokePermission(controller, 'transcription');
  await acceptSmokePermission(controller, 'transcription');
  await waitForCondition(
    () => {
      const snapshot = jobs.activitySnapshot();
      return (
        snapshot.running === 1 &&
        snapshot.queued >= 1 &&
        snapshot.jobs.some((item) => item.origin === transcriptionOrigin && item.state === 'queued')
      );
    },
    'cross-origin global queue',
    180_000,
  );
  const queueObservedAt = performance.now();
  const queueObserved = true;
  const trustedUIShowsBoth = (await controller.trustedWebContents.executeJavaScript(`
    (async () => {
      if (!document.querySelector('[aria-label="Oryn Local AI"]')) {
        document.querySelector('[aria-label="Local AI activity"]')?.click();
      }
      await new Promise((resolve) => setTimeout(resolve, 0));
      const text = document.querySelector('[aria-label="Oryn Local AI"]')?.textContent ?? '';
      document.querySelector('[aria-label="Close Local AI"]')?.click();
      return text.includes(${JSON.stringify(chatRepository)}) &&
        text.includes(${JSON.stringify(transcriptionRepository)}) &&
        text.includes('streaming') && text.includes('queued');
    })()
  `)) as boolean;

  const websiteIsolation =
    ((await chatContents.executeJavaScript(
      `!document.body.textContent.includes(${JSON.stringify(transcriptionRepository)})`,
    )) as boolean) &&
    ((await transcriptionContents.executeJavaScript(
      `!document.body.textContent.includes(${JSON.stringify(chatRepository)})`,
    )) as boolean);

  await waitForCondition(
    () =>
      !jobs
        .activitySnapshot()
        .jobs.some((item) => item.origin === transcriptionOrigin && item.state === 'queued'),
    'queued transcription start',
    240_000,
  );
  const queueWaitMs = performance.now() - queueObservedAt;
  await waitForCondition(
    () =>
      Promise.all([
        chatContents.executeJavaScript(
          `(() => {
            const statuses = Array.from(document.querySelectorAll('[data-chat-job] .job-status'))
              .map((item) => item.textContent ?? '');
            return statuses.length >= 6 &&
              statuses.some((value) => value === 'Chat complete.') &&
              statuses.every((value) => value === 'Chat complete.' || value === 'Chat cancelled.');
          })()`,
        ),
        transcriptionContents.executeJavaScript(
          `(document.querySelector('#status')?.textContent ?? '') === 'Transcription complete.'`,
        ),
      ]).then((values) => values.every(Boolean)),
    'two-origin workload completion',
    240_000,
  );
  const generationMs = performance.now() - generationStartedAt;
  const inventory = await engine.getModelInventory();
  const bothResident = [chatRepository, transcriptionRepository].every((repository) =>
    inventory.models.some(
      (item) => item.model.repository === repository && item.residencyState === 'loaded',
    ),
  );
  const artifactPerformance = await measureLargeArtifactPerformance(controller, chatTabId);
  const passed =
    downloadCompleted &&
    queueObserved &&
    trustedUIShowsBoth &&
    websiteIsolation &&
    bothResident &&
    engineStartupMs > 0 &&
    inspectMs > 0 &&
    downloadMs > 0 &&
    loadMs > 0 &&
    modelInspectMs > 0 &&
    modelPrepareMs > 0 &&
    queueWaitMs > 0 &&
    generationMs > 0 &&
    browserFrameMs > 0 &&
    cancellationLatencyMs > 0 &&
    artifactPerformance.throughputMiBps > 0;
  return {
    passed,
    status: passed ? 'completed' : 'failed',
    downloadCompleted,
    queueObserved,
    trustedUIShowsBoth,
    websiteIsolation,
    bothResident,
    performance: {
      engineStartupMs: Math.round(engineStartupMs),
      inspectMs: Math.round(inspectMs),
      downloadMs: Math.round(downloadMs),
      loadMs: Math.round(loadMs),
      modelInspectMs: Math.round(modelInspectMs),
      modelPrepareMs: Math.round(modelPrepareMs),
      queueWaitMs: Math.round(queueWaitMs),
      generationMs: Math.round(generationMs),
      browserFrameMs: Math.round(browserFrameMs * 100) / 100,
      cancellationLatencyMs: Math.round(cancellationLatencyMs * 100) / 100,
      artifactThroughputMiBps: artifactPerformance.throughputMiBps,
      mainProcessRssDeltaBytes: artifactPerformance.rssDeltaBytes,
    },
  };
}

async function measureLargeArtifactPerformance(
  controller: BrowserWindowController,
  ownerTabId: string,
): Promise<{ throughputMiBps: number; rssDeltaBytes: number }> {
  const engineInstanceId = engine.engineInstanceId;
  if (!engineInstanceId) throw new Error('Engine identity unavailable for artifact measurement.');
  controller.activateTab(ownerTabId);
  const owner = smokeCallerContext(controller, ownerTabId);
  const jobId = 'job_alpha_artifact_performance';
  const outputRoot = join(artifacts.storage.engineStagingRoot, engineInstanceId, jobId);
  mkdirSync(outputRoot, { recursive: true, mode: 0o700 });
  const outputPath = join(outputRoot, 'large-media.bin');
  const chunk = Buffer.alloc(1024 * 1024, 0x5a);
  const file = openSync(outputPath, 'wx', 0o600);
  try {
    for (let index = 0; index < 32; index += 1) writeSync(file, chunk);
  } finally {
    closeSync(file);
  }
  const published = (await artifacts.publishEngineResult(owner, engineInstanceId, jobId, {
    kind: 'artifact',
    artifact: {
      outputId: 'output_alpha_large_media',
      internalPath: outputPath,
      mimeType: 'application/octet-stream',
      sizeBytes: 32 * 1024 * 1024,
      name: 'large-media.bin',
    },
  })) as { artifact: { id: string } };
  const url = artifacts.getUrl(owner, published.artifact.id);
  const rssBefore = process.memoryUsage().rss;
  let peakRss = rssBefore;
  let bytes = 0;
  const startedAt = performance.now();
  const response = await new ArtifactProtocolHandler(artifacts).serve({ url, method: 'GET' });
  if (!response.body) throw new Error('Large artifact stream is unavailable.');
  for await (const value of response.body) {
    bytes += Buffer.byteLength(value);
    peakRss = Math.max(peakRss, process.memoryUsage().rss);
  }
  const durationMs = performance.now() - startedAt;
  await artifacts.release(owner, published.artifact.id);
  if (bytes !== 32 * 1024 * 1024 || durationMs <= 0) {
    throw new Error('Large artifact measurement was incomplete.');
  }
  return {
    throughputMiBps: Math.round((bytes / 1024 / 1024 / (durationMs / 1000)) * 100) / 100,
    rssDeltaBytes: Math.max(0, peakRss - rssBefore),
  };
}

async function waitForSmokePermission(
  controller: BrowserWindowController,
  workload: string,
): Promise<void> {
  await waitForCondition(
    () =>
      controller.trustedWebContents
        .executeJavaScript(
          'Boolean(document.querySelector(\'[aria-label="Local AI permission"]\'))',
        )
        .then(Boolean),
    `${workload} permission prompt`,
    60_000,
  );
}

async function acceptSmokePermission(
  controller: BrowserWindowController,
  workload: string,
): Promise<void> {
  const allowed = (await controller.trustedWebContents.executeJavaScript(`
    (() => {
      const dialog = document.querySelector('[aria-label="Local AI permission"]');
      const button = Array.from(dialog?.querySelectorAll('button') ?? [])
        .find((candidate) =>
          ['Allow once', 'Trust and allow once', 'Use anyway once'].includes(
            candidate.textContent?.trim() ?? '',
          ),
        );
      button?.click();
      return Boolean(button);
    })()
  `)) as boolean;
  if (!allowed) throw new Error(`${workload} permission could not be accepted.`);
}

async function runSharedTaskSmoke(
  controller: BrowserWindowController,
  tabId: string,
): Promise<void> {
  controller.activateTab(tabId);
  const contents = controller.activeWebsiteWebContents;
  await contents.executeJavaScript(`
    globalThis.sharedTaskSmoke = null;
    void (async () => {
      const api = globalThis.localAI;
      const session = await api.requestModel('model_request_shared_task', {
        role: 'classification',
        preferredModel: {provider: 'huggingface', repository: 'cross-encoder/nli-deberta-v3-xsmall', revision: 'a150876415327c80daeff35ca6f68f5ed8cf5c24'},
        requirements: {task: 'zero-shot-classification', inputModalities: ['text'], outputModalities: ['structured']}
      });
      if (!session.ok) throw new Error(session.error.code);
      if ('runtime' in session.value) throw new Error('Website runtime privacy violation');
      const input = {text: 'A cat is sleeping on a sofa.', labels: ['animals', 'finance']};
      const rejected = await api.invokeJob({sessionId: session.value.sessionId, input: {...input, backend: 'cpu'}});
      if (rejected.ok || rejected.error.code !== 'JOB_INPUT_INVALID') throw new Error('Runtime options were accepted');
      const terminal = new Promise(resolve => {
        const stop = api.onJobEvent(event => {
          if (event.sessionId === session.value.sessionId && ['job.completed', 'job.failed', 'job.cancelled'].includes(event.event)) {
            stop(); resolve(event);
          }
        });
      });
      const invoked = await api.invokeJob({sessionId: session.value.sessionId, input});
      if (!invoked.ok) throw new Error(invoked.error.code);
      const completed = await terminal;
      const predictions = completed.data?.result?.value?.predictions;
      if (completed.event !== 'job.completed' || predictions?.[0]?.label !== 'animals') throw new Error('Shared task result failed');
      await api.releaseModelSession(session.value.sessionId);
      globalThis.sharedTaskSmoke = {passed: true};
    })().catch(error => {globalThis.sharedTaskSmoke = {error: error.message}});
  `);
  await waitForCondition(
    async () => {
      const error = await contents.executeJavaScript('globalThis.sharedTaskSmoke?.error');
      if (error) throw new Error(String(error));
      return controller.trustedWebContents
        .executeJavaScript(`Boolean(document.querySelector('[aria-label="Local AI permission"]'))`)
        .then(Boolean);
    },
    'shared task permission prompt',
    60_000,
  );
  await acceptSmokePermission(controller, 'shared task');
  await waitForCondition(
    async () => {
      const result = await contents.executeJavaScript('globalThis.sharedTaskSmoke');
      if (result?.error) throw new Error(String(result.error));
      return result?.passed === true;
    },
    'shared task website completion',
    180_000,
  );
  console.log(
    'ORYN_SHARED_TASK_SMOKE_RESULT',
    JSON.stringify({
      passed: true,
      task: 'zero-shot-classification',
      runtimeFieldsRejected: true,
      sessionRuntimeHidden: true,
    }),
  );
}

async function runTranscriptionFoundation(
  controller: BrowserWindowController,
  tabId: string,
): Promise<{ passed: boolean; status: string; details: string; installed: boolean }> {
  if (process.env.ORYN_TRANSCRIPTION_SMOKE !== '1') {
    return { passed: true, status: 'not-requested', details: '', installed: false };
  }
  controller.activateTab(tabId);
  const contents = controller.activeWebsiteWebContents;
  await waitUntilLoaded(contents);
  const origin = new URL(contents.getURL()).origin;
  const repository = 'Xenova/tiny-random-WhisperForConditionalGeneration';
  modelPreferences.setSite(origin, 'transcription', {
    provider: 'huggingface',
    repository,
    revision: 'e1a9e01a339a6c8ba2aadb92a271624e46f86d05',
  });
  await contents.executeJavaScript(`
    (() => {
      const sampleRate = 16000;
      const frames = 3200;
      const buffer = new ArrayBuffer(44 + frames * 2);
      const view = new DataView(buffer);
      const write = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
      write(0, 'RIFF'); view.setUint32(4, 36 + frames * 2, true); write(8, 'WAVE');
      write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
      view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
      view.setUint16(34, 16, true); write(36, 'data'); view.setUint32(40, frames * 2, true);
      for (let index = 0; index < frames; index += 1) {
        view.setInt16(44 + index * 2, Math.sin(index / sampleRate * Math.PI * 880) * 4000, true);
      }
      const transfer = new DataTransfer();
      transfer.items.add(new File([buffer], 'smoke.wav', { type: 'audio/wav' }));
      const input = document.querySelector('#audio-file');
      if (!(input instanceof HTMLInputElement)) throw new Error('Transcription file input missing.');
      input.files = transfer.files;
      document.querySelector('#transcription-form')?.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true })
      );
    })()
  `);
  await waitForCondition(
    () =>
      controller.trustedWebContents
        .executeJavaScript(
          'Boolean(document.querySelector(\'[aria-label="Local AI permission"]\'))',
        )
        .then(Boolean),
    'transcription permission prompt',
    60_000,
  );
  await acceptSmokePermission(controller, 'Transcription');
  await waitForCondition(
    () =>
      contents
        .executeJavaScript(
          `
          ['Transcription complete.', 'Transcription cancelled.'].includes(
            document.querySelector('#status')?.textContent ?? ''
          ) || (document.querySelector('#status')?.textContent ?? '').startsWith('Could not transcribe:')
        `,
        )
        .then(Boolean),
    'complete website transcription',
    180_000,
  );
  const page = (await contents.executeJavaScript(`
    ({
      status: document.querySelector('#status')?.textContent ?? '',
      details: document.querySelector('#details')?.textContent ?? '',
      transcript: document.querySelector('#transcript')?.textContent ?? '',
      localPathsHidden: !document.body.textContent.includes('/Users/')
    })
  `)) as { status: string; details: string; transcript: string; localPathsHidden: boolean };
  const inventory = await engine.getModelInventory();
  const installed = inventory.models.some(
    (item) => item.model.repository === repository && item.installationState === 'installed',
  );
  return {
    passed:
      page.status === 'Transcription complete.' &&
      page.details.includes(repository) &&
      page.transcript.length > 0 &&
      page.localPathsHidden &&
      installed,
    status: page.status,
    details: page.details,
    installed,
  };
}

async function runChatFoundation(
  controller: BrowserWindowController,
  tabId: string,
): Promise<{
  passed: boolean;
  status: string;
  model: string;
  backend: string;
  deltaCount: number;
  installed: boolean;
}> {
  if (process.env.ORYN_CHAT_SMOKE !== '1') {
    return {
      passed: true,
      status: 'not-requested',
      model: '',
      backend: '',
      deltaCount: 0,
      installed: false,
    };
  }
  controller.activateTab(tabId);
  const contents = controller.activeWebsiteWebContents;
  await waitUntilLoaded(contents);
  const origin = new URL(contents.getURL()).origin;
  const repository = 'trl-internal-testing/tiny-Qwen2ForCausalLM-2.5';
  modelPreferences.setSite(origin, 'chat', {
    provider: 'huggingface',
    repository,
    revision: 'f56c2fbf7fabc03ffff3494d3c2f8d06c774493d',
  });
  await contents.executeJavaScript(`
    (() => {
      const prompt = document.querySelector('#prompt');
      if (!(prompt instanceof HTMLTextAreaElement)) throw new Error('Chat prompt missing.');
      prompt.value = 'Reply with a short greeting.';
      document.querySelector('#chat-form')?.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true })
      );
    })()
  `);
  await waitForCondition(
    async () => {
      const status = await contents.executeJavaScript(
        "document.querySelector('[data-chat-job] .job-status')?.textContent ?? ''",
      );
      if (String(status).startsWith('Chat failed:')) throw new Error(String(status));
      return controller.trustedWebContents
        .executeJavaScript(
          `
          (() => {
            const dialog = document.querySelector('[aria-label="Local AI permission"]');
            const button = Array.from(dialog?.querySelectorAll('button') ?? []).find((candidate) =>
              ['Allow once', 'Trust and allow once', 'Use anyway once'].includes(
                candidate.textContent?.trim() ?? ''
              ) && !candidate.disabled
            );
            button?.click();
            return Boolean(button);
          })()
        `,
        )
        .then(Boolean);
    },
    'accept actionable chat permission prompt',
    60_000,
  );
  await waitForCondition(
    () =>
      contents
        .executeJavaScript(
          `
          (() => {
            const value = document.querySelector('[data-chat-job] .job-status')?.textContent ?? '';
            return value === 'Chat complete.' || value === 'Chat cancelled.' || value.startsWith('Chat failed:');
          })()
        `,
        )
        .then(Boolean),
    'complete website chat generation',
    180_000,
  );
  const page = (await contents.executeJavaScript(`
    (() => {
      const card = document.querySelector('[data-chat-job]');
      return {
        status: card?.querySelector('.job-status')?.textContent ?? '',
        answer: card?.querySelector('.assistant')?.textContent ?? '',
        model: card?.getAttribute('data-model') ?? '',
        backend: card?.getAttribute('data-backend') ?? '',
        deltaCount: Number(card?.getAttribute('data-delta-count') ?? '0'),
        localPathsHidden: !document.body.textContent.includes('/Users/')
      };
    })()
  `)) as {
    status: string;
    answer: string;
    model: string;
    backend: string;
    deltaCount: number;
    localPathsHidden: boolean;
  };
  const inventory = await engine.getModelInventory();
  const installed = inventory.models.some(
    (item) => item.model.repository === repository && item.installationState === 'installed',
  );
  return {
    passed:
      page.status === 'Chat complete.' &&
      page.answer.length > 0 &&
      page.model === repository &&
      page.backend === '' &&
      page.deltaCount > 0 &&
      page.localPathsHidden &&
      installed,
    status: page.status,
    model: page.model,
    backend: page.backend,
    deltaCount: page.deltaCount,
    installed,
  };
}

async function runSongFoundation(
  controller: BrowserWindowController,
  tabId: string,
  backgroundTabId: string,
): Promise<{
  passed: boolean;
  status: string;
  model: string;
  backend: string;
  artifactBytes: number;
  runtimeVisible: boolean;
  completedInBackground: boolean;
  audioPlayable: boolean;
  saveAsAvailable: boolean;
  explicitUploadAvailable: boolean;
  insecureUploadRejected: boolean;
  installed: boolean;
}> {
  if (process.env.ORYN_SONG_SMOKE !== '1') {
    return {
      passed: true,
      status: 'not-requested',
      model: '',
      backend: '',
      artifactBytes: 0,
      runtimeVisible: false,
      completedInBackground: false,
      audioPlayable: false,
      saveAsAvailable: false,
      explicitUploadAvailable: false,
      insecureUploadRejected: false,
      installed: false,
    };
  }
  controller.activateTab(tabId);
  const contents = controller.activeWebsiteWebContents;
  await waitUntilLoaded(contents);
  const origin = new URL(contents.getURL()).origin;
  const repository =
    process.env.ORYN_EXPECT_SONG_MODEL ?? 'hf-tiny-v2/tiny-random-MusicgenForConditionalGeneration';
  if (process.env.ORYN_CAPABILITY_SONG_SMOKE !== '1') {
    modelPreferences.setSite(origin, 'song-generation', {
      provider: 'huggingface',
      repository,
      revision: '4606e560821ed4c10d76d6a6513c893ec3cf2ae0',
    });
  }
  await contents.executeJavaScript(`
    (() => {
      const prompt = document.querySelector('#prompt');
      const duration = document.querySelector('#duration');
      if (!(prompt instanceof HTMLTextAreaElement) || !(duration instanceof HTMLSelectElement)) {
        throw new Error('Song controls missing.');
      }
      prompt.value = 'Dark atmospheric techno with a slow pulse';
      duration.value = '15';
      document.querySelector('#song-form')?.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true })
      );
    })()
  `);
  await waitForCondition(
    () =>
      controller.trustedWebContents
        .executeJavaScript(
          'Boolean(document.querySelector(\'[aria-label="Local AI permission"]\'))',
        )
        .then(Boolean),
    'song-generation permission prompt',
    60_000,
  );
  await acceptSmokePermission(controller, 'Song-generation');
  await waitForCondition(
    () =>
      controller.trustedWebContents
        .executeJavaScript(
          `(document.querySelector('[aria-label="Local AI activity"]')?.textContent ?? '').includes('running')`,
        )
        .then(Boolean),
    'trusted song runtime indicator',
    120_000,
  );
  const runtimeVisible = true;
  controller.activateTab(backgroundTabId);
  await waitForCondition(
    () =>
      contents
        .executeJavaScript(
          `
          (() => {
            const value = document.querySelector('#status')?.textContent ?? '';
            return value === 'Song generation complete.' ||
              value === 'Song generation cancelled.' ||
              value.startsWith('Song generation failed:');
          })()
        `,
        )
        .then(Boolean),
    'complete background song generation',
    240_000,
  );
  const completedInBackground = controller.activeTabId === backgroundTabId;
  controller.activateTab(tabId);
  const page = (await contents.executeJavaScript(`
    (async () => {
      const result = document.querySelector('#result');
      const audio = document.querySelector('#audio');
      if (!(audio instanceof HTMLAudioElement)) throw new Error('Song audio control missing.');
      const completedStatus = document.querySelector('#status')?.textContent ?? '';
      if (completedStatus === 'Song generation complete.' && audio.readyState < 1) {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('Audio metadata timed out.')), 10000);
          audio.addEventListener('loadedmetadata', () => { clearTimeout(timeout); resolve(); }, { once: true });
          audio.addEventListener('error', () => { clearTimeout(timeout); reject(new Error('Audio failed.')); }, { once: true });
          audio.load();
        });
      }
      const uploadUrl = document.querySelector('#upload-url');
      if (!(uploadUrl instanceof HTMLInputElement)) throw new Error('Upload URL control missing.');
      uploadUrl.value = 'http://insecure.example/upload';
      document.querySelector('#upload')?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
      return {
        status: completedStatus,
        model: result?.getAttribute('data-model') ?? '',
        backend: result?.getAttribute('data-backend') ?? '',
        artifactBytes: Number(result?.getAttribute('data-artifact-bytes') ?? '0'),
        audioPlayable: Number.isFinite(audio.duration) && audio.duration > 0,
        saveAsAvailable: document.querySelector('#save') instanceof HTMLButtonElement,
        explicitUploadAvailable: document.querySelector('#upload') instanceof HTMLButtonElement,
        insecureUploadRejected: (document.querySelector('#status')?.textContent ?? '').includes('Enter an HTTPS destination'),
        pathsHidden: !document.body.textContent.includes('/Users/')
      };
    })()
  `)) as {
    status: string;
    model: string;
    backend: string;
    artifactBytes: number;
    audioPlayable: boolean;
    saveAsAvailable: boolean;
    explicitUploadAvailable: boolean;
    insecureUploadRejected: boolean;
    pathsHidden: boolean;
  };
  const inventory = await engine.getModelInventory();
  const installed = inventory.models.some(
    (item) => item.model.repository === repository && item.installationState === 'installed',
  );
  const passed =
    page.status === 'Song generation complete.' &&
    page.model === repository &&
    page.backend === '' &&
    page.artifactBytes > 1_000_000 &&
    runtimeVisible &&
    completedInBackground &&
    page.audioPlayable &&
    page.saveAsAvailable &&
    page.explicitUploadAvailable &&
    page.insecureUploadRejected &&
    page.pathsHidden &&
    installed;
  return {
    passed,
    status: page.status,
    model: page.model,
    backend: page.backend,
    artifactBytes: page.artifactBytes,
    runtimeVisible,
    completedInBackground,
    audioPlayable: page.audioPlayable,
    saveAsAvailable: page.saveAsAvailable,
    explicitUploadAvailable: page.explicitUploadAvailable,
    insecureUploadRejected: page.insecureUploadRejected,
    installed,
  };
}

function finishSmokeTest(exitCode: number): void {
  requestedExitCode = exitCode;
  app.quit();
}

async function verifyMarkers(
  controller: BrowserWindowController,
  tabs: Array<{ tabId: string; marker: string }>,
): Promise<boolean> {
  for (const tab of tabs) {
    controller.activateTab(tab.tabId);
    const marker = (await controller.activeWebsiteWebContents.executeJavaScript(
      'globalThis.__orynTabMarker',
    )) as unknown;
    if (marker !== tab.marker) {
      return false;
    }
  }
  return true;
}

function waitUntilLoaded(
  contents: WebContents,
  expectedURL?: string,
  label = 'smoke page load',
): Promise<void> {
  return waitForSmokePage(contents, { expectedURL, label });
}

function waitForNavigation(contents: WebContents): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => finish(new Error('Timed out waiting for navigation.')),
      20_000,
    );
    const navigated = () => finish();
    const failed = (_event: Electron.Event, code: number, description: string) => {
      if (code !== -3) {
        finish(new Error(`Navigation failed (${code}): ${description}`));
      }
    };
    const finish = (error?: Error) => {
      clearTimeout(timeout);
      contents.removeListener('did-navigate', navigated);
      contents.removeListener('did-fail-load', failed);
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    };
    contents.once('did-navigate', navigated);
    contents.once('did-fail-load', failed);
  });
}

async function navigateAndWait(contents: WebContents, navigate: () => void): Promise<void> {
  const completion = waitForNavigation(contents);
  navigate();
  await completion;
}

async function waitForCondition(
  condition: () => boolean | Promise<boolean>,
  label: string,
  timeoutMs = 20_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!(await condition())) {
    if (Date.now() >= deadline) {
      throw new Error(`Timed out waiting for ${label}.`);
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
}

const pendingLinks: string[] = [];
function deliverOSLink(value: string): void {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return;
  }
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    value.length > 8192
  )
    return;
  const owner =
    [...browserControllers].find((item) => !item.isPrivate && item.window.isFocused()) ??
    [...browserControllers].filter((item) => !item.isPrivate).at(-1);
  if (owner) {
    owner.createTab(url.toString(), true);
    owner.window.show();
    owner.window.focus();
  } else if (app.isReady()) void createBrowser(undefined, url.toString());
  else pendingLinks.push(url.toString());
}
app.on('open-url', (event, url) => {
  event.preventDefault();
  deliverOSLink(url);
});
const pendingFiles: string[] = [];
function deliverOSFile(value: string): void {
  try {
    const local = value.startsWith('file:') ? fileURLToPath(value) : resolve(value);
    if (!/\.(?:html?|xhtml|shtml)$/iu.test(local)) return;
    const url = pathToFileURL(local).toString();
    if (!app.isReady()) {
      pendingFiles.push(url);
      return;
    }
    const owner = [...browserControllers].find(
      (item) => !item.isPrivate && item.window.isFocused(),
    );
    if (owner) {
      owner.createTab(url, true);
      owner.window.show();
      owner.window.focus();
    } else void createBrowser(undefined, url);
  } catch {
    /* Ignore malformed OS file delivery. */
  }
}
app.on('open-file', (event, value) => {
  event.preventDefault();
  deliverOSFile(value);
});
app.on('second-instance', (_event, argv) => {
  for (const value of argv) if (/^https?:\/\//u.test(value)) deliverOSLink(value);
  for (const value of argv)
    if (value.startsWith('/') || value.startsWith('file:')) deliverOSFile(value);
});
const ownsProfileLock = app.requestSingleInstanceLock();
if (!ownsProfileLock) app.quit();

app.whenReady().then(async () => {
  if (!ownsProfileLock) return;
  try {
    const cleanup = privateCleanup.recover();
    privateCleanupReady = cleanup.failed === 0;
    if (cleanup.failed)
      await dialog.showMessageBox({
        type: 'warning',
        message: 'Private temporary-file cleanup is pending',
        detail: 'Some temporary files could not be removed. Oryn will retry on the next startup.',
        buttons: ['OK'],
      });
  } catch {
    await dialog.showMessageBox({
      type: 'warning',
      message: 'Private cleanup needs attention',
      detail:
        'The cleanup journal could not be read safely. Private windows are unavailable until it is repaired.',
      buttons: ['OK'],
    });
  }
  try {
    const cleanup = await cleanupImportSnapshots(
      join(browserData.paths.browserCoreRoot, 'import-temp'),
    );
    if (cleanup.failed)
      await dialog.showMessageBox({
        type: 'warning',
        message: 'Browser-import cleanup is pending',
        detail: 'Some temporary copies could not be removed. Oryn will retry at the next startup.',
        buttons: ['OK'],
      });
  } catch {
    await dialog.showMessageBox({
      type: 'warning',
      message: 'Browser-import cleanup is pending',
      detail: 'Temporary copies could not be cleaned safely. Oryn will retry at the next startup.',
      buttons: ['OK'],
    });
  }
  app.on(
    'certificate-error',
    (_event, contents, url, _error, certificate, callback, isMainFrame) => {
      if (isMainFrame)
        for (const controller of browserControllers)
          controller.recordCertificateError(contents, url, certificate);
      callback(false);
    },
  );
  const examplesRoot = app.isPackaged
    ? join(process.resourcesPath, 'reference-demos')
    : resolve(app.getAppPath(), '../../../oryn-examples');
  try {
    referenceDemoServers = await startReferenceDemoServers(examplesRoot);
    console.info(
      `Oryn reference demos ready: ${referenceDemoServers.map((server) => server.url).join(', ')}`,
    );
  } catch (error) {
    console.warn(
      `Oryn reference demos are unavailable: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  installNewTabProtocol(session.defaultSession, {
    workspaceRendererRoot: join(__dirname, '../renderer'),
    workspaceDevelopmentURL: process.env.ELECTRON_RENDERER_URL,
    hasWorkspaceReference: (reference) => Boolean(browserData.modelWorkspaces.get(reference)),
    getFavorites: () =>
      browserData.bookmarks
        .list()
        .flatMap((item) =>
          item.kind === 'bookmark' && item.favorite && !item.trashedAt
            ? [{ title: item.title, url: item.url, thumbnail: item.thumbnail }]
            : [],
        ),
  });
  installArtifactProtocol();
  const restoreWindows = windowRegistry.list();
  if (!restoreWindows.length) await createBrowser();
  else for (const id of restoreWindows) await createBrowser(id);
  installApplicationMenu();
  for (const url of pendingLinks.splice(0)) deliverOSLink(url);
  for (const value of pendingFiles.splice(0)) deliverOSFile(value);

  powerMonitor.on('resume', () => {
    if (appUpdates.snapshot().automaticChecks) void appUpdates.check(false);
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createBrowser();
    }
  });
});

function installApplicationMenu(): void {
  const template: MenuItemConstructorOptions[] = [
    ...(process.platform === 'darwin'
      ? [
          {
            label: app.name,
            submenu: [
              { role: 'about' as const },
              { type: 'separator' as const },
              {
                label: 'Check for Updates…',
                click: () => openUpdatesAndCheck(),
              },
              { type: 'separator' as const },
              { role: 'services' as const },
              { type: 'separator' as const },
              { role: 'hide' as const },
              { role: 'hideOthers' as const },
              { role: 'unhide' as const },
              { type: 'separator' as const },
              { role: 'quit' as const },
            ],
          },
        ]
      : []),
    {
      label: 'File',
      submenu: [
        {
          label: 'New Window',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            void createBrowser();
          },
        },
        {
          label: 'New Private Window',
          accelerator: 'CmdOrCtrl+Shift+N',
          click: () => {
            void createBrowser(undefined, undefined, true);
          },
        },
        {
          label: 'Open File…',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            void openLocalFile();
          },
        },
        {
          label: 'Save Page As…',
          accelerator: 'CmdOrCtrl+S',
          click: (_item, window) => {
            const owner = [...browserControllers].find((item) => item.window === window);
            void owner?.savePageAs();
          },
        },
        {
          label: 'Print…',
          accelerator: 'CmdOrCtrl+P',
          click: (_item, window) => {
            const owner = [...browserControllers].find((item) => item.window === window);
            void owner?.printPage();
          },
        },
        { role: 'close' },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        ...(process.platform === 'darwin'
          ? [{ type: 'separator' as const }, { role: 'front' as const }]
          : [{ role: 'close' as const }]),
      ],
    },
    ...(process.platform === 'darwin'
      ? []
      : [
          {
            label: 'Help',
            submenu: [{ label: 'Check for Updates…', click: () => openUpdatesAndCheck() }],
          },
        ]),
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
  app.dock?.setMenu(
    Menu.buildFromTemplate([
      {
        label: 'New Window',
        click: () => {
          void createBrowser();
        },
      },
      {
        label: 'New Private Window',
        click: () => {
          void createBrowser(undefined, undefined, true);
        },
      },
    ]),
  );
}

async function openLocalFile(): Promise<void> {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'HTML documents', extensions: ['html', 'htm', 'xhtml', 'shtml'] }],
  });
  if (result.canceled || !result.filePaths[0]) return;
  const owner = [...browserControllers].find((item) => item.window.isFocused());
  if (owner) owner.createTab(pathToFileURL(result.filePaths[0]).toString(), true);
  else deliverOSFile(result.filePaths[0]);
}

function openUpdatesAndCheck(): void {
  const controller =
    [...browserControllers].find((candidate) => candidate.window.isFocused()) ??
    [...browserControllers][0];
  controller?.trustedWebContents.send(IPC.browserOpenPanel, 'settings');
  void appUpdates.check(true);
}

app.on('will-quit', () => {
  artifacts.storage.close();
  appUpdates.stop();
  const servers = referenceDemoServers;
  referenceDemoServers = [];
  void Promise.allSettled(servers.map((server) => server.close()));
});

function installArtifactProtocol(): void {
  const handler = new ArtifactProtocolHandler(artifacts);
  installBrowserRequestGate(session.defaultSession, (details) => {
    const token = ArtifactCapabilityService.tokenFromUrl(details.url);
    const frameId = details.frame?.routingId;
    const allowed = Boolean(
      token &&
      details.webContentsId !== undefined &&
      artifacts.capabilities.authorizeNetworkRequest(token, details.webContentsId, frameId),
    );
    return allowed;
  });
  session.defaultSession.protocol.handle(ARTIFACT_SCHEME, async (request) => {
    const response = await handler.serve({
      url: request.url,
      method: request.method,
      range: request.headers.get('range') ?? undefined,
    });
    return new Response(
      response.body
        ? (Readable.toWeb(response.body) as unknown as ReadableStream<Uint8Array>)
        : null,
      { status: response.status, headers: response.headers },
    );
  });
}

let shutdownStarted = false;
let shutdownComplete = false;
let requestedExitCode = 0;

async function prepareAppUpdateInstall(): Promise<{ accepted: boolean; reason?: string }> {
  if (shutdownStarted) return { accepted: false, reason: 'Oryn is already shutting down.' };
  const activity = jobs.activitySnapshot();
  if (activity.running > 0 || activity.queued > 0) {
    return { accepted: false, reason: 'Local AI work is still running or queued.' };
  }
  if (artifacts.hasPendingWrites()) {
    return { accepted: false, reason: 'An artifact is still being written.' };
  }
  const activeDownload = [...windowData.values()].some((data) => data.hasActiveDownloads());
  if (activeDownload) {
    return { accepted: false, reason: 'A browser download is still active.' };
  }
  for (const controller of browserControllers) {
    if (controller.getState().huggingFacePageAction?.status === 'downloading') {
      return { accepted: false, reason: 'A model download is still active.' };
    }
    if (!(await controller.allowsUpdateRestart())) {
      return {
        accepted: false,
        reason:
          'Close open pages before restarting to update. Page close checks preserve any work you choose to keep.',
      };
    }
  }
  for (const controller of browserControllers) controller.tabs.freezeForRestart();
  shutdownStarted = true;
  try {
    await engine.shutdown();
    browserData.history.flush();
    browserData.downloads.flush();
    for (const controller of browserControllers) controller.allowPreparedClose();
    await finishPreparedWindows(true);
    shutdownComplete = true;
    return { accepted: true };
  } catch (error) {
    console.error('Local AI Engine shutdown failed before update installation.', error);
    throw error;
  }
}

async function recoverAppUpdateInstall(): Promise<void> {
  shutdownStarted = false;
  shutdownComplete = false;
  for (const controller of browserControllers) {
    controller.tabs.cancelRestart();
    controller.cancelPreparedClose();
  }
  await engine.start();
}

app.on('before-quit', (event) => {
  if (shutdownComplete) return;
  event.preventDefault();
  if (shutdownStarted) return;
  shutdownStarted = true;
  void (async () => {
    const activity = jobs.activitySnapshot();
    if (
      activity.running ||
      activity.queued ||
      artifacts.hasPendingWrites() ||
      [...windowData.values()].some((data) => data.hasActiveDownloads()) ||
      [...browserControllers].some(
        (owner) => owner.getState().huggingFacePageAction?.status === 'downloading',
      )
    ) {
      shutdownStarted = false;
      await dialog.showMessageBox({
        message: 'Work is still active',
        detail: 'Finish or cancel Local AI work, exports and downloads before quitting.',
        buttons: ['Keep Oryn open'],
      });
      return;
    }
    for (const controller of browserControllers) {
      if (!(await controller.prepareClose())) {
        shutdownStarted = false;
        for (const owner of browserControllers) owner.cancelPreparedClose();
        return;
      }
    }
    await engine.shutdown();
    await artifacts.storage.shutdown();
    browserData.downloads.flush();
    browserData.history.flush();
    await finishPreparedWindows();
    shutdownComplete = true;
    process.exitCode = requestedExitCode;
    app.quit();
  })().catch((error) => {
    shutdownStarted = false;
    for (const owner of browserControllers) owner.cancelPreparedClose();
    console.error('Oryn could not safely finish quitting.', error);
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
