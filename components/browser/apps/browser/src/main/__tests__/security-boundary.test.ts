import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it, vi } from 'vitest';
import { runInNewContext } from 'node:vm';
import { ModuleKind, transpileModule } from 'typescript';
import {
  permitsEmbeddedPermission,
  type PermissionFrameContext,
} from '../data/permission-frame-policy';

import { ARTIFACT_SCHEME } from '../artifact-manager/artifact-capability-service';
import { LocalAIPermissionStore } from '../model-resolution/local-ai-permission-store';
import { OrynSettingsService } from '../settings/oryn-settings-service';
import { OrynSettingsStore } from '../settings/oryn-settings-store';

const preload = readFileSync(join(import.meta.dirname, '../../preload/website-preload.ts'), 'utf8');
const trustedPreload = readFileSync(
  join(import.meta.dirname, '../../preload/trusted-ui-preload.ts'),
  'utf8',
);
const browserWindow = readFileSync(join(import.meta.dirname, '../browser-window.ts'), 'utf8');
const ipcHandlers = readFileSync(join(import.meta.dirname, '../ipc-handlers.ts'), 'utf8');
const browserDataService = readFileSync(
  join(import.meta.dirname, '../data/browser-data-service.ts'),
  'utf8',
);
const workspacePreload = readFileSync(
  join(import.meta.dirname, '../../preload/model-workspace-preload.ts'),
  'utf8',
);
const workspaceDocument = readFileSync(
  join(import.meta.dirname, '../../renderer/model-workspace.html'),
  'utf8',
);
const internalProtocol = readFileSync(join(import.meta.dirname, '../new-tab-protocol.ts'), 'utf8');
const sidebarWebApps = readFileSync(
  join(import.meta.dirname, '../sidebar-web-app-controller.ts'),
  'utf8',
);

describe('website attack surface', () => {
  it('keeps Electron and Node disabled for all managed website views', () => {
    expect(browserWindow).toContain('nodeIntegration: false');
    expect(browserWindow).toContain('contextIsolation: true');
    expect(browserWindow).toContain('sandbox: true');
    expect(browserWindow).toContain('webSecurity: true');
    expect(preload).toContain("contextBridge.exposeInMainWorld('localAI', localAI)");
    expect(preload).not.toContain("exposeInMainWorld('electron'");
    expect(preload).not.toContain("exposeInMainWorld('ipcRenderer'");
  });

  it('contains only the origin-scoped website channel allowlist', () => {
    const channels = [...preload.matchAll(/['"](local-ai:[a-z-]+)['"]/gu)].map((match) => match[1]);
    expect(new Set(channels)).toEqual(
      new Set([
        'local-ai:get-capabilities',
        'local-ai:model-request',
        'local-ai:model-request-cancel',
        'local-ai:model-session-release',
        'local-ai:model-request-progress',
        'local-ai:artifact-create-begin',
        'local-ai:artifact-create-chunk',
        'local-ai:artifact-create-finish',
        'local-ai:artifact-create-abort',
        'local-ai:artifact-get',
        'local-ai:artifact-get-url',
        'local-ai:artifact-save-as',
        'local-ai:artifact-release',
        'local-ai:job-invoke',
        'local-ai:job-cancel',
        'local-ai:job-event',
      ]),
    );
    // The isolated preload also handles a genuine OS HTML drop. It does not expose
    // that event-only browser channel as a page-callable bridge operation.
    const exposedBridge = preload.slice(
      preload.indexOf('const localAI:'),
      preload.indexOf('contextBridge.exposeInMainWorld'),
    );
    expect(exposedBridge).not.toMatch(/browser:|models-get|settings-|activity-|remove-trusted/gu);
    expect([...preload.matchAll(/['"](browser:[a-z-]+)['"]/gu)].map((match) => match[1])).toEqual([
      'browser:native-file-drop',
    ]);
    expect(preload).not.toMatch(/hugging-face-page-action|model-workspace/gu);
    expect(preload).not.toMatch(/credential|token/gu);
    expect(preload).not.toMatch(/app-update|AppUpdate|checkForAppUpdate/gu);
    expect(trustedPreload).toContain("appUpdateCheck: 'app-update:check'");
    expect(ipcHandlers).toContain('ensureTrustedUI(event)');
  });

  it('uses a separate narrow preload for the internal model workspace', () => {
    expect(workspacePreload).toContain("contextBridge.exposeInMainWorld('orynWorkspace', api)");
    expect(workspacePreload).not.toContain("exposeInMainWorld('localAI'");
    expect(workspacePreload).not.toContain("exposeInMainWorld('browser'");
    expect(workspacePreload).not.toMatch(/permission|models-get|settings-|remove-trusted/gu);
    expect(ipcHandlers).toContain('callerContexts.deriveModelWorkspace(event)');
  });

  it('exposes permission recovery only through the trusted UI', () => {
    expect(trustedPreload).toContain(
      "localAISettingsSiteAskAgain: 'local-ai:settings-site-ask-again'",
    );
    expect(trustedPreload).toContain('askOrynSiteAgain:');
    expect(preload).not.toContain('askOrynSiteAgain');
    expect(workspacePreload).not.toContain('askOrynSiteAgain');
    expect(ipcHandlers).toMatch(
      /handle\(IPC\.localAISettingsSiteAskAgain,[\s\S]*?ensureTrustedUI\(event\);[\s\S]*?orynSettings\.askSiteAgain\(origin\);/u,
    );
  });

  it('validates the origin before resetting a saved Local AI block', () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-permission-recovery-'));
    const permissions = new LocalAIPermissionStore(join(root, 'permissions.json'));
    const key = {
      origin: 'http://127.0.0.1:5189',
      role: 'transcription',
      model: {
        provider: 'huggingface' as const,
        repository: 'org/model',
        revision: 'a'.repeat(40),
      },
    };
    permissions.recordTrustedDecision(key, 'deny');
    const service = new OrynSettingsService(
      new OrynSettingsStore(join(root, 'settings.json')),
      { configureInitialRuntimePolicy: () => undefined } as never,
      {} as never,
      permissions,
      {} as never,
      {} as never,
      {} as never,
      {} as never,
    );

    for (const origin of [`${key.origin}/path`, 'file:///tmp', 'oryn://model-workspace']) {
      expect(() => service.askSiteAgain(origin)).toThrow('INVALID_ORIGIN');
      expect(permissions.decision(key)).toBe('deny');
    }
    service.askSiteAgain(key.origin);
    expect(permissions.decision(key)).toBeUndefined();
  });

  it('allows only the canonical capability scheme for workspace media', () => {
    expect(workspaceDocument).toContain(`img-src ${ARTIFACT_SCHEME}: data: blob:`);
    expect(workspaceDocument).toContain(`media-src ${ARTIFACT_SCHEME}: blob:`);
    expect(internalProtocol).toContain('img-src ${ARTIFACT_SCHEME}: data: blob:');
    expect(internalProtocol).toContain('media-src ${ARTIFACT_SCHEME}: blob:');
    expect(`${workspaceDocument}\n${internalProtocol}`).not.toMatch(
      /(?:img|media)-src artifact:/gu,
    );
  });

  it('composites trusted chrome in an isolated topmost view', () => {
    expect(browserWindow).toContain('new WebContentsView({');
    expect(browserWindow).toContain('transparent: true');
    expect(browserWindow).toContain("this.trustedView.setBackgroundColor('#00000000')");
    expect(browserWindow).toContain('this.window.contentView.addChildView(this.trustedView)');
    expect(browserWindow).toContain("withSurface(trustedUIBaseURL, 'passive-overlay')");
    expect(browserWindow).toContain('height: this.chromeCoversWindow ? height : contentHeight');
    expect(ipcHandlers).toContain('controller.isTrustedUIWebContents(event.sender)');
    expect(browserWindow).toContain('installTrustedUINavigationPolicy');
    expect(browserWindow).toContain("action: 'deny'");
    expect(browserWindow).toContain("contents.on('will-navigate'");
    expect(ipcHandlers).toContain('event.senderFrame !== event.sender.mainFrame');
  });

  it('applies one deny-by-default policy to Chromium permission requests and checks', () => {
    expect(browserDataService).toContain('setPermissionRequestHandler');
    expect(browserDataService).toContain('setPermissionCheckHandler');
    // Both paths use the same browser-derived frame resolver and qualification gate.
    expect(browserDataService.match(/resolvePermissionFrame\(/g)?.length).toBeGreaterThanOrEqual(3);
    expect(browserDataService.match(/permitsEmbeddedPermission\(/g)?.length).toBe(2);
    const frame: PermissionFrameContext = {
      topOrigin: 'https://top.example',
      origin: 'https://embedded.example',
      embedded: true,
      isMainFrame: false,
      ancestorOrigins: ['https://top.example'],
      frameKey: 'frame',
      identity: 'document',
    };
    for (const permission of [
      'camera',
      'microphone',
      'notifications',
      'midi',
      'idle-detection',
      'unknown',
      'local-ai',
    ])
      expect(permitsEmbeddedPermission(frame, permission)).toBe(false);
    for (const permission of ['fullscreen', 'geolocation', 'display-capture'])
      expect(permitsEmbeddedPermission(frame, permission)).toBe(true);
  });

  it('gives every sidebar web app an isolated partition without a privileged preload', () => {
    expect(sidebarWebApps).toContain('persist:oryn-sidebar-${config.appId}');
    expect(sidebarWebApps).toContain('nodeIntegration: false');
    expect(sidebarWebApps).toContain('contextIsolation: true');
    expect(sidebarWebApps).toContain('sandbox: true');
    expect(sidebarWebApps).not.toContain('preload:');
    expect(sidebarWebApps).not.toMatch(/localAI|orynWorkspace|contextBridge/gu);
  });

  it('enforces hidden-background policy before job submission', () => {
    const store = new OrynSettingsStore(
      join(mkdtempSync(join(tmpdir(), 'oryn-security-settings-')), 'settings.json'),
    );
    const engine = { configureInitialRuntimePolicy: () => undefined };
    const service = new OrynSettingsService(
      store,
      engine as never,
      {} as never,
      {} as never,
      {} as never,
      {} as never,
      {} as never,
      {} as never,
    );

    store.updateCompute({ backgroundJobs: 'ask' });
    expect(() => service.assertBackgroundAllowed(false)).toThrow(
      'BACKGROUND_COMPUTE_REQUIRES_ATTENTION',
    );
    expect(() => service.assertBackgroundAllowed(true)).not.toThrow();
    store.updateCompute({ backgroundJobs: 'disable' });
    expect(() => service.assertBackgroundAllowed(false)).toThrow('BACKGROUND_COMPUTE_DISABLED');
  });
  it('admits native file-drop IPC only for an unhandled trusted main-frame OS drop', () => {
    let drop!: (event: {
      isTrusted: boolean;
      defaultPrevented: boolean;
      dataTransfer: { files: { name: string }[] };
      preventDefault: () => void;
    }) => void;
    const invoke = vi.fn();
    const expose = vi.fn();
    const getPathForFile = vi.fn(() => '/tmp/user-selected.html');
    const processDouble = { isMainFrame: true };
    const source = transpileModule(preload, {
      compilerOptions: { module: ModuleKind.CommonJS },
    }).outputText;
    runInNewContext(source, {
      exports: {},
      process: processDouble,
      require: (name: string) => {
        if (name !== 'electron') throw Error('Unexpected preload import');
        return {
          contextBridge: { exposeInMainWorld: expose },
          ipcRenderer: { invoke },
          webUtils: { getPathForFile },
        };
      },
      window: {
        addEventListener: (name: string, listener: typeof drop) => {
          if (name === 'drop') drop = listener;
        },
      },
    });
    const event = {
      isTrusted: false,
      defaultPrevented: false,
      dataTransfer: { files: [{ name: 'user-selected.html' }] },
      preventDefault: vi.fn(),
    };
    drop(event);
    expect(invoke).not.toHaveBeenCalled();
    expect(getPathForFile).not.toHaveBeenCalled();
    event.isTrusted = true;
    event.defaultPrevented = true;
    drop(event);
    expect(invoke).not.toHaveBeenCalled();
    event.defaultPrevented = false;
    processDouble.isMainFrame = false;
    drop(event);
    expect(invoke).not.toHaveBeenCalled();
    processDouble.isMainFrame = true;
    drop(event);
    expect(invoke).toHaveBeenCalledExactlyOnceWith('browser:native-file-drop', [
      '/tmp/user-selected.html',
    ]);
    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(expose).toHaveBeenCalledExactlyOnceWith('localAI', expect.any(Object));
    const bridge = expose.mock.calls[0]![1] as Record<string, unknown>;
    expect(bridge).not.toHaveProperty('nativeFileDrop');
    expect(bridge).not.toHaveProperty('getPathForFile');
  });
});
