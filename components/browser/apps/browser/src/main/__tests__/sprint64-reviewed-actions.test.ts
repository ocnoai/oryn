import { EventEmitter } from 'node:events';
import { X509Certificate } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Session, WebContents } from 'electron';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeSession, electronDouble, installFixture, page } from './security-audit-ipc-fixture';
import { IPC, type SiteContext } from '../../shared/ipc';
import { BrowserDataService } from '../data/browser-data-service';
import { parseCertificateChain, type ConnectionSecuritySnapshot } from '../connection-security';
import { installIpcHandlers } from '../ipc-handlers';

const directories: string[] = [];
const cleanups: (() => void)[] = [];
const nativeDialog = Object.assign(electronDouble.dialog, {
  showSaveDialog: vi.fn(),
  showMessageBoxSync: vi.fn(),
});
Object.assign(electronDouble, {
  systemPreferences: { getMediaAccessStatus: () => 'granted' },
});
const pem = readFileSync(
  new URL('./fixtures/connection-security-cert.pem', import.meta.url),
  'utf8',
);

beforeEach(() => {
  nativeDialog.showMessageBox.mockReset();
  nativeDialog.showSaveDialog.mockReset();
  nativeDialog.showMessageBoxSync.mockReset();
});
afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) cleanup();
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
});

function setup() {
  const root = mkdtempSync(join(tmpdir(), 'oryn-reviewed-actions-'));
  directories.push(root);
  const data = new BrowserDataService(root);
  const session = new FakeSession();
  data.installChromiumSessionHandlers(session as unknown as Session);
  const pages = [page(10, session), page(11, session)].map((contents) => {
    let destroyed = false;
    return Object.assign(contents, {
      getTitle: () => 'Reviewed fixture',
      isDestroyed: () => destroyed,
      close: vi.fn(() => {
        destroyed = true;
        contents.emit('destroyed');
      }),
    });
  });
  const fixture = installFixture(data, new EventEmitter(), undefined, pages);
  cleanups.push(fixture.cleanup);
  return { ...fixture, data, session, pages, root };
}

function reviewedSite(): SiteContext {
  return {
    tabId: 'tab_10',
    documentId: 'document_10',
    origin: 'https://example.com',
    partition: 'default',
  };
}

function certificateFixture(fixture: ReturnType<typeof setup>) {
  const snapshot: ConnectionSecuritySnapshot = {
    state: 'secure',
    url: fixture.pages[0]!.getURL(),
    certificates: parseCertificateChain([new X509Certificate(pem).raw.toString('base64')]),
    chainIncludesRoot: false,
    issues: [],
  };
  Object.assign(fixture.controller, { connectionSnapshot: () => snapshot });
  return {
    snapshot,
    expected: {
      tabId: 'tab_10',
      documentId: 'document_10',
      url: snapshot.url,
      fingerprint256: snapshot.certificates[0]!.fingerprint256,
    },
  };
}

async function grantCamera(fixture: ReturnType<typeof setup>) {
  nativeDialog.showMessageBox.mockResolvedValue({ response: 2, checkboxChecked: false });
  const granted = await new Promise<boolean>((resolve) =>
    fixture.session.request(fixture.pages[0] as unknown as WebContents, 'media', resolve, {
      requestingUrl: fixture.pages[0]!.getURL(),
      isMainFrame: true,
      mediaTypes: ['video'],
    }),
  );
  expect(granted).toBe(true);
  return fixture.data.captureSnapshot()[0]!;
}

describe('embedded permission controls bind to the displayed site context', () => {
  it.each(['tab', 'document', 'authority'] as const)(
    'rejects a stale %s before changing the permission store',
    (change) => {
      const fixture = setup();
      fixture.data.sitePermissions.setEmbedded(
        'https://example.com',
        'https://embedded.test',
        'camera',
        'allow',
        ['https://example.com'],
      );
      const before = readFileSync(fixture.data.paths.sitePermissions, 'utf8');
      if (change === 'tab') fixture.select(fixture.pages[1]!);
      if (change === 'document') fixture.callers.getDocumentId = () => 'document_replacement';
      if (change === 'authority') fixture.callers.hasDocumentAuthority = () => false;
      expect(() =>
        fixture.invoke(
          IPC.browserEmbeddedPermissionRevoke,
          'https://embedded.test',
          'camera',
          reviewedSite(),
        ),
      ).toThrow('SITE_CONTEXT_CHANGED');
      expect(readFileSync(fixture.data.paths.sitePermissions, 'utf8')).toBe(before);
    },
  );

  it('revokes the reviewed pair and capability without clearing another partition or permission', () => {
    const fixture = setup();
    for (const [permission, partition] of [
      ['camera', 'default'],
      ['microphone', 'default'],
      ['camera', 'persist:sidebar'],
    ] as const)
      fixture.data.sitePermissions.setEmbedded(
        'https://example.com',
        'https://embedded.test',
        permission,
        'allow',
        ['https://example.com'],
        partition,
      );
    fixture.invoke(
      IPC.browserEmbeddedPermissionRevoke,
      'https://embedded.test',
      'camera',
      reviewedSite(),
    );
    expect(
      fixture.data.sitePermissions
        .listEmbedded('https://example.com')
        .map((record) => record.permission),
    ).toEqual(['microphone']);
    expect(
      fixture.data.sitePermissions.listEmbedded('https://example.com', 'persist:sidebar'),
    ).toHaveLength(1);
  });
});

describe('certificate export uses the reviewed document and certificate', () => {
  it.each(['tabId', 'documentId', 'url', 'fingerprint256'] as const)(
    'rejects stale %s before showing a Save dialog',
    async (field) => {
      const fixture = setup();
      const { expected } = certificateFixture(fixture);
      const path = join(fixture.root, 'never-saved.pem');
      nativeDialog.showSaveDialog.mockResolvedValue({ canceled: false, filePath: path });
      await expect(
        fixture.invoke(IPC.browserCertificateExport, { ...expected, [field]: 'stale' }),
      ).rejects.toThrow('CERTIFICATE_DOCUMENT_CHANGED');
      expect(nativeDialog.showSaveDialog).not.toHaveBeenCalled();
      expect(existsSync(path)).toBe(false);
    },
  );

  it('rejects replacement while Save is open and never writes the selected file', async () => {
    const fixture = setup();
    const { expected } = certificateFixture(fixture);
    const path = join(fixture.root, 'never-saved.pem');
    let finish!: (value: { canceled: boolean; filePath: string }) => void;
    nativeDialog.showSaveDialog.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const exportResult = fixture.invoke(IPC.browserCertificateExport, expected);
    fixture.callers.getDocumentId = () => 'document_replacement';
    finish({ canceled: false, filePath: path });
    await expect(exportResult).rejects.toThrow('CERTIFICATE_DOCUMENT_CHANGED');
    expect(existsSync(path)).toBe(false);
  });

  it('writes only the reviewed chain after explicit Save selection', async () => {
    const fixture = setup();
    const { snapshot, expected } = certificateFixture(fixture);
    const path = join(fixture.root, 'reviewed-chain.pem');
    nativeDialog.showSaveDialog.mockResolvedValue({ canceled: false, filePath: path });
    await fixture.invoke(IPC.browserCertificateExport, expected);
    expect(readFileSync(path, 'utf8').trim()).toBe(snapshot.certificates[0]!.pem.trim());
  });
});

describe('capture Stop cannot act on replacement capture authority', () => {
  it('rejects an old generation before confirmation or grant mutation', async () => {
    const fixture = setup();
    const old = await grantCamera(fixture);
    fixture.pages[0]!.emit('did-navigate');
    const replacement = await grantCamera(fixture);
    expect(replacement.generation).not.toBe(old.generation);
    nativeDialog.showMessageBox.mockClear();
    const store = readFileSync(fixture.data.paths.sitePermissions, 'utf8');
    await expect(
      fixture.invoke(IPC.browserCaptureStop, old.webContentsId, old.generation),
    ).rejects.toThrow('CAPTURE_TARGET_CHANGED');
    expect(nativeDialog.showMessageBox).not.toHaveBeenCalled();
    expect(fixture.pages[0]!.close).not.toHaveBeenCalled();
    expect(fixture.data.captureSnapshot()).toEqual([replacement]);
    expect(readFileSync(fixture.data.paths.sitePermissions, 'utf8')).toBe(store);
  });

  it('rechecks generation after async confirmation before closing or revoking the replacement', async () => {
    const fixture = setup();
    const old = await grantCamera(fixture);
    let finish!: (value: { response: number; checkboxChecked: boolean }) => void;
    nativeDialog.showMessageBox.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const stop = fixture.invoke(IPC.browserCaptureStop, old.webContentsId, old.generation);
    fixture.pages[0]!.emit('did-start-navigation');
    fixture.pages[0]!.emit('did-navigate');
    const replacement = await grantCamera(fixture);
    const store = readFileSync(fixture.data.paths.sitePermissions, 'utf8');
    finish({ response: 1, checkboxChecked: false });
    await expect(stop).rejects.toThrow('CAPTURE_TARGET_CHANGED');
    expect(fixture.pages[0]!.close).not.toHaveBeenCalled();
    expect(fixture.data.captureSnapshot()).toEqual([replacement]);
    expect(readFileSync(fixture.data.paths.sitePermissions, 'utf8')).toBe(store);
  });

  it('closes the confirmed live page and clears its remembered capture grant', async () => {
    const fixture = setup();
    const capture = await grantCamera(fixture);
    nativeDialog.showMessageBox.mockResolvedValue({ response: 1, checkboxChecked: false });
    await expect(
      fixture.invoke(IPC.browserCaptureStop, capture.webContentsId, capture.generation),
    ).resolves.toBe(true);
    expect(fixture.pages[0]!.close).toHaveBeenCalledWith({ waitForBeforeUnload: true });
    expect(fixture.data.captureSnapshot()).toEqual([]);
    expect(fixture.data.sitePermissions.get('https://example.com', 'camera')).toBeUndefined();
  });
});

describe('private organization commands cannot mutate ordinary windows', () => {
  it('rejects all cross-window workspace operations before closing, recovering or changing normal stores', async () => {
    const fixture = setup();
    const normalData = new BrowserDataService(join(fixture.root, 'normal-profile'));
    const workspace = normalData.experience.createWorkspace('Keep normal workspace');
    const store = readFileSync(normalData.paths.browserExperience, 'utf8');
    Object.assign(fixture.controller, { isPrivate: true });
    const normalTabs = {
      getTabs: vi.fn(() => [{ tabId: 'normal_tab', workspaceId: workspace.workspaceId }]),
      closeTabs: vi.fn(async () => true),
      recoverOrganization: vi.fn(),
      moveIslandToWorkspace: vi.fn(),
    };
    const normal = { isTrustedUIWebContents: () => false, tabs: normalTabs };
    // Reinstall the same registered-handler fixture with an actual second owner;
    // a regression must not reach that owner's tabs or physical profile stores.
    fixture.cleanup();
    const transfer = vi.fn(async () => undefined);
    const remove = installIpcHandlers(
      fixture.controller as never,
      {} as never,
      fixture.callers as never,
      fixture.data,
      {} as never,
      {} as never,
      new EventEmitter() as never,
      {} as never,
      new EventEmitter() as never,
      {} as never,
      {} as never,
      new EventEmitter() as never,
      {} as never,
      {
        all: () => [fixture.controller, normal] as never,
        data: (owner) => (owner === (normal as unknown) ? normalData : fixture.data),
        create: async () => undefined,
        transfer,
      },
    );
    cleanups.push(remove);
    for (const command of [
      { type: 'workspace-delete', workspaceId: workspace.workspaceId, tabs: 'close' },
      { type: 'tab-move-workspace', tabId: 'normal_tab', workspaceId: workspace.workspaceId },
      {
        type: 'island-move-workspace',
        islandId: 'normal_island',
        workspaceId: workspace.workspaceId,
      },
      { type: 'tab-new-window', tabId: 'normal_tab' },
    ])
      await expect(fixture.invoke(IPC.browserExperienceCommand, command)).rejects.toThrow(
        'PRIVATE_FEATURE_UNAVAILABLE',
      );
    expect(normalTabs.closeTabs).not.toHaveBeenCalled();
    expect(normalTabs.recoverOrganization).not.toHaveBeenCalled();
    expect(normalTabs.moveIslandToWorkspace).not.toHaveBeenCalled();
    expect(transfer).not.toHaveBeenCalled();
    expect(readFileSync(normalData.paths.browserExperience, 'utf8')).toBe(store);
  });
});
