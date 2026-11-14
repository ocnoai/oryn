import type { Session, WebContents } from 'electron';
import { IPC } from '../../shared/ipc';
import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { FakeSession, page, installFixture, electronDouble } from './security-audit-ipc-fixture';
import { BrowserDataService } from '../data/browser-data-service';

it('trusted permission changes revoke all matching page grants without crossing partitions', async () => {
  const root = mkdtempSync(join(tmpdir(), 'oryn-permission-validation-'));
  const service = new BrowserDataService(root),
    session = new FakeSession();
  service.installChromiumSessionHandlers(session as unknown as Session);
  const a = page(10, session),
    b = page(11, session);
  const ipc = installFixture(service, new EventEmitter(), undefined, [a, b]);
  const details = { requestingUrl: 'https://example.com/page', isMainFrame: true };
  const check = (p: ReturnType<typeof page>) =>
    session.check(p as unknown as WebContents, 'notifications', 'https://example.com', details);
  const ask = (p: ReturnType<typeof page>) =>
    new Promise<boolean>((resolve) =>
      session.request(p as unknown as WebContents, 'notifications', resolve, details),
    );
  try {
    electronDouble.dialog.showMessageBox.mockResolvedValue({ response: 1, checkboxChecked: false });
    expect(await ask(a)).toBe(true);
    expect(check(a)).toBe(true);
    expect(check(b)).toBe(false);
    ipc.select(b);
    await ipc.invoke(IPC.browserPermissionSet, 'notifications', 'deny', {
      tabId: 'tab_11',
      documentId: 'document_11',
      origin: 'https://example.com',
      partition: 'default',
    });
    expect(service.sitePermissions.get('https://example.com', 'notifications', 'default')).toBe(
      'deny',
    );
    expect(check(a)).toBe(false);
    expect(await ask(a)).toBe(false);
    expect(check(b)).toBe(false);
    expect(await ask(b)).toBe(false);
    const sidebar = new FakeSession();
    service.installChromiumSessionHandlers(
      sidebar as unknown as Session,
      'persist:validation-sidebar',
    );
    expect(
      sidebar.check(
        page(12, sidebar) as unknown as WebContents,
        'notifications',
        'https://example.com',
        details,
      ),
    ).toBe(false);
    expect(
      session.check(a as unknown as WebContents, 'notifications', 'https://foreign.test', details),
    ).toBe(false);
    await ipc.invoke(IPC.browserPermissionSet, 'notifications', 'ask', {
      tabId: 'tab_11',
      documentId: 'document_11',
      origin: 'https://example.com',
      partition: 'default',
    });
    expect(check(a)).toBe(false);
    a.emit('did-navigate');
    expect(check(a)).toBe(false);
    service.sitePermissions.clear('https://example.com', 'default', 'notifications');
    expect(await ask(a)).toBe(true);
    await service.clearSiteData(session as unknown as Session, 'https://example.com');
    expect(check(a)).toBe(false);
    console.log(
      JSON.stringify({
        id: 'NV-11',
        scope:
          'actual BrowserDataService and registered trusted IPC handler; fake Electron permission transport',
        denyPersisted: true,
        retainedPageGrant: false,
        navigationRevoked: true,
        siteClearRevoked: true,
      }),
    );
  } finally {
    ipc.cleanup();
    rmSync(root, { recursive: true, force: true });
  }
});
