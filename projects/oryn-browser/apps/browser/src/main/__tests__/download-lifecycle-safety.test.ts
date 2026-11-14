import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { DownloadItem, Session } from 'electron';
import { BrowserDataService } from '../data/browser-data-service';
import {
  DownloadSafetyError,
  type DownloadSafety,
  type DownloadFileIdentity,
} from '../download-safety';

vi.mock('electron', () => ({ app: {}, dialog: {}, shell: {}, systemPreferences: {} }));
const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
const identity: DownloadFileIdentity = {
  device: '1',
  inode: '2',
  size: '8',
  modified: '3',
  changed: '4',
};
class FakeSession extends EventEmitter {
  setDisplayMediaRequestHandler = vi.fn();
  setPermissionRequestHandler = vi.fn();
  setPermissionCheckHandler = vi.fn();
  setDownloadPath = vi.fn();
}
function setup(privateMode = false) {
  const root = mkdtempSync(join(tmpdir(), 'oryn-download-lifecycle-'));
  roots.push(root);
  const safety = {
    check: vi.fn<DownloadSafety['check']>(async () => identity),
    open: vi.fn<DownloadSafety['open']>(async () => true),
  };
  const service = new BrowserDataService(
    privateMode ? `memory:private-${root}` : root,
    safety as unknown as DownloadSafety,
  );
  const session = new FakeSession();
  service.installChromiumSessionHandlers(session as unknown as Session);
  return { root, service, session, safety };
}
function complete(value: ReturnType<typeof setup>, redirected = false) {
  const path = join(value.root, 'explicit-save.txt');
  const item = Object.assign(new EventEmitter(), {
    getURL: () => `https://fixture.invalid/${redirected ? 'redirected' : 'direct'}`,
    getFilename: () => 'explicit-save.txt',
    getSavePath: () => path,
    getTotalBytes: () => 8,
    getReceivedBytes: () => 8,
  });
  value.session.emit('will-download', {}, item as unknown as DownloadItem);
  item.emit('done', {}, 'completed');
  return value.service.downloads.list()[0]!.downloadId;
}
describe('download lifecycle quarantine gate', () => {
  it.each([false, true])(
    'checks direct and redirected completions in %s-private stores',
    async (privateMode) => {
      for (const redirected of [false, true]) {
        const value = setup(privateMode);
        const id = complete(value, redirected);
        expect(value.service.downloads.list()[0]?.openState).toBe('pending');
        await vi.waitFor(() => expect(value.service.downloads.list()[0]?.openState).toBe('ready'));
        expect(value.safety.check).toHaveBeenCalledOnce();
        await value.service.controlDownload(id, 'open');
        expect(value.safety.open).toHaveBeenCalledOnce();
      }
    },
  );
  it('never opens pending downloads; failed checks are visible and explicitly retryable', async () => {
    const value = setup();
    value.safety.check.mockRejectedValue(new DownloadSafetyError('quarantine'));
    const id = complete(value);
    await expect(value.service.controlDownload(id, 'open')).rejects.toThrow(
      'Check the downloaded file',
    );
    await vi.waitFor(() =>
      expect(value.service.downloads.list()[0]).toMatchObject({
        openState: 'error',
        openError: 'quarantine',
      }),
    );
    expect(value.safety.open).not.toHaveBeenCalled();
    value.safety.check.mockResolvedValue(identity);
    await value.service.controlDownload(id, 'check');
    expect(value.service.downloads.list()[0]).toMatchObject({
      openState: 'ready',
      fileIdentity: identity,
    });
  });
  it('revalidates completed persisted records after restart rather than trusting saved ready', async () => {
    const value = setup();
    complete(value);
    await vi.waitFor(() => expect(value.service.downloads.list()[0]?.openState).toBe('ready'));
    const check = vi.fn(async () => {
      throw new DownloadSafetyError('file-changed');
    });
    const restarted = new BrowserDataService(value.root, { check } as unknown as DownloadSafety);
    expect(restarted.downloads.list()[0]?.openState).toBe('pending');
    await vi.waitFor(() =>
      expect(restarted.downloads.list()[0]).toMatchObject({
        openState: 'error',
        openError: 'file-changed',
      }),
    );
    expect(check).toHaveBeenCalledOnce();
  });
  it('contains OS/helper errors in the stable record state and never publishes their raw message', async () => {
    const value = setup();
    const id = complete(value);
    await vi.waitFor(() => expect(value.service.downloads.list()[0]?.openState).toBe('ready'));
    value.safety.open.mockRejectedValue(new Error('secret native OS path'));
    await expect(value.service.controlDownload(id, 'open')).rejects.toThrow('could not be opened');
    expect(value.service.downloads.list()[0]).toMatchObject({
      openState: 'error',
      openError: 'open-failed',
    });
    expect(JSON.stringify(value.service.downloads.list())).not.toContain('secret');
  });
  it('deduplicates simultaneous Open requests and invalidates review when the record is removed', async () => {
    const value = setup();
    const id = complete(value);
    await vi.waitFor(() => expect(value.service.downloads.list()[0]?.openState).toBe('ready'));
    let finish!: (accepted: boolean) => void;
    value.safety.open.mockImplementation(async (_record, current: () => boolean) => {
      await new Promise<boolean>((resolve) => {
        finish = resolve;
      });
      if (!current()) throw new DownloadSafetyError('file-changed');
      return true;
    });
    const first = value.service.controlDownload(id, 'open');
    await expect(value.service.controlDownload(id, 'open')).rejects.toThrow();
    await value.service.controlDownload(id, 'remove');
    finish(true);
    await expect(first).rejects.toThrow('could not be opened');
    expect(value.safety.open).toHaveBeenCalledOnce();
  });
});
