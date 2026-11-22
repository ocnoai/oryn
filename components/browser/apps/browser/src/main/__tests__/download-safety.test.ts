import { mkdtemp, writeFile, rename, symlink, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DownloadSafety, isRiskyDownload } from '../download-safety';
import type { DownloadRecord } from '../data/download-store';

vi.mock('electron', () => ({ app: {}, dialog: {}, shell: {} }));
const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});
async function fixture(name = 'harmless.txt', contents = 'harmless text') {
  const root = await mkdtemp(join(tmpdir(), 'oryn-download-safety-'));
  roots.push(root);
  const path = join(root, name);
  await writeFile(path, contents);
  const record: DownloadRecord = {
    downloadId: 'download_fixture',
    url: 'https://fixture.invalid/download',
    filename: name,
    savePath: path,
    receivedBytes: contents.length,
    totalBytes: contents.length,
    state: 'completed',
    startedAt: '2026-01-01T00:00:00.000Z',
    updatedAt: '2026-01-01T00:00:00.000Z',
  };
  const native = {
    ensureQuarantine: vi.fn(() => 'preserved' as const),
    hasQuarantine: vi.fn(() => true),
    fileReferenceURL: vi.fn(() => 'file:///.file/id=1.2'),
  };
  const confirm = vi.fn(async () => true);
  const openFile = vi.fn(async () => '');
  const safety = new DownloadSafety({ native: () => native, confirm, openFile });
  record.fileIdentity = await safety.check(record);
  return { root, path, record, native, confirm, openFile, safety };
}

describe('Browser download safety', () => {
  it('checks a held file, preserves native quarantine and hands off only an internal file-ID URL', async () => {
    const value = await fixture();
    expect(await value.safety.open(value.record, () => true)).toBe(true);
    expect(value.native.ensureQuarantine).toHaveBeenCalledTimes(2);
    expect(value.confirm).not.toHaveBeenCalled();
    expect(value.openFile).toHaveBeenCalledExactlyOnceWith('file:///.file/id=1.2');
  });
  it('requires explicit confirmation for script/installer names, executable bits and disguised native/script bytes', () => {
    for (const name of [
      'sample.DMG',
      'sample.pkg',
      'sample.app',
      'sample.command',
      'sample.sh',
      'sample.terminal',
    ])
      expect(isRiskyDownload(name, Buffer.from('text'), 0o644n)).toBe(true);
    expect(isRiskyDownload('innocent.txt', Buffer.from('#!/bin/sh\n'), 0o644n)).toBe(true);
    expect(isRiskyDownload('innocent.txt', Buffer.from('cffaedfe', 'hex'), 0o644n)).toBe(true);
    expect(isRiskyDownload('innocent.txt', Buffer.from('text'), 0o755n)).toBe(true);
    expect(isRiskyDownload('innocent.txt', Buffer.from('text'), 0o644n)).toBe(false);
  });
  it('Cancel never hands off a risky file', async () => {
    const value = await fixture('sample.command');
    value.confirm.mockResolvedValue(false);
    expect(await value.safety.open(value.record, () => true)).toBe(false);
    expect(value.native.fileReferenceURL).not.toHaveBeenCalled();
    expect(value.openFile).not.toHaveBeenCalled();
  });
  it.each(['replace', 'rewrite'] as const)(
    'refuses a %s during the native confirmation',
    async (change) => {
      const value = await fixture('sample.command');
      value.confirm.mockImplementation(async () => {
        if (change === 'replace') await rename(value.path, join(value.root, 'held.command'));
        await writeFile(value.path, 'changed script bytes');
        return true;
      });
      await expect(value.safety.open(value.record, () => true)).rejects.toMatchObject({
        reason: 'file-changed',
      });
      expect(value.openFile).not.toHaveBeenCalled();
    },
  );
  it('a changed file after completion requires a fresh explicit check', async () => {
    const value = await fixture();
    await writeFile(value.path, 'replacement bytes');
    await expect(value.safety.check(value.record)).rejects.toMatchObject({
      reason: 'file-changed',
    });
    await expect(
      value.safety.check({ ...value.record, fileIdentity: undefined }),
    ).resolves.toBeDefined();
  });
  it('stale/removed download records cannot launch after confirmation', async () => {
    const value = await fixture('sample.command');
    await expect(value.safety.open(value.record, () => false)).rejects.toMatchObject({
      reason: 'file-changed',
    });
    expect(value.openFile).not.toHaveBeenCalled();
  });
  it('missing quarantine after confirmation blocks opening', async () => {
    const value = await fixture('sample.command');
    value.native.hasQuarantine.mockReturnValue(false);
    await expect(value.safety.open(value.record, () => true)).rejects.toMatchObject({
      reason: 'quarantine',
    });
    expect(value.openFile).not.toHaveBeenCalled();
  });
  it('denies symlink substitution and never sends its target to the native helper', async () => {
    const value = await fixture();
    await rename(value.path, join(value.root, 'target.txt'));
    await symlink(join(value.root, 'target.txt'), value.path);
    value.native.ensureQuarantine.mockClear();
    await expect(value.safety.open(value.record, () => true)).rejects.toMatchObject({
      reason: 'file-unavailable',
    });
    expect(value.native.ensureQuarantine).not.toHaveBeenCalled();
    expect(value.openFile).not.toHaveBeenCalled();
  });
  it('does not accept a mutable path or arbitrary URL as the native file-reference result', async () => {
    const value = await fixture();
    value.native.fileReferenceURL.mockReturnValue('file:///tmp/changed.command');
    await expect(value.safety.open(value.record, () => true)).rejects.toMatchObject({
      reason: 'file-changed',
    });
    expect(value.openFile).not.toHaveBeenCalled();
  });
  it('returns only stable errors for missing files, tagging failures and OS open errors', async () => {
    const value = await fixture();
    value.native.ensureQuarantine.mockImplementation(() => {
      throw new Error('secret native path');
    });
    await expect(value.safety.check(value.record)).rejects.toMatchObject({
      reason: 'quarantine',
      message: 'DOWNLOAD_QUARANTINE',
    });
    value.native.ensureQuarantine.mockReturnValue('preserved');
    value.openFile.mockRejectedValue(new Error('secret OS error/path'));
    await expect(value.safety.open(value.record, () => true)).rejects.toMatchObject({
      reason: 'open-failed',
      message: 'DOWNLOAD_OPEN_FAILED',
    });
    await rm(value.path);
    await expect(value.safety.check(value.record)).rejects.toMatchObject({
      reason: 'file-unavailable',
      message: 'DOWNLOAD_FILE_UNAVAILABLE',
    });
  });
});
