import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BookmarkStore } from '../data/bookmark-store';
import { DownloadStore } from '../data/download-store';
import { SitePermissionStore } from '../data/site-permission-store';
import { JsonFileStore } from '../data/json-file-store';

const directories: string[] = [];
function path() {
  const dir = mkdtempSync(join(tmpdir(), 'oryn-s60-data-'));
  directories.push(dir);
  return join(dir, 'data.json');
}
afterEach(() => {
  vi.useRealTimers();
  for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe('Sprint 60 persisted data regressions', () => {
  it('B17 copies descendants once with independent IDs and remapped parents', () => {
    const file = path();
    const store = new BookmarkStore(file);
    const folder = store.createFolder('Parent');
    const nested = store.createFolder('Nested', folder.bookmarkId);
    const child = store.createBookmark('Example', 'https://example.com', nested.bookmarkId);
    store.move([folder.bookmarkId, child.bookmarkId], null, true);
    const nodes = new BookmarkStore(file).list();
    expect(nodes).toHaveLength(6);
    const copiedRoot = nodes.find(
      (node) => node.title === 'Parent' && node.bookmarkId !== folder.bookmarkId,
    )!;
    const copiedNested = nodes.find((node) => node.parentId === copiedRoot.bookmarkId)!;
    expect(nodes.find((node) => node.parentId === copiedNested.bookmarkId)?.title).toBe('Example');
    store.deletePermanently([folder.bookmarkId]);
    expect(store.list()).toHaveLength(3);
  });
  it('B18 restores required ancestors so a child is reachable', () => {
    const store = new BookmarkStore(path());
    const folder = store.createFolder('Folder');
    const child = store.createBookmark('Example', 'https://example.com', folder.bookmarkId);
    store.trash([folder.bookmarkId]);
    store.restore([child.bookmarkId]);
    expect(store.list().every((node) => !node.trashedAt)).toBe(true);
  });
  it('rejects cycles atomically and rejects active bookmark schemes', () => {
    const store = new BookmarkStore(path());
    const folder = store.createFolder('Folder');
    const child = store.createFolder('Child', folder.bookmarkId);
    const before = store.list();
    expect(() => store.move([folder.bookmarkId], child.bookmarkId)).toThrow('BOOKMARK_CYCLE');
    expect(store.list()).toEqual(before);
    expect(() => store.createBookmark('Script', 'javascript:alert(1)')).toThrow(
      'BOOKMARK_URL_UNSAFE',
    );
  });
  it('quarantines a persisted cyclic graph instead of looping during a later operation', () => {
    const file = path();
    const store = new BookmarkStore(file);
    const folder = store.createFolder('Folder');
    const persisted = JSON.parse(readFileSync(file, 'utf8'));
    persisted.nodes[0].parentId = folder.bookmarkId;
    writeFileSync(file, JSON.stringify(persisted));
    expect(new BookmarkStore(file).list()).toEqual([]);
    expect(readdirSync(join(file, '..')).some((name) => name.includes('.corrupt-'))).toBe(true);
  });
  it('B04 recovers stale transfers without claiming resumability', () => {
    const file = path();
    const old = new DownloadStore(file);
    const record = old.start({ url: 'https://example.com/file', filename: 'file', totalBytes: 10 });
    old.update(record.downloadId, { state: 'paused' });
    old.flush();
    const restored = new DownloadStore(file);
    expect(restored.list()[0]?.state).toBe('interrupted');
    expect(JSON.parse(readFileSync(file, 'utf8')).records[0].state).toBe('interrupted');
  });
  it('B20 coalesces progress and publishes terminal state immediately', () => {
    vi.useFakeTimers();
    const store = new DownloadStore(path());
    const record = store.start({
      url: 'https://example.com/file',
      filename: 'file',
      totalBytes: 100,
    });
    const changed = vi.fn();
    store.on('changed', changed);
    for (let i = 1; i <= 50; i++) store.update(record.downloadId, { receivedBytes: i });
    expect(changed).not.toHaveBeenCalled();
    vi.advanceTimersByTime(250);
    expect(changed).toHaveBeenCalledTimes(1);
    store.update(record.downloadId, { state: 'completed', receivedBytes: 100 });
    expect(changed).toHaveBeenCalledTimes(2);
  });
  it('isolates grants and clearing across ordinary and sidebar partitions', () => {
    const file = path();
    const store = new SitePermissionStore(file);
    store.set('https://example.com', 'media', 'allow');
    expect(store.get('https://example.com', 'media', 'persist:sidebar')).toBeUndefined();
    store.set('https://example.com', 'media', 'deny', 'persist:sidebar');
    store.clear('https://example.com');
    expect(
      new SitePermissionStore(file).get('https://example.com', 'media', 'persist:sidebar'),
    ).toBe('deny');
  });
  it('migrates origin-only grants into ordinary browsing only', () => {
    const file = path();
    writeFileSync(
      file,
      JSON.stringify({
        version: 1,
        records: [
          {
            origin: 'https://example.com',
            permission: 'media',
            decision: 'allow',
            updatedAt: new Date().toISOString(),
          },
        ],
      }),
    );
    const store = new SitePermissionStore(file);
    expect(store.get('https://example.com', 'media')).toBe('allow');
    expect(store.get('https://example.com', 'media', 'sidebar')).toBeUndefined();
  });
  it('B13 quarantines corruption, preserves exact bytes and recovers without authority', () => {
    const file = path();
    writeFileSync(file, '{broken');
    expect(new SitePermissionStore(file).get('https://example.com', 'media')).toBeUndefined();
    const quarantined = readdirSync(join(file, '..')).find((name) => name.includes('.corrupt-'))!;
    expect(readFileSync(join(file, '..', quarantined), 'utf8')).toBe('{broken');
  });
  it('does not mistake an I/O failure for corrupt JSON or discard it', () => {
    const dir = path();
    const store = new JsonFileStore(
      join(dir, '..'),
      () => ({ safe: true }),
      (value): value is { safe: boolean } => typeof value === 'object',
    );
    expect(() => store.read()).toThrow();
  });
});
