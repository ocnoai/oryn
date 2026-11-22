import { createHash } from 'node:crypto';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { DatabaseSync } from 'node:sqlite';
import { BookmarkStore } from '../data/bookmark-store';
import { BrowsingHistoryStore } from '../data/history-store';
import { BrowserImportService } from '../data/browser-import-service';
import { parseChromiumImport } from '../data/browser-import-chromium';
import { cleanupImportSnapshots, createImportSnapshot } from '../data/browser-import-snapshot';
import { emptyImportData, normalizeChromiumBookmarks } from '../data/browser-import-normalize';
import type { BrowserImportWorkerRequest } from '../data/browser-import-types';

const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

function setup(actualParser = false) {
  const homePath = mkdtempSync(join(tmpdir(), 'oryn-browser-import-test-'));
  directories.push(homePath);
  const profile = join(homePath, 'Library', 'Application Support', 'Google', 'Chrome', 'Default');
  mkdirSync(profile, { recursive: true });
  writeFileSync(
    join(profile, '..', 'Local State'),
    JSON.stringify({
      profile: { info_cache: { Default: { name: 'Personal' }, '../outside': { name: 'Unsafe' } } },
    }),
  );
  writeFileSync(
    join(profile, 'Bookmarks'),
    JSON.stringify({
      roots: {
        bookmark_bar: {
          type: 'folder',
          id: '1',
          name: 'Bookmarks',
          children: [{ type: 'url', id: '2', name: 'Oryn', url: 'https://oryn.ocno.ai/' }],
        },
      },
    }),
  );
  const temporaryRoot = join(homePath, 'import-temp');
  mkdirSync(temporaryRoot);
  const bookmarks = new BookmarkStore(join(homePath, 'bookmarks.json'));
  const history = new BrowsingHistoryStore(join(homePath, 'history.json'));
  let now = 100;
  const service = new BrowserImportService({
    bookmarks,
    history,
    homePath,
    temporaryRoot,
    workerPath: '/unused',
    nativeSandboxPath: '/unused',
    now: () => now,
    runWorker: actualParser
      ? parseChromiumImport
      : async (request) => {
          const data = emptyImportData();
          normalizeChromiumBookmarks(
            JSON.parse(readFileSync(join(request.directory, 'Bookmarks'), 'utf8')),
            request,
            data,
          );
          return data;
        },
  });
  return {
    homePath,
    profile,
    temporaryRoot,
    bookmarks,
    history,
    service,
    advance: () => {
      now += 300_001;
    },
  };
}

describe('trusted browser import coordinator', () => {
  it('discovers opaque source IDs and commits only a same-owner, unexpired, single-use preview', async () => {
    const state = setup();
    const sources = await state.service.discover();
    const chrome = sources.find((source) => source.browser === 'chrome')!;
    expect(chrome.profileLabel).toBe('Personal');
    expect(JSON.stringify(sources)).not.toContain(state.homePath);
    const owner = { webContentsId: 1, documentId: 'document-a' };
    const before = createHash('sha256')
      .update(readFileSync(join(state.profile, 'Bookmarks')))
      .digest('hex');
    const preview = await state.service.preview(owner, {
      sourceId: chrome.sourceId,
      dataTypes: ['bookmarks'],
    });
    expect(preview.counts).toEqual({ bookmarks: 1, folders: 2, history: 0 });
    expect(state.bookmarks.list()).toEqual([]);
    expect(readdirSync(state.temporaryRoot)).toEqual([]);
    expect(() => state.service.commit({ webContentsId: 2 }, preview.token)).toThrow(
      'IMPORT_PREVIEW_EXPIRED',
    );
    expect(() =>
      state.service.commit({ webContentsId: 1, documentId: 'document-b' }, preview.token),
    ).toThrow('IMPORT_PREVIEW_EXPIRED');
    expect(state.service.commit(owner, preview.token)).toMatchObject({ bookmarks: 1, folders: 2 });
    expect(() => state.service.commit(owner, preview.token)).toThrow('IMPORT_PREVIEW_EXPIRED');
    expect(
      createHash('sha256')
        .update(readFileSync(join(state.profile, 'Bookmarks')))
        .digest('hex'),
    ).toBe(before);
    const repeat = await state.service.preview(owner, {
      sourceId: chrome.sourceId,
      dataTypes: ['bookmarks'],
    });
    expect(repeat.counts).toEqual({ bookmarks: 0, folders: 0, history: 0 });
    state.service.commit(owner, repeat.token);
    expect(state.bookmarks.list()).toHaveLength(3);
    state.service.dispose();
  });

  it('expiry, cancellation and owner destruction leave stores unchanged', async () => {
    const state = setup();
    const source = (await state.service.discover()).find((source) => source.browser === 'chrome')!;
    const owner = { webContentsId: 3 };
    const request = { sourceId: source.sourceId, dataTypes: ['bookmarks'] as const };
    const first = await state.service.preview(owner, {
      ...request,
      dataTypes: [...request.dataTypes],
    });
    state.advance();
    expect(() => state.service.commit(owner, first.token)).toThrow('IMPORT_PREVIEW_EXPIRED');
    const next = await state.service.preview(owner, {
      ...request,
      dataTypes: [...request.dataTypes],
    });
    state.service.disposeOwner(owner);
    expect(() => state.service.commit(owner, next.token)).toThrow('IMPORT_PREVIEW_EXPIRED');
    expect(state.bookmarks.list()).toEqual([]);
    expect(state.history.search()).toEqual([]);
    state.service.dispose();
  });

  it('round-trips UTF-8 bookmark folders and titles through a real copied-source parse and committed store', async () => {
    const state = setup(true);
    const folder = 'Práce 🧪 — 資料';
    const title = 'Příliš žluťoučký kůň 🦊 — أهلاً';
    writeFileSync(
      join(state.profile, 'Bookmarks'),
      JSON.stringify({
        roots: {
          bookmark_bar: {
            id: '1',
            type: 'folder',
            name: folder,
            children: [
              { id: '2', type: 'url', name: title, url: 'https://unicode.test/' },
              { id: '3', type: 'url', name: 'Browser settings', url: 'chrome://settings/' },
            ],
          },
        },
      }),
    );
    const original = readFileSync(join(state.profile, 'Bookmarks'));
    const source = (await state.service.discover()).find((entry) => entry.browser === 'chrome')!;
    const owner = { webContentsId: 10, documentId: 'utf8-document' };
    try {
      const preview = await state.service.preview(owner, {
        sourceId: source.sourceId,
        dataTypes: ['bookmarks'],
      });
      expect(preview.counts).toEqual({ bookmarks: 1, folders: 2, history: 0 });
      expect(preview.skipped).toContainEqual({ reason: 'unsafe-bookmark-url', count: 1 });
      expect(state.bookmarks.list()).toEqual([]);
      expect(readdirSync(state.temporaryRoot)).toEqual([]);
      expect(state.service.commit(owner, preview.token)).toMatchObject({
        bookmarks: 1,
        folders: 2,
      });
      const reopened = new BookmarkStore(join(state.homePath, 'bookmarks.json')).list();
      expect(reopened).toEqual(state.bookmarks.list());
      expect(reopened).toContainEqual(expect.objectContaining({ kind: 'folder', title: folder }));
      expect(reopened).toContainEqual(
        expect.objectContaining({ kind: 'bookmark', title, url: 'https://unicode.test/' }),
      );
      const stored = readFileSync(join(state.homePath, 'bookmarks.json'), 'utf8');
      expect(stored).toContain(folder);
      expect(stored).toContain(title);
      expect(stored).not.toContain('chrome:');
      expect(readFileSync(join(state.profile, 'Bookmarks'))).toEqual(original);
    } finally {
      state.service.dispose();
    }
  });

  it.each(['malformed bookmarks JSON', 'corrupt SQLite history'] as const)(
    'fails closed on %s without changing existing stores or original bytes',
    async (corruption) => {
      const state = setup(true);
      state.bookmarks.createBookmark('Existing bookmark', 'https://existing.test/');
      state.history.recordVisit(
        'https://existing.test/history',
        'Existing history',
        new Date('2026-11-01T00:00:00.000Z'),
      );
      state.history.flush();
      const dataTypes: ('bookmarks' | 'history')[] = ['bookmarks'];
      const originals = [join(state.profile, 'Bookmarks')];
      if (corruption === 'malformed bookmarks JSON')
        writeFileSync(originals[0]!, '{"roots":{"bookmark_bar":');
      else {
        const history = join(state.profile, 'History');
        writeFileSync(
          history,
          Buffer.from('This is a corrupt SQLite source, not a database.\0', 'utf8'),
        );
        originals.push(history);
        dataTypes.push('history');
      }
      const sourceHashes = () =>
        originals.map((path) => createHash('sha256').update(readFileSync(path)).digest('hex'));
      const beforeHashes = sourceHashes();
      const beforeBookmarks = state.bookmarks.list();
      const beforeHistory = state.history.search();
      const persistentFiles = ['bookmarks.json', 'history.json'].map((name) =>
        join(state.homePath, name),
      );
      const beforeFiles = persistentFiles.map((path) => readFileSync(path));
      const source = (await state.service.discover()).find((entry) => entry.browser === 'chrome')!;
      try {
        await expect(
          state.service.preview({ webContentsId: 11 }, { sourceId: source.sourceId, dataTypes }),
        ).rejects.toThrow(
          corruption === 'malformed bookmarks JSON'
            ? 'IMPORT_JSON_INVALID'
            : /not a database|malformed/iu,
        );
        expect(state.bookmarks.list()).toEqual(beforeBookmarks);
        expect(state.history.search()).toEqual(beforeHistory);
        expect(persistentFiles.map((path) => readFileSync(path))).toEqual(beforeFiles);
        expect(sourceHashes()).toEqual(beforeHashes);
        expect(readdirSync(state.temporaryRoot)).toEqual([]);
        // A failed preparation must release the active slot rather than strand future imports.
        await expect(
          state.service.preview({ webContentsId: 11 }, { sourceId: source.sourceId, dataTypes }),
        ).rejects.toThrow(
          corruption === 'malformed bookmarks JSON'
            ? 'IMPORT_JSON_INVALID'
            : /not a database|malformed/iu,
        );
        expect(persistentFiles.map((path) => readFileSync(path))).toEqual(beforeFiles);
        expect(sourceHashes()).toEqual(beforeHashes);
        expect(readdirSync(state.temporaryRoot)).toEqual([]);
      } finally {
        state.service.dispose();
      }
    },
  );
});

describe('read-only selected source snapshots', () => {
  it('copies live SQLite WAL data and parses it without changing any source file', async () => {
    const state = setup();
    const database = new DatabaseSync(join(state.profile, 'History'));
    database.exec(
      'PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT, title TEXT); CREATE TABLE visits (id INTEGER PRIMARY KEY, url INTEGER, visit_time INTEGER)',
    );
    database
      .prepare('INSERT INTO urls VALUES (?, ?, ?)')
      .run(1, 'https://history.test/', 'History');
    database.prepare('INSERT INTO visits VALUES (?, ?, ?)').run(1, 1, 13_331_212_800_000_000n);
    const files = ['History', 'History-wal', 'History-shm'];
    const hashes = () =>
      files.map((name) =>
        createHash('sha256')
          .update(readFileSync(join(state.profile, name)))
          .digest('hex'),
      );
    const before = hashes();
    const snapshot = await createImportSnapshot({
      profileDirectory: state.profile,
      temporaryRoot: state.temporaryRoot,
      dataTypes: ['history'],
    });
    try {
      const request: BrowserImportWorkerRequest = {
        version: 2,
        kind: 'chromium',
        directory: snapshot.directory,
        sourceKey: 'a'.repeat(64),
        sourceLabel: 'Google Chrome',
        dataTypes: ['history'],
        bookmarkCapacity: 20_000,
        existingBookmarkIds: [],
      };
      const parsed = await parseChromiumImport(request);
      expect(parsed.history).toHaveLength(1);
      expect(parsed.history[0]?.url).toBe('https://history.test/');
      expect(hashes()).toEqual(before);
    } finally {
      await snapshot.dispose();
      database.close();
      state.service.dispose();
    }
  });

  it('rejects symlinked source members and cleans only its dedicated stale snapshot names', async () => {
    const state = setup();
    rmSync(join(state.profile, 'Bookmarks'));
    const outside = join(state.homePath, 'outside.json');
    writeFileSync(outside, 'protected');
    symlinkSync(outside, join(state.profile, 'Bookmarks'));
    await expect(
      createImportSnapshot({
        profileDirectory: state.profile,
        temporaryRoot: state.temporaryRoot,
        dataTypes: ['bookmarks'],
      }),
    ).rejects.toThrow('IMPORT_SOURCE_FILE_UNSAFE');
    mkdirSync(join(state.temporaryRoot, 'oryn-browser-import-ABC123'));
    writeFileSync(join(state.temporaryRoot, 'oryn-browser-import-ABC123', 'History'), 'temporary');
    symlinkSync(outside, join(state.temporaryRoot, 'oryn-browser-import-DEF456'));
    writeFileSync(join(state.temporaryRoot, 'unrelated'), 'preserved');
    expect(await cleanupImportSnapshots(state.temporaryRoot)).toEqual({ removed: 2, failed: 0 });
    expect(readFileSync(outside, 'utf8')).toBe('protected');
    expect(readdirSync(state.temporaryRoot)).toEqual(['unrelated']);
    state.service.dispose();
  });
});
