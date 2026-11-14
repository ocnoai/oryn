import { describe, expect, it } from 'vitest';
import { BookmarkStore } from '../data/bookmark-store';
import { BrowsingHistoryStore } from '../data/history-store';
import {
  emptyImportData,
  normalizeChromiumBookmarks,
  normalizeSafariBookmarks,
} from '../data/browser-import-normalize';
import type { BrowserImportWorkerRequest } from '../data/browser-import-types';

function request(
  capacity = 20_000,
  existingBookmarkIds: string[] = [],
): BrowserImportWorkerRequest {
  return {
    version: 2,
    kind: 'chromium',
    directory: '/unused',
    sourceKey: 'a'.repeat(64),
    sourceLabel: 'Google Chrome',
    dataTypes: ['bookmarks'],
    bookmarkCapacity: capacity,
    existingBookmarkIds,
  };
}

describe('browser import graph normalization', () => {
  it('counts the imported wrapper against the destination cap and reports every dropped item', () => {
    const data = emptyImportData();
    normalizeChromiumBookmarks(
      {
        roots: {
          bookmark_bar: {
            type: 'folder',
            id: '1',
            name: 'Bar',
            children: Array.from({ length: 30_000 }, (_, index) => ({
              type: 'url',
              id: String(index + 2),
              name: `Page ${index}`,
              url: `https://example.test/${index}`,
            })),
          },
        },
      },
      request(),
      data,
    );
    expect(data.bookmarks).toHaveLength(20_000);
    expect(data.bookmarks.filter((item) => item.kind === 'bookmark')).toHaveLength(19_998);
    expect(data.skipped).toContainEqual({ reason: 'bookmark-capacity', count: 10_002 });
    const store = new BookmarkStore(`memory:normalize-${crypto.randomUUID()}`);
    store.importNodes(data.bookmarks);
    expect(store.list()).toHaveLength(20_000);
  });

  it('flattens deeper folders without creating a ninth-level bookmark', () => {
    let node: unknown = { type: 'url', id: 'leaf', name: 'Leaf', url: 'https://example.test/deep' };
    for (let level = 12; level >= 1; level--)
      node = { type: 'folder', id: `folder${level}`, name: `Level ${level}`, children: [node] };
    const data = emptyImportData();
    normalizeChromiumBookmarks({ roots: { bookmark_bar: node } }, request(), data);
    const store = new BookmarkStore(`memory:depth-${crypto.randomUUID()}`);
    expect(() => store.importNodes(data.bookmarks)).not.toThrow();
    expect(store.list().some((item) => item.kind === 'bookmark')).toBe(true);
    expect(data.skipped).toContainEqual({ reason: 'flattened-folder', count: 6 });
  });

  it('skips unsafe URLs, bounds titles and imports added items into an existing wrapper once', () => {
    const source = (children: unknown[]) => ({
      roots: { other: { type: 'folder', id: 'root', name: 'Other', children } },
    });
    const first = emptyImportData();
    normalizeChromiumBookmarks(
      source([{ type: 'url', id: 'a', name: 'A', url: 'https://example.test/a' }]),
      request(),
      first,
    );
    const store = new BookmarkStore(`memory:repeat-${crypto.randomUUID()}`);
    store.importNodes(first.bookmarks);
    const next = emptyImportData();
    normalizeChromiumBookmarks(
      source([
        { type: 'url', id: 'a', name: 'A', url: 'https://example.test/a' },
        { type: 'url', id: 'b', name: 'x'.repeat(900), url: 'https://example.test/b' },
        { type: 'url', id: 'unsafe', name: 'Script', url: 'javascript:alert(1)' },
        { type: 'url', id: 'internal', name: 'Settings', url: 'chrome://settings/' },
        {
          type: 'url',
          id: 'extension',
          name: 'Extension',
          url: 'chrome-extension://fixture/options.html',
        },
        {
          type: 'url',
          id: 'credential',
          name: 'Credential',
          url: 'https://secret:password@example.test/',
        },
      ]),
      request(
        20_000 - store.list().length,
        store.list().map((item) => item.bookmarkId),
      ),
      next,
    );
    expect(next.bookmarks).toHaveLength(1);
    expect(next.bookmarks[0]?.title).toHaveLength(512);
    expect(() => store.importNodes(next.bookmarks)).not.toThrow();
    expect(store.list()).toHaveLength(4);
    expect(next.skipped).toContainEqual({ reason: 'unsafe-bookmark-url', count: 4 });
    const previous = store.list();
    expect(() =>
      store.importNodes([{ ...next.bookmarks[0]!, parentId: next.bookmarks[0]!.bookmarkId }]),
    ).toThrow();
    expect(store.list()).toEqual(previous);
  });

  it('keeps Safari active HTML inert and skips unsafe links without rejecting safe siblings', () => {
    const data = emptyImportData();
    normalizeSafariBookmarks(
      '<DL><H3>Research &amp; notes</H3><DL><A HREF="https://example.test/">Safe</A><A HREF="javascript:bad()">Unsafe</A><script><A HREF="https://hidden.test/">Hidden</A></script></DL></DL>',
      request(),
      data,
    );
    expect(data.bookmarks.filter((item) => item.kind === 'bookmark')).toHaveLength(1);
    expect(data.bookmarks.some((item) => item.title === 'Hidden')).toBe(false);
    expect(data.skipped).toContainEqual({ reason: 'unsafe-bookmark-url', count: 1 });
  });
});

describe('imported history merge', () => {
  it('retains the newest combined 10,000 visits, deduplicates reimports and rejects invalid batches atomically', () => {
    const history = new BrowsingHistoryStore(`memory:history-${crypto.randomUUID()}`);
    history.recordVisit(
      'https://normal.test/newest',
      'Normal visit',
      new Date('2026-11-04T00:00:00Z'),
    );
    history.flush();
    const entries = Array.from({ length: 10_000 }, (_, index) => ({
      url: `https://old.test/${index}`,
      title: String(index),
      visitedAt: new Date(Date.UTC(2025, 0, 1) + index * 1000).toISOString(),
    }));
    expect(history.mergeImported(entries)).toEqual({ added: 9999, duplicates: 0, dropped: 1 });
    expect(history.search()).toHaveLength(10_000);
    expect(history.search()[0]?.url).toBe('https://normal.test/newest');
    expect(history.mergeImported(entries)).toEqual({ added: 0, duplicates: 9999, dropped: 1 });
    const before = history.search();
    expect(() =>
      history.mergeImported([
        { url: 'javascript:bad()', title: 'Bad', visitedAt: new Date().toISOString() },
      ]),
    ).toThrow('HISTORY_IMPORT_INVALID');
    expect(history.search()).toEqual(before);
  });
});
