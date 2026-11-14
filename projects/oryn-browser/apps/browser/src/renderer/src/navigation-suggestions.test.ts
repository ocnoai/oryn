import { describe, expect, it } from 'vitest';

import type { BrowserBookmarkItem, BrowserHistoryItem } from '../../shared/ipc';
import { getNavigationSuggestions } from './navigation-suggestions';

function bookmark(url: string, title: string): BrowserBookmarkItem {
  return {
    kind: 'bookmark',
    bookmarkId: url,
    parentId: null,
    title,
    url,
    createdAt: '2026-01-01T00:00:00Z',
    order: 0,
    favorite: false,
  };
}

function visit(url: string, title: string): BrowserHistoryItem {
  return { historyId: url, url, title, visitedAt: '2026-01-01T00:00:00Z' };
}

describe('local navigation suggestions', () => {
  it('matches titles and addresses case-insensitively, including words across both', () => {
    const history = [visit('https://example.com/reference', 'TypeScript Handbook')];
    expect(getNavigationSuggestions('  TYPESCRIPT  ', [], [], history)[0]?.title).toBe(
      'TypeScript Handbook',
    );
    expect(getNavigationSuggestions('EXAMPLE.COM', [], [], history)).toHaveLength(1);
    expect(getNavigationSuggestions('handbook   example', [], [], history)).toHaveLength(1);
    expect(getNavigationSuggestions('handbook missing', [], [], history)).toEqual([]);
  });

  it('ranks an address prefix ahead of a title match and uses source priority for ties', () => {
    const result = getNavigationSuggestions(
      'oryn',
      [{ url: 'https://tab.example/', title: 'About Oryn' }],
      [bookmark('https://bookmark.example/', 'About Oryn')],
      [visit('https://history.example/', 'About Oryn'), visit('https://oryn.example/', 'Browser')],
    );
    expect(result.map((item) => item.url)).toEqual([
      'https://oryn.example/',
      'https://tab.example/',
      'https://bookmark.example/',
      'https://history.example/',
    ]);
    expect(result.map((item) => item.source)).toEqual(['history', 'tab', 'bookmark', 'history']);
  });

  it('deduplicates addresses across sources, retaining the best matching source', () => {
    const result = getNavigationSuggestions(
      'example',
      [{ url: 'https://example.com', title: 'Open page' }],
      [bookmark('https://example.com/', 'Saved page')],
      [visit('https://example.com/', 'Recent page'), visit('https://example.com/', 'Earlier page')],
    );
    expect(result).toEqual([{ url: 'https://example.com', title: 'Open page', source: 'tab' }]);
    expect(
      getNavigationSuggestions(
        'saved',
        [{ url: 'https://example.com/', title: 'Open page' }],
        [bookmark('https://example.com/', 'Saved page')],
        [],
      )[0]?.source,
    ).toBe('bookmark');
  });

  it('excludes folders, trashed bookmarks, malformed addresses and internal pages', () => {
    const saved = bookmark('https://trashed.example/', 'Example');
    const folder: BrowserBookmarkItem = {
      kind: 'folder',
      bookmarkId: 'folder',
      parentId: null,
      title: 'Example',
      createdAt: '2026-01-01T00:00:00Z',
      order: 0,
      favorite: false,
    };
    const result = getNavigationSuggestions(
      'example',
      [
        { url: '', title: 'Example' },
        { url: 'oryn://newtab', title: 'Example' },
        { url: 'file:///example', title: 'Example' },
        { url: 'javascript:example()', title: 'Example' },
      ],
      [folder, { ...saved, trashedAt: '2026-02-01T00:00:00Z' }],
      [visit('invalid example', 'Example'), visit('https://safe.example/', 'Example')],
    );
    expect(result.map((item) => item.url)).toEqual(['https://safe.example/']);
  });

  it('searches older history before limiting results and limits after deduplication', () => {
    const history = [
      ...Array.from({ length: 150 }, (_, index) =>
        visit(`https://recent.example/${index}`, `Recent page ${index}`),
      ),
      visit('https://old.example/', 'Rare Handbook'),
    ];
    expect(getNavigationSuggestions('rare', [], [], history)[0]?.url).toBe('https://old.example/');
    expect(getNavigationSuggestions('recent', [], [], history)).toHaveLength(8);
    expect(getNavigationSuggestions('recent', [], [], history, 2)).toHaveLength(2);
    expect(
      getNavigationSuggestions('example', [], [], [history[0]!, history[0]!, history[1]!], 2),
    ).toHaveLength(2);
  });

  it('does not offer suggestions for an empty query or a zero limit', () => {
    const history = [visit('https://example.com/', 'Example')];
    expect(getNavigationSuggestions('   ', [], [], history)).toEqual([]);
    expect(getNavigationSuggestions('example', [], [], history, 0)).toEqual([]);
  });
});
