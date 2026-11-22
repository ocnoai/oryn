import type { BrowserBookmarkItem, BrowserHistoryItem, BrowserTabState } from '../../shared/ipc';

export interface NavigationSuggestion {
  url: string;
  title: string;
  source: 'tab' | 'bookmark' | 'history';
}

const sourceOrder = { tab: 0, bookmark: 1, history: 2 };

export function getNavigationSuggestions(
  query: string,
  tabs: readonly Pick<BrowserTabState, 'url' | 'title'>[],
  bookmarks: readonly BrowserBookmarkItem[],
  history: readonly BrowserHistoryItem[],
  limit = 8,
): NavigationSuggestion[] {
  const normalized = query.trim().toLocaleLowerCase().replace(/\s+/gu, ' ');
  if (!normalized || limit <= 0) return [];
  const terms = normalized.split(' ');
  const matches = new Map<
    string,
    { suggestion: NavigationSuggestion; rank: number; order: number }
  >();
  let order = 0;

  const add = (item: { url: string; title: string }, source: NavigationSuggestion['source']) => {
    let address: URL;
    try {
      address = new URL(item.url);
    } catch {
      return;
    }
    if (address.protocol !== 'https:' && address.protocol !== 'http:') return;
    const url = item.url.toLocaleLowerCase();
    const host = address.hostname.toLocaleLowerCase().replace(/^www\./u, '');
    const displayURL = url.replace(/^https?:\/\/(?:www\.)?/u, '');
    const title = item.title.toLocaleLowerCase();
    const searchable = `${url} ${title}`;
    if (!terms.every((term) => searchable.includes(term))) return;

    const rank =
      url === normalized || host === normalized || displayURL === normalized
        ? 0
        : host.startsWith(normalized)
          ? 1
          : url.startsWith(normalized) || displayURL.startsWith(normalized)
            ? 2
            : url.includes(normalized)
              ? 3
              : title.startsWith(normalized)
                ? 4
                : title.includes(normalized)
                  ? 5
                  : 6;
    const candidate = {
      suggestion: { url: item.url, title: item.title || address.hostname, source },
      rank,
      order: order++,
    };
    const previous = matches.get(address.href);
    if (
      !previous ||
      rank < previous.rank ||
      (rank === previous.rank && sourceOrder[source] < sourceOrder[previous.suggestion.source])
    ) {
      matches.set(address.href, candidate);
    }
  };

  for (const tab of tabs) add(tab, 'tab');
  for (const bookmark of bookmarks) {
    if (bookmark.kind === 'bookmark' && !bookmark.trashedAt) add(bookmark, 'bookmark');
  }
  for (const entry of history) add(entry, 'history');

  return [...matches.values()]
    .sort(
      (left, right) =>
        left.rank - right.rank ||
        sourceOrder[left.suggestion.source] - sourceOrder[right.suggestion.source] ||
        left.order - right.order,
    )
    .slice(0, Math.floor(limit))
    .map(({ suggestion }) => suggestion);
}
