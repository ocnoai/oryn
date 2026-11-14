import { JsonFileStore, isRecord } from './json-file-store';
import { isFaviconDataURL } from '../favicon';

export interface RestorableTab {
  url: string;
  tabId?: string;
  title?: string;
  /** Cached image bytes for an open tab; never retained in recently closed history. */
  favicon?: string;
  workspaceId?: string;
  islandId?: string;
  pinned?: boolean;
  closedAt?: string;
}

interface BrowserSessionData {
  version: 1;
  openTabs: RestorableTab[];
  activeIndex: number;
  recentlyClosed: RestorableTab[];
}

export class BrowserSessionStore {
  private readonly store: JsonFileStore<BrowserSessionData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(
      filePath,
      () => ({ version: 1, openTabs: [], activeIndex: 0, recentlyClosed: [] }),
      isBrowserSessionData,
    );
  }

  getRestoreState(): Pick<BrowserSessionData, 'openTabs' | 'activeIndex'> {
    const { openTabs, activeIndex } = this.store.read();
    return { openTabs, activeIndex };
  }

  saveOpenTabs(openTabs: RestorableTab[], activeIndex: number): void {
    const data = this.store.read();
    if (
      JSON.stringify([data.openTabs, data.activeIndex]) === JSON.stringify([openTabs, activeIndex])
    )
      return;
    this.store.write({ ...data, openTabs, activeIndex });
  }

  pushRecentlyClosed(tab: RestorableTab): void {
    const closed = { ...tab };
    delete closed.favicon;
    this.store.update((data) => ({
      ...data,
      recentlyClosed: [...data.recentlyClosed, closed].slice(-25),
    }));
  }

  popRecentlyClosed(): RestorableTab | undefined {
    let restored: RestorableTab | undefined;
    this.store.update((data) => {
      restored = data.recentlyClosed.at(-1);
      return { ...data, recentlyClosed: data.recentlyClosed.slice(0, -1) };
    });
    return restored;
  }

  listRecentlyClosed(): RestorableTab[] {
    return this.store.read().recentlyClosed.reverse();
  }
}

function isBrowserSessionData(value: unknown): value is BrowserSessionData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    Array.isArray(value.openTabs) &&
    value.openTabs.every(isRestorableTab) &&
    Number.isInteger(value.activeIndex) &&
    typeof value.activeIndex === 'number' &&
    Array.isArray(value.recentlyClosed) &&
    value.recentlyClosed.every(isRestorableTab)
  );
}

function isRestorableTab(value: unknown): value is RestorableTab {
  return (
    isRecord(value) &&
    typeof value.url === 'string' &&
    (value.tabId === undefined || typeof value.tabId === 'string') &&
    (value.title === undefined || typeof value.title === 'string') &&
    (value.favicon === undefined || isFaviconDataURL(value.favicon)) &&
    (value.workspaceId === undefined || typeof value.workspaceId === 'string') &&
    (value.islandId === undefined || typeof value.islandId === 'string') &&
    (value.pinned === undefined || typeof value.pinned === 'boolean') &&
    (value.closedAt === undefined || typeof value.closedAt === 'string')
  );
}
