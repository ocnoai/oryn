import { createHash, randomUUID } from 'node:crypto';

import { JsonFileStore, isRecord } from './json-file-store';

export interface BrowsingHistoryEntry {
  historyId: string;
  url: string;
  title: string;
  visitedAt: string;
}

export interface ImportedHistoryEntry {
  url: string;
  title: string;
  visitedAt: string;
}

interface BrowsingHistoryData {
  version: 1;
  entries: BrowsingHistoryEntry[];
}

export class BrowsingHistoryStore {
  private readonly store: JsonFileStore<BrowsingHistoryData>;
  private data: BrowsingHistoryData;
  private pendingWrite?: ReturnType<typeof setTimeout>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(
      filePath,
      () => ({ version: 1, entries: [] }),
      isBrowsingHistoryData,
    );
    this.data = this.store.read();
  }

  recordVisit(
    url: string,
    title: string,
    visitedAt = new Date(),
  ): BrowsingHistoryEntry | undefined {
    if (!isHistoryURL(url)) {
      return undefined;
    }
    const entry: BrowsingHistoryEntry = {
      historyId: `history_${randomUUID().replaceAll('-', '')}`,
      url,
      title: title || new URL(url).hostname,
      visitedAt: visitedAt.toISOString(),
    };
    this.data.entries = [...this.data.entries, entry].slice(-10_000);
    this.schedule();
    return entry;
  }

  updateTitle(url: string, title: string): void {
    if (!title || !isHistoryURL(url)) return;
    const entry = [...this.data.entries].reverse().find((item) => item.url === url);
    if (!entry || entry.title === title) return;
    entry.title = title;
    this.schedule();
  }

  search(query = ''): BrowsingHistoryEntry[] {
    const normalized = query.trim().toLocaleLowerCase();
    return structuredClone(
      this.data.entries
        .filter(
          (entry) =>
            !normalized ||
            entry.url.toLocaleLowerCase().includes(normalized) ||
            entry.title.toLocaleLowerCase().includes(normalized),
        )
        .sort((left, right) => right.visitedAt.localeCompare(left.visitedAt)),
    );
  }

  /** Merge visits in timestamp order; importing old history must not evict newer normal visits. */
  mergeImported(entries: readonly ImportedHistoryEntry[]): {
    added: number;
    duplicates: number;
    dropped: number;
  } {
    if (!Array.isArray(entries) || entries.length > 10_000) throw new Error('HISTORY_IMPORT_LIMIT');
    const imported = entries.map((entry) => {
      if (
        !entry ||
        typeof entry.url !== 'string' ||
        entry.url.length > 8192 ||
        !['http:', 'https:'].includes(new URL(entry.url).protocol) ||
        typeof entry.title !== 'string' ||
        entry.title.length > 512 ||
        typeof entry.visitedAt !== 'string' ||
        !Number.isFinite(Date.parse(entry.visitedAt))
      )
        throw new Error('HISTORY_IMPORT_INVALID');
      const parsed = new URL(entry.url);
      if (parsed.username || parsed.password) throw new Error('HISTORY_IMPORT_INVALID');
      const url = parsed.toString();
      const visitedAt = new Date(entry.visitedAt).toISOString();
      const key = `${url}\0${visitedAt}`;
      return {
        url,
        title: entry.title || parsed.hostname,
        visitedAt,
        historyId: `history_${createHash('sha256').update(key).digest('hex')}`,
      };
    });
    const keyFor = (entry: ImportedHistoryEntry) =>
      `${new URL(entry.url).toString()}\0${new Date(entry.visitedAt).toISOString()}`;
    const seen = new Set(this.data.entries.map(keyFor));
    const additions: BrowsingHistoryEntry[] = [];
    let duplicates = 0;
    for (const entry of imported) {
      const key = keyFor(entry);
      if (seen.has(key)) {
        duplicates++;
        continue;
      }
      seen.add(key);
      additions.push(entry);
    }
    const combined = [...this.data.entries, ...additions].sort(
      (left, right) =>
        Date.parse(left.visitedAt) - Date.parse(right.visitedAt) ||
        left.historyId.localeCompare(right.historyId),
    );
    const retained = combined.slice(-10_000);
    const retainedIds = new Set(retained.map((entry) => entry.historyId));
    const added = additions.filter((entry) => retainedIds.has(entry.historyId)).length;
    // Persist before publishing the changed in-memory state. A failed write leaves it untouched.
    const next = { ...this.data, entries: retained };
    this.store.write(next);
    this.data = next;
    clearTimeout(this.pendingWrite);
    this.pendingWrite = undefined;
    return { added, duplicates, dropped: additions.length - added };
  }

  delete(historyId: string): boolean {
    const count = this.data.entries.length;
    this.data.entries = this.data.entries.filter((entry) => entry.historyId !== historyId);
    this.flush();
    return this.data.entries.length !== count;
  }

  clear(since = 0): void {
    this.data.entries = this.data.entries.filter((entry) => Date.parse(entry.visitedAt) < since);
    this.flush();
  }

  flush(): void {
    clearTimeout(this.pendingWrite);
    this.pendingWrite = undefined;
    this.store.write(this.data);
  }

  private schedule(): void {
    if (!this.pendingWrite) this.pendingWrite = setTimeout(() => this.flush(), 250);
  }
}

function isHistoryURL(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === 'https:' ||
      url.protocol === 'http:' ||
      (url.protocol === 'file:' &&
        (url.hostname === '' || url.hostname === 'localhost') &&
        !url.username &&
        !url.password)
    );
  } catch {
    return false;
  }
}

function isBrowsingHistoryData(value: unknown): value is BrowsingHistoryData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    Array.isArray(value.entries) &&
    value.entries.every(
      (entry) =>
        isRecord(entry) &&
        typeof entry.historyId === 'string' &&
        typeof entry.url === 'string' &&
        typeof entry.title === 'string' &&
        typeof entry.visitedAt === 'string',
    )
  );
}
