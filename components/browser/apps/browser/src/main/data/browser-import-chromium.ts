import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { isRecord } from './json-file-store';
import {
  emptyImportData,
  importHistoryEntry,
  normalizeChromiumBookmarks,
  skipImport,
} from './browser-import-normalize';
import {
  BROWSER_IMPORT_LIMITS,
  type BrowserImportData,
  type BrowserImportWorkerRequest,
} from './browser-import-types';

const CHROMIUM_EPOCH_MICROSECONDS = 11_644_473_600_000_000n;

export function chromiumVisitTime(value: unknown): string | undefined {
  try {
    if (typeof value !== 'bigint' && (typeof value !== 'number' || !Number.isSafeInteger(value)))
      return undefined;
    const milliseconds = Number((BigInt(value) - CHROMIUM_EPOCH_MICROSECONDS) / 1000n);
    if (milliseconds < 0 || milliseconds > 8.64e15) return undefined;
    return new Date(milliseconds).toISOString();
  } catch {
    return undefined;
  }
}

async function readBoundedJSON(path: string, maximum: number): Promise<unknown> {
  const data = await readFile(path);
  if (data.length > maximum) throw new Error('IMPORT_FILE_TOO_LARGE');
  try {
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(data));
  } catch {
    throw new Error('IMPORT_JSON_INVALID');
  }
}

export async function parseChromiumImport(
  request: BrowserImportWorkerRequest,
): Promise<BrowserImportData> {
  const data = emptyImportData();
  if (request.dataTypes.includes('bookmarks')) {
    normalizeChromiumBookmarks(
      await readBoundedJSON(
        join(request.directory, 'Bookmarks'),
        BROWSER_IMPORT_LIMITS.bookmarksBytes,
      ),
      request,
      data,
    );
  }
  if (request.dataTypes.includes('history')) {
    const database = new DatabaseSync(join(request.directory, 'History'), {
      readOnly: true,
      allowExtension: false,
      enableDoubleQuotedStringLiterals: false,
      timeout: 0,
    });
    try {
      database.exec(
        'PRAGMA query_only=ON; PRAGMA trusted_schema=OFF; PRAGMA temp_store=MEMORY; PRAGMA cache_size=-4096',
      );
      const tables = database
        .prepare("SELECT name, type FROM sqlite_schema WHERE name IN ('urls', 'visits')")
        .all();
      if (tables.length !== 2 || tables.some((row) => row.type !== 'table'))
        throw new Error('IMPORT_HISTORY_SCHEMA_INVALID');
      const check = database.prepare('PRAGMA quick_check(1)').get();
      if (!check || Object.values(check)[0] !== 'ok') throw new Error('IMPORT_HISTORY_CORRUPT');
      const query = database.prepare(
        'SELECT u.url, u.title, v.visit_time FROM visits AS v JOIN urls AS u ON u.id = v.url ORDER BY v.visit_time DESC, v.id DESC LIMIT ?',
      );
      query.setReadBigInts(true);
      const seen = new Set<string>();
      let examined = 0;
      for (const row of query.iterate(BROWSER_IMPORT_LIMITS.sourceNodes)) {
        examined++;
        const entry = importHistoryEntry({
          url: row.url,
          title: row.title,
          visitedAt: chromiumVisitTime(row.visit_time),
        });
        if (!entry) {
          skipImport(data, 'invalid-history-entry');
          continue;
        }
        const key = `${entry.url}\0${entry.visitedAt}`;
        if (seen.has(key)) {
          skipImport(data, 'duplicate-history-entry');
          continue;
        }
        seen.add(key);
        if (data.history.length < BROWSER_IMPORT_LIMITS.historyEntries) data.history.push(entry);
        else skipImport(data, 'history-capacity');
      }
      const countQuery = database.prepare('SELECT COUNT(*) AS count FROM visits');
      countQuery.setReadBigInts(true);
      const count = countQuery.get()?.count;
      if (typeof count === 'bigint' && count > BigInt(examined))
        skipImport(data, 'history-source-limit', Number(count - BigInt(examined)));
    } finally {
      database.close();
    }
  }
  if (request.dataTypes.includes('searchEngine')) {
    const preferences = await readBoundedJSON(
      join(request.directory, 'Preferences'),
      BROWSER_IMPORT_LIMITS.preferencesBytes,
    );
    const outer =
      isRecord(preferences) && isRecord(preferences.default_search_provider_data)
        ? preferences.default_search_provider_data
        : undefined;
    const template =
      outer && isRecord(outer.template_url_data) ? outer.template_url_data : undefined;
    const search = template?.url;
    if (typeof search === 'string' && search.length < 8192) {
      try {
        const url = new URL(search.replaceAll('{searchTerms}', 'oryn'));
        if (url.protocol !== 'https:' || url.username || url.password) throw new Error();
        if (['www.google.com', 'google.com'].includes(url.hostname) && url.pathname === '/search')
          data.searchProvider = 'google';
        else if (['www.bing.com', 'bing.com'].includes(url.hostname) && url.pathname === '/search')
          data.searchProvider = 'bing';
        else if (
          ['www.duckduckgo.com', 'duckduckgo.com'].includes(url.hostname) &&
          url.pathname === '/'
        )
          data.searchProvider = 'duckduckgo';
      } catch {
        /* An unknown search template never becomes imported executable configuration. */
      }
    }
    if (!data.searchProvider) skipImport(data, 'unsupported-search-provider');
  }
  return data;
}
