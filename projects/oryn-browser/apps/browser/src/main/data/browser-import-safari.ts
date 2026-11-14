import { constants } from 'node:fs';
import { lstat, open, type FileHandle } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { fromRandomAccessReaderPromise, RandomAccessReader, type Entry, type ZipFile } from 'yauzl';
import { isSafariArchiveIdentity, sameSafariArchive } from './browser-import-safari-source';
import { isRecord } from './json-file-store';
import {
  emptyImportData,
  importHistoryEntry,
  normalizeSafariBookmarks,
  skipImport,
} from './browser-import-normalize';
import {
  BROWSER_IMPORT_LIMITS,
  type BrowserImportData,
  type BrowserImportWorkerRequest,
} from './browser-import-types';

// Closed filename recognition. Do not probe arbitrary JSON members to discover their data_type:
// that would open payment-card/extension payloads. Unknown locales require a reviewed alias first.
const BOOKMARK_NAMES = new Set([
  'Bookmarks',
  'Signets',
  'Lesezeichen',
  'Segnalibri',
  'Marcadores',
  'Favoritos',
  'Bladwijzers',
  'Záložky',
]);
const HISTORY_NAMES = [
  'History',
  'Historique',
  'Verlauf',
  'Cronologia',
  'Historial',
  'Histórico',
  'Geschiedenis',
  'História',
  'Historie',
];

// ZIP integrity check; independent of Node's newer zlib.crc32 so source tests still run on Node 22.
const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, value) => {
  let current = value;
  for (let bit = 0; bit < 8; bit++)
    current = current & 1 ? 0xedb88320 ^ (current >>> 1) : current >>> 1;
  return current >>> 0;
});
export function importCRC32(buffer: Buffer, previous = 0): number {
  let value = previous ^ 0xffffffff;
  for (const byte of buffer) value = CRC_TABLE[(value ^ byte) & 0xff]! ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
}

export function safariMemberType(name: string): 'bookmarks' | 'history' | undefined {
  if (
    name.length > 4096 ||
    name.includes('\\') ||
    name.startsWith('/') ||
    name.split('/').some((part) => part === '..' || part === '.')
  )
    return undefined;
  const base = name.split('/').at(-1) ?? '';
  if (base.endsWith('.html') && BOOKMARK_NAMES.has(base.slice(0, -5))) return 'bookmarks';
  if (!base.endsWith('.json')) return undefined;
  const stem = base.slice(0, -5);
  if (
    HISTORY_NAMES.some(
      (prefix) =>
        stem === prefix || (stem.startsWith(prefix) && /^[- _ (]/.test(stem.slice(prefix.length))),
    )
  )
    return 'history';
  return undefined;
}

async function readMember(zip: ZipFile, entry: Entry): Promise<Buffer> {
  if (
    entry.isEncrypted() ||
    !entry.canDecodeFileData() ||
    entry.uncompressedSize > BROWSER_IMPORT_LIMITS.archiveMemberBytes ||
    entry.uncompressedSize >
      Math.max(1, entry.compressedSize) * BROWSER_IMPORT_LIMITS.archiveRatio ||
    ((entry.externalFileAttributes >>> 16) & 0o170000) === 0o120000
  )
    throw new Error('IMPORT_ARCHIVE_MEMBER_UNSAFE');
  const stream = await zip.openReadStreamPromise(entry);
  const chunks: Buffer[] = [];
  let bytes = 0;
  let checksum = 0;
  try {
    for await (const chunk of stream) {
      const part = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      bytes += part.length;
      if (bytes > entry.uncompressedSize || bytes > BROWSER_IMPORT_LIMITS.archiveMemberBytes)
        throw new Error('IMPORT_ARCHIVE_MEMBER_TOO_LARGE');
      checksum = importCRC32(part, checksum);
      chunks.push(part);
    }
  } finally {
    stream.destroy();
  }
  if (bytes !== entry.uncompressedSize || checksum >>> 0 !== entry.crc32 >>> 0)
    throw new Error('IMPORT_ARCHIVE_MEMBER_CORRUPT');
  return Buffer.concat(chunks, bytes);
}

function decodeUTF8(buffer: Buffer): string {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(buffer);
  } catch {
    throw new Error('IMPORT_TEXT_INVALID');
  }
}

export function normalizeSafariHistory(value: unknown, data: BrowserImportData): void {
  if (
    !isRecord(value) ||
    !isRecord(value.metadata) ||
    value.metadata.data_type !== 'history' ||
    value.metadata.schema_version !== 1 ||
    !['Safari', 'Safari Technology Preview'].includes(String(value.metadata.browser_name)) ||
    !Array.isArray(value.history) ||
    value.history.length > BROWSER_IMPORT_LIMITS.sourceNodes
  )
    throw new Error('IMPORT_SAFARI_HISTORY_INVALID');
  for (const item of value.history) {
    if (
      !isRecord(item) ||
      typeof item.time_usec !== 'number' ||
      !Number.isSafeInteger(item.time_usec) ||
      item.time_usec < 0
    ) {
      skipImport(data, 'invalid-history-entry');
      continue;
    }
    // Both Apple's documented visits_count and older exports' visit_count are aggregate metadata.
    // Oryn records the one exported latest visit, rather than inventing missing individual visits.
    const entry = importHistoryEntry({
      url: item.url,
      title: item.title,
      visitedAt: new Date(Math.floor(item.time_usec / 1000)).toISOString(),
    });
    if (!entry) {
      skipImport(data, 'invalid-history-entry');
      continue;
    }
    data.history.push(entry);
  }
}

export async function parseSafariImport(
  request: BrowserImportWorkerRequest,
): Promise<BrowserImportData> {
  const identity = request.safariArchive;
  if (!isSafariArchiveIdentity(identity)) throw new Error('IMPORT_SOURCE_FILE_UNSAFE');
  const source = await open(
    identity.path,
    constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
  );
  let zip: ZipFile | undefined;
  try {
    if (
      !sameSafariArchive(await source.stat({ bigint: true }), identity) ||
      !sameSafariArchive(await lstat(identity.path, { bigint: true }), identity)
    )
      throw new Error('IMPORT_SOURCE_CHANGED');
    // A held descriptor and bounded random-access ranges avoid copying the container,
    // including compressed forbidden payloads, onto disk or into a whole-archive RAM buffer.
    class SelectedArchiveReader extends RandomAccessReader {
      constructor(
        private readonly file: FileHandle,
        private readonly size: number,
      ) {
        super();
      }
      override _readStreamForRange(start: number, end: number) {
        if (
          !Number.isSafeInteger(start) ||
          !Number.isSafeInteger(end) ||
          start < 0 ||
          end <= start ||
          end > this.size
        )
          throw new Error('IMPORT_ARCHIVE_RANGE_INVALID');
        const file = this.file;
        return Readable.from(
          (async function* () {
            let position = start;
            while (position < end) {
              const buffer = Buffer.alloc(Math.min(64 * 1024, end - position));
              const { bytesRead } = await file.read(buffer, 0, buffer.length, position);
              if (!bytesRead) throw new Error('IMPORT_SOURCE_CHANGED');
              position += bytesRead;
              yield buffer.subarray(0, bytesRead);
            }
          })(),
          { objectMode: false, highWaterMark: 64 * 1024 },
        );
      }
    }
    zip = await fromRandomAccessReaderPromise(
      new SelectedArchiveReader(source, identity.size),
      identity.size,
      {
        lazyEntries: true,
        autoClose: false,
        validateEntrySizes: true,
        strictFileNames: true,
      },
    );
    const data = emptyImportData();
    const names = new Set<string>();
    const matched = new Set<string>();
    let entries = 0;
    let selectedBytes = 0;
    let bookmarkFiles = 0;
    let historyFiles = 0;
    {
      if (zip.entryCount > BROWSER_IMPORT_LIMITS.archiveEntries)
        throw new Error('IMPORT_ARCHIVE_ENTRY_LIMIT');
      for await (const entry of zip.eachEntry()) {
        if (++entries > BROWSER_IMPORT_LIMITS.archiveEntries)
          throw new Error('IMPORT_ARCHIVE_ENTRY_LIMIT');
        const type = safariMemberType(entry.fileName);
        if (!type || !request.dataTypes.includes(type)) continue;
        if (names.has(entry.fileName)) throw new Error('IMPORT_ARCHIVE_DUPLICATE_MEMBER');
        names.add(entry.fileName);
        selectedBytes += entry.uncompressedSize;
        if (selectedBytes > BROWSER_IMPORT_LIMITS.archiveTotalBytes)
          throw new Error('IMPORT_ARCHIVE_TOTAL_LIMIT');
        const content = decodeUTF8(await readMember(zip, entry));
        matched.add(type);
        if (type === 'bookmarks') {
          if (++bookmarkFiles > 1) throw new Error('IMPORT_ARCHIVE_DUPLICATE_BOOKMARKS');
          normalizeSafariBookmarks(content, request, data);
        } else {
          if (++historyFiles > 64) throw new Error('IMPORT_SAFARI_PROFILE_LIMIT');
          let value: unknown;
          try {
            value = JSON.parse(content);
          } catch {
            throw new Error('IMPORT_JSON_INVALID');
          }
          normalizeSafariHistory(value, data);
        }
      }
    }
    for (const type of request.dataTypes) {
      if (type !== 'searchEngine' && !matched.has(type))
        throw new Error('IMPORT_SAFARI_MEMBER_MISSING');
    }
    const seen = new Set<string>();
    data.history = data.history
      .sort((a, b) => Date.parse(b.visitedAt) - Date.parse(a.visitedAt))
      .filter((entry) => {
        const key = `${entry.url}\0${entry.visitedAt}`;
        if (seen.has(key)) {
          skipImport(data, 'duplicate-history-entry');
          return false;
        }
        seen.add(key);
        return true;
      });
    if (data.history.length > BROWSER_IMPORT_LIMITS.historyEntries) {
      skipImport(
        data,
        'history-capacity',
        data.history.length - BROWSER_IMPORT_LIMITS.historyEntries,
      );
      data.history.length = BROWSER_IMPORT_LIMITS.historyEntries;
    }
    data.warnings.push(
      'Safari exports are unencrypted. Delete the export when you no longer need it.',
    );
    if (
      !sameSafariArchive(await source.stat({ bigint: true }), identity) ||
      !sameSafariArchive(await lstat(identity.path, { bigint: true }), identity)
    )
      throw new Error('IMPORT_SOURCE_CHANGED');
    return data;
  } finally {
    zip?.close();
    await source.close();
  }
}
