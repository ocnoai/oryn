import {
  appendFileSync,
  fstatSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { deflateRawSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ZipFile } from 'yauzl';
import { BookmarkStore } from '../data/bookmark-store';
import { BrowsingHistoryStore } from '../data/history-store';
import { BrowserImportService } from '../data/browser-import-service';
import { inspectSafariArchiveSource } from '../data/browser-import-safari-source';
import { importCRC32, parseSafariImport, safariMemberType } from '../data/browser-import-safari';
import type { BrowserImportWorkerRequest } from '../data/browser-import-types';

const directories: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
});

// Deliberately small fixture writer, not a production ZIP implementation.
function fixtureZIP(
  entries: {
    name: string;
    content: string;
    corruptCompressed?: boolean;
    advertisedSize?: number;
  }[],
): Buffer {
  const locals: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name);
    const content = Buffer.from(entry.content);
    const compressed = entry.corruptCompressed
      ? Buffer.from([0xff, 0xff, 0xff])
      : deflateRawSync(content);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x800, 6);
    local.writeUInt16LE(8, 8);
    local.writeUInt32LE(importCRC32(content), 14);
    local.writeUInt32LE(compressed.length, 18);
    local.writeUInt32LE(entry.advertisedSize ?? content.length, 22);
    local.writeUInt16LE(name.length, 26);
    const record = Buffer.alloc(46);
    record.writeUInt32LE(0x02014b50, 0);
    record.writeUInt16LE(20, 4);
    record.writeUInt16LE(20, 6);
    record.writeUInt16LE(0x800, 8);
    record.writeUInt16LE(8, 10);
    record.writeUInt32LE(importCRC32(content), 16);
    record.writeUInt32LE(compressed.length, 20);
    record.writeUInt32LE(entry.advertisedSize ?? content.length, 24);
    record.writeUInt16LE(name.length, 28);
    record.writeUInt32LE(offset, 42);
    locals.push(local, name, compressed);
    central.push(record, name);
    offset += local.length + name.length + compressed.length;
  }
  const directory = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, directory, end]);
}

function request(
  entries: Parameters<typeof fixtureZIP>[0],
  dataTypes: BrowserImportWorkerRequest['dataTypes'] = ['bookmarks', 'history'],
): BrowserImportWorkerRequest {
  const directory = mkdtempSync(join(tmpdir(), 'oryn-safari-import-test-'));
  directories.push(directory);
  const archive = join(directory, 'Safari.zip');
  writeFileSync(archive, fixtureZIP(entries));
  const path = realpathSync(archive);
  const stats = lstatSync(path, { bigint: true });
  return {
    version: 2,
    kind: 'safari',
    directory: '',
    safariArchive: {
      path,
      dev: stats.dev.toString(),
      ino: stats.ino.toString(),
      size: Number(stats.size),
      mtimeNs: stats.mtimeNs.toString(),
      ctimeNs: stats.ctimeNs.toString(),
    },
    sourceKey: 'b'.repeat(64),
    sourceLabel: 'Safari',
    dataTypes,
    bookmarkCapacity: 20_000,
    existingBookmarkIds: [],
  };
}

function history(url: string, time = 1_730_764_800_000_000): string {
  return JSON.stringify({
    metadata: { browser_name: 'Safari', schema_version: 1, data_type: 'history' },
    history: [{ url, title: 'Page', time_usec: time, visits_count: 2 }],
  });
}

describe('selective Safari export parsing', () => {
  it('reads bookmarks and multiple profile histories while never opening forbidden payload streams', async () => {
    const read = vi.spyOn(ZipFile.prototype, 'openReadStreamPromise');
    const data = await parseSafariImport(
      request([
        {
          name: 'Bookmarks.html',
          content:
            '<DL><H3>Práce 🧪 — 資料 &amp; notes</H3><DL><A HREF="https://bookmark.test/">Příliš žluťoučký kůň 🦊 — أهلاً</A></DL></DL>',
        },
        { name: 'History.json', content: history('https://history.test/') },
        {
          name: 'History (Work).json',
          content: history('https://work.test/', 1_730_764_801_000_000),
        },
        { name: 'Passwords.csv', content: 'secret', corruptCompressed: true },
        { name: 'PaymentCards.json', content: 'secret', corruptCompressed: true },
        { name: 'Extensions.json', content: 'secret', corruptCompressed: true },
      ]),
    );
    expect(data.bookmarks.filter((item) => item.kind === 'bookmark')).toHaveLength(1);
    expect(data.bookmarks).toContainEqual(
      expect.objectContaining({ kind: 'folder', title: 'Práce 🧪 — 資料 & notes' }),
    );
    expect(data.bookmarks).toContainEqual(
      expect.objectContaining({ kind: 'bookmark', title: 'Příliš žluťoučký kůň 🦊 — أهلاً' }),
    );
    expect(data.history.map((item) => item.url)).toEqual([
      'https://work.test/',
      'https://history.test/',
    ]);
    expect(read.mock.calls.map(([entry]) => entry.fileName)).toEqual([
      'Bookmarks.html',
      'History.json',
      'History (Work).json',
    ]);
  });

  it('rejects duplicate, oversized, traversal and malformed selected members', async () => {
    await expect(
      parseSafariImport(
        request(
          [
            { name: 'History.json', content: history('https://one.test/') },
            { name: 'History.json', content: history('https://two.test/') },
          ],
          ['history'],
        ),
      ),
    ).rejects.toThrow('IMPORT_ARCHIVE_DUPLICATE_MEMBER');
    await expect(
      parseSafariImport(
        request(
          [{ name: 'History.json', content: '{}', advertisedSize: 20 * 1024 * 1024 }],
          ['history'],
        ),
      ),
    ).rejects.toThrow('IMPORT_ARCHIVE_MEMBER_UNSAFE');
    await expect(
      parseSafariImport(
        request([{ name: '../History.json', content: history('https://one.test/') }], ['history']),
      ),
    ).rejects.toThrow();
    await expect(
      parseSafariImport(
        request(
          [
            {
              name: 'History.json',
              content: JSON.stringify({ metadata: { data_type: 'payment_cards' }, history: [] }),
            },
          ],
          ['history'],
        ),
      ),
    ).rejects.toThrow('IMPORT_SAFARI_HISTORY_INVALID');
    await expect(
      parseSafariImport(request([{ name: 'History.json', content: '{bad' }], ['history'])),
    ).rejects.toThrow('IMPORT_JSON_INVALID');
  });

  it('has closed filename recognition and CRC coverage', () => {
    expect(safariMemberType('Historique (Travail).json')).toBe('history');
    expect(safariMemberType('Passwords.json')).toBeUndefined();
    expect(safariMemberType('My History.json')).toBeUndefined();
    expect(safariMemberType('unknown-localized-name.json')).toBeUndefined();
    expect(importCRC32(Buffer.from('123456789'))).toBe(0xcbf43926);
    expect(importCRC32(Buffer.from('456789'), importCRC32(Buffer.from('123')))).toBe(0xcbf43926);
  });

  it('rejects a valid archive with no required members without opening excluded payloads or retaining its descriptor', async () => {
    const selected = request([
      { name: 'Passwords.csv', content: 'forbidden-password' },
      { name: 'PaymentCards.json', content: 'forbidden-payment' },
      { name: 'Extensions.json', content: 'forbidden-extension' },
    ]);
    const identity = selected.safariArchive!;
    const descriptorCount = () =>
      readdirSync('/dev/fd').filter((name) => {
        if (!/^[0-9]+$/u.test(name)) return false;
        try {
          const stats = fstatSync(Number(name), { bigint: true });
          return stats.dev.toString() === identity.dev && stats.ino.toString() === identity.ino;
        } catch {
          return false;
        }
      }).length;
    expect(descriptorCount()).toBe(0);
    const openMember = vi.spyOn(ZipFile.prototype, 'openReadStreamPromise');
    const eachEntry = ZipFile.prototype.eachEntry;
    let observedHeldDescriptor = false;
    vi.spyOn(ZipFile.prototype, 'eachEntry').mockImplementation(function (this: ZipFile) {
      expect(descriptorCount()).toBe(1);
      observedHeldDescriptor = true;
      return eachEntry.call(this);
    });
    await expect(parseSafariImport(selected)).rejects.toThrow('IMPORT_SAFARI_MEMBER_MISSING');
    expect(observedHeldDescriptor).toBe(true);
    expect(openMember).not.toHaveBeenCalled();
    expect(descriptorCount()).toBe(0);
  });

  it('never creates an archive scratch copy during preview, cancellation or commit', async () => {
    const selected = request(
      [
        {
          name: 'Bookmarks.html',
          content: '<DL><A HREF="https://selected.test/">Selected</A></DL>',
        },
        { name: 'Passwords.csv', content: 'forbidden-password-bytes', corruptCompressed: true },
      ],
      ['bookmarks'],
    );
    const root = mkdtempSync(join(tmpdir(), 'oryn-safari-no-copy-'));
    directories.push(root);
    const temporaryRoot = join(root, 'scratch');
    mkdirSync(temporaryRoot);
    const bookmarks = new BookmarkStore(join(root, 'bookmarks.json'));
    const historyStore = new BrowsingHistoryStore(join(root, 'history.json'));
    const original = ZipFile.prototype.openReadStreamPromise;
    vi.spyOn(ZipFile.prototype, 'openReadStreamPromise').mockImplementation(function (
      this: ZipFile,
      entry,
      options,
    ) {
      expect(readdirSync(temporaryRoot)).toEqual([]);
      return original.call(this, entry, options);
    });
    const service = new BrowserImportService({
      bookmarks,
      history: historyStore,
      homePath: root,
      temporaryRoot,
      workerPath: '/unused',
      nativeSandboxPath: '/unused',
      runWorker: async (workerRequest) => {
        expect(workerRequest.directory).toBe('');
        expect(workerRequest.safariArchive?.path).toBe(selected.safariArchive!.path);
        expect(readdirSync(temporaryRoot)).toEqual([]);
        return parseSafariImport(workerRequest);
      },
    });
    const source = (await service.discover()).find((item) => item.browser === 'safari')!;
    const owner = { webContentsId: 15 };
    try {
      const preview = await service.preview(owner, {
        sourceId: source.sourceId,
        dataTypes: ['bookmarks'],
        safariArchivePath: selected.safariArchive!.path,
      });
      expect(preview.counts.bookmarks).toBe(1);
      expect(bookmarks.list()).toEqual([]);
      service.cancel(owner, preview.token);
      expect(bookmarks.list()).toEqual([]);
      expect(readdirSync(temporaryRoot)).toEqual([]);
      const confirmed = await service.preview(owner, {
        sourceId: source.sourceId,
        dataTypes: ['bookmarks'],
        safariArchivePath: selected.safariArchive!.path,
      });
      expect(service.commit(owner, confirmed.token).bookmarks).toBe(1);
      expect(readdirSync(temporaryRoot)).toEqual([]);
    } finally {
      service.dispose();
    }
  });

  it('rejects source symlinks and changes before or during selective reading', async () => {
    const selected = request(
      [{ name: 'History.json', content: history('https://selected.test/') }],
      ['history'],
    );
    const path = selected.safariArchive!.path;
    const link = `${path}.link`;
    symlinkSync(path, link);
    await expect(inspectSafariArchiveSource(link)).rejects.toThrow('IMPORT_SOURCE_FILE_UNSAFE');
    const original = ZipFile.prototype.openReadStreamPromise;
    vi.spyOn(ZipFile.prototype, 'openReadStreamPromise').mockImplementationOnce(function (
      this: ZipFile,
      entry,
      options,
    ) {
      appendFileSync(path, 'changed-after-central-directory-read');
      return original.call(this, entry, options);
    });
    await expect(parseSafariImport(selected)).rejects.toThrow('IMPORT_SOURCE_CHANGED');
    await expect(parseSafariImport(selected)).rejects.toThrow('IMPORT_SOURCE_CHANGED');
  });
});
