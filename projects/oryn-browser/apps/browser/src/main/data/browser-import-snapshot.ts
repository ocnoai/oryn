import { constants, type Stats } from 'node:fs';
import { chmod, lstat, mkdir, mkdtemp, open, readdir, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { BROWSER_IMPORT_LIMITS, type BrowserImportDataType } from './browser-import-types';

interface SourceFile {
  source: string;
  name: string;
  maximum: number;
  optional?: boolean;
}
interface SnapshotFile extends SourceFile {
  identity: Stats;
}

function sameIdentity(left: Stats, right: Stats): boolean {
  return (
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeMs === right.mtimeMs &&
    left.ctimeMs === right.ctimeMs
  );
}

async function copyBounded(file: SnapshotFile, destination: string): Promise<void> {
  const source = await open(
    file.source,
    constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
  );
  let target: Awaited<ReturnType<typeof open>> | undefined;
  try {
    target = await open(
      destination,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL,
      0o600,
    );
    const initial = await source.stat();
    if (!initial.isFile() || initial.size > file.maximum || !sameIdentity(file.identity, initial))
      throw new Error('IMPORT_SOURCE_CHANGED');
    const buffer = Buffer.alloc(64 * 1024);
    let position = 0;
    while (position < initial.size) {
      const count = Math.min(buffer.length, initial.size - position);
      const { bytesRead } = await source.read(buffer, 0, count, position);
      if (!bytesRead) throw new Error('IMPORT_SOURCE_CHANGED');
      let written = 0;
      while (written < bytesRead) {
        const next = await target.write(buffer, written, bytesRead - written, position + written);
        if (!next.bytesWritten) throw new Error('IMPORT_COPY_FAILED');
        written += next.bytesWritten;
      }
      position += bytesRead;
    }
    if (!sameIdentity(initial, await source.stat())) throw new Error('IMPORT_SOURCE_CHANGED');
  } finally {
    await source.close();
    await target?.close();
  }
  await chmod(destination, 0o400);
}

/** A live SQLite database and its WAL must be copied as one stable observed set, never independently trusted. */
export async function createImportSnapshot(options: {
  profileDirectory?: string;
  dataTypes: BrowserImportDataType[];
  temporaryRoot?: string;
}): Promise<{ directory: string; dispose(): Promise<void> }> {
  const files: SourceFile[] = [];
  {
    if (!options.profileDirectory) throw new Error('IMPORT_SOURCE_INVALID');
    if (options.dataTypes.includes('bookmarks'))
      files.push({
        source: join(options.profileDirectory, 'Bookmarks'),
        name: 'Bookmarks',
        maximum: BROWSER_IMPORT_LIMITS.bookmarksBytes,
      });
    if (options.dataTypes.includes('history')) {
      files.push({
        source: join(options.profileDirectory, 'History'),
        name: 'History',
        maximum: BROWSER_IMPORT_LIMITS.historyBytes,
      });
      for (const suffix of ['-wal', '-shm'])
        files.push({
          source: join(options.profileDirectory, `History${suffix}`),
          name: `History${suffix}`,
          maximum: BROWSER_IMPORT_LIMITS.historyBytes,
          optional: true,
        });
    }
    if (options.dataTypes.includes('searchEngine'))
      files.push({
        source: join(options.profileDirectory, 'Preferences'),
        name: 'Preferences',
        maximum: BROWSER_IMPORT_LIMITS.preferencesBytes,
      });
  }
  const root = options.temporaryRoot ?? tmpdir();
  await mkdir(root, { recursive: true });
  for (let attempt = 0; attempt < 3; attempt++) {
    const directory = await mkdtemp(join(root, 'oryn-browser-import-'));
    await chmod(directory, 0o700);
    try {
      const parentIdentity = options.profileDirectory
        ? await lstat(options.profileDirectory)
        : undefined;
      if (parentIdentity && (!parentIdentity.isDirectory() || parentIdentity.isSymbolicLink()))
        throw new Error('IMPORT_SOURCE_FILE_UNSAFE');
      const found: SnapshotFile[] = [];
      const absent = new Set<string>();
      for (const file of files) {
        const identity = await lstat(file.source).catch((error: NodeJS.ErrnoException) => {
          if (error.code === 'ENOENT' && file.optional) return undefined;
          throw error;
        });
        if (!identity) {
          absent.add(file.source);
          continue;
        }
        if (!identity.isFile() || identity.isSymbolicLink() || identity.size > file.maximum)
          throw new Error('IMPORT_SOURCE_FILE_UNSAFE');
        found.push({ ...file, identity });
      }
      for (const file of found) await copyBounded(file, join(directory, file.name));
      for (const file of found)
        if (!sameIdentity(file.identity, await stat(file.source)))
          throw new Error('IMPORT_SOURCE_CHANGED');
      for (const missing of absent)
        if (
          await lstat(missing).then(
            () => true,
            (error: NodeJS.ErrnoException) => {
              if (error.code === 'ENOENT') return false;
              throw error;
            },
          )
        )
          throw new Error('IMPORT_SOURCE_CHANGED');
      if (parentIdentity && !sameIdentity(parentIdentity, await lstat(options.profileDirectory!)))
        throw new Error('IMPORT_SOURCE_CHANGED');
      return { directory, dispose: () => rm(directory, { recursive: true, force: true }) };
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      if (!(error instanceof Error) || error.message !== 'IMPORT_SOURCE_CHANGED' || attempt === 2)
        throw error;
    }
  }
  throw new Error('IMPORT_SOURCE_CHANGED');
}

/** Startup-only: caller must hold the Browser profile lock and use a dedicated import scratch root. */
export async function cleanupImportSnapshots(
  temporaryRoot: string,
): Promise<{ removed: number; failed: number }> {
  const information = await lstat(temporaryRoot).catch((error: NodeJS.ErrnoException) => {
    if (error.code === 'ENOENT') return undefined;
    throw error;
  });
  if (!information) return { removed: 0, failed: 0 };
  if (!information.isDirectory() || information.isSymbolicLink())
    throw new Error('IMPORT_TEMP_ROOT_UNSAFE');
  const entries = await readdir(temporaryRoot, { withFileTypes: true });
  if (entries.length > 1024) throw new Error('IMPORT_TEMP_ROOT_LIMIT');
  let removed = 0;
  let failed = 0;
  for (const entry of entries) {
    if (!/^oryn-browser-import-[A-Za-z0-9]{6}$/.test(entry.name)) continue;
    try {
      // rm removes a symlink itself instead of following it. No archive/source path enters this API.
      await rm(join(temporaryRoot, entry.name), { recursive: true, force: true });
      removed++;
    } catch {
      failed++;
    }
  }
  return { removed, failed };
}
