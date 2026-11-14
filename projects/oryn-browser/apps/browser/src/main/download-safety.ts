import { constants, type BigIntStats } from 'node:fs';
import { open, lstat, type FileHandle } from 'node:fs/promises';
import { basename, isAbsolute, join } from 'node:path';
import { createRequire } from 'node:module';
import { app, dialog, shell } from 'electron';
import type { DownloadRecord } from './data/download-store';

export type DownloadOpenError = 'quarantine' | 'file-unavailable' | 'file-changed' | 'open-failed';
export interface DownloadFileIdentity {
  device: string;
  inode: string;
  size: string;
  modified: string;
  changed: string;
}
export interface DownloadSafetyNative {
  ensureQuarantine(fd: number): 'preserved' | 'added';
  hasQuarantine(fd: number): boolean;
  fileReferenceURL(fd: number): string;
}
export class DownloadSafetyError extends Error {
  constructor(readonly reason: DownloadOpenError) {
    super(`DOWNLOAD_${reason.toUpperCase().replaceAll('-', '_')}`);
  }
}

interface Dependencies {
  native?: () => DownloadSafetyNative;
  confirm?: (filename: string) => Promise<boolean>;
  openFile?: (fileReferenceURL: string) => Promise<string>;
}

/** Only Browser-owned records enter this service; no website path or command API is exposed. */
export class DownloadSafety {
  private nativeModule?: DownloadSafetyNative;
  constructor(private readonly dependencies: Dependencies = {}) {}

  private native(): DownloadSafetyNative {
    if (this.dependencies.native) return this.dependencies.native();
    if (process.platform !== 'darwin') throw new DownloadSafetyError('quarantine');
    try {
      this.nativeModule ??= createRequire(import.meta.url)(
        app.isPackaged
          ? join(process.resourcesPath, 'helpers', 'download_safety.node')
          : join(app.getAppPath(), 'build', 'helpers', 'download_safety.node'),
      ) as DownloadSafetyNative;
      return this.nativeModule;
    } catch {
      throw new DownloadSafetyError('quarantine');
    }
  }

  async check(record: DownloadRecord): Promise<DownloadFileIdentity> {
    return this.withFile(record, record.fileIdentity, async (_handle, identity) => identity);
  }

  async open(record: DownloadRecord, isCurrent: () => boolean): Promise<boolean> {
    return this.withFile(record, record.fileIdentity, async (handle, identity, native) => {
      const prefix = Buffer.alloc(512);
      const { bytesRead } = await handle.read(prefix, 0, prefix.length, 0);
      const information = await handle.stat({ bigint: true });
      if (isRiskyDownload(record.savePath!, prefix.subarray(0, bytesRead), information.mode)) {
        const name = basename(record.savePath!)
          .replace(/[\p{Cc}\p{Cf}]/gu, '')
          .slice(0, 255);
        const accepted = this.dependencies.confirm
          ? await this.dependencies.confirm(name)
          : (
              await dialog.showMessageBox({
                type: 'warning',
                title: 'Open downloaded file?',
                message: `Open ${name}?`,
                detail: 'This downloaded installer or script may run software on your Mac.',
                buttons: ['Cancel', 'Open'],
                defaultId: 0,
                cancelId: 0,
                noLink: true,
              })
            ).response === 1;
        if (!accepted) return false;
      }
      if (!isCurrent()) throw new DownloadSafetyError('file-changed');
      await verifyIdentity(handle, record.savePath!, identity);
      if (!native.hasQuarantine(handle.fd)) throw new DownloadSafetyError('quarantine');
      let bound: string;
      try {
        bound = native.fileReferenceURL(handle.fd);
      } catch {
        throw new DownloadSafetyError('file-changed');
      }
      if (!/^file:\/\/\/\.file\/id=\d+\.\d+$/u.test(bound) || !isCurrent())
        throw new DownloadSafetyError('file-changed');
      // Electron 45 macOS passes this NSURL directly to NSWorkspace. A POSIX openPath would
      // resolve its mutable filename instead. This internal file-ID URL never comes from IPC.
      // Keep the descriptor alive through the OS handoff; name replacement cannot retarget it.
      try {
        const error = this.dependencies.openFile
          ? await this.dependencies.openFile(bound)
          : await shell.openExternal(bound);
        if (error) throw new DownloadSafetyError('open-failed');
      } catch {
        throw new DownloadSafetyError('open-failed');
      }
      return true;
    });
  }

  private async withFile<T>(
    record: DownloadRecord,
    expected: DownloadFileIdentity | undefined,
    action: (
      handle: FileHandle,
      identity: DownloadFileIdentity,
      native: DownloadSafetyNative,
    ) => Promise<T>,
  ): Promise<T> {
    if (
      record.state !== 'completed' ||
      !record.savePath ||
      !isAbsolute(record.savePath) ||
      record.savePath.length > 8192 ||
      record.savePath.includes('\0')
    )
      throw new DownloadSafetyError('file-unavailable');
    let handle: FileHandle | undefined;
    try {
      handle = await open(
        record.savePath,
        constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
      );
      const initial = await handle.stat({ bigint: true });
      if (!initial.isFile() || initial.nlink !== 1n)
        throw new DownloadSafetyError('file-unavailable');
      if (expected && !sameIdentity(expected, identify(initial)))
        throw new DownloadSafetyError('file-changed');
      const native = this.native();
      try {
        native.ensureQuarantine(handle.fd);
      } catch {
        throw new DownloadSafetyError('quarantine');
      }
      const current = identify(await handle.stat({ bigint: true }));
      // Adding our missing xattr may change ctime, but cannot change the held file's bytes.
      if (!sameIdentity(identify(initial), current, false))
        throw new DownloadSafetyError('file-changed');
      await verifyIdentity(handle, record.savePath, current);
      return await action(handle, current, native);
    } catch (error) {
      if (error instanceof DownloadSafetyError) throw error;
      throw new DownloadSafetyError('file-unavailable');
    } finally {
      await handle?.close();
    }
  }
}

function identify(value: BigIntStats): DownloadFileIdentity {
  return {
    device: String(value.dev),
    inode: String(value.ino),
    size: String(value.size),
    modified: String(value.mtimeNs),
    changed: String(value.ctimeNs),
  };
}
function sameIdentity(
  left: DownloadFileIdentity,
  right: DownloadFileIdentity,
  includeChange = true,
): boolean {
  return (
    left.device === right.device &&
    left.inode === right.inode &&
    left.size === right.size &&
    left.modified === right.modified &&
    (!includeChange || left.changed === right.changed)
  );
}
async function verifyIdentity(
  handle: FileHandle,
  path: string,
  identity: DownloadFileIdentity,
): Promise<void> {
  const [held, named] = await Promise.all([
    handle.stat({ bigint: true }),
    lstat(path, { bigint: true }),
  ]);
  if (
    !named.isFile() ||
    named.nlink !== 1n ||
    !sameIdentity(identity, identify(held)) ||
    !sameIdentity(identity, identify(named))
  )
    throw new DownloadSafetyError('file-changed');
}

export function isRiskyDownload(path: string, prefix: Uint8Array, mode: bigint): boolean {
  if (
    /\.(?:dmg|pkg|app|command|sh|terminal|bash|zsh|scpt|applescript|workflow|py|pl|rb|js)$/iu.test(
      path,
    ) ||
    (mode & 0o111n) !== 0n
  )
    return true;
  if (prefix[0] === 0x23 && prefix[1] === 0x21) return true;
  const magic = Buffer.from(prefix.subarray(0, 4)).toString('hex');
  return [
    'feedface',
    'feedfacf',
    'cefaedfe',
    'cffaedfe',
    'cafebabe',
    'bebafeca',
    'cafebabf',
    'bfbafeca',
  ].includes(magic);
}
