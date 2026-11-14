import type { BigIntStats } from 'node:fs';
import { lstat, realpath } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';
import { BROWSER_IMPORT_LIMITS, type SafariArchiveIdentity } from './browser-import-types';

export function isSafariArchiveIdentity(value: unknown): value is SafariArchiveIdentity {
  if (!value || typeof value !== 'object') return false;
  const item = value as SafariArchiveIdentity;
  return (
    typeof item.path === 'string' &&
    isAbsolute(item.path) &&
    item.path.length < 4096 &&
    Array.from(item.path).every(
      (character) => character.charCodeAt(0) >= 32 && character.charCodeAt(0) !== 127,
    ) &&
    Number.isSafeInteger(item.size) &&
    item.size >= 0 &&
    item.size <= BROWSER_IMPORT_LIMITS.archiveBytes &&
    [item.dev, item.ino, item.mtimeNs, item.ctimeNs].every(
      (part) => typeof part === 'string' && /^[0-9]{1,30}$/u.test(part),
    )
  );
}

export function sameSafariArchive(stats: BigIntStats, identity: SafariArchiveIdentity): boolean {
  return (
    stats.isFile() &&
    !stats.isSymbolicLink() &&
    stats.dev.toString() === identity.dev &&
    stats.ino.toString() === identity.ino &&
    stats.size === BigInt(identity.size) &&
    stats.mtimeNs.toString() === identity.mtimeNs &&
    stats.ctimeNs.toString() === identity.ctimeNs
  );
}

/** Metadata only in main. Safari data is never copied into Browser-owned disk storage. */
export async function inspectSafariArchiveSource(path: string): Promise<SafariArchiveIdentity> {
  const selected = resolve(path);
  const initial = await lstat(selected, { bigint: true });
  if (
    !initial.isFile() ||
    initial.isSymbolicLink() ||
    initial.size > BigInt(BROWSER_IMPORT_LIMITS.archiveBytes)
  )
    throw new Error('IMPORT_SOURCE_FILE_UNSAFE');
  const canonical = await realpath(selected);
  const identity: SafariArchiveIdentity = {
    path: canonical,
    dev: initial.dev.toString(),
    ino: initial.ino.toString(),
    size: Number(initial.size),
    mtimeNs: initial.mtimeNs.toString(),
    ctimeNs: initial.ctimeNs.toString(),
  };
  if (
    !isSafariArchiveIdentity(identity) ||
    !sameSafariArchive(await lstat(selected, { bigint: true }), identity) ||
    !sameSafariArchive(await lstat(canonical, { bigint: true }), identity)
  )
    throw new Error('IMPORT_SOURCE_CHANGED');
  return identity;
}
