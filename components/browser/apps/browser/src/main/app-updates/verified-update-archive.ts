import { spawn } from 'node:child_process';
import { createHash, timingSafeEqual } from 'node:crypto';
import { closeSync, constants, fstatSync, openSync, readSync, type BigIntStats } from 'node:fs';
import { chmod, mkdir, mkdtemp, open, realpath, rm, statfs, unlink } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { Readable } from 'node:stream';
import { RandomAccessReader, ZipFile } from 'yauzl';

import { isStrictlyNewerVersion } from './app-update-service';
import {
  checkedZipDirectory,
  checkedLocalRange,
  checkLocalCoverage,
  checkSymlinkGraph,
  MAX_UPDATE_ZIP_ENTRIES,
} from './update-zip-layout';

const GiB = 1024 ** 3;
const MAX_ZIP_BYTES = 8 * GiB;
const MAX_UNPACKED_BYTES = 32 * GiB;
const APP_NAME = 'Oryn Browser.app';

export interface UpdatePayloadIdentity {
  version: string;
  bundleVersion: string;
  size: number;
  sha512: string;
  unpackedSize: number;
}

export interface NativeVerificationPolicy {
  expectedAppId: string;
  expectedTeamId: string;
  currentVersion: string;
  currentSystemVersion: string;
  verificationRoot: string;
}

export interface NativeCommandOptions {
  cwd: string;
  signal: AbortSignal;
  archiveFd?: number;
}

export type NativeCommandRunner = (
  command: string,
  args: readonly string[],
  options: NativeCommandOptions,
) => Promise<string>;

export function validBundleVersion(value: unknown): value is string {
  return typeof value === 'string' && /^[\x21-\x7e]{1,64}$/u.test(value);
}

export function validUpdatePayloadSizes(size: unknown, unpackedSize: unknown): boolean {
  return (
    typeof size === 'number' &&
    Number.isSafeInteger(size) &&
    size > 0 &&
    size <= MAX_ZIP_BYTES &&
    typeof unpackedSize === 'number' &&
    Number.isSafeInteger(unpackedSize) &&
    unpackedSize > 0 &&
    unpackedSize <= MAX_UNPACKED_BYTES
  );
}

function supportsSystem(minimum: string, current: string): boolean {
  const parse = (value: string) =>
    /^\d{1,4}\.\d{1,4}(?:\.\d{1,4})?$/u.test(value) ? value.split('.').map(Number) : undefined;
  const left = parse(minimum);
  const right = parse(current);
  if (!left || !right) return false;
  for (let index = 0; index < 3; index++) {
    if ((right[index] ?? 0) !== (left[index] ?? 0)) return (right[index] ?? 0) > (left[index] ?? 0);
  }
  return true;
}

function fail(code: string): never {
  throw new Error(code);
}

function unsafePathText(value: string): boolean {
  return (
    value.includes('\\') ||
    [...value].some((character) => {
      const code = character.charCodeAt(0);
      return code < 32 || code === 127;
    })
  );
}

function identity(fd: number): BigIntStats {
  const value = fstatSync(fd, { bigint: true });
  if (!value.isFile()) fail('APP_UPDATE_ARCHIVE_INVALID');
  return value;
}

function sameFile(left: BigIntStats, right: BigIntStats): boolean {
  return (
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs
  );
}

function expectedDigest(value: string): Buffer {
  const bytes = Buffer.from(value, 'base64');
  if (bytes.length !== 64 || bytes.toString('base64') !== value)
    fail('APP_UPDATE_ZIP_METADATA_INVALID');
  return bytes;
}

function descriptorStream(fd: number, start: number, end: number, current = () => true): Readable {
  let position = start;
  return new Readable({
    read() {
      // Each bounded read yields to cancellation/UI work. No fs.ReadStream owns
      // this descriptor: destroy/early iterator exit must never close/reuse it.
      setImmediate(() => {
        if (this.destroyed) return;
        if (!current()) {
          this.destroy(new Error('APP_UPDATE_PAYLOAD_CHANGED'));
          return;
        }
        if (position >= end) {
          this.push(null);
          return;
        }
        try {
          const buffer = Buffer.alloc(Math.min(1024 * 1024, end - position));
          const length = readSync(fd, buffer, 0, buffer.length, position);
          if (!length) throw new Error('APP_UPDATE_PAYLOAD_CHANGED');
          position += length;
          this.push(buffer.subarray(0, length));
        } catch {
          this.destroy(new Error('APP_UPDATE_PAYLOAD_CHANGED'));
        }
      });
    },
  });
}

export class VerifiedUpdateArchive {
  private closed = false;
  private readonly original: BigIntStats;

  constructor(
    private readonly fd: number,
    readonly payload: UpdatePayloadIdentity,
  ) {
    this.original = identity(fd);
    if (this.original.size !== BigInt(payload.size)) fail('APP_UPDATE_PAYLOAD_CHANGED');
  }

  async revalidate(signal?: AbortSignal): Promise<void> {
    if (this.closed || !sameFile(this.original, identity(this.fd)))
      fail('APP_UPDATE_PAYLOAD_CHANGED');
    const hash = createHash('sha512');
    for await (const chunk of this.stream()) {
      signal?.throwIfAborted();
      hash.update(chunk);
    }
    if (
      !sameFile(this.original, identity(this.fd)) ||
      !timingSafeEqual(hash.digest(), expectedDigest(this.payload.sha512))
    )
      fail('APP_UPDATE_PAYLOAD_CHANGED');
  }

  stream(): Readable {
    if (this.closed || !sameFile(this.original, identity(this.fd)))
      fail('APP_UPDATE_PAYLOAD_CHANGED');
    // The private copy is unlinked and its write handle is closed. Native staging
    // reads this retained read-only descriptor; it never reopens a cache pathname.
    return descriptorStream(this.fd, 0, this.payload.size, () => !this.closed);
  }

  descriptor(): number {
    if (this.closed) fail('APP_UPDATE_PAYLOAD_CHANGED');
    return this.fd;
  }

  isOpen(): boolean {
    return !this.closed;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    closeSync(this.fd);
  }
}

class DescriptorReader extends RandomAccessReader {
  constructor(private readonly archive: VerifiedUpdateArchive) {
    super();
  }

  override _readStreamForRange(start: number, end: number): Readable {
    return descriptorStream(this.archive.descriptor(), start, end, () => this.archive.isOpen());
  }

  override close(callback: (error: Error | null) => void): void {
    // yauzl does not own the retained archive descriptor.
    callback(null);
  }
}

export async function inspectUpdateZip(
  archive: VerifiedUpdateArchive,
  signal: AbortSignal,
): Promise<void> {
  const directory = await checkedZipDirectory(archive.descriptor(), archive.payload.size, signal);
  // Use yauzl's public record decoder with our independently checked complete
  // directory count. Its EOCD parser truncates classic ditto count-overflow ZIPs.
  const reader = new DescriptorReader(archive) as DescriptorReader & { ref(): void };
  reader.ref();
  const zip = new ZipFile(
    reader,
    directory.offset,
    archive.payload.size,
    directory.count,
    '',
    true,
    true,
    true,
    true,
  ) as ZipFile & { strictFileNames: boolean };
  zip.strictFileNames = true;
  const ranges: Array<[number, number]> = [];
  const names = new Set<string>();
  const links = new Map<string, string>();
  const regularFiles = new Set<string>();
  let bytes = 0;
  let count = 0;
  let hasInfo = false;
  try {
    if (zip.entryCount < 1 || zip.entryCount > MAX_UPDATE_ZIP_ENTRIES)
      fail('APP_UPDATE_ARCHIVE_INVALID');
    for await (const entry of zip.eachEntry()) {
      signal.throwIfAborted();
      const name = entry.fileName;
      const rawName = entry.fileNameRaw.toString('utf8');
      const trimmed = name.endsWith('/') ? name.slice(0, -1) : name;
      const parts = trimmed.split('/');
      const metadata = parts[0] === '__MACOSX';
      if (
        ++count > MAX_UPDATE_ZIP_ENTRIES ||
        rawName !== name ||
        !Buffer.from(rawName, 'utf8').equals(entry.fileNameRaw) ||
        Buffer.byteLength(name) > 1024 ||
        parts.length > 64 ||
        parts.some((part) => !part || part === '.' || part === '..' || unsafePathText(part)) ||
        (parts[0] !== APP_NAME &&
          !(
            metadata &&
            (parts[1] === APP_NAME ||
              parts.length === 1 ||
              (parts.length === 2 && parts[1] === `._${APP_NAME}`))
          )) ||
        !entry.canDecodeFileData() ||
        entry.isEncrypted() ||
        !Number.isSafeInteger(entry.uncompressedSize) ||
        entry.uncompressedSize < 0
      )
        fail('APP_UPDATE_ARCHIVE_INVALID');
      ranges.push(
        checkedLocalRange(
          archive.descriptor(),
          archive.payload.size,
          directory.offset,
          entry,
          await zip.readLocalFileHeaderPromise(entry),
        ),
      );
      const key = trimmed.normalize('NFC').toLowerCase();
      if (names.has(key)) fail('APP_UPDATE_ARCHIVE_INVALID');
      names.add(key);
      bytes += entry.uncompressedSize;
      if (!Number.isSafeInteger(bytes) || bytes > archive.payload.unpackedSize)
        fail('APP_UPDATE_ARCHIVE_INVALID');
      const mode = (entry.externalFileAttributes >>> 16) & 0xf000;
      if (![0, 0x4000, 0x8000, 0xa000].includes(mode)) fail('APP_UPDATE_ARCHIVE_INVALID');
      if (
        ((entry.externalFileAttributes >>> 16) & 0x0e00) !== 0 ||
        (name.endsWith('/') && (entry.uncompressedSize !== 0 || (mode !== 0 && mode !== 0x4000))) ||
        (!name.endsWith('/') && mode === 0x4000)
      )
        fail('APP_UPDATE_ARCHIVE_INVALID');
      if (mode === 0xa000) {
        if (
          metadata ||
          entry.uncompressedSize > 4096 ||
          entry.compressedSize > 64 * 1024 ||
          name.endsWith('/')
        )
          fail('APP_UPDATE_ARCHIVE_INVALID');
        const stream = await zip.openReadStreamPromise(entry);
        const chunks: Buffer[] = [];
        let length = 0;
        const abort = () => stream.destroy(new Error('APP_UPDATE_CANCELLED'));
        signal.addEventListener('abort', abort, { once: true });
        try {
          for await (const value of stream) {
            signal.throwIfAborted();
            const chunk = Buffer.from(value);
            length += chunk.length;
            if (length > 4096) fail('APP_UPDATE_ARCHIVE_INVALID');
            chunks.push(chunk);
          }
        } finally {
          signal.removeEventListener('abort', abort);
          stream.destroy();
        }
        const rawTarget = Buffer.concat(chunks);
        const target = rawTarget.toString('utf8');
        const destination = resolve('/', dirname(trimmed), target);
        if (
          !target ||
          !Buffer.from(target, 'utf8').equals(rawTarget) ||
          isAbsolute(target) ||
          unsafePathText(target) ||
          !destination.startsWith(`/${APP_NAME}/`)
        )
          fail('APP_UPDATE_ARCHIVE_INVALID');
        links.set(key, target);
      } else if (!name.endsWith('/')) {
        regularFiles.add(key);
        // Validate actual inflation before native extraction. An attacker cannot
        // obtain an unbounded ditto write by lying in the central directory.
        const stream = await zip.openReadStreamPromise(entry);
        let actual = 0;
        const abort = () => stream.destroy(new Error('APP_UPDATE_CANCELLED'));
        signal.addEventListener('abort', abort, { once: true });
        try {
          for await (const chunk of stream) {
            signal.throwIfAborted();
            actual += chunk.length;
            if (actual > entry.uncompressedSize) fail('APP_UPDATE_ARCHIVE_INVALID');
          }
          if (actual !== entry.uncompressedSize) fail('APP_UPDATE_ARCHIVE_INVALID');
        } finally {
          signal.removeEventListener('abort', abort);
          stream.destroy();
        }
      }
      hasInfo ||= name === `${APP_NAME}/Contents/Info.plist` && (mode === 0 || mode === 0x8000);
    }
    if (!hasInfo) fail('APP_UPDATE_ARCHIVE_INVALID');
    checkLocalCoverage(ranges, directory.offset);
    await checkSymlinkGraph(links, APP_NAME, signal);
    // No archive entry may write through a previously created symlink. Framework
    // aliases are retained, but their real children must use the real directory.
    for (const name of names) {
      const parts = name.split('/');
      while (parts.length > 1) {
        parts.pop();
        if (links.has(parts.join('/')) || regularFiles.has(parts.join('/')))
          fail('APP_UPDATE_ARCHIVE_INVALID');
      }
    }
  } finally {
    zip.close();
  }
}

export const runNativeVerificationCommand: NativeCommandRunner = (command, args, options) =>
  new Promise((resolveResult, reject) => {
    const child = spawn(command, [...args], {
      cwd: options.cwd,
      env: {
        PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
        HOME: options.cwd,
        TMPDIR: options.cwd,
      },
      stdio: ['ignore', 'pipe', 'pipe', options.archiveFd ?? 'ignore'],
      signal: options.signal,
      timeout: 120_000,
      killSignal: 'SIGKILL',
    });
    const chunks: Buffer[] = [];
    let length = 0;
    const collect = (chunk: Buffer) => {
      length += chunk.length;
      if (length > 128 * 1024) child.kill('SIGKILL');
      else chunks.push(chunk);
    };
    child.stdout?.on('data', collect);
    child.stderr?.on('data', collect);
    child.once('error', () => reject(new Error('APP_UPDATE_NATIVE_CHECK_FAILED')));
    child.once('close', (code) => {
      if (code !== 0 || length > 128 * 1024) reject(new Error('APP_UPDATE_NATIVE_CHECK_FAILED'));
      else resolveResult(Buffer.concat(chunks).toString('utf8'));
    });
  });

export function validateMacExecutable(fd: number): void {
  const size = Number(identity(fd).size);
  const read = (offset: number, length: number) => {
    if (offset < 0 || length < 0 || offset + length > size) fail('APP_UPDATE_ARCH_MISMATCH');
    const value = Buffer.alloc(length);
    if (readSync(fd, value, 0, length, offset) !== length) fail('APP_UPDATE_ARCH_MISMATCH');
    return value;
  };
  const magic = read(0, 8);
  let slice = 0;
  let sliceSize = size;
  if (magic.readUInt32BE() === 0xcafebabe || magic.readUInt32BE() === 0xcafebabf) {
    const wide = magic.readUInt32BE() === 0xcafebabf;
    const count = magic.readUInt32BE(4);
    if (count < 1 || count > 16) fail('APP_UPDATE_ARCH_MISMATCH');
    let found = false;
    for (let index = 0; index < count; index++) {
      const item = read(8 + index * (wide ? 32 : 20), wide ? 32 : 20);
      if (item.readUInt32BE() !== 0x0100000c) continue;
      if (found) fail('APP_UPDATE_ARCH_MISMATCH');
      found = true;
      slice = wide ? Number(item.readBigUInt64BE(8)) : item.readUInt32BE(8);
      sliceSize = wide ? Number(item.readBigUInt64BE(16)) : item.readUInt32BE(12);
      if (
        !Number.isSafeInteger(slice) ||
        !Number.isSafeInteger(sliceSize) ||
        slice + sliceSize > size
      )
        fail('APP_UPDATE_ARCH_MISMATCH');
    }
    if (!found) fail('APP_UPDATE_ARCH_MISMATCH');
  }
  const header = read(slice, 32);
  if (header.readUInt32LE() !== 0xfeedfacf || header.readUInt32LE(4) !== 0x0100000c)
    fail('APP_UPDATE_ARCH_MISMATCH');
  const commands = header.readUInt32LE(16);
  const bytes = header.readUInt32LE(20);
  if (commands < 1 || commands > 8192 || bytes > 4 * 1024 * 1024 || bytes + 32 > sliceSize)
    fail('APP_UPDATE_ARCH_MISMATCH');
  const data = read(slice + 32, bytes);
  let offset = 0;
  let macOS = false;
  for (let index = 0; index < commands; index++) {
    if (offset + 8 > bytes) fail('APP_UPDATE_ARCH_MISMATCH');
    const command = data.readUInt32LE(offset);
    const length = data.readUInt32LE(offset + 4);
    if (length < 8 || offset + length > bytes) fail('APP_UPDATE_ARCH_MISMATCH');
    if (command === 0x32) {
      if (length < 24 || data.readUInt32LE(offset + 8) !== 1) fail('APP_UPDATE_ARCH_MISMATCH');
      macOS = true;
    }
    if (command === 0x24) macOS = true;
    offset += length;
  }
  if (!macOS || offset !== bytes) fail('APP_UPDATE_ARCH_MISMATCH');
}

export async function verifySignedMacApp(
  appPath: string,
  payload: UpdatePayloadIdentity,
  policy: NativeVerificationPolicy,
  options: NativeCommandOptions,
  run: NativeCommandRunner = runNativeVerificationCommand,
): Promise<void> {
  const requirement = `anchor apple generic and identifier "${policy.expectedAppId}" and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "${policy.expectedTeamId}"`;
  try {
    await run(
      '/usr/bin/codesign',
      ['--verify', '--deep', '--strict', `-R=${requirement}`, appPath],
      options,
    );
    const details = await run('/usr/bin/codesign', ['--display', '--verbose=4', appPath], options);
    if (!/flags=0x[0-9a-f]+\([^)]*runtime/iu.test(details)) fail('APP_UPDATE_SIGNATURE_INVALID');
  } catch {
    fail('APP_UPDATE_SIGNATURE_INVALID');
  }
  try {
    const assessment = await run(
      '/usr/sbin/spctl',
      ['--assess', '--type', 'execute', '--ignore-cache', '--no-cache', '--verbose=2', appPath],
      options,
    );
    if (!/source=Notarized Developer ID/u.test(assessment)) fail('APP_UPDATE_NOTARIZATION_INVALID');
  } catch {
    fail('APP_UPDATE_NOTARIZATION_INVALID');
  }
  const plist = join(appPath, 'Contents/Info.plist');
  const field = async (name: string) =>
    (await run('/usr/bin/plutil', ['-extract', name, 'raw', '-o', '-', plist], options)).replace(
      /\r?\n$/u,
      '',
    );
  const appId = await field('CFBundleIdentifier');
  const version = await field('CFBundleShortVersionString');
  const build = await field('CFBundleVersion');
  if (appId !== policy.expectedAppId) fail('APP_UPDATE_IDENTITY_MISMATCH');
  if (
    version !== payload.version ||
    build !== payload.bundleVersion ||
    !isStrictlyNewerVersion(version, policy.currentVersion)
  )
    fail('APP_UPDATE_VERSION_MISMATCH');
  if (!supportsSystem(await field('LSMinimumSystemVersion'), policy.currentSystemVersion))
    fail('APP_UPDATE_OS_UNSUPPORTED');
  const executable = await field('CFBundleExecutable');
  if (
    !executable ||
    executable.length > 255 ||
    executable.includes('/') ||
    unsafePathText(executable) ||
    executable === '.' ||
    executable === '..'
  )
    fail('APP_UPDATE_IDENTITY_MISMATCH');
  const path = join(appPath, 'Contents/MacOS', executable);
  if ((await realpath(path)) !== path) fail('APP_UPDATE_ARCH_MISMATCH');
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    validateMacExecutable(fd);
  } finally {
    closeSync(fd);
  }
}

export async function verifyUpdateArchive(
  source: string,
  payload: UpdatePayloadIdentity,
  policy: NativeVerificationPolicy,
  signal: AbortSignal,
  run: NativeCommandRunner = runNativeVerificationCommand,
): Promise<VerifiedUpdateArchive> {
  if (
    (process.platform !== 'darwin' && run === runNativeVerificationCommand) ||
    !/^[A-Za-z0-9.-]{1,128}$/u.test(policy.expectedAppId) ||
    !/^[A-Z0-9]{10}$/u.test(policy.expectedTeamId) ||
    !validBundleVersion(payload.bundleVersion) ||
    !validUpdatePayloadSizes(payload.size, payload.unpackedSize) ||
    !isStrictlyNewerVersion(payload.version, policy.currentVersion)
  )
    fail('APP_UPDATE_VERIFICATION_UNAVAILABLE');
  expectedDigest(payload.sha512);
  signal = AbortSignal.any([signal, AbortSignal.timeout(10 * 60_000)]);
  signal.throwIfAborted();
  await mkdir(policy.verificationRoot, { recursive: true, mode: 0o700 });
  const root = await realpath(policy.verificationRoot);
  if (root !== resolve(policy.verificationRoot)) fail('APP_UPDATE_VERIFICATION_UNAVAILABLE');
  const available = await statfs(root, { bigint: true });
  if (available.bavail * available.bsize < BigInt(payload.size + payload.unpackedSize + GiB))
    fail('APP_UPDATE_DISK_SPACE_LOW');
  const scratch = await mkdtemp(join(root, 'verify-'));
  let archive: VerifiedUpdateArchive | undefined;
  let retained: number | undefined;
  try {
    const input = await open(
      source,
      constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
    );
    const outputPath = join(scratch, 'payload.zip');
    const output = await open(outputPath, 'wx', 0o600).catch(async (error: unknown) => {
      await input.close();
      throw error;
    });
    try {
      const before = await input.stat({ bigint: true });
      if (!before.isFile() || before.size !== BigInt(payload.size))
        fail('APP_UPDATE_PAYLOAD_CHANGED');
      const hash = createHash('sha512');
      let bytes = 0;
      for await (const chunk of input.createReadStream({ autoClose: false })) {
        signal.throwIfAborted();
        bytes += chunk.length;
        if (bytes > payload.size) fail('APP_UPDATE_PAYLOAD_CHANGED');
        hash.update(chunk);
        await output.writeFile(chunk);
      }
      if (
        bytes !== payload.size ||
        !sameFile(before, await input.stat({ bigint: true })) ||
        !timingSafeEqual(hash.digest(), expectedDigest(payload.sha512))
      )
        fail('APP_UPDATE_PAYLOAD_CHANGED');
      await output.sync();
    } finally {
      await input.close();
      await output.close();
    }
    await chmod(outputPath, 0o400);
    retained = openSync(outputPath, constants.O_RDONLY | constants.O_NOFOLLOW);
    await unlink(outputPath);
    archive = new VerifiedUpdateArchive(retained, payload);
    retained = undefined;
    await inspectUpdateZip(archive, signal);
    const extracted = join(scratch, 'extracted');
    await mkdir(extracted, { mode: 0o700 });
    // macOS startup reads the root directory itself; this grants no descendants.
    const profile = `(version 1)(deny default)(allow process-exec (literal "/usr/bin/ditto"))(allow sysctl-read)(allow file-read* (literal "/") (subpath "/System") (subpath "/usr/lib") (subpath "/Library/Apple") (subpath "/dev/fd") (literal "/dev/null") (subpath ${JSON.stringify(scratch)}))(allow file-write* (subpath ${JSON.stringify(extracted)}))`;
    const commandOptions = {
      cwd: scratch,
      signal,
      archiveFd: archive.descriptor(),
    };
    await run(
      '/usr/bin/sandbox-exec',
      ['-p', profile, '/usr/bin/ditto', '-x', '-k', '/dev/fd/3', extracted],
      commandOptions,
    );
    const appPath = join(extracted, APP_NAME);
    if (
      (await realpath(appPath)) !== appPath ||
      relative(extracted, appPath).startsWith(`..${sep}`)
    )
      fail('APP_UPDATE_ARCHIVE_INVALID');
    await verifySignedMacApp(appPath, payload, policy, commandOptions, run);
    await archive.revalidate(signal);
    signal.throwIfAborted();
    return archive;
  } catch (error) {
    archive?.close();
    throw error;
  } finally {
    if (retained !== undefined) closeSync(retained);
    try {
      await rm(scratch, { recursive: true, force: true });
    } catch {
      archive?.close();
      fail('APP_UPDATE_VERIFICATION_CLEANUP_FAILED');
    }
  }
}
