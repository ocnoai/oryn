import { closeSync, constants, fstatSync, openSync, readSync } from 'node:fs';
import { Buffer } from 'node:buffer';

const ARM64 = 0x0100000c;
const MAX_COMMAND_BYTES = 1024 * 1024;
const MAX_COMMANDS = 16384;
const MAX_FAT_SLICES = 64;

function requireCondition(value, message) {
  if (!value) throw new Error(message);
}
export function parseMacOSVersion(value) {
  requireCondition(
    typeof value === 'string' && /^\d{1,5}(?:\.\d{1,5}){0,2}$/u.test(value),
    'Invalid macOS version.',
  );
  const result = value.split('.').map(Number);
  while (result.length < 3) result.push(0);
  requireCondition(
    result[0] > 0 && result.every((part) => part <= 65535),
    'Invalid macOS version component.',
  );
  return result;
}
export function compareMacOSVersions(left, right) {
  const a = parseMacOSVersion(left),
    b = parseMacOSVersion(right);
  for (let index = 0; index < 3; index++)
    if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
  return 0;
}
function packedVersion(value) {
  const result = `${value >>> 16}.${(value >>> 8) & 255}.${value & 255}`;
  parseMacOSVersion(result);
  return result;
}

/** Parse only bounded headers/load commands, without mapping or executing a native payload.
 * The current release target is arm64. Inactive x86 slices do not establish Intel qualification.
 */
export function readMachOMinimumOS(path) {
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const initial = fstatSync(fd, { bigint: true });
    requireCondition(
      initial.isFile() && initial.size >= 32n && initial.size <= BigInt(Number.MAX_SAFE_INTEGER),
      'Invalid native file.',
    );
    const size = Number(initial.size);
    function read(offset, length) {
      requireCondition(
        Number.isSafeInteger(offset) &&
          offset >= 0 &&
          Number.isSafeInteger(length) &&
          length >= 0 &&
          length <= MAX_COMMAND_BYTES &&
          offset <= size - length,
        'Native header read exceeds bounds.',
      );
      const buffer = Buffer.alloc(length);
      let done = 0;
      while (done < length) {
        const received = readSync(fd, buffer, done, length - done, offset + done);
        requireCondition(received > 0, 'Truncated native header.');
        done += received;
      }
      return buffer;
    }
    const header = read(0, 8);
    const magic = header.readUInt32BE(0);
    let slices;
    if ([0xcafebabe, 0xbebafeca, 0xcafebabf, 0xbfbafeca].includes(magic)) {
      const little = magic === 0xbebafeca || magic === 0xbfbafeca;
      const wide = magic === 0xcafebabf || magic === 0xbfbafeca;
      const uint32 = (buffer, offset) =>
        little ? buffer.readUInt32LE(offset) : buffer.readUInt32BE(offset);
      const uint64 = (buffer, offset) => {
        const result = little ? buffer.readBigUInt64LE(offset) : buffer.readBigUInt64BE(offset);
        requireCondition(
          result <= BigInt(Number.MAX_SAFE_INTEGER),
          'Unsafe native slice offset/length.',
        );
        return Number(result);
      };
      const count = uint32(header, 4),
        entrySize = wide ? 32 : 20;
      requireCondition(count > 0 && count <= MAX_FAT_SLICES, 'Invalid native fat-slice count.');
      const tableSize = 8 + count * entrySize,
        table = read(8, count * entrySize);
      const all = [];
      for (let index = 0; index < count; index++) {
        const at = index * entrySize;
        const offset = wide ? uint64(table, at + 8) : uint32(table, at + 8);
        const length = wide ? uint64(table, at + 16) : uint32(table, at + 12);
        const cpu = uint32(table, at),
          subtype = uint32(table, at + 4);
        requireCondition(
          offset >= tableSize && length >= 32 && offset <= size - length,
          'Native fat slice exceeds file bounds.',
        );
        requireCondition(
          !all.some(
            (slice) =>
              (slice.cpu === cpu && slice.subtype === subtype) ||
              (offset < slice.offset + slice.length && slice.offset < offset + length),
          ),
          'Duplicate or overlapping native fat slices.',
        );
        all.push({ offset, length, cpu, subtype });
      }
      slices = all.filter((slice) => slice.cpu === ARM64);
      requireCondition(slices.length > 0, 'Native payload has no arm64 slice.');
    } else slices = [{ offset: 0, length: size }];
    const minimums = [];
    for (const slice of slices) {
      const mach = read(slice.offset, 32);
      // Apple Silicon macOS uses 64-bit little-endian Mach-O.
      requireCondition(
        mach.readUInt32LE(0) === 0xfeedfacf && mach.readUInt32LE(4) === ARM64,
        'Unsupported native architecture/header.',
      );
      if (slice.cpu !== undefined)
        requireCondition(
          mach.readUInt32LE(8) === slice.subtype,
          'Native fat/header architecture mismatch.',
        );
      const count = mach.readUInt32LE(16),
        bytes = mach.readUInt32LE(20);
      requireCondition(
        count > 0 &&
          count <= MAX_COMMANDS &&
          bytes >= count * 8 &&
          bytes <= MAX_COMMAND_BYTES &&
          bytes <= slice.length - 32,
        'Invalid native load-command bounds.',
      );
      const commands = read(slice.offset + 32, bytes);
      let offset = 0,
        minimum;
      for (let index = 0; index < count; index++) {
        requireCondition(offset <= bytes - 8, 'Truncated native load command.');
        const type = commands.readUInt32LE(offset),
          length = commands.readUInt32LE(offset + 4);
        requireCondition(
          length >= 8 && length % 8 === 0 && length <= bytes - offset,
          'Invalid native load-command size.',
        );
        if (type === 0x32) {
          requireCondition(
            length >= 24 && commands.readUInt32LE(offset + 8) === 1,
            'Native payload is not a macOS platform build.',
          );
          const tools = commands.readUInt32LE(offset + 20);
          requireCondition(
            tools <= 64 && length === 24 + tools * 8 && minimum === undefined,
            'Invalid/duplicate native build-version command.',
          );
          minimum = packedVersion(commands.readUInt32LE(offset + 12));
        } else if (type === 0x24) {
          requireCondition(
            length === 16 && minimum === undefined,
            'Invalid/duplicate native minimum-version command.',
          );
          minimum = packedVersion(commands.readUInt32LE(offset + 8));
        } else if ([0x25, 0x2f, 0x30].includes(type)) {
          throw new Error('Native payload declares a non-macOS minimum platform.');
        }
        offset += length;
      }
      requireCondition(
        offset === bytes && minimum !== undefined,
        'Missing or inconsistent native minimum-version commands.',
      );
      minimums.push(minimum);
    }
    const final = fstatSync(fd, { bigint: true });
    requireCondition(
      initial.dev === final.dev &&
        initial.ino === final.ino &&
        initial.size === final.size &&
        initial.mtimeNs === final.mtimeNs &&
        initial.ctimeNs === final.ctimeNs,
      'Native file changed during header validation.',
    );
    return minimums.reduce((left, right) =>
      compareMacOSVersions(left, right) >= 0 ? left : right,
    );
  } finally {
    closeSync(fd);
  }
}

export function inspectNativeMinimumOS(nativeFiles) {
  requireCondition(
    Array.isArray(nativeFiles) &&
      nativeFiles.length > 0 &&
      nativeFiles.length <= 8192 &&
      new Set(nativeFiles).size === nativeFiles.length,
    'Invalid native minimum-version input set.',
  );
  const files = nativeFiles.map((path) => ({ path, minimum: readMachOMinimumOS(path) }));
  return {
    architecture: 'arm64',
    files,
    minimum: files
      .map((entry) => entry.minimum)
      .reduce((left, right) => (compareMacOSVersions(left, right) >= 0 ? left : right)),
  };
}
export function assertNativeMinimumOS(declared, nativeFiles) {
  parseMacOSVersion(declared);
  const inspected = inspectNativeMinimumOS(nativeFiles);
  const exceeding = inspected.files.filter(
    (entry) => compareMacOSVersions(entry.minimum, declared) > 0,
  );
  requireCondition(
    exceeding.length === 0,
    `Declared macOS ${declared} is below retained native floor ${inspected.minimum}: ${exceeding.map((entry) => entry.path).join(', ')}`,
  );
  return {
    architecture: inspected.architecture,
    declared,
    nativeFloor: inspected.minimum,
    nativeFiles: inspected.files.length,
  };
}
