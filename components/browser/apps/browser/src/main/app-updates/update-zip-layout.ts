import { readSync } from 'node:fs';
import { getFileNameLowLevel, parseExtraFields, type Entry, type LocalFileHeader } from 'yauzl';

export const MAX_UPDATE_ZIP_ENTRIES = 250_000;
const MAX_CENTRAL_BYTES = 256 * 1024 * 1024;

function invalid(): never {
  throw new Error('APP_UPDATE_ARCHIVE_INVALID');
}

export function checkUnicodePathFields(
  fields: Array<{ id: number; data: Buffer }>,
  raw: Buffer,
): void {
  for (const field of fields) {
    if (
      field.id === 0x7075 &&
      (field.data.length < 5 || field.data[0] !== 1 || !field.data.subarray(5).equals(raw))
    )
      invalid();
  }
}

function read(fd: number, size: number, position: number, length: number): Buffer {
  if (!Number.isSafeInteger(position) || position < 0 || length < 0 || position + length > size)
    invalid();
  const data = Buffer.alloc(length);
  if (readSync(fd, data, 0, length, position) !== length) invalid();
  return data;
}

function uint64(value: bigint): number {
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result < 0) invalid();
  return result;
}

/** Walk the complete central-directory byte range, not the classic 16-bit count.
 * macOS ditto can wrap that count instead of emitting a ZIP64 locator. Only that
 * exact modulo form is accepted, after validating every central record. */
export async function checkedZipDirectory(fd: number, size: number, signal: AbortSignal) {
  const start = Math.max(0, size - 65_557);
  const tail = read(fd, size, start, size - start);
  let end = -1;
  for (let index = tail.length - 22; index >= 0; index--) {
    if (
      tail.readUInt32LE(index) === 0x06054b50 &&
      index + 22 + tail.readUInt16LE(index + 20) === tail.length
    ) {
      end = start + index;
      break;
    }
  }
  if (end < 0) invalid();
  const record = read(fd, size, end, 22);
  if (
    record.readUInt16LE(4) !== 0 ||
    record.readUInt16LE(6) !== 0 ||
    record.readUInt16LE(8) !== record.readUInt16LE(10)
  )
    invalid();
  let announced = record.readUInt16LE(10);
  let bytes = record.readUInt32LE(12);
  let offset = record.readUInt32LE(16);
  let terminator = end;
  let zip64 = false;
  if (end >= 20 && read(fd, size, end - 20, 4).readUInt32LE() === 0x07064b50) {
    zip64 = true;
    const locator = read(fd, size, end - 20, 20);
    if (locator.readUInt32LE(4) !== 0 || locator.readUInt32LE(16) !== 1) invalid();
    terminator = uint64(locator.readBigUInt64LE(8));
    const wide = read(fd, size, terminator, 56);
    const wideBytes = uint64(wide.readBigUInt64LE(4));
    if (
      wide.readUInt32LE() !== 0x06064b50 ||
      wideBytes < 44 ||
      wideBytes > 1024 * 1024 ||
      terminator + 12 + wideBytes !== end - 20 ||
      wide.readUInt32LE(16) !== 0 ||
      wide.readUInt32LE(20) !== 0 ||
      wide.readBigUInt64LE(24) !== wide.readBigUInt64LE(32)
    )
      invalid();
    announced = uint64(wide.readBigUInt64LE(32));
    bytes = uint64(wide.readBigUInt64LE(40));
    offset = uint64(wide.readBigUInt64LE(48));
  } else if (bytes === 0xffffffff || offset === 0xffffffff) invalid();
  if (bytes < 46 || bytes > MAX_CENTRAL_BYTES || offset < 1 || offset + bytes !== terminator)
    invalid();
  let cursor = offset;
  let count = 0;
  const limit = offset + bytes;
  while (cursor < limit) {
    signal.throwIfAborted();
    if (++count > MAX_UPDATE_ZIP_ENTRIES) invalid();
    const item = read(fd, size, cursor, 46);
    const name = item.readUInt16LE(28);
    if (item.readUInt32LE() !== 0x02014b50 || !name || name > 1024 || item.readUInt16LE(34) !== 0)
      invalid();
    cursor += 46 + name + item.readUInt16LE(30) + item.readUInt16LE(32);
    if (cursor > limit) invalid();
    if (count % 256 === 0) await new Promise<void>((done) => setImmediate(done));
  }
  if (
    !count ||
    (count !== announced && (zip64 || count <= 0xffff || announced !== count % 0x10000))
  )
    invalid();
  return { offset, end: limit, count };
}

/** Central/local names, flags and physical coverage must agree. Otherwise a
 * streaming native extractor could see hidden members omitted by a ZIP reader. */
export function checkedLocalRange(
  fd: number,
  size: number,
  centralOffset: number,
  entry: Entry,
  header: LocalFileHeader,
): [number, number] {
  checkUnicodePathFields(entry.extraFields, entry.fileNameRaw);
  checkUnicodePathFields(parseExtraFields(header.extraField), header.fileName);
  if (
    !header.fileName.equals(entry.fileNameRaw) ||
    getFileNameLowLevel(
      header.generalPurposeBitFlag,
      header.fileName,
      parseExtraFields(header.extraField),
      true,
    ) !== entry.fileName ||
    header.generalPurposeBitFlag !== entry.generalPurposeBitFlag ||
    header.compressionMethod !== entry.compressionMethod
  )
    invalid();
  const begin = entry.relativeOffsetOfLocalHeader;
  let end = header.fileDataStart + entry.compressedSize;
  if (
    !Number.isSafeInteger(begin) ||
    begin < 0 ||
    begin + 30 > header.fileDataStart ||
    end > centralOffset ||
    end < header.fileDataStart
  )
    invalid();
  if (entry.generalPurposeBitFlag & 8) {
    const wide =
      header.compressedSize === 0xffffffff ||
      header.uncompressedSize === 0xffffffff ||
      entry.compressedSize >= 0xffffffff ||
      entry.uncompressedSize >= 0xffffffff;
    const length = wide ? 20 : 12;
    const value = read(fd, size, end, Math.min(length + 4, centralOffset - end));
    if (value.length < length) invalid();
    let match: number | undefined;
    for (const skip of [0, 4]) {
      if (skip === 4 && value.readUInt32LE() !== 0x08074b50) continue;
      if (value.length < skip + length || value.readUInt32LE(skip) !== entry.crc32) continue;
      const compressed = wide
        ? uint64(value.readBigUInt64LE(skip + 4))
        : value.readUInt32LE(skip + 4);
      const unpacked = wide
        ? uint64(value.readBigUInt64LE(skip + 12))
        : value.readUInt32LE(skip + 8);
      if (compressed === entry.compressedSize && unpacked === entry.uncompressedSize) {
        if (match !== undefined) invalid();
        match = skip + length;
      }
    }
    if (match === undefined) invalid();
    end += match;
  } else if (
    header.compressedSize !== entry.compressedSize ||
    header.uncompressedSize !== entry.uncompressedSize ||
    header.crc32 !== entry.crc32
  )
    invalid();
  return [begin, end];
}

export async function checkSymlinkGraph(
  links: Map<string, string>,
  appName: string,
  signal: AbortSignal,
): Promise<void> {
  const canonical = (value: string) => value.normalize('NFC').toLowerCase();
  const root = canonical(appName);
  let count = 0;
  for (const link of links.keys()) {
    signal.throwIfAborted();
    const pending = link.split('/').reverse();
    const resolved: string[] = [];
    const visited = new Set<string>();
    let steps = 0;
    let expanded = link.length;
    while (pending.length) {
      if (++steps > 8192) invalid();
      const part = pending.pop()!;
      if (!part || part === '.') continue;
      if (part === '..') {
        if (resolved.length <= 1) invalid();
        resolved.pop();
        continue;
      }
      resolved.push(canonical(part));
      if (resolved[0] !== root) invalid();
      const path = resolved.join('/');
      const target = links.get(path);
      if (target !== undefined) {
        if (visited.has(path) || visited.size >= 40) invalid();
        visited.add(path);
        resolved.pop();
        expanded += target.length;
        if (expanded > 8192) invalid();
        pending.push(...target.split('/').reverse());
      }
    }
    if (++count % 256 === 0) await new Promise<void>((done) => setImmediate(done));
  }
}

export function checkLocalCoverage(ranges: Array<[number, number]>, centralOffset: number): void {
  ranges.sort((left, right) => left[0] - right[0]);
  let cursor = 0;
  for (const [start, end] of ranges) {
    if (start !== cursor || end < start) invalid();
    cursor = end;
  }
  if (cursor !== centralOffset) invalid();
}
