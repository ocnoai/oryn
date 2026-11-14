export interface ArchiveEntry { name: string; blob: Blob }

/** Store already-compressed videos without another compression/memory pass. */
export async function createVideoArchive(entries: ArchiveEntry[], signal?: AbortSignal, onProgress?: (completed: number) => void): Promise<Blob> {
  if (!entries.length || entries.length > 65_535) throw new Error('Choose between 1 and 65,535 videos to download.');
  const parts: BlobPart[] = [], directory: Uint8Array<ArrayBuffer>[] = [];
  const names = new Set<string>();
  const encoder = new TextEncoder();
  let offset = 0;
  for (const [index, entry] of entries.entries()) {
    signal?.throwIfAborted();
    const safe = entry.name.replace(/[\\/\x00-\x1f]/gu, '-').replace(/^\.+/u, '') || 'video.mp4';
    const dot = safe.lastIndexOf('.');
    const base = dot > 0 ? safe.slice(0, dot) : safe, extension = dot > 0 ? safe.slice(dot) : '';
    let name = safe, suffix = 2;
    while (names.has(name.toLowerCase())) name = `${base}-${suffix++}${extension}`;
    names.add(name.toLowerCase());
    const filename = encoder.encode(name);
    if (filename.length > 65_535 || entry.blob.size + offset + filename.length + 30 >= 0xffffffff)
      throw new Error('This ZIP exceeds the 4 GB download limit. Save videos individually or use a smaller batch.');
    const crc = await checksum(entry.blob, signal);
    const local = new Uint8Array(30 + filename.length), header = new DataView(local.buffer);
    header.setUint32(0, 0x04034b50, true);
    header.setUint16(4, 20, true);
    header.setUint16(6, 0x800, true); // UTF-8; stored (uncompressed).
    header.setUint16(12, 0x21, true); // 1980-01-01 DOS date.
    header.setUint32(14, crc, true);
    header.setUint32(18, entry.blob.size, true);
    header.setUint32(22, entry.blob.size, true);
    header.setUint16(26, filename.length, true);
    local.set(filename, 30);
    const central = new Uint8Array(46 + filename.length), record = new DataView(central.buffer);
    record.setUint32(0, 0x02014b50, true);
    record.setUint16(4, 20, true);
    central.set(local.subarray(4, 30), 6);
    record.setUint32(42, offset, true);
    central.set(filename, 46);
    parts.push(local, entry.blob);
    directory.push(central);
    offset += local.length + entry.blob.size;
    onProgress?.(index + 1);
  }
  const size = directory.reduce((total, entry) => total + entry.length, 0);
  if (offset + size + 22 >= 0xffffffff) throw new Error('This ZIP exceeds the 4 GB download limit.');
  const end = new Uint8Array(22), record = new DataView(end.buffer);
  record.setUint32(0, 0x06054b50, true);
  record.setUint16(8, entries.length, true);
  record.setUint16(10, entries.length, true);
  record.setUint32(12, size, true);
  record.setUint32(16, offset, true);
  signal?.throwIfAborted();
  return new Blob([...parts, ...directory, end], {type: 'application/zip'});
}

const table = Uint32Array.from({length: 256}, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
  return value >>> 0;
});
async function checksum(blob: Blob, signal?: AbortSignal): Promise<number> {
  let crc = 0xffffffff;
  // Bounded slices keep checksum work cancellable, even for a large output.
  for (let offset = 0; offset < blob.size; offset += 1024 * 1024) {
    signal?.throwIfAborted();
    const bytes = new Uint8Array(await blob.slice(offset, offset + 1024 * 1024).arrayBuffer());
    for (const byte of bytes) crc = table[(crc ^ byte) & 255]! ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
