// @ts-nocheck
import test from 'node:test';
import assert from 'node:assert/strict';
import { crc32 } from 'node:zlib';
import { createVideoArchive } from '../src/video-archive.ts';

test('ZIP preserves video bytes, UTF-8 names, checksums and central-directory offsets', async () => {
  const content = new Uint8Array(1024 * 1024 + 7).map((_, i) => i % 251);
  const names = ['Príbeh.mp4', 'príbeh.mp4', '../other.webm'];
  const progress = [];
  const archive = await createVideoArchive(names.map(name => ({name, blob: new Blob([content])})), undefined, n => progress.push(n));
  assert.equal(archive.type, 'application/zip');
  const bytes = new Uint8Array(await archive.arrayBuffer()), view = new DataView(bytes.buffer);
  const end = bytes.length - 22;
  assert.equal(view.getUint32(end, true), 0x06054b50);
  assert.equal(view.getUint16(end + 10, true), 3);
  let cursor = view.getUint32(end + 16, true);
  const found = [];
  for (let i = 0; i < 3; i++) {
    assert.equal(view.getUint32(cursor, true), 0x02014b50);
    assert.equal(view.getUint16(cursor + 8, true), 0x800);
    assert.equal(view.getUint16(cursor + 10, true), 0);
    const local = view.getUint32(cursor + 42, true), length = view.getUint16(cursor + 28, true);
    assert.equal(view.getUint32(local, true), 0x04034b50);
    assert.equal(view.getUint32(local + 14, true), crc32(content));
    assert.equal(view.getUint32(cursor + 16, true), crc32(content));
    assert.equal(view.getUint32(local + 18, true), content.length);
    const name = new TextDecoder().decode(bytes.slice(cursor + 46, cursor + 46 + length));
    found.push(name);
    assert.equal(new TextDecoder().decode(bytes.slice(local + 30, local + 30 + length)), name);
    assert.deepEqual(bytes.slice(local + 30 + length, local + 30 + length + content.length), content);
    cursor += 46 + length;
  }
  assert.equal(cursor, end);
  assert.deepEqual(found, ['Príbeh.mp4', 'príbeh-2.mp4', '-other.webm']);
  assert.deepEqual(progress, [1, 2, 3]);
});

test('ZIP handles suffix collisions, empty files and rejects empty batches', async () => {
  const archive = await createVideoArchive(['a.mp4', 'a-2.mp4', 'a.mp4'].map(name => ({name, blob: new Blob([])})));
  const text = await archive.text();
  assert.ok(text.includes('a-3.mp4'));
  await assert.rejects(createVideoArchive([]), /Choose between/);
});

test('ZIP supports cancellation before and between files', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(createVideoArchive([{name: 'a.mp4', blob: new Blob(['test'])}], controller.signal), {name: 'AbortError'});
  const active = new AbortController();
  await assert.rejects(createVideoArchive(['a', 'b'].map(name => ({name, blob: new Blob(['test'])})), active.signal, () => active.abort()), {name: 'AbortError'});
});
