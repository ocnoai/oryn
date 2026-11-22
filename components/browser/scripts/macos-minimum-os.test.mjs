import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import {
  assertNativeMinimumOS,
  compareMacOSVersions,
  parseMacOSVersion,
  readMachOMinimumOS,
} from './macos-minimum-os.mjs';

const root = mkdtempSync(join(tmpdir(), 'oryn-native-minimum-os-'));
after(() => rmSync(root, { recursive: true, force: true }));
let counter = 0;
function fixture(bytes) {
  const path = join(root, `${counter++}.node`);
  writeFileSync(path, bytes);
  return path;
}
function thin({ major = 14, minor = 0, cpu = 0x0100000c, platform = 1, legacy = false } = {}) {
  const bytes = Buffer.alloc(legacy ? 48 : 56);
  bytes.writeUInt32LE(0xfeedfacf, 0);
  bytes.writeUInt32LE(cpu, 4);
  bytes.writeUInt32LE(8, 12);
  bytes.writeUInt32LE(1, 16);
  bytes.writeUInt32LE(bytes.length - 32, 20);
  bytes.writeUInt32LE(legacy ? 0x24 : 0x32, 32);
  bytes.writeUInt32LE(bytes.length - 32, 36);
  if (!legacy) bytes.writeUInt32LE(platform, 40);
  bytes.writeUInt32LE((major << 16) | (minor << 8), legacy ? 40 : 44);
  return bytes;
}
function fat({ wide = false, cpu = 0x0100000c, overlap = false } = {}) {
  const x64 = thin({ cpu: 0x01000007, major: 30 }),
    arm = thin({ major: 26, minor: 2 });
  const bytes = Buffer.alloc(512 + arm.length),
    entry = wide ? 32 : 20;
  bytes.writeUInt32BE(wide ? 0xcafebabf : 0xcafebabe, 0);
  bytes.writeUInt32BE(2, 4);
  for (const [index, type, offset, length] of [
    [0, 0x01000007, 256, x64.length],
    [1, cpu, overlap ? 256 : 512, arm.length],
  ]) {
    const at = 8 + index * entry;
    bytes.writeUInt32BE(type, at);
    if (wide) {
      bytes.writeBigUInt64BE(BigInt(offset), at + 8);
      bytes.writeBigUInt64BE(BigInt(length), at + 16);
    } else {
      bytes.writeUInt32BE(offset, at + 8);
      bytes.writeUInt32BE(length, at + 12);
    }
  }
  x64.copy(bytes, 256);
  arm.copy(bytes, 512);
  return bytes;
}
test('compares macOS versions numerically and validates bounded declaration strings', () => {
  assert.equal(compareMacOSVersions('14', '14.0.0'), 0);
  assert.equal(compareMacOSVersions('26.2', '14.10'), 1);
  for (const value of ['0', '14.x', '14.0.0.1', ' 14.0', '14.0\n', '999999'])
    assert.throws(() => parseMacOSVersion(value));
});
test('reads macOS thin build-version and legacy minimum commands', () => {
  assert.equal(readMachOMinimumOS(fixture(thin())), '14.0.0');
  assert.equal(readMachOMinimumOS(fixture(thin({ legacy: true, major: 13, minor: 3 }))), '13.3.0');
});
test('selects arm64 from both bounded fat table formats, without claiming x86 qualification', () => {
  for (const wide of [false, true])
    assert.equal(readMachOMinimumOS(fixture(fat({ wide }))), '26.2.0');
});
test('fails declared macOS 14 for retained 26.2 native bytes, accepts an honestly declared 26.2 floor', () => {
  const files = [fixture(thin()), fixture(thin({ major: 26, minor: 2 }))];
  assert.throws(() => assertNativeMinimumOS('14.0', files), /below retained native floor 26.2.0/u);
  assert.equal(assertNativeMinimumOS('26.2', files).nativeFloor, '26.2.0');
});
test('rejects wrong architecture/platform, missing minimums and malformed command bounds', () => {
  for (const bytes of [thin({ cpu: 0x01000007 }), thin({ platform: 2 }), Buffer.alloc(16)])
    assert.throws(() => readMachOMinimumOS(fixture(bytes)));
  for (const [offset, value] of [
    [16, 0],
    [16, 20000],
    [20, 2 ** 21],
    [36, 8],
    [32, 1],
    [52, 65],
  ]) {
    const bytes = thin();
    bytes.writeUInt32LE(value, offset);
    assert.throws(() => readMachOMinimumOS(fixture(bytes)));
  }
});
test('rejects overlapping/out-of-bounds fat slices and a mismatched embedded architecture', () => {
  assert.throws(() => readMachOMinimumOS(fixture(fat({ overlap: true }))));
  const beyond = fat();
  beyond.writeUInt32BE(0xffffffff, 8 + 20 + 8);
  assert.throws(() => readMachOMinimumOS(fixture(beyond)));
  const mismatch = fat();
  mismatch.writeUInt32LE(0x01000007, 512 + 4);
  assert.throws(() => readMachOMinimumOS(fixture(mismatch)));
  const duplicate = thin();
  duplicate.writeUInt32LE(2, 16);
  duplicate.writeUInt32LE(48, 20);
  assert.throws(() =>
    readMachOMinimumOS(fixture(Buffer.concat([duplicate, duplicate.subarray(32)]))),
  );
});
