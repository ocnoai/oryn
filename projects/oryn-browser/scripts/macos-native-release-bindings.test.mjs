import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import {
  assertNativeReleaseUnchanged,
  completeReleaseBindings,
  nativeBindingName,
  releaseInputDigest,
  snapshotReleaseNativeInputs,
} from './macos-native-release-bindings.mjs';

function fixture(t) {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'oryn-release-binding-')));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const engineRoot = join(directory, 'engine');
  const native = join(engineRoot, 'runtime/bin/python3.12');
  const capabilities = join(
    engineRoot,
    'runtime/lib/python3.12/site-packages/oryn_engine/data/capabilities.json',
  );
  mkdirSync(join(native, '..'), { recursive: true });
  mkdirSync(join(capabilities, '..'), { recursive: true });
  writeFileSync(native, Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 1]));
  writeFileSync(capabilities, '{"reviewed":true}\n');
  return { directory, engineRoot, native, capabilities };
}

test('release binding retains the original native digest and binds final signed bytes', (t) => {
  const { engineRoot, native, capabilities } = fixture(t);
  const original = snapshotReleaseNativeInputs(engineRoot, [native]);
  const before = releaseInputDigest(native);
  writeFileSync(native, Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 1, 2]));
  const sealed = completeReleaseBindings(engineRoot, original, []);
  assert.equal(sealed.capabilitySha256, releaseInputDigest(capabilities));
  assert.deepEqual(sealed.files, [
    {
      path: 'runtime/bin/python3.12',
      originalSha256: before,
      signedSha256: releaseInputDigest(native),
    },
  ]);
  assert.notEqual(sealed.files[0].originalSha256, sealed.files[0].signedSha256);
});

test('native inventory rejects duplicates, non-Mach-O, escapes and symlink traversal', (t) => {
  const { directory, engineRoot, native, capabilities } = fixture(t);
  assert.throws(() => snapshotReleaseNativeInputs(engineRoot, [native, native]), /Duplicate/);
  assert.throws(() => snapshotReleaseNativeInputs(engineRoot, [capabilities]), /Mach-O/);
  assert.throws(
    () => snapshotReleaseNativeInputs(engineRoot, [join(directory, 'outside')]),
    /escapes/,
  );
  symlinkSync(join(engineRoot, 'runtime'), join(engineRoot, 'alias'));
  assert.throws(
    () => snapshotReleaseNativeInputs(engineRoot, [join(engineRoot, 'alias/bin/python3.12')]),
    /symlinks/,
  );
});

test('enclosing app signing cannot change native bytes or the authenticated map', (t) => {
  const { engineRoot, native } = fixture(t);
  const original = snapshotReleaseNativeInputs(engineRoot, [native]);
  const prepared = { engineRoot, original, sourceInputs: [] };
  const sealed = completeReleaseBindings(engineRoot, original, []);
  const mapPath = join(engineRoot, nativeBindingName, 'Contents/Resources/native-hashes.json');
  mkdirSync(join(mapPath, '..'), { recursive: true });
  writeFileSync(mapPath, JSON.stringify(sealed));
  assertNativeReleaseUnchanged(prepared, sealed);
  writeFileSync(mapPath, JSON.stringify({ ...sealed, capabilitySha256: '0'.repeat(64) }));
  assert.throws(() => assertNativeReleaseUnchanged(prepared, sealed), /binding changed/);
  writeFileSync(mapPath, JSON.stringify(sealed));
  writeFileSync(native, Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 3]));
  assert.throws(() => assertNativeReleaseUnchanged(prepared, sealed), /inputs changed/);
});

test('source records must describe the exact final bytes and cannot duplicate native inputs', (t) => {
  const { engineRoot, native, capabilities } = fixture(t);
  const original = snapshotReleaseNativeInputs(engineRoot, [native]);
  assert.throws(
    () =>
      completeReleaseBindings(engineRoot, original, [
        { ...original[0], signedSha256: '0'.repeat(64) },
      ]),
    /duplicate/,
  );
  const source = {
    path: 'runtime/lib/python3.12/site-packages/oryn_engine/data/capabilities.json',
    originalSha256: releaseInputDigest(capabilities),
    signedSha256: '0'.repeat(64),
  };
  assert.throws(() => completeReleaseBindings(engineRoot, original, [source]), /Invalid final/);
  assert.throws(
    () => completeReleaseBindings(engineRoot, [{ ...original[0], path: '../escape' }], []),
    /escapes/,
  );
});
