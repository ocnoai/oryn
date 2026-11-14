import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFileSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import process from 'node:process';
import { test } from 'node:test';
import {
  assertDevelopmentIdentity,
  assertDevelopmentEngineCoverage,
  developmentAppId,
  developmentEntitlementKeys,
  developmentEnginePath,
  developmentMainExecutable,
  developmentProductName,
  developmentSigningOptions,
  scanDevelopmentBundle,
} from './sign-macos-development.mjs';
import { assertDevelopmentSignature } from './verify-macos-development.mjs';
import { createEngineInventory, verifyEnginePreservation } from './macos-development-engine.mjs';

const info = {
  CFBundleIdentifier: developmentAppId,
  CFBundleExecutable: developmentProductName,
  CFBundleName: developmentProductName,
};
const metadata = { orynBuildFlavor: 'local-development', orynUpdate: { enabled: false } };
const grants = Object.fromEntries(developmentEntitlementKeys.map((key) => [key, true]));
const signature = 'CodeDirectory flags=0x2(adhoc)\nSignature=adhoc\nTeamIdentifier=not set\n';

function fixture(t) {
  const directory = mkdtempSync(join(tmpdir(), 'oryn-development-sign-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const app = join(directory, `${developmentProductName}.app`);
  for (const path of [
    developmentMainExecutable,
    'Contents/Resources/engine/runtime/bin/python3.12',
  ]) {
    mkdirSync(join(app, path, '..'), { recursive: true });
    writeFileSync(join(app, path), Buffer.from([0xcf, 0xfa, 0xed, 0xfe]));
  }
  writeFileSync(join(app, 'Contents/Resources/inert.txt'), 'resource');
  return { directory, app };
}

test('development identity requires separate app name/ID, immutable flavor and disabled updater', () => {
  assertDevelopmentIdentity(info, metadata);
  for (const changed of [
    { ...info, CFBundleIdentifier: 'ai.ocno.oryn.browser' },
    { ...info, CFBundleExecutable: 'Oryn Browser' },
    { ...info, CFBundleName: 'Oryn Browser' },
  ])
    assert.throws(() => assertDevelopmentIdentity(changed, metadata), /separate app identity/u);
  for (const changed of [
    { ...metadata, orynBuildFlavor: 'release' },
    { ...metadata, orynUpdate: { enabled: true } },
    {},
  ])
    assert.throws(() => assertDevelopmentIdentity(info, changed), /separate app identity/u);
});

test('development signer preserves the entire Engine and clears flags only for application code', (t) => {
  const f = fixture(t);
  const options = developmentSigningOptions({
    app: f.app,
    identity: 'ambient-keychain-identity',
    optionsForFile: () => ({
      hardenedRuntime: true,
      entitlements: 'unsafe.plist',
      signatureFlags: ['runtime'],
    }),
  });
  assert.equal(options.identity, '-');
  assert.equal(options.preAutoEntitlements, false);
  assert.equal(options.preEmbedProvisioningProfile, false);
  for (const path of scanDevelopmentBundle(f.app).nativeFiles) {
    if (path.startsWith(join(options.app, developmentEnginePath))) {
      assert.equal(options.ignore(path), true);
      continue;
    }
    assert.equal(options.ignore(path), false);
    const policy = options.optionsForFile(path);
    assert.equal(policy.hardenedRuntime, false);
    assert.deepEqual(policy.signatureFlags, ['0x0']);
    assert.equal(policy.entitlements.endsWith('entitlements.mac.inherit.plist'), true);
    assert.equal(policy.timestamp, 'none');
  }
  assert.equal(options.ignore(join(f.app, 'Contents/Resources/inert.txt')), true);
  const lookalike = join(f.app, 'Contents/Resources/engine-not-preserved/library.dylib');
  mkdirSync(join(lookalike, '..'), { recursive: true });
  writeFileSync(lookalike, Buffer.from([0xcf, 0xfa, 0xed, 0xfe]));
  assert.equal(
    options.ignore(join(options.app, 'Contents/Resources/engine-not-preserved/library.dylib')),
    false,
  );
  const preserved = join(options.app, developmentEnginePath, 'runtime/bin/python3.12');
  assertDevelopmentEngineCoverage(options.app, [preserved], { nativeFiles: 1 });
  assert.throws(
    () =>
      assertDevelopmentEngineCoverage(
        options.app,
        [preserved, join(options.app, developmentEnginePath, '.DS_Store')],
        { nativeFiles: 1 },
      ),
    /outside the preserved/u,
  );
});

test(
  'Engine preservation binds unchanged original native bytes and all Python/capability data',
  { skip: process.platform !== 'darwin' },
  async (t) => {
    const f = fixture(t);
    const stage = join(f.directory, 'original-stage');
    mkdirSync(join(stage, 'bin'), { recursive: true });
    const original = join(stage, 'bin/python-fixture');
    copyFileSync('/usr/bin/true', original);
    const signed = spawnSync(
      '/usr/bin/codesign',
      ['--force', '--sign', '-', '--options', '0x0', '--timestamp=none', original],
      { encoding: 'utf8' },
    );
    assert.equal(signed.status, 0, signed.stderr);
    writeFileSync(join(stage, 'worker.py'), 'original reviewed source\n');
    writeFileSync(join(stage, 'capabilities.json'), '{"pinned":"unchanged"}\n');
    const inventory = await createEngineInventory(stage);
    const packaged = join(f.directory, 'packaged-engine');
    cpSync(stage, packaged, { recursive: true });
    const originalHash = createHash('sha256').update(readFileSync(original)).digest('hex');
    const proof = await verifyEnginePreservation(packaged, inventory);
    assert.equal(proof.payloadPreserved, true);
    assert.equal(proof.nativeFiles, 1);
    assert.equal(
      createHash('sha256')
        .update(readFileSync(join(packaged, 'bin/python-fixture')))
        .digest('hex'),
      originalHash,
    );
    writeFileSync(join(packaged, 'worker.py'), 'tampered Python source\n');
    await assert.rejects(
      () => verifyEnginePreservation(packaged, inventory),
      /differs from the original/u,
    );
    writeFileSync(join(packaged, 'worker.py'), readFileSync(join(stage, 'worker.py')));
    writeFileSync(join(packaged, 'capabilities.json'), '{"pinned":"silently rewritten"}\n');
    await assert.rejects(
      () => verifyEnginePreservation(packaged, inventory),
      /differs from the original/u,
    );
    writeFileSync(
      join(packaged, 'capabilities.json'),
      readFileSync(join(stage, 'capabilities.json')),
    );
    writeFileSync(
      join(packaged, 'bin/python-fixture'),
      Buffer.concat([readFileSync(original), Buffer.from('changed signature bytes')]),
    );
    await assert.rejects(
      () => verifyEnginePreservation(packaged, inventory),
      /differs from the original/u,
    );
  },
);

test(
  'unexpected hardened upstream Engine code is rejected instead of rewritten or accepted',
  { skip: process.platform !== 'darwin' },
  async (t) => {
    const f = fixture(t);
    const stage = join(f.directory, 'hardened-stage');
    mkdirSync(stage);
    const executable = join(stage, 'hardened-fixture');
    copyFileSync('/usr/bin/true', executable);
    const signed = spawnSync(
      '/usr/bin/codesign',
      ['--force', '--sign', '-', '--options', 'runtime', '--timestamp=none', executable],
      { encoding: 'utf8' },
    );
    assert.equal(signed.status, 0, signed.stderr);
    const inventory = await createEngineInventory(stage);
    await assert.rejects(() => verifyEnginePreservation(stage, inventory), /original non-runtime/u);
  },
);

test('Engine inventory rejects escaping symlinks and executable or non-file Finder metadata', async (t) => {
  const f = fixture(t);
  const stage = join(f.directory, 'bounded-stage');
  mkdirSync(stage);
  symlinkSync('/usr/bin/true', join(stage, 'escape'));
  await assert.rejects(() => createEngineInventory(stage), /Absolute development Engine symlink/u);
  rmSync(join(stage, 'escape'));
  writeFileSync(join(stage, '.DS_Store'), Buffer.from([0xcf, 0xfa, 0xed, 0xfe]));
  await assert.rejects(
    () => createEngineInventory(stage),
    /Unexpected executable or non-file Finder metadata/u,
  );
  rmSync(join(stage, '.DS_Store'));
  mkdirSync(join(stage, '.DS_Store'));
  await assert.rejects(
    () => createEngineInventory(stage),
    /Unexpected executable or non-file Finder metadata/u,
  );
});

test('development verifier rejects runtime/library flags, publisher identity and device/exception entitlements', () => {
  assertDevelopmentSignature(signature, grants);
  assertDevelopmentSignature(signature, {}, { executable: false });
  assert.throws(() => assertDevelopmentSignature(signature, {}), /exact JIT/u);
  for (const details of [
    signature.replace('0x2(adhoc)', '0x10002(adhoc,runtime)'),
    signature.replace('0x2(adhoc)', '0x2002(adhoc,library)'),
    signature.replace('not set', 'ABCDEFGHIJ'),
  ])
    assert.throws(() => assertDevelopmentSignature(details, grants), /no runtime/u);
  for (const key of [
    'com.apple.security.device.camera',
    'com.apple.security.device.audio-input',
    'com.apple.security.personal-information.location',
    'com.apple.security.cs.disable-library-validation',
  ]) {
    assert.throws(
      () => assertDevelopmentSignature(signature, { ...grants, [key]: true }),
      /no device/u,
    );
    assert.throws(
      () => assertDevelopmentSignature(signature, { [key]: true }, { executable: false }),
      /no device/u,
    );
  }
});

test('development scan rejects symlink escapes before signing', (t) => {
  const f = fixture(t);
  symlinkSync(f.directory, join(f.app, 'Contents/Resources/escape'));
  assert.throws(() => scanDevelopmentBundle(f.app), /escapes/u);
});

test(
  'installed codesign explicitly removes runtime from an already-hardened ad-hoc Mach-O',
  { skip: process.platform !== 'darwin' },
  (t) => {
    const directory = mkdtempSync(join(tmpdir(), 'oryn-codesign-zero-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    const executable = join(directory, 'fixture');
    copyFileSync('/usr/bin/true', executable);
    for (const flags of ['runtime', '0x0']) {
      const result = spawnSync(
        '/usr/bin/codesign',
        ['--force', '--sign', '-', '--options', flags, '--timestamp=none', executable],
        { encoding: 'utf8' },
      );
      assert.equal(result.status, 0, result.stderr);
      const display = spawnSync('/usr/bin/codesign', ['-d', '--verbose=4', executable], {
        encoding: 'utf8',
      });
      assert.equal(display.status, 0, display.stderr);
      if (flags === 'runtime') assert.match(display.stderr, /flags=.*\bruntime\b/u);
      else assert.match(display.stderr, /flags=0x2\(adhoc\)/u);
    }
    assert.equal(spawnSync('/usr/bin/codesign', ['--verify', '--strict', executable]).status, 0);
    assert.equal(spawnSync(executable).status, 0);
  },
);
