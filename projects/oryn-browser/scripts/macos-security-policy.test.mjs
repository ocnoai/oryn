import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { FuseState, FuseV1Options } from './macos-packaging-dependencies.mjs';
import YAML from 'yaml';
import {
  assertDocumentAssociations,
  assertEntitlements,
  assertFusePolicy,
  assertPrivacyDescriptions,
  browserRoot,
  deviceEntitlements,
  entitlementFile,
  isExecutableMachO,
  expectedEntitlements,
  fusePolicy,
  mainExecutable,
  localSigningEnvironment,
  relativeBundlePath,
  scanBundle,
  usageDescriptions,
} from './macos-security-policy.mjs';
import { signingOptions } from './sign-macos.mjs';

function wire(variant = 'release') {
  return Object.fromEntries(
    Object.entries(fusePolicy(variant))
      .filter(([key]) => key === 'version' || /^\d+$/u.test(key))
      .map(([key, value]) => [
        key,
        key === 'version' ? value : value ? FuseState.ENABLE : FuseState.DISABLE,
      ]),
  );
}

function fixture(t) {
  const directory = mkdtempSync(join(tmpdir(), 'oryn-macos-policy-'));
  t.after(() => rmSync(directory, { force: true, recursive: true }));
  const app = join(directory, 'Oryn Browser.app');
  const paths = [
    mainExecutable,
    'Contents/Resources/engine/runtime/bin/python3.12',
    'Contents/Frameworks/Oryn Browser Helper.app/Contents/MacOS/Oryn Browser Helper',
  ];
  for (const path of paths) {
    mkdirSync(join(app, path, '..'), { recursive: true });
    writeFileSync(join(app, path), Buffer.from([0xcf, 0xfa, 0xed, 0xfe]));
  }
  writeFileSync(join(app, 'Contents/Resources/inert.txt'), 'inert resource');
  return { app, directory, paths };
}

test('release rejects inspector, removed/unknown fuses and schema drift', () => {
  assertFusePolicy(wire());
  assert.throws(() => assertFusePolicy(wire('inspect-test')), /EnableNodeCliInspectArguments/u);
  const removed = wire();
  removed[FuseV1Options.RunAsNode] = FuseState.REMOVED;
  assert.throws(() => assertFusePolicy(removed), /RunAsNode/u);
  assert.throws(() => assertFusePolicy({ ...wire(), 9: FuseState.ENABLE }), /schema/u);
  assert.throws(() => assertFusePolicy({ ...wire(), version: '2' }), /schema/u);
  const missing = wire();
  delete missing[FuseV1Options.WasmTrapHandlers];
  assert.throws(() => assertFusePolicy(missing), /schema/u);
  assert.throws(() => fusePolicy('development'), /variant/u);
});

test('explicit inspector lane differs in exactly the inspect fuse', () => {
  const release = fusePolicy();
  const inspect = fusePolicy('inspect-test');
  assert.deepEqual(
    Object.keys(release).filter((key) => release[key] !== inspect[key]),
    [String(FuseV1Options.EnableNodeCliInspectArguments)],
  );
  assertFusePolicy(wire('inspect-test'), 'inspect-test');
});

test('local packaging cannot import signing credentials or use an ambient identity', () => {
  const environment = localSigningEnvironment(
    {
      PATH: '/safe/bin',
      CSC_LINK: 'credential-fixture',
      CSC_KEY_PASSWORD: 'password-fixture',
      APPLE_KEYCHAIN_PROFILE: 'profile-fixture',
      ORYN_APPLE_TEAM_ID: 'team-fixture',
      ORYN_MAC_SIGNING_MODE: 'release',
    },
    'inspect-test',
  );
  assert.deepEqual(environment, {
    PATH: '/safe/bin',
    CSC_IDENTITY_AUTO_DISCOVERY: 'false',
    ORYN_MAC_SIGNING_MODE: 'ad-hoc',
    ORYN_MAC_SECURITY_VARIANT: 'inspect-test',
  });
});

test('device entitlements are exact-main-only and never inherited by Engine or helper lookalikes', () => {
  for (const path of ['', mainExecutable]) {
    assertEntitlements(
      path,
      Object.fromEntries(expectedEntitlements(path).map((key) => [key, true])),
    );
  }
  for (const path of [
    'Contents/Resources/engine/runtime/bin/python3.12',
    'Contents/Frameworks/Oryn Browser Helper.app/Contents/MacOS/Oryn Browser Helper',
    `Contents/Resources/engine/${mainExecutable}`,
    `${mainExecutable}.untrusted`,
  ]) {
    const entitlements = Object.fromEntries(expectedEntitlements(path).map((key) => [key, true]));
    assert.equal(
      deviceEntitlements.some((key) => key in entitlements),
      false,
    );
    assert.throws(
      () => assertEntitlements(path, { ...entitlements, [deviceEntitlements[0]]: true }),
      /Unexpected entitlements/u,
    );
  }
  assert.throws(
    () => relativeBundlePath('/bundle.app', '/bundle.app.evil/Contents/MacOS/Oryn Browser'),
    /escapes/u,
  );
  assert.throws(() => entitlementFile('/bundle.app', '/another.app'), /escapes/u);
  assertEntitlements('Contents/Frameworks/library.dylib', {}, { executable: false });
  assert.throws(
    () => assertEntitlements('Contents/Resources/engine/python', {}),
    /Unexpected entitlements/u,
  );
  assert.throws(
    () =>
      assertEntitlements(
        'Contents/Frameworks/library.dylib',
        { [deviceEntitlements[0]]: true },
        { executable: false },
      ),
    /Unexpected entitlements/u,
  );
});

test('Mach-O executable detection distinguishes process code from libraries', (t) => {
  const f = fixture(t);
  const path = join(f.app, mainExecutable);
  const header = Buffer.alloc(32);
  header.writeUInt32BE(0xcffaedfe);
  header.writeUInt32LE(2, 12);
  writeFileSync(path, header);
  assert.equal(isExecutableMachO(path), true);
  header.writeUInt32LE(6, 12);
  writeFileSync(path, header);
  assert.equal(isExecutableMachO(path), false);
});

test('signer preserves non-entitlement options and does not default to osx-sign device grants', (t) => {
  const f = fixture(t);
  const options = signingOptions({
    app: f.app,
    identity: 'real-identity',
    optionsForFile: () => ({
      timestamp: 'https://timestamp.apple.com/ts01',
      requirements: '=test',
      entitlements: 'unsafe.plist',
      hardenedRuntime: false,
    }),
  });
  const engine = options.optionsForFile(join(f.app, f.paths[1]));
  assert.equal(engine.entitlements.endsWith('entitlements.mac.inherit.plist'), true);
  assert.equal(engine.hardenedRuntime, true);
  assert.equal(engine.requirements, '=test');
  assert.equal(options.identity, 'real-identity');
  assert.equal(options.preAutoEntitlements, false);
  assert.equal(options.ignore(join(f.app, 'Contents/Resources/inert.txt')), true);
  const local = signingOptions({ app: f.app }, { adHoc: true });
  assert.equal(local.identity, '-');
  assert.equal(local.optionsForFile(join(f.app, mainExecutable)).timestamp, 'none');
});

test('bundle scanner sees all Mach-O files without following symlinks or allowing escapes', (t) => {
  const f = fixture(t);
  symlinkSync('python3.12', join(f.app, 'Contents/Resources/engine/runtime/bin/python'));
  assert.equal(scanBundle(f.app).nativeFiles.length, 3);
  symlinkSync(f.directory, join(f.app, 'Contents/Resources/escape'));
  assert.throws(() => scanBundle(f.app), /escapes/u);
});

test('builder has exact branded descriptions and only Alternate Viewer HTML associations', () => {
  const config = YAML.parse(
    readFileSync(join(browserRoot, 'apps/browser/electron-builder.yml'), 'utf8'),
  );
  assertPrivacyDescriptions(config.mac.extendInfo);
  const generated = config.fileAssociations.map((entry) => ({
    CFBundleTypeExtensions: [entry.ext].flat(),
    CFBundleTypeRole: entry.role,
    LSHandlerRank: entry.rank,
  }));
  const info = {
    ...config.mac.extendInfo,
    CFBundleDocumentTypes: [...config.mac.extendInfo.CFBundleDocumentTypes, ...generated],
  };
  assertDocumentAssociations(info);
  assert.throws(
    () =>
      assertPrivacyDescriptions({
        ...usageDescriptions,
        NSCameraUsageDescription: 'This app needs access to the camera',
      }),
    /privacy string/u,
  );
  info.CFBundleDocumentTypes[0].LSHandlerRank = 'Default';
  assert.throws(() => assertDocumentAssociations(info), /Alternate Viewer/u);
});
