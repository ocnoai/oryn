import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { URL } from 'node:url';
import YAML from 'yaml';
import {
  assertPackagedAppUpdateConfiguration,
  validateAppUpdateProviderConfiguration,
} from './macos-app-update-config.mjs';

const appRequire = createRequire(new URL('../apps/browser/package.json', import.meta.url));
const builderRequire = createRequire(appRequire.resolve('electron-builder'));
const packagerRequire = createRequire(builderRequire.resolve('app-builder-lib'));
const { createPackage } = packagerRequire('@electron/asar');
const { PublishManager } = builderRequire('app-builder-lib/out/publish/PublishManager');
const { Platform } = builderRequire('app-builder-lib');
const update = {
  enabled: true,
  baseUrl: 'https://registry.oryn.ocno.ai/',
  expectedTeamId: '3AWD2NU8UF',
  stagedRolloutQualified: false,
  channel: 'stable',
};
const provider = {
  provider: 'generic',
  url: update.baseUrl,
  channel: 'latest',
  updaterCacheDirName: 'oryn-browser-updater',
};

async function fixture(t, updater = update) {
  const root = mkdtempSync(join(tmpdir(), 'oryn-app-update-config-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const app = join(root, 'Oryn Browser.app');
  const resources = join(app, 'Contents/Resources');
  const source = join(root, 'package-source');
  mkdirSync(resources, { recursive: true });
  mkdirSync(source);
  writeFileSync(
    join(source, 'package.json'),
    JSON.stringify({ name: 'oryn-browser', version: '0.1.4', orynUpdate: updater }),
  );
  await createPackage(source, join(resources, 'app.asar'));
  return { app, resources, config: join(resources, 'app-update.yml') };
}

test('updater-enabled packaged apps require a matching provider resource', async (t) => {
  const f = await fixture(t);
  assert.throws(() => assertPackagedAppUpdateConfiguration(f.app), /APP_UPDATE_CONFIG_MISSING/u);
  writeFileSync(f.config, YAML.stringify(provider));
  assert.equal(assertPackagedAppUpdateConfiguration(f.app, update).enabled, true);
  writeFileSync(f.config, YAML.stringify({ ...provider, url: 'https://other.example/' }));
  assert.throws(() => assertPackagedAppUpdateConfiguration(f.app), /APP_UPDATE_CONFIG_MISMATCH/u);
});

test('the provider must agree with the signed channel, team and planned cache', () => {
  const metadata = { orynUpdate: update };
  for (const changed of [
    { ...provider, provider: 'github' },
    { ...provider, channel: 'beta' },
    { ...provider, updaterCacheDirName: '../escape-updater' },
  ]) {
    assert.throws(
      () => validateAppUpdateProviderConfiguration(metadata, changed),
      /APP_UPDATE_CONFIG_MISMATCH/u,
    );
  }
  assert.throws(
    () =>
      validateAppUpdateProviderConfiguration(metadata, provider, { expectedTeamId: 'OTHERTEAM0' }),
    /APP_UPDATE_METADATA_MISMATCH/u,
  );
  assert.throws(
    () =>
      validateAppUpdateProviderConfiguration(metadata, provider, {
        updaterCacheDirName: 'other-updater',
      }),
    /APP_UPDATE_CONFIG_MISMATCH/u,
  );
});

test('beta and manual updater-disabled apps retain their supported configuration', async (t) => {
  const beta = { ...update, channel: 'beta' };
  assert.equal(
    validateAppUpdateProviderConfiguration({ orynUpdate: beta }, { ...provider, channel: 'beta' })
      .channel,
    'beta',
  );
  const f = await fixture(t, { enabled: false });
  assert.deepEqual(assertPackagedAppUpdateConfiguration(f.app), { enabled: false });
  assert.throws(
    () => assertPackagedAppUpdateConfiguration(f.app, { enabled: true }),
    /APP_UPDATE_METADATA_MISMATCH/u,
  );
});

test('provider resources reject symlinks and oversized data', async (t) => {
  const f = await fixture(t);
  const target = join(f.resources, 'other.yml');
  writeFileSync(target, YAML.stringify(provider));
  symlinkSync(target, f.config);
  assert.throws(() => assertPackagedAppUpdateConfiguration(f.app), /APP_UPDATE_CONFIG_INVALID/u);
  rmSync(f.config);
  writeFileSync(f.config, 'x'.repeat(16 * 1024 + 1));
  assert.throws(() => assertPackagedAppUpdateConfiguration(f.app), /APP_UPDATE_CONFIG_INVALID/u);
});

test('pinned builder skips dir targets and materializes the resource for zip before signing', async (t) => {
  const f = await fixture(t);
  let afterPack;
  const packager = {
    cancellationToken: { cancelled: false },
    config: {
      publish: { provider: provider.provider, url: provider.url, channel: provider.channel },
    },
    appInfo: { updaterCacheDirName: provider.updaterCacheDirName },
    onAfterPack(handler) {
      afterPack = handler;
    },
    onArtifactCreated() {},
  };
  new PublishManager(packager, { publish: 'never' });
  const platformPackager = {
    platform: Platform.MAC,
    platformSpecificBuildOptions: {},
    config: packager.config,
    appInfo: packager.appInfo,
    info: packager,
    expandMacro(value) {
      return value;
    },
    getResourcesDir() {
      return f.resources;
    },
  };
  const event = { electronPlatformName: 'darwin', packager: platformPackager, arch: 3 };
  await afterPack({ ...event, targets: [{ name: 'dir' }] });
  assert.equal(existsSync(f.config), false);
  assert.throws(() => assertPackagedAppUpdateConfiguration(f.app), /APP_UPDATE_CONFIG_MISSING/u);
  await afterPack({ ...event, targets: [{ name: 'zip' }] });
  assert.equal(assertPackagedAppUpdateConfiguration(f.app, update).enabled, true);
});
