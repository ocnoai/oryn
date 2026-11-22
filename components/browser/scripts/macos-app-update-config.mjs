/* global console, process */
import { lstatSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import YAML from 'yaml';
import { releaseBaseUrl } from './app-release-validation.mjs';

const appRequire = createRequire(new URL('../apps/browser/package.json', import.meta.url));
const builderRequire = createRequire(appRequire.resolve('electron-builder'));
const packagerRequire = createRequire(builderRequire.resolve('app-builder-lib'));
const { extractFile } = packagerRequire('@electron/asar');

function requireCondition(value, code, message) {
  if (!value) throw new Error(`${code}: ${message}`);
}

function record(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function validateAppUpdateProviderConfiguration(metadata, provider, expected = {}) {
  const update = metadata?.orynUpdate;
  if (!record(update) || update.enabled !== true) {
    requireCondition(
      expected.enabled !== true,
      'APP_UPDATE_METADATA_MISMATCH',
      'The app does not enable the expected updater.',
    );
    return { enabled: false };
  }
  requireCondition(
    expected.enabled !== false &&
      typeof update.expectedTeamId === 'string' &&
      /^[A-Z0-9]{10}$/u.test(update.expectedTeamId) &&
      ['stable', 'beta'].includes(update.channel) &&
      typeof update.stagedRolloutQualified === 'boolean',
    'APP_UPDATE_METADATA_INVALID',
    'The packaged updater metadata is invalid.',
  );
  const baseUrl = releaseBaseUrl(update.baseUrl);
  requireCondition(
    (expected.baseUrl === undefined || baseUrl === releaseBaseUrl(expected.baseUrl)) &&
      (expected.expectedTeamId === undefined ||
        update.expectedTeamId === expected.expectedTeamId) &&
      (expected.channel === undefined || update.channel === expected.channel),
    'APP_UPDATE_METADATA_MISMATCH',
    'The packaged updater metadata differs from the release plan.',
  );
  requireCondition(
    record(provider),
    'APP_UPDATE_CONFIG_MISSING',
    'The updater-enabled app is missing Contents/Resources/app-update.yml.',
  );
  const channel = update.channel === 'beta' ? 'beta' : 'latest';
  requireCondition(
    provider.provider === 'generic' &&
      typeof provider.url === 'string' &&
      releaseBaseUrl(provider.url) === baseUrl &&
      provider.channel === channel &&
      typeof provider.updaterCacheDirName === 'string' &&
      /^[A-Za-z0-9@_-][A-Za-z0-9@._-]{0,127}$/u.test(provider.updaterCacheDirName) &&
      provider.updaterCacheDirName.endsWith('-updater') &&
      (expected.updaterCacheDirName === undefined ||
        provider.updaterCacheDirName === expected.updaterCacheDirName),
    'APP_UPDATE_CONFIG_MISMATCH',
    'The bundled update provider, URL, channel or cache directory differs from the app metadata.',
  );
  return {
    enabled: true,
    baseUrl,
    channel: update.channel,
    expectedTeamId: update.expectedTeamId,
    updaterCacheDirName: provider.updaterCacheDirName,
  };
}

// Call after signature verification to authenticate the read metadata; the same
// check also runs before signing to catch incomplete packaging without sealing it.
export function assertPackagedAppUpdateConfiguration(app, expected = {}) {
  const resources = join(resolve(app), 'Contents/Resources');
  const metadata = JSON.parse(extractFile(join(resources, 'app.asar'), 'package.json').toString());
  let provider;
  if (metadata.orynUpdate?.enabled === true) {
    const path = join(resources, 'app-update.yml');
    let stat;
    try {
      stat = lstatSync(path);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (stat) {
      requireCondition(
        stat.isFile() && !stat.isSymbolicLink() && stat.size > 0 && stat.size <= 16 * 1024,
        'APP_UPDATE_CONFIG_INVALID',
        'The bundled updater configuration must be a bounded regular file.',
      );
      provider = YAML.parse(readFileSync(path, 'utf8'), { maxAliasCount: 0 });
    }
  }
  return validateAppUpdateProviderConfiguration(metadata, provider, expected);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [app, baseUrl, expectedTeamId, channel = 'stable', ...extra] = process.argv.slice(2);
  if (!app || !baseUrl || !expectedTeamId || extra.length) {
    throw new Error('Usage: macos-app-update-config.mjs <app> <base-url> <team-id> [stable|beta]');
  }
  const result = assertPackagedAppUpdateConfiguration(app, {
    enabled: true,
    baseUrl,
    expectedTeamId,
    channel,
  });
  console.log(`APP_UPDATE_CONFIG_RESULT ${JSON.stringify({ passed: true, ...result })}`);
}
