/* global console, process */
import { spawnSync } from 'node:child_process';
import {
  lstatSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve, sep } from 'node:path';
import { fileURLToPath, URL } from 'node:url';

import YAML from 'yaml';
import {
  sha512,
  validateBundleVersions,
  validateReleaseArtifacts,
} from './app-release-validation.mjs';
import {
  macOSReleaseBuilderConfiguration,
  macOSReleaseOptions,
  manualReleaseArtifacts,
} from './macos-release-config.mjs';

const repositoryRoot = resolve(fileURLToPath(new URL('..', import.meta.url)));
if (
  (process.env.ORYN_MAC_SIGNING_MODE && process.env.ORYN_MAC_SIGNING_MODE !== 'release') ||
  (process.env.ORYN_MAC_SECURITY_VARIANT && process.env.ORYN_MAC_SECURITY_VARIANT !== 'release')
) {
  throw new Error('Production packaging cannot use local ad-hoc signing or the inspect-test fuse.');
}
const appDirectory = join(repositoryRoot, 'apps/browser');
const version = JSON.parse(readFileSync(join(appDirectory, 'package.json'), 'utf8')).version;
const releaseDirectory = resolve(
  process.env.ORYN_RELEASE_OUTPUT ?? join(appDirectory, 'release-candidates', version),
);
const releaseOptions = macOSReleaseOptions(version, process.env);
const { updatesEnabled, baseUrl, teamId, channel, stagedRolloutQualified, percentage } =
  releaseOptions;
if (existsSync(releaseDirectory))
  throw new Error(
    'Release output already exists. Use a fresh output directory; never overwrite a candidate.',
  );
mkdirSync(releaseDirectory, { recursive: true });

const temporaryDirectory = mkdtempSync(join(tmpdir(), 'oryn-release-config-'));
const configPath = join(temporaryDirectory, 'electron-builder.yml');
try {
  const config = macOSReleaseBuilderConfiguration(appDirectory, releaseDirectory, releaseOptions);
  writeFileSync(configPath, YAML.stringify(config), { mode: 0o600 });
  run(
    'pnpm',
    [
      'exec',
      'electron-builder',
      '--config',
      configPath,
      '--mac',
      'dmg',
      'zip',
      '--arm64',
      '--publish',
      'never',
    ],
    appDirectory,
  );

  const appPath = join(releaseDirectory, 'mac-arm64', 'Oryn Browser.app');
  const zipPath = join(releaseDirectory, `Oryn-Browser-${version}-arm64.zip`);
  // Verify the final distributed ZIP as well as the unpacked build. This is a
  // required packaging gate, not a separate optional operator command.
  run(
    process.execPath,
    [join(repositoryRoot, 'scripts/verify-macos-release.mjs'), appPath, zipPath],
    appDirectory,
  );
  const identity = validateBundleVersions(
    {
      version: plistValue(appPath, 'CFBundleShortVersionString'),
      bundleVersion: plistValue(appPath, 'CFBundleVersion'),
    },
    { version },
  );
  const unpackedSize = allocatedTreeSize(appPath);
  let metadataPath;
  let metadata;
  if (updatesEnabled) {
    metadataPath = join(releaseDirectory, `${channel === 'beta' ? 'beta' : 'latest'}-mac.yml`);
    metadata = YAML.parse(readFileSync(metadataPath, 'utf8'));
    if (!metadata || typeof metadata !== 'object' || !Array.isArray(metadata.files)) {
      throw new Error('electron-builder produced invalid update metadata.');
    }
    if (metadata.version !== version) throw new Error('Generated metadata has the wrong version.');
    const zipFiles = metadata.files.filter(
      (file) => file && typeof file.url === 'string' && file.url.toLowerCase().endsWith('.zip'),
    );
    if (zipFiles.length !== 1) throw new Error('Release metadata must contain exactly one ZIP.');
    for (const file of metadata.files) {
      if (!file || typeof file.url !== 'string' || typeof file.sha512 !== 'string') {
        throw new Error('Release metadata has an invalid file entry.');
      }
      const filename = decodeURIComponent(basename(new URL(file.url, baseUrl).pathname));
      if (!filename || filename.includes(sep))
        throw new Error('Release metadata has an unsafe filename.');
      const artifactPath = join(releaseDirectory, filename);
      const size = statSync(artifactPath).size;
      if (!Number.isSafeInteger(size) || size <= 0)
        throw new Error(`Invalid artifact size for ${filename}.`);
      file.size = size;
    }
    metadata.oryn = {
      schemaVersion: 2,
      platform: 'darwin',
      arch: 'arm64',
      unpackedSize,
      bundleVersion: identity.bundleVersion,
    };
    if (channel === 'stable' && stagedRolloutQualified && percentage !== undefined) {
      metadata.stagingPercentage = percentage;
    } else {
      delete metadata.stagingPercentage;
    }
    writeFileSync(metadataPath, YAML.stringify(metadata), { mode: 0o644 });
  } else {
    if (
      ['latest-mac.yml', 'beta-mac.yml'].some((name) => existsSync(join(releaseDirectory, name)))
    ) {
      throw new Error('A manual release must not contain update feed metadata.');
    }
    metadata = await manualReleaseArtifacts(releaseDirectory, version, unpackedSize);
  }
  await validateReleaseArtifacts(releaseDirectory, metadata, { manual: !updatesEnabled });
  const revision = spawnSync('git', ['rev-parse', 'HEAD'], {
    cwd: repositoryRoot,
    encoding: 'utf8',
  });
  if (revision.status !== 0) throw new Error('Could not record release source revision.');
  writeFileSync(
    join(releaseDirectory, 'release-manifest.json'),
    `${JSON.stringify(
      {
        schemaVersion: 1,
        version,
        bundleVersion: identity.bundleVersion,
        channel,
        baseUrl,
        teamId,
        updatesEnabled,
        distribution: updatesEnabled ? 'updater' : 'manual',
        stagedRolloutQualified,
        browserRevision: revision.stdout.trim(),
        runtimeManifestSha512: await sha512(
          join(appPath, 'Contents/Resources/engine/release-manifest.json'),
        ),
        ...(metadataPath ? { metadataSha512: await sha512(metadataPath) } : {}),
        signedAndNotarized: true,
        files: metadata.files,
      },
      null,
      2,
    )}\n`,
  );
  console.log(
    `${updatesEnabled ? 'ORYN_UPDATE_METADATA' : 'ORYN_MANUAL_RELEASE'} ${JSON.stringify({ channel, metadata: metadataPath ?? null, unpackedSize, stagedRolloutQualified, percentage: metadata.stagingPercentage ?? null })}`,
  );
} finally {
  rmSync(temporaryDirectory, { recursive: true, force: true });
}

function allocatedTreeSize(root) {
  const entry = lstatSync(root);
  if (entry.isSymbolicLink()) return roundAllocation(entry.size);
  if (entry.isFile()) return roundAllocation(entry.size);
  if (!entry.isDirectory()) return 0;
  return readdirSync(root).reduce(
    (total, child) => total + allocatedTreeSize(join(root, child)),
    roundAllocation(entry.size),
  );
}

function roundAllocation(size) {
  return Math.max(4096, Math.ceil(size / 4096) * 4096);
}

function run(command, arguments_, cwd) {
  const result = spawnSync(command, arguments_, { cwd, stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`${command} failed with status ${result.status}.`);
}

function plistValue(app, name) {
  const result = spawnSync(
    '/usr/libexec/PlistBuddy',
    ['-c', `Print :${name}`, join(app, 'Contents/Info.plist')],
    { encoding: 'utf8', maxBuffer: 64 * 1024 },
  );
  if (result.status !== 0) throw new Error(`Could not read the verified app's ${name}.`);
  return result.stdout.replace(/\r?\n$/u, '');
}
