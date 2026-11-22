/* global console, process, fetch, AbortSignal */
import {
  constants,
  closeSync,
  copyFileSync,
  existsSync,
  fsyncSync,
  linkSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { Buffer } from 'node:buffer';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import YAML from 'yaml';
import {
  assertNewerRelease,
  assertReleaseChannel,
  readMetadata,
  releaseBaseUrl,
  sha512,
  validatePromotionEvidence,
  validateReleaseArtifacts,
} from './app-release-validation.mjs';

// A provider-neutral static-host filesystem backend. target must be the mounted
// directory actually served at baseUrl. Object-store hosting needs its own
// conditional-write transport; never emulate atomic publication with a bulk sync.
export async function publishRelease({
  directory,
  target,
  channel,
  baseUrl,
  evidence,
  verifyHosted = verifyHostedObject,
  verifySigned = verifySignedZip,
}) {
  const manifest = JSON.parse(readFileSync(join(directory, 'release-manifest.json'), 'utf8'));
  baseUrl = releaseBaseUrl(baseUrl);
  if (
    manifest.schemaVersion !== 1 ||
    manifest.baseUrl !== baseUrl ||
    !manifest.signedAndNotarized ||
    !/^[A-Z0-9]{10}$/u.test(manifest.teamId) ||
    !['beta', 'stable'].includes(manifest.channel)
  )
    throw new Error('Invalid release manifest or feed authority mismatch.');
  const source = join(directory, `${manifest.channel === 'stable' ? 'latest' : 'beta'}-mac.yml`);
  if ((await sha512(source)) !== manifest.metadataSha512)
    throw new Error('Release metadata changed after verification.');
  const metadata = readMetadata(source);
  if (manifest.version !== metadata.version) throw new Error('Manifest version mismatch.');
  assertReleaseChannel(metadata.version, channel);
  await validateReleaseArtifacts(directory, metadata);
  validatePromotionEvidence(evidence, metadata, manifest.teamId);
  if (channel === 'beta') delete metadata.stagingPercentage;
  else if (metadata.stagingPercentage !== undefined && !manifest.stagedRolloutQualified)
    throw new Error('Percentage rollout is not qualified.');
  await verifySigned(directory, metadata, manifest.teamId);

  // All releases share the same lock, including beta/stable promotions. A stale
  // lock needs operator inspection; silently stealing it could corrupt a feed.
  const lock = join(target, '.oryn-publication-lock');
  mkdirSync(lock);
  let temporary;
  try {
    const channelFile = `${channel === 'stable' ? 'latest' : 'beta'}-mac.yml`;
    const destination = join(target, channelFile);
    const bytes = YAML.stringify(metadata);
    if (existsSync(destination)) {
      const previous = readMetadata(destination);
      if (previous.version === metadata.version && readFileSync(destination, 'utf8') === bytes) {
        for (const file of metadata.files) await verifyHosted(new URL(file.url, baseUrl), file);
        await verifyHosted(new URL(channelFile, baseUrl), {
          size: Buffer.byteLength(bytes),
          sha512: createHash('sha512').update(bytes).digest('base64'),
        });
        return { version: metadata.version, channel, alreadyPublished: true };
      }
      assertNewerRelease(metadata.version, previous.version);
    }
    temporary = mkdtempSync(join(target, '.oryn-upload-'));
    for (const file of metadata.files) {
      const immutable = join(target, file.url);
      if (existsSync(immutable)) {
        if ((await sha512(immutable)) !== file.sha512)
          throw new Error('An immutable release filename already contains different bytes.');
      } else {
        const staged = join(temporary, file.url);
        copyFileSync(join(directory, file.url), staged, constants.COPYFILE_EXCL);
        syncFile(staged);
        linkSync(staged, immutable); // Atomic create-only publication, never overwrite.
      }
      // Fetch the final CDN/HTTPS object, not a storage API response, before
      // publishing an offer that clients can see.
      await verifyHosted(new URL(file.url, baseUrl), file);
    }
    const stagedMetadata = join(temporary, channelFile);
    writeFileSync(stagedMetadata, bytes, { mode: 0o644, flag: 'wx' });
    syncFile(stagedMetadata);
    renameSync(stagedMetadata, destination); // Only this channel changes, last.
    syncFile(target);
    await verifyHosted(new URL(channelFile, baseUrl), {
      size: Buffer.byteLength(bytes),
      sha512: createHash('sha512').update(bytes).digest('base64'),
    });
    return { version: metadata.version, channel, alreadyPublished: false };
  } finally {
    if (temporary) rmSync(temporary, { recursive: true, force: true });
    rmSync(lock, { recursive: true });
  }
}

function syncFile(file) {
  const fd = openSync(file, 'r');
  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function verifySignedZip(directory, metadata, teamId) {
  const script = fileURLToPath(new URL('./verify-macos-release.mjs', import.meta.url));
  const zip = metadata.files.find((file) => file.url.endsWith('.zip'));
  const result = spawnSync(
    process.execPath,
    [script, '-', join(directory, zip.url), metadata.version, metadata.oryn.bundleVersion],
    {
      env: { ...process.env, ORYN_APPLE_TEAM_ID: teamId },
      stdio: 'inherit',
    },
  );
  if (result.status !== 0) throw new Error('Signed ZIP verification failed.');
}

export async function verifyHostedObject(url, file) {
  const response = await fetch(url, {
    redirect: 'error',
    headers: { 'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache' },
    signal: AbortSignal.timeout(30 * 60 * 1000),
  });
  if (
    response.status !== 200 ||
    response.headers.get('content-length') !== String(file.size) ||
    ![null, 'identity'].includes(response.headers.get('content-encoding')) ||
    response.headers.has('transfer-encoding')
  )
    throw new Error(
      'Hosted release must return a direct, uncompressed 200 with exact Content-Length.',
    );
  const hash = createHash('sha512');
  let received = 0;
  for await (const chunk of response.body) {
    received += chunk.length;
    if (received > file.size) throw new Error('Hosted release exceeds its approved size.');
    hash.update(chunk);
  }
  if (received !== file.size || hash.digest('base64') !== file.sha512)
    throw new Error('Hosted release bytes do not match the signed candidate.');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [directory, channel, evidencePath] = process.argv.slice(2);
  if (
    !directory ||
    !channel ||
    !process.env.ORYN_RELEASE_PUBLISH_DIR ||
    !process.env.ORYN_UPDATE_BASE_URL
  )
    throw new Error(
      'Usage: publish-app-update.mjs <candidate-dir> <beta|stable> [qualification.json]; set ORYN_RELEASE_PUBLISH_DIR and ORYN_UPDATE_BASE_URL.',
    );
  console.log(
    JSON.stringify(
      await publishRelease({
        directory: resolve(directory),
        channel,
        target: resolve(process.env.ORYN_RELEASE_PUBLISH_DIR),
        baseUrl: process.env.ORYN_UPDATE_BASE_URL,
        evidence: evidencePath ? JSON.parse(readFileSync(evidencePath, 'utf8')) : undefined,
      }),
    ),
  );
}
