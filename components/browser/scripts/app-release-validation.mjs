import { createHash } from 'node:crypto';
import { createReadStream, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { URL } from 'node:url';
import YAML from 'yaml';

export function releaseVersion(value) {
  if (typeof value !== 'string') throw new Error('Release version must be X.Y.Z or X.Y.Z-beta.N.');
  const match = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-beta\.(0|[1-9]\d*))?$/u.exec(
    value ?? '',
  );
  if (!match) throw new Error('Release version must be X.Y.Z or X.Y.Z-beta.N.');
  const parts = match.slice(1).map((part) => (part === undefined ? Infinity : Number(part)));
  if (
    parts.slice(0, 3).some((part) => !Number.isSafeInteger(part)) ||
    (match[4] !== undefined && !Number.isSafeInteger(parts[3]))
  )
    throw new Error('Invalid version components.');
  return parts;
}

export function assertNewerRelease(next, previous) {
  const left = releaseVersion(next);
  const right = releaseVersion(previous);
  for (let i = 0; i < left.length; i++) {
    if (left[i] > right[i]) return;
    if (left[i] < right[i]) break;
  }
  throw new Error('Publication must advance the channel version.');
}

export function assertReleaseChannel(version, channel) {
  const parts = releaseVersion(version);
  if (!['stable', 'beta'].includes(channel)) throw new Error('Invalid release channel.');
  if (channel === 'stable' && parts[3] !== Infinity)
    throw new Error('Stable cannot publish a prerelease.');
}

export function releaseBaseUrl(value) {
  const url = new URL(value);
  if (
    url.protocol !== 'https:' ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    url.hostname === 'localhost' ||
    url.hostname.endsWith('.localhost') ||
    url.hostname.endsWith('.invalid')
  )
    throw new Error('A production HTTPS release directory is required.');
  url.pathname = `${url.pathname.replace(/\/+$/u, '')}/`;
  return url.toString();
}

export async function sha512(file) {
  const hash = createHash('sha512');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('base64');
}

export function readMetadata(file) {
  if (statSync(file).size > 256 * 1024) throw new Error('Update metadata exceeds its size limit.');
  return YAML.parse(readFileSync(file, 'utf8'), { maxAliasCount: 0 });
}

export function releaseBundleVersion(value) {
  if (typeof value !== 'string' || !/^[\x21-\x7e]{1,64}$/u.test(value))
    throw new Error('Bundle version must be 1–64 ASCII characters without whitespace or controls.');
  return value;
}

// Call only after native code-signature verification has authenticated Info.plist.
// This helper validates values; it does not itself authenticate an app bundle.
export function validateBundleVersions(actual, expected = {}) {
  releaseVersion(actual?.version);
  releaseBundleVersion(actual?.bundleVersion);
  if (expected.version !== undefined && actual.version !== expected.version)
    throw new Error('The signed app has a different release version.');
  if (expected.bundleVersion !== undefined && actual.bundleVersion !== expected.bundleVersion)
    throw new Error('The signed app has a different bundle version.');
  return { version: actual.version, bundleVersion: actual.bundleVersion };
}

export async function validateReleaseArtifacts(directory, metadata, { manual = false } = {}) {
  releaseVersion(metadata?.version);
  if (!Array.isArray(metadata.files) || metadata.files.length !== 2)
    throw new Error('Expected exactly one ZIP and one DMG.');
  const expected = new Set(
    ['zip', 'dmg'].map((ext) => `Oryn-Browser-${metadata.version}-arm64.${ext}`),
  );
  for (const file of metadata.files) {
    if (!expected.delete(file?.url)) throw new Error('Unexpected or duplicate release filename.');
    const artifact = join(directory, file.url);
    const info = statSync(artifact);
    if (
      !info.isFile() ||
      !Number.isSafeInteger(file.size) ||
      file.size <= 0 ||
      file.size !== info.size
    )
      throw new Error('Release size mismatch.');
    if (file.sha512 !== (await sha512(artifact))) throw new Error('Release SHA-512 mismatch.');
  }
  const fields = manual
    ? 'arch,platform,schemaVersion,unpackedSize'
    : 'arch,bundleVersion,platform,schemaVersion,unpackedSize';
  if (
    !metadata.oryn ||
    Object.keys(metadata.oryn).sort().join(',') !== fields ||
    metadata.oryn.schemaVersion !== (manual ? 1 : 2) ||
    metadata.oryn.platform !== 'darwin' ||
    metadata.oryn.arch !== 'arm64' ||
    !Number.isSafeInteger(metadata.oryn.unpackedSize) ||
    metadata.oryn.unpackedSize <= 0
  )
    throw new Error('Invalid release metadata schema or unpacked footprint.');
  if (!manual) releaseBundleVersion(metadata.oryn.bundleVersion);
}

export function validatePromotionEvidence(evidence, metadata, teamId) {
  const zip = metadata.files.find((file) => file.url.endsWith('.zip'));
  if (
    evidence?.schemaVersion !== 2 ||
    evidence.version !== metadata.version ||
    evidence.bundleVersion !== metadata.oryn.bundleVersion ||
    evidence.zipSha512 !== zip.sha512 ||
    evidence.teamId !== teamId
  )
    throw new Error('Qualification evidence does not match this exact signed artifact.');
  const checks = [
    'signedUpgrade',
    'dataPreserved',
    'engineHealthy',
    'wrongPublisherRejected',
    'interruptionRecovery',
    'networkPrivacy',
    'channelIsolation',
    'securityReview',
  ];
  for (const check of checks) {
    if (
      evidence.checks?.[check]?.passed !== true ||
      typeof evidence.checks[check].evidence !== 'string' ||
      !evidence.checks[check].evidence.trim()
    )
      throw new Error(`Missing release evidence: ${check}.`);
  }
}
