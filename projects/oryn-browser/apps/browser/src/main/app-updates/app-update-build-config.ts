import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export interface AppUpdateBuildConfiguration {
  enabled: boolean;
  baseUrl?: string;
  expectedTeamId?: string;
  stagedRolloutQualified?: boolean;
  channel?: 'stable' | 'beta';
}

export function readAppUpdateBuildConfiguration(options: {
  packaged: boolean;
  appPath: string;
}): AppUpdateBuildConfiguration {
  if (!options.packaged) return { enabled: false };
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(join(options.appPath, 'package.json'), 'utf8')) as unknown;
  } catch {
    return { enabled: false };
  }
  if (
    !isRecord(parsed) ||
    parsed.orynBuildFlavor === 'local-development' ||
    !isRecord(parsed.orynUpdate) ||
    parsed.orynUpdate.enabled !== true
  ) {
    return { enabled: false };
  }
  const baseUrl = validateUpdateBaseUrl(parsed.orynUpdate.baseUrl);
  const expectedTeamId = parsed.orynUpdate.expectedTeamId;
  if (typeof expectedTeamId !== 'string' || !/^[A-Z0-9]{10}$/u.test(expectedTeamId)) {
    throw new Error('The updater-enabled build has an invalid Apple Team ID.');
  }
  if (typeof parsed.orynUpdate.stagedRolloutQualified !== 'boolean') {
    throw new Error('The updater-enabled build must declare staged-rollout qualification.');
  }
  const channel =
    parsed.orynUpdate.channel ??
    (typeof parsed.version === 'string' && /^\d+\.\d+\.\d+-beta\./u.test(parsed.version)
      ? 'beta'
      : 'stable');
  if (channel !== 'stable' && channel !== 'beta')
    throw new Error('Invalid packaged update channel.');
  return {
    enabled: true,
    baseUrl,
    expectedTeamId,
    channel,
    stagedRolloutQualified: parsed.orynUpdate.stagedRolloutQualified,
  };
}

export function validateUpdateBaseUrl(value: unknown): string {
  if (typeof value !== 'string' || value.length > 2048) {
    throw new Error('The update base URL is missing or invalid.');
  }
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
  ) {
    throw new Error('The update base URL must be a production HTTPS URL without credentials.');
  }
  url.pathname = `${url.pathname.replace(/\/+$/u, '')}/`;
  return url.toString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
