import { JsonFileStore, isRecord } from '../data/json-file-store';
import { existsSync } from 'node:fs';

export type AppUpdateChannel = 'stable' | 'beta';

export interface AppUpdateSettingsData {
  version: 1;
  automaticChecks: boolean;
  channel: AppUpdateChannel;
  skipped?: {
    channel: AppUpdateChannel;
    version: string;
  };
  deferred?: {
    channel: AppUpdateChannel;
    version: string;
    until: number;
  };
  lastCheckAt?: number;
}

export class AppUpdateSettingsStore {
  private readonly store: JsonFileStore<AppUpdateSettingsData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(filePath, defaults, isSettings);
  }

  initializeChannel(channel: AppUpdateChannel): void {
    // Seed only a new profile. Installing beta must not override an explicit
    // choice already stored by a stable or beta installation.
    if (!existsSync(this.store.filePath)) this.update({ channel });
  }

  read(): AppUpdateSettingsData {
    try {
      return structuredClone(this.store.read());
    } catch {
      return defaults();
    }
  }

  update(value: Partial<Omit<AppUpdateSettingsData, 'version'>>): AppUpdateSettingsData {
    const next = { ...this.read(), ...value, version: 1 as const };
    this.store.write(next);
    return structuredClone(next);
  }
}

function defaults(): AppUpdateSettingsData {
  return { version: 1, automaticChecks: true, channel: 'stable' };
}

function isSettings(value: unknown): value is AppUpdateSettingsData {
  const lastCheckAt = value && isRecord(value) ? value.lastCheckAt : undefined;
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    typeof value.automaticChecks !== 'boolean' ||
    (value.channel !== 'stable' && value.channel !== 'beta') ||
    (lastCheckAt !== undefined &&
      (typeof lastCheckAt !== 'number' || !Number.isSafeInteger(lastCheckAt) || lastCheckAt < 0))
  ) {
    return false;
  }
  const skippedValid =
    value.skipped === undefined ||
    (isRecord(value.skipped) &&
      (value.skipped.channel === 'stable' || value.skipped.channel === 'beta') &&
      typeof value.skipped.version === 'string' &&
      value.skipped.version.length > 0 &&
      value.skipped.version.length <= 128);
  if (!skippedValid) return false;
  if (value.deferred === undefined) return true;
  return (
    isRecord(value.deferred) &&
    (value.deferred.channel === 'stable' || value.deferred.channel === 'beta') &&
    typeof value.deferred.version === 'string' &&
    value.deferred.version.length > 0 &&
    value.deferred.version.length <= 128 &&
    typeof value.deferred.until === 'number' &&
    Number.isSafeInteger(value.deferred.until) &&
    value.deferred.until >= 0
  );
}
