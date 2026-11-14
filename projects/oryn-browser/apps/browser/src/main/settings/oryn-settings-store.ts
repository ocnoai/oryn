import { JsonFileStore, isRecord } from '../data/json-file-store';

export interface OrynSettingsData {
  version: 1;
  compute: {
    memoryBudgetBytes: number | null;
    backgroundJobs: 'allow' | 'ask' | 'disable';
    batteryMode: 'normal' | 'reduce' | 'disable';
  };
  developer: {
    showDiagnostics: boolean;
    showResolutionDetails: boolean;
    showProtocolDiagnostics: boolean;
  };
}

export class OrynSettingsStore {
  private readonly store: JsonFileStore<OrynSettingsData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(filePath, defaults, isSettings);
  }

  read(): OrynSettingsData {
    return structuredClone(this.store.read());
  }

  updateCompute(value: Partial<OrynSettingsData['compute']>): OrynSettingsData['compute'] {
    validateCompute(value);
    const result = this.store.update((current) => ({
      ...current,
      compute: { ...current.compute, ...value },
    }));
    return structuredClone(result.compute);
  }

  updateDeveloper(value: Partial<OrynSettingsData['developer']>): OrynSettingsData['developer'] {
    if (
      !isRecord(value) ||
      Object.keys(value).some(
        (key) =>
          !['showDiagnostics', 'showResolutionDetails', 'showProtocolDiagnostics'].includes(key),
      ) ||
      Object.values(value).some((item) => typeof item !== 'boolean')
    ) {
      throw new Error('INVALID_DEVELOPER_SETTINGS');
    }
    const result = this.store.update((current) => ({
      ...current,
      developer: { ...current.developer, ...value },
    }));
    return structuredClone(result.developer);
  }
}

function defaults(): OrynSettingsData {
  return {
    version: 1,
    compute: { memoryBudgetBytes: null, backgroundJobs: 'allow', batteryMode: 'normal' },
    developer: {
      showDiagnostics: false,
      showResolutionDetails: false,
      showProtocolDiagnostics: false,
    },
  };
}

function validateCompute(value: Partial<OrynSettingsData['compute']>): void {
  if (
    !isRecord(value) ||
    Object.keys(value).some(
      (key) => !['memoryBudgetBytes', 'backgroundJobs', 'batteryMode'].includes(key),
    ) ||
    (value.memoryBudgetBytes !== undefined &&
      value.memoryBudgetBytes !== null &&
      (!Number.isSafeInteger(value.memoryBudgetBytes) ||
        value.memoryBudgetBytes < 512 * 1024 * 1024 ||
        value.memoryBudgetBytes > 1024 * 1024 * 1024 * 1024)) ||
    (value.backgroundJobs !== undefined &&
      !['allow', 'ask', 'disable'].includes(value.backgroundJobs)) ||
    (value.batteryMode !== undefined &&
      !['normal', 'reduce', 'disable'].includes(value.batteryMode))
  ) {
    throw new Error('INVALID_COMPUTE_SETTINGS');
  }
}

function isSettings(value: unknown): value is OrynSettingsData {
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    !isRecord(value.compute) ||
    !isRecord(value.developer)
  ) {
    return false;
  }
  try {
    validateCompute(value.compute);
  } catch {
    return false;
  }
  return (
    typeof value.developer.showDiagnostics === 'boolean' &&
    typeof value.developer.showResolutionDetails === 'boolean' &&
    typeof value.developer.showProtocolDiagnostics === 'boolean'
  );
}
