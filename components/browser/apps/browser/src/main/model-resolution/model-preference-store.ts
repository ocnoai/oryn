import { JsonFileStore, isRecord } from '../data/json-file-store';
import type { ModelRef } from './types';

export type OptimizationPreference = 'balanced' | 'quality' | 'speed' | 'memory';

export interface SiteModelPreference {
  origin: string;
  role: string;
  model: ModelRef;
}

export interface GlobalRolePreference {
  role: string;
  model: ModelRef;
}

export interface GlobalFamilyPreference {
  role: string;
  family: string;
}

interface ModelPreferenceData {
  version: 1;
  preferInstalled: boolean;
  optimization: OptimizationPreference;
  site: SiteModelPreference[];
  global: GlobalRolePreference[];
  families?: GlobalFamilyPreference[];
}

export class ModelPreferenceStore {
  private readonly store: JsonFileStore<ModelPreferenceData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(
      filePath,
      () => ({
        version: 1,
        preferInstalled: true,
        optimization: 'balanced',
        site: [],
        global: [],
        families: [],
      }),
      isModelPreferenceData,
    );
  }

  get sitePreferences(): SiteModelPreference[] {
    return this.store.read().site;
  }

  get globalPreferences(): GlobalRolePreference[] {
    return this.store.read().global;
  }

  get familyPreferences(): GlobalFamilyPreference[] {
    return this.store.read().families ?? [];
  }

  get preferInstalled(): boolean {
    return this.store.read().preferInstalled;
  }

  get optimization(): OptimizationPreference {
    return this.store.read().optimization;
  }

  getSite(origin: string, role: string): ModelRef | undefined {
    return this.store.read().site.find((item) => item.origin === origin && item.role === role)
      ?.model;
  }

  getGlobal(role: string): ModelRef | undefined {
    return this.store.read().global.find((item) => item.role === role)?.model;
  }

  getFamily(role: string): string | undefined {
    return this.familyPreferences.find((item) => item.role === role)?.family;
  }

  setSite(origin: string, role: string, model: ModelRef): void {
    const normalizedOrigin = new URL(origin).origin;
    this.store.update((current) => ({
      ...current,
      site: [
        ...current.site.filter((item) => item.origin !== normalizedOrigin || item.role !== role),
        { origin: normalizedOrigin, role, model },
      ],
    }));
  }

  setGlobal(role: string, model: ModelRef): void {
    this.store.update((current) => ({
      ...current,
      global: [...current.global.filter((item) => item.role !== role), { role, model }],
    }));
  }

  clearGlobal(role: string): void {
    this.store.update((current) => ({
      ...current,
      global: current.global.filter((item) => item.role !== role),
    }));
  }

  setGlobalFamily(role: string, family: string | undefined): void {
    if (!/^[a-z][a-z0-9-]{0,63}$/u.test(role)) throw new Error('INVALID_MODEL_ROLE');
    if (family !== undefined && (family.trim().length === 0 || family.length > 128)) {
      throw new Error('INVALID_MODEL_FAMILY');
    }
    this.store.update((current) => ({
      ...current,
      families: [
        ...(current.families ?? []).filter((item) => item.role !== role),
        ...(family === undefined ? [] : [{ role, family: family.trim() }]),
      ],
    }));
  }

  setRanking(options: { preferInstalled?: boolean; optimization?: OptimizationPreference }): void {
    this.store.update((current) => ({ ...current, ...options }));
  }

  clearSite(origin: string, role: string): void {
    const normalizedOrigin = new URL(origin).origin;
    this.store.update((current) => ({
      ...current,
      site: current.site.filter((item) => item.origin !== normalizedOrigin || item.role !== role),
    }));
  }

  clearOrigin(origin: string): void {
    const normalizedOrigin = new URL(origin).origin;
    this.store.update((current) => ({
      ...current,
      site: current.site.filter((item) => item.origin !== normalizedOrigin),
    }));
  }
}

function isModelPreferenceData(value: unknown): value is ModelPreferenceData {
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    typeof value.preferInstalled !== 'boolean' ||
    !['balanced', 'quality', 'speed', 'memory'].includes(String(value.optimization)) ||
    !Array.isArray(value.site) ||
    !Array.isArray(value.global)
  ) {
    return false;
  }
  return (
    value.site.every(
      (item) =>
        isRecord(item) &&
        typeof item.origin === 'string' &&
        typeof item.role === 'string' &&
        isModelRef(item.model),
    ) &&
    value.global.every(
      (item) => isRecord(item) && typeof item.role === 'string' && isModelRef(item.model),
    ) &&
    (value.families === undefined ||
      (Array.isArray(value.families) &&
        value.families.every(
          (item) =>
            isRecord(item) && typeof item.role === 'string' && typeof item.family === 'string',
        )))
  );
}

function isModelRef(value: unknown): value is ModelRef {
  return (
    isRecord(value) &&
    value.provider === 'huggingface' &&
    typeof value.repository === 'string' &&
    (value.revision === undefined || typeof value.revision === 'string')
  );
}
