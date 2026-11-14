import { validateSelectionModelRef, type ResolvedModelRef } from '@ocno/oryn-protocol';

import type { ArtifactManager } from '../artifact-manager/artifact-manager';
import type { AIActivityStore } from '../data/ai-activity-store';
import { isRecord } from '../data/json-file-store';
import type { EngineSupervisor } from '../engine-supervisor';
import type { BrowserJobRegistry } from '../jobs/browser-job-registry';
import type { LocalAIPermissionStore } from '../model-resolution/local-ai-permission-store';
import type { BrowserModelSessionRegistry } from '../model-resolution/browser-model-sessions';
import type {
  ModelPreferenceStore,
  OptimizationPreference,
} from '../model-resolution/model-preference-store';
import type {
  OrynComputeSettings,
  OrynDeveloperSettings,
  OrynSettingsSnapshot,
} from '../../shared/ipc';
import type { OrynSettingsData, OrynSettingsStore } from './oryn-settings-store';

export class OrynSettingsService {
  constructor(
    private readonly store: OrynSettingsStore,
    private readonly engine: EngineSupervisor,
    private readonly preferences: ModelPreferenceStore,
    private readonly permissions: LocalAIPermissionStore,
    private readonly sessions: BrowserModelSessionRegistry,
    private readonly jobs: BrowserJobRegistry,
    private readonly activity: AIActivityStore,
    private readonly artifacts: ArtifactManager,
  ) {
    const compute = this.store.read().compute;
    this.engine.configureInitialRuntimePolicy(enginePolicy(compute, this.preferences.optimization));
  }

  async snapshot(): Promise<OrynSettingsSnapshot> {
    const [inventory, resources] = await Promise.all([
      this.engine.getModelInventory(),
      this.engine.getRuntimeResources(),
    ]);
    const persisted = this.store.read();
    const activity = this.activity.list();
    const artifactStorage = this.artifacts.storageSnapshot();
    const origins = new Set([
      ...this.permissions.list().map((item) => item.origin),
      ...this.preferences.sitePreferences.map((item) => item.origin),
      ...activity.map((item) => item.origin),
    ]);
    const sites = [...origins].sort().map((origin) => {
      const records = activity.filter((item) => item.origin === origin);
      return {
        origin,
        permissions: this.permissions
          .list()
          .filter((item) => item.origin === origin)
          .map((item) => ({
            role: item.role,
            repository: item.model.repository,
            decision: item.decision,
            updatedAt: item.updatedAt,
          })),
        modelChoices: this.preferences.sitePreferences
          .filter((item) => item.origin === origin)
          .map((item) => ({ role: item.role, repository: item.model.repository })),
        activityCount: records.length,
        lastActivityAt:
          records.length > 0 ? Math.max(...records.map((item) => item.createdAt)) : undefined,
      };
    });
    const modelBytes = inventory.models.reduce((sum, item) => sum + item.logicalSizeBytes, 0);
    const modelReclaimableBytes = inventory.models.reduce(
      (sum, item) => sum + item.reclaimableSizeBytes,
      0,
    );
    return {
      preferences: {
        preferInstalled: this.preferences.preferInstalled,
        optimization: this.preferences.optimization,
        globalModels: this.preferences.globalPreferences,
        globalFamilies: this.preferences.familyPreferences,
        siteChoices: this.preferences.sitePreferences.map((item) => ({
          origin: item.origin,
          role: item.role,
          repository: item.model.repository,
        })),
        availableModels: inventory.models
          .filter((item) => item.installationState === 'installed')
          .map((item) => item.model),
      },
      compute: {
        ...persisted.compute,
        memoryBudgetBytes:
          resources.localAI.memoryBudgetBytes ?? persisted.compute.memoryBudgetBytes,
        generationConcurrency: 1,
      },
      sites,
      storage: {
        modelBytes,
        modelReclaimableBytes,
        modelSharedBytes: Math.max(0, modelBytes - modelReclaimableBytes),
        artifactBytes: artifactStorage.inputBytes + artifactStorage.publishedBytes,
        temporaryArtifactBytes: artifactStorage.temporaryBytes,
        readyArtifacts: artifactStorage.readyArtifacts,
        reclaimableArtifacts: artifactStorage.reclaimableArtifacts,
      },
      developer: { ...persisted.developer, experimentalRunnersAllowed: false },
    };
  }

  async updateRanking(value: {
    preferInstalled?: boolean;
    optimization?: OptimizationPreference;
  }): Promise<void> {
    if (
      !isRecord(value) ||
      Object.keys(value).some((key) => !['preferInstalled', 'optimization'].includes(key)) ||
      (value.preferInstalled !== undefined && typeof value.preferInstalled !== 'boolean') ||
      (value.optimization !== undefined &&
        !['balanced', 'quality', 'speed', 'memory'].includes(value.optimization))
    ) {
      throw new Error('INVALID_MODEL_PREFERENCES');
    }
    this.preferences.setRanking(value);
    await this.engine.updateRuntimePolicy(
      enginePolicy(this.store.read().compute, this.preferences.optimization),
    );
  }

  async updateRole(value: {
    role: string;
    model?: ResolvedModelRef;
    family?: string;
  }): Promise<void> {
    if (!isRecord(value)) throw new Error('INVALID_MODEL_PREFERENCES');
    if (!/^[a-z][a-z0-9-]{0,63}$/u.test(value.role)) throw new Error('INVALID_MODEL_ROLE');
    if (value.model === undefined) {
      this.preferences.clearGlobal(value.role);
    } else {
      validateModel(value.model);
      if (value.model.files) {
        const profile = await this.engine.inspectModel(value.model);
        if (!profile.executable) throw new Error('MODEL_NOT_EXECUTABLE');
      }
      this.preferences.setGlobal(value.role, value.model);
    }
    this.preferences.setGlobalFamily(value.role, value.family?.trim() || undefined);
  }

  async updateCompute(value: Partial<OrynSettingsData['compute']>): Promise<OrynComputeSettings> {
    const compute = this.store.updateCompute(value);
    await this.engine.updateRuntimePolicy(enginePolicy(compute, this.preferences.optimization));
    return { ...compute, generationConcurrency: 1 };
  }

  resetSiteChoices(origin: string): void {
    this.preferences.clearOrigin(requireOrigin(origin));
  }

  askSiteAgain(origin: string): void {
    this.permissions.resetBlockedOrigin(requireOrigin(origin));
  }

  async revokeSite(origin: string, cancelActive: boolean): Promise<void> {
    const normalized = requireOrigin(origin);
    this.permissions.revokeOrigin(normalized);
    if (cancelActive) {
      await this.jobs.cancelAllForOrigin(normalized);
      await this.sessions.releaseForOrigin(normalized);
    }
  }

  assertBackgroundAllowed(tabIsActive: boolean): void {
    if (tabIsActive) return;
    const policy = this.store.read().compute.backgroundJobs;
    if (policy === 'disable') throw new Error('BACKGROUND_COMPUTE_DISABLED');
    if (policy === 'ask') throw new Error('BACKGROUND_COMPUTE_REQUIRES_ATTENTION');
  }

  async cleanupStorage(target: 'artifacts' | 'temporary'): Promise<void> {
    if (target === 'artifacts') {
      await this.artifacts.cleanupReclaimable();
    } else if (target === 'temporary') {
      this.artifacts.cleanupTemporary();
    } else {
      throw new Error('INVALID_STORAGE_CLEANUP');
    }
  }

  updateDeveloper(value: Partial<OrynDeveloperSettings>): OrynSettingsData['developer'] {
    if ('experimentalRunnersAllowed' in value && value.experimentalRunnersAllowed !== false) {
      throw new Error('EXPERIMENTAL_RUNNERS_REQUIRE_STRONGER_DEVELOPER_MODE');
    }
    const safe = { ...value };
    delete safe.experimentalRunnersAllowed;
    return this.store.updateDeveloper(safe);
  }
}

function enginePolicy(compute: OrynSettingsData['compute'], optimization: OptimizationPreference) {
  return {
    ...(compute.memoryBudgetBytes === null ? {} : { memoryBudgetBytes: compute.memoryBudgetBytes }),
    batteryMode: compute.batteryMode,
    allowSpeculativePreload: compute.batteryMode === 'normal',
    runtimeSelectionPolicy: {
      balanced: 'balanced',
      quality: 'quality',
      speed: 'performance',
      memory: 'memory-saver',
    }[optimization] as 'balanced' | 'quality' | 'performance' | 'memory-saver',
  };
}

function requireOrigin(value: string): string {
  const normalized = new URL(value).origin;
  if (normalized !== value || !/^https?:\/\//u.test(value)) throw new Error('INVALID_ORIGIN');
  return normalized;
}

function validateModel(value: ResolvedModelRef): void {
  validateSelectionModelRef(value);
}
