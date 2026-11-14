import { taskRequirementsFrom } from '@ocno/oryn-protocol';
import type { RegistryService } from '../model-resolution/registry-service';
import type { ModelInspector } from '../model-resolution/types';
import type { TrustedAlternativeSearch } from '../../shared/ipc';
import { exactModelKey } from '@ocno/oryn-protocol';
import { randomUUID } from 'node:crypto';

import type {
  EngineModelInventoryRecord,
  ModelRemovePreviewResult,
  ModelRemoveResult,
  ModelResidencyResult,
  RuntimePackCleanupResult,
  RuntimePackInventoryResult,
  RuntimeOverrideResult,
  ResolvedModelRef,
} from '@ocno/oryn-protocol';

import type { AIActivityStore } from '../data/ai-activity-store';
import type { BrowserJobRegistry } from '../jobs/browser-job-registry';
import type { BrowserModelSessionRegistry } from '../model-resolution/browser-model-sessions';
import type { ModelPreferenceStore } from '../model-resolution/model-preference-store';
import { ModelResolutionError, type ModelResolver } from '../model-resolution/model-resolver';
import type { ModelRequest } from '../model-resolution/types';
import type {
  ModelManagerSnapshot,
  TrustedInstalledModel,
  TrustedModelSubstitute,
} from '../../shared/ipc';

interface ModelManagerEngine {
  getModelInventory(): Promise<{ models: EngineModelInventoryRecord[] }>;
  getRuntimePackInventory(): Promise<RuntimePackInventoryResult>;
  setModelRuntimeOverride(
    model: ResolvedModelRef,
    candidateId?: string,
  ): Promise<RuntimeOverrideResult>;
  cleanupRuntimePacks(): Promise<RuntimePackCleanupResult>;
  unloadModel(
    operationId: string,
    model: ResolvedModelRef,
    mode: 'fail-if-busy' | 'when-idle',
  ): Promise<ModelResidencyResult>;
  previewModelRemoval(installationId: string): Promise<ModelRemovePreviewResult>;
  removeModel(
    operationId: string,
    installationId: string,
    mode: 'fail-if-busy' | 'when-idle',
  ): Promise<ModelRemoveResult>;
}

export class ModelManagerService {
  constructor(
    private readonly engine: ModelManagerEngine,
    private readonly resolver: ModelResolver,
    private readonly preferences: ModelPreferenceStore,
    private readonly sessions: BrowserModelSessionRegistry,
    private readonly jobs: BrowserJobRegistry,
    private readonly activity: AIActivityStore,
    private readonly registry?: RegistryService,
    private readonly inspector?: ModelInspector,
  ) {}

  registryStatus() {
    return (
      this.registry?.status() ?? {
        enabled: false,
        configured: false,
        status: 'unavailable',
        catalogVersion: 0,
        publishedAt: '',
        stale: true,
      }
    );
  }
  setRegistryRefresh(enabled: boolean) {
    if (typeof enabled !== 'boolean') throw new Error('INVALID_REGISTRY_SETTING');
    return this.registry?.setEnabled(enabled) ?? this.registryStatus();
  }

  async checkBetterAlternatives(
    installationId: string,
    signal: AbortSignal,
  ): Promise<TrustedAlternativeSearch> {
    const record = await this.requireRecord(installationId);
    if (!this.registry || !this.inspector) throw new Error('SEARCH_UNAVAILABLE');
    const session = this.sessions
      .snapshot()
      .find((s) => modelKey(s.resolution.resolvedModel) === modelKey(record.model));
    const requirements = session?.request.requirements ?? {
      task: record.technicalCapabilities?.task,
    };
    const result = await this.registry.search(record.model, signal);
    const snapshot = this.registry.capture();
    const variants: TrustedAlternativeSearch['variants'] = [];
    for (const variant of result.variants) {
      signal.throwIfAborted();
      const model = { provider: 'huggingface' as const, ...variant.artifact };
      if (modelKey(model) === modelKey(record.model)) continue;
      try {
        const profile = await this.inspector.inspectModel(
          model,
          taskRequirementsFrom(requirements),
        );
        signal.throwIfAborted();
        const revoked = this.registry.isRevoked(profile.model, snapshot);
        variants.push({
          model: profile.model,
          executable: profile.executable && !revoked,
          lineage: variant.lineage,
          reasons: [
            ...(profile.blockingReasons ?? []),
            ...(revoked ? ['This artifact is revoked by the active security policy.'] : []),
            'Publisher metadata is not verified equivalence. No comparable speed measurement is known.',
          ],
          requiredMemoryBytes: profile.estimatedMemoryBytes,
          downloadBytes: profile.estimatedDiskBytes,
        });
      } catch {
        signal.throwIfAborted();
        variants.push({
          model,
          executable: false,
          lineage: variant.lineage,
          reasons: ['Local inspection unavailable. Try again before choosing this artifact.'],
        });
      }
    }
    return {
      suggestedRole:
        session?.request.role ??
        (record.technicalCapabilities?.task === 'text-generation'
          ? 'chat'
          : record.technicalCapabilities?.task === 'automatic-speech-recognition'
            ? 'transcription'
            : (record.technicalCapabilities?.task ?? 'model')),
      status: !variants.length
        ? 'no-results'
        : variants.some((v) => v.executable)
          ? 'results'
          : 'no-better-option',
      variants,
    };
  }

  async snapshot(): Promise<ModelManagerSnapshot> {
    const inventory = await this.engine.getModelInventory();
    const models = inventory.models.map((record) => this.project(record));
    return {
      totalLogicalBytes: models.reduce((total, model) => total + model.logicalSizeBytes, 0),
      totalReclaimableBytes: models.reduce((total, model) => total + model.reclaimableSizeBytes, 0),
      models,
      runtimePacks: [],
      runtimePackPolicy: {
        catalogVersion: 0,
        catalogDigest: '',
        thirdPartyPublishersEnabled: false,
        automaticExecutableUpdatesEnabled: false,
      },
    };
  }

  async runtimePacks(): Promise<Pick<ModelManagerSnapshot, 'runtimePacks' | 'runtimePackPolicy'>> {
    const runtimePacks = await this.engine.getRuntimePackInventory();
    return {
      runtimePacks: runtimePacks.packs,
      runtimePackPolicy: {
        catalogVersion: runtimePacks.catalogVersion,
        catalogDigest: runtimePacks.catalogDigest,
        thirdPartyPublishersEnabled: runtimePacks.thirdPartyPublishersEnabled,
        automaticExecutableUpdatesEnabled: runtimePacks.automaticExecutableUpdatesEnabled,
      },
    };
  }

  async cleanupRuntimePacks(): Promise<RuntimePackCleanupResult> {
    return this.engine.cleanupRuntimePacks();
  }

  async unload(installationId: string): Promise<ModelResidencyResult> {
    const record = await this.requireRecord(installationId);
    return this.engine.unloadModel(operationId('unload'), record.model, 'when-idle');
  }

  async setRuntimeOverride(installationId: string, candidateId?: string): Promise<void> {
    const record = await this.requireRecord(installationId);
    await this.engine.setModelRuntimeOverride(record.model, candidateId);
    await this.engine.unloadModel(operationId('runtime-change'), record.model, 'when-idle');
  }

  async previewRemoval(installationId: string): Promise<ModelRemovePreviewResult> {
    await this.requireRecord(installationId);
    return this.engine.previewModelRemoval(installationId);
  }

  async remove(
    installationId: string,
    action: 'delete' | 'when-idle' | 'cancel-and-delete',
  ): Promise<ModelRemoveResult> {
    const record = await this.requireRecord(installationId);
    if (action === 'cancel-and-delete') {
      await this.jobs.cancelAllForModel(record.model);
      await this.sessions.releaseForModel(record.model);
    }
    return this.engine.removeModel(
      operationId('remove'),
      installationId,
      action === 'delete' ? 'fail-if-busy' : 'when-idle',
    );
  }

  async sourceUrl(installationId: string): Promise<string> {
    const record = await this.requireRecord(installationId);
    return sourceUrl(record.model);
  }

  private async requireRecord(installationId: string): Promise<EngineModelInventoryRecord> {
    const record = (await this.engine.getModelInventory()).models.find(
      (candidate) => candidate.installationId === installationId,
    );
    if (!record) throw new Error('MODEL_NOT_INSTALLED');
    return record;
  }

  async modelSubstitutes(installationId: string): Promise<TrustedModelSubstitute[]> {
    const inventory = (await this.engine.getModelInventory()).models;
    const record = inventory.find((item) => item.installationId === installationId);
    if (!record) throw new Error('MODEL_NOT_INSTALLED');
    const session = this.sessions
      .snapshot()
      .find((item) => modelKey(item.resolution.resolvedModel) === modelKey(record.model));
    return this.substitutes(
      record,
      inventory,
      session?.request ?? syntheticRequest(record),
      session?.origin ?? 'oryn://model-manager',
    );
  }

  private project(record: EngineModelInventoryRecord): TrustedInstalledModel {
    const key = modelKey(record.model);
    const sessions = this.sessions
      .snapshot()
      .filter((session) => modelKey(session.resolution.resolvedModel) === key);
    const activity = this.activity.list().filter((item) => modelKey(item.resolvedModel) === key);
    const usedBy = [
      ...new Set([...sessions.map((item) => item.origin), ...activity.map((item) => item.origin)]),
    ].slice(0, 32);
    const preferredFor = [
      ...this.preferences.sitePreferences
        .filter((item) => sameLogicalModel(item.model, record.model))
        .map((item) => `${item.origin} / ${item.role}`),
      ...this.preferences.globalPreferences
        .filter((item) => sameLogicalModel(item.model, record.model))
        .map((item) => `All sites / ${item.role}`),
    ];
    const technical = record.technicalCapabilities;
    const architectures = technical?.architectures ?? [];
    return {
      installationId: record.installationId,
      displayName: record.model.repository.split('/').at(-1) ?? record.model.repository,
      publisher: record.model.repository.split('/')[0] ?? 'Unknown',
      repository: record.model.repository,
      exactRevision: record.model.revision,
      sourceUrl: sourceUrl(record.model),
      task: technical?.task,
      modalities: [
        ...(technical?.inputModalities ?? []).map((value) => `Input: ${value}`),
        ...(technical?.outputModalities ?? []).map((value) => `Output: ${value}`),
      ],
      family: architectures[0]?.replace(/For(?:ConditionalGeneration|CausalLM)$/u, ''),
      parameterCount: technical?.parameterCount,
      precision: undefined,
      runner: record.runner,
      ...(record.runtimePack
        ? {
            runtimePack: {
              packId: record.runtimePack.packId,
              packVersion: record.runtimePack.packVersion,
              runnerId: record.runtimePack.runnerId,
              taskContract: record.runtimePack.taskContract,
              compatibilityLevel: record.runtimePack.compatibilityLevel,
              ...(record.runtimePack.trustedMetadata
                ? { trustedMetadata: record.runtimePack.trustedMetadata }
                : {}),
            },
          }
        : {}),
      ...(record.runtimePlan
        ? {
            runtimeSelection: {
              candidateId: record.runtimePlan.selected.candidateId,
              automatic: !record.runtimePlan.reasonCodes.includes('USER_OVERRIDE'),
              policy: record.runtimePlan.policy,
              foundationId: record.runtimePlan.selected.foundation.id,
              foundationVersion: record.runtimePlan.selected.foundation.version,
              adapterId: record.runtimePlan.selected.adapter.id,
              adapterVersion: record.runtimePlan.selected.adapter.version,
              backend: record.runtimePlan.selected.backend,
              precision: record.runtimePlan.selected.precision,
              ...(record.runtimePlan.selected.quantization
                ? { quantization: record.runtimePlan.selected.quantization }
                : {}),
              estimatedMemoryBytes: record.runtimePlan.selected.estimatedMemoryBytes,
              fallbackCount: record.runtimePlan.fallbacks.length,
              reasonCodes: record.runtimePlan.reasonCodes,
              alternatives: record.runtimePlan.fallbacks.map((candidate) => ({
                candidateId: candidate.candidateId,
                foundationId: candidate.foundation.id,
                foundationVersion: candidate.foundation.version,
                adapterId: candidate.adapter.id,
                adapterVersion: candidate.adapter.version,
                backend: candidate.backend,
                precision: candidate.precision,
                ...(candidate.quantization ? { quantization: candidate.quantization } : {}),
                estimatedMemoryBytes: candidate.estimatedMemoryBytes,
              })),
            },
          }
        : {}),
      executableBackends: record.executableBackends ?? [],
      logicalSizeBytes: record.logicalSizeBytes,
      reclaimableSizeBytes: record.reclaimableSizeBytes,
      estimatedResidentBytes: record.estimatedResidentBytes,
      installationState: record.installationState,
      residencyState: record.residencyState,
      activeSessions: record.activeSessions,
      runningJobs: record.runningJobs,
      queuedJobs: record.queuedJobs,
      installedAt: record.installedAt,
      lastUsedAt: record.lastUsedAt,
      usedBy,
      preferredFor,
      substitutes: [],
    };
  }

  private async substitutes(
    installed: EngineModelInventoryRecord,
    inventory: EngineModelInventoryRecord[],
    request: ModelRequest,
    origin: string,
  ): Promise<TrustedModelSubstitute[]> {
    const alternatives = inventory.filter(
      (candidate) =>
        candidate.installationState === 'installed' &&
        modelKey(candidate.model) !== modelKey(installed.model),
    );
    const assessed = await Promise.all(
      alternatives.map(async (candidate): Promise<TrustedModelSubstitute | undefined> => {
        try {
          const result = await this.resolver.resolve(request, {
            origin,
            inventory: [installed, candidate],
            userOverride: candidate.model,
          });
          return {
            repository: candidate.model.repository,
            exactRevision: candidate.model.revision,
            compatibility: result.resolution.compatibility,
            developerCertified: result.resolution.developerCertified,
            reasons: result.resolution.compatibilityReasons.map((reason) => reason.detail),
            contextOrigin: origin === 'oryn://model-manager' ? undefined : origin,
            contextRole: request.role,
          };
        } catch (error) {
          if (error instanceof ModelResolutionError) return undefined;
          throw error;
        }
      }),
    );
    return assessed.filter((value): value is TrustedModelSubstitute => value !== undefined);
  }
}

function syntheticRequest(record: EngineModelInventoryRecord): ModelRequest {
  const technical = record.technicalCapabilities;
  return {
    role: 'model-manager',
    preferredModel: record.model,
    requirements: {
      task: technical?.task,
      architectures: undefined,
      inputModalities: technical?.inputModalities,
      outputModalities: technical?.outputModalities,
      capabilities: Object.fromEntries(
        Object.entries(technical?.capabilities ?? {}).flatMap(([name, value]) =>
          value === true ? [[name, true]] : [],
        ),
      ),
    },
    substitution: { automatic: 'compatible' },
  } as ModelRequest;
}

function sourceUrl(model: ResolvedModelRef): string {
  const [publisher, name] = model.repository.split('/');
  return `https://huggingface.co/${encodeURIComponent(publisher!)}/${encodeURIComponent(name!)}/tree/${encodeURIComponent(model.revision)}`;
}

function modelKey(model: ResolvedModelRef): string {
  return exactModelKey(model);
}

function sameLogicalModel(
  preference: { provider: string; repository: string; revision?: string },
  model: ResolvedModelRef,
): boolean {
  return (
    preference.provider === model.provider &&
    preference.repository === model.repository &&
    (preference.revision === undefined || preference.revision === model.revision)
  );
}

function operationId(prefix: string): string {
  return `${prefix}_${randomUUID().replaceAll('-', '')}`;
}
