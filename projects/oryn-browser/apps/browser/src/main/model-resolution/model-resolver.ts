import { comparable, type Snapshot } from '@ocno/oryn-registry';
import type { RegistryService } from './registry-service';
import { exactModelKey } from '@ocno/oryn-protocol';
import { taskRequirementsFrom, taskRequirementFailures } from '@ocno/oryn-protocol';
import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  ResolvedModelRef,
  TechnicalCapabilityValue,
} from '@ocno/oryn-protocol';

import { normalizeModelRef, validateModelRequest } from './model-request';
import type { ModelPreferenceStore, OptimizationPreference } from './model-preference-store';
import type {
  Compatibility,
  CompatibilityReason,
  ModelCandidateAssessment,
  ModelInspector,
  ModelRef,
  ModelRequest,
  ModelResolutionContext,
  ModelResolutionResult,
  ModelRequirements,
} from './types';

type CandidateSource = ModelCandidateAssessment['sources'][number];

export class ModelResolutionError extends Error {
  constructor(
    readonly code:
      | 'NO_EXECUTABLE_MODEL'
      | 'MODEL_NOT_EXECUTABLE'
      | 'MODEL_SECURITY_BLOCKED'
      | 'MODEL_REFERENCE_INVALID',
    message: string,
  ) {
    super(message);
    this.name = 'ModelResolutionError';
  }
}

export class ModelResolver {
  constructor(
    private readonly inspector: ModelInspector,
    private readonly preferences: ModelPreferenceStore,
    private readonly registry?: RegistryService,
  ) {}

  async resolve(value: unknown, context: ModelResolutionContext): Promise<ModelResolutionResult> {
    const request = validateModelRequest(value);
    const origin = new URL(context.origin).origin;
    const snapshot = this.registry?.capture();
    if (context.signal?.aborted) throw new Error('REQUEST_ABORTED');
    const candidates = await this.generateCandidates(request, origin, context, snapshot);
    if (context.signal?.aborted) throw new Error('REQUEST_ABORTED');
    const inspected: Array<ModelCandidateAssessment | undefined> = [];
    // One outstanding inspection per admitted request leaves Engine control capacity
    // for unrelated origins and stops new work after the caller cancels.
    for (const { ref, sources } of candidates) {
      if (context.signal?.aborted) throw new Error('REQUEST_ABORTED');
      try {
        inspected.push(
          await this.assess(ref, sources, request, context.inventory, snapshot, context.signal),
        );
      } catch (error) {
        if (
          sources.includes('override') ||
          (sources.includes('preferred') &&
            (!request.selection || request.selection.intent === 'exact-artifact'))
        )
          throw error;
        inspected.push(undefined);
      }
      if (context.signal?.aborted) throw new Error('REQUEST_ABORTED');
    }
    const assessments = consolidateAssessments(
      inspected.filter((v): v is ModelCandidateAssessment => !!v),
      request.requirements,
    );
    const preferred = assessments.find((item) => item.sources.includes('preferred'));
    if (request.preferredModel !== undefined && !preferred && !request.selection) {
      throw new ModelResolutionError(
        'MODEL_REFERENCE_INVALID',
        'Preferred model could not be resolved.',
      );
    }

    const source =
      request.selection?.intent === 'source-checkpoint'
        ? (preferred?.model ?? normalizeModelRef(request.selection.sourceCheckpoint))
        : request.selection?.intent === 'preferred-variant'
          ? (assessments.find((c) => c.sources.includes('checkpoint'))?.model ??
            request.selection.sourceCheckpoint)
          : undefined;
    const sourceVariant =
      preferred && snapshot ? this.registry?.variant(preferred.model, snapshot) : undefined;
    const checkpoint =
      source ?? (sourceVariant?.lineage === 'verified' ? sourceVariant.checkpoint : undefined);
    for (const candidate of assessments) {
      const variant = snapshot ? this.registry?.variant(candidate.model, snapshot) : undefined;
      candidate.lineage = variant?.lineage ?? 'unknown';
      const declarations = candidate.profile.lineageDeclarations ?? [];
      const conflicting =
        !!variant?.checkpoint &&
        variant.relationship !== 'original' &&
        declarations.length > 0 &&
        (declarations.length !== 1 || declarations[0] !== variant.checkpoint.repository);
      if (conflicting) candidate.lineage = 'conflicting';
      candidate.sameCheckpoint =
        !conflicting &&
        !!checkpoint &&
        ((candidate.model.repository === checkpoint.repository &&
          candidate.model.revision === checkpoint.revision &&
          'files' in checkpoint &&
          Array.isArray(checkpoint.files) &&
          JSON.stringify([...(candidate.model.files ?? [])].sort()) ===
            JSON.stringify([...checkpoint.files].sort()) &&
          !variant) ||
          (variant?.lineage === 'verified' &&
            ['original', 'conversion', 'quantization'].includes(variant.relationship) &&
            variant.checkpoint?.repository === checkpoint.repository &&
            variant.checkpoint?.revision === checkpoint.revision));
    }
    const eligible = assessments.filter(
      (c) =>
        isSelectable(c) &&
        isAutomaticSelectionEligible(c) &&
        (request.substitution?.automatic !== 'same-checkpoint' ||
          c.sameCheckpoint ||
          c.sources.includes('preferred')),
    );
    const evidence = eligible.map((c) => c.profile.recommendationEvidence);
    if (
      eligible.length > 1 &&
      evidence.every(
        (e) =>
          e &&
          e.sampleCount >= 2 &&
          e.measuredAt <= Date.now() &&
          Date.now() < e.expiresAt &&
          e.comparisonKey === evidence[0]?.comparisonKey &&
          e.metric === evidence[0]?.metric &&
          e.phase === evidence[0]?.phase,
      )
    ) {
      for (const c of eligible) {
        const e = c.profile.recommendationEvidence!;
        c.performanceRank = e.metric === 'decode-tokens-per-second' ? -e.value : e.value;
      }
    }
    if (
      snapshot &&
      this.registry &&
      eligible.length > 1 &&
      eligible.every((c) => c.performanceRank === undefined)
    ) {
      const references = eligible.map((c) => this.registry!.referenceEvidence(c.profile, snapshot));
      if (references.every((e) => e && references[0] && comparable(e, references[0]))) {
        for (const [index, candidate] of eligible.entries()) {
          const e = references[index]!;
          candidate.referenceRank = e.metric === 'decode-tokens-per-second' ? -e.value : e.value;
          candidate.recommendation!.reasons.push(
            `Reference evidence ${e.id} (${e.kind}); not a local measurement.`,
          );
        }
      }
    }
    const selected = this.select(request, assessments, context.userOverride !== undefined);
    const resolutionReason = this.resolutionReason(
      request,
      selected,
      preferred,
      context.userOverride !== undefined,
    );
    return {
      resolution: {
        preferredModel: preferred?.model ?? selected.model,
        resolvedModel: selected.model,
        compatibility: selected.compatibility,
        executability: selected.executability,
        security: selected.security,
        trust: modelTrust(selected.profile),
        developerCertified: selected.developerCertified,
        substituted: preferred ? modelKey(selected.model) !== modelKey(preferred.model) : false,
        overriddenByUser: context.userOverride !== undefined,
        resolutionReason,
        compatibilityReasons: selected.compatibilityReasons,
        downloadRequired: !selected.installed,
        ...(selected.recommendation
          ? {
              selection: {
                version: 1 as const,
                kind: selected.sources.includes('preferred')
                  ? ('requested-artifact' as const)
                  : selected.sameCheckpoint
                    ? ('same-checkpoint' as const)
                    : ('different-model' as const),
                source:
                  selected.performanceRank === undefined
                    ? selected.recommendation.source
                    : ('local-measurements' as const),
                confidence:
                  selected.performanceRank === undefined
                    ? selected.recommendation.confidence
                    : ('measured' as const),
                catalogVersion: selected.recommendation.catalogVersion,
                rulesVersion: selected.recommendation.rulesVersion,
                ruleIds: selected.recommendation.ruleIds,
                reasons: [
                  ...selected.recommendation.reasons,
                  ...(selected.sameCheckpoint && !selected.sources.includes('preferred')
                    ? ['Verified checkpoint lineage; quantization may change output quality.']
                    : []),
                  ...(!selected.sameCheckpoint && !selected.sources.includes('preferred')
                    ? ['This selection uses a different model.']
                    : []),
                ],
              },
            }
          : {}),
      },
      selected,
      candidates: assessments.sort((a, b) => this.compareCandidates(request, a, b)),
    };
  }

  private async generateCandidates(
    request: ModelRequest,
    origin: string,
    context: ModelResolutionContext,
    snapshot?: Snapshot,
  ): Promise<Array<{ ref: ResolvedModelRef; sources: CandidateSource[] }>> {
    const values = new Map<string, { ref: ResolvedModelRef; sources: CandidateSource[] }>();
    const add = (ref: ModelRef, source: CandidateSource): void => {
      const normalized = normalizeModelRef(ref);
      const key = modelKey(normalized);
      const existing = values.get(key);
      if (existing) {
        if (!existing.sources.includes(source)) existing.sources.push(source);
      } else {
        values.set(key, { ref: normalized, sources: [source] });
      }
    };
    if (request.preferredModel !== undefined) {
      add(request.preferredModel, 'preferred');
      if (request.selection?.intent === 'preferred-variant' && request.selection.sourceCheckpoint)
        add(request.selection.sourceCheckpoint, 'checkpoint');
    } else {
      const catalog = this.inspector.matchModelCatalog;
      if (!catalog) {
        throw new ModelResolutionError(
          'NO_EXECUTABLE_MODEL',
          'The trusted runtime-pack catalog is unavailable.',
        );
      }
      const result = await catalog.call(
        this.inspector,
        request.requirements!.task!,
        request.requirements!.outputModalities!,
      );
      for (const model of result.models) add(model, 'catalog');
    }
    if (snapshot && this.registry) {
      const requested =
        request.selection?.intent === 'source-checkpoint'
          ? request.selection.sourceCheckpoint
          : request.selection?.intent === 'preferred-variant'
            ? (request.selection.sourceCheckpoint ?? request.preferredModel)
            : request.preferredModel;
      const known =
        requested &&
        this.registry
          .variants(snapshot)
          .find(
            (v) =>
              v.artifact.repository === requested.repository &&
              (!requested.revision ||
                requested.revision === 'main' ||
                requested.revision === v.artifact.revision),
          );
      const source = known?.checkpoint ?? requested;
      if (source)
        for (const variant of this.registry.variants(snapshot)) {
          if (
            variant.checkpoint?.repository === source.repository &&
            (!source.revision ||
              source.revision === 'main' ||
              variant.checkpoint.revision === source.revision)
          )
            add({ provider: 'huggingface', ...variant.artifact }, 'catalog');
        }
    }
    for (const certified of ('certifiedModels' in request ? request.certifiedModels : []) ?? []) {
      add(certified, 'certified');
    }
    for (const installed of context.inventory) {
      if (
        installed.admissionScope === 'huggingface-page' ||
        installed.technicalCapabilities?.architectures?.includes('OrynReviewedLfm3ForCausalLM')
      )
        continue;
      const installedTask = installed.technicalCapabilities?.task;
      const requestedTask = request.requirements?.task;
      if (requestedTask && installedTask && installedTask !== requestedTask) {
        continue;
      }
      add(installed.model, 'installed');
    }
    const site = this.preferences.getSite(origin, request.role);
    if (site) add(site, 'site-preference');
    const global = this.preferences.getGlobal(request.role);
    if (global) add(global, 'global-preference');
    if (context.userOverride) add(context.userOverride, 'override');
    if (values.size > 64) {
      throw new ModelResolutionError(
        'MODEL_REFERENCE_INVALID',
        'Model candidate set is too large.',
      );
    }
    return [...values.values()].sort(
      (a, b) => Number(b.sources.includes('installed')) - Number(a.sources.includes('installed')),
    );
  }

  private async assess(
    requested: ResolvedModelRef,
    sources: CandidateSource[],
    request: ModelRequest,
    inventory: EngineModelInventoryRecord[],
    snapshot?: Snapshot,
    signal?: AbortSignal,
  ): Promise<ModelCandidateAssessment> {
    const hint =
      snapshot &&
      this.registry?.decision(
        {
          model: requested,
          technicalCapabilities: { task: request.requirements?.task },
        } as CandidateExecutionProfile,
        snapshot,
      );
    if (signal?.aborted) throw new Error('REQUEST_ABORTED');
    let profile = await this.inspector.inspectModel(
      requested,
      taskRequirementsFrom(request.requirements),
      hint ? { snapshotDigest: hint.snapshotDigest, engines: hint.engines } : undefined,
    );
    // One snapshot, at most two inspections. Reinspect the immutable revision;
    // an explicit file pin is retained, while an unpinned source may select the
    // preferred provider's complete representation. Engine still owns all gates.
    const recommendation = snapshot && this.registry?.decision(profile, snapshot);
    if (
      recommendation &&
      JSON.stringify(recommendation.engines) !== JSON.stringify(hint?.engines)
    ) {
      if (signal?.aborted) throw new Error('REQUEST_ABORTED');
      profile = await this.inspector.inspectModel(
        { ...requested, revision: profile.exactRevision },
        taskRequirementsFrom(request.requirements),
        { snapshotDigest: recommendation.snapshotDigest, engines: recommendation.engines },
      );
    }
    const selectedEngine =
      profile.runtimePack?.packId ??
      (profile.runner === 'transformers' ? 'oryn.transformers-pytorch' : undefined);
    // Explicit references/preferences cannot promote Browser-only admission
    // into an SDK offer either. This check precedes trust/permission ranking.
    if (
      profile.admissionScope === 'huggingface-page' ||
      profile.sourceRecognition?.admitted === true ||
      profile.technicalCapabilities.architectures?.includes('OrynReviewedLfm3ForCausalLM')
    ) {
      profile = {
        ...profile,
        executable: false,
        executableBackends: [],
        loadDisposition: 'unsupported',
        blockingReasons: [...(profile.blockingReasons ?? []), 'BROWSER_ONLY_ADMISSION'],
      };
    }
    if (
      recommendation?.engines.length &&
      selectedEngine &&
      selectedEngine !== recommendation.engines[0]
    ) {
      recommendation.reasons.push(
        'Engine retained a different route after applying admission, evidence and existing runtime bindings.',
      );
    }
    // Revocation is keyed by the selected artifact, even when the caller only
    // supplied a checkpoint and its catalog variant has since been removed.
    const selectedArtifact = profile.selectedFiles?.length
      ? { ...profile.model, files: profile.selectedFiles.map((file) => file.name).sort() }
      : profile.model;
    if (snapshot && this.registry?.isRevoked(selectedArtifact, snapshot)) {
      profile.executable = false;
      profile.executableBackends = [];
    }
    if (request.selection && !profile.model.files && profile.selectedFiles?.length) {
      profile.model = { ...profile.model, files: profile.selectedFiles.map((f) => f.name).sort() };
    }
    const exact = profile.model;
    const installedRecord = inventory.find((item) => modelKey(item.model) === modelKey(exact));
    const developerCertified = sources.includes('certified');
    const compatibilityReasons = evaluateRequirements(request.requirements, profile);
    const compatibility = classifyCompatibility(developerCertified, compatibilityReasons);
    const auditedAdapter = profile.runtimePack?.compatibilityLevel === 'audited-adapter';
    const security =
      (profile.requiresCustomCode && !auditedAdapter) ||
      profile.blockingReasons?.some(isSecurityBlock)
        ? 'blocked'
        : 'allowed';
    const executability = profile.executable ? 'executable' : 'not-executable';
    return {
      model: exact,
      profile,
      compatibility,
      executability,
      security,
      developerCertified,
      installed:
        installedRecord?.installationState === 'installed' && !profile.installationRequired,
      loaded: installedRecord?.residencyState === 'loaded',
      sources,
      compatibilityReasons,
      ...(snapshot && this.registry
        ? {
            recommendation,
            representation: this.registry.variant(profile.model, snapshot)?.representation,
          }
        : {}),
    };
  }

  private select(
    request: ModelRequest,
    candidates: ModelCandidateAssessment[],
    hasOverride: boolean,
  ): ModelCandidateAssessment {
    if (hasOverride) {
      const override = candidates.find((item) => item.sources.includes('override'));
      if (!override)
        throw new ModelResolutionError('MODEL_REFERENCE_INVALID', 'Override is invalid.');
      if (request.selection && override.compatibility === 'incompatible')
        throw new ModelResolutionError(
          'MODEL_NOT_EXECUTABLE',
          'Required features and limits remain binding.',
        );
      this.assertSelectable(override);
      return override;
    }
    if (request.selection?.intent === 'exact-artifact') {
      const pinned = candidates.find((item) => item.sources.includes('preferred'));
      if (!pinned)
        throw new ModelResolutionError(
          'MODEL_REFERENCE_INVALID',
          'Pinned artifact is unavailable.',
        );
      if (pinned.compatibility === 'incompatible')
        throw new ModelResolutionError(
          'MODEL_NOT_EXECUTABLE',
          'Required features and limits remain binding.',
        );
      this.assertSelectable(pinned);
      return pinned;
    }
    const site = candidates.find((item) => item.sources.includes('site-preference'));
    if (
      site &&
      isSelectable(site) &&
      isAutomaticSelectionEligible(site) &&
      (!request.selection ||
        request.substitution?.automatic === 'compatible' ||
        site.sameCheckpoint ||
        site.sources.includes('preferred'))
    )
      return site;
    const global = candidates.find((item) => item.sources.includes('global-preference'));
    if (
      global &&
      isSelectable(global) &&
      isAutomaticSelectionEligible(global) &&
      (!request.selection ||
        request.substitution?.automatic === 'compatible' ||
        global.sameCheckpoint ||
        global.sources.includes('preferred'))
    )
      return global;

    const mode = request.substitution?.automatic ?? 'none';
    const automaticCandidates = candidates.filter(
      (candidate) =>
        isAutomaticSelectionEligible(candidate) &&
        isSelectable(candidate) &&
        (!request.selection || ['compatible', 'certified'].includes(candidate.compatibility)),
    );
    let pool: ModelCandidateAssessment[];
    if (mode === 'none') {
      pool = automaticCandidates.filter((item) => item.sources.includes('preferred'));
    } else if (mode === 'same-checkpoint') {
      pool = automaticCandidates.filter(
        (item) => item.sources.includes('preferred') || item.sameCheckpoint,
      );
    } else if (mode === 'certified-only') {
      pool = automaticCandidates.filter(
        (item) => item.sources.includes('preferred') || item.developerCertified,
      );
    } else {
      pool = automaticCandidates.filter(
        (item) => item.compatibility === 'certified' || item.compatibility === 'compatible',
      );
      if (pool.length === 0 && !request.selection) {
        pool = automaticCandidates.filter((item) => item.compatibility === 'uncertain');
      }
    }
    const selectable = pool.sort((a, b) => this.compareCandidates(request, a, b));
    if (!selectable[0]) {
      const preferred = candidates.find((item) => item.sources.includes('preferred'));
      if (preferred && preferred.security !== 'allowed') {
        throw new ModelResolutionError(
          'MODEL_SECURITY_BLOCKED',
          'Model security policy blocked selection.',
        );
      }
      throw new ModelResolutionError(
        'NO_EXECUTABLE_MODEL',
        'No executable model satisfies resolution policy.',
      );
    }
    return selectable[0];
  }

  private assertSelectable(candidate: ModelCandidateAssessment): void {
    if (
      candidate.compatibilityReasons.some(
        (reason) => reason.requirement.startsWith('TASK_') && reason.outcome === 'not-satisfied',
      )
    ) {
      throw new ModelResolutionError(
        'MODEL_NOT_EXECUTABLE',
        'The selected model cannot fulfill the required task contract.',
      );
    }
    if (candidate.security !== 'allowed') {
      throw new ModelResolutionError(
        'MODEL_SECURITY_BLOCKED',
        'User override cannot bypass security policy.',
      );
    }
    if (candidate.executability !== 'executable') {
      throw new ModelResolutionError(
        'MODEL_NOT_EXECUTABLE',
        'User override cannot bypass runtime executability.',
      );
    }
  }

  private compareCandidates(
    request: ModelRequest,
    a: ModelCandidateAssessment,
    b: ModelCandidateAssessment,
  ): number {
    const compatibility = { certified: 0, compatible: 1, uncertain: 2, incompatible: 3 };
    const compatibilityDifference = compatibility[a.compatibility] - compatibility[b.compatibility];
    if (compatibilityDifference !== 0) return compatibilityDifference;
    const family = this.preferences.getFamily(request.role);
    if (family) {
      const aMatches = matchesFamily(a, family);
      const bMatches = matchesFamily(b, family);
      if (aMatches !== bMatches) return aMatches ? -1 : 1;
    }
    if (this.preferences.preferInstalled && a.installed !== b.installed)
      return a.installed ? -1 : 1;
    if (a.loaded !== b.loaded) return a.loaded ? -1 : 1;
    if (
      a.performanceRank !== undefined &&
      b.performanceRank !== undefined &&
      ['balanced', 'speed'].includes(this.preferences.optimization)
    ) {
      const difference = a.performanceRank - b.performanceRank;
      if (difference) return difference;
    }
    if (
      a.referenceRank !== undefined &&
      b.referenceRank !== undefined &&
      ['balanced', 'speed'].includes(this.preferences.optimization)
    ) {
      const difference = a.referenceRank - b.referenceRank;
      if (difference) return difference;
    }
    if (
      a.recommendation &&
      b.recommendation &&
      ['balanced', 'speed'].includes(this.preferences.optimization)
    ) {
      const rank = (c: ModelCandidateAssessment): number => {
        const engine = c.profile.runtimePack?.packId ?? c.profile.runner;
        const index = c.recommendation!.engines.indexOf(engine ?? '');
        return index < 0 ? c.recommendation!.engines.length : index;
      };
      const difference = rank(a) - rank(b);
      if (difference) return difference;
      const representationRank = (c: ModelCandidateAssessment): number => {
        const index = c.recommendation!.representations.indexOf(c.representation ?? '');
        return index < 0 ? c.recommendation!.representations.length : index;
      };
      const representationDifference = representationRank(a) - representationRank(b);
      if (representationDifference) return representationDifference;
    }
    return compareOptimization(a, b, this.preferences.optimization);
  }

  private resolutionReason(
    request: ModelRequest,
    selected: ModelCandidateAssessment,
    preferred: ModelCandidateAssessment | undefined,
    override: boolean,
  ): ModelResolutionResult['resolution']['resolutionReason'] {
    if (override) return 'user-override';
    if (!preferred) return 'capability-match';
    if (selected.sources.includes('site-preference')) return 'site-preference';
    if (selected.sources.includes('global-preference')) return 'global-user-preference';
    if (modelKey(selected.model) === modelKey(preferred.model)) return 'preferred-model';
    const family = this.preferences.getFamily(request.role);
    if (family && matchesFamily(selected, family)) return 'global-user-preference';
    if (selected.developerCertified) return 'developer-certified-alternative';
    if (selected.installed) return 'already-installed';
    return 'automatic-compatible-substitution';
  }
}

function matchesFamily(candidate: ModelCandidateAssessment, family: string): boolean {
  const needle = family.toLowerCase().replaceAll(/[^a-z0-9]/gu, '');
  if (!needle) return false;
  return [
    candidate.model.repository,
    ...(candidate.profile.technicalCapabilities.architectures ?? []),
  ]
    .map((value) => value.toLowerCase().replaceAll(/[^a-z0-9]/gu, ''))
    .some((value) => value.includes(needle));
}

function evaluateRequirements(
  requirements: ModelRequirements | undefined,
  profile: CandidateExecutionProfile,
): CompatibilityReason[] {
  if (!requirements) return [];
  const technical = profile.technicalCapabilities;
  const reasons: CompatibilityReason[] = [];
  for (const failure of taskRequirementFailures(
    profile.taskProfile,
    taskRequirementsFrom(requirements) ?? {},
  )) {
    reasons.push({
      requirement: failure,
      outcome: 'not-satisfied',
      detail: 'The model does not satisfy the requested task contract.',
    });
  }
  compareScalar(reasons, 'task', requirements.task, technical.task);
  compareOverlap(reasons, 'architectures', requirements.architectures, technical.architectures);
  if (requirements.parameters?.minBillions !== undefined) {
    compareMinimum(
      reasons,
      'minimum parameter count',
      requirements.parameters.minBillions * 1_000_000_000,
      technical.parameterCount,
    );
  }
  if (requirements.parameters?.maxBillions !== undefined) {
    compareMaximum(
      reasons,
      'maximum parameter count',
      requirements.parameters.maxBillions * 1_000_000_000,
      technical.parameterCount,
    );
  }
  if (requirements.minimumContextLength !== undefined) {
    compareMinimum(
      reasons,
      'minimum context length',
      requirements.minimumContextLength,
      technical.contextLength,
    );
  }
  compareSubset(
    reasons,
    'input modalities',
    requirements.inputModalities,
    technical.inputModalities,
  );
  compareSubset(
    reasons,
    'output modalities',
    requirements.outputModalities,
    technical.outputModalities,
  );
  for (const [capability, expected] of Object.entries(requirements.capabilities ?? {})) {
    const actual = technical.capabilities?.[capability];
    compareCapability(reasons, capability, expected, actual);
  }
  if (requirements.memory?.maximumRecommendedBytes !== undefined) {
    compareMaximum(
      reasons,
      'maximum recommended memory',
      requirements.memory.maximumRecommendedBytes,
      profile.estimatedMemoryBytes,
    );
  }
  return reasons;
}

function classifyCompatibility(
  developerCertified: boolean,
  reasons: CompatibilityReason[],
): Compatibility {
  if (reasons.some((item) => item.outcome === 'not-satisfied')) return 'incompatible';
  if (reasons.some((item) => item.outcome === 'unknown')) return 'uncertain';
  return developerCertified ? 'certified' : 'compatible';
}

function compareScalar(
  reasons: CompatibilityReason[],
  requirement: string,
  expected: string | undefined,
  actual: string | undefined,
): void {
  if (expected === undefined) return;
  reasons.push(
    reason(requirement, actual === undefined ? undefined : actual === expected, `${expected}`),
  );
}

function compareOverlap(
  reasons: CompatibilityReason[],
  requirement: string,
  expected: string[] | undefined,
  actual: string[] | undefined,
): void {
  if (!expected?.length) return;
  reasons.push(
    reason(
      requirement,
      actual === undefined ? undefined : expected.some((item) => actual.includes(item)),
      expected.join(', '),
    ),
  );
}

function compareSubset(
  reasons: CompatibilityReason[],
  requirement: string,
  expected: string[] | undefined,
  actual: string[] | undefined,
): void {
  if (!expected?.length) return;
  reasons.push(
    reason(
      requirement,
      actual === undefined ? undefined : expected.every((item) => actual.includes(item as never)),
      expected.join(', '),
    ),
  );
}

function compareMinimum(
  reasons: CompatibilityReason[],
  requirement: string,
  expected: number,
  actual: number | undefined,
): void {
  reasons.push(
    reason(requirement, actual === undefined ? undefined : actual >= expected, String(expected)),
  );
}

function compareMaximum(
  reasons: CompatibilityReason[],
  requirement: string,
  expected: number,
  actual: number | undefined,
): void {
  reasons.push(
    reason(requirement, actual === undefined ? undefined : actual <= expected, String(expected)),
  );
}

function compareCapability(
  reasons: CompatibilityReason[],
  requirement: string,
  expected: boolean | undefined,
  actual: TechnicalCapabilityValue | undefined,
): void {
  if (expected === undefined) return;
  reasons.push(
    reason(
      requirement,
      actual === undefined || actual === 'unknown' ? undefined : actual === expected,
      String(expected),
    ),
  );
}

function reason(
  requirement: string,
  result: boolean | undefined,
  detail: string,
): CompatibilityReason {
  return {
    requirement,
    outcome: result === undefined ? 'unknown' : result ? 'satisfied' : 'not-satisfied',
    detail,
  };
}

function isSecurityBlock(reason: string): boolean {
  return (
    reason === 'CUSTOM_CODE_REQUIRED' ||
    reason === 'MODEL_SECURITY_BLOCKED' ||
    reason === 'UNSAFE_MODEL_FORMAT'
  );
}

function isSelectable(candidate: ModelCandidateAssessment): boolean {
  return (
    !candidate.compatibilityReasons.some(
      (reason) => reason.requirement.startsWith('TASK_') && reason.outcome === 'not-satisfied',
    ) &&
    candidate.executability === 'executable' &&
    candidate.security === 'allowed'
  );
}

function isAutomaticSelectionEligible(candidate: ModelCandidateAssessment): boolean {
  if (candidate.profile.loadDisposition !== undefined) {
    return (
      candidate.profile.loadDisposition === 'verified-load' ||
      candidate.sources.some((source) =>
        ['preferred', 'site-preference', 'global-preference'].includes(source),
      )
    );
  }
  const conformance = candidate.profile.runtimePack?.trustedMetadata?.conformance.status;
  return conformance !== 'fixture-only' || candidate.sources.includes('preferred');
}

function modelTrust(
  profile: CandidateExecutionProfile,
): 'oryn-verified' | 'user-approved-unverified' {
  if (profile.loadDisposition !== undefined) {
    return profile.loadDisposition === 'verified-load'
      ? 'oryn-verified'
      : 'user-approved-unverified';
  }
  return profile.runtimePack &&
    profile.runtimePack.trustedMetadata?.conformance.status !== 'fixture-only'
    ? 'oryn-verified'
    : 'user-approved-unverified';
}

function compareOptimization(
  a: ModelCandidateAssessment,
  b: ModelCandidateAssessment,
  preference: OptimizationPreference,
): number {
  const aParameters = a.profile.technicalCapabilities.parameterCount;
  const bParameters = b.profile.technicalCapabilities.parameterCount;
  const aMemory = a.profile.estimatedMemoryBytes;
  const bMemory = b.profile.estimatedMemoryBytes;
  if (preference === 'quality' && aParameters !== undefined && bParameters !== undefined) {
    return bParameters - aParameters;
  }
  if (
    (preference === 'speed' || preference === 'memory') &&
    aMemory !== undefined &&
    bMemory !== undefined
  ) {
    return aMemory - bMemory;
  }
  return modelKey(a.model).localeCompare(modelKey(b.model));
}

function modelKey(model: ResolvedModelRef): string {
  return exactModelKey(model);
}

function consolidateAssessments(
  values: ModelCandidateAssessment[],
  requirements: ModelRequirements | undefined,
): ModelCandidateAssessment[] {
  const result = new Map<string, ModelCandidateAssessment>();
  for (const candidate of values) {
    const key = modelKey(candidate.model);
    const existing = result.get(key);
    if (!existing) {
      result.set(key, candidate);
      continue;
    }
    existing.sources = [...new Set([...existing.sources, ...candidate.sources])];
    existing.developerCertified ||= candidate.developerCertified;
    existing.installed ||= candidate.installed;
    existing.loaded ||= candidate.loaded;
    existing.compatibilityReasons = evaluateRequirements(requirements, existing.profile);
    existing.compatibility = classifyCompatibility(
      existing.developerCertified,
      existing.compatibilityReasons,
    );
  }
  return [...result.values()];
}
