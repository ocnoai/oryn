import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { assertRuntimeResourceSnapshot } from '@ocno/oryn-protocol';

import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  ResolvedModelRef,
  RuntimeResourceSnapshot,
} from '@ocno/oryn-protocol';

import { parseHuggingFaceModelPage } from './hugging-face-model-url';
import type { HuggingFacePageActionState, TrustedModelOperation } from '../../shared/ipc';
import { formatStorageSize as formatBytes } from '../../shared/format-storage-size';
import {
  isOrynVerified,
  isUserTrustEligible,
  taskContractForProfile,
  type ModelWorkspaceTrustMode,
} from '../model-workspace/model-workspace-eligibility';

const INSPECTION_DEBOUNCE_MS = 220;
const CACHE_TTL_MS = 5 * 60_000;
const CACHE_LIMIT = 64;
// Routing can add a summary reason alongside the selected model's resource reason.
// Prefer the concrete reason regardless of the order in the Engine response.
const RESOURCE_BLOCKING_REASONS = [
  'DISK_SPACE_INSUFFICIENT',
  'MODEL_MEMORY_INSUFFICIENT',
  'MODEL_MEMORY_BUDGET_EXCEEDED',
  'RUNTIME_RESOURCES_INSUFFICIENT',
];

export interface PageActionTabIdentity {
  tabId: string;
  documentId: string;
  url: string;
}

export interface PageActionEngine {
  inspectModel(model: ResolvedModelRef): Promise<CandidateExecutionProfile>;
  inspectHuggingFaceModel?(
    model: ResolvedModelRef,
    context: { sourceURL: string; tabId: string; documentId: string },
    signal?: AbortSignal,
  ): Promise<CandidateExecutionProfile>;
  getModelInventory(): Promise<{ models: EngineModelInventoryRecord[] }>;
  getRuntimeResources(): Promise<RuntimeResourceSnapshot>;
  installModel(operationId: string, model: ResolvedModelRef): Promise<unknown>;
  installHuggingFaceModel?(
    operationId: string,
    model: ResolvedModelRef,
    context: { sourceURL: string; tabId: string; documentId: string },
  ): Promise<unknown>;
  cancelModelInstall(operationId: string): Promise<boolean>;
  on(event: string, listener: (value: unknown) => void): unknown;
  removeListener(event: string, listener: (value: unknown) => void): unknown;
}

interface CachedInspection {
  expiresAt: number;
  profile: CandidateExecutionProfile;
}

export class HuggingFacePageActionController extends EventEmitter {
  private state: HuggingFacePageActionState | undefined;
  private identity: PageActionTabIdentity | undefined;
  private revision = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private operationAbort: AbortController | undefined;
  private sourceInspectionAbort: AbortController | undefined;
  private activeInstall: { operationId: string; state: HuggingFacePageActionState } | undefined;
  private inspectionLane: Promise<void> = Promise.resolve();
  private readonly cache = new Map<string, CachedInspection>();
  private readonly resourcesListener = (resources: unknown) => {
    if (!this.state) return;
    try {
      assertRuntimeResourceSnapshot(resources);
    } catch {
      return;
    }
    const availableMemoryBytes = availableModelMemory(resources);
    if (this.state.availableMemoryBytes !== availableMemoryBytes) {
      this.setState({ ...this.state, availableMemoryBytes });
    }
  };
  private readonly inventoryListener = () => {
    // model.install publishes inventory changes while its authoritative request is still in
    // flight. Re-inspecting here would discard the operation id/progress and race a second Engine
    // request against the install. confirmInstall owns the terminal installed/failed transition.
    if (this.identity && this.state?.status !== 'downloading') {
      this.activeTabChangedAfterInvalidation();
    }
  };

  constructor(
    private readonly engine: PageActionEngine,
    private readonly getActiveTab: () => PageActionTabIdentity,
  ) {
    super();
    engine.on('model-inventory-updated', this.inventoryListener);
    engine.on('resources-updated', this.resourcesListener);
  }

  snapshot(): HuggingFacePageActionState | undefined {
    return this.state ? structuredClone(this.state) : undefined;
  }

  activeTabChanged(): void {
    const next = this.getActiveTab();
    const page = parseHuggingFaceModelPage(next.url);
    if (!page) {
      this.invalidate();
      return;
    }
    if (
      this.identity?.tabId === next.tabId &&
      this.identity.documentId === next.documentId &&
      this.identity.url === next.url
    ) {
      return;
    }
    this.sourceInspectionAbort?.abort();
    this.revision += 1;
    const revision = this.revision;
    this.identity = next;
    if (this.timer) clearTimeout(this.timer);
    this.setState({
      status: 'inspecting',
      repository: page.repository,
      sourceURL: page.sourceURL,
      selectedFiles: [],
      message: 'Checking exact-model support…',
    });
    this.timer = setTimeout(() => {
      this.timer = undefined;
      this.inspectionLane = this.inspectionLane
        .catch(() => undefined)
        .then(() => this.inspect(next, revision));
    }, INSPECTION_DEBOUNCE_MS);
  }

  async confirmInstall(trustMode: ModelWorkspaceTrustMode): Promise<void> {
    const current = this.requireCurrent(trustMode === 'oryn-verified' ? 'loadable' : 'unverified');
    if (current.trustMode !== trustMode) {
      throw new Error('HUGGING_FACE_PAGE_ACTION_TRUST_MODE_MISMATCH');
    }
    const exact = exactModel(current);
    if (this.activeInstall) await this.cancelInstall(this.activeInstall.operationId);
    const operationId = `hf_install_${randomUUID().replaceAll('-', '')}`;
    const abort = new AbortController();
    this.operationAbort?.abort();
    this.operationAbort = abort;
    this.activeInstall = { operationId, state: current };
    const identity = this.identity as PageActionTabIdentity;
    const listener = (value: unknown) => {
      if (!isInstallProgress(value) || value.operationId !== operationId || abort.signal.aborted) {
        return;
      }
      this.publishOperation(current, operationId, 'downloading', {
        downloadedBytes: value.downloadedBytes,
        totalBytes: value.totalBytes,
      });
      if (!this.isCurrent(identity)) return;
      this.setState({
        ...current,
        status: 'downloading',
        operationId,
        downloadedBytes: value.downloadedBytes,
        totalBytes: value.totalBytes,
        message: 'Downloading and verifying the exact revision…',
      });
    };
    this.engine.on('model-install-progress', listener);
    this.publishOperation(current, operationId, 'downloading', {
      downloadedBytes: 0,
      totalBytes: current.diskBytes,
    });
    this.setState({
      ...current,
      status: 'downloading',
      operationId,
      downloadedBytes: 0,
      totalBytes: current.diskBytes,
      message: 'Preparing the exact model files…',
    });
    try {
      if (this.engine.installHuggingFaceModel) {
        await this.engine.installHuggingFaceModel(operationId, exact, {
          sourceURL: current.sourceURL,
          tabId: identity.tabId,
          documentId: identity.documentId,
        });
      } else {
        await this.engine.installModel(operationId, exact);
      }
      if (!abort.signal.aborted) this.publishOperation(current, operationId, 'ready');
      if (!abort.signal.aborted && this.isCurrent(identity)) {
        this.setState({
          ...current,
          status: 'installed',
          message:
            trustMode === 'oryn-verified'
              ? 'Installed and ready in Oryn.'
              : 'Installed with your explicit trust decision.',
        });
      }
    } catch (error) {
      if (abort.signal.aborted) return;
      const classified = classifyInstallFailure({ ...current, operationId }, error);
      this.publishOperation(current, operationId, 'failed', {
        errorCode: (classified.blockingReasons?.[0] ?? errorCode(error)) || undefined,
      });
      if (this.isCurrent(identity)) {
        this.setState(classified);
      }
    } finally {
      this.engine.removeListener('model-install-progress', listener);
      if (this.operationAbort === abort) this.operationAbort = undefined;
      if (this.activeInstall?.operationId === operationId) this.activeInstall = undefined;
    }
  }

  async cancelInstall(expectedOperationId?: string): Promise<boolean> {
    const activeInstall = this.activeInstall;
    const operationId = activeInstall?.operationId;
    if (!operationId || (expectedOperationId && operationId !== expectedOperationId)) return false;
    this.operationAbort?.abort();
    this.operationAbort = undefined;
    this.activeInstall = undefined;
    this.publishOperation(activeInstall.state, operationId, 'cancelled');
    await this.engine.cancelModelInstall(operationId).catch(() => false);
    if (this.state?.operationId === operationId) this.activeTabChangedAfterInvalidation();
    return true;
  }

  requireInstalled(): HuggingFacePageActionState {
    return this.requireCurrent('installed');
  }

  retry(): void {
    this.cache.clear();
    this.sourceInspectionAbort?.abort();
    const identity = this.identity;
    if (
      identity &&
      this.isCurrent(identity) &&
      this.engine.inspectHuggingFaceModel &&
      (this.state?.blockingReasons?.includes('CUSTOM_CODE_REQUIRED') ||
        this.state?.implementationDetails)
    ) {
      const revision = ++this.revision;
      this.setState({
        ...this.state!,
        status: 'inspecting',
        message: 'Checking reviewed built-in implementations…',
      });
      // Called only by trusted Chrome IPC; the page supplies neither the
      // context nor the action. Automatic navigation inspection stays inert.
      this.inspectionLane = this.inspectionLane
        .catch(() => undefined)
        .then(() => this.inspect(identity, revision, true));
      return;
    }
    this.activeTabChangedAfterInvalidation();
  }

  credentialsChanged(): void {
    this.cache.clear();
    this.activeTabChangedAfterInvalidation();
  }

  destroy(): void {
    if (this.timer) clearTimeout(this.timer);
    this.operationAbort?.abort();
    this.sourceInspectionAbort?.abort();
    this.engine.removeListener('model-inventory-updated', this.inventoryListener);
    this.engine.removeListener('resources-updated', this.resourcesListener);
  }

  private async inspect(
    identity: PageActionTabIdentity,
    revision: number,
    reviewed = false,
  ): Promise<void> {
    if (revision !== this.revision || !this.isCurrent(identity)) return;
    const page = parseHuggingFaceModelPage(identity.url);
    if (!page) return;
    const requestedRevision = page.revision ?? 'main';
    const immutableRequest = /^[0-9a-f]{40,64}$/u.test(requestedRevision);
    const key = `${page.repository}@${requestedRevision}`;
    const sourceAbort = reviewed ? new AbortController() : undefined;
    if (sourceAbort) this.sourceInspectionAbort = sourceAbort;
    try {
      let profile = !reviewed && immutableRequest ? this.cached(key) : undefined;
      if (!profile) {
        const model: ResolvedModelRef = {
          provider: 'huggingface',
          repository: page.repository,
          revision: requestedRevision,
        };
        profile =
          reviewed && this.engine.inspectHuggingFaceModel
            ? await this.engine.inspectHuggingFaceModel(
                model,
                {
                  sourceURL: page.sourceURL,
                  tabId: identity.tabId,
                  documentId: identity.documentId,
                },
                sourceAbort?.signal,
              )
            : await this.engine.inspectModel(model);
        if (!reviewed)
          this.cache.set(`${page.repository}@${profile.exactRevision}`, {
            profile: structuredClone(profile),
            expiresAt: Date.now() + CACHE_TTL_MS,
          });
        while (this.cache.size > CACHE_LIMIT)
          this.cache.delete(this.cache.keys().next().value as string);
      }
      const [inventory, resources] = await Promise.all([
        this.engine.getModelInventory(),
        this.engine.getRuntimeResources().catch(() => undefined),
      ]);
      if (revision !== this.revision || !this.isCurrent(identity)) return;
      this.setState({
        ...toState(page.sourceURL, profile, inventory.models),
        availableMemoryBytes: resources ? availableModelMemory(resources) : undefined,
      });
    } catch (error) {
      if (revision !== this.revision || !this.isCurrent(identity)) return;
      const seed: HuggingFacePageActionState = {
        status: 'failed',
        repository: page.repository,
        sourceURL: page.sourceURL,
        selectedFiles: [],
      };
      this.setState(classifyFailure(seed, error));
    } finally {
      if (this.sourceInspectionAbort === sourceAbort) this.sourceInspectionAbort = undefined;
    }
  }

  private cached(key: string): CandidateExecutionProfile | undefined {
    const item = this.cache.get(key);
    if (!item || item.expiresAt <= Date.now()) {
      this.cache.delete(key);
      return undefined;
    }
    this.cache.delete(key);
    this.cache.set(key, item);
    return structuredClone(item.profile);
  }

  private requireCurrent(status: HuggingFacePageActionState['status']): HuggingFacePageActionState {
    if (!this.state || this.state.status !== status || !this.identity) {
      throw new Error('HUGGING_FACE_PAGE_ACTION_NOT_AVAILABLE');
    }
    const current = this.getActiveTab();
    if (!this.isCurrent(current)) throw new Error('HUGGING_FACE_PAGE_ACTION_STALE');
    return structuredClone(this.state);
  }

  private isCurrent(identity: PageActionTabIdentity): boolean {
    const active = this.getActiveTab();
    return (
      active.tabId === identity.tabId &&
      active.documentId === identity.documentId &&
      active.url === identity.url
    );
  }

  private invalidate(): void {
    this.sourceInspectionAbort?.abort();
    this.revision += 1;
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    this.identity = undefined;
    this.state = undefined;
    this.emit('changed');
  }

  private activeTabChangedAfterInvalidation(): void {
    this.identity = undefined;
    this.state = undefined;
    this.activeTabChanged();
  }

  private setState(state: HuggingFacePageActionState): void {
    this.state = structuredClone(state);
    this.emit('changed');
  }

  private publishOperation(
    state: HuggingFacePageActionState,
    operationId: string,
    stage: TrustedModelOperation['stage'],
    progress: Pick<TrustedModelOperation, 'downloadedBytes' | 'totalBytes' | 'errorCode'> = {},
  ): void {
    let origin = 'https://huggingface.co';
    try {
      origin = new URL(state.sourceURL).origin;
    } catch {
      // The inspected page identity is already trusted; retain the known provider origin.
    }
    this.emit('operation-changed', {
      operationId,
      origin,
      stage,
      repository: state.repository,
      ...(state.exactRevision ? { revision: state.exactRevision } : {}),
      ...progress,
    } satisfies TrustedModelOperation);
  }
}

function toState(
  sourceURL: string,
  profile: CandidateExecutionProfile,
  inventory: EngineModelInventoryRecord[],
): HuggingFacePageActionState {
  const pack = profile.runtimePack;
  const taskContract = taskContractForProfile(profile);
  const verified = isOrynVerified(profile);
  const userTrustEligible = isUserTrustEligible(profile);
  const blocking = profile.blockingReasons ?? [];
  const installed = inventory.some(
    (item) =>
      item.installationState === 'installed' &&
      item.model.repository === profile.model.repository &&
      item.model.revision === profile.exactRevision,
  );
  const base: HuggingFacePageActionState = {
    status: 'unsupported',
    repository: profile.model.repository,
    exactRevision: profile.exactRevision,
    sourceURL,
    task: profile.technicalCapabilities.task,
    taskContract,
    runner: profile.runner,
    compatibility: pack?.compatibilityLevel,
    securityStatus:
      profile.requiresCustomCode && pack?.compatibilityLevel !== 'audited-adapter'
        ? 'blocked'
        : 'allowed',
    license: pack?.trustedMetadata?.license,
    selectedFiles: summarizeSelectedFiles(profile),
    diskBytes: profile.estimatedDiskBytes,
    memoryBytes: profile.estimatedMemoryBytes,
    blockingReasons: blocking,
    ...(profile.sourceRecognition?.state === 'recognized'
      ? {
          implementationNotice:
            'Recognition selects only Oryn-reviewed built-in code. Repository Python is never run. Reviewed code can still contain bugs; source matching does not verify this model’s outputs.',
          implementationDetails: `${profile.sourceRecognition.match === 'exact' ? 'Exact' : 'Structural'} source match · ${profile.sourceRecognition.profileId} · ${profile.sourceRecognition.representation}`,
        }
      : {}),
  };
  if (
    profile.requiresCustomCode &&
    !profile.executable &&
    profile.sourceRecognition?.state === 'inspection-unavailable'
  ) {
    return {
      ...base,
      status: 'unsupported',
      message:
        'Reviewed source inspection is unavailable for this installed runtime. Repository Python remains blocked.',
    };
  }
  if (
    profile.requiresCustomCode &&
    !profile.executable &&
    profile.sourceRecognition?.state === 'unknown'
  ) {
    return {
      ...base,
      status: 'blocked',
      message:
        'Repository source does not match an installed reviewed implementation. Oryn will not run repository Python.',
    };
  }
  if (blocking.includes('REVIEWED_REPRESENTATION_UNSUPPORTED')) {
    return {
      ...base,
      status: 'unsupported',
      message:
        'The source implementation is recognized, but this weight format, configuration or backend has not been qualified for the installed adapter.',
    };
  }
  if (installed && (verified || userTrustEligible)) {
    return {
      ...base,
      status: 'installed',
      trustMode: verified ? 'oryn-verified' : 'user-trusted',
      message: verified ? 'Installed · Open in Oryn' : 'Installed with user trust · Open in Oryn',
    };
  }
  if (blocking.includes('MODEL_ACCESS_RESTRICTED')) {
    return {
      ...base,
      status: 'access-required',
      message: 'Access approval is required on Hugging Face.',
    };
  }
  if (blocking.some((reason) => ['CUSTOM_CODE_REQUIRED', 'UNSAFE_MODEL_FORMAT'].includes(reason))) {
    return {
      ...base,
      status: 'blocked',
      securityStatus: 'blocked',
      message: securityMessage(blocking),
    };
  }
  const resourceReason = RESOURCE_BLOCKING_REASONS.find((reason) => blocking.includes(reason));
  if (
    resourceReason &&
    blocking.every(
      (reason) => reason === 'CONFORMANCE_REQUIRED' || RESOURCE_BLOCKING_REASONS.includes(reason),
    )
  ) {
    return {
      ...classifyFailure(base, new Error(resourceReason)),
      blockingReasons: blocking,
    };
  }
  if (verified) {
    return {
      ...base,
      status: 'loadable',
      trustMode: 'oryn-verified',
      message: 'Supported by an active signed Oryn runtime pack.',
    };
  }
  if (userTrustEligible) {
    return {
      ...base,
      status: 'unverified',
      trustMode: 'user-trusted',
      compatibility: 'Not Oryn verified',
      message:
        'The sealed Oryn runtime can attempt this safe declarative model, but this exact revision has not passed Oryn conformance.',
    };
  }
  const evidencePassed = pack?.trustedMetadata?.conformance.status !== 'fixture-only';
  return {
    ...base,
    status: 'unsupported',
    message:
      blocking.length > 0
        ? humanizeBlockingReason(
            blocking.find(
              (reason) =>
                reason !== 'CONFORMANCE_REQUIRED' && !RESOURCE_BLOCKING_REASONS.includes(reason),
            ) ?? blocking[0],
          )
        : !pack
          ? 'No installed runtime supports this model’s architecture and file format.'
          : !evidencePassed
            ? 'This exact model has not passed runtime-pack conformance testing.'
            : 'This exact model is not currently supported by Oryn.',
  };
}

function classifyFailure(
  base: HuggingFacePageActionState,
  error: unknown,
): HuggingFacePageActionState {
  const code = errorCode(error);
  if (code === 'MODEL_ACCESS_RESTRICTED') {
    return {
      ...base,
      status: 'access-required',
      blockingReasons: [code],
      message: 'Access approval is required on Hugging Face.',
    };
  }
  if (['CUSTOM_CODE_REQUIRED', 'UNSAFE_MODEL_FORMAT'].includes(code)) {
    return {
      ...base,
      status: 'blocked',
      securityStatus: 'blocked',
      blockingReasons: [code],
      message: securityMessage([code]),
    };
  }
  if (RESOURCE_BLOCKING_REASONS.includes(code)) {
    return {
      ...base,
      status: 'resource-error',
      blockingReasons: [code],
      message:
        code === 'DISK_SPACE_INSUFFICIENT'
          ? 'Not enough disk space. Remove models or free storage, then retry.'
          : code === 'RUNTIME_RESOURCES_INSUFFICIENT'
            ? 'The model exceeds the currently available runtime resources. Free resources, then retry.'
            : 'Not enough available model memory. Unload another model, then retry.',
    };
  }
  return {
    ...base,
    status: 'failed',
    blockingReasons: code ? [code] : undefined,
    message: 'Oryn could not inspect this model. Retry from the page action.',
  };
}

function classifyInstallFailure(
  base: HuggingFacePageActionState,
  error: unknown,
): HuggingFacePageActionState {
  const classified = classifyFailure(base, error);
  const disk = diskSpaceDetails(error);
  if (classified.status !== 'failed') return { ...classified, ...disk };
  const code = errorCode(error);
  return {
    ...classified,
    ...disk,
    message: code
      ? `The model download could not be started (${code}). Retry from the page action.`
      : 'The model download could not be started. Retry from the page action.',
  };
}

function availableModelMemory(resources: RuntimeResourceSnapshot): number | undefined {
  const available = resources.systemMemory?.availableBytes;
  if (available === undefined || !Number.isFinite(available) || available < 0) return undefined;
  const budget = resources.localAI.memoryBudgetBytes;
  if (budget === undefined) return available;
  // Match the Engine's admission calculation; system available memory already
  // excludes resident allocations, so subtract models only from the configured budget.
  return Math.min(available, Math.max(0, budget - resources.localAI.estimatedResidentModelBytes));
}

function diskSpaceDetails(
  error: unknown,
): Pick<HuggingFacePageActionState, 'requiredDiskBytes' | 'availableDiskBytes'> {
  if (!(error instanceof Error) || !('protocolError' in error)) return {};
  const protocolError = error.protocolError;
  if (
    typeof protocolError !== 'object' ||
    protocolError === null ||
    !('details' in protocolError)
  ) {
    return {};
  }
  const details = protocolError.details;
  if (typeof details !== 'object' || details === null) return {};
  const required = 'requiredBytes' in details ? details.requiredBytes : undefined;
  const available = 'availableBytes' in details ? details.availableBytes : undefined;
  return {
    ...(typeof required === 'number' && Number.isFinite(required) && required >= 0
      ? { requiredDiskBytes: required }
      : {}),
    ...(typeof available === 'number' && Number.isFinite(available) && available >= 0
      ? { availableDiskBytes: available }
      : {}),
  };
}

function errorCode(error: unknown): string {
  if (typeof error === 'object' && error && 'code' in error && typeof error.code === 'string')
    return error.code;
  if (error instanceof Error && 'protocolError' in error) {
    const protocolError = error.protocolError;
    if (
      typeof protocolError === 'object' &&
      protocolError !== null &&
      'code' in protocolError &&
      typeof protocolError.code === 'string'
    ) {
      return protocolError.code;
    }
  }
  if (error instanceof Error) {
    const match = error.message.match(/\b[A-Z][A-Z0-9_]{3,}\b/u);
    return match?.[0] ?? '';
  }
  return '';
}

function securityMessage(reasons: string[]): string {
  return reasons.includes('UNSAFE_MODEL_FORMAT')
    ? 'Blocked: this repository does not provide a safe declarative model format.'
    : 'Blocked: this repository requires untrusted custom code.';
}

function humanizeBlockingReason(reason?: string): string {
  const messages: Record<string, string> = {
    MODEL_ARCHITECTURE_UNRESOLVED:
      'The runtime cannot determine the base architecture from this checkpoint’s metadata.',
    QUANTIZATION_UNSUPPORTED:
      'This checkpoint uses a quantization that the installed backend cannot load.',
    QUANTIZATION_METADATA_INVALID: 'This checkpoint’s quantization metadata is invalid.',
    TENSOR_METADATA_UNAVAILABLE: 'The runtime could not read the checkpoint’s tensor metadata.',
    MODEL_SHARDS_INVALID: 'The checkpoint’s weight index is invalid.',
    MODEL_TOKENIZER_MISSING: 'This checkpoint is missing its tokenizer files.',
    RUNNER_UNSUPPORTED: 'No bundled runner supports this model architecture.',
    RUNTIME_CAPABILITY_UNSUPPORTED:
      'The model format is recognized, but its embedded architecture is not supported by the bundled runtime.',
    GGUF_VARIANT_UNSUPPORTED:
      'No single-file GGUF variant in this repository matches the bundled llama.cpp policy.',
    TASK_ADAPTER_UNAVAILABLE:
      'The runtime recognizes this model, but Oryn has no sealed adapter for its requested task.',
    BACKEND_UNAVAILABLE: 'The required local compute backend is unavailable.',
    RUNTIME_DEPENDENCY_UNAVAILABLE: 'The required signed runtime is unavailable.',
    MODEL_SUBSET_INCOMPLETE: 'The repository is missing files required by its runtime pack.',
  };
  return (reason && messages[reason]) || 'This exact model is not currently supported by Oryn.';
}

function exactModel(state: HuggingFacePageActionState): ResolvedModelRef {
  if (!state.exactRevision) throw new Error('HUGGING_FACE_EXACT_REVISION_REQUIRED');
  return { provider: 'huggingface', repository: state.repository, revision: state.exactRevision };
}

function isInstallProgress(
  value: unknown,
): value is { operationId: string; downloadedBytes: number; totalBytes?: number } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'operationId' in value &&
    typeof value.operationId === 'string' &&
    'downloadedBytes' in value &&
    typeof value.downloadedBytes === 'number' &&
    (!('totalBytes' in value) || typeof value.totalBytes === 'number')
  );
}

function summarizeSelectedFiles(profile: CandidateExecutionProfile): string[] {
  if (!profile.selectedFiles?.length) {
    return profile.runtimePack?.trustedMetadata?.modelFile
      ? [profile.runtimePack.trustedMetadata.modelFile]
      : [];
  }
  const visible = profile.selectedFiles
    .slice(0, 32)
    .map((file) => `${file.name} · ${formatBytes(file.sizeBytes)}`);
  if (profile.selectedFiles.length > visible.length) {
    visible.push(`…and ${profile.selectedFiles.length - visible.length} more selected files`);
  }
  return visible;
}
