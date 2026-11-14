import { exactModelKey } from '@ocno/oryn-protocol';
import {
  taskRequirementsFrom,
  taskRequirementFailures,
  type TaskRequirements,
  type TaskProfile,
} from '@ocno/oryn-protocol';
import { randomUUID } from 'node:crypto';

import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  EngineSessionAttached,
  ResolvedModelRef,
} from '@ocno/oryn-protocol';

import type { BrowserCallerContext, BrowserCallerContextService } from '../browser-caller-context';
import type { LocalAIPermissionStore } from './local-ai-permission-store';
import type { ModelPreferenceStore } from './model-preference-store';
import type { ModelResolver } from './model-resolver';
import type { PermissionPromptCoordinator } from './permission-prompt-coordinator';
import { validateModelRequest } from './model-request';
import type { ModelRequest, ModelResolution, ModelResolutionResult } from './types';
import {
  isOrynVerified,
  isUserTrustEligible,
  taskContractForProfile,
  type ModelWorkspaceTrustMode,
} from '../model-workspace/model-workspace-eligibility';

interface EngineSessionClient {
  getModelInventory(): Promise<{
    models: EngineModelInventoryRecord[];
  }>;
  attachModelSession(
    sessionId: string,
    model: ResolvedModelRef,
    ensureLoaded?: boolean,
    requirements?: TaskRequirements,
  ): Promise<EngineSessionAttached>;
  releaseModelSession(sessionId: string): Promise<boolean>;
  installModel(
    operationId: string,
    model: ResolvedModelRef,
    requirements?: TaskRequirements,
    recommendation?: { snapshotDigest: string; engines: string[] },
  ): Promise<unknown>;
  cancelModelInstall(operationId: string): Promise<boolean>;
  on(event: string, listener: (value: unknown) => void): unknown;
  removeListener(event: string, listener: (value: unknown) => void): unknown;
}

export type BrowserModelRequestProgress =
  | { stage: 'resolving' }
  | { stage: 'awaiting-permission' }
  | { stage: 'awaiting-model-choice' }
  | {
      stage: 'downloading';
      model: ResolvedModelRef;
      downloadedBytes: number;
      totalBytes?: number;
    }
  | { stage: 'loading'; model: ResolvedModelRef; progress?: number }
  | { stage: 'ready'; sessionId: string };

export interface BrowserModelRequestOptions {
  operationId: string;
  signal?: AbortSignal;
  onProgress?: (progress: BrowserModelRequestProgress) => void;
}

export interface BrowserModelSession {
  sessionId: string;
  role: string;
  origin: string;
  tabId: string;
  webContentsId: number;
  documentId: string;
  resolution: ModelResolution;
  runtime: EngineSessionAttached['runtime'];
  capabilities: EngineSessionAttached['technicalCapabilities'];
  request: ModelRequest;
  taskProfile?: TaskProfile;
}

export class BrowserModelSessionError extends Error {
  constructor(
    readonly code: 'USER_DENIED' | 'DOWNLOAD_REQUIRED' | 'SESSION_NOT_OWNED',
    readonly savedDecision?: 'deny' | 'revoked',
  ) {
    super(code);
    this.name = 'BrowserModelSessionError';
  }
}

export class BrowserModelSessionRegistry {
  private readonly sessions = new Map<string, BrowserModelSession>();
  private readonly pendingOrigins = new Set<string>();
  private static readonly MAX_PENDING_MODEL_REQUESTS = 4;

  constructor(
    private readonly engine: EngineSessionClient,
    private readonly resolver: ModelResolver,
    private readonly permissions: LocalAIPermissionStore,
    private readonly preferences: ModelPreferenceStore,
    private readonly prompts: PermissionPromptCoordinator,
    callerContexts?: BrowserCallerContextService,
  ) {
    callerContexts?.on(
      'document-invalidated',
      (owner: Pick<BrowserCallerContext, 'tabId' | 'webContentsId' | 'documentId'>) => {
        void this.releaseForDocument(owner);
      },
    );
  }

  async requestSession(
    request: unknown,
    owner: BrowserCallerContext,
    options: BrowserModelRequestOptions = { operationId: `model_request_${randomUUID()}` },
  ): Promise<BrowserModelSession> {
    throwIfAborted(options.signal);
    const origin = new URL(owner.origin).origin;
    if (
      this.pendingOrigins.has(origin) ||
      this.pendingOrigins.size >= BrowserModelSessionRegistry.MAX_PENDING_MODEL_REQUESTS
    )
      throw new Error('LOCAL_AI_MODEL_REQUEST_BUSY');
    this.pendingOrigins.add(origin);
    try {
      return await this.requestAdmittedSession(request, owner, options);
    } finally {
      this.pendingOrigins.delete(origin);
    }
  }

  private async requestAdmittedSession(
    request: unknown,
    owner: BrowserCallerContext,
    options: BrowserModelRequestOptions,
  ): Promise<BrowserModelSession> {
    throwIfAborted(options.signal);
    options.onProgress?.({ stage: 'resolving' });
    const validatedRequest = validateModelRequest(request);
    const role = validatedRequest.role;
    const inventory = await this.engine.getModelInventory();
    let result = await this.resolver.resolve(validatedRequest, {
      origin: owner.origin,
      inventory: inventory.models,
      signal: options.signal,
    });
    let key = permissionKey(owner.origin, role, result);
    const existingDecision = this.permissions.decision(key);
    if (existingDecision === 'deny' || existingDecision === 'revoked') {
      throw new BrowserModelSessionError('USER_DENIED', existingDecision);
    }
    if (
      result.resolution.downloadRequired ||
      (existingDecision !== 'always-allow' && existingDecision !== 'allow-once')
    ) {
      options.onProgress?.({ stage: 'awaiting-permission' });
      const response = await this.prompts.request(owner.origin, role, result, options.signal, () =>
        options.onProgress?.({ stage: 'awaiting-model-choice' }),
      );
      if (response.action === 'deny') {
        // Dismissing this request must not silently create a permanent site block.
        throw new BrowserModelSessionError('USER_DENIED');
      }
      if (response.selectedModel) {
        result = await this.resolver.resolve(validatedRequest, {
          origin: owner.origin,
          inventory: inventory.models,
          userOverride: response.selectedModel,
          signal: options.signal,
        });
        key = permissionKey(owner.origin, role, result);
        if (response.rememberSiteChoice) {
          this.preferences.setSite(owner.origin, role, response.selectedModel);
        }
      }
      this.permissions.recordTrustedDecision(key, response.action);
    }
    if (result.resolution.downloadRequired || result.selected.profile.runtimePreparationRequired) {
      await this.installResolvedModel(
        result,
        options,
        taskRequirementsFrom(validatedRequest.requirements),
      );
    }
    throwIfAborted(options.signal);
    if (this.permissions.decision(key) === 'allow-once') this.permissions.consumeOnce(key);

    const sessionId = `session_${randomUUID().replaceAll('-', '')}`;
    options.onProgress?.({ stage: 'loading', model: result.resolution.resolvedModel });
    const attached = await this.engine.attachModelSession(
      sessionId,
      result.resolution.resolvedModel,
      true,
      taskRequirementsFrom(validatedRequest.requirements),
    );
    if (
      taskRequirementFailures(
        attached.taskProfile,
        taskRequirementsFrom(validatedRequest.requirements) ?? {},
      ).length
    ) {
      await this.engine.releaseModelSession(sessionId);
      throw new Error('TASK_CONTRACT_INCOMPATIBLE');
    }
    const session: BrowserModelSession = Object.freeze({
      sessionId,
      role,
      origin: owner.origin,
      tabId: owner.tabId,
      webContentsId: owner.webContentsId,
      documentId: owner.documentId,
      resolution: structuredClone(result.resolution),
      runtime: attached.runtime,
      capabilities: attached.technicalCapabilities,
      ...(attached.taskProfile ? { taskProfile: structuredClone(attached.taskProfile) } : {}),
      request: structuredClone(validatedRequest),
    });
    this.sessions.set(sessionId, session);
    options.onProgress?.({ stage: 'ready', sessionId });
    return session;
  }

  /** Attaches an already-installed exact model for a browser-owned workspace document. */
  async attachModelWorkspaceSession(
    profile: CandidateExecutionProfile,
    taskContract: string,
    owner: BrowserCallerContext,
    trustMode: ModelWorkspaceTrustMode = 'oryn-verified',
  ): Promise<BrowserModelSession> {
    const eligible =
      trustMode === 'oryn-verified'
        ? isOrynVerified(profile)
        : isUserTrustEligible(profile) || isOrynVerified(profile);
    if (!eligible || taskContractForProfile(profile) !== taskContract) {
      throw new Error('MODEL_WORKSPACE_REVALIDATION_FAILED');
    }
    const inventory = await this.engine.getModelInventory();
    const installed = inventory.models.some(
      (item) =>
        item.installationState === 'installed' && modelKey(item.model) === modelKey(profile.model),
    );
    if (!installed) throw new Error('MODEL_WORKSPACE_MODEL_REMOVED');

    const requirements: TaskRequirements = {
      task: profile.taskProfile?.task ?? profile.technicalCapabilities.task,
      contract: profile.taskProfile?.contract ?? { id: taskContract, revision: 1 },
    };
    if (profile.runtimePreparationRequired) {
      await this.engine.installModel(
        `install_${randomUUID().replaceAll('-', '')}`,
        profile.model,
        requirements,
      );
    }
    const sessionId = `session_${randomUUID().replaceAll('-', '')}`;
    const attached = await this.engine.attachModelSession(
      sessionId,
      profile.model,
      true,
      requirements,
    );
    if (profile.taskProfile && taskRequirementFailures(attached.taskProfile, requirements).length) {
      await this.engine.releaseModelSession(sessionId);
      throw new Error('MODEL_WORKSPACE_REVALIDATION_FAILED');
    }
    const resolution: ModelResolution = {
      preferredModel: structuredClone(profile.model),
      resolvedModel: structuredClone(profile.model),
      compatibility: trustMode === 'oryn-verified' ? 'certified' : 'uncertain',
      executability: 'executable',
      security: 'allowed',
      trust: trustMode === 'oryn-verified' ? 'oryn-verified' : 'user-approved-unverified',
      developerCertified: trustMode === 'oryn-verified',
      substituted: false,
      overriddenByUser: trustMode === 'user-trusted',
      resolutionReason: trustMode === 'oryn-verified' ? 'preferred-model' : 'user-override',
      compatibilityReasons: [],
      downloadRequired: false,
    };
    const request: ModelRequest = {
      role: `workspace-${taskContract}`,
      preferredModel: structuredClone(profile.model),
      requirements: {
        ...requirements,
        task: attached.taskProfile?.task ?? profile.technicalCapabilities.task,
        inputModalities:
          attached.taskProfile?.inputModalities ?? profile.technicalCapabilities.inputModalities,
        outputModalities:
          attached.taskProfile?.outputModalities ?? profile.technicalCapabilities.outputModalities,
      },
      substitution: { automatic: 'none' },
    };
    const session: BrowserModelSession = Object.freeze({
      sessionId,
      role: `workspace-${taskContract}`,
      origin: owner.origin,
      tabId: owner.tabId,
      webContentsId: owner.webContentsId,
      documentId: owner.documentId,
      resolution,
      runtime: attached.runtime,
      capabilities: attached.technicalCapabilities,
      ...(attached.taskProfile ? { taskProfile: structuredClone(attached.taskProfile) } : {}),
      request,
    });
    this.sessions.set(sessionId, session);
    return session;
  }

  requireOwned(sessionId: string, owner: BrowserCallerContext): BrowserModelSession {
    const session = this.sessions.get(sessionId);
    if (
      !session ||
      session.origin !== owner.origin ||
      session.tabId !== owner.tabId ||
      session.webContentsId !== owner.webContentsId ||
      session.documentId !== owner.documentId
    ) {
      throw new BrowserModelSessionError('SESSION_NOT_OWNED');
    }
    return session;
  }

  async releaseOwned(sessionId: string, owner: BrowserCallerContext): Promise<void> {
    this.requireOwned(sessionId, owner);
    this.sessions.delete(sessionId);
    await this.engine.releaseModelSession(sessionId);
  }

  async releaseForDocument(
    owner: Pick<BrowserCallerContext, 'tabId' | 'webContentsId' | 'documentId'>,
  ): Promise<void> {
    const ids = [...this.sessions.values()]
      .filter(
        (session) =>
          session.tabId === owner.tabId &&
          session.webContentsId === owner.webContentsId &&
          session.documentId === owner.documentId,
      )
      .map((session) => session.sessionId);
    for (const id of ids) {
      this.sessions.delete(id);
      await this.engine.releaseModelSession(id);
    }
  }

  async releaseForModel(model: ResolvedModelRef): Promise<void> {
    const ids = [...this.sessions.values()]
      .filter((session) => modelKey(session.resolution.resolvedModel) === modelKey(model))
      .map((session) => session.sessionId);
    for (const id of ids) {
      this.sessions.delete(id);
      await this.engine.releaseModelSession(id);
    }
  }

  async releaseForOrigin(origin: string): Promise<void> {
    const normalized = new URL(origin).origin;
    const ids = [...this.sessions.values()]
      .filter((session) => session.origin === normalized)
      .map((session) => session.sessionId);
    for (const id of ids) {
      this.sessions.delete(id);
      await this.engine.releaseModelSession(id);
    }
  }

  snapshot(): BrowserModelSession[] {
    return [...this.sessions.values()];
  }

  private async installResolvedModel(
    result: ModelResolutionResult,
    options: BrowserModelRequestOptions,
    requirements?: TaskRequirements,
  ): Promise<void> {
    const listener = (value: unknown) => {
      if (!isInstallProgress(value) || value.operationId !== options.operationId) return;
      options.onProgress?.({
        stage: 'downloading',
        model: result.resolution.resolvedModel,
        downloadedBytes: value.downloadedBytes,
        totalBytes: value.totalBytes,
      });
    };
    const abort = () => void this.engine.cancelModelInstall(options.operationId);
    this.engine.on('model-install-progress', listener);
    options.signal?.addEventListener('abort', abort, { once: true });
    try {
      throwIfAborted(options.signal);
      await this.engine.installModel(
        options.operationId,
        result.resolution.resolvedModel,
        requirements,
        result.selected.recommendation
          ? {
              snapshotDigest: result.selected.recommendation.snapshotDigest,
              engines: result.selected.recommendation.engines,
            }
          : undefined,
      );
      throwIfAborted(options.signal);
    } finally {
      this.engine.removeListener('model-install-progress', listener);
      options.signal?.removeEventListener('abort', abort);
    }
  }
}

function permissionKey(origin: string, role: string, result: ModelResolutionResult) {
  return {
    origin,
    role,
    model: result.resolution.resolvedModel,
  };
}

function modelKey(model: ResolvedModelRef): string {
  return exactModelKey(model);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new Error('REQUEST_ABORTED');
}

function isInstallProgress(value: unknown): value is {
  operationId: string;
  downloadedBytes: number;
  totalBytes: number;
} {
  return (
    typeof value === 'object' &&
    value !== null &&
    'operationId' in value &&
    typeof value.operationId === 'string' &&
    'downloadedBytes' in value &&
    typeof value.downloadedBytes === 'number' &&
    'totalBytes' in value &&
    typeof value.totalBytes === 'number'
  );
}
