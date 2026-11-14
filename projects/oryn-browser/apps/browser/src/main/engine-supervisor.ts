import {
  validateTaskProfile,
  validateTaskRequirements,
  type TaskRequirements,
} from '@ocno/oryn-protocol';
import { SHARED_TASK_CONTRACTS } from '@ocno/oryn-protocol';
import { supportsTaskNegotiation, validateNegotiatedTasks } from './task-negotiation';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { mkdirSync, realpathSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { Readable, Writable } from 'node:stream';

import {
  assertCandidateExecutionProfile,
  assertEngineSessionAttached,
  assertEngineJobEvent,
  assertEngineJobReference,
  assertEngineJobStatus,
  assertHandshakeResponse,
  assertModelInstallProgressEvent,
  assertModelCatalogMatchResult,
  assertModelInstallResult,
  assertModelInstallStateEvent,
  assertModelInventoryChangedEvent,
  assertModelInventoryResult,
  assertModelRemovePreviewResult,
  assertModelRemoveResult,
  assertModelResidencyChangedEvent,
  assertModelResidencyResult,
  assertProtocolEvent,
  assertProtocolPingResult,
  assertProtocolResponse,
  assertRuntimeCapabilities,
  assertRuntimeMemoryPressureChangedEvent,
  assertRuntimePackCleanupResult,
  assertRuntimePackInventoryResult,
  assertRuntimePolicyUpdateResult,
  assertRuntimeOverrideResult,
  assertRuntimeResourceSnapshot,
  assertRuntimeResourcesUpdatedEvent,
  assertRuntimeShutdownResult,
  assertRuntimeStateEvent,
  assertSessionReleaseResult,
  MAX_MESSAGE_BYTES,
  PROTOCOL_VERSION,
  ProtocolEventSequence,
  ProtocolLineDecoder,
  parseProtocolJSONLine,
  type EngineRuntimeCapabilities,
  type EngineInputArtifactBinding,
  type EngineJobEvent,
  type EngineJobReference,
  type EngineJobStatus,
  type EngineSessionAttached,
  type CandidateExecutionProfile,
  type ModelInstallResult,
  type ModelCatalogMatchResult,
  type ModelInventoryResult,
  type ModelRemovePreviewResult,
  type ModelRemoveResult,
  type ModelResidencyResult,
  type ProtocolError,
  type ProtocolResponse,
  type RuntimePolicyUpdate,
  type RuntimePolicyUpdateResult,
  type RuntimeOverrideResult,
  type RuntimeResourceSnapshot,
  type RuntimePackCleanupResult,
  type RuntimePackInventoryResult,
  type ResolvedModelRef,
} from '@ocno/oryn-protocol';

import { engineLaunchPaths } from './paths';
import type { EngineFailureReason, EngineFailureSummary } from './browser-diagnostics';
import {
  EngineDiagnosticBuffer,
  MAX_PENDING_PROTOCOL_REQUESTS,
  sanitizeProtocolError,
  SerializedProtocolWriter,
} from './protocol-transport';

export type EngineSupervisorState = 'starting' | 'ready' | 'error' | 'stopped';

interface PendingRequest {
  kind: 'handshake' | 'request';
  method: string;
  resolve(value: unknown): void;
  reject(error: Error): void;
  timeout?: NodeJS.Timeout;
}

export class EngineRequestError extends Error {
  constructor(readonly protocolError: ProtocolError) {
    super(protocolError.message);
    this.name = 'EngineRequestError';
  }
}

export class EngineSupervisor extends EventEmitter {
  private child?: ChildProcessWithoutNullStreams;
  private writer?: SerializedProtocolWriter;
  private readonly pending = new Map<string, PendingRequest>();
  // A caller timeout does not mean Engine stopped processing the request.
  private readonly timedOutRequests = new Map<string, PendingRequest['kind']>();
  private readonly eventSequence = new ProtocolEventSequence();
  private readonly diagnostics = new EngineDiagnosticBuffer();
  private lineDecoder = new ProtocolLineDecoder();
  private startPromise?: Promise<void>;
  private stopping = false;
  private restartAttempts = 0;
  private diagnosticFailures: EngineFailureSummary = { count: 0 };
  private failureReportedForAttempt = false;
  private _state: EngineSupervisorState = 'stopped';
  private _engineInstanceId?: string;
  private maxMessageBytes = MAX_MESSAGE_BYTES;
  private runtimePolicy: RuntimePolicyUpdate = {
    policyRevision: 1,
    batteryMode: 'normal',
    allowSpeculativePreload: false,
  };
  private _lastResources?: RuntimeResourceSnapshot;
  private _modelInventory: ModelInventoryResult = { models: [] };
  private credentialInput?: Writable;
  private credentialAckBuffer = '';
  private credentialRevision = 0;
  private readonly pendingCredentialAcks = new Map<
    number,
    { resolve(): void; reject(error: Error): void; timeout: NodeJS.Timeout }
  >();

  constructor(
    private readonly options: {
      modelRoot?: string;
      runtimePackRoot?: string;
      artifactInputRoot?: string;
      artifactOutputRoot?: string;
      runtimeScratchRoot?: string;
      huggingFaceToken?: () => string | undefined;
    } = {},
  ) {
    super();
  }

  get state(): EngineSupervisorState {
    return this._state;
  }

  get engineInstanceId(): string | undefined {
    return this._engineInstanceId;
  }

  get diagnosticLog(): string {
    return this.diagnostics.read();
  }

  get failureSummary(): EngineFailureSummary {
    return structuredClone(this.diagnosticFailures);
  }

  get lastResources(): RuntimeResourceSnapshot | undefined {
    return this._lastResources;
  }

  get modelInventory(): ModelInventoryResult {
    return this._modelInventory;
  }

  async start(): Promise<void> {
    if (this._state === 'ready') {
      return;
    }
    if (this.startPromise) {
      return this.startPromise;
    }

    this.stopping = false;
    this.startPromise = this.startEngine()
      .catch(async (error: unknown) => {
        const failure = asError(error);
        this.handleProcessFailure(failure);
        this.rejectCredentialAcks(failure);
        // Keep concurrent callers on this attempt until its process exits. A
        // rejected initialization must not leave a live, unhandshaken Engine.
        const child = this.child;
        if (child) {
          await new Promise<void>((resolve) => {
            const finished = () => {
              clearTimeout(forceExit);
              child.removeListener('exit', finished);
              child.removeListener('close', finished);
              if (this.child === child) this.handleExit(child.exitCode, child.signalCode);
              resolve();
            };
            const forceExit = setTimeout(() => child.kill('SIGKILL'), 1_000);
            child.once('exit', finished);
            child.once('close', finished);
            child.stdin.end();
            // The macOS launcher forwards SIGTERM to its sandboxed Engine.
            child.kill('SIGTERM');
            if (child.exitCode !== null || child.signalCode !== null || !child.pid) finished();
          });
        }
        throw failure;
      })
      .finally(() => {
        this.startPromise = undefined;
      });
    return this.startPromise;
  }

  private async startEngine(): Promise<void> {
    this.failureReportedForAttempt = false;
    const deadline = Date.now() + 30_000;
    const remainingStartupMs = () => {
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw new Error('Local AI Engine startup timed out.');
      return remaining;
    };
    this.setState('starting');
    this.lineDecoder = new ProtocolLineDecoder(MAX_MESSAGE_BYTES);
    this.eventSequence.reset();
    this.diagnostics.clear();

    const launch = engineLaunchPaths();
    const modelRoot = secureDirectory(
      this.options.modelRoot ?? resolve(launch.engineDirectory, 'models'),
    );
    const runtimePackRoot = secureDirectory(
      this.options.runtimePackRoot ?? resolve(launch.engineDirectory, 'runtime-packs'),
    );
    const artifactInputRoot = secureDirectory(
      this.options.artifactInputRoot ?? resolve(launch.engineDirectory, 'artifacts', 'input'),
    );
    const artifactOutputRoot = secureDirectory(
      this.options.artifactOutputRoot ?? resolve(launch.engineDirectory, 'artifacts', 'output'),
    );
    const runtimeScratchRoot = secureDirectory(
      this.options.runtimeScratchRoot ?? resolve(launch.engineDirectory, 'runtime-temp'),
    );
    const environment: NodeJS.ProcessEnv = {
      PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
      HOME: runtimeScratchRoot,
      TMPDIR: runtimeScratchRoot,
      LANG: safeLocale(process.env.LANG),
      LC_CTYPE: safeLocale(process.env.LC_CTYPE),
      ORYN_MODEL_ROOT: modelRoot,
      ORYN_RUNTIME_PACK_ROOT: runtimePackRoot,
      ORYN_ARTIFACT_INPUT_ROOT: artifactInputRoot,
      ORYN_ARTIFACT_OUTPUT_ROOT: artifactOutputRoot,
      ORYN_ENGINE_ROOT: resolve(launch.engineDirectory),
      ORYN_SANDBOX_REQUIRED: launch.sandboxRequired ? '1' : '0',
      PYTHONNOUSERSITE: '1',
      PYTHONDONTWRITEBYTECODE: '1',
      PYTHONUNBUFFERED: '1',
      HF_HUB_DISABLE_TELEMETRY: '1',
    };
    if (launch.pythonPath) {
      environment.PYTHONPATH = launch.pythonPath;
    }
    if (!launch.packaged && process.env.ORYN_ALLOW_RUNTIME_OVERRIDES === '1') {
      environment.ORYN_PACK_OVERRIDES_ALLOWED = '1';
      for (const [key, value] of Object.entries(process.env)) {
        if ((key.startsWith('ORYN_PACK_') || key.startsWith('ORYN_FOUNDATION_')) && value) {
          environment[key] = value;
        }
      }
    } else {
      environment.ORYN_PACK_OVERRIDES_ALLOWED = '0';
    }
    if (launch.runtimePackSecurityProfile) {
      environment.ORYN_PACK_SANDBOX_PROFILE = resolve(launch.runtimePackSecurityProfile);
    }

    environment.ORYN_HF_CREDENTIAL_INPUT_FD = '3';
    environment.ORYN_HF_CREDENTIAL_ACK_FD = '4';
    const pythonRoot = dirname(dirname(realpathSync(launch.pythonExecutable)));
    const sandboxArguments = launch.engineSecurityProfile
      ? [
          '-f',
          resolve(launch.engineSecurityProfile),
          '-D',
          `ENGINE_ROOT=${resolve(launch.engineDirectory)}`,
          '-D',
          `PYTHON_ROOT=${pythonRoot}`,
          '-D',
          `SECURITY_ROOT=${dirname(resolve(launch.engineSecurityProfile))}`,
          '-D',
          `MODEL_ROOT=${modelRoot}`,
          '-D',
          `PACK_ROOT=${runtimePackRoot}`,
          '-D',
          `ARTIFACT_INPUT_ROOT=${artifactInputRoot}`,
          '-D',
          `ARTIFACT_OUTPUT_ROOT=${artifactOutputRoot}`,
          '-D',
          `SCRATCH_ROOT=${runtimeScratchRoot}`,
          launch.pythonExecutable,
          ...launch.pythonArguments,
        ]
      : launch.pythonArguments;
    // Seatbelt cannot be reapplied inside Engine. The bundled launcher starts
    // Engine and signed workers as separately sandboxed children.
    const executable = launch.pythonExecutable;
    const argumentsToLaunch = launch.engineSecurityProfile
      ? [
          ...(launch.packaged ? ['-I', '-B'] : ['-B']),
          '-m',
          'oryn_engine.sandbox_launcher',
          ...sandboxArguments,
        ]
      : sandboxArguments;
    const child = spawn(executable, argumentsToLaunch, {
      cwd: launch.engineDirectory,
      env: environment,
      stdio: ['pipe', 'pipe', 'pipe', 'pipe', 'pipe'],
    });
    this.child = child as ChildProcessWithoutNullStreams;
    this.writer = new SerializedProtocolWriter(child.stdin);
    this.credentialInput = child.stdio[3] as Writable;
    const credentialAcknowledgments = child.stdio[4] as Readable;
    credentialAcknowledgments.on('data', (chunk: Buffer) => {
      if (this.child === child) this.handleCredentialAck(chunk);
    });

    child.stdout.on('data', (chunk: Buffer) => {
      if (this.child === child) this.handleStdout(chunk);
    });
    child.stdout.once('end', () => {
      if (this.child === child) this.handleStdoutEnd();
    });
    child.stderr.on('data', (chunk: Buffer) => {
      if (this.child !== child) return;
      const raw = chunk.toString('utf8');
      this.diagnostics.append(raw);
      const text = raw.trimEnd();
      if (text) {
        console.error(`[engine] ${text}`);
      }
    });
    child.once('error', (error) => {
      if (this.child !== child) return;
      this.handleProcessFailure(error);
      this.rejectCredentialAcks(error);
    });
    child.once('exit', (code, signal) => {
      if (this.child === child) this.handleExit(code, signal);
    });

    await this.sendSupervisedCredential(this.options.huggingFaceToken?.(), remainingStartupMs());

    const handshakeId = this.createRequestId();
    const result = await this.send(
      {
        type: 'request',
        id: handshakeId,
        method: 'protocol.handshake',
        params: {
          supportedVersions: [PROTOCOL_VERSION],
          browserVersion: '0.1.0',
          maxMessageBytes: MAX_MESSAGE_BYTES,
        },
      },
      'handshake',
      'protocol.handshake',
      remainingStartupMs(),
    );

    if (!isRecord(result)) {
      throw new Error('Engine handshake returned an invalid result.');
    }
    this._engineInstanceId = String(result.engineInstanceId);
    this.maxMessageBytes = Number(result.maxMessageBytes);
    this.lineDecoder = new ProtocolLineDecoder(this.maxMessageBytes);

    // This additive extension first ships in Engine 0.2. Older engines retain base tasks.
    if (supportsTaskNegotiation(String(result.engineVersion))) {
      const negotiated = await this.request(
        'task.contracts.negotiate',
        { contracts: [...SHARED_TASK_CONTRACTS] },
        remainingStartupMs(),
      );
      validateNegotiatedTasks(negotiated);
    }
    await this.applyCurrentRuntimePolicy(remainingStartupMs());
    await this.ping(remainingStartupMs());
    await this.reconcileModelInventory(remainingStartupMs());
    remainingStartupMs();
    this.restartAttempts = 0;
    this.setState('ready');
  }

  async ping(timeoutMs = 15_000): Promise<void> {
    const result = await this.request('protocol.ping', {}, timeoutMs);
    assertProtocolPingResult(result);
  }

  async getRuntimeCapabilities(): Promise<EngineRuntimeCapabilities> {
    await this.start();
    const result = await this.request('runtime.capabilities', {});
    assertRuntimeCapabilities(result);
    return result;
  }

  async getRuntimeResources(): Promise<RuntimeResourceSnapshot> {
    await this.start();
    const result = await this.request('runtime.resources', {});
    assertRuntimeResourceSnapshot(result);
    this._lastResources = result;
    return result;
  }

  async getRuntimePackInventory(): Promise<RuntimePackInventoryResult> {
    await this.start();
    const result = await this.request('runtime.pack.inventory', {});
    assertRuntimePackInventoryResult(result);
    return result;
  }

  async cleanupRuntimePacks(): Promise<RuntimePackCleanupResult> {
    await this.start();
    const result = await this.request('runtime.pack.cleanup', {});
    assertRuntimePackCleanupResult(result);
    return result;
  }

  async matchModelCatalog(
    task: string,
    outputModalities: string[],
  ): Promise<ModelCatalogMatchResult> {
    await this.start();
    const result = await this.request('model.catalog.match', { task, outputModalities });
    assertModelCatalogMatchResult(result);
    return result;
  }

  async updateRuntimePolicy(
    policy: Omit<RuntimePolicyUpdate, 'policyRevision'>,
  ): Promise<RuntimePolicyUpdateResult> {
    this.runtimePolicy = {
      ...policy,
      policyRevision: this.runtimePolicy.policyRevision + 1,
    };
    return this.applyCurrentRuntimePolicy();
  }

  configureInitialRuntimePolicy(policy: Omit<RuntimePolicyUpdate, 'policyRevision'>): void {
    if (this._state !== 'stopped' || this.startPromise) {
      throw new Error('ENGINE_POLICY_ALREADY_ACTIVE');
    }
    this.runtimePolicy = { ...policy, policyRevision: this.runtimePolicy.policyRevision };
  }

  private async negotiatedRequirements(
    requirements?: TaskRequirements,
  ): Promise<{ requirements?: TaskRequirements }> {
    if (!requirements) return {};
    validateTaskRequirements(requirements);
    const capabilities = await this.getRuntimeCapabilities();
    if (capabilities.features.taskProfiles) return { requirements };
    if (
      requirements.contract ||
      requirements.requiredFeatures?.length ||
      Object.keys(requirements.parameterValues ?? {}).length
    )
      throw new Error('TASK_CONTRACT_UNSUPPORTED');
    return {};
  }

  async inspectModel(
    model: ResolvedModelRef,
    requirements?: TaskRequirements,
    recommendation?: { snapshotDigest: string; engines: string[] },
  ): Promise<CandidateExecutionProfile> {
    await this.start();
    const taskRequirements = await this.negotiatedRequirements(requirements);
    const result = await this.request(
      'model.inspect',
      { model, ...taskRequirements, ...(recommendation ? { recommendation } : {}) },
      240_000,
    );
    assertCandidateExecutionProfile(result);
    if (result.taskProfile) validateTaskProfile(result.taskProfile);
    return result;
  }

  async updateHuggingFaceCredential(token: string | undefined): Promise<void> {
    await this.start();
    await this.sendSupervisedCredential(token);
  }

  /** Main-process HF page action only; never forwarded from website/SDK IPC. */
  async inspectHuggingFaceModel(
    model: ResolvedModelRef,
    sourceContext: { sourceURL: string; tabId: string; documentId: string },
    signal?: AbortSignal,
  ): Promise<CandidateExecutionProfile> {
    const capabilities = await this.getRuntimeCapabilities();
    if (signal?.aborted) throw new Error('Source inspection cancelled.');
    if (!capabilities.features.reviewedSourceInspection) {
      const profile = await this.inspectModel(model);
      return profile.requiresCustomCode
        ? { ...profile, sourceRecognition: { state: 'inspection-unavailable' } }
        : profile;
    }
    const operationId =
      'source_' +
      createHash('sha256')
        .update(JSON.stringify([sourceContext.tabId, sourceContext.documentId]))
        .digest('hex');
    let settled = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const cancel = () => {
      if (settled || attempts++ >= 20) return;
      // Cancellation can overtake registration on the Engine control pool.
      void this.cancelModelInstall(operationId)
        .then((accepted) => {
          if (!accepted && !settled) retry = setTimeout(cancel, 100);
        })
        .catch(() => undefined);
    };
    const request = this.request('model.source.inspect', { model, sourceContext }, 240_000);
    signal?.addEventListener('abort', cancel, { once: true });
    if (signal?.aborted) cancel();
    try {
      const result = await request;
      assertCandidateExecutionProfile(result);
      if (result.taskProfile) validateTaskProfile(result.taskProfile);
      return result;
    } finally {
      settled = true;
      if (retry) clearTimeout(retry);
      signal?.removeEventListener('abort', cancel);
    }
  }

  async installHuggingFaceModel(
    operationId: string,
    model: ResolvedModelRef,
    sourceContext: { sourceURL: string; tabId: string; documentId: string },
  ): Promise<ModelInstallResult> {
    const capabilities = await this.getRuntimeCapabilities();
    if (!capabilities.features.reviewedSourceInspection)
      return this.installModel(operationId, model);
    const result = await this.request(
      'model.source.install',
      { operationId, model, sourceContext },
      7_200_000,
    );
    assertModelInstallResult(result);
    return result;
  }

  async installModel(
    operationId: string,
    model: ResolvedModelRef,
    requirements?: TaskRequirements,
    recommendation?: { snapshotDigest: string; engines: string[] },
  ): Promise<ModelInstallResult> {
    await this.start();
    const result = await this.request('model.install', {
      ...(recommendation ? { recommendation } : {}),
      operationId,
      model,
      ...(await this.negotiatedRequirements(requirements)),
    });
    assertModelInstallResult(result);
    return result;
  }

  async cancelModelInstall(operationId: string): Promise<boolean> {
    await this.start();
    const result = await this.request('model.install.cancel', { operationId });
    if (!isRecord(result) || typeof result.accepted !== 'boolean') {
      throw new Error('Engine returned an invalid model install cancellation result.');
    }
    return result.accepted;
  }

  async loadModel(operationId: string, model: ResolvedModelRef): Promise<ModelResidencyResult> {
    await this.start();
    const result = await this.request('model.load', { operationId, model });
    assertModelResidencyResult(result);
    return result;
  }

  async unloadModel(
    operationId: string,
    model: ResolvedModelRef,
    mode: 'fail-if-busy' | 'when-idle' = 'fail-if-busy',
  ): Promise<ModelResidencyResult> {
    await this.start();
    const result = await this.request('model.unload', { operationId, model, mode });
    assertModelResidencyResult(result);
    return result;
  }

  async getModelInventory(): Promise<ModelInventoryResult> {
    await this.start();
    return this.reconcileModelInventory();
  }

  async setModelRuntimeOverride(
    model: ResolvedModelRef,
    candidateId?: string,
  ): Promise<RuntimeOverrideResult> {
    await this.start();
    const result = await this.request('model.runtime.override', {
      model,
      candidateId: candidateId ?? null,
    });
    assertRuntimeOverrideResult(result);
    await this.reconcileModelInventory();
    return result;
  }

  async previewModelRemoval(installationId: string): Promise<ModelRemovePreviewResult> {
    await this.start();
    const result = await this.request('model.remove.preview', { installationId });
    assertModelRemovePreviewResult(result);
    return result;
  }

  async removeModel(
    operationId: string,
    installationId: string,
    mode: 'fail-if-busy' | 'when-idle' = 'fail-if-busy',
  ): Promise<ModelRemoveResult> {
    await this.start();
    const result = await this.request('model.remove', { operationId, installationId, mode });
    assertModelRemoveResult(result);
    return result;
  }

  async attachModelSession(
    sessionId: string,
    model: ResolvedModelRef,
    ensureLoaded = true,
    requirements?: TaskRequirements,
  ): Promise<EngineSessionAttached> {
    await this.start();
    const result = await this.request('session.attach', {
      sessionId,
      model,
      ensureLoaded,
      ...(await this.negotiatedRequirements(requirements)),
    });
    assertEngineSessionAttached(result);
    if (result.taskProfile) validateTaskProfile(result.taskProfile);
    return result;
  }

  async releaseModelSession(sessionId: string): Promise<boolean> {
    await this.start();
    const result = await this.request('session.release', { sessionId });
    assertSessionReleaseResult(result);
    return result.released;
  }

  async createJob(
    jobId: string,
    sessionId: string,
    input: unknown,
    inputArtifacts: EngineInputArtifactBinding[],
  ): Promise<EngineJobReference> {
    await this.start();
    const result = await this.request('job.create', {
      jobId,
      sessionId,
      input,
      ...(inputArtifacts.length > 0 ? { inputArtifacts } : {}),
    });
    assertEngineJobReference(result);
    return result;
  }

  async getJobStatus(jobId: string): Promise<EngineJobStatus> {
    const result = await this.request('job.status', { jobId });
    assertEngineJobStatus(result);
    return result;
  }

  async cancelJob(
    jobId: string,
    reason: 'website' | 'user' | 'resource-policy' | 'runtime-shutdown',
  ): Promise<EngineJobStatus> {
    const result = await this.request('job.cancel', { jobId, reason });
    assertEngineJobStatus(result);
    return result;
  }

  async request(
    method: string,
    params: Record<string, unknown>,
    timeoutMs = isControlMethod(method) ? 15_000 : undefined,
  ): Promise<unknown> {
    const negotiated =
      this._state === 'ready' || (this._state === 'starting' && this._engineInstanceId);
    if (!negotiated || !this.child) {
      throw new Error('Local AI Engine is not ready.');
    }
    return this.send(
      {
        v: PROTOCOL_VERSION,
        type: 'request',
        id: this.createRequestId(),
        method,
        params,
      },
      'request',
      method,
      timeoutMs,
    );
  }

  stop(): void {
    void this.shutdown().catch((error) => this.handleProcessFailure(asError(error)));
  }

  async shutdown(reason = 'browser-exit'): Promise<void> {
    this.stopping = true;
    const child = this.child;
    if (!child || child.killed) {
      this.setState('stopped');
      return;
    }
    const exited = new Promise<void>((resolve) => child.once('exit', () => resolve()));
    try {
      if (this._state === 'ready') {
        await this.request('runtime.shutdown', { reason });
        child.stdin.end();
      } else {
        child.kill('SIGTERM');
      }
      const graceful = await Promise.race([
        exited.then(() => true),
        new Promise<false>((resolve) => setTimeout(() => resolve(false), 10_000)),
      ]);
      if (!graceful && !child.killed) {
        child.kill('SIGKILL');
        await exited;
      }
    } catch (error) {
      if (!child.killed) child.kill('SIGTERM');
      await Promise.race([exited, new Promise<void>((resolve) => setTimeout(resolve, 10_000))]);
      if (this.child === child && !child.killed) child.kill('SIGKILL');
      throw error;
    }
  }

  async restart(): Promise<void> {
    await this.shutdown('runtime-restart');
    this.stopping = false;
    await this.start();
  }

  forceExitForSmokeTest(): void {
    if (process.env.LOCAL_AI_SMOKE_TEST !== '1') {
      throw new Error('Forced engine exit is available only in the isolated smoke profile.');
    }
    this.child?.kill('SIGKILL');
  }

  private send(
    message: Record<string, unknown>,
    kind: PendingRequest['kind'],
    method: string,
    timeoutMs: number | undefined,
  ): Promise<unknown> {
    const child = this.child;
    const requestId = String(message.id);
    if (!child?.stdin.writable || !this.writer) {
      return Promise.reject(new Error('Local AI Engine stdin is unavailable.'));
    }
    if (this.pending.size + this.timedOutRequests.size >= MAX_PENDING_PROTOCOL_REQUESTS) {
      return Promise.reject(new Error('Too many pending engine protocol requests.'));
    }

    const serialized = `${JSON.stringify(message)}\n`;
    if (Buffer.byteLength(serialized, 'utf8') > this.maxMessageBytes + 1) {
      return Promise.reject(new Error('Protocol request exceeds the negotiated message limit.'));
    }

    return new Promise<unknown>((resolve, reject) => {
      const timeout = timeoutMs
        ? setTimeout(() => {
            this.pending.delete(requestId);
            this.timedOutRequests.set(requestId, kind);
            reject(new Error(`Engine request timed out: ${String(message.method)}`));
          }, timeoutMs)
        : undefined;
      this.pending.set(requestId, { kind, method, resolve, reject, timeout });
      void this.writer?.write(serialized).catch((error: unknown) => {
        this.timedOutRequests.delete(requestId);
        if (this.pending.has(requestId)) {
          if (timeout) clearTimeout(timeout);
          this.pending.delete(requestId);
          reject(asError(error));
        }
      });
    });
  }

  private handleStdout(chunk: Buffer): void {
    try {
      for (const line of this.lineDecoder.push(chunk)) {
        this.handleLine(line);
      }
    } catch (error) {
      this.handleProtocolCorruption(error);
    }
  }

  private handleLine(line: string): void {
    let message: unknown;
    try {
      message = parseProtocolJSONLine(line);
    } catch (error) {
      this.handleProtocolCorruption(error);
      return;
    }

    if (!isRecord(message)) {
      this.handleProtocolCorruption(new Error('Engine emitted a non-object protocol value.'));
      return;
    }

    if (message.type === 'event') {
      assertProtocolEvent(message);
      this.eventSequence.accept(message);
      if (message.event === 'runtime.state') assertRuntimeStateEvent(message);
      if (message.event === 'runtime.resources.updated') {
        assertRuntimeResourcesUpdatedEvent(message);
        this._lastResources = message.data;
        this.emit('resources-updated', message.data);
      }
      if (message.event === 'runtime.memoryPressure.changed') {
        assertRuntimeMemoryPressureChangedEvent(message);
        this.emit('memory-pressure-changed', message.data);
      }
      if (message.event === 'model.install.progress') {
        assertModelInstallProgressEvent(message);
        this.emit('model-install-progress', message.data);
      }
      if (message.event === 'model.install.state') {
        assertModelInstallStateEvent(message);
        this.emit('model-install-state', message.data);
      }
      if (message.event === 'model.residency.changed') {
        assertModelResidencyChangedEvent(message);
        this.emit('model-residency-changed', message.data);
      }
      if (message.event === 'model.inventory.changed') {
        assertModelInventoryChangedEvent(message);
        this.emit('model-inventory-invalidated', message.data);
        if (!this.stopping && this._state === 'ready') {
          void this.reconcileModelInventory()
            .then((inventory) => this.emit('model-inventory-updated', inventory))
            .catch((error) => {
              if (!this.stopping) console.error('Failed to reconcile model inventory:', error);
            });
        }
      }
      if (message.event.startsWith('job.')) {
        assertEngineJobEvent(message);
        this.emit('job-event', message as EngineJobEvent);
      }
      this.emit('event', message);
      return;
    }

    if (message.type !== 'response' || typeof message.id !== 'string') {
      this.handleProtocolCorruption(new Error('Engine emitted an invalid protocol envelope.'));
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending && this.timedOutRequests.has(message.id)) {
      try {
        if (this.timedOutRequests.get(message.id) === 'handshake') assertHandshakeResponse(message);
        else assertProtocolResponse(message);
        this.timedOutRequests.delete(message.id);
      } catch (error) {
        this.handleProtocolCorruption(error);
      }
      return;
    }
    if (!pending) {
      this.handleProtocolCorruption(new Error('Engine response has no pending request.'));
      return;
    }

    try {
      if (pending.kind === 'handshake') {
        assertHandshakeResponse(message);
      } else {
        assertProtocolResponse(message);
      }
      this.resolvePending(message.id, message);
    } catch (error) {
      this.handleProtocolCorruption(error);
    }
  }

  private resolvePending(
    requestId: string,
    response: ProtocolResponse | Record<string, unknown>,
  ): void {
    const pending = this.pending.get(requestId);
    if (!pending) {
      return;
    }
    // Keep the request registered until validation and settlement succeed. If an
    // assertion throws, handleProtocolCorruption must still be able to reject it.
    if (response.ok === true && 'result' in response) {
      if (pending.method === 'protocol.ping') assertProtocolPingResult(response.result);
      if (pending.method === 'runtime.capabilities') assertRuntimeCapabilities(response.result);
      if (pending.method === 'runtime.resources') assertRuntimeResourceSnapshot(response.result);
      if (pending.method === 'runtime.pack.inventory') {
        assertRuntimePackInventoryResult(response.result);
      }
      if (pending.method === 'runtime.pack.cleanup') {
        assertRuntimePackCleanupResult(response.result);
      }
      if (pending.method === 'runtime.policy.update') {
        assertRuntimePolicyUpdateResult(response.result);
      }
      if (pending.method === 'runtime.shutdown') assertRuntimeShutdownResult(response.result);
      if (pending.method === 'model.inspect' || pending.method === 'model.source.inspect')
        assertCandidateExecutionProfile(response.result);
      if (pending.method === 'model.install' || pending.method === 'model.source.install')
        assertModelInstallResult(response.result);
      if (pending.method === 'model.load' || pending.method === 'model.unload') {
        assertModelResidencyResult(response.result);
      }
      if (pending.method === 'model.inventory.list') assertModelInventoryResult(response.result);
      if (pending.method === 'model.remove.preview') {
        assertModelRemovePreviewResult(response.result);
      }
      if (pending.method === 'session.attach') assertEngineSessionAttached(response.result);
      if (pending.method === 'session.release') assertSessionReleaseResult(response.result);
      if (pending.method === 'job.create') assertEngineJobReference(response.result);
      if (pending.method === 'job.status' || pending.method === 'job.cancel') {
        assertEngineJobStatus(response.result);
      }
      pending.resolve(response.result);
    } else if (response.ok === false && 'error' in response) {
      pending.reject(
        new EngineRequestError(sanitizeProtocolError(response.error as ProtocolError)),
      );
    } else {
      pending.reject(new Error('Engine returned a malformed response.'));
    }
    if (pending.timeout) clearTimeout(pending.timeout);
    this.pending.delete(requestId);
  }

  private handleProtocolCorruption(error: unknown): void {
    if (!this.stopping) this.recordDiagnosticFailure('protocol-error', null);
    const protocolError = error instanceof Error ? error : new Error(String(error));
    console.error('Engine protocol corruption:', protocolError);
    this.rejectPending(protocolError);
    this.setState('error');
    this.child?.kill('SIGKILL');
  }

  private handleStdoutEnd(): void {
    try {
      this.lineDecoder.finish();
    } catch (error) {
      this.handleProtocolCorruption(error);
    }
  }

  private handleProcessFailure(error: Error): void {
    if (this.stopping) return;
    this.recordDiagnosticFailure('process-error', null);
    console.error('Local AI Engine process failure:', error);
    this.rejectPending(error);
    this.setState('error');
  }

  private handleExit(code: number | null, signal: NodeJS.Signals | null): void {
    const child = this.child;
    this.child = undefined;
    this.writer = undefined;
    this.credentialInput = undefined;
    this.credentialAckBuffer = '';
    this.rejectCredentialAcks(new Error('Local AI Engine credential channel closed.'));
    this._engineInstanceId = undefined;
    this._lastResources = undefined;
    this._modelInventory = { models: [] };
    this.rejectPending(
      new Error(`Local AI Engine exited (code ${String(code)}, signal ${String(signal)}).`),
    );

    if (this.stopping) {
      this.setState('stopped');
      return;
    }

    this.recordDiagnosticFailure(signal ? 'signal' : 'unexpected-exit', code);
    this.setState('error');
    if (child && this.restartAttempts < 3) {
      const delayMs = 500 * 2 ** this.restartAttempts;
      this.restartAttempts += 1;
      setTimeout(
        () => void this.start().catch((error) => this.handleProcessFailure(asError(error))),
        delayMs,
      );
    }
  }

  private rejectPending(error: Error): void {
    for (const request of this.pending.values()) {
      if (request.timeout) clearTimeout(request.timeout);
      request.reject(error);
    }
    this.pending.clear();
    this.timedOutRequests.clear();
  }

  private recordDiagnosticFailure(reason: EngineFailureReason, exitCode: number | null): void {
    if (!this.failureReportedForAttempt) {
      this.diagnosticFailures.count = Math.min(this.diagnosticFailures.count + 1, 1_000_000);
      this.diagnosticFailures.last = { reason, exitCode };
      this.failureReportedForAttempt = true;
    } else if (this.diagnosticFailures.last && exitCode !== null) {
      this.diagnosticFailures.last.exitCode = exitCode;
    }
  }

  private sendSupervisedCredential(token: string | undefined, timeoutMs = 5_000): Promise<void> {
    const input = this.credentialInput;
    if (!input?.writable) {
      return Promise.reject(new Error('Local AI Engine credential channel is unavailable.'));
    }
    const revision = ++this.credentialRevision;
    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingCredentialAcks.delete(revision);
        reject(new Error('Local AI Engine did not acknowledge credential state.'));
      }, timeoutMs);
      this.pendingCredentialAcks.set(revision, { resolve, reject, timeout });
      input.write(`${JSON.stringify({ revision, token: token ?? null })}\n`, (error) => {
        if (!error) return;
        const pending = this.pendingCredentialAcks.get(revision);
        if (!pending) return;
        clearTimeout(pending.timeout);
        this.pendingCredentialAcks.delete(revision);
        pending.reject(new Error('Local AI Engine credential channel write failed.'));
      });
    });
  }

  private handleCredentialAck(chunk: Buffer): void {
    this.credentialAckBuffer += chunk.toString('utf8');
    if (this.credentialAckBuffer.length > 4096) {
      this.rejectCredentialAcks(new Error('Local AI Engine credential acknowledgment is invalid.'));
      return;
    }
    while (this.credentialAckBuffer.includes('\n')) {
      const newline = this.credentialAckBuffer.indexOf('\n');
      const line = this.credentialAckBuffer.slice(0, newline).replace(/\r$/u, '');
      this.credentialAckBuffer = this.credentialAckBuffer.slice(newline + 1);
      if (!/^[1-9][0-9]{0,9}$/u.test(line)) {
        this.rejectCredentialAcks(
          new Error('Local AI Engine credential acknowledgment is invalid.'),
        );
        return;
      }
      const revision = Number(line);
      const pending = this.pendingCredentialAcks.get(revision);
      if (!pending) continue;
      clearTimeout(pending.timeout);
      this.pendingCredentialAcks.delete(revision);
      pending.resolve();
    }
  }

  private rejectCredentialAcks(error: Error): void {
    for (const pending of this.pendingCredentialAcks.values()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pendingCredentialAcks.clear();
  }

  private createRequestId(): string {
    return `req_${randomUUID().replaceAll('-', '')}`;
  }

  private async applyCurrentRuntimePolicy(timeoutMs = 15_000): Promise<RuntimePolicyUpdateResult> {
    const result = await this.request(
      'runtime.policy.update',
      { ...this.runtimePolicy },
      timeoutMs,
    );
    assertRuntimePolicyUpdateResult(result);
    this._lastResources = result.resources;
    return result;
  }

  private async reconcileModelInventory(timeoutMs = 15_000): Promise<ModelInventoryResult> {
    const result = await this.request('model.inventory.list', {}, timeoutMs);
    assertModelInventoryResult(result);
    this._modelInventory = result;
    return result;
  }

  private setState(state: EngineSupervisorState): void {
    if (this._state === state) {
      return;
    }
    this._state = state;
    this.emit('state-changed', state);
  }
}

function secureDirectory(path: string): string {
  const absolute = resolve(path);
  mkdirSync(absolute, { recursive: true, mode: 0o700 });
  return realpathSync(absolute);
}

function safeLocale(value: string | undefined): string {
  return value && /^[A-Za-z0-9_.@-]{1,64}$/.test(value) ? value : 'C.UTF-8';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function isControlMethod(method: string): boolean {
  return (
    method === 'protocol.ping' ||
    method === 'runtime.capabilities' ||
    method === 'runtime.resources' ||
    method === 'runtime.pack.inventory' ||
    method === 'runtime.pack.cleanup' ||
    method === 'runtime.policy.update' ||
    method === 'runtime.shutdown'
  );
}
