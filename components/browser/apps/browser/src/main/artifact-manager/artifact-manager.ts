import { randomBytes } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';

import type { ArtifactIoRuntime } from './artifact-file-service';
import type { BrowserCallerContext } from '../browser-caller-context';
import { ArtifactGarbageCollector } from './artifact-garbage-collector';
import { ArtifactCapabilityService } from './artifact-capability-service';
import { ArtifactLeaseManager, type ArtifactLeaseSet } from './artifact-lease-manager';
import { ArtifactQuotaManager, type ArtifactQuotaPolicy } from './artifact-quota-manager';
import { ArtifactReconciler } from './artifact-reconciler';
import { ArtifactRegistry } from './artifact-registry';
import { ArtifactStorage, type ArtifactWriteTarget } from './artifact-storage';
import {
  ArtifactError,
  type ArtifactLifetime,
  type ArtifactMetadata,
  type ArtifactOwner,
  type ArtifactRecord,
  type ArtifactReference,
  type EngineInferenceResult,
  type EngineOutputArtifact,
  sameArtifactOwner,
} from './types';

export interface BeginArtifactCreate {
  mimeType: string;
  sizeBytes: number;
  name?: string;
  lifetime?: 'ephemeral' | 'session';
}

interface PendingUpload {
  artifactId: string;
  owner: ArtifactOwner;
  declaredSize: number;
  writtenBytes: number;
  target: ArtifactWriteTarget;
}

const MAX_CHUNK_BYTES = 256 * 1024;
const EPHEMERAL_TTL_MS = 10 * 60 * 1000;

export class ArtifactManager {
  readonly registry: ArtifactRegistry;
  readonly storage: ArtifactStorage;
  readonly leases: ArtifactLeaseManager;
  readonly quota: ArtifactQuotaManager;
  readonly garbageCollector: ArtifactGarbageCollector;
  readonly reconciler: ArtifactReconciler;
  readonly capabilities: ArtifactCapabilityService;
  private readonly publications = new Map<
    string,
    { owner: ArtifactOwner; jobId: string; abort: AbortController }
  >();
  private readonly uploads = new Map<string, PendingUpload>();
  private readonly hasDocumentAuthority: (owner: ArtifactOwner) => boolean;
  private readonly now: () => number;

  constructor(
    registryPath: string,
    storageRoot: string,
    options: {
      artifactIo?: ArtifactIoRuntime;
      quotaPolicy?: ArtifactQuotaPolicy;
      availableBytes?: () => number;
      hasDocumentAuthority?: (owner: ArtifactOwner) => boolean;
      now?: () => number;
    } = {},
  ) {
    this.hasDocumentAuthority = options.hasDocumentAuthority ?? (() => true);
    this.now = options.now ?? Date.now;
    this.registry = new ArtifactRegistry(registryPath);
    this.storage = new ArtifactStorage(storageRoot, options.artifactIo);
    this.capabilities = new ArtifactCapabilityService(this.hasDocumentAuthority);
    this.leases = new ArtifactLeaseManager(this.registry, this.storage);
    this.quota = new ArtifactQuotaManager(
      storageRoot,
      () => this.registry.list(),
      options.quotaPolicy,
      options.availableBytes,
    );
    this.garbageCollector = new ArtifactGarbageCollector(this.registry, this.storage, this.leases);
    this.leases.setCleanupHandler((artifactId) =>
      this.garbageCollector.collectOne(artifactId).then(() => undefined),
    );
    this.reconciler = new ArtifactReconciler(this.registry, this.storage);
    try {
      this.reconciler.reconcile();
    } catch (error) {
      this.storage.close();
      throw error;
    }
  }

  hasPendingWrites(): boolean {
    return this.uploads.size > 0 || this.publications.size > 0;
  }

  async beginCreate(
    context: BrowserCallerContext,
    input: BeginArtifactCreate,
  ): Promise<{ uploadId: string }> {
    const metadata = validateCreate(input);
    const owner = ownerFromContext(context);
    const artifactId = opaqueId('artifact');
    const uploadId = opaqueId('artifact_upload');
    this.quota.reserve(uploadId, metadata.sizeBytes);
    let target: ArtifactWriteTarget | undefined;
    try {
      target = await this.storage.allocateInput(uploadId);
      const createdAt = this.now();
      const record: ArtifactRecord = {
        id: artifactId,
        state: 'creating',
        source: 'website-input',
        lifetime: metadata.lifetime,
        owner,
        metadata: {
          mimeType: metadata.mimeType,
          sizeBytes: metadata.sizeBytes,
          name: metadata.name,
          createdAt,
          ...(metadata.lifetime === 'ephemeral' ? { expiresAt: createdAt + EPHEMERAL_TTL_MS } : {}),
        },
        storage: { root: 'input', relativePath: target.relativePath },
      };
      this.registry.create(record);
      this.uploads.set(uploadId, {
        artifactId,
        owner,
        declaredSize: metadata.sizeBytes,
        writtenBytes: 0,
        target,
      });
      return { uploadId };
    } catch (error) {
      if (target) await this.storage.abort(target);
      this.registry.remove(artifactId);
      this.quota.release(uploadId);
      throw error;
    }
  }

  async appendChunk(
    context: BrowserCallerContext,
    uploadId: string,
    value: unknown,
  ): Promise<void> {
    const upload = this.requireUpload(context, uploadId);
    if (
      !(value instanceof ArrayBuffer) ||
      value.byteLength === 0 ||
      value.byteLength > MAX_CHUNK_BYTES
    ) {
      throw new ArtifactError('ARTIFACT_INVALID');
    }
    if (upload.writtenBytes + value.byteLength > upload.declaredSize) {
      await this.abortCreate(context, uploadId);
      throw new ArtifactError('ARTIFACT_INVALID');
    }
    try {
      await this.storage.write(upload.target, new Uint8Array(value));
      upload.writtenBytes += value.byteLength;
    } catch {
      await this.abortCreate(context, uploadId);
      throw new ArtifactError('ARTIFACT_IO_ERROR');
    }
  }

  async finishCreate(context: BrowserCallerContext, uploadId: string): Promise<ArtifactReference> {
    const upload = this.requireUpload(context, uploadId);
    if (upload.writtenBytes !== upload.declaredSize) {
      await this.abortCreate(context, uploadId);
      throw new ArtifactError('ARTIFACT_INVALID');
    }
    try {
      await this.storage.finalize(upload.target);
      const record = this.registry.get(upload.artifactId);
      if (!record) throw new ArtifactError('ARTIFACT_INVALID');
      this.registry.replace({
        ...record,
        state: 'ready',
        metadata: { ...record.metadata, sizeBytes: upload.writtenBytes },
      });
      this.uploads.delete(uploadId);
      this.quota.release(uploadId);
      return reference(record, upload.writtenBytes);
    } catch (error) {
      this.uploads.delete(uploadId);
      this.quota.release(uploadId);
      throw error instanceof ArtifactError ? error : new ArtifactError('ARTIFACT_IO_ERROR');
    }
  }

  async abortCreate(context: BrowserCallerContext, uploadId: string): Promise<void> {
    const upload = this.uploads.get(uploadId);
    if (!upload || !sameArtifactOwner(upload.owner, ownerFromContext(context))) return;
    this.uploads.delete(uploadId);
    this.quota.release(uploadId);
    await this.storage.abort(upload.target);
    this.registry.remove(upload.artifactId);
  }

  get(context: BrowserCallerContext, artifactId: string): ArtifactMetadata {
    const record = this.authorizeReady(context, artifactId);
    return {
      ...reference(record),
      createdAt: record.metadata.createdAt,
      expiresAt: record.metadata.expiresAt,
    };
  }

  getUrl(context: BrowserCallerContext, artifactId: string): string {
    return this.capabilities.issue(this.authorizeReady(context, artifactId));
  }

  acquireCapabilityRead(token: string): {
    record: ArtifactRecord;
    path: string;
    release(): Promise<void>;
  } {
    const capability = this.capabilities.resolve(token);
    if (!capability) throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
    const record = this.registry.get(capability.artifactId);
    if (record && this.expireIfDue(record)) throw new ArtifactError('ARTIFACT_EXPIRED');
    if (!record || record.state !== 'ready' || !sameArtifactOwner(record.owner, capability.owner)) {
      this.capabilities.revokeArtifact(capability.artifactId);
      throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
    }
    const lease = this.leases.acquireWebsiteRead(record);
    return { record, path: lease.path, release: lease.release };
  }

  async saveTo(
    context: BrowserCallerContext,
    artifactId: string,
    destination: string,
  ): Promise<void> {
    const record = this.authorizeReady(context, artifactId);
    const lease = this.leases.acquireWebsiteRead(record);
    try {
      await pipeline(createReadStream(lease.path), createWriteStream(destination, { mode: 0o600 }));
    } catch {
      throw new ArtifactError('ARTIFACT_IO_ERROR');
    } finally {
      await lease.release();
    }
  }

  async saveTrustedTo(artifactId: string, destination: string): Promise<void> {
    const record = this.registry.get(artifactId);
    if (!record || record.state !== 'ready') throw new ArtifactError('ARTIFACT_NOT_FOUND');
    const lease = this.leases.acquireWebsiteRead(record);
    try {
      await pipeline(createReadStream(lease.path), createWriteStream(destination, { mode: 0o600 }));
    } catch {
      throw new ArtifactError('ARTIFACT_IO_ERROR');
    } finally {
      await lease.release();
    }
  }

  async publishEngineResult(
    context: BrowserCallerContext,
    engineInstanceId: string,
    jobId: string,
    value: unknown,
    preview = false,
  ): Promise<unknown> {
    const owner = ownerFromContext(context);
    const reservationId = opaqueId('artifact_publication');
    const abort = new AbortController();
    this.publications.set(reservationId, { owner, jobId, abort });
    let stagingFiles: string[] = [];
    let stagingCleaned = false;
    const records: ArtifactRecord[] = [];
    const adoptedPaths: string[] = [];
    try {
      if (!this.hasDocumentAuthority(owner)) throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      const result = validateEngineResult(value);
      const descriptors = collectEngineOutputs(result);
      const outputIds = new Set(descriptors.map((descriptor) => descriptor.outputId));
      if (outputIds.size !== descriptors.length) throw new ArtifactError('ARTIFACT_INVALID');
      const normalized = descriptors.map(normalizeEngineOutput);
      const validated = normalized.map((descriptor) =>
        this.storage.prepareEngineOutput(engineInstanceId, jobId, descriptor),
      );
      stagingFiles = validated.map((output) => output.sourceRelativePath);
      this.quota.reserve(
        reservationId,
        normalized.reduce((total, descriptor) => total + descriptor.sizeBytes, 0),
      );
      const adopted = await this.storage.adoptEngineOutputs(validated, abort.signal);
      const createdAt = this.now();
      for (const output of adopted) {
        adoptedPaths.push(output.relativePath);
        records.push({
          id: opaqueId('artifact'),
          state: 'ready',
          source: preview ? 'engine-preview' : 'engine-output',
          lifetime: preview ? 'ephemeral' : 'session',
          owner,
          metadata: {
            mimeType: output.descriptor.mimeType,
            sizeBytes: output.descriptor.sizeBytes,
            name: output.descriptor.name,
            createdAt,
            ...(preview ? { expiresAt: createdAt + EPHEMERAL_TTL_MS } : {}),
          },
          storage: { root: 'published', relativePath: output.relativePath },
        });
      }
      await this.storage.removeEngineJobStaging(
        engineInstanceId,
        jobId,
        preview ? stagingFiles : undefined,
      );
      stagingCleaned = true;
      if (abort.signal.aborted || !this.hasDocumentAuthority(owner))
        throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      for (const record of records) this.registry.create(record);
      const metadata = new Map(
        normalized.map((descriptor, index) => [
          descriptor.outputId,
          metadataFromRecord(records[index]!),
        ]),
      );
      return transformEngineResult(result, metadata);
    } catch (error) {
      for (const record of records) this.registry.remove(record.id);
      for (const relativePath of adoptedPaths) this.storage.delete('published', relativePath);
      throw error instanceof ArtifactError ? error : new ArtifactError('ARTIFACT_IO_ERROR');
    } finally {
      if (!stagingCleaned)
        await this.storage.removeEngineJobStaging(
          engineInstanceId,
          jobId,
          preview ? stagingFiles : undefined,
        );
      this.quota.release(reservationId);
      this.publications.delete(reservationId);
    }
  }

  cancelPublication(jobId: string): void {
    for (const pending of this.publications.values()) {
      if (pending.jobId === jobId) pending.abort.abort();
    }
  }

  async cleanupEngineJob(engineInstanceId: string, jobId: string): Promise<void> {
    await this.storage.removeEngineJobStaging(engineInstanceId, jobId);
  }

  storageSnapshot(): {
    inputBytes: number;
    publishedBytes: number;
    temporaryBytes: number;
    readyArtifacts: number;
    reclaimableArtifacts: number;
  } {
    const usage = this.storage.usage();
    const records = this.registry.list();
    return {
      ...usage,
      readyArtifacts: records.filter((record) => record.state === 'ready').length,
      reclaimableArtifacts: records.filter((record) =>
        ['released', 'expired', 'deleting', 'error'].includes(record.state),
      ).length,
    };
  }

  async cleanupReclaimable(): Promise<number> {
    for (const record of this.registry.list()) this.expireIfDue(record);
    return (await this.garbageCollector.collect()).length;
  }

  cleanupTemporary(): void {
    if (this.hasPendingWrites()) throw new ArtifactError('RESOURCE_LIMIT_EXCEEDED');
    // Engine staging may contain active jobs; only startup and per-job cleanup
    // may remove it. Live settings cleanup touches Browser-private partials.
    this.storage.clearTemporary();
  }

  async release(context: BrowserCallerContext, artifactId: string): Promise<void> {
    const record = this.authorizeReady(context, artifactId);
    this.capabilities.revokeArtifact(record.id);
    this.registry.transition(record.id, 'released');
    await this.garbageCollector.collectOne(record.id);
  }

  acquireEngineInputLeases(context: BrowserCallerContext, artifactIds: string[]): ArtifactLeaseSet {
    if (!Array.isArray(artifactIds) || artifactIds.length === 0 || artifactIds.length > 32) {
      throw new ArtifactError('ARTIFACT_INVALID');
    }
    for (const artifactId of artifactIds) this.authorizeReady(context, artifactId);
    return this.leases.acquireEngineReads(ownerFromContext(context), artifactIds);
  }

  async invalidateDocument(
    owner: Pick<ArtifactOwner, 'tabId' | 'webContentsId' | 'documentId'>,
  ): Promise<void> {
    this.capabilities.revokeDocument(owner);
    for (const pending of this.publications.values()) {
      if (matchesDocument(pending.owner, owner)) pending.abort.abort();
    }
    const cleanup: Promise<unknown>[] = [];
    for (const [uploadId, upload] of this.uploads) {
      if (matchesDocument(upload.owner, owner)) {
        this.uploads.delete(uploadId);
        this.quota.release(uploadId);
        this.registry.remove(upload.artifactId);
        cleanup.push(this.storage.abort(upload.target));
      }
    }
    for (const record of this.registry.list()) {
      if (
        record.lifetime !== 'persistent' &&
        record.state === 'ready' &&
        matchesDocument(record.owner, owner)
      ) {
        this.registry.transition(record.id, 'released');
        cleanup.push(this.garbageCollector.collectOne(record.id));
      }
    }
    // Revoke every record before yielding. Navigation and destruction can invalidate
    // the same document concurrently; an awaited loop retained stale ready records.
    await Promise.all(cleanup);
  }

  private requireUpload(context: BrowserCallerContext, uploadId: string): PendingUpload {
    if (typeof uploadId !== 'string' || !/^artifact_upload_[A-Za-z0-9_-]{20,80}$/u.test(uploadId)) {
      throw new ArtifactError('ARTIFACT_INVALID');
    }
    const upload = this.uploads.get(uploadId);
    if (!upload || !sameArtifactOwner(upload.owner, ownerFromContext(context))) {
      throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
    }
    return upload;
  }

  private authorizeReady(context: BrowserCallerContext, artifactId: string): ArtifactRecord {
    if (typeof artifactId !== 'string' || !/^artifact_[A-Za-z0-9_-]{20,80}$/u.test(artifactId)) {
      throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
    }
    const record = this.registry.get(artifactId);
    if (!record || !sameArtifactOwner(record.owner, ownerFromContext(context))) {
      throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
    }
    if (this.expireIfDue(record)) throw new ArtifactError('ARTIFACT_EXPIRED');
    if (record.state === 'expired') throw new ArtifactError('ARTIFACT_EXPIRED');
    if (record.state !== 'ready') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
    if (!this.storage.exists(record.storage.root, record.storage.relativePath)) {
      this.registry.transition(record.id, 'error');
      throw new ArtifactError('ARTIFACT_NOT_FOUND');
    }
    return record;
  }

  private expireIfDue(record: ArtifactRecord): boolean {
    if (
      record.state !== 'ready' ||
      record.metadata.expiresAt === undefined ||
      record.metadata.expiresAt > this.now()
    ) {
      return record.state === 'expired';
    }
    this.capabilities.revokeArtifact(record.id);
    this.registry.transition(record.id, 'expired');
    setImmediate(() => void this.garbageCollector.collectOne(record.id));
    return true;
  }
}

const SAFE_INLINE_MIME = new Set([
  'image/png',
  'image/jpeg',
  'image/gif',
  'image/webp',
  'image/avif',
  'audio/wav',
  'audio/mpeg',
  'audio/ogg',
  'audio/flac',
  'audio/aac',
  'audio/mp4',
  'video/mp4',
  'video/webm',
  'video/ogg',
  'text/plain',
  'application/pdf',
]);

function normalizeEngineOutput(value: EngineOutputArtifact): EngineOutputArtifact {
  const mimeType = value.mimeType.trim().toLowerCase().split(';', 1)[0] ?? '';
  const name = value.name === undefined ? undefined : sanitizeFilename(value.name);
  return {
    ...value,
    mimeType: SAFE_INLINE_MIME.has(mimeType) ? mimeType : 'application/octet-stream',
    ...(name ? { name } : { name: undefined }),
  };
}

function sanitizeFilename(value: string): string | undefined {
  const sanitized = value
    .replaceAll(/[\\/]/gu, '_')
    .split('')
    .map((character) => {
      const code = character.charCodeAt(0);
      return code < 32 || code === 127 ? '_' : character;
    })
    .join('')
    .replaceAll(/\.\.+/gu, '_')
    .trim()
    .slice(0, 255);
  return sanitized && sanitized !== '.' ? sanitized : undefined;
}

function validateEngineResult(value: unknown, depth = 0): EngineInferenceResult {
  if (depth > 8 || !value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ArtifactError('ARTIFACT_INVALID');
  }
  const record = value as Record<string, unknown>;
  if (record.kind === 'structured' && 'value' in record) {
    return { kind: 'structured', value: structuredClone(record.value) };
  }
  if (record.kind === 'artifact' && isEngineOutputArtifact(record.artifact)) {
    return { kind: 'artifact', artifact: { ...record.artifact } };
  }
  if (
    record.kind === 'multi' &&
    Array.isArray(record.outputs) &&
    record.outputs.length > 0 &&
    record.outputs.length <= 64
  ) {
    return {
      kind: 'multi',
      outputs: record.outputs.map((output) => validateEngineResult(output, depth + 1)),
    };
  }
  throw new ArtifactError('ARTIFACT_INVALID');
}

function isEngineOutputArtifact(value: unknown): value is EngineOutputArtifact {
  return Boolean(
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    typeof (value as EngineOutputArtifact).outputId === 'string' &&
    typeof (value as EngineOutputArtifact).internalPath === 'string' &&
    typeof (value as EngineOutputArtifact).mimeType === 'string' &&
    typeof (value as EngineOutputArtifact).sizeBytes === 'number' &&
    ((value as EngineOutputArtifact).name === undefined ||
      typeof (value as EngineOutputArtifact).name === 'string'),
  );
}

function collectEngineOutputs(value: EngineInferenceResult): EngineOutputArtifact[] {
  if (value.kind === 'structured') return [];
  if (value.kind === 'artifact') return [value.artifact];
  return value.outputs.flatMap(collectEngineOutputs);
}

function transformEngineResult(
  value: EngineInferenceResult,
  metadata: Map<string, ArtifactMetadata>,
): unknown {
  if (value.kind === 'structured') return value;
  if (value.kind === 'artifact') {
    const artifact = metadata.get(value.artifact.outputId);
    if (!artifact) throw new ArtifactError('ARTIFACT_INVALID');
    return { kind: 'artifact', artifact };
  }
  return {
    kind: 'multi',
    outputs: value.outputs.map((output) => transformEngineResult(output, metadata)),
  };
}

function metadataFromRecord(record: ArtifactRecord): ArtifactMetadata {
  return {
    ...reference(record),
    createdAt: record.metadata.createdAt,
    expiresAt: record.metadata.expiresAt,
  };
}

function validateCreate(input: BeginArtifactCreate): {
  mimeType: string;
  sizeBytes: number;
  name?: string;
  lifetime: Exclude<ArtifactLifetime, 'persistent'>;
} {
  if (!input || typeof input !== 'object') throw new ArtifactError('ARTIFACT_INVALID');
  if (!Number.isSafeInteger(input.sizeBytes) || input.sizeBytes < 0) {
    throw new ArtifactError('ARTIFACT_INVALID');
  }
  const mimeType = input.mimeType.trim().toLowerCase() || 'application/octet-stream';
  if (mimeType.length > 255 || /[\r\n\0]/u.test(mimeType))
    throw new ArtifactError('ARTIFACT_INVALID');
  if (
    input.name !== undefined &&
    (typeof input.name !== 'string' || input.name.length > 255 || /[\r\n\0]/u.test(input.name))
  ) {
    throw new ArtifactError('ARTIFACT_INVALID');
  }
  if (input.lifetime !== undefined && !['ephemeral', 'session'].includes(input.lifetime)) {
    throw new ArtifactError('ARTIFACT_INVALID');
  }
  return {
    mimeType,
    sizeBytes: input.sizeBytes,
    name: input.name,
    lifetime: input.lifetime ?? 'session',
  };
}

function ownerFromContext(context: BrowserCallerContext): ArtifactOwner {
  return {
    origin: context.origin,
    tabId: context.tabId,
    webContentsId: context.webContentsId,
    frameId: context.frameId,
    documentId: context.documentId,
  };
}

function matchesDocument(
  owner: ArtifactOwner,
  target: Pick<ArtifactOwner, 'tabId' | 'webContentsId' | 'documentId'>,
): boolean {
  return (
    owner.tabId === target.tabId &&
    owner.webContentsId === target.webContentsId &&
    owner.documentId === target.documentId
  );
}

function opaqueId(prefix: string): string {
  return `${prefix}_${randomBytes(24).toString('base64url')}`;
}

function reference(
  record: ArtifactRecord,
  sizeBytes = record.metadata.sizeBytes,
): ArtifactReference {
  return {
    id: record.id,
    mimeType: record.metadata.mimeType,
    sizeBytes,
    name: record.metadata.name,
  };
}
