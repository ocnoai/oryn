import { randomBytes } from 'node:crypto';
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
} from 'node:fs';
import { open, type FileHandle } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

import { ArtifactError, type EngineOutputArtifact } from './types';
import { ArtifactFileService, type ArtifactIoRuntime } from './artifact-file-service';

export interface ArtifactWriteTarget {
  uploadId: string;
  relativePath: string;
  temporaryPath: string;
  objectPath: string;
  handle: FileHandle;
}

export interface PreparedEngineOutput {
  descriptor: EngineOutputArtifact;
  sourceRelativePath: string;
}

export interface AdoptedEngineOutput extends PreparedEngineOutput {
  relativePath: string;
  objectPath: string;
}

export class ArtifactStorage {
  readonly inputRoot: string;
  readonly publishedRoot: string;
  readonly temporaryRoot: string;
  readonly engineStagingRoot: string;

  private readonly files: ArtifactFileService;

  constructor(
    readonly root: string,
    runtime?: ArtifactIoRuntime,
  ) {
    // Resolve trusted profile aliases once, before any Engine process starts.
    // Engine also canonicalizes its root (e.g. /var -> /private/var on macOS).
    mkdirSync(root, { recursive: true, mode: 0o700 });
    this.root = realpathSync(root);
    this.inputRoot = join(this.root, 'input');
    this.publishedRoot = join(this.root, 'published');
    this.temporaryRoot = join(this.root, 'temporary');
    this.engineStagingRoot = join(this.root, 'result', 'staging');
    for (const directory of [
      this.root,
      this.inputRoot,
      this.publishedRoot,
      this.temporaryRoot,
      this.engineStagingRoot,
    ]) {
      mkdirSync(directory, { recursive: true, mode: 0o700 });
    }
    this.files = new ArtifactFileService(
      this.engineStagingRoot,
      this.temporaryRoot,
      this.publishedRoot,
      runtime,
    );
  }

  close(): void {
    this.files.close();
  }

  async shutdown(): Promise<void> {
    await this.files.shutdown();
  }

  async allocateInput(uploadId: string): Promise<ArtifactWriteTarget> {
    const objectName = `object_${randomBytes(24).toString('base64url')}`;
    const relativePath = join(objectName.slice(-2), objectName);
    const objectPath = this.resolveControlled('input', relativePath, false);
    mkdirSync(dirname(objectPath), { recursive: true, mode: 0o700 });
    const temporaryPath = this.resolveWithin(this.temporaryRoot, `${uploadId}.partial`, false);
    const handle = await open(temporaryPath, 'wx', 0o600);
    return { uploadId, relativePath, temporaryPath, objectPath, handle };
  }

  async write(target: ArtifactWriteTarget, chunk: Uint8Array): Promise<void> {
    let offset = 0;
    while (offset < chunk.byteLength) {
      const { bytesWritten } = await target.handle.write(
        chunk,
        offset,
        chunk.byteLength - offset,
        null,
      );
      if (bytesWritten <= 0) throw new Error('ARTIFACT_SHORT_WRITE');
      offset += bytesWritten;
    }
  }

  async finalize(target: ArtifactWriteTarget): Promise<void> {
    await target.handle.sync();
    await target.handle.close();
    renameSync(target.temporaryPath, target.objectPath);
    chmodSync(target.objectPath, 0o400);
  }

  async abort(target: ArtifactWriteTarget): Promise<void> {
    try {
      await target.handle.close();
    } catch {
      // The handle may already be closed after a failed finalize.
    }
    rmSync(target.temporaryPath, { force: true });
  }

  resolveReady(root: 'input' | 'published', relativePath: string): string {
    const path = this.resolveControlled(root, relativePath, true);
    const stat = lstatSync(path);
    if (!stat.isFile() || stat.isSymbolicLink()) throw new ArtifactError('ARTIFACT_PATH_INVALID');
    const canonical = realpathSync(path);
    this.assertContained(root === 'input' ? this.inputRoot : this.publishedRoot, canonical);
    return canonical;
  }

  exists(root: 'input' | 'published', relativePath: string): boolean {
    try {
      this.resolveReady(root, relativePath);
      return true;
    } catch {
      return false;
    }
  }

  delete(root: 'input' | 'published', relativePath: string): void {
    const path = this.resolveControlled(root, relativePath, false);
    rmSync(path, { force: true });
  }

  prepareEngineOutput(
    engineInstanceId: string,
    jobId: string,
    descriptor: EngineOutputArtifact,
  ): PreparedEngineOutput {
    if (
      !/^[A-Za-z][A-Za-z0-9_-]{1,127}$/u.test(engineInstanceId) ||
      !/^[A-Za-z][A-Za-z0-9_-]{1,127}$/u.test(jobId) ||
      !descriptor ||
      typeof descriptor !== 'object' ||
      !/^output_[A-Za-z0-9_-]{8,128}$/u.test(descriptor.outputId) ||
      typeof descriptor.internalPath !== 'string' ||
      !isAbsolute(descriptor.internalPath) ||
      typeof descriptor.mimeType !== 'string' ||
      !Number.isSafeInteger(descriptor.sizeBytes) ||
      descriptor.sizeBytes < 0
    ) {
      throw new ArtifactError('ARTIFACT_INVALID');
    }
    // Only lexical validation here. The helper opens every component no-follow
    // from the retained staging directory and validates the actual source FD.
    const jobRoot = join(this.engineStagingRoot, engineInstanceId, jobId);
    const withinJob = relative(jobRoot, descriptor.internalPath);
    if (
      !withinJob ||
      isAbsolute(withinJob) ||
      withinJob.split(sep).some((part) => !part || part === '..' || part === '.') ||
      descriptor.internalPath.includes('\0')
    ) {
      throw new ArtifactError('ARTIFACT_PATH_INVALID');
    }
    return { descriptor, sourceRelativePath: join(engineInstanceId, jobId, withinJob) };
  }

  async adoptEngineOutputs(
    outputs: PreparedEngineOutput[],
    signal?: AbortSignal,
  ): Promise<AdoptedEngineOutput[]> {
    if (outputs.length === 0) return [];
    const adopted = outputs.map((output) => {
      const objectName = `object_${randomBytes(24).toString('base64url')}`;
      const relativePath = join(objectName.slice(-2), objectName);
      return {
        ...output,
        relativePath,
        objectPath: this.resolveControlled('published', relativePath, false),
      };
    });
    try {
      await this.files.run(
        {
          operation: 'copy',
          outputs: adopted.map((output) => ({
            source: output.sourceRelativePath,
            object: basename(output.objectPath),
            size: output.descriptor.sizeBytes,
          })),
        },
        signal,
      );
      return adopted;
    } catch (error) {
      // The helper has exited before rollback. These names are Browser-generated
      // and inside private roots; no Engine path is reopened by Browser.
      for (const output of adopted) {
        rmSync(join(this.temporaryRoot, `${basename(output.objectPath)}.adopting`), {
          force: true,
        });
        rmSync(output.objectPath, { force: true });
      }
      throw error;
    }
  }

  async removeEngineJobStaging(
    engineInstanceId: string,
    jobId: string,
    files?: string[],
  ): Promise<void> {
    if (![engineInstanceId, jobId].every((id) => /^[A-Za-z][A-Za-z0-9_-]{1,127}$/u.test(id)))
      return;
    try {
      await this.files.run({
        operation: 'cleanup',
        job: join(engineInstanceId, jobId),
        ...(files ? { files } : {}),
      });
    } catch {
      // Bounded, descriptor-relative cleanup may leave entries for startup retry.
      console.error('Artifact staging cleanup incomplete; startup reconciliation will retry.');
    }
  }

  clearEngineStaging(): { removedRoots: number; complete: boolean } {
    const count = readdirSync(this.engineStagingRoot).length;
    const complete = this.files.cleanupAtStartup();
    return { removedRoots: complete ? count : 0, complete };
  }

  listObjectFiles(root: 'input' | 'published'): string[] {
    const base = root === 'input' ? this.inputRoot : this.publishedRoot;
    if (!existsSync(base)) return [];
    const result: string[] = [];
    for (const entry of readdirSync(base, { recursive: true, withFileTypes: true })) {
      if (entry.isFile() && !entry.isSymbolicLink()) {
        const absolute = join(entry.parentPath, entry.name);
        result.push(relative(base, absolute));
      }
    }
    return result;
  }

  clearTemporary(): void {
    for (const entry of readdirSync(this.temporaryRoot, { withFileTypes: true })) {
      const target = this.resolveWithin(this.temporaryRoot, basename(entry.name), false);
      rmSync(target, { recursive: entry.isDirectory(), force: true });
    }
  }

  usage(): { inputBytes: number; publishedBytes: number; temporaryBytes: number } {
    return {
      inputBytes: this.directoryBytes(this.inputRoot),
      publishedBytes: this.directoryBytes(this.publishedRoot),
      temporaryBytes:
        this.directoryBytes(this.temporaryRoot) + this.directoryBytes(this.engineStagingRoot),
    };
  }

  private directoryBytes(root: string): number {
    if (!existsSync(root)) return 0;
    let total = 0;
    for (const entry of readdirSync(root, { recursive: true, withFileTypes: true })) {
      if (!entry.isFile() || entry.isSymbolicLink()) continue;
      const path = join(entry.parentPath, entry.name);
      try {
        const stat = lstatSync(path);
        if (stat.isFile() && !stat.isSymbolicLink()) total += stat.size;
      } catch {
        // Concurrent cleanup is reflected by the next snapshot.
      }
    }
    return total;
  }

  private resolveControlled(
    root: 'input' | 'published',
    relativePath: string,
    mustExist: boolean,
  ): string {
    const base = root === 'input' ? this.inputRoot : this.publishedRoot;
    return this.resolveWithin(base, relativePath, mustExist);
  }

  private resolveWithin(base: string, value: string, mustExist: boolean): string {
    if (!value || isAbsolute(value) || value.includes('\0')) {
      throw new ArtifactError('ARTIFACT_PATH_INVALID');
    }
    const canonicalBase = existsSync(base) ? realpathSync(base) : resolve(base);
    const candidate = resolve(canonicalBase, value);
    this.assertContained(canonicalBase, candidate);
    if (mustExist && !existsSync(candidate)) throw new ArtifactError('ARTIFACT_NOT_FOUND');
    return candidate;
  }

  private assertContained(base: string, candidate: string): void {
    const canonicalBase = existsSync(base) ? realpathSync(base) : resolve(base);
    const relation = relative(canonicalBase, resolve(candidate));
    if (relation === '..' || relation.startsWith(`..${sep}`) || isAbsolute(relation)) {
      throw new ArtifactError('ARTIFACT_PATH_INVALID');
    }
  }
}
