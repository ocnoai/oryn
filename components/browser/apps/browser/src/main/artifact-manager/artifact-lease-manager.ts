import type { ArtifactRegistry } from './artifact-registry';
import type { ArtifactStorage } from './artifact-storage';
import {
  ArtifactError,
  type ArtifactOwner,
  type EngineInputArtifactBinding,
  sameArtifactOwner,
} from './types';

export interface ArtifactLeaseSet {
  bindings: EngineInputArtifactBinding[];
  release(): Promise<void>;
}

export interface ArtifactReadLease {
  artifactId: string;
  path: string;
  release(): Promise<void>;
}

export class ArtifactLeaseManager {
  private readonly engineReaders = new Map<string, number>();
  private readonly websiteReaders = new Map<string, number>();
  private cleanup?: (artifactId: string) => Promise<void>;

  constructor(
    private readonly registry: ArtifactRegistry,
    private readonly storage: ArtifactStorage,
  ) {}

  setCleanupHandler(cleanup: (artifactId: string) => Promise<void>): void {
    this.cleanup = cleanup;
  }

  acquireEngineReads(owner: ArtifactOwner, artifactIds: string[]): ArtifactLeaseSet {
    const uniqueIds = [...new Set(artifactIds)];
    const bindings = uniqueIds.map((artifactId) => {
      const record = this.registry.get(artifactId);
      if (!record || !sameArtifactOwner(record.owner, owner)) {
        throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      }
      if (record.state !== 'ready') throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      const internalPath = this.storage.resolveReady(
        record.storage.root,
        record.storage.relativePath,
      );
      return {
        artifactId: record.id,
        internalPath,
        mimeType: record.metadata.mimeType,
        sizeBytes: record.metadata.sizeBytes,
        name: record.metadata.name,
        access: 'read' as const,
      };
    });

    for (const artifactId of uniqueIds) {
      this.engineReaders.set(artifactId, this.count(artifactId) + 1);
    }
    let released = false;
    return {
      bindings,
      release: async () => {
        if (released) return;
        released = true;
        for (const artifactId of uniqueIds) {
          const next = Math.max(0, this.count(artifactId) - 1);
          if (next === 0) this.engineReaders.delete(artifactId);
          else this.engineReaders.set(artifactId, next);
          await this.cleanup?.(artifactId);
        }
      },
    };
  }

  count(artifactId: string): number {
    return (this.engineReaders.get(artifactId) ?? 0) + (this.websiteReaders.get(artifactId) ?? 0);
  }

  acquireWebsiteRead(record: {
    id: string;
    storage: { root: 'input' | 'published'; relativePath: string };
  }): ArtifactReadLease {
    const path = this.storage.resolveReady(record.storage.root, record.storage.relativePath);
    this.websiteReaders.set(record.id, (this.websiteReaders.get(record.id) ?? 0) + 1);
    let released = false;
    return {
      artifactId: record.id,
      path,
      release: async () => {
        if (released) return;
        released = true;
        const next = Math.max(0, (this.websiteReaders.get(record.id) ?? 0) - 1);
        if (next === 0) this.websiteReaders.delete(record.id);
        else this.websiteReaders.set(record.id, next);
        await this.cleanup?.(record.id);
      },
    };
  }
}
