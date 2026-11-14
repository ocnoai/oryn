import type { ArtifactRegistry } from './artifact-registry';
import type { ArtifactStorage } from './artifact-storage';

export class ArtifactReconciler {
  constructor(
    private readonly registry: ArtifactRegistry,
    private readonly storage: ArtifactStorage,
  ) {}

  reconcile(): {
    removedPartials: number;
    removedOrphans: number;
    removedStagingRoots: number;
    stagingCleanupComplete: boolean;
    invalidRecords: number;
  } {
    const creating = this.registry.list().filter((record) => record.state === 'creating');
    for (const record of creating) this.registry.remove(record.id);
    this.storage.clearTemporary();
    const stagingCleanup = this.storage.clearEngineStaging();

    let invalidRecords = 0;
    for (const record of this.registry.list()) {
      if (
        record.state === 'ready' &&
        !this.storage.exists(record.storage.root, record.storage.relativePath)
      ) {
        this.registry.transition(record.id, 'error');
        invalidRecords += 1;
      }
    }

    const authorized = new Set(
      this.registry
        .list()
        .filter((record) => record.state === 'ready')
        .map((record) => `${record.storage.root}:${record.storage.relativePath}`),
    );
    let removedOrphans = 0;
    for (const root of ['input', 'published'] as const) {
      for (const relativePath of this.storage.listObjectFiles(root)) {
        if (!authorized.has(`${root}:${relativePath}`)) {
          this.storage.delete(root, relativePath);
          removedOrphans += 1;
        }
      }
    }
    return {
      removedPartials: creating.length,
      removedOrphans,
      removedStagingRoots: stagingCleanup.removedRoots,
      stagingCleanupComplete: stagingCleanup.complete,
      invalidRecords,
    };
  }
}
