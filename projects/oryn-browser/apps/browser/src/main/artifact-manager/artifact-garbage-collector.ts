import type { ArtifactLeaseManager } from './artifact-lease-manager';
import type { ArtifactRegistry } from './artifact-registry';
import type { ArtifactStorage } from './artifact-storage';

export class ArtifactGarbageCollector {
  constructor(
    private readonly registry: ArtifactRegistry,
    private readonly storage: ArtifactStorage,
    private readonly leases: ArtifactLeaseManager,
  ) {}

  async collectOne(artifactId: string): Promise<boolean> {
    const record = this.registry.get(artifactId);
    if (!record || !['released', 'expired', 'deleting', 'error'].includes(record.state)) {
      return false;
    }
    if (this.leases.count(artifactId) > 0) return false;
    if (record.state !== 'deleting') this.registry.transition(artifactId, 'deleting');
    this.storage.delete(record.storage.root, record.storage.relativePath);
    this.registry.remove(artifactId);
    return true;
  }

  async collect(): Promise<string[]> {
    const removed: string[] = [];
    for (const record of this.registry.list()) {
      if (await this.collectOne(record.id)) removed.push(record.id);
    }
    return removed;
  }
}
