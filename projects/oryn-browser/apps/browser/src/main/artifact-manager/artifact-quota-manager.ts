import { statfsSync } from 'node:fs';

import type { ArtifactRecord } from './types';
import { ArtifactError } from './types';

export interface ArtifactQuotaPolicy {
  maxArtifactBytes: number;
  maxTemporaryBytes: number;
  freeDiskReserveBytes: number;
}

const DEFAULT_POLICY: ArtifactQuotaPolicy = {
  maxArtifactBytes: 20 * 1024 ** 3,
  maxTemporaryBytes: 10 * 1024 ** 3,
  freeDiskReserveBytes: 1024 ** 3,
};

export class ArtifactQuotaManager {
  private readonly reservations = new Map<string, number>();

  constructor(
    private readonly storageRoot: string,
    private readonly listRecords: () => ArtifactRecord[],
    private readonly policy: ArtifactQuotaPolicy = DEFAULT_POLICY,
    private readonly availableBytes: () => number = () => {
      const stat = statfsSync(this.storageRoot);
      return Number(stat.bavail) * Number(stat.bsize);
    },
  ) {}

  reserve(operationId: string, sizeBytes: number, alreadyStored = false): void {
    if (!Number.isSafeInteger(sizeBytes) || sizeBytes < 0) {
      throw new ArtifactError('ARTIFACT_INVALID');
    }
    if (this.reservations.has(operationId)) throw new ArtifactError('ARTIFACT_INVALID');
    const retained = this.listRecords()
      .filter((record) => record.state !== 'creating')
      .reduce((total, record) => total + record.metadata.sizeBytes, 0);
    const pending = [...this.reservations.values()].reduce((total, value) => total + value, 0);
    if (
      pending + sizeBytes > this.policy.maxTemporaryBytes ||
      retained + pending + sizeBytes > this.policy.maxArtifactBytes
    ) {
      throw new ArtifactError('RESOURCE_LIMIT_EXCEEDED');
    }
    const additionalBytes = pending + (alreadyStored ? 0 : sizeBytes);
    if (this.availableBytes() - this.policy.freeDiskReserveBytes < additionalBytes) {
      throw new ArtifactError('DISK_SPACE_INSUFFICIENT');
    }
    this.reservations.set(operationId, sizeBytes);
  }

  release(operationId: string): void {
    this.reservations.delete(operationId);
  }
}
