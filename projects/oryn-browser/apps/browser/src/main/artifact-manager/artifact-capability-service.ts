import { randomBytes } from 'node:crypto';

import type { ArtifactOwner, ArtifactRecord } from './types';

const CAPABILITY_TTL_MS = 15 * 60 * 1000;
export const ARTIFACT_SCHEME = 'oryn-artifact';

interface ArtifactCapability {
  token: string;
  artifactId: string;
  owner: ArtifactOwner;
  expiresAt: number;
}

export class ArtifactCapabilityService {
  private readonly capabilities = new Map<string, ArtifactCapability>();

  constructor(private readonly hasDocumentAuthority: (owner: ArtifactOwner) => boolean) {}

  issue(record: ArtifactRecord): string {
    this.prune();
    const existing = [...this.capabilities.values()].find(
      (capability) =>
        capability.artifactId === record.id &&
        capability.owner.webContentsId === record.owner.webContentsId &&
        capability.owner.frameId === record.owner.frameId &&
        capability.owner.documentId === record.owner.documentId,
    );
    const capability =
      existing ??
      ({
        token: randomBytes(32).toString('base64url'),
        artifactId: record.id,
        owner: { ...record.owner },
        expiresAt: Date.now() + CAPABILITY_TTL_MS,
      } satisfies ArtifactCapability);
    this.capabilities.set(capability.token, capability);
    return `${ARTIFACT_SCHEME}://capability/${capability.token}`;
  }

  resolve(token: string): ArtifactCapability | undefined {
    const capability = this.capabilities.get(token);
    if (
      !capability ||
      capability.expiresAt <= Date.now() ||
      !this.hasDocumentAuthority(capability.owner)
    ) {
      if (capability) this.capabilities.delete(token);
      return undefined;
    }
    return capability;
  }

  authorizeNetworkRequest(token: string, webContentsId: number, frameId?: number): boolean {
    const capability = this.resolve(token);
    return Boolean(
      capability &&
      capability.owner.webContentsId === webContentsId &&
      (frameId === undefined || capability.owner.frameId === frameId),
    );
  }

  revokeArtifact(artifactId: string): void {
    for (const [token, capability] of this.capabilities) {
      if (capability.artifactId === artifactId) this.capabilities.delete(token);
    }
  }

  revokeDocument(owner: Pick<ArtifactOwner, 'webContentsId' | 'documentId'>): void {
    for (const [token, capability] of this.capabilities) {
      if (
        capability.owner.webContentsId === owner.webContentsId &&
        capability.owner.documentId === owner.documentId
      ) {
        this.capabilities.delete(token);
      }
    }
  }

  static tokenFromUrl(value: string): string | undefined {
    try {
      const url = new URL(value);
      const token = url.protocol === `${ARTIFACT_SCHEME}:` ? url.pathname.slice(1) : '';
      return /^[A-Za-z0-9_-]{40,80}$/u.test(token) ? token : undefined;
    } catch {
      return undefined;
    }
  }

  private prune(): void {
    for (const token of this.capabilities.keys()) this.resolve(token);
  }
}
