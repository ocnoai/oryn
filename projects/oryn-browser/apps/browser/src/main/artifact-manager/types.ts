export type ArtifactState = 'creating' | 'ready' | 'released' | 'expired' | 'deleting' | 'error';
export type ArtifactLifetime = 'ephemeral' | 'session' | 'persistent';
export type ArtifactSource = 'website-input' | 'engine-output' | 'engine-preview';

export interface ArtifactOwner {
  origin: string;
  tabId: string;
  webContentsId: number;
  frameId: number;
  documentId: string;
}

export interface ArtifactRecord {
  id: string;
  state: ArtifactState;
  source: ArtifactSource;
  lifetime: ArtifactLifetime;
  owner: ArtifactOwner;
  metadata: {
    mimeType: string;
    sizeBytes: number;
    name?: string;
    createdAt: number;
    expiresAt?: number;
  };
  storage: {
    root: 'input' | 'published';
    relativePath: string;
  };
}

export interface ArtifactReference {
  id: string;
  mimeType: string;
  sizeBytes: number;
  name?: string;
}

export interface ArtifactMetadata extends ArtifactReference {
  createdAt: number;
  expiresAt?: number;
}

export interface EngineInputArtifactBinding {
  artifactId: string;
  internalPath: string;
  mimeType: string;
  sizeBytes: number;
  name?: string;
  access: 'read';
}

export interface EngineOutputArtifact {
  outputId: string;
  internalPath: string;
  mimeType: string;
  sizeBytes: number;
  name?: string;
}

export type EngineInferenceResult =
  | { kind: 'structured'; value: unknown }
  | { kind: 'artifact'; artifact: EngineOutputArtifact }
  | { kind: 'multi'; outputs: EngineInferenceResult[] };

export type ArtifactErrorCode =
  | 'ARTIFACT_NOT_FOUND'
  | 'ARTIFACT_ACCESS_DENIED'
  | 'ARTIFACT_EXPIRED'
  | 'ARTIFACT_INVALID'
  | 'ARTIFACT_IO_ERROR'
  | 'ARTIFACT_PATH_INVALID'
  | 'RESOURCE_LIMIT_EXCEEDED'
  | 'DISK_SPACE_INSUFFICIENT';

export class ArtifactError extends Error {
  constructor(readonly code: ArtifactErrorCode) {
    super(code);
    this.name = 'ArtifactError';
  }
}

export function sameArtifactOwner(left: ArtifactOwner, right: ArtifactOwner): boolean {
  return (
    left.origin === right.origin &&
    left.tabId === right.tabId &&
    left.webContentsId === right.webContentsId &&
    left.frameId === right.frameId &&
    left.documentId === right.documentId
  );
}
