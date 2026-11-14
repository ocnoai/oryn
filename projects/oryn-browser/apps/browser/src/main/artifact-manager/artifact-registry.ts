import { JsonFileStore, isRecord } from '../data/json-file-store';
import type { ArtifactRecord, ArtifactState } from './types';

interface ArtifactRegistryDocument {
  version: 1;
  records: ArtifactRecord[];
}

const ALLOWED_TRANSITIONS: Record<ArtifactState, ArtifactState[]> = {
  creating: ['ready', 'error', 'deleting'],
  ready: ['released', 'expired', 'deleting', 'error'],
  released: ['deleting'],
  expired: ['deleting'],
  deleting: ['error'],
  error: ['deleting'],
};

export class ArtifactRegistry {
  private readonly store: JsonFileStore<ArtifactRegistryDocument>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(
      filePath,
      () => ({ version: 1, records: [] }),
      isArtifactRegistryDocument,
    );
  }

  list(): ArtifactRecord[] {
    return this.store.read().records;
  }

  get(id: string): ArtifactRecord | undefined {
    return this.list().find((record) => record.id === id);
  }

  create(record: ArtifactRecord): void {
    this.store.update((document) => {
      if (document.records.some((item) => item.id === record.id)) {
        throw new Error('ARTIFACT_ID_CONFLICT');
      }
      return { ...document, records: [...document.records, record] };
    });
  }

  replace(record: ArtifactRecord): void {
    this.store.update((document) => {
      const previous = document.records.find((item) => item.id === record.id);
      if (!previous) throw new Error('ARTIFACT_RECORD_NOT_FOUND');
      if (previous.state !== record.state) this.assertTransition(previous.state, record.state);
      return {
        ...document,
        records: document.records.map((item) => (item.id === record.id ? record : item)),
      };
    });
  }

  transition(id: string, state: ArtifactState): ArtifactRecord {
    const record = this.get(id);
    if (!record) throw new Error('ARTIFACT_RECORD_NOT_FOUND');
    const next = { ...record, state };
    this.replace(next);
    return next;
  }

  remove(id: string): void {
    this.store.update((document) => ({
      ...document,
      records: document.records.filter((record) => record.id !== id),
    }));
  }

  private assertTransition(from: ArtifactState, to: ArtifactState): void {
    if (!ALLOWED_TRANSITIONS[from].includes(to)) {
      throw new Error(`ARTIFACT_STATE_TRANSITION_INVALID:${from}:${to}`);
    }
  }
}

function isArtifactRegistryDocument(value: unknown): value is ArtifactRegistryDocument {
  return (
    isRecord(value) &&
    value.version === 1 &&
    Array.isArray(value.records) &&
    value.records.every(isArtifactRecord)
  );
}

function isArtifactRecord(value: unknown): value is ArtifactRecord {
  if (!isRecord(value) || !isRecord(value.owner) || !isRecord(value.metadata)) return false;
  if (!isRecord(value.storage)) return false;
  return (
    typeof value.id === 'string' &&
    ['creating', 'ready', 'released', 'expired', 'deleting', 'error'].includes(
      String(value.state),
    ) &&
    ['website-input', 'engine-output', 'engine-preview'].includes(String(value.source)) &&
    ['ephemeral', 'session', 'persistent'].includes(String(value.lifetime)) &&
    typeof value.owner.origin === 'string' &&
    typeof value.owner.tabId === 'string' &&
    typeof value.owner.webContentsId === 'number' &&
    typeof value.owner.frameId === 'number' &&
    typeof value.owner.documentId === 'string' &&
    typeof value.metadata.mimeType === 'string' &&
    typeof value.metadata.sizeBytes === 'number' &&
    typeof value.metadata.createdAt === 'number' &&
    (value.metadata.name === undefined || typeof value.metadata.name === 'string') &&
    (value.metadata.expiresAt === undefined || typeof value.metadata.expiresAt === 'number') &&
    ['input', 'published'].includes(String(value.storage.root)) &&
    typeof value.storage.relativePath === 'string'
  );
}
