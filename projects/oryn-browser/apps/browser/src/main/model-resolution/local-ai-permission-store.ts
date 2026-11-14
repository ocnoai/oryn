import { exactModelKey } from '@ocno/oryn-protocol';
import type { ResolvedModelRef } from '@ocno/oryn-protocol';

import { JsonFileStore, isRecord } from '../data/json-file-store';

export type LocalAIPermissionDecision = 'allow-once' | 'always-allow' | 'deny' | 'revoked';

export interface LocalAIPermissionKey {
  origin: string;
  role: string;
  model: ResolvedModelRef;
}

interface PersistedPermission extends LocalAIPermissionKey {
  decision: Exclude<LocalAIPermissionDecision, 'allow-once'>;
  updatedAt: string;
}

interface PermissionData {
  version: 1;
  records: PersistedPermission[];
}

export class LocalAIPermissionStore {
  private readonly store: JsonFileStore<PermissionData>;
  private readonly once = new Set<string>();

  constructor(filePath: string) {
    this.store = new JsonFileStore(filePath, () => ({ version: 1, records: [] }), isPermissionData);
  }

  decision(key: LocalAIPermissionKey): LocalAIPermissionDecision | undefined {
    const normalized = normalizeKey(key);
    if (this.once.has(permissionKey(normalized))) return 'allow-once';
    const persisted = this.store
      .read()
      .records.find((item) => permissionKey(item) === permissionKey(normalized))?.decision;
    return persisted === 'always-allow' && isLoopbackOrigin(normalized.origin)
      ? undefined
      : persisted;
  }

  recordTrustedDecision(key: LocalAIPermissionKey, decision: LocalAIPermissionDecision): void {
    const normalized = normalizeKey(key);
    const id = permissionKey(normalized);
    if (decision === 'always-allow' && isLoopbackOrigin(normalized.origin)) {
      this.once.add(id);
      return;
    }
    if (decision === 'allow-once') {
      this.once.add(id);
      return;
    }
    this.once.delete(id);
    this.store.update((current) => ({
      ...current,
      records: [
        ...current.records.filter((item) => permissionKey(item) !== id),
        { ...normalized, decision, updatedAt: new Date().toISOString() },
      ],
    }));
  }

  consumeOnce(key: LocalAIPermissionKey): boolean {
    return this.once.delete(permissionKey(normalizeKey(key)));
  }

  revoke(key: LocalAIPermissionKey): void {
    this.recordTrustedDecision(key, 'revoked');
  }

  list(): PersistedPermission[] {
    return this.store
      .read()
      .records.map((record) =>
        record.decision === 'always-allow' && isLoopbackOrigin(record.origin)
          ? { ...record, decision: 'revoked' }
          : record,
      );
  }

  revokeOrigin(origin: string): void {
    const normalized = new URL(origin).origin;
    this.store.update((current) => ({
      ...current,
      records: current.records.map((record) =>
        record.origin === normalized
          ? { ...record, decision: 'revoked' as const, updatedAt: new Date().toISOString() }
          : record,
      ),
    }));
    for (const value of [...this.once]) {
      if (value.startsWith(`${normalized}\n`)) this.once.delete(value);
    }
  }

  /** Returns saved blocks to the prompt state without granting access. */
  resetBlockedOrigin(origin: string): void {
    const normalized = new URL(origin).origin;
    this.store.update((current) => ({
      ...current,
      records: current.records.filter(
        (record) =>
          record.origin !== normalized ||
          (record.decision !== 'deny' && record.decision !== 'revoked'),
      ),
    }));
  }
}

function normalizeKey(key: LocalAIPermissionKey): LocalAIPermissionKey {
  return { ...key, origin: new URL(key.origin).origin };
}

function permissionKey(key: LocalAIPermissionKey): string {
  return `${key.origin}\n${key.role}\n${exactModelKey(key.model)}`;
}

export function isLoopbackOrigin(origin: string): boolean {
  try {
    const hostname = new URL(origin).hostname.toLowerCase();
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]';
  } catch {
    return false;
  }
}

function isPermissionData(value: unknown): value is PermissionData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    Array.isArray(value.records) &&
    value.records.every(
      (item) =>
        isRecord(item) &&
        typeof item.origin === 'string' &&
        typeof item.role === 'string' &&
        isRecord(item.model) &&
        item.model.provider === 'huggingface' &&
        typeof item.model.repository === 'string' &&
        typeof item.model.revision === 'string' &&
        ['always-allow', 'deny', 'revoked'].includes(String(item.decision)) &&
        typeof item.updatedAt === 'string',
    )
  );
}
