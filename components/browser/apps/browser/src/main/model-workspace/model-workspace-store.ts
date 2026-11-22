import { randomBytes } from 'node:crypto';

import {
  validateTaskRequirements,
  type ResolvedModelRef,
  type TaskRequirements,
} from '@ocno/oryn-protocol';

import { isRecord, JsonFileStore } from '../data/json-file-store';

const MAX_BINDINGS = 128;

export interface ModelWorkspaceBinding {
  reference: string;
  model: ResolvedModelRef;
  taskContract: string;
  selectedFiles?: string[];
  requirements?: TaskRequirements;
  trustMode?: 'oryn-verified' | 'user-trusted';
  runtimePack?: { packId: string; packVersion: string };
  createdAt: number;
}

interface WorkspaceData {
  version: 1;
  bindings: ModelWorkspaceBinding[];
}

export class ModelWorkspaceStore {
  private readonly store: JsonFileStore<WorkspaceData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(filePath, () => ({ version: 1, bindings: [] }), isWorkspaceData);
  }

  create(input: Omit<ModelWorkspaceBinding, 'reference' | 'createdAt'>): ModelWorkspaceBinding {
    const binding: ModelWorkspaceBinding = {
      ...structuredClone(input),
      reference: randomBytes(24).toString('base64url'),
      createdAt: Date.now(),
    };
    this.store.update((current) => ({
      version: 1,
      bindings: [
        ...current.bindings.filter((item) => item.reference !== binding.reference),
        binding,
      ]
        .sort((a, b) => b.createdAt - a.createdAt)
        .slice(0, MAX_BINDINGS),
    }));
    return structuredClone(binding);
  }

  get(reference: string): ModelWorkspaceBinding | undefined {
    const value = this.store.read().bindings.find((item) => item.reference === reference);
    return value ? structuredClone(value) : undefined;
  }
}

function isWorkspaceData(value: unknown): value is WorkspaceData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    Array.isArray(value.bindings) &&
    value.bindings.length <= MAX_BINDINGS &&
    value.bindings.every(isBinding)
  );
}

function isBinding(value: unknown): value is ModelWorkspaceBinding {
  if (!(
    isRecord(value) &&
    typeof value.reference === 'string' &&
    /^[A-Za-z0-9_-]{32,64}$/u.test(value.reference) &&
    typeof value.createdAt === 'number' &&
    Number.isSafeInteger(value.createdAt) &&
    typeof value.taskContract === 'string' &&
    value.taskContract.length > 0 &&
    value.taskContract.length <= 128 &&
    isRecord(value.model) &&
    value.model.provider === 'huggingface' &&
    typeof value.model.repository === 'string' &&
    typeof value.model.revision === 'string'
  )) {
    return false;
  }
  if (
    value.selectedFiles !== undefined &&
    (!Array.isArray(value.selectedFiles) ||
      value.selectedFiles.length < 1 ||
      value.selectedFiles.length > 4096 ||
      new Set(value.selectedFiles).size !== value.selectedFiles.length ||
      value.selectedFiles.some(
        (name: unknown) =>
          typeof name !== 'string' ||
          !name ||
          name.length > 1024 ||
          name.startsWith('/') ||
          name.includes('\\') ||
          name.includes('\0') ||
          name.split('/').includes('..'),
      ))
  )
    return false;
  const trustMode = value.trustMode ?? 'oryn-verified';
  if (value.requirements !== undefined) {
    try {
      const requirements = validateTaskRequirements(value.requirements);
      if (requirements.contract?.id !== value.taskContract) return false;
    } catch {
      return false;
    }
  }
  if (trustMode !== 'oryn-verified' && trustMode !== 'user-trusted') return false;
  if (value.runtimePack === undefined) return trustMode === 'user-trusted';
  return (
    isRecord(value.runtimePack) &&
    typeof value.runtimePack.packId === 'string' &&
    typeof value.runtimePack.packVersion === 'string'
  );
}
