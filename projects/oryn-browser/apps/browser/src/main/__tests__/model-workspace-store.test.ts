import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { ModelWorkspaceStore } from '../model-workspace/model-workspace-store';
import {
  createOrynModelWorkspaceURL,
  readOrynModelWorkspaceReference,
} from '../../shared/browser-urls';

describe('ModelWorkspaceStore', () => {
  it('persists only opaque references while keeping exact authority browser-owned', () => {
    const store = new ModelWorkspaceStore(
      join(mkdtempSync(join(tmpdir(), 'oryn-workspace-')), 'bindings.json'),
    );
    const binding = store.create({
      model: {
        provider: 'huggingface',
        repository: 'google/gemma-3-1b-it',
        revision: 'a'.repeat(40),
      },
      taskContract: 'chat-v1',
      runtimePack: { packId: 'oryn.chat', packVersion: '1.0.0' },
    });
    const url = createOrynModelWorkspaceURL(binding.reference);
    expect(url).not.toContain('google');
    expect(url).not.toContain('gemma');
    expect(url).not.toContain(binding.model.revision);
    expect(readOrynModelWorkspaceReference(url)).toBe(binding.reference);
    expect(store.get(binding.reference)).toEqual(binding);
  });

  it.each([
    'oryn://model/short',
    'oryn://model/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/extra',
    'oryn://model/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?model=secret',
    'https://model/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  ])('rejects invalid workspace URLs: %s', (url) => {
    expect(readOrynModelWorkspaceReference(url)).toBeUndefined();
  });

  it('persists a user-trusted declarative binding without inventing a runtime pack', () => {
    const store = new ModelWorkspaceStore(
      join(mkdtempSync(join(tmpdir(), 'oryn-user-workspace-')), 'bindings.json'),
    );
    const binding = store.create({
      model: {
        provider: 'huggingface',
        repository: 'Qwen/Qwen3.5-4B',
        revision: 'b'.repeat(40),
      },
      taskContract: 'chat-v1',
      trustMode: 'user-trusted',
    });
    expect(store.get(binding.reference)).toEqual(binding);
    expect(binding.runtimePack).toBeUndefined();
  });
});

it('retains a complete file-set guard without changing the installed model identity', () => {
  const path = join(mkdtempSync(join(tmpdir(), 'oryn-workspace-files-')), 'bindings.json');
  const store = new ModelWorkspaceStore(path);
  const model = {
    provider: 'huggingface' as const,
    repository: 'org/model',
    revision: 'a'.repeat(40),
  };
  const binding = store.create({
    model,
    selectedFiles: ['config.json', 'model.safetensors'],
    taskContract: 'chat-v1',
    trustMode: 'user-trusted',
    requirements: { task: 'text-generation', contract: { id: 'chat-v1', revision: 1 } },
  });
  expect(new ModelWorkspaceStore(path).get(binding.reference)).toEqual(binding);
  expect(binding.model).toEqual(model);
  expect(binding.model).not.toHaveProperty('files');
});
