import { EventEmitter } from 'node:events';
import { expect, it, vi } from 'vitest';
import { BrowserJobRegistry, BrowserJobError } from '../jobs/browser-job-registry';
import { owner, setup } from './security-audit-job-fixture';

it('rejects an unowned session before walking a shared input graph', async () => {
  const engine = new EventEmitter();
  const requireOwned = vi.fn(() => {
    throw new BrowserJobError('SESSION_NOT_OWNED');
  });
  const registry = new BrowserJobRegistry(
    engine as never,
    { requireOwned } as never,
    {} as never,
    {} as never,
    {} as never,
  );
  const nodes = new WeakSet<object>();
  let input: unknown = null;
  for (let depth = 0; depth < 10; depth++) {
    input = [input, input];
    nodes.add(input as object);
  }
  const original = Array.prototype[Symbol.iterator];
  let edges = 0,
    error: unknown;
  Array.prototype[Symbol.iterator] = function* () {
    for (const child of original.call(this)) {
      if (nodes.has(this)) edges++;
      yield child;
    }
    return undefined;
  };
  try {
    await registry.invoke(owner, { sessionId: 'session_unknown', input });
  } catch (value) {
    error = value;
  } finally {
    Array.prototype[Symbol.iterator] = original;
    registry.dispose();
  }
  expect(error).toMatchObject({ code: 'SESSION_NOT_OWNED' });
  expect(requireOwned).toHaveBeenCalledOnce();
  expect(edges).toBe(0);
});

it('rejects excessive expanded JSON before serializing the container and accepts a normal owned input', async () => {
  const f = setup();
  // Six small shared arrays and one 20 KB string expand to about 1.28 MiB.
  // The test forbids serializing that container, rather than allocating a stress payload.
  let input: unknown = 'x'.repeat(20_000);
  for (let depth = 0; depth < 6; depth++) input = [input, input];
  const stringify = JSON.stringify;
  let serializedContainer = false,
    error: unknown;
  const spy = vi.spyOn(JSON, 'stringify').mockImplementation(((value: unknown) => {
    if (value === input) {
      serializedContainer = true;
      throw new Error('AUDIT_WHOLE_GRAPH_SERIALIZE');
    }
    return stringify(value);
  }) as never);
  try {
    await f.registry.invoke(owner, { sessionId: 'session_one', input });
  } catch (value) {
    error = value;
  } finally {
    spy.mockRestore();
  }
  try {
    expect(error).toMatchObject({ code: 'JOB_INPUT_INVALID' });
    expect(serializedContainer).toBe(false);
    expect(f.engine.createJob).not.toHaveBeenCalled();
    const cycle: unknown[] = [];
    cycle.push(cycle);
    await expect(
      f.registry.invoke(owner, { sessionId: 'session_one', input: cycle }),
    ).rejects.toMatchObject({ code: 'JOB_INPUT_INVALID' });
    await f.registry.invoke(owner, {
      sessionId: 'session_one',
      input: { audioArtifactId: 'artifact_abcdefghijklmnopqrstuvwxyz' },
    });
    expect(f.engine.createJob).toHaveBeenCalledOnce();
  } finally {
    f.registry.dispose();
  }
});
