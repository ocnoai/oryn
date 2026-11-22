import { expect, it, vi } from 'vitest';
import { fixture, owner, exact, request, profile } from './security-audit-model-fixture';

it('bounds same-origin admission before inventory and stops further inspection after cancellation', async () => {
  const f = fixture([]),
    admitted: Array<{
      model: ReturnType<typeof exact>;
      resolve: (value: ReturnType<typeof profile>) => void;
    }> = [];
  f.engine.inspectModel = vi.fn(
    (model) =>
      new Promise<ReturnType<typeof profile>>((resolve) => admitted.push({ model, resolve })),
  );
  f.prompts.on('prompt', (p) =>
    f.prompts.respondFromTrustedUI({ promptId: p.promptId, action: 'deny' }),
  );
  const abort = new AbortController();
  const r = {
    ...request(exact('validation/preferred')),
    certifiedModels: Array.from({ length: 32 }, (_, i) => exact(`validation/cert-${i}`)),
  };
  const pending = f.sessions
    .requestSession(r, owner, { operationId: 'model_request_admission', signal: abort.signal })
    .catch((error) => error);
  try {
    for (let i = 0; i < 20 && !admitted.length; i++) await Promise.resolve();
    expect(admitted).toHaveLength(1);
    await expect(f.sessions.requestSession(request(), owner)).rejects.toThrow(
      'LOCAL_AI_MODEL_REQUEST_BUSY',
    );
    expect(f.engine.getModelInventory).toHaveBeenCalledTimes(1);
    const peer = f.sessions
      .requestSession(request(exact('validation/peer')), {
        ...owner,
        origin: 'https://peer.test',
        frameUrl: 'https://peer.test/app',
        documentId: 'document_peer',
      })
      .catch((error) => error);
    for (let i = 0; i < 20 && admitted.length < 2; i++) await Promise.resolve();
    expect(admitted).toHaveLength(2);
    admitted[1]!.resolve(profile(admitted[1]!.model));
    expect((await peer).code).toBe('USER_DENIED');
    abort.abort();
    await expect(f.sessions.requestSession(request(), owner)).rejects.toThrow(
      'LOCAL_AI_MODEL_REQUEST_BUSY',
    );
    admitted[0]!.resolve(profile(admitted[0]!.model));
    expect((await pending).message).toBe('REQUEST_ABORTED');
    expect(admitted).toHaveLength(2); // No second A candidate after cancellation.
    f.engine.inspectModel = vi.fn(async (model) => profile(model));
    await expect(f.sessions.requestSession(request(), owner)).rejects.toMatchObject({
      code: 'USER_DENIED',
    });
    expect(f.engine.installModel).not.toHaveBeenCalled();
  } finally {
    abort.abort();
    for (const item of admitted) item.resolve(profile(item.model));
    await pending;
    f.cleanup();
  }
});

it('caps outstanding origins before Engine inventory and releases every slot on denial', async () => {
  const f = fixture([]),
    admitted: Array<{
      model: ReturnType<typeof exact>;
      resolve: (value: ReturnType<typeof profile>) => void;
    }> = [];
  f.engine.inspectModel = vi.fn(
    (model) =>
      new Promise<ReturnType<typeof profile>>((resolve) => admitted.push({ model, resolve })),
  );
  f.prompts.on('prompt', (p) =>
    f.prompts.respondFromTrustedUI({ promptId: p.promptId, action: 'deny' }),
  );
  const pending = Array.from({ length: 4 }, (_, i) =>
    f.sessions
      .requestSession(request(exact(`validation/model-${i}`)), {
        ...owner,
        origin: `https://site-${i}.test`,
      })
      .catch((error) => error),
  );
  try {
    for (let i = 0; i < 20 && admitted.length < 4; i++) await Promise.resolve();
    expect(admitted).toHaveLength(4);
    await expect(
      f.sessions.requestSession(request(), { ...owner, origin: 'https://site-over-cap.test' }),
    ).rejects.toThrow('LOCAL_AI_MODEL_REQUEST_BUSY');
    expect(f.engine.getModelInventory).toHaveBeenCalledTimes(4);
    for (const item of admitted) item.resolve(profile(item.model));
    expect((await Promise.all(pending)).every((error) => error.code === 'USER_DENIED')).toBe(true);
    f.engine.inspectModel = vi.fn(async (model) => profile(model));
    await expect(f.sessions.requestSession(request(), owner)).rejects.toMatchObject({
      code: 'USER_DENIED',
    });
  } finally {
    for (const item of admitted) item.resolve(profile(item.model));
    await Promise.all(pending);
    f.cleanup();
  }
});
