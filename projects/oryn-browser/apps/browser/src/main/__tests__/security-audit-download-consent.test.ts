import { expect, it, vi } from 'vitest';
import { ModelManagerService } from '../model-manager/model-manager-service';
import { fixture, owner, exact, request } from './security-audit-model-fixture';

it('a deleted model needs fresh download consent even with saved Always Allow', async () => {
  const f = fixture();
  try {
    let promptCount = 0;
    f.prompts.on('prompt', (p) => {
      promptCount++;
      f.prompts.respondFromTrustedUI({
        promptId: p.promptId,
        action: promptCount === 2 ? 'deny' : 'always-allow',
      });
    });
    const session = await f.sessions.requestSession(request(), owner);
    await f.sessions.releaseOwned(session.sessionId, owner);
    expect(promptCount).toBe(1);
    expect(f.engine.installModel).not.toHaveBeenCalled();
    const manager = new ModelManagerService(
      f.engine as unknown as ConstructorParameters<typeof ModelManagerService>[0],
      f.resolver,
      f.preferences,
      f.sessions,
      { cancelAllForModel: vi.fn() } as unknown as ConstructorParameters<
        typeof ModelManagerService
      >[4],
      {} as ConstructorParameters<typeof ModelManagerService>[5],
    );
    expect((await manager.remove('install_0', 'delete')).state).toBe('removed');
    expect(f.models).toHaveLength(0);
    expect(f.permissions.decision({ origin: owner.origin, role: 'chat', model: exact() })).toBe(
      'always-allow',
    );
    await expect(f.sessions.requestSession(request(), owner)).rejects.toMatchObject({
      code: 'USER_DENIED',
    });
    expect(promptCount).toBe(2);
    expect(f.engine.installModel).not.toHaveBeenCalled();
    const again = await f.sessions.requestSession(request(), owner);
    expect(again.resolution.downloadRequired).toBe(true);
    expect(f.engine.installModel).toHaveBeenCalledTimes(1);
    expect(f.engine.installModel.mock.calls[0]![1]).toEqual(exact());
    expect(promptCount).toBe(3);
    await f.sessions.releaseOwned(again.sessionId, owner);
    f.permissions.revoke({ origin: owner.origin, role: 'chat', model: exact() });
    f.models.splice(0);
    await expect(f.sessions.requestSession(request(), owner)).rejects.toMatchObject({
      code: 'USER_DENIED',
    });
    expect(f.engine.installModel).toHaveBeenCalledTimes(1);
    console.log(
      JSON.stringify({
        id: 'NV-06',
        scope: 'real Browser orchestration; simulated Engine',
        promptCount,
        installCalls: 1,
        revokedBlocks: true,
      }),
    );
  } finally {
    f.cleanup();
  }
});
