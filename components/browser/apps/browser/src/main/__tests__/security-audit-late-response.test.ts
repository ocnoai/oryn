import { Writable } from 'node:stream';
import { expect, it, vi } from 'vitest';
import { PROTOCOL_VERSION } from '@ocno/oryn-protocol';
vi.mock('../paths', () => ({ engineLaunchPaths: vi.fn() }));
import { EngineSupervisor } from '../engine-supervisor';
import { SerializedProtocolWriter } from '../protocol-transport';
import { BrowserJobRegistry } from '../jobs/browser-job-registry';
import { owner, setup } from './security-audit-job-fixture';
interface SupervisorInternals {
  child: { stdin?: Writable; kill: ReturnType<typeof vi.fn> };
  writer: SerializedProtocolWriter;
  _state: string;
  _engineInstanceId: string;
  handleLine(line: string): void;
  pending: Map<
    string,
    {
      kind: string;
      method: string;
      resolve: ReturnType<typeof vi.fn>;
      reject: ReturnType<typeof vi.fn>;
    }
  >;
}
interface RegistryInternals {
  jobs: Map<string, { state: string }>;
}

it.each([false, true])(
  'a valid response with late=%s does not fail unrelated work',
  async (late) => {
    vi.useFakeTimers();
    const supervisor = new EngineSupervisor();
    const internals = supervisor as unknown as SupervisorInternals;
    const wire: Array<{ id: string }> = [];
    const stdin = new Writable({
      write(chunk, _encoding, callback) {
        wire.push(JSON.parse(chunk.toString()));
        callback();
      },
    });
    const kill = vi.fn();
    internals.child = { stdin, kill };
    internals.writer = new SerializedProtocolWriter(stdin);
    internals._state = 'ready';
    internals._engineInstanceId = 'engine_validation';
    const f = setup();
    f.registry.dispose();
    // Create a real registry job through public invoke; only Engine job creation is doubled.
    vi.spyOn(supervisor, 'createJob').mockImplementation(async (jobId, sessionId) => ({
      jobId,
      sessionId,
      state: 'running',
    }));
    const source = f;
    const jobs = new BrowserJobRegistry(
      supervisor,
      source.sessions as unknown as ConstructorParameters<typeof BrowserJobRegistry>[1],
      source.artifacts as unknown as ConstructorParameters<typeof BrowserJobRegistry>[2],
      source.callerContexts as unknown as ConstructorParameters<typeof BrowserJobRegistry>[3],
      source.activity as unknown as ConstructorParameters<typeof BrowserJobRegistry>[4],
    );
    try {
      const job = await jobs.invoke(
        { ...owner, origin: 'https://peer-b.test' },
        {
          sessionId: 'session_one',
          input: { audioArtifactId: 'artifact_abcdefghijklmnopqrstuvwxyz' },
        },
      );
      const p1 = supervisor.request('model.inventory.list', {}, 10).then(
        (value) => ({ ok: true, value }),
        (e) => ({ ok: false, message: e.message }),
      );
      const p2 = supervisor.request('model.inventory.list', {}, 1000).then(
        (value) => ({ ok: true, value }),
        (e) => ({ ok: false, message: e.message }),
      );
      await vi.advanceTimersByTimeAsync(0);
      expect(wire).toHaveLength(2);
      if (late) await vi.advanceTimersByTimeAsync(11);
      internals.handleLine(
        JSON.stringify({
          v: PROTOCOL_VERSION,
          type: 'response',
          id: wire[0]!.id,
          ok: true,
          result: { models: [] },
        }),
      );
      if (late) {
        expect(await p1).toMatchObject({
          ok: false,
          message: expect.stringContaining('timed out'),
        });
        expect(supervisor.state).toBe('ready');
        expect(kill).not.toHaveBeenCalled();
        expect((jobs as unknown as RegistryInternals).jobs.get(job.jobId)?.state).toBe('running');
        expect(f.activity.complete).not.toHaveBeenCalled();
        internals.handleLine(
          JSON.stringify({
            v: PROTOCOL_VERSION,
            type: 'response',
            id: wire[1]!.id,
            ok: true,
            result: { models: [] },
          }),
        );
        expect(await p2).toMatchObject({ ok: true });
      } else {
        internals.handleLine(
          JSON.stringify({
            v: PROTOCOL_VERSION,
            type: 'response',
            id: wire[1]!.id,
            ok: true,
            result: { models: [] },
          }),
        );
        expect(await p1).toMatchObject({ ok: true });
        expect(await p2).toMatchObject({ ok: true });
        expect(supervisor.state).toBe('ready');
        expect(kill).not.toHaveBeenCalled();
        expect((jobs as unknown as RegistryInternals).jobs.get(job.jobId)?.state).toBe('running');
      }
      console.log(
        JSON.stringify({
          id: 'NV-09',
          late,
          scope:
            'real Supervisor timers/decoder/settlement and Registry peer-job failure; fake child',
          state: supervisor.state,
          killCalls: kill.mock.calls.length,
        }),
      );
    } finally {
      jobs.dispose();
      vi.clearAllTimers();
      vi.useRealTimers();
      vi.restoreAllMocks();
      stdin.destroy();
    }
  },
);

it('still treats an unsolicited response as protocol corruption', () => {
  const supervisor = new EngineSupervisor();
  const kill = vi.fn(),
    reject = vi.fn();
  const state = supervisor as unknown as SupervisorInternals;
  state.child = { kill };
  state._state = 'ready';
  state.pending.set('request_pending', {
    kind: 'request',
    method: 'model.inventory.list',
    resolve: vi.fn(),
    reject,
  });
  state.handleLine(
    JSON.stringify({
      v: PROTOCOL_VERSION,
      type: 'response',
      id: 'request_never_sent',
      ok: true,
      result: { models: [] },
    }),
  );
  expect(supervisor.state).toBe('error');
  expect(kill).toHaveBeenCalledExactlyOnceWith('SIGKILL');
  expect(reject).toHaveBeenCalledWith(expect.any(Error));
});
