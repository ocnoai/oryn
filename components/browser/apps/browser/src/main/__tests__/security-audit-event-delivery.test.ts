import { IPC } from '../../shared/ipc';
import { expect, it, vi } from 'vitest';
import { owner, setup } from './security-audit-job-fixture';
import { FakeSession, page, installFixture } from './security-audit-ipc-fixture';

it.each(['stable', 'replace-after-check', 'revoke-before-check'])(
  'revalidates document authority at final delivery: %s',
  async (mode) => {
    const f = setup();
    const target = page(owner.webContentsId, new FakeSession());
    const ipc = installFixture({}, f.registry, undefined, [target]);
    let document = 'A';
    const delivered: Array<{ document: string; channel: string; event: unknown }> = [];
    target.send.mockImplementation((channel, event) =>
      delivered.push({ document, channel, event }),
    );
    const caller = f.callerContexts;
    caller.hasDocumentAuthority = () => document === 'A';
    ipc.callers.hasDocumentAuthority = () => document === 'A';
    const scheduled: Array<() => void> = [];
    const original = setImmediate;
    try {
      const ref = await f.registry.invoke(owner, {
        sessionId: 'session_one',
        input: { audioArtifactId: 'artifact_abcdefghijklmnopqrstuvwxyz' },
      });
      vi.stubGlobal('setImmediate', (f: () => void) => {
        scheduled.push(f);
        return {};
      });
      if (mode === 'revoke-before-check') document = 'B';
      f.engine.emit('job-event', {
        v: 1,
        type: 'event',
        event: 'job.completed',
        seq: 1,
        timestamp: 1,
        data: {
          jobId: ref.jobId,
          sessionId: 'session_one',
          sequence: 1,
          result: { kind: 'structured', data: { text: 'AUDIT_CANARY_A' } },
        },
      });
      const chain = (
        f.registry as unknown as { eventChains: Map<string, Promise<void>> }
      ).eventChains.get(ref.jobId);
      await chain;
      if (mode === 'replace-after-check') document = 'B';
      expect(scheduled.length).toBe(mode === 'revoke-before-check' ? 0 : 1);
      for (const callback of scheduled) callback();
      if (mode !== 'stable') expect(delivered).toHaveLength(0);
      else {
        expect(delivered).toHaveLength(1);
        expect(delivered[0]!.document).toBe('A');
        expect(delivered[0]!.channel).toBe(IPC.localAIJobEvent);
        expect(JSON.stringify(delivered[0]!.event)).toContain('AUDIT_CANARY_A');
      }
      console.log(
        JSON.stringify({
          id: 'NV-04',
          scope: 'real registry and actual IPC forwarding; synthetic navigation boundary',
          mode,
          deliveredTo: delivered.map((v) => v.document),
        }),
      );
    } finally {
      vi.stubGlobal('setImmediate', original);
      ipc.cleanup();
      f.registry.dispose();
      vi.unstubAllGlobals();
    }
  },
);
