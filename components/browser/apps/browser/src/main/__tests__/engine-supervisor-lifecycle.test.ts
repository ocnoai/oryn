import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PassThrough, Writable } from 'node:stream';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('node:child_process', () => ({ spawn: vi.fn() }));
vi.mock('../paths', () => ({ engineLaunchPaths: vi.fn() }));

import { spawn } from 'node:child_process';
import { engineLaunchPaths } from '../paths';
import { EngineSupervisor } from '../engine-supervisor';

const resources = {
  snapshotId: 'resource_startup',
  timestamp: 1,
  memoryPressure: 'normal',
  systemMemory: { totalBytes: 32_000_000_000, availableBytes: 16_000_000_000 },
  processMemory: { residentBytes: 100_000_000 },
  localAI: { memoryBudgetBytes: 16_000_000_000, estimatedResidentModelBytes: 0 },
  accelerator: { backend: 'cpu' },
  generationLane: { capacity: 1, running: 0, queued: 0 },
};

class FakeEngine extends EventEmitter {
  readonly pid = 123;
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  killed = false;
  credentialDelay: number | undefined = 0;
  inventoryDelay = 0;
  failInventory = false;
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly acknowledgments = new PassThrough();
  readonly stdin = new Writable({
    write: (chunk, _encoding, callback) => {
      const request = JSON.parse(chunk.toString());
      const respond = () => {
        let result: unknown;
        if (request.method === 'protocol.handshake') {
          result = {
            selectedVersion: 1,
            engineVersion: '0.1.0',
            engineInstanceId: 'engine_startup',
            maxMessageBytes: 16 * 1024 * 1024,
          };
        } else if (request.method === 'runtime.policy.update') {
          result = { appliedRevision: request.params.policyRevision, resources };
        } else if (request.method === 'protocol.ping') {
          result = { engineInstanceId: 'engine_startup', state: 'ready' };
        } else if (request.method === 'model.inventory.list') {
          result = { models: [] };
        }
        const failure = this.failInventory && request.method === 'model.inventory.list';
        this.stdout.write(
          `${JSON.stringify({
            ...(request.method === 'protocol.handshake' ? {} : { v: 1 }),
            type: 'response',
            id: request.id,
            ok: !failure,
            ...(failure
              ? {
                  error: {
                    code: 'PROTOCOL_INTERNAL_ERROR',
                    category: 'protocol',
                    message: 'Inventory unavailable',
                    retryable: false,
                  },
                }
              : { result }),
          })}\n`,
        );
      };
      setTimeout(respond, request.method === 'model.inventory.list' ? this.inventoryDelay : 0);
      callback();
    },
  });
  readonly credentialInput = new Writable({
    write: (chunk, _encoding, callback) => {
      const { revision } = JSON.parse(chunk.toString());
      if (this.credentialDelay !== undefined) {
        setTimeout(() => this.acknowledgments.write(`${revision}\n`), this.credentialDelay);
      }
      callback();
    },
  });
  readonly stdio = [
    this.stdin,
    this.stdout,
    this.stderr,
    this.credentialInput,
    this.acknowledgments,
  ];
  readonly kill = vi.fn((signal: NodeJS.Signals) => {
    this.killed = true;
    setTimeout(() => {
      this.signalCode = signal;
      this.emit('exit', null, signal);
      this.emit('close', null, signal);
    }, 50);
    return true;
  });
}

let root: string;
beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  root = mkdtempSync(join(tmpdir(), 'oryn-engine-startup-'));
  vi.mocked(engineLaunchPaths).mockReturnValue({
    engineDirectory: root,
    pythonExecutable: process.execPath,
    pythonArguments: [],
    packaged: false,
    sandboxRequired: false,
  });
});
afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.mocked(spawn).mockReset();
  rmSync(root, { recursive: true, force: true });
});

function useChildren(...children: FakeEngine[]) {
  for (const child of children) {
    vi.mocked(spawn).mockReturnValueOnce(child as unknown as ReturnType<typeof spawn>);
  }
}

describe('EngineSupervisor startup lifecycle', () => {
  it('allows a cold credential startup beyond five seconds and shares concurrent startup', async () => {
    const child = new FakeEngine();
    child.credentialDelay = 6_000;
    useChildren(child);
    const supervisor = new EngineSupervisor();
    const first = supervisor.start();
    const concurrent = supervisor.start();
    await vi.advanceTimersByTimeAsync(6_100);
    await Promise.all([first, concurrent]);
    expect(spawn).toHaveBeenCalledTimes(1);
    expect(supervisor.state).toBe('ready');
    expect(supervisor.lastResources).toEqual(resources);
    expect(child.kill).not.toHaveBeenCalled();

    // Credentials changed on an already running Engine keep their shorter deadline.
    child.credentialDelay = undefined;
    const update = supervisor.updateHuggingFaceCredential(undefined);
    const rejected = expect(update).rejects.toThrow('did not acknowledge credential state');
    await vi.advanceTimersByTimeAsync(5_000);
    await rejected;
    expect(supervisor.state).toBe('ready');
  });

  it('uses one startup deadline, cleans up before retry, and ignores stale child callbacks', async () => {
    const slow = new FakeEngine();
    slow.credentialDelay = 25_000;
    slow.inventoryDelay = 6_000;
    const replacement = new FakeEngine();
    useChildren(slow, replacement);
    const supervisor = new EngineSupervisor();
    const first = supervisor.start();
    const rejected = expect(first).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(30_000);
    expect(supervisor.state).toBe('error');
    expect(slow.kill).toHaveBeenCalledWith('SIGTERM');
    const concurrent = expect(supervisor.start()).rejects.toThrow('timed out');
    expect(spawn).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(50);
    await Promise.all([rejected, concurrent]);
    await vi.advanceTimersByTimeAsync(600);
    expect(spawn).toHaveBeenCalledTimes(2);
    expect(supervisor.state).toBe('ready');

    slow.stdout.write('unrelated stale output\n');
    slow.acknowledgments.write('invalid stale acknowledgment\n');
    slow.stderr.write('old diagnostics');
    slow.emit('error', new Error('old process failure'));
    slow.emit('exit', 1, null);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(supervisor.state).toBe('ready');
    expect(supervisor.engineInstanceId).toBe('engine_startup');
    expect(supervisor.diagnosticLog).toBe('');
    expect(replacement.kill).not.toHaveBeenCalled();
    expect(spawn).toHaveBeenCalledTimes(2);
  });

  it('retains only a closed failure reason and exit code across process error and exit', async () => {
    const child = new FakeEngine();
    useChildren(child);
    const supervisor = new EngineSupervisor();
    const started = supervisor.start();
    await vi.advanceTimersByTimeAsync(100);
    await started;
    expect(supervisor.failureSummary).toEqual({ count: 0 });
    child.emit('error', new Error('secret stderr https://private.example /private/profile'));
    child.exitCode = 23;
    child.emit('exit', 23, null);
    expect(supervisor.failureSummary).toEqual({
      count: 1,
      last: { reason: 'process-error', exitCode: 23 },
    });
    const summary = supervisor.failureSummary;
    summary.count = 999;
    expect(supervisor.failureSummary.count).toBe(1);
    expect(JSON.stringify(supervisor.failureSummary)).not.toContain('secret');
  });

  it('records an unexpected exit independently from planned shutdown', async () => {
    const child = new FakeEngine();
    useChildren(child);
    const supervisor = new EngineSupervisor();
    const started = supervisor.start();
    await vi.advanceTimersByTimeAsync(100);
    await started;
    child.exitCode = 7;
    child.emit('exit', 7, null);
    expect(supervisor.failureSummary).toEqual({
      count: 1,
      last: { reason: 'unexpected-exit', exitCode: 7 },
    });
  });

  it('does not count an intentional Engine shutdown as a failure', async () => {
    const child = new FakeEngine();
    useChildren(child);
    const supervisor = new EngineSupervisor();
    const started = supervisor.start();
    await vi.advanceTimersByTimeAsync(100);
    await started;
    const stopped = supervisor.stop();
    await vi.advanceTimersByTimeAsync(100);
    await stopped;
    expect(supervisor.state).toBe('stopped');
    expect(supervisor.failureSummary).toEqual({ count: 0 });
  });

  it('keeps repeated failures after the handshake within the restart limit', async () => {
    const children = Array.from({ length: 4 }, () => {
      const child = new FakeEngine();
      child.failInventory = true;
      return child;
    });
    useChildren(...children);
    const supervisor = new EngineSupervisor();
    const rejected = expect(supervisor.start()).rejects.toThrow('Inventory unavailable');
    await vi.advanceTimersByTimeAsync(10_000);
    await rejected;
    expect(supervisor.state).toBe('error');
    expect(spawn).toHaveBeenCalledTimes(4);
    expect(children.every((child) => child.kill.mock.calls.length === 1)).toBe(true);
  });
});
