import { PROTOCOL_VERSION } from '@ocno/oryn-protocol';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('../paths', () => ({ engineLaunchPaths: vi.fn() }));

import { EngineSupervisor } from '../engine-supervisor';

afterEach(() => vi.restoreAllMocks());

describe('engine response settlement', () => {
  it.each(['model.inspect', 'model.inventory.list'])(
    'rejects the waiting %s request when response validation fails',
    (method) => {
      vi.spyOn(console, 'error').mockImplementation(() => {});
      const supervisor = new EngineSupervisor();
      const resolve = vi.fn();
      const reject = vi.fn();
      const kill = vi.fn();
      const internals = supervisor as unknown as {
        pending: Map<string, unknown>;
        child: unknown;
        handleLine(line: string): void;
      };
      internals.child = { kill };
      internals.pending.set('request_test', { kind: 'request', method, resolve, reject });

      internals.handleLine(
        JSON.stringify({
          v: PROTOCOL_VERSION,
          type: 'response',
          id: 'request_test',
          ok: true,
          result: {},
        }),
      );

      expect(reject).toHaveBeenCalledExactlyOnceWith(expect.any(Error));
      expect(resolve).not.toHaveBeenCalled();
      expect(internals.pending.size).toBe(0);
      expect(supervisor.state).toBe('error');
      expect(kill).toHaveBeenCalledWith('SIGKILL');
    },
  );
});
