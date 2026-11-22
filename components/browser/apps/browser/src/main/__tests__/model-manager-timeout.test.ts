import { afterEach, expect, it, vi } from 'vitest';
import { withRequestTimeout } from '../../renderer/src/request-timeout';

afterEach(() => vi.useRealTimers());

it('bounds an unresponsive inventory request and ignores a late response', async () => {
  vi.useFakeTimers();
  let finish!: (value: string) => void;
  const pending = new Promise<string>((resolve) => {
    finish = resolve;
  });
  const result = withRequestTimeout(pending);
  const rejected = expect(result).rejects.toThrow('Request timed out');
  await vi.advanceTimersByTimeAsync(30_000);
  await rejected;
  finish('late inventory');
  await expect(result).rejects.toThrow('Request timed out');
  expect(vi.getTimerCount()).toBe(0);
});

it('clears deadlines on successful and failed refreshes', async () => {
  vi.useFakeTimers();
  await expect(withRequestTimeout(Promise.resolve(['installed']))).resolves.toEqual(['installed']);
  await expect(withRequestTimeout(Promise.reject(new Error('Engine unavailable')))).rejects.toThrow(
    'Engine unavailable',
  );
  expect(vi.getTimerCount()).toBe(0);
});
