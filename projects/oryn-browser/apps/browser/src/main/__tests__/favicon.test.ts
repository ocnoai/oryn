import type { WebContents } from 'electron';
import { expect, it, vi } from 'vitest';
import { resolveFaviconDataURL } from '../favicon';

it('fetches page-selected icons without session credentials and rejects URL credentials', async () => {
  const fetch = vi.fn(
    async () =>
      new Response(new Uint8Array([1, 2, 3]), {
        headers: { 'Content-Type': 'image/png' },
      }),
  );
  const value = await resolveFaviconDataURL(
    { session: { fetch } } as unknown as WebContents,
    ['https://user:password@fixture.test/icon', 'https://fixture.test/icon'],
    new AbortController().signal,
  );
  expect(fetch).toHaveBeenCalledExactlyOnceWith(
    'https://fixture.test/icon',
    expect.objectContaining({ credentials: 'omit' }),
  );
  expect(value).toBe('data:image/png;base64,AQID');
});

it('bounds actual streamed bytes even without Content-Length and cancels the stream', async () => {
  const cancel = vi.fn();
  const fetch = vi.fn(
    async () =>
      new Response(
        new ReadableStream({
          pull(controller) {
            controller.enqueue(new Uint8Array(128 * 1024));
          },
          cancel,
        }),
        { headers: { 'Content-Type': 'image/png' } },
      ),
  );
  const contents = { session: { fetch } } as unknown as WebContents;
  const value = await resolveFaviconDataURL(
    contents,
    ['https://fixture.test/huge'],
    new AbortController().signal,
  );
  expect(value).toBeUndefined();
  expect(cancel).toHaveBeenCalledTimes(1);
});

it('caps advertised candidates and propagates cancellation to network requests', async () => {
  const abort = new AbortController();
  const fetch = vi.fn(async (_url: string, options: { signal: AbortSignal }) => {
    expect(options.signal.aborted).toBe(true);
    throw new Error('aborted');
  });
  abort.abort();
  await resolveFaviconDataURL(
    { session: { fetch } } as unknown as WebContents,
    Array(100).fill('https://fixture.test/icon'),
    abort.signal,
  );
  expect(fetch.mock.calls.length).toBeLessThanOrEqual(4);
});
