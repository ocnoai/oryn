import type { WebContents } from 'electron';

const MAX_FAVICON_BYTES = 512 * 1024;

export function isFaviconDataURL(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.startsWith('data:image/') &&
    value.length <= MAX_FAVICON_BYTES * 2
  );
}

export async function resolveFaviconDataURL(
  contents: WebContents,
  candidates: string[],
  signal: AbortSignal,
): Promise<string | undefined> {
  for (const candidate of candidates.slice(0, 4)) {
    try {
      if (isFaviconDataURL(candidate)) {
        return candidate;
      }
      const url = new URL(candidate);
      if (!['http:', 'https:'].includes(url.protocol)) continue;
      if (url.username || url.password) continue;
      const response = await contents.session.fetch(url.toString(), {
        signal: AbortSignal.any([signal, AbortSignal.timeout(5000)]),
        cache: 'force-cache',
        // The page chooses this URL, but the request runs in the Browser process.
        // Do not grant it the session's cookies or cached authentication.
        credentials: 'omit',
      });
      if (!response.ok) continue;
      const responseType = response.headers.get('content-type')?.split(';')[0]?.trim();
      if (responseType && !responseType.startsWith('image/')) {
        await response.body?.cancel();
        continue;
      }
      const declaredLength = Number(response.headers.get('content-length') ?? 0);
      if (declaredLength > MAX_FAVICON_BYTES) {
        await response.body?.cancel();
        continue;
      }
      const reader = response.body?.getReader();
      if (!reader) continue;
      const chunks: Uint8Array[] = [];
      let length = 0;
      try {
        while (true) {
          const chunk = await reader.read();
          if (chunk.done) break;
          length += chunk.value.byteLength;
          if (length > MAX_FAVICON_BYTES) break;
          chunks.push(chunk.value);
        }
      } finally {
        await reader.cancel();
      }
      if (length > MAX_FAVICON_BYTES) continue;
      const bytes = Buffer.concat(chunks);
      if (bytes.byteLength === 0 || bytes.byteLength > MAX_FAVICON_BYTES) continue;
      const mimeType = responseType?.startsWith('image/') === true ? responseType : 'image/x-icon';
      return `data:${mimeType};base64,${bytes.toString('base64')}`;
    } catch {
      // Try the next icon candidate when a page advertises an unavailable resource.
    }
  }
  return undefined;
}
