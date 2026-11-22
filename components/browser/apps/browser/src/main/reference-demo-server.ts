import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { extname, resolve, sep } from 'node:path';

export interface ReferenceDemoDefinition {
  id: 'chat' | 'song-generation' | 'transcription' | 'image-generation' | 'video-subtitles';
  /** Optional test-only port. Production demo servers always request an ephemeral port. */
  port?: number;
}

export interface ReferenceDemoServer {
  id: ReferenceDemoDefinition['id'];
  url: string;
  close(): Promise<void>;
}

export const REFERENCE_DEMOS: readonly ReferenceDemoDefinition[] = [
  { id: 'chat' },
  { id: 'song-generation' },
  { id: 'transcription' },
  { id: 'image-generation' },
  { id: 'video-subtitles' },
];

const CONTENT_TYPES: Readonly<Record<string, string>> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.srt': 'text/plain; charset=utf-8',
  '.mp4': 'video/mp4',
};

export async function startReferenceDemoServers(
  examplesRoot: string,
  definitions: readonly ReferenceDemoDefinition[] = REFERENCE_DEMOS,
): Promise<ReferenceDemoServer[]> {
  const servers: ReferenceDemoServer[] = [];
  try {
    for (const definition of definitions) {
      const siteRoot = resolve(examplesRoot, definition.id, 'dist');
      if (!existsSync(resolve(siteRoot, 'index.html'))) {
        throw new Error(`Reference demo build is missing: ${siteRoot}`);
      }
      servers.push(await startReferenceDemoServer(siteRoot, definition));
    }
    return servers;
  } catch (error) {
    await Promise.allSettled(servers.map((server) => server.close()));
    throw error;
  }
}

async function startReferenceDemoServer(
  siteRoot: string,
  definition: ReferenceDemoDefinition,
): Promise<ReferenceDemoServer> {
  const server = createServer((request, response) => {
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('X-Content-Type-Options', 'nosniff');

    try {
      const requestURL = new URL(
        request.url ?? '/',
        `http://${request.headers.host ?? 'localhost'}`,
      );
      if (requestURL.pathname === '/__oryn_demo_health') {
        response.setHeader('Content-Type', 'application/json; charset=utf-8');
        response.end(JSON.stringify({ id: definition.id, ready: true }));
        return;
      }

      if (request.method !== 'GET' && request.method !== 'HEAD') {
        response.statusCode = 405;
        response.setHeader('Allow', 'GET, HEAD');
        response.end('Method not allowed');
        return;
      }

      const relativePath = decodeURIComponent(
        requestURL.pathname === '/' ? 'index.html' : requestURL.pathname.slice(1),
      );
      const target = resolve(siteRoot, relativePath);
      if (target !== siteRoot && !target.startsWith(`${siteRoot}${sep}`)) {
        response.statusCode = 404;
        response.end('Not found');
        return;
      }
      if (!existsSync(target) || !statSync(target).isFile()) {
        response.statusCode = 404;
        response.end('Not found');
        return;
      }

      response.setHeader(
        'Content-Type',
        CONTENT_TYPES[extname(target)] ?? 'application/octet-stream',
      );
      if (request.method === 'HEAD') {
        response.end();
        return;
      }
      createReadStream(target).pipe(response);
    } catch {
      response.statusCode = 400;
      response.end('Bad request');
    }
  });

  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(definition.port ?? 0, '127.0.0.1', () => {
      server.removeListener('error', reject);
      resolveListen();
    });
  });
  const address = server.address();
  if (!address || typeof address === 'string') {
    await closeServer(server);
    throw new Error(`Reference demo ${definition.id} did not expose a TCP address.`);
  }

  return {
    id: definition.id,
    url: `http://localhost:${address.port}/`,
    close: () => closeServer(server),
  };
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolveClose, reject) => {
    server.close((error) => (error ? reject(error) : resolveClose()));
  });
}
