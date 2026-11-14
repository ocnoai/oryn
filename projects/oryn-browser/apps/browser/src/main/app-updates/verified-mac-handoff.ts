import { Server, type IncomingMessage, type ServerResponse } from 'node:http';
import type { Readable } from 'node:stream';

import type { VerifiedUpdateArchive } from './verified-update-archive';

type RequestListener = (request: IncomingMessage, response: ServerResponse) => void;

/** Pinned MacUpdater 6.8.9 adapter: keep its authenticated root feed/native
 * installer, but serve the payload through retained verified read authority.
 * Its original payload handler reopens a mutable cache pathname. */
export class VerifiedMacHandoff {
  private readonly server: Server;
  private readonly original: RequestListener;
  private readonly handler: RequestListener;
  private readonly streams = new Set<Readable>();
  private archive: VerifiedUpdateArchive | undefined;
  private payloadPath: string | undefined;
  private closed = false;

  constructor(
    private readonly updater: { server?: Server },
    private readonly isCurrent: () => boolean,
  ) {
    const server = updater.server;
    const address = server?.address();
    const listeners = server?.listeners('request');
    if (
      !(server instanceof Server) ||
      !address ||
      typeof address === 'string' ||
      address.address !== '127.0.0.1' ||
      !listeners ||
      listeners.length !== 1 ||
      typeof listeners[0] !== 'function'
    )
      throw new Error('APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE');
    this.server = server;
    this.original = listeners[0] as RequestListener;
    this.handler = (request, response) => this.request(request, response);
    server.removeListener('request', this.original);
    server.on('request', this.handler);
  }

  attach(archive: VerifiedUpdateArchive): void {
    this.assertCurrent();
    if (this.archive) throw new Error('APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE');
    this.archive = archive;
  }

  async revalidate(): Promise<void> {
    this.assertCurrent();
    if (!this.archive) throw new Error('APP_UPDATE_PAYLOAD_UNVERIFIED');
    await this.archive.revalidate(AbortSignal.timeout(120_000));
    this.assertCurrent();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const stream of this.streams) stream.destroy();
    this.streams.clear();
    // Never restore the pathname-based payload handler after cancellation/failure.
    this.server.closeAllConnections();
    this.server.close();
    this.archive?.close();
    this.archive = undefined;
  }

  private assertCurrent(): void {
    if (
      this.closed ||
      !this.isCurrent() ||
      this.updater.server !== this.server ||
      this.server.listeners('request').length !== 1 ||
      this.server.listeners('request')[0] !== this.handler
    )
      throw new Error('APP_UPDATE_STALE_PAYLOAD');
  }

  private request(request: IncomingMessage, response: ServerResponse): void {
    try {
      this.assertCurrent();
      if (request.method !== 'GET' || !this.archive)
        throw new Error('APP_UPDATE_PAYLOAD_UNVERIFIED');
      if (request.url === '/') {
        // Retain the maintained updater's random Basic-auth credential and feed
        // response. Capture only its successful fixed-size loopback URL so the
        // payload route can be matched exactly (never a prefix match).
        const end = response.end;
        response.end = ((chunk?: unknown, ...rest: unknown[]) => {
          if (response.statusCode === 200) {
            try {
              const data = typeof chunk === 'string' ? Buffer.from(chunk) : chunk;
              if (!Buffer.isBuffer(data) || data.length > 4096) throw new Error();
              const parsed = JSON.parse(data.toString('utf8')) as {
                url?: unknown;
              };
              if (typeof parsed.url !== 'string') throw new Error();
              const url = new URL(parsed.url);
              const address = this.server.address();
              if (
                !address ||
                typeof address === 'string' ||
                url.origin !== `http://127.0.0.1:${address.port}` ||
                url.username ||
                url.password ||
                url.search ||
                url.hash ||
                !/^\/[a-f0-9]{128}\.zip$/u.test(url.pathname) ||
                (this.payloadPath !== undefined && this.payloadPath !== url.pathname)
              )
                throw new Error();
              this.payloadPath = url.pathname;
            } catch {
              response.statusCode = 503;
              return Reflect.apply(end, response, []);
            }
          }
          return Reflect.apply(end, response, [chunk, ...rest]);
        }) as ServerResponse['end'];
        this.original(request, response);
        return;
      }
      if (!this.payloadPath || request.url !== this.payloadPath) throw new Error();
      void this.serve(request, response).catch(() => {
        if (response.headersSent) response.destroy();
        else {
          response.statusCode = 503;
          response.end();
        }
      });
    } catch {
      response.statusCode = 503;
      response.end();
    }
  }

  private async serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
    await this.revalidate();
    const archive = this.archive!;
    const stream = archive.stream();
    this.streams.add(stream);
    const finish = () => {
      this.streams.delete(stream);
      stream.destroy();
    };
    response.once('close', finish);
    request.once('aborted', finish);
    stream.once('error', () => response.destroy());
    response.writeHead(200, {
      'Content-Type': 'application/zip',
      'Content-Length': archive.payload.size,
    });
    stream.pipe(response);
  }
}
