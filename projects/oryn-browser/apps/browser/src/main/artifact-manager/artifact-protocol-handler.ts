import { createReadStream, type ReadStream } from 'node:fs';

import { ArtifactCapabilityService } from './artifact-capability-service';
import type { ArtifactManager } from './artifact-manager';

export interface ArtifactProtocolRequest {
  url: string;
  method: string;
  range?: string;
}

export interface ArtifactProtocolResponse {
  status: number;
  headers: Record<string, string>;
  body?: ReadStream;
}

export class ArtifactProtocolHandler {
  constructor(private readonly artifacts: ArtifactManager) {}

  async serve(request: ArtifactProtocolRequest): Promise<ArtifactProtocolResponse> {
    const token = ArtifactCapabilityService.tokenFromUrl(request.url);
    if (!token) return errorResponse(404);
    if (request.method !== 'GET' && request.method !== 'HEAD') return errorResponse(405);

    let access: ReturnType<ArtifactManager['acquireCapabilityRead']>;
    try {
      access = this.artifacts.acquireCapabilityRead(token);
    } catch {
      return errorResponse(403);
    }
    const { record } = access;
    const size = record.metadata.sizeBytes;
    const headers: Record<string, string> = {
      'Content-Type': record.metadata.mimeType,
      'Content-Length': String(size),
      'Accept-Ranges': 'bytes',
      'X-Content-Type-Options': 'nosniff',
      'Cache-Control': 'private, no-store',
      'Access-Control-Allow-Origin': record.owner.origin,
      Vary: 'Origin',
    };
    if (record.metadata.mimeType === 'application/octet-stream') {
      headers['Content-Disposition'] =
        `attachment; filename="${headerFilename(record.metadata.name)}"`;
      headers['Content-Security-Policy'] = "default-src 'none'; sandbox";
    }

    const range = request.range === undefined ? undefined : parseRange(request.range, size);
    if (request.range !== undefined && !range) {
      await access.release();
      return {
        status: 416,
        headers: {
          ...headers,
          'Content-Range': `bytes */${size}`,
          'Content-Length': '0',
        },
      };
    }
    if (range) {
      headers['Content-Range'] = `bytes ${range.start}-${range.end}/${size}`;
      headers['Content-Length'] = String(range.end - range.start + 1);
    }
    if (request.method === 'HEAD') {
      await access.release();
      return { status: range ? 206 : 200, headers };
    }

    const body = createReadStream(access.path, range ? range : undefined);
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      void access.release();
    };
    body.once('end', release);
    body.once('close', release);
    body.once('error', release);
    return { status: range ? 206 : 200, headers, body };
  }
}

function parseRange(value: string, size: number): { start: number; end: number } | undefined {
  const match = /^bytes=(\d*)-(\d*)$/u.exec(value.trim());
  if (!match || size <= 0) return undefined;
  const [, startValue = '', endValue = ''] = match;
  if (!startValue && !endValue) return undefined;
  if (!startValue) {
    const suffix = Number(endValue);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return undefined;
    return { start: Math.max(0, size - suffix), end: size - 1 };
  }
  const start = Number(startValue);
  const requestedEnd = endValue ? Number(endValue) : size - 1;
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(requestedEnd) ||
    start < 0 ||
    start >= size ||
    requestedEnd < start
  ) {
    return undefined;
  }
  return { start, end: Math.min(requestedEnd, size - 1) };
}

function errorResponse(status: number): ArtifactProtocolResponse {
  return {
    status,
    headers: {
      'Content-Length': '0',
      'Cache-Control': 'private, no-store',
      'X-Content-Type-Options': 'nosniff',
    },
  };
}

function headerFilename(value?: string): string {
  return (value ?? 'artifact.bin').replaceAll(/["\\\r\n]/gu, '_').slice(0, 255);
}
