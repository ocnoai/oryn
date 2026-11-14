import { createRequire } from 'node:module';
import { basename, dirname, isAbsolute, resolve } from 'node:path';
import { parseChromiumImport } from './browser-import-chromium';
import { parseSafariImport } from './browser-import-safari';
import { isSafariArchiveIdentity } from './browser-import-safari-source';
import {
  BROWSER_IMPORT_LIMITS,
  type BrowserImportWorkerReply,
  type BrowserImportWorkerRequest,
} from './browser-import-types';

function isRequest(value: unknown): value is BrowserImportWorkerRequest {
  if (!value || typeof value !== 'object') return false;
  const request = value as BrowserImportWorkerRequest;
  return (
    request.version === 2 &&
    ['chromium', 'safari'].includes(request.kind) &&
    typeof request.directory === 'string' &&
    request.directory.length < 4096 &&
    (request.kind === 'safari'
      ? request.directory === '' && isSafariArchiveIdentity(request.safariArchive)
      : !request.safariArchive &&
        isAbsolute(request.directory) &&
        /^oryn-browser-import-[A-Za-z0-9]{6}$/u.test(basename(request.directory))) &&
    typeof request.sourceKey === 'string' &&
    /^[a-f0-9]{64}$/.test(request.sourceKey) &&
    typeof request.sourceLabel === 'string' &&
    request.sourceLabel.length <= 256 &&
    Number.isInteger(request.bookmarkCapacity) &&
    request.bookmarkCapacity >= 0 &&
    request.bookmarkCapacity <= BROWSER_IMPORT_LIMITS.bookmarkNodes &&
    Array.isArray(request.existingBookmarkIds) &&
    request.existingBookmarkIds.length <= BROWSER_IMPORT_LIMITS.bookmarkNodes &&
    request.existingBookmarkIds.every(
      (value) => typeof value === 'string' && value.length <= 128,
    ) &&
    Array.isArray(request.dataTypes) &&
    request.dataTypes.length > 0 &&
    request.dataTypes.length <= 3 &&
    request.dataTypes.every((type) => ['bookmarks', 'history', 'searchEngine'].includes(type))
  );
}

export function importSandboxProfile(directory?: string): string {
  if (directory === undefined)
    return `(version 1)\n(deny default)\n(allow file-write* (literal "/dev/null"))\n(allow process-info* (target self))\n(allow sysctl-read)\n(allow system-sched)\n(deny network*)\n`;
  const path = resolve(directory);
  if (path.includes('\0') || /[\r\n]/.test(path)) throw new Error('IMPORT_SANDBOX_UNAVAILABLE');
  const quoted = JSON.stringify(path);
  const ancestors: string[] = [];
  let parent = dirname(path);
  while (true) {
    ancestors.push(`(literal ${JSON.stringify(parent)})`);
    if (parent === '/') break;
    parent = dirname(parent);
  }
  return `(version 1)\n(deny default)\n(allow file-read* (subpath ${quoted}))\n(allow file-read-metadata ${ancestors.join(' ')})\n(allow file-write* (literal "/dev/null"))\n(allow process-info* (target self))\n(allow sysctl-read)\n(allow system-sched)\n(deny network*)\n`;
}

const port = process.parentPort;
if (port) {
  let used = false;
  port.once('message', async (event) => {
    if (used) return;
    used = true;
    let reply: BrowserImportWorkerReply;
    try {
      const envelope = event.data as { request?: unknown; nativeSandboxPath?: unknown };
      if (
        !envelope ||
        !isRequest(envelope.request) ||
        typeof envelope.nativeSandboxPath !== 'string' ||
        process.platform !== 'darwin'
      )
        throw new Error('IMPORT_SANDBOX_UNAVAILABLE');
      // All reviewed JS/native imports above have initialized before filesystem/network authority drops.
      const require = createRequire(import.meta.url);
      const sandbox = require(envelope.nativeSandboxPath) as {
        protocolVersion?: number;
        confine?: (profile: string, cpu: number, fileBytes: number, exactFile?: string) => boolean;
      };
      if (
        sandbox.protocolVersion !== 2 ||
        typeof sandbox.confine !== 'function' ||
        !sandbox.confine(
          importSandboxProfile(
            envelope.request.kind === 'safari' ? undefined : envelope.request.directory,
          ),
          10,
          0,
          envelope.request.safariArchive?.path,
        )
      )
        throw new Error('IMPORT_SANDBOX_UNAVAILABLE');
      const data =
        envelope.request.kind === 'chromium'
          ? await parseChromiumImport(envelope.request)
          : await parseSafariImport(envelope.request);
      reply = { version: 2, ok: true, data };
    } catch (error) {
      const code =
        error instanceof Error && /^IMPORT_[A-Z_]+$/.test(error.message)
          ? error.message
          : 'IMPORT_PARSE_FAILED';
      reply = { version: 2, ok: false, code };
    }
    port.postMessage(reply);
    // Main waits for exit before deleting copied inputs or accepting worker output.
    setImmediate(() => process.exit(0));
  });
}
