import { realpath } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import {
  BROWSER_IMPORT_LIMITS,
  type BrowserImportData,
  type BrowserImportWorkerReply,
  type BrowserImportWorkerRequest,
} from './browser-import-types';

export interface BrowserImportWorkerOptions {
  workerPath: string;
  nativeSandboxPath: string;
  signal?: AbortSignal;
}

/** Utility-process transport is isolation from Browser main, not itself an OS security sandbox. */
export async function runBrowserImportWorker(
  request: BrowserImportWorkerRequest,
  options: BrowserImportWorkerOptions,
): Promise<BrowserImportData> {
  if (process.platform !== 'darwin') throw new Error('IMPORT_SANDBOX_UNAVAILABLE');
  const nativeSandboxPath = await realpath(options.nativeSandboxPath).catch(() => {
    throw new Error('IMPORT_SANDBOX_UNAVAILABLE');
  });
  const directory = request.kind === 'safari' ? '/' : await realpath(request.directory);
  const { app, session, utilityProcess } = await import('electron');
  if (!app.isReady()) throw new Error('IMPORT_SANDBOX_UNAVAILABLE');
  const isolatedSession = session.fromPartition(`oryn-import-${randomUUID()}`);
  isolatedSession.webRequest.onBeforeRequest((_details, callback) => callback({ cancel: true }));
  const child = utilityProcess.fork(options.workerPath, [], {
    cwd: directory,
    env: { LANG: 'en_US.UTF-8', LC_ALL: 'en_US.UTF-8', TMPDIR: directory },
    // Electron's Node utility ignores Node/V8 heap flags (qualified on Electron 45).
    // The measured RSS watchdog below is the memory boundary; native confine limits CPU/files.
    execArgv: [],
    session: isolatedSession,
    stdio: 'ignore',
    serviceName: 'Oryn browser data transfer',
    allowLoadingUnsignedLibraries: false,
    respondToAuthRequestsFromMainProcess: false,
  });
  return await new Promise<BrowserImportData>((resolve, reject) => {
    let reply: BrowserImportWorkerReply | undefined;
    let error: Error | undefined;
    let settled = false;
    const started = Date.now();
    const stop = (code: string) => {
      error ??= new Error(code);
      if (child.pid) {
        try {
          process.kill(child.pid, 'SIGKILL');
        } catch {
          child.kill();
        }
      } else child.kill();
    };
    const abort = () => stop('IMPORT_CANCELLED');
    const timeout = setTimeout(() => stop('IMPORT_TIMEOUT'), BROWSER_IMPORT_LIMITS.workerTimeoutMs);
    const memory = setInterval(() => {
      const pid = child.pid;
      if (!pid) return;
      const metric = app.getAppMetrics().find((item) => item.pid === pid);
      const measured = metric?.memory?.workingSetSize;
      if (
        (typeof measured !== 'number' || !Number.isFinite(measured) || measured <= 0) &&
        Date.now() - started > 2000
      ) {
        stop('IMPORT_MEMORY_UNAVAILABLE');
        return;
      }
      const bytes = (measured ?? 0) * 1024;
      if (bytes > BROWSER_IMPORT_LIMITS.workerMemoryBytes) stop('IMPORT_MEMORY_LIMIT');
    }, 100);
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      clearInterval(memory);
      options.signal?.removeEventListener('abort', abort);
      void isolatedSession.closeAllConnections();
      if (error) reject(error);
      else if (reply?.ok) resolve(reply.data);
      else
        reject(
          new Error(
            reply && !reply.ok && /^IMPORT_[A-Z_]+$/.test(reply.code)
              ? reply.code
              : 'IMPORT_WORKER_FAILED',
          ),
        );
    };
    child.on('message', (message: unknown) => {
      if (
        reply ||
        !message ||
        typeof message !== 'object' ||
        (message as BrowserImportWorkerReply).version !== 2
      ) {
        stop('IMPORT_WORKER_INVALID');
        return;
      }
      reply = message as BrowserImportWorkerReply;
    });
    child.on('error', () => stop('IMPORT_WORKER_FAILED'));
    child.once('spawn', () => {
      if (error) stop(error.message);
    });
    child.once('exit', (code) => {
      if (code !== 0) error ??= new Error('IMPORT_WORKER_FAILED');
      finish();
    });
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) abort();
    else
      child.postMessage({
        request: { ...request, directory: request.kind === 'safari' ? '' : directory },
        nativeSandboxPath,
      });
  });
}
