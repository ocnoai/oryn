import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { closeSync, constants, openSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { win32 } from 'node:path';

import { ArtifactError } from './types';

export interface ArtifactIoRuntime {
  pythonExecutable: string;
  helperPath: string;
}

/** Platform helpers retain root handles; neither backend copies by an Engine pathname. */
export class ArtifactFileService {
  private readonly descriptors: number[] = [];
  private windowsHandles: number[] = [];
  private queue: Promise<void> = Promise.resolve();
  private closed = false;
  private activeChild?: ChildProcess;

  constructor(
    staging: string,
    temporary: string,
    published: string,
    private readonly runtime: ArtifactIoRuntime = {
      // Source/test default. Packaged Browser always supplies its bundled runtime.
      pythonExecutable:
        process.platform === 'win32' ? (process.env.ORYN_ENGINE_PYTHON ?? 'python') : 'python3',
      helperPath: fileURLToPath(
        new URL(
          `../../../build/helpers/${process.platform === 'win32' ? 'artifact_io_windows.py' : 'artifact_io.py'}`,
          import.meta.url,
        ),
      ),
    },
  ) {
    if (process.platform === 'win32') {
      if (!win32.isAbsolute(this.runtime.pythonExecutable)) {
        throw new Error(
          'Windows artifact I/O requires an absolute bundled or configured Python path.',
        );
      }
      // Python opens the roots before Engine startup and duplicates non-inheritable
      // native handles into Browser. Later helpers duplicate those handles back;
      // no Engine-controlled path is reopened to locate a root.
      const result = this.runSync({
        operation: 'initialize',
        roots: [staging, temporary, published],
      });
      if (result.error || result.status !== 0) throw helperError(result.stdout ?? '');
      const handles: unknown = JSON.parse(result.stdout);
      if (
        !Array.isArray(handles) ||
        handles.length !== 3 ||
        !handles.every((handle) => Number.isSafeInteger(handle) && handle > 0)
      ) {
        throw new ArtifactError('ARTIFACT_IO_ERROR');
      }
      this.windowsHandles = handles as number[];
      return;
    }
    if (!constants.O_NOFOLLOW || !constants.O_DIRECTORY) {
      throw new ArtifactError('ARTIFACT_IO_ERROR');
    }
    try {
      for (const path of [staging, temporary, published]) {
        this.descriptors.push(
          openSync(path, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW),
        );
      }
    } catch (error) {
      this.close();
      throw error;
    }
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.activeChild?.kill('SIGKILL');
    for (const descriptor of this.descriptors.splice(0)) closeSync(descriptor);
    if (this.windowsHandles.length) {
      // Active helpers own duplicates; killing/draining them remains authoritative.
      // Browser process exit also closes its handles if this bounded release fails.
      this.runSync({ operation: 'close' });
      this.windowsHandles = [];
    }
  }

  async shutdown(): Promise<void> {
    this.close();
    await this.queue;
  }

  /** Startup only, before Engine launch. Work and wall time are bounded. */
  cleanupAtStartup(): boolean {
    this.assertOpen();
    const result = this.runSync({ operation: 'cleanup', all: true });
    if (result.error || result.status !== 0) {
      console.error(
        'Artifact staging reconciliation incomplete; stale entries remain for a later retry.',
      );
      return false;
    }
    return true;
  }

  async run(
    request: { operation: 'copy' | 'cleanup'; [key: string]: unknown },
    signal?: AbortSignal,
  ): Promise<void> {
    const previous = this.queue;
    let release!: () => void;
    this.queue = new Promise<void>((resolve) => {
      release = resolve;
    });
    try {
      await previous;
      this.assertOpen();
      if (signal?.aborted) throw new ArtifactError('ARTIFACT_ACCESS_DENIED');
      const input = this.input(request);
      if (Buffer.byteLength(input) > 256 * 1024) throw new ArtifactError('ARTIFACT_INVALID');
      await new Promise<void>((resolve, reject) => {
        const child = spawn(this.runtime.pythonExecutable, this.arguments(), {
          env: this.environment(),
          cwd: this.workingDirectory(),
          windowsHide: true,
          stdio: ['pipe', 'pipe', 'ignore', ...this.descriptors],
        });
        this.activeChild = child;
        let output = '';
        let failed = false;
        const stop = () => {
          failed = true;
          child.kill('SIGKILL');
        };
        const timeout = setTimeout(stop, request.operation === 'copy' ? 15 * 60_000 : 10_000);
        signal?.addEventListener('abort', stop, { once: true });
        child.stdout!.on('data', (chunk: Buffer) => {
          if (output.length + chunk.length > 4096) stop();
          else output += chunk.toString('utf8');
        });
        // Wait for close (not just error/exit) before rolling back private files.
        child.on('error', () => {
          failed = true;
        });
        child.stdin!.on('error', () => {
          failed = true;
        });
        child.on('close', (code) => {
          if (this.activeChild === child) this.activeChild = undefined;
          clearTimeout(timeout);
          signal?.removeEventListener('abort', stop);
          if (signal?.aborted) reject(new ArtifactError('ARTIFACT_ACCESS_DENIED'));
          else if (failed || code !== 0) reject(helperError(output));
          else resolve();
        });
        child.stdin!.end(input);
        if (signal?.aborted) stop();
      });
    } finally {
      release();
    }
  }

  private arguments(): string[] {
    return ['-I', '-S', '-B', this.runtime.helperPath];
  }

  private input(request: Record<string, unknown>): string {
    return JSON.stringify({
      ...request,
      parentPid: process.pid,
      ...(process.platform === 'win32' ? { handles: this.windowsHandles } : {}),
    });
  }

  private environment(): NodeJS.ProcessEnv {
    // Windows needs its OS directory to load system DLLs. No Python settings,
    // user PATH, credentials or Engine environment cross this boundary.
    return process.platform === 'win32'
      ? { SystemRoot: process.env.SystemRoot, PATH: '' }
      : { PATH: '/usr/local/bin:/usr/bin:/bin' };
  }

  private workingDirectory(): string {
    return process.platform === 'win32' ? win32.dirname(this.runtime.helperPath) : '/';
  }

  private runSync(request: Record<string, unknown>) {
    return spawnSync(this.runtime.pythonExecutable, this.arguments(), {
      env: this.environment(),
      cwd: this.workingDirectory(),
      windowsHide: true,
      input: this.input(request),
      stdio: ['pipe', 'pipe', 'ignore', ...this.descriptors],
      encoding: 'utf8',
      timeout: 10_000,
      killSignal: 'SIGKILL',
      maxBuffer: 4096,
    });
  }

  private assertOpen(): void {
    if (this.closed) throw new ArtifactError('ARTIFACT_IO_ERROR');
  }
}

function helperError(output: string): ArtifactError {
  const code = output.trim();
  return new ArtifactError(
    [
      'ARTIFACT_INVALID',
      'ARTIFACT_PATH_INVALID',
      'RESOURCE_LIMIT_EXCEEDED',
      'DISK_SPACE_INSUFFICIENT',
    ].includes(code)
      ? (code as ConstructorParameters<typeof ArtifactError>[0])
      : 'ARTIFACT_IO_ERROR',
  );
}
