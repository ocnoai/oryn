import {
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
  openSync,
  fsyncSync,
  closeSync,
} from 'node:fs';
import { dirname } from 'node:path';

export const recoveredStores = new Set<string>();

export class JsonFileStore<T> {
  private memoryValue?: T;
  constructor(
    readonly filePath: string,
    private readonly createDefault: () => T,
    private readonly validate: (value: unknown) => value is T,
  ) {}

  read(): T {
    if (this.filePath.startsWith('memory:'))
      return structuredClone(this.memoryValue ?? this.createDefault());
    try {
      const value = JSON.parse(readFileSync(this.filePath, 'utf8')) as unknown;
      if (!this.validate(value)) {
        throw new Error(`Invalid persisted data in ${this.filePath}`);
      }
      return value;
    } catch (error) {
      if (isMissingFileError(error)) {
        return structuredClone(this.createDefault());
      }
      // Preserve corrupt bytes for diagnosis; defaults never restore grants or credentials.
      if (
        !(error instanceof SyntaxError) &&
        !(error instanceof Error && error.message.startsWith('Invalid persisted data'))
      )
        throw error;
      renameSync(this.filePath, `${this.filePath}.corrupt-${Date.now()}`);
      recoveredStores.add(this.filePath.split('/').at(-1) ?? 'browser data');
      const fallback = structuredClone(this.createDefault());
      this.write(fallback);
      return fallback;
    }
  }

  write(value: T): void {
    if (!this.validate(value)) {
      throw new Error(`Refusing to persist invalid data to ${this.filePath}`);
    }
    if (this.filePath.startsWith('memory:')) {
      this.memoryValue = structuredClone(value);
      return;
    }
    mkdirSync(dirname(this.filePath), { recursive: true });
    const temporaryPath = `${this.filePath}.tmp`;
    writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    });
    const descriptor = openSync(temporaryPath, 'r');
    try {
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    renameSync(temporaryPath, this.filePath);
    if (process.platform !== 'win32') {
      const directory = openSync(dirname(this.filePath), 'r');
      try {
        fsyncSync(directory);
      } finally {
        closeSync(directory);
      }
    }
  }

  update(mutator: (current: T) => T): T {
    const next = mutator(this.read());
    this.write(next);
    return next;
  }
}

function isMissingFileError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT';
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
