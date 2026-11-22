import { Buffer } from 'node:buffer';
import { randomUUID } from 'node:crypto';
import {
  closeSync,
  constants,
  existsSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readSync,
  realpathSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join } from 'node:path';

const validID = (value: unknown): value is string =>
  typeof value === 'string' && /^[a-f0-9]{32}$/u.test(value);
const MAX_NAMESPACES = 128;
const MAX_JOURNAL_BYTES = 16384;

/** Contains no browsing data. Register before disk writes; acknowledge only after deletion. */
export class PrivateCleanupQueue {
  readonly journalPath: string;
  readonly temporaryRoot: string;
  private readonly profileRoot: string;
  private readonly active = new Set<string>();
  private failed = 0;

  constructor(
    profileRoot: string,
    private readonly operations: { remove?: (path: string) => void } = {},
  ) {
    mkdirSync(profileRoot, { recursive: true, mode: 0o700 });
    this.profileRoot = realpathSync(profileRoot);
    this.journalPath = join(this.profileRoot, 'browser-core', 'private-cleanup.json');
    this.temporaryRoot = join(this.profileRoot, 'private-temp');
  }

  get pendingCount(): number {
    return this.read().length;
  }
  get status(): { pending: number; failed: number } {
    return { pending: this.pendingCount, failed: this.failed };
  }

  register(): { id: string; directory: string } {
    this.assertRoot();
    const ids = this.read();
    if (ids.length >= MAX_NAMESPACES) throw new Error('PRIVATE_CLEANUP_QUEUE_FULL');
    const id = randomUUID().replaceAll('-', '');
    this.persist([...ids, id]);
    const directory = join(this.temporaryRoot, id);
    mkdirSync(this.temporaryRoot, { recursive: true, mode: 0o700 });
    this.assertRoot();
    mkdirSync(directory, { mode: 0o700 });
    this.active.add(id);
    return { id, directory };
  }

  cleanup(id: string): boolean {
    if (!validID(id)) throw new Error('PRIVATE_CLEANUP_ID_INVALID');
    this.active.delete(id);
    return this.remove(id);
  }

  recover(): { pending: number; failed: number } {
    this.failed = 0;
    for (const id of this.read()) if (!this.active.has(id) && !this.remove(id)) this.failed++;
    return this.status;
  }

  private remove(id: string): boolean {
    const ids = this.read();
    if (!ids.includes(id)) return true;
    try {
      this.assertRoot();
      const target = join(this.temporaryRoot, id);
      let present = false;
      try {
        const stat = lstatSync(target);
        if (!stat.isDirectory() || stat.isSymbolicLink())
          throw new Error('PRIVATE_CLEANUP_UNSAFE_TARGET');
        present = true;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
      if (present) {
        if (this.operations.remove) this.operations.remove(target);
        else rmSync(target, { recursive: true, force: true });
      }
      // Persist the directory removal before acknowledging it in another directory.
      // Otherwise a crash could recover the namespace after its retry record vanished.
      this.assertRoot();
      if (existsSync(this.temporaryRoot)) this.flushDirectory(this.temporaryRoot);
      this.persist(ids.filter((value) => value !== id));
      return true;
    } catch {
      return false;
    }
  }

  private assertManagedParents(): void {
    for (const path of [this.profileRoot, dirname(this.journalPath)]) {
      try {
        const stat = lstatSync(path);
        if (!stat.isDirectory() || stat.isSymbolicLink())
          throw new Error('PRIVATE_CLEANUP_UNSAFE_PARENT');
      } catch (error) {
        if (
          path === dirname(this.journalPath) &&
          (error as NodeJS.ErrnoException).code === 'ENOENT'
        )
          continue;
        throw error;
      }
    }
  }

  private assertRoot(): void {
    this.assertManagedParents();
    try {
      const stat = lstatSync(this.temporaryRoot);
      if (!stat.isDirectory() || stat.isSymbolicLink())
        throw new Error('PRIVATE_CLEANUP_UNSAFE_ROOT');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }

  private read(): string[] {
    this.assertManagedParents();
    let descriptor: number;
    try {
      descriptor = openSync(
        this.journalPath,
        constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
      throw new Error('PRIVATE_CLEANUP_JOURNAL_INVALID');
    }
    try {
      const stat = fstatSync(descriptor);
      if (!stat.isFile() || stat.size > MAX_JOURNAL_BYTES) throw new Error();
      const bytes = Buffer.alloc(MAX_JOURNAL_BYTES + 1);
      const count = readSync(descriptor, bytes, 0, bytes.length, 0);
      if (count > MAX_JOURNAL_BYTES) throw new Error();
      const value = JSON.parse(bytes.toString('utf8', 0, count)) as {
        version?: unknown;
        ids?: unknown;
      };
      if (
        value.version !== 1 ||
        !Array.isArray(value.ids) ||
        value.ids.length > MAX_NAMESPACES ||
        !value.ids.every(validID) ||
        new Set(value.ids).size !== value.ids.length
      )
        throw new Error();
      return value.ids;
    } catch {
      throw new Error('PRIVATE_CLEANUP_JOURNAL_INVALID');
    } finally {
      closeSync(descriptor);
    }
  }

  private flushDirectory(path: string): void {
    const descriptor = openSync(
      path,
      constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
    );
    try {
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
  }

  private persist(ids: string[]): void {
    this.assertManagedParents();
    mkdirSync(dirname(this.journalPath), { recursive: true, mode: 0o700 });
    this.assertManagedParents();
    this.flushDirectory(this.profileRoot);
    const temporary = join(dirname(this.journalPath), `private-cleanup.${randomUUID()}.tmp`);
    const descriptor = openSync(
      temporary,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o600,
    );
    try {
      try {
        writeFileSync(descriptor, `${JSON.stringify({ version: 1, ids })}\n`);
        fsyncSync(descriptor);
      } finally {
        closeSync(descriptor);
      }
      this.assertManagedParents();
      renameSync(temporary, this.journalPath);
      this.flushDirectory(dirname(this.journalPath));
    } finally {
      rmSync(temporary, { force: true });
    }
  }
}
