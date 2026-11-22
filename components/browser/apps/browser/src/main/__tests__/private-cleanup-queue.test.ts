import {
  mkdtempSync,
  readFileSync,
  writeFileSync,
  existsSync,
  mkdirSync,
  symlinkSync,
  rmSync,
  linkSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { ModuleKind, ScriptTarget, transpileModule } from 'typescript';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { PrivateCleanupQueue } from '../data/private-cleanup-queue';

const roots: string[] = [];
const root = () => {
  const value = mkdtempSync(join(tmpdir(), 'oryn-private-cleanup-test-'));
  roots.push(value);
  return value;
};
afterEach(() => {
  vi.restoreAllMocks();
  for (const value of roots.splice(0)) rmSync(value, { recursive: true, force: true });
});

describe('private cleanup journal', () => {
  it('registers a content-free namespace durably before returning a writable directory', () => {
    const queue = new PrivateCleanupQueue(root());
    const entry = queue.register();
    expect(existsSync(entry.directory)).toBe(true);
    const journal = readFileSync(queue.journalPath, 'utf8');
    expect(journal).toContain(entry.id);
    writeFileSync(join(entry.directory, 'temporary'), 'private-browsing-token');
    expect(journal).not.toContain('private-browsing-token');
    expect(journal).not.toContain(entry.directory);
    expect(queue.cleanup(entry.id)).toBe(true);
    expect(existsSync(entry.directory)).toBe(false);
    expect(queue.pendingCount).toBe(0);
  });

  it('recovers crash residue on next startup without deleting explicit saves or active windows', () => {
    const profile = root();
    const queue = new PrivateCleanupQueue(profile);
    const first = queue.register();
    const second = queue.register();
    const saved = join(profile, 'user-saved.txt');
    writeFileSync(saved, 'explicit-save');
    writeFileSync(join(first.directory, 'temporary'), 'old-session');
    queue.cleanup(first.id);
    expect(existsSync(second.directory)).toBe(true);
    expect(queue.recover().pending).toBe(1);
    const nextStartup = new PrivateCleanupQueue(profile);
    expect(nextStartup.recover()).toEqual({ pending: 0, failed: 0 });
    expect(existsSync(second.directory)).toBe(false);
    expect(readFileSync(saved, 'utf8')).toBe('explicit-save');
    expect(nextStartup.recover()).toEqual({ pending: 0, failed: 0 });
  });

  it('retains failed work for a later startup and treats an already removed target as success', () => {
    const profile = root();
    const queue = new PrivateCleanupQueue(profile, {
      remove: () => {
        throw new Error('locked');
      },
    });
    const entry = queue.register();
    expect(queue.cleanup(entry.id)).toBe(false);
    expect(queue.pendingCount).toBe(1);
    rmSync(entry.directory, { recursive: true });
    expect(new PrivateCleanupQueue(profile).recover()).toEqual({ pending: 0, failed: 0 });
  });

  it('does not create a writable namespace when durable registration fails', () => {
    const profile = root();
    const queue = new PrivateCleanupQueue(profile);
    vi.spyOn(queue as unknown as { persist(ids: string[]): void }, 'persist').mockImplementation(
      () => {
        throw new Error('simulated journal write failure');
      },
    );
    expect(() => queue.register()).toThrow('simulated journal write failure');
    expect(existsSync(queue.temporaryRoot)).toBe(false);
    expect(existsSync(queue.journalPath)).toBe(false);
  });

  it.each(['during-deletion', 'after-deletion-before-acknowledgement'])(
    'recovers after a real child process is killed %s',
    (phase) => {
      const scratch = root();
      const profile = join(scratch, 'profile');
      const source = readFileSync(
        join(import.meta.dirname, '../data/private-cleanup-queue.ts'),
        'utf8',
      );
      writeFileSync(
        join(scratch, 'queue.cjs'),
        transpileModule(source, {
          compilerOptions: { module: ModuleKind.CommonJS, target: ScriptTarget.ES2022 },
        }).outputText,
      );
      writeFileSync(
        join(scratch, 'crash.cjs'),
        `
        const fs = require('node:fs');
        const {join} = require('node:path');
        const {PrivateCleanupQueue} = require('./queue.cjs');
        const [profile, phase] = process.argv.slice(2);
        const queue = new PrivateCleanupQueue(profile, {remove(target) {
          fs.rmSync(join(target, 'first'));
          if (phase === 'during-deletion') process.kill(process.pid, 'SIGKILL');
          fs.rmSync(target, {recursive:true});
        }});
        const entry = queue.register();
        fs.writeFileSync(join(profile, 'explicit-save'), 'keep-user-save');
        fs.writeFileSync(join(entry.directory, 'first'), 'private-first');
        fs.writeFileSync(join(entry.directory, 'second'), 'private-second');
        if (phase === 'after-deletion-before-acknowledgement')
          queue.persist = () => process.kill(process.pid, 'SIGKILL');
        queue.cleanup(entry.id);
        process.exit(2);
      `,
      );
      const child = spawnSync(process.execPath, [join(scratch, 'crash.cjs'), profile, phase], {
        timeout: 5000,
        encoding: 'utf8',
      });
      expect(child.error).toBeUndefined();
      expect(child.signal).toBe('SIGKILL');
      const queue = new PrivateCleanupQueue(profile);
      expect(queue.pendingCount).toBe(1);
      const ids = JSON.parse(readFileSync(queue.journalPath, 'utf8')).ids as string[];
      const directory = join(queue.temporaryRoot, ids[0]!);
      expect(existsSync(directory)).toBe(phase === 'during-deletion');
      if (phase === 'during-deletion')
        expect(readFileSync(join(directory, 'second'), 'utf8')).toBe('private-second');
      expect(queue.recover()).toEqual({ pending: 0, failed: 0 });
      expect(existsSync(directory)).toBe(false);
      expect(queue.recover()).toEqual({ pending: 0, failed: 0 });
      expect(readFileSync(join(profile, 'explicit-save'), 'utf8')).toBe('keep-user-save');
    },
  );

  it('rejects malformed journal targets without deleting unrelated data', () => {
    const profile = root();
    const queue = new PrivateCleanupQueue(profile);
    const entry = queue.register();
    writeFileSync(queue.journalPath, JSON.stringify({ version: 1, ids: ['../../other-data'] }));
    expect(() => new PrivateCleanupQueue(profile).recover()).toThrow(
      'PRIVATE_CLEANUP_JOURNAL_INVALID',
    );
    expect(existsSync(entry.directory)).toBe(true);
  });

  it('does not follow a namespace or private-root symlink', () => {
    const profile = root();
    const external = root();
    writeFileSync(join(external, 'keep'), 'normal-data');
    const queue = new PrivateCleanupQueue(profile);
    const entry = queue.register();
    rmSync(entry.directory, { recursive: true });
    symlinkSync(external, entry.directory, 'dir');
    expect(queue.cleanup(entry.id)).toBe(false);
    expect(readFileSync(join(external, 'keep'), 'utf8')).toBe('normal-data');
    const another = root();
    mkdirSync(join(another, 'browser-core'), { recursive: true });
    symlinkSync(external, join(another, 'private-temp'), 'dir');
    expect(() => new PrivateCleanupQueue(another).register()).toThrow(
      'PRIVATE_CLEANUP_UNSAFE_ROOT',
    );
  });

  it('never truncates a pre-existing temporary-file hardlink to an explicit save', () => {
    const profile = root();
    const queue = new PrivateCleanupQueue(profile);
    mkdirSync(join(profile, 'browser-core'));
    const saved = join(profile, 'explicit-save.txt');
    writeFileSync(saved, 'user-saved-content');
    linkSync(saved, `${queue.journalPath}.tmp`);
    const entry = queue.register();
    expect(readFileSync(saved, 'utf8')).toBe('user-saved-content');
    expect(queue.cleanup(entry.id)).toBe(true);
    expect(readFileSync(saved, 'utf8')).toBe('user-saved-content');
  });

  it('rejects a browser-core parent symlink before reading or writing an external journal', () => {
    const profile = root();
    const external = root();
    const externalJournal = join(external, 'private-cleanup.json');
    const sentinel = JSON.stringify({ version: 1, ids: [] });
    writeFileSync(externalJournal, sentinel);
    symlinkSync(external, join(profile, 'browser-core'), 'dir');
    const queue = new PrivateCleanupQueue(profile);
    expect(() => queue.register()).toThrow('PRIVATE_CLEANUP_UNSAFE_PARENT');
    expect(() => queue.recover()).toThrow('PRIVATE_CLEANUP_UNSAFE_PARENT');
    expect(readFileSync(externalJournal, 'utf8')).toBe(sentinel);
    expect(existsSync(join(profile, 'private-temp'))).toBe(false);
  });

  it('rejects oversized or non-file journals and retains dangling namespace symlinks for retry', () => {
    const profile = root();
    const queue = new PrivateCleanupQueue(profile);
    const entry = queue.register();
    const original = readFileSync(queue.journalPath, 'utf8');
    writeFileSync(queue.journalPath, ' '.repeat(16385));
    expect(() => queue.recover()).toThrow('PRIVATE_CLEANUP_JOURNAL_INVALID');
    rmSync(queue.journalPath);
    mkdirSync(queue.journalPath);
    expect(() => queue.recover()).toThrow('PRIVATE_CLEANUP_JOURNAL_INVALID');
    rmSync(queue.journalPath, { recursive: true });
    writeFileSync(queue.journalPath, original);
    rmSync(entry.directory, { recursive: true });
    symlinkSync(join(profile, 'missing'), entry.directory, 'dir');
    expect(queue.cleanup(entry.id)).toBe(false);
    expect(queue.pendingCount).toBe(1);
  });
});
