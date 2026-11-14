import { execFileSync } from 'node:child_process';
import {
  constants,
  openSync,
  closeSync,
  mkdtempSync,
  writeFileSync,
  readFileSync,
  renameSync,
  rmSync,
  linkSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { DownloadSafetyNative } from '../download-safety';

const require = createRequire(import.meta.url);
let root: string;
let native: DownloadSafetyNative;
let reader: string;
// The product target is macOS; other platforms must qualify a separate quarantine implementation.
describe.runIf(process.platform === 'darwin')('native macOS download safety', () => {
  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'oryn-download-native-'));
    const include = join(dirname(require.resolve('node-api-headers/package.json')), 'include');
    const source = fileURLToPath(
      new URL('../../../build/helpers/download_safety.mm', import.meta.url),
    );
    const helper = join(root, 'download_safety.node');
    execFileSync(
      '/usr/bin/clang++',
      [
        '-bundle',
        '-undefined',
        'dynamic_lookup',
        '-I',
        include,
        '-framework',
        'Foundation',
        source,
        '-o',
        helper,
      ],
      { timeout: 30_000, stdio: 'pipe' },
    );
    native = require(helper) as DownloadSafetyNative;
    reader = join(root, 'read-reference');
    const readerSource = join(root, 'read-reference.mm');
    writeFileSync(
      readerSource,
      `#import <Foundation/Foundation.h>
      int main(int argc, char **argv) { @autoreleasepool {
        if (argc != 2) return 1;
        NSURL *url = [NSURL URLWithString:[NSString stringWithUTF8String:argv[1]]];
        NSData *data = [NSData dataWithContentsOfURL:url];
        if (!data || [data length] > 512) return 1;
        fwrite([data bytes], 1, [data length], stdout); return 0;
      } }`,
    );
    execFileSync('/usr/bin/clang++', ['-framework', 'Foundation', readerSource, '-o', reader], {
      timeout: 30_000,
      stdio: 'pipe',
    });
  });
  afterAll(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });
  function held(name: string, contents = 'harmless native fixture') {
    const path = join(root, name);
    writeFileSync(path, contents);
    return { path, fd: openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW) };
  }
  function attribute(path: string) {
    return execFileSync('/usr/bin/xattr', ['-p', 'com.apple.quarantine', path], {
      encoding: 'utf8',
    }).trim();
  }
  it('adds quarantine only when missing and preserves the complete native attribute', () => {
    const value = held('fallback.txt');
    try {
      expect(native.hasQuarantine(value.fd)).toBe(false);
      expect(native.ensureQuarantine(value.fd)).toBe('added');
      expect(attribute(value.path)).toMatch(/^0001;[0-9a-f]+;Oryn Browser;$/u);
      expect(native.hasQuarantine(value.fd)).toBe(true);
      const original = '0083;12345678;Native browser;native-existing-identity';
      execFileSync('/usr/bin/xattr', ['-w', 'com.apple.quarantine', original, value.path]);
      expect(native.ensureQuarantine(value.fd)).toBe('preserved');
      expect(attribute(value.path)).toBe(original);
    } finally {
      closeSync(value.fd);
    }
  });
  it('persists the quarantine tag after all held handles close and the file is reopened', () => {
    const value = held('restart.txt');
    native.ensureQuarantine(value.fd);
    closeSync(value.fd);
    const fd = openSync(value.path, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      expect(native.hasQuarantine(fd)).toBe(true);
    } finally {
      closeSync(fd);
    }
  });
  it('refuses malformed/non-quarantine attributes without overwriting them', () => {
    const value = held('invalid-attribute.txt');
    try {
      for (const attribute of [' +1;12345678;invalid;', '0000;12345678;no quarantine;']) {
        execFileSync('/usr/bin/xattr', ['-w', 'com.apple.quarantine', attribute, value.path]);
        expect(native.hasQuarantine(value.fd)).toBe(false);
        expect(() => native.ensureQuarantine(value.fd)).toThrow('DOWNLOAD_QUARANTINE_UNAVAILABLE');
        expect(
          execFileSync('/usr/bin/xattr', ['-p', 'com.apple.quarantine', value.path], {
            encoding: 'utf8',
          }).trim(),
        ).toBe(attribute.trim());
      }
    } finally {
      closeSync(value.fd);
    }
  });
  it('the native file-reference URL still reads the held bytes after pathname replacement', () => {
    const value = held('reference.command', 'held harmless bytes');
    try {
      native.ensureQuarantine(value.fd);
      const url = native.fileReferenceURL(value.fd);
      expect(url).toMatch(/^file:\/\/\/\.file\/id=\d+\.\d+$/u);
      renameSync(value.path, join(root, 'held-original.command'));
      writeFileSync(value.path, 'replacement must never be opened');
      expect(readFileSync(value.path, 'utf8')).toBe('replacement must never be opened');
      expect(execFileSync(reader, [url], { encoding: 'utf8', timeout: 5_000 })).toBe(
        'held harmless bytes',
      );
    } finally {
      closeSync(value.fd);
    }
  });
  it('rejects hardlinked files and invalid descriptors without tagging another inode', () => {
    const value = held('hardlink.txt');
    try {
      linkSync(value.path, join(root, 'other-name.txt'));
      expect(() => native.ensureQuarantine(value.fd)).toThrow('DOWNLOAD_QUARANTINE_UNAVAILABLE');
      expect(() => native.fileReferenceURL(value.fd)).toThrow('DOWNLOAD_FILE_CHANGED');
      expect(() => native.ensureQuarantine(0)).toThrow('DOWNLOAD_QUARANTINE_UNAVAILABLE');
    } finally {
      closeSync(value.fd);
    }
  });
});
