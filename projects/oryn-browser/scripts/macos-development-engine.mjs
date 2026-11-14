import { Buffer } from 'node:buffer';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  closeSync,
  constants,
  fstatSync,
  lstatSync,
  openSync,
  readdirSync,
  readlinkSync,
  readSync,
  realpathSync,
} from 'node:fs';
import { isAbsolute, join, relative, sep } from 'node:path';
import { isMachO } from './macos-security-policy.mjs';

export const engineInventoryResource = 'oryn-local-development-engine.json';
const maxFiles = 250_000;
const maxBytes = 16 * 1024 ** 3;
const hashPattern = /^[a-f0-9]{64}$/u;

function contained(root, path) {
  const value = relative(root, path);
  if (isAbsolute(value) || value === '..' || value.startsWith(`..${sep}`))
    throw new Error('Development Engine path escapes its payload.');
  return value;
}

function inventory(engineRoot) {
  const root = realpathSync(engineRoot);
  const tree = createHash('sha256').update('oryn-development-engine-tree-v1\n');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  let regularFiles = 0;
  let symbolicLinks = 0;
  let objects = 0;
  let logicalBytes = 0;
  const nativeFiles = [];
  function walk(directory) {
    for (const name of readdirSync(directory).sort()) {
      if (++objects > maxFiles)
        throw new Error('Development Engine exceeds the reviewed inventory bounds.');
      const path = join(directory, name);
      const entry = contained(root, path);
      const stat = lstatSync(path);
      if (name === '.DS_Store') {
        if (!stat.isFile() || isMachO(path))
          throw new Error('Unexpected executable or non-file Finder metadata.');
        continue; // Only inert Finder metadata is excluded by Builder.
      }
      if (stat.isSymbolicLink()) {
        const target = readlinkSync(path);
        if (isAbsolute(target)) throw new Error('Absolute development Engine symlink.');
        contained(root, realpathSync(path));
        symbolicLinks++;
        tree.update(`${JSON.stringify([entry, 'symlink', target])}\n`);
      } else if (stat.isDirectory()) walk(path);
      else if (stat.isFile()) {
        regularFiles++;
        logicalBytes += stat.size;
        if (regularFiles > maxFiles || logicalBytes > maxBytes || stat.size > 2 * 1024 ** 3)
          throw new Error('Development Engine exceeds the reviewed inventory bounds.');
        const descriptor = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
        const hash = createHash('sha256');
        try {
          const opened = fstatSync(descriptor);
          if (!opened.isFile() || opened.ino !== stat.ino || opened.size !== stat.size)
            throw new Error('Development Engine changed while inventorying.');
          let bytes;
          while ((bytes = readSync(descriptor, buffer, 0, buffer.length, null)) > 0)
            hash.update(buffer.subarray(0, bytes));
          const after = fstatSync(descriptor);
          if (after.size !== opened.size || after.mtimeMs !== opened.mtimeMs)
            throw new Error('Development Engine changed while inventorying.');
        } finally {
          closeSync(descriptor);
        }
        const sha256 = hash.digest('hex');
        tree.update(`${JSON.stringify([entry, 'file', stat.size, stat.mode & 0o111, sha256])}\n`);
        if (isMachO(path)) nativeFiles.push({ path: entry, sha256 });
      } else throw new Error('Unexpected development Engine filesystem object.');
      if (regularFiles + symbolicLinks > maxFiles)
        throw new Error('Development Engine exceeds the reviewed inventory bounds.');
    }
  }
  walk(root);
  return {
    schemaVersion: 1,
    kind: 'oryn-local-development-engine',
    architecture: 'arm64',
    regularFiles,
    symbolicLinks,
    logicalBytes,
    treeSha256: tree.digest('hex'),
    nativeFiles: nativeFiles.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0)),
  };
}

export async function createEngineInventory(stageRoot) {
  return inventory(stageRoot);
}

function capture(args, path, input) {
  const result = spawnSync('/usr/bin/codesign', args, {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024,
    input,
  });
  if (result.error || result.status !== 0)
    throw new Error(`Invalid preserved arm64 Engine signature at ${path}: ${result.stderr}`);
  return `${result.stdout}\n${result.stderr}`;
}

function assertUpstreamSignature(path) {
  capture(['--verify', '--strict', '--architecture', 'arm64', path], path);
  const details = capture(['-d', '--verbose=4', '--architecture', 'arm64', path], path);
  const flags = /flags=(0x[0-9a-f]+)/iu.exec(details)?.[1];
  if (
    !flags ||
    ![2, 0x20002].includes(Number.parseInt(flags, 16)) ||
    !/^Signature=adhoc$/mu.test(details) ||
    !/^TeamIdentifier=not set$/mu.test(details)
  )
    throw new Error('Preserved Engine requires original non-runtime ad-hoc/linker signatures.');
  const grants = capture(
    ['-d', '--entitlements', '-', '--xml', '--architecture', 'arm64', path],
    path,
  );
  const start = grants.indexOf('<?xml');
  const end = grants.indexOf('</plist>', start);
  if (start >= 0 || end >= 0) {
    if (start < 0 || end < 0) throw new Error('Malformed preserved Engine entitlements.');
    const parsed = spawnSync('/usr/bin/plutil', ['-convert', 'json', '-o', '-', '-'], {
      encoding: 'utf8',
      input: grants.slice(start, end + 8),
      maxBuffer: 1024 * 1024,
    });
    if (parsed.error || parsed.status !== 0 || Object.keys(JSON.parse(parsed.stdout)).length !== 0)
      throw new Error(
        'Preserved Engine must have no entitlements or device/validation exceptions.',
      );
  }
}

export async function verifyEnginePreservation(engineRoot, expected) {
  if (
    expected?.schemaVersion !== 1 ||
    expected.kind !== 'oryn-local-development-engine' ||
    expected.architecture !== 'arm64' ||
    !hashPattern.test(expected.treeSha256) ||
    !Array.isArray(expected.nativeFiles) ||
    expected.nativeFiles.length < 1 ||
    expected.nativeFiles.length > 5000
  )
    throw new Error('Invalid preserved Engine inventory.');
  const actual = inventory(engineRoot);
  if (JSON.stringify(actual) !== JSON.stringify(expected))
    throw new Error(
      'Packaged Engine differs from the original staged payload; pinned bytes must remain unchanged.',
    );
  const root = realpathSync(engineRoot);
  for (const entry of actual.nativeFiles) assertUpstreamSignature(join(root, entry.path));
  return {
    passed: true,
    payloadPreserved: true,
    architecture: 'arm64',
    treeSha256: actual.treeSha256,
    regularFiles: actual.regularFiles,
    logicalBytes: actual.logicalBytes,
    nativeFiles: actual.nativeFiles.length,
    nativeSigning: 'original ad-hoc/linker; arm64 signatures verified',
    deviceEntitlements: [],
    hardenedRuntime: false,
    capabilityHashesRewritten: false,
  };
}
