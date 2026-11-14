/* global process, console */
import { Buffer } from 'node:buffer';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  closeSync,
  constants,
  fstatSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  realpathSync,
  writeFileSync,
} from 'node:fs';
import { isAbsolute, join, relative, sep } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import { isMachO } from './macos-security-policy.mjs';

export const releasePublisherTeam = '3AWD2NU8UF';
export const nativeBindingIdentifier = 'ai.ocno.oryn.browser.nativebindings';
export const nativeBindingName = 'native-release-bindings.bundle';
const hashPattern = /^[a-f0-9]{64}$/u;

function containedFile(root, path) {
  const name = relative(root, path);
  if (!name || isAbsolute(name) || name === '..' || name.startsWith(`..${sep}`))
    throw new Error('Native release input escapes its Engine payload.');
  let current = root;
  for (const part of name.split(sep)) {
    current = join(current, part);
    if (lstatSync(current).isSymbolicLink())
      throw new Error('Native release inputs must not traverse symlinks.');
  }
  if (!lstatSync(path).isFile()) throw new Error('Native release input is not a regular file.');
  return name.split(sep).join('/');
}

export function releaseInputDigest(path) {
  const descriptor = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  const hash = createHash('sha256');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  try {
    const before = fstatSync(descriptor);
    if (!before.isFile()) throw new Error('Native release input is not a regular file.');
    let bytes;
    while ((bytes = readSync(descriptor, buffer, 0, buffer.length, null)) > 0)
      hash.update(buffer.subarray(0, bytes));
    const after = fstatSync(descriptor);
    if (after.size !== before.size || after.mtimeMs !== before.mtimeMs)
      throw new Error('Native release input changed while hashing.');
    return hash.digest('hex');
  } finally {
    closeSync(descriptor);
  }
}

export function snapshotReleaseNativeInputs(engineRoot, paths) {
  const root = realpathSync(engineRoot);
  const entries = paths.map((path) => {
    const name = containedFile(root, path);
    if (!isMachO(path)) throw new Error('Native release input is not Mach-O.');
    return { path: name, originalSha256: releaseInputDigest(path) };
  });
  if (!entries.length || entries.length > 8192)
    throw new Error('Unexpected native release input count.');
  if (new Set(entries.map((entry) => entry.path)).size !== entries.length)
    throw new Error('Duplicate native release input.');
  return entries.sort((a, b) => a.path.localeCompare(b.path));
}

export function completeReleaseBindings(engineRoot, nativeInputs, sourceInputs) {
  const root = realpathSync(engineRoot);
  const nativeFiles = nativeInputs.map((entry) => {
    const path = join(root, entry.path);
    containedFile(root, path);
    if (!isMachO(path)) throw new Error('Final native release input is not Mach-O.');
    return { ...entry, signedSha256: releaseInputDigest(path) };
  });
  const files = [...nativeFiles, ...sourceInputs];
  if (files.length > 8192 || new Set(files.map((entry) => entry.path)).size !== files.length)
    throw new Error('Unexpected or duplicate native release binding.');
  for (const entry of files) {
    if (
      Object.keys(entry).sort().join(',') !== 'originalSha256,path,signedSha256' ||
      !hashPattern.test(entry.originalSha256) ||
      !hashPattern.test(entry.signedSha256) ||
      containedFile(root, join(root, entry.path)) !== entry.path ||
      releaseInputDigest(join(root, entry.path)) !== entry.signedSha256
    )
      throw new Error('Invalid final native release binding.');
  }
  const capabilityPath = join(
    root,
    'runtime/lib/python3.12/site-packages/oryn_engine/data/capabilities.json',
  );
  return {
    schemaVersion: 1,
    capabilitySha256: releaseInputDigest(capabilityPath),
    files: files.sort((a, b) => a.path.localeCompare(b.path)),
  };
}

function run(command, args, { capture = false } = {}) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    stdio: capture ? 'pipe' : 'inherit',
    maxBuffer: 8 * 1024 * 1024,
  });
  if (result.error || result.status !== 0)
    throw new Error(
      `${command} failed: ${result.stderr || result.error?.message || result.status}`,
    );
  return result.stdout;
}

export function verifyReleaseRuntime(app) {
  run(process.execPath, [
    fileURLToPath(new URL('./verify-packaged-runtime.mjs', import.meta.url)),
    app,
  ]);
}

export function prepareNativeRelease(app, nativeFiles) {
  if (process.env.ORYN_APPLE_TEAM_ID !== releasePublisherTeam)
    throw new Error('Native release bindings require the pinned Oryn publisher team.');
  const engineRoot = join(realpathSync(app), 'Contents/Resources/engine');
  // Authenticate and exercise the reviewed, unmodified runtime before any file
  // is transformed or re-signed. The original RSA artifacts stay byte-identical.
  verifyReleaseRuntime(app);
  const original = snapshotReleaseNativeInputs(
    engineRoot,
    nativeFiles.filter((path) => path.startsWith(`${engineRoot}${sep}`)),
  );
  const sourceInputs = JSON.parse(
    run(
      join(engineRoot, 'runtime/bin/python3.12'),
      [
        '-I',
        '-B',
        '-c',
        'import json,sys; from pathlib import Path; from oryn_engine.native_release_bindings import prepare_release_sources; print(json.dumps(prepare_release_sources(Path(sys.argv[1]))))',
        engineRoot,
      ],
      { capture: true },
    ),
  );
  return { engineRoot, original, sourceInputs };
}

export function sealNativeRelease(prepared, options) {
  const { engineRoot, original, sourceInputs } = prepared;
  const manifest = completeReleaseBindings(engineRoot, original, sourceInputs);
  if (Buffer.byteLength(JSON.stringify(manifest)) > 2 * 1024 * 1024)
    throw new Error('Native release binding exceeds its reviewed size bound.');
  const bundle = join(engineRoot, nativeBindingName);
  const contents = join(bundle, 'Contents');
  mkdirSync(join(contents, 'Resources'), { recursive: true });
  writeFileSync(
    join(contents, 'Info.plist'),
    `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>${nativeBindingIdentifier}</string>
<key>CFBundleName</key><string>Oryn Native Release Bindings</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundlePackageType</key><string>BNDL</string>
</dict></plist>\n`,
  );
  writeFileSync(join(contents, 'Resources/native-hashes.json'), `${JSON.stringify(manifest)}\n`);
  const args = ['--force', '--sign', options.identity, '--options', 'runtime', '--timestamp'];
  if (options.keychain) args.push('--keychain', options.keychain);
  run('/usr/bin/codesign', [...args, bundle]);
  run('/usr/bin/codesign', [
    '--verify',
    '--strict',
    `-R=anchor apple generic and identifier "${nativeBindingIdentifier}" and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "${releasePublisherTeam}"`,
    bundle,
  ]);
  console.log(`ORYN_NATIVE_RELEASE_BINDINGS ${JSON.stringify({ files: manifest.files.length })}`);
  return manifest;
}

export function assertNativeReleaseUnchanged(prepared, sealed) {
  const current = completeReleaseBindings(
    prepared.engineRoot,
    prepared.original,
    prepared.sourceInputs,
  );
  if (JSON.stringify(current) !== JSON.stringify(sealed))
    throw new Error('Engine inputs changed after their final release binding.');
  // Also catch accidental writes to the signed map during the enclosing seal.
  const stored = JSON.parse(
    readFileSync(
      join(prepared.engineRoot, nativeBindingName, 'Contents/Resources/native-hashes.json'),
      'utf8',
    ),
  );
  if (JSON.stringify(stored) !== JSON.stringify(sealed))
    throw new Error('Native release binding changed during the enclosing app seal.');
}
