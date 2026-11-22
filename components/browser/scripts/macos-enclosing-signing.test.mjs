import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import process from 'node:process';
import { test } from 'node:test';
import { signAsync } from './macos-packaging-dependencies.mjs';
import { enclosingAppSigningOptions, signingOptions } from './sign-macos.mjs';

function codesign(args) {
  return spawnSync('/usr/bin/codesign', args, { encoding: 'utf8', timeout: 15_000 });
}

function digest(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

test(
  'real enclosing signer preserves child signatures and leaves binary resources unsigned',
  { skip: process.platform !== 'darwin', timeout: 30_000 },
  async (t) => {
    const directory = mkdtempSync(join(tmpdir(), 'oryn-enclosing-sign-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    const app = join(directory, 'Oryn Browser.app');
    const executable = join(app, 'Contents/MacOS/Oryn Browser');
    const child = join(app, 'Contents/Resources/engine/runtime/bin/python3.12');
    const inert = join(app, 'Contents/Resources/engine/static/main.js.br');
    for (const path of [executable, child, inert]) mkdirSync(dirname(path), { recursive: true });
    copyFileSync('/usr/bin/true', executable);
    copyFileSync('/usr/bin/true', child);
    // osx-sign classifies this as binary data. A lost ignore predicate would
    // therefore attempt to sign it even though it is an inert web resource.
    writeFileSync(inert, Buffer.from([0, 255, 128, 1, 2, 3, 0, 254]));
    writeFileSync(
      join(app, 'Contents/Info.plist'),
      `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>ai.ocno.oryn.fixture.enclosing</string>
<key>CFBundleExecutable</key><string>Oryn Browser</string>
<key>CFBundleName</key><string>Oryn Browser</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1</string>
</dict></plist>\n`,
    );
    // Give the child a distinct valid policy. Accidentally applying the main
    // app's inherited hardened policy again must visibly change these bytes.
    const childSign = codesign([
      '--force',
      '--sign',
      '-',
      '--identifier',
      'ai.ocno.oryn.fixture.child',
      '--options',
      '0x0',
      '--timestamp=none',
      child,
    ]);
    assert.equal(childSign.status, 0, childSign.stderr);
    const childBefore = digest(child);
    const inertBefore = digest(inert);
    assert.notEqual(codesign(['--display', inert]).status, 0);

    const signing = signingOptions(
      { app, platform: 'darwin', gatekeeperAssess: false },
      { adHoc: true },
    );
    // Exercise the actual dependency normalizer and signer, rather than merely
    // calling the predicate: osx-sign 1.3.1 silently drops array-form filters.
    await signAsync(enclosingAppSigningOptions(signing));

    assert.equal(digest(child), childBefore, 'Enclosing seal changed a child Mach-O signature');
    assert.equal(digest(inert), inertBefore, 'Enclosing seal changed an inert binary resource');
    assert.notEqual(
      codesign(['--display', inert]).status,
      0,
      'Inert resource acquired a signature',
    );
    const verify = codesign(['--verify', '--deep', '--strict', app]);
    assert.equal(verify.status, 0, verify.stderr);
    const details = codesign(['--display', '--verbose=4', app]);
    assert.equal(details.status, 0, details.stderr);
    assert.match(details.stderr, /flags=.*\bruntime\b/u);
  },
);
