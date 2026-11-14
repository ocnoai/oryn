/* global console, process */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import {
  FuseState,
  FuseV1Options,
  FuseVersion,
  flipFuses,
  getCurrentFuseWire,
} from './macos-packaging-dependencies.mjs';

const appRequire = createRequire(new URL('../apps/browser/package.json', import.meta.url));
async function digest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
function run(command, args) {
  const result = spawnSync(command, args, { encoding: 'utf8' });
  if (result.error || result.status !== 0)
    throw new Error(`${command} failed: ${result.stderr || result.error?.message}`);
  return result.stdout;
}

/** Source-only Electron copy: narrow file authority without changing installed node_modules. */
export async function prepareSourceElectron(outputDirectory) {
  if (process.platform !== 'darwin') throw new Error('This source test lane requires macOS.');
  const originalExecutable = appRequire('electron');
  const source = resolve(dirname(originalExecutable), '../..');
  const output = resolve(outputDirectory);
  const relation = relative(source, output);
  if (!isAbsolute(relation) && relation !== '..' && !relation.startsWith(`..${sep}`))
    throw new Error('Test Electron output must be outside the installed bundle.');
  const app = join(output, 'Electron.app');
  const evidencePath = join(output, 'source-electron-evidence.json');
  if (existsSync(app) || existsSync(evidencePath))
    throw new Error('Source Electron test output already exists.');
  mkdirSync(output, { recursive: true, mode: 0o700 });
  const before = await getCurrentFuseWire(source);
  const frameworkRelative =
    'Contents/Frameworks/Electron Framework.framework/Versions/A/Electron Framework';
  const sourceFrameworkHash = await digest(join(source, frameworkRelative));
  const sourceExecutableHash = await digest(originalExecutable);
  run('/bin/cp', ['-cR', source, app]);
  await flipFuses(app, {
    version: FuseVersion.V1,
    [FuseV1Options.GrantFileProtocolExtraPrivileges]: false,
    resetAdHocDarwinSignature: true,
  });
  const after = await getCurrentFuseWire(app);
  for (const [key, value] of Object.entries(before)) {
    if (key === String(FuseV1Options.GrantFileProtocolExtraPrivileges)) {
      if (after[key] !== FuseState.DISABLE)
        throw new Error('Test Electron file privilege fuse was not disabled.');
    } else if (after[key] !== value)
      throw new Error('Source Electron preparation changed another fuse.');
  }
  run('/usr/bin/codesign', ['--verify', '--deep', '--strict', app]);
  if (
    JSON.stringify(await getCurrentFuseWire(source)) !== JSON.stringify(before) ||
    (await digest(join(source, frameworkRelative))) !== sourceFrameworkHash ||
    (await digest(originalExecutable)) !== sourceExecutableHash
  )
    throw new Error('The original installed Electron changed.');
  for (const payload of ['Contents/Info.plist', 'Contents/Resources/default_app.asar']) {
    if ((await digest(join(source, payload))) !== (await digest(join(app, payload))))
      throw new Error('Source Electron application metadata or default loader changed.');
  }
  const evidence = {
    schemaVersion: 1,
    lane: 'source-electron-file-policy',
    app,
    executable: join(app, 'Contents/MacOS/Electron'),
    signing: 'ad-hoc',
    notarized: false,
    distributionQualified: false,
    change:
      'Only GrantFileProtocolExtraPrivileges is disabled; ad-hoc code signatures/resource seals are regenerated with original entitlements, requirements, flags and runtime metadata preserved.',
    originalUnchanged: true,
    originalExecutableSHA256: sourceExecutableHash,
    originalFrameworkSHA256: sourceFrameworkHash,
    sourceFuses: before,
    testFuses: after,
  };
  writeFileSync(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`, {
    flag: 'wx',
    mode: 0o600,
  });
  return evidence;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  if (process.argv.length !== 3)
    throw new Error('Pass a fresh source Electron test output directory.');
  console.log(JSON.stringify(await prepareSourceElectron(process.argv[2]), null, 2));
}
