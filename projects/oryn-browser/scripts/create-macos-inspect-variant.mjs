/* global console, process */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { basename, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { flipFuses, getCurrentFuseWire } from './macos-packaging-dependencies.mjs';
import { assertFusePolicy, fusePolicy, scanBundle } from './macos-security-policy.mjs';
import { signWithSecurityPolicy } from './sign-macos.mjs';
import { verifyMacOSSecurity } from './verify-macos-security.mjs';

async function digest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

// The caller first verifies the source under release policy. Derive the test
// artifact from those exact bytes rather than rebuilding while source can drift.
export async function createInspectVariant(sourceApp, outputDirectory) {
  if (process.platform !== 'darwin') throw new Error('The inspector variant requires macOS.');
  const source = scanBundle(resolve(sourceApp)).root;
  const output = resolve(outputDirectory);
  const relation = relative(source, output);
  if (!isAbsolute(relation) && relation !== '..' && !relation.startsWith(`..${sep}`))
    throw new Error('Inspector output must be outside the source application bundle.');
  if (!basename(output).includes('inspect-test'))
    throw new Error('The output must be explicitly labelled inspect-test.');
  if (existsSync(output)) throw new Error('Inspector variant output already exists.');
  assertFusePolicy(await getCurrentFuseWire(source), 'release');
  mkdirSync(join(output, 'mac-arm64'), { recursive: true });
  const app = join(output, 'mac-arm64/Oryn Browser.app');
  const copy = spawnSync('/bin/cp', ['-cR', source, app], { stdio: 'inherit' });
  if (copy.error || copy.status !== 0)
    throw new Error('Cloning the inspector test artifact failed.');
  await flipFuses(app, fusePolicy('inspect-test'));
  await signWithSecurityPolicy(
    { app, identity: '-', platform: 'darwin', type: 'distribution' },
    { variant: 'inspect-test', mode: 'ad-hoc' },
  );
  const security = await verifyMacOSSecurity(app, { variant: 'inspect-test' });
  const equalPayloads = {};
  for (const path of ['Contents/Info.plist', 'Contents/Resources/app.asar']) {
    const sourceHash = await digest(join(source, path));
    const testHash = await digest(join(app, path));
    if (sourceHash !== testHash)
      throw new Error('Inspector test variant changed application code or metadata.');
    equalPayloads[path] = sourceHash;
  }
  const evidence = {
    schemaVersion: 1,
    app,
    sourceApp: source,
    variant: 'inspect-test',
    signing: 'ad-hoc',
    notarized: false,
    distributionQualified: false,
    executionQualified: false,
    security,
    identicalApplicationPayloads: equalPayloads,
    change:
      'The inspector fuse is enabled; code signatures/resource seals are regenerated. Source app payload and metadata are preserved.',
    nativeExecutionGate:
      'Ad-hoc hardened library validation has no Apple Team ID; native execution needs a real identity without weakening policy.',
  };
  writeFileSync(
    join(output, 'macos-security-evidence.json'),
    `${JSON.stringify(evidence, null, 2)}\n`,
  );
  return evidence;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 4)
    throw new Error(
      'Pass a release-policy Oryn Browser.app and a fresh inspect-test output directory.',
    );
  await verifyMacOSSecurity(process.argv[2]);
  console.log(
    `MACOS_INSPECT_VARIANT_RESULT ${JSON.stringify(await createInspectVariant(process.argv[2], process.argv[3]))}`,
  );
}
