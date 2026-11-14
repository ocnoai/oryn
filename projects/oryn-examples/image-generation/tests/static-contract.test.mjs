import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import test from 'node:test';

const packageRoot = new URL('../', import.meta.url);

test('production and fixture builds identify distinct model profiles', async () => {
  const production = await readFile(new URL('dist/index.html', packageRoot), 'utf8');
  const fixture = await readFile(new URL('dist-fixture/index.html', packageRoot), 'utf8');

  assert.match(production, /name="oryn-model-profile" content="production"/);
  assert.match(fixture, /name="oryn-model-profile" content="fixture"/);
});

test('first render exposes the required accessible controls and truth fields', async () => {
  const html = await readFile(new URL('index.html', packageRoot), 'utf8');
  for (const id of [
    'prompt',
    'steps',
    'use-seed',
    'generate',
    'cancel',
    'image-preview',
    'save',
    'read',
    'upload',
    'requested-model',
    'resolved-model',
    'substitution',
    'runtime',
    'backend',
  ]) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
  assert.match(html, /role="status" aria-live="polite"/);
  assert.match(html, /Nothing is uploaded automatically/);
});

test('functional icons use the semantic Oryn adapter', async () => {
  const html = await readFile(new URL('index.html', packageRoot), 'utf8');
  const adapter = await readFile(new URL('src/icons.ts', packageRoot), 'utf8');

  assert.match(html, /data-oryn-icon="local-privacy"/);
  assert.match(html, /data-oryn-icon="generate"/);
  assert.match(adapter, /nameAttr: 'data-oryn-icon'/);
  assert.match(adapter, /LocalPrivacy: ShieldCheck/);
  assert.match(adapter, /Generate: Sparkles/);
});

test('example source has no privileged or runtime-selection escape hatches', async () => {
  const sourceDirectory = new URL('src/', packageRoot);
  const source = (
    await Promise.all(
      (await readdir(sourceDirectory))
        .filter((name) => name.endsWith('.ts'))
        .map((name) => readFile(new URL(name, sourceDirectory), 'utf8')),
    )
  ).join('\n');
  const forbidden = [
    'ipcRenderer',
    'child_process',
    'pythonSource',
    'shellCommand',
    'entrypointPath',
    'deviceMap',
    'offloadStrategy',
    'quantizationStrategy',
    'filesystemPath',
    'runtimePackId',
  ];

  for (const term of forbidden) assert.equal(source.includes(term), false, term);
  assert.match(source, /createLocalAIClient/);
  assert.match(source, /invokeAndWait/);
  assert.match(source, /artifacts\.getUrl/);
  assert.match(source, /artifacts\.readBlob/);
  assert.match(source, /artifacts\.saveAs/);
  assert.match(source, /steps: Number\(steps\.value\)/);
  assert.doesNotMatch(source, /numInferenceSteps|outputFormat/);
});

test('website controls enforce bounded image input choices', async () => {
  const html = await readFile(new URL('index.html', packageRoot), 'utf8');
  assert.deepEqual(
    [...html.matchAll(/name="size" value="(\d+x\d+)"/g)].map((match) => match[1]),
    ['512x512', '768x512', '512x768', '256x256'],
  );
  assert.deepEqual(
    [...html.matchAll(/<option value="(\d+)"[^>]*>[^<]*(?:fastest|quick|balanced|detailed)/g)].map(
      (match) => Number(match[1]),
    ),
    [1, 4, 12, 20],
  );
  assert.match(html, /<option value="20" selected>20 · detailed<\/option>/);
  assert.match(html, /min="0" max="4294967295"/);
  assert.match(html, /maxlength="2000" required/);
});
