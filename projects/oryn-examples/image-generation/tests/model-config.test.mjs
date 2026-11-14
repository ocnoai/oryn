import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const sourceUrl = new URL('../src/model-config.ts', import.meta.url);

async function loadConfig(profile) {
  const source = (await readFile(sourceUrl, 'utf8')).replace(
    '__ORYN_IMAGE_MODEL_PROFILE__',
    JSON.stringify(profile),
  );
  const output = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  return import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}#${profile}`);
}

test('production request pins FLUX and permits compatible substitution', async () => {
  const config = await loadConfig('production');
  const request = config.createImageModelRequest();

  assert.deepEqual(request.preferredModel, {
    provider: 'huggingface',
    repository: 'black-forest-labs/FLUX.1-dev',
    revision: '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21',
  });
  assert.deepEqual(request.certifiedModels, [
    {
      provider: 'huggingface',
      repository: 'segmind/Segmind-Vega',
      revision: '7714c4363e5856ff974a4f4b068e8691f26d0b40',
    },
  ]);
  assert.deepEqual(request.requirements, {
    task: 'text-to-image',
    inputModalities: ['text'],
    outputModalities: ['image'],
    capabilities: { deterministicSeed: true },
  });
  assert.deepEqual(request.substitution, { automatic: 'compatible' });
});

test('fixture profile changes only the example-owned request configuration', async () => {
  const config = await loadConfig('fixture');
  const request = config.createImageModelRequest();

  assert.equal(request.preferredModel.repository, 'dg845/tiny-random-stable-diffusion-xl');
  assert.equal(request.preferredModel.revision, '7ae769dd7f63b06308f2f23881d329a0cb1c89d2');
  assert.deepEqual(request.certifiedModels, [request.preferredModel]);
  assert.deepEqual(request.substitution, { automatic: 'none' });
  assert.equal(request.role, 'image-generation');
  assert.equal(request.requirements.task, 'text-to-image');
  assert.deepEqual(request.requirements.outputModalities, ['image']);
});
