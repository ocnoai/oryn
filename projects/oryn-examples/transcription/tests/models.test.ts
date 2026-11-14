// @ts-nocheck -- Browser workspace tests run directly in Node's type-stripper.
import assert from 'node:assert/strict';
import test from 'node:test';
import { buildModelRequest, defaultModelForLanguage, getSelection, LANGUAGES, modelsForLanguage } from '../src/models.ts';

const modelIds = language => modelsForLanguage(language).map(model => model.id);

test('the language picker includes all 100 Whisper language tokens plus automatic detection', () => {
  assert.equal(LANGUAGES.length, 101);
  assert.equal(LANGUAGES[0].id, 'auto');
  assert.equal(new Set(LANGUAGES.map(language => language.id)).size, LANGUAGES.length);
  for (const id of ['sk', 'cs', 'hr', 'no', 'uk', 'bg', 'sr', 'tl', 'yue']) {
    assert.ok(LANGUAGES.some(language => language.id === id));
  }
});

test('Slovak keeps multilingual Whisper choices while Czech also offers Qwen', () => {
  assert.deepEqual(modelIds('sk'), ['whisper-small', 'whisper-tiny', 'whisper-large-v3-turbo', 'whisper-large-v3']);
  assert.ok(modelIds('cs').includes('qwen3-asr-0.6b'));
  assert.ok(modelIds('cs').includes('qwen3-asr-1.7b'));
  assert.equal(getSelection('sk', 'whisper-small').inputLanguage, 'sk');
  assert.equal(getSelection('cs', 'qwen3-asr-1.7b').inputLanguage, 'Czech');
});

test('Qwen is offered for exactly its 30 named languages and receives official processor names', () => {
  const qwenLanguages = LANGUAGES.filter(language => modelIds(language.id).includes('qwen3-asr-0.6b'));
  assert.equal(qwenLanguages.length, 30);
  assert.equal(getSelection('zh', 'qwen3-asr-0.6b').inputLanguage, 'Chinese');
  assert.equal(getSelection('en', 'qwen3-asr-0.6b').inputLanguage, 'English');
  assert.equal(getSelection('tl', 'qwen3-asr-0.6b').inputLanguage, 'Filipino');
  assert.equal(getSelection('tl', 'whisper-small').inputLanguage, 'tl');
});

test('English-only Whisper is available only for English and omits forced language/task arguments via auto', () => {
  assert.ok(modelIds('en').includes('whisper-small.en'));
  assert.ok(!modelIds('cs').includes('whisper-small.en'));
  assert.ok(!modelIds('auto').includes('whisper-small.en'));
  const selection = getSelection('en', 'whisper-small.en');
  assert.equal(selection.inputLanguage, 'auto');
  assert.equal(buildModelRequest(selection, true).requirements.parameterValues.language, 'auto');
});

test('Cantonese excludes Whisper checkpoints without a Cantonese token', () => {
  assert.deepEqual(modelIds('yue'), ['whisper-large-v3-turbo', 'whisper-large-v3', 'qwen3-asr-0.6b', 'qwen3-asr-1.7b']);
  assert.equal(defaultModelForLanguage('yue').id, 'whisper-large-v3-turbo');
  assert.equal(getSelection('yue', 'whisper-large-v3').inputLanguage, 'yue');
});

test('automatic detection excludes language-restricted choices and preserves the original Whisper Small default', () => {
  assert.deepEqual(modelIds('auto'), ['whisper-small', 'whisper-tiny', 'whisper-large-v3-turbo', 'whisper-large-v3']);
  assert.equal(getSelection('auto', 'whisper-small').inputLanguage, 'auto');
  assert.equal(defaultModelForLanguage('auto').repository, 'openai/whisper-small');
  assert.equal(defaultModelForLanguage('en').repository, 'openai/whisper-small');
  assert.equal(defaultModelForLanguage('sk').repository, 'openai/whisper-small');
});

test('unknown language, unknown model, and unsupported language/model pairs fail before requesting a model', () => {
  assert.throws(() => modelsForLanguage('unsupported'), /Choose a language/);
  assert.throws(() => defaultModelForLanguage('unsupported'), /Choose a language/);
  assert.throws(() => getSelection('en', 'unknown'), /Choose a transcription model/);
  assert.throws(() => getSelection('sk', 'qwen3-asr-0.6b'), /Choose a transcription model/);
  assert.throws(() => getSelection('cs', 'whisper-small.en'), /Choose a transcription model/);
  assert.throws(() => getSelection('yue', 'whisper-small'), /Choose a transcription model/);
  assert.throws(() => getSelection('auto', 'qwen3-asr-1.7b'), /Choose a transcription model/);
});

test('requests preserve the chosen repository, reject automatic substitution, and declare language requirements', () => {
  const request = buildModelRequest(getSelection('cs', 'qwen3-asr-1.7b'), true, true);
  assert.deepEqual(request.preferredModel, {
    provider: 'huggingface', repository: 'Qwen/Qwen3-ASR-1.7B-hf', revision: 'bcd2b5b7f32b480ab5790554cfa8347f246a14f3',
  });
  assert.deepEqual(request.substitution, { automatic: 'none' });
  assert.deepEqual(request.selection, { version: 1, intent: 'preferred-variant' });
  assert.deepEqual(request.requirements, {
    task: 'automatic-speech-recognition', inputModalities: ['audio'], outputModalities: ['text'],
    contract: { id: 'transcription-v1', revision: 1 }, requiredFeatures: ['speechRecognition'],
    parameterValues: { language: 'Czech', timestamps: false },
  });
});

test('older runtimes receive the chosen model and standard modalities without unsupported profile fields', () => {
  const request = buildModelRequest(getSelection('sk', 'whisper-small'), false);
  assert.deepEqual(request.preferredModel, { provider: 'huggingface', repository: 'openai/whisper-small' });
  assert.deepEqual(request.requirements, {
    task: 'automatic-speech-recognition', inputModalities: ['audio'], outputModalities: ['text'],
  });
  assert.equal(request.selection, undefined);
  assert.deepEqual(request.substitution, { automatic: 'none' });
});
