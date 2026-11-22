import { describe, expect, it } from 'vitest';

import { parseHuggingFaceModelPage } from '../model-page/hugging-face-model-url';

describe('parseHuggingFaceModelPage', () => {
  it.each([
    ['https://huggingface.co/google/gemma-3-1b-it', 'google/gemma-3-1b-it', undefined, 'root'],
    ['https://huggingface.co/Qwen/Qwen3.5-0.8B/', 'Qwen/Qwen3.5-0.8B', undefined, 'root'],
    [
      'https://huggingface.co/google/gemma-3-1b-it/tree/main',
      'google/gemma-3-1b-it',
      'main',
      'tree',
    ],
    [
      'https://huggingface.co/google/gemma-3-1b-it/blob/v1/config.json',
      'google/gemma-3-1b-it',
      'v1',
      'blob',
    ],
  ])('accepts %s', (url, repository, revision, route) => {
    const parsed = parseHuggingFaceModelPage(url);
    expect(parsed).toMatchObject({ repository, route });
    expect(parsed?.revision).toBe(revision);
  });

  it.each([
    'http://huggingface.co/google/gemma-3-1b-it',
    'https://www.huggingface.co/google/gemma-3-1b-it',
    'https://huggingface.co.evil.test/google/gemma-3-1b-it',
    'https://user@huggingface.co/google/gemma-3-1b-it',
    'https://huggingface.co:444/google/gemma-3-1b-it',
    'https://huggingface.co/datasets/google/gemma',
    'https://huggingface.co/spaces/google/gemma',
    'https://huggingface.co/settings/tokens',
    'https://huggingface.co/google/gemma/discussions',
    'https://huggingface.co/google/gemma/tree/main/extra',
    'https://huggingface.co/google/gemma/blob/main',
    'https://huggingface.co/google/gemma%2fescape',
    'https://huggingface.co/google/%E0%A4%A',
    `https://huggingface.co/${'a'.repeat(97)}/model`,
  ])('rejects %s', (url) => {
    expect(parseHuggingFaceModelPage(url)).toBeUndefined();
  });
});
