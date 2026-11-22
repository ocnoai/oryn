import { describe, expect, it } from 'vitest';

import {
  MODEL_WORKSPACE_TASK_CONTRACTS,
  validateModelWorkspaceInput,
} from '../model-workspace/task-contract-registry';

const artifactId = `artifact_${'a'.repeat(24)}`;

it('preserves explicit audio classifier heads and rejects unknown heads', () => {
  for (const classificationHead of ['language', 'gender']) {
    expect(
      validateModelWorkspaceInput('audio-classification-v1', {
        audioArtifactId: artifactId,
        classificationHead,
      }),
    ).toEqual({ audioArtifactId: artifactId, classificationHead, topK: 5 });
  }
  expect(() =>
    validateModelWorkspaceInput('audio-classification-v1', {
      audioArtifactId: artifactId,
      classificationHead: 'other',
    }),
  ).toThrow();
});

const valid: Record<(typeof MODEL_WORKSPACE_TASK_CONTRACTS)[number], unknown> = {
  'pose-estimation-v1': { imageArtifactId: artifactId, boxes: [[0, 0, 1, 1]] },
  'unconditional-image-generation-v1': { width: 32, height: 32, steps: 2, seed: 0 },
  'audio-question-answering-v1': {
    audioArtifactId: artifactId,
    instruction: 'Describe the sound.',
  },
  'audio-transformation-v1': { operation: 'enhance', audioArtifactId: artifactId },
  'multimodal-generation-v1': {
    inputMode: 'image',
    instruction: 'Describe this image.',
    imageArtifactId: artifactId,
    outputMode: 'text',
  },
  'image-text-to-video-v1': { imageArtifactId: artifactId, prompt: 'The cube turns.' },
  'video-editing-v1': { videoArtifactId: artifactId, operation: 'restore' },
  'tabular-classification-v1': {
    columns: [{ name: 'x', type: 'numeric' }],
    trainingRows: [[1], [2]],
    targets: ['cold', 'warm'],
    queryRows: [[1.5]],
  },
  'tabular-regression-v1': {
    columns: [{ name: 'x', type: 'numeric' }],
    trainingRows: [[1], [2]],
    targets: [2, 4],
    queryRows: [[1.5]],
  },
  'time-series-forecasting-v1': { context: [1, 2, 3], horizon: 2 },
  'visual-document-retrieval-v1': { imageArtifactId: artifactId, queries: ['Invoice total'] },
  'table-question-answering-v1': { question: 'Which city?', columns: ['City'], rows: [['Paris']] },
  'document-question-answering-v1': {
    imageArtifactId: artifactId,
    question: 'Total?',
    words: ['42'],
    boxes: [[0.1, 0.1, 0.2, 0.2]],
  },
  'keypoint-detection-v1': { imageArtifactId: artifactId },
  'video-classification-v1': { videoArtifactId: artifactId },
  'video-question-answering-v1': { videoArtifactId: artifactId, question: 'What happens first?' },
  'voice-activity-detection-v1': { audioArtifactId: artifactId },
  'zero-shot-classification-v1': { text: 'A lovely sunny day', labels: ['weather', 'finance'] },
  'image-embeddings-v1': { imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa' },
  'image-segmentation-v1': { imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa' },
  'mask-generation-v1': {
    imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa',
    points: [{ x: 0.5, y: 0.5, label: 1 }],
  },
  'zero-shot-object-detection-v1': {
    imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa',
    labels: ['cat', 'dog'],
  },
  'image-editing-v1': {
    imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa',
    prompt: 'Make the sky blue',
  },
  'image-layers-v1': {
    imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa',
    prompt: 'Separate layers',
  },
  'image-to-text-v1': { imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa' },
  'visual-question-answering-v1': {
    imageArtifactId: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa',
    question: 'What is in the image?',
  },
  'chat-v1': {
    messages: [{ role: 'user', content: 'Hello' }],
    maxTokens: 64,
    temperature: 0.7,
    topP: 0.9,
  },
  'chat-v2': {
    messages: [{ role: 'user', content: 'Hello' }],
    maxTokens: 4096,
    thinking: false,
  },
  'text-to-image-v1': {
    prompt: 'A lighthouse',
    width: 512,
    height: 512,
    steps: 4,
    guidanceScale: 1,
    seed: 2,
  },
  'image-to-image-v1': {
    prompt: 'Watercolor',
    width: 512,
    height: 512,
    steps: 4,
    guidanceScale: 1,
    seed: 2,
    imageArtifactId: artifactId,
    strength: 0.8,
  },
  'text-to-music-v1': {
    prompt: 'Quiet piano',
    lyrics: '[Instrumental]',
    durationSeconds: 10,
    seed: 2,
  },
  'speech-synthesis-v1': { text: 'Hello', voice: 'af_heart', language: 'en-us', speed: 1 },
  'transcription-v1': { audioArtifactId: artifactId, language: 'auto', timestamps: false },
  'media-audio-extract-v1': { videoArtifactId: artifactId, track: 'default' },
  'audio-source-separation-v1': { audioArtifactId: artifactId, target: 'vocals' },
  'forced-alignment-v1': {
    audioArtifactId: artifactId,
    transcript: 'Hello world',
    language: 'en',
    timestamps: 'word',
  },
  'subtitle-export-v1': {
    format: 'subrip',
    segments: [{ startMs: 0, endMs: 1_000, text: 'Hello world' }],
  },
  'embeddings-v1': { texts: ['one', 'two'], normalize: true },
  'reranking-v1': {
    query: 'fruit',
    documents: ['apple', 'chair'],
    instruction: 'Rank relevance',
    topK: 2,
  },
  'text-to-video-v1': {
    prompt: 'A cloud',
    width: 512,
    height: 512,
    steps: 4,
    guidanceScale: 1,
    seed: 2,
    frames: 16,
    fps: 8,
  },
  'image-to-video-v1': {
    prompt: 'Move gently',
    width: 512,
    height: 512,
    steps: 4,
    guidanceScale: 1,
    seed: 2,
    frames: 16,
    fps: 8,
    imageArtifactId: artifactId,
  },
  'text-classification-v1': { text: 'Oryn is useful', topK: 5 },
  'token-classification-v1': { text: 'Boris uses Oryn' },
  'question-answering-v1': { question: 'Who?', context: 'Boris uses Oryn.', maxAnswerLength: 16 },
  'fill-mask-v1': { text: 'Oryn runs [MASK].', topK: 5 },
  'translation-v1': { text: 'Hello', maxTokens: 64 },
  'summarization-v1': { text: 'A sufficiently long article.', maxTokens: 64 },
  'image-classification-v1': { imageArtifactId: artifactId, topK: 5 },
  'audio-classification-v1': { audioArtifactId: artifactId, topK: 5 },
  'object-detection-v1': { imageArtifactId: artifactId, threshold: 0.5 },
  'depth-estimation-v1': { imageArtifactId: artifactId },
  'zero-shot-image-classification-v1': { imageArtifactId: artifactId, labels: ['cat', 'dog'] },
};

describe('model workspace task-contract registry', () => {
  it('accepts best-effort thinking and larger bounded budgets only under chat-v2', () => {
    const messages = [{ role: 'user', content: 'Write a short article.' }];
    for (const thinking of [true, false]) {
      expect(
        validateModelWorkspaceInput('chat-v2', { messages, thinking, maxTokens: 8192 }),
      ).toMatchObject({ thinking, maxTokens: 8192 });
    }
    expect(validateModelWorkspaceInput('chat-v2', { messages })).toMatchObject({ maxTokens: 4096 });
    for (const input of [
      { messages, thinking: 'off' },
      { messages, maxTokens: 8193 },
      { messages, enable_thinking: false },
    ]) {
      expect(() => validateModelWorkspaceInput('chat-v2', input)).toThrow();
    }
    expect(() => validateModelWorkspaceInput('chat-v1', { messages, thinking: false })).toThrow();
    expect(() => validateModelWorkspaceInput('chat-v1', { messages, maxTokens: 513 })).toThrow();
  });
  it('uses the instruction-editing contract for a runtime that declares it', () => {
    const input = { ...(valid['text-to-image-v1'] as object), imageArtifactId: artifactId };
    const capabilities = { instructionEditing: true };
    expect(validateModelWorkspaceInput('text-to-image-v1', input, capabilities)).toEqual(input);
    expect(() =>
      validateModelWorkspaceInput(
        'text-to-image-v1',
        {
          ...input,
          strength: 0.8,
        },
        capabilities,
      ),
    ).toThrow('MODEL_WORKSPACE_INPUT_INVALID');
    expect(
      validateModelWorkspaceInput('text-to-image-v1', {
        ...input,
        strength: 0.8,
      }),
    ).toMatchObject({ strength: 0.8 });
  });

  it('covers and validates every shipped interactive task contract', () => {
    expect([...MODEL_WORKSPACE_TASK_CONTRACTS].sort()).toEqual(Object.keys(valid).sort());
    for (const contract of MODEL_WORKSPACE_TASK_CONTRACTS) {
      expect(validateModelWorkspaceInput(contract, valid[contract])).toBeTruthy();
    }
  });

  it('rejects executable, scheduler, oversized, and cross-contract fields', () => {
    expect(() =>
      validateModelWorkspaceInput('chat-v1', {
        ...(valid['chat-v1'] as object),
        pythonSource: 'x',
      }),
    ).toThrow('MODEL_WORKSPACE_INPUT_INVALID');
    expect(() =>
      validateModelWorkspaceInput('text-to-image-v1', {
        ...(valid['text-to-image-v1'] as object),
        frames: 16,
      }),
    ).toThrow('MODEL_WORKSPACE_INPUT_INVALID');
    expect(() =>
      validateModelWorkspaceInput('embeddings-v1', { texts: Array(65).fill('x') }),
    ).toThrow('MODEL_WORKSPACE_INPUT_INVALID');
    expect(() =>
      validateModelWorkspaceInput('transcription-v1', { audioArtifactId: 'artifact_guess' }),
    ).toThrow('MODEL_WORKSPACE_INPUT_INVALID');
    expect(() =>
      validateModelWorkspaceInput('subtitle-export-v1', {
        format: 'subrip',
        segments: [
          { startMs: 100, endMs: 200, text: 'one' },
          { startMs: 199, endMs: 300, text: 'overlap' },
        ],
      }),
    ).toThrow('MODEL_WORKSPACE_INPUT_INVALID');
  });
});

it('accepts browser-owned image references in chat and rejects paths or loader controls', () => {
  const messages = [{ role: 'user', content: 'Describe this image.' }];
  const imageArtifactId = 'artifact_abcdefghijklmnopqrstuvwx';
  expect(validateModelWorkspaceInput('chat-v1', { messages, imageArtifactId })).toMatchObject({
    imageArtifactId,
  });
  expect(() =>
    validateModelWorkspaceInput('chat-v1', { messages, imageArtifactId: '/tmp/image.png' }),
  ).toThrow();
  expect(() =>
    validateModelWorkspaceInput('chat-v1', { messages, runtime: 'custom.py' }),
  ).toThrow();
});

it('validates optional image editing with the existing artifact contract', () => {
  const imageArtifactId = 'artifact_abcdefghijklmnopqrstuvwx';
  expect(
    validateModelWorkspaceInput('text-to-image-v1', {
      prompt: 'A blue bird',
      imageArtifactId,
      strength: 0.5,
    }),
  ).toMatchObject({ imageArtifactId, strength: 0.5 });
  expect(() =>
    validateModelWorkspaceInput('text-to-image-v1', {
      prompt: 'A blue bird',
      imageArtifactId,
      strength: 2,
    }),
  ).toThrow();
});

it('preserves an absent reranking instruction for runtimes that do not support one', () => {
  expect(
    validateModelWorkspaceInput('reranking-v1', { query: 'cats', documents: ['cat', 'boat'] }),
  ).not.toHaveProperty('instruction');
  expect(
    validateModelWorkspaceInput('reranking-v1', {
      query: 'cats',
      documents: ['cat'],
      instruction: 'Rank relevance',
    }),
  ).toHaveProperty('instruction', 'Rank relevance');
});

it('preserves reference speech audio and rejects contradictory fixed voices', () => {
  expect(
    validateModelWorkspaceInput('speech-synthesis-v1', {
      text: 'Hello',
      audioArtifactId: artifactId,
    }),
  ).toEqual({
    text: 'Hello',
    audioArtifactId: artifactId,
    voice: 'reference',
    language: 'en',
    speed: 1,
  });
  expect(() =>
    validateModelWorkspaceInput('speech-synthesis-v1', { text: 'Hello', voice: 'reference' }),
  ).toThrow();
  expect(() =>
    validateModelWorkspaceInput('speech-synthesis-v1', {
      text: 'Hello',
      audioArtifactId: artifactId,
      voice: 'af_heart',
    }),
  ).toThrow();
});

it('preserves automatic multilingual speech without requiring reference audio', () => {
  expect(
    validateModelWorkspaceInput('speech-synthesis-v1', {
      text: 'Ahoj, svet!',
      voice: 'auto',
      language: 'sk',
      speed: 1,
    }),
  ).toEqual({ text: 'Ahoj, svet!', voice: 'auto', language: 'sk', speed: 1 });
  expect(() =>
    validateModelWorkspaceInput('speech-synthesis-v1', {
      text: 'Hello',
      voice: 'auto',
      language: 'en',
      audioArtifactId: artifactId,
    }),
  ).toThrow();
});

it('preserves automatic speech language selection without reference audio', () => {
  expect(
    validateModelWorkspaceInput('speech-synthesis-v1', {
      text: 'Hello! 早安！',
      voice: 'auto',
      language: 'auto',
      speed: 1,
    }),
  ).toEqual({ text: 'Hello! 早安！', voice: 'auto', language: 'auto', speed: 1 });
});

it('keeps codec encode and decode inputs distinct', () => {
  expect(
    validateModelWorkspaceInput('audio-transformation-v1', {
      operation: 'encode',
      audioArtifactId: artifactId,
      sampleRate: 24000,
    }),
  ).toEqual({ operation: 'encode', audioArtifactId: artifactId, sampleRate: 24000 });
  expect(
    validateModelWorkspaceInput('audio-transformation-v1', {
      operation: 'decode',
      codes: [Array(16).fill(1)],
      sampleRate: 24000,
    }),
  ).toEqual({ operation: 'decode', codes: [Array(16).fill(1)], sampleRate: 24000 });
  expect(() =>
    validateModelWorkspaceInput('audio-transformation-v1', {
      operation: 'decode',
      codes: [Array(16).fill(1)],
      audioArtifactId: artifactId,
    }),
  ).toThrow();
});
