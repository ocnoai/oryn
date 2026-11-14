import {
  isSharedTaskContract,
  validateSharedTaskInput,
  SHARED_TASK_CONTRACTS,
} from '@ocno/oryn-protocol';
import type { ModelWorkspaceTaskContract } from '../../shared/ipc';

export const MODEL_WORKSPACE_TASK_CONTRACTS: readonly ModelWorkspaceTaskContract[] = Object.freeze([
  ...SHARED_TASK_CONTRACTS,
  'chat-v1',
  'text-to-image-v1',
  'image-to-image-v1',
  'text-to-music-v1',
  'speech-synthesis-v1',
  'transcription-v1',
  'media-audio-extract-v1',
  'audio-source-separation-v1',
  'forced-alignment-v1',
  'subtitle-export-v1',
  'embeddings-v1',
  'reranking-v1',
  'text-to-video-v1',
  'image-to-video-v1',
  'text-classification-v1',
  'token-classification-v1',
  'question-answering-v1',
  'fill-mask-v1',
  'translation-v1',
  'summarization-v1',
  'image-classification-v1',
  'audio-classification-v1',
  'object-detection-v1',
  'depth-estimation-v1',
  'zero-shot-image-classification-v1',
]);

const ARTIFACT_ID = /^artifact_[A-Za-z0-9_-]{20,80}$/u;

export function isModelWorkspaceTaskContract(value: string): value is ModelWorkspaceTaskContract {
  return (MODEL_WORKSPACE_TASK_CONTRACTS as readonly string[]).includes(value);
}

/** Browser-side defense in depth; Engine remains the authoritative task validator. */
export function validateModelWorkspaceInput(
  contract: ModelWorkspaceTaskContract,
  value: unknown,
  capabilities?: Record<string, unknown>,
): unknown {
  const input = record(value);
  if (
    contract === 'text-to-image-v1' &&
    input.imageArtifactId !== undefined &&
    capabilities?.instructionEditing === true
  ) {
    contract = 'image-editing-v1';
  }
  if (isSharedTaskContract(contract)) {
    try {
      const checked = validateSharedTaskInput(contract, input);
      if (
        Array.isArray(input.labels) &&
        new Set(input.labels.map((label) => String(label).trim())).size !== input.labels.length
      )
        invalid();
      if (
        contract === 'mask-generation-v1' &&
        Array.isArray(input.points) &&
        !input.points.some((point) => record(point).label === 1)
      )
        invalid();
      return checked;
    } catch {
      invalid();
    }
  }
  switch (contract) {
    case 'chat-v1': {
      exactKeys(input, ['messages', 'maxTokens', 'temperature', 'topP', 'imageArtifactId']);
      if (input.imageArtifactId !== undefined && !artifactId(input.imageArtifactId)) invalid();
      if (
        !Array.isArray(input.messages) ||
        input.messages.length < 1 ||
        input.messages.length > 128
      )
        invalid();
      let total = 0;
      const messages = input.messages.map((message) => {
        const item = record(message);
        exactKeys(item, ['role', 'content'], true);
        if (
          !['system', 'user', 'assistant'].includes(String(item.role)) ||
          !boundedString(item.content, 1, 32_768)
        )
          invalid();
        total += item.content.length;
        return { role: item.role as 'system' | 'user' | 'assistant', content: item.content };
      });
      if (total > 65_536 || !messages.some((item) => item.role === 'user')) invalid();
      return {
        messages,
        ...(input.imageArtifactId !== undefined ? { imageArtifactId: input.imageArtifactId } : {}),
        maxTokens: boundedInteger(input.maxTokens ?? 128, 1, 512),
        temperature: boundedNumber(input.temperature ?? 0.7, 0, 2),
        topP: boundedNumber(input.topP ?? 0.9, Number.EPSILON, 1),
      };
    }
    case 'text-to-image-v1':
    case 'image-to-image-v1':
    case 'text-to-video-v1':
    case 'image-to-video-v1':
      return validateMedia(contract, input);
    case 'text-to-music-v1': {
      exactKeys(input, ['prompt', 'lyrics', 'durationSeconds', 'seed']);
      if (!boundedString(input.prompt, 1, 2_000)) invalid();
      if (input.lyrics !== undefined && !boundedString(input.lyrics, 0, 12_000)) invalid();
      return {
        prompt: input.prompt.trim(),
        ...(input.lyrics !== undefined ? { lyrics: input.lyrics } : {}),
        durationSeconds: boundedNumber(input.durationSeconds ?? 10, 1, 180),
        ...(input.seed !== undefined ? { seed: boundedInteger(input.seed, -1, 2 ** 32 - 1) } : {}),
      };
    }
    case 'speech-synthesis-v1': {
      exactKeys(input, ['text', 'voice', 'language', 'speed', 'audioArtifactId']);
      const reference = input.audioArtifactId !== undefined;
      if (
        (reference && !artifactId(input.audioArtifactId)) ||
        ((input.voice ?? (reference ? 'reference' : 'af_heart')) === 'reference') !== reference
      )
        invalid();
      if (
        !boundedString(input.text, 1, 350) ||
        !boundedString(input.voice ?? 'af_heart', 1, 64) ||
        !boundedString(input.language ?? 'en-us', 1, 32)
      )
        invalid();
      return {
        text: input.text.trim(),
        voice: input.voice ?? (reference ? 'reference' : 'af_heart'),
        language: input.language ?? (reference ? 'en' : 'en-us'),
        ...(reference ? { audioArtifactId: input.audioArtifactId } : {}),
        speed: boundedNumber(input.speed ?? 1, 0.5, 2),
      };
    }
    case 'transcription-v1':
      exactKeys(input, ['audioArtifactId', 'language', 'timestamps']);
      if (
        !artifactId(input.audioArtifactId) ||
        !boundedString(input.language ?? 'auto', 1, 32) ||
        (input.timestamps !== undefined && typeof input.timestamps !== 'boolean')
      )
        invalid();
      return {
        audioArtifactId: input.audioArtifactId,
        language: input.language ?? 'auto',
        timestamps: input.timestamps ?? false,
      };
    case 'media-audio-extract-v1':
      exactKeys(input, ['videoArtifactId', 'track'], true);
      if (!artifactId(input.videoArtifactId) || input.track !== 'default') invalid();
      return { videoArtifactId: input.videoArtifactId, track: 'default' as const };
    case 'audio-source-separation-v1':
      exactKeys(input, ['audioArtifactId', 'target'], true);
      if (!artifactId(input.audioArtifactId) || input.target !== 'vocals') invalid();
      return { audioArtifactId: input.audioArtifactId, target: 'vocals' as const };
    case 'forced-alignment-v1':
      exactKeys(input, ['audioArtifactId', 'transcript', 'language', 'timestamps'], true);
      if (
        !artifactId(input.audioArtifactId) ||
        !boundedString(input.transcript, 1, 262_144) ||
        !boundedString(input.language, 1, 32) ||
        !['word', 'segment'].includes(String(input.timestamps))
      )
        invalid();
      return {
        audioArtifactId: input.audioArtifactId,
        transcript: input.transcript.trim(),
        language: input.language,
        timestamps: input.timestamps as 'word' | 'segment',
      };
    case 'subtitle-export-v1': {
      exactKeys(input, ['format', 'segments'], true);
      if (input.format !== 'subrip' || !Array.isArray(input.segments)) invalid();
      if (input.segments.length < 1 || input.segments.length > 20_000) invalid();
      let previousEnd = 0;
      let totalText = 0;
      const segments = input.segments.map((segment) => {
        const item = record(segment);
        exactKeys(item, ['startMs', 'endMs', 'text'], true);
        const startMs = boundedInteger(item.startMs, 0, 14_400_000);
        const endMs = boundedInteger(item.endMs, 1, 14_400_000);
        if (startMs < previousEnd || endMs <= startMs || !boundedString(item.text, 1, 2_000))
          invalid();
        previousEnd = endMs;
        totalText += item.text.length;
        return { startMs, endMs, text: item.text.trim() };
      });
      if (totalText > 2_000_000) invalid();
      return { format: 'subrip' as const, segments };
    }
    case 'embeddings-v1':
      exactKeys(input, ['texts', 'normalize']);
      if (
        !Array.isArray(input.texts) ||
        input.texts.length < 1 ||
        input.texts.length > 64 ||
        input.texts.some((item) => !boundedString(item, 1, 8_192)) ||
        input.texts.reduce((sum, item) => sum + item.length, 0) > 65_536 ||
        (input.normalize !== undefined && typeof input.normalize !== 'boolean')
      )
        invalid();
      return { texts: input.texts.map((item) => item.trim()), normalize: input.normalize ?? true };
    case 'reranking-v1': {
      exactKeys(input, ['query', 'documents', 'instruction', 'topK']);
      const instruction = input.instruction;
      if (
        !boundedString(input.query, 1, 8_192) ||
        !Array.isArray(input.documents) ||
        input.documents.length < 1 ||
        input.documents.length > 64 ||
        input.documents.some((item) => !boundedString(item, 1, 16_384)) ||
        (instruction !== undefined && !boundedString(instruction, 1, 2_048))
      )
        invalid();
      return {
        query: input.query.trim(),
        documents: input.documents.map((item) => item.trim()),
        ...(typeof instruction === 'string' ? { instruction: instruction.trim() } : {}),
        topK: boundedInteger(input.topK ?? input.documents.length, 1, input.documents.length),
      };
    }
    case 'text-classification-v1':
    case 'fill-mask-v1':
      exactKeys(input, ['text', 'topK']);
      if (!boundedString(input.text, 1, 32_768)) invalid();
      return { text: input.text.trim(), topK: boundedInteger(input.topK ?? 5, 1, 20) };
    case 'token-classification-v1':
      exactKeys(input, ['text']);
      if (!boundedString(input.text, 1, 32_768)) invalid();
      return { text: input.text.trim() };
    case 'question-answering-v1':
      exactKeys(input, ['question', 'context', 'maxAnswerLength']);
      if (!boundedString(input.question, 1, 4_096) || !boundedString(input.context, 1, 65_536))
        invalid();
      return {
        question: input.question.trim(),
        context: input.context.trim(),
        maxAnswerLength: boundedInteger(input.maxAnswerLength ?? 64, 1, 128),
      };
    case 'translation-v1':
    case 'summarization-v1':
      exactKeys(input, ['text', 'maxTokens']);
      if (!boundedString(input.text, 1, 32_768)) invalid();
      return { text: input.text.trim(), maxTokens: boundedInteger(input.maxTokens ?? 256, 1, 512) };
    case 'image-classification-v1':
      exactKeys(input, ['imageArtifactId', 'topK']);
      if (!artifactId(input.imageArtifactId)) invalid();
      return {
        imageArtifactId: input.imageArtifactId,
        topK: boundedInteger(input.topK ?? 5, 1, 20),
      };
    case 'audio-classification-v1':
      exactKeys(input, ['audioArtifactId', 'topK', 'classificationHead']);
      if (
        input.classificationHead !== undefined &&
        !['language', 'gender'].includes(input.classificationHead as string)
      )
        invalid();
      if (!artifactId(input.audioArtifactId)) invalid();
      return {
        audioArtifactId: input.audioArtifactId,
        ...(input.classificationHead === undefined
          ? {}
          : { classificationHead: input.classificationHead }),
        topK: boundedInteger(input.topK ?? 5, 1, 20),
      };
    case 'object-detection-v1':
      exactKeys(input, ['imageArtifactId', 'threshold']);
      if (!artifactId(input.imageArtifactId)) invalid();
      return {
        imageArtifactId: input.imageArtifactId,
        threshold: boundedNumber(input.threshold ?? 0.5, 0, 1),
      };
    case 'depth-estimation-v1':
      exactKeys(input, ['imageArtifactId']);
      if (!artifactId(input.imageArtifactId)) invalid();
      return { imageArtifactId: input.imageArtifactId };
    case 'zero-shot-image-classification-v1':
      exactKeys(input, ['imageArtifactId', 'labels'], true);
      if (
        !artifactId(input.imageArtifactId) ||
        !Array.isArray(input.labels) ||
        input.labels.length < 2 ||
        input.labels.length > 64 ||
        input.labels.some((item) => !boundedString(item, 1, 128))
      )
        invalid();
      return {
        imageArtifactId: input.imageArtifactId,
        labels: input.labels.map((item) => item.trim()),
      };
  }
}

function validateMedia(
  contract: ModelWorkspaceTaskContract,
  input: Record<string, unknown>,
): unknown {
  if (contract === 'text-to-image-v1' && input.imageArtifactId !== undefined)
    contract = 'image-to-image-v1';
  const video = contract.includes('video');
  const needsImage = contract.startsWith('image-to-');
  const allowed = ['prompt', 'negativePrompt', 'width', 'height', 'steps', 'guidanceScale', 'seed'];
  if (video) allowed.push('frames', 'fps');
  if (needsImage) allowed.push('imageArtifactId');
  if (contract === 'image-to-image-v1') allowed.push('strength');
  exactKeys(input, allowed);
  if (
    !boundedString(input.prompt, 1, 4_096) ||
    (input.negativePrompt !== undefined && !boundedString(input.negativePrompt, 0, 4_096))
  )
    invalid();
  const width = boundedInteger(input.width ?? 512, 64, video ? 1_280 : 2_048);
  const height = boundedInteger(input.height ?? 512, 64, video ? 1_280 : 2_048);
  if (width % 8 || height % 8 || width * height > (video ? 1_280 * 720 : 2_048 * 2_048)) invalid();
  const result: Record<string, unknown> = {
    prompt: input.prompt.trim(),
    negativePrompt: input.negativePrompt ?? '',
    width,
    height,
    steps: boundedInteger(input.steps ?? 20, 1, 100),
    guidanceScale: boundedNumber(input.guidanceScale ?? 7.5, 0, 30),
    seed: boundedInteger(input.seed ?? 0, 0, 2 ** 32 - 1),
  };
  if (video) {
    const frames = boundedInteger(input.frames ?? 16, 1, 120);
    const fps = boundedInteger(input.fps ?? 8, 1, 60);
    if (frames / fps > 20 || width * height * frames > 96 * 1024 * 1024) invalid();
    Object.assign(result, { frames, fps });
  }
  if (needsImage) {
    if (!artifactId(input.imageArtifactId)) invalid();
    result.imageArtifactId = input.imageArtifactId;
  }
  if (contract === 'image-to-image-v1')
    result.strength = boundedNumber(input.strength ?? 0.8, 0.05, 1);
  return result;
}

function record(value: unknown): Record<string, unknown> {
  if (
    typeof value !== 'object' ||
    value === null ||
    Array.isArray(value) ||
    Object.getPrototypeOf(value) !== Object.prototype
  )
    invalid();
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, allowed: string[], requireAll = false): void {
  const keys = Object.keys(value);
  if (
    keys.some((key) => !allowed.includes(key)) ||
    (requireAll && allowed.some((key) => !keys.includes(key)))
  )
    invalid();
}

function boundedString(value: unknown, minimum: number, maximum: number): value is string {
  return typeof value === 'string' && value.trim().length >= minimum && value.length <= maximum;
}

function boundedInteger(value: unknown, minimum: number, maximum: number): number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  )
    invalid();
  return value;
}

function boundedNumber(value: unknown, minimum: number, maximum: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum)
    invalid();
  return value;
}

function artifactId(value: unknown): value is string {
  return typeof value === 'string' && ARTIFACT_ID.test(value);
}

function invalid(): never {
  throw new Error('MODEL_WORKSPACE_INPUT_INVALID');
}
