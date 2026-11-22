import { matchesSchema as matches } from './schema-match.js';
import { SHARED_TASK_CONTRACTS, type SharedTaskContract, type SharedTaskInputs } from './shared-tasks.js';
import { sharedTaskSchemas } from './shared-task-schemas.js';
import { validTaskRelations } from './task-relations.js';

export const SHARED_MODEL_TASK_CONTRACTS: Readonly<Record<string, SharedTaskContract>> = Object.freeze({
  'visual-document-retrieval': 'visual-document-retrieval-v1',
  'zero-shot-classification': 'zero-shot-classification-v1',
  'image-feature-extraction': 'image-embeddings-v1',
  'image-segmentation': 'image-segmentation-v1',
  'mask-generation': 'mask-generation-v1',
  'zero-shot-object-detection': 'zero-shot-object-detection-v1',
  'image-to-text': 'image-to-text-v1',
  'visual-question-answering': 'visual-question-answering-v1',
  'table-question-answering': 'table-question-answering-v1',
  'document-question-answering': 'document-question-answering-v1',
  'keypoint-detection': 'keypoint-detection-v1',
  'video-classification': 'video-classification-v1',
  'video-text-to-text': 'video-question-answering-v1',
  'voice-activity-detection': 'voice-activity-detection-v1',
  'unconditional-image-generation': 'unconditional-image-generation-v1',
  'audio-text-to-text': 'audio-question-answering-v1',
  'audio-to-audio': 'audio-transformation-v1',
  'any-to-any': 'multimodal-generation-v1',
  'image-text-to-video': 'image-text-to-video-v1',
  'video-to-video': 'video-editing-v1',
  'tabular-classification': 'tabular-classification-v1',
  'tabular-regression': 'tabular-regression-v1',
  'time-series-forecasting': 'time-series-forecasting-v1',
  'pose-estimation': 'pose-estimation-v1',
});

/** Canonical data validation only. Browser still owns permissions and artifact leases. */
export function validateSharedTaskInput<C extends SharedTaskContract>(contract: C, input: unknown): SharedTaskInputs[C] {
  if (!(SHARED_TASK_CONTRACTS as readonly string[]).includes(contract) || !record(input) ||
      !matches(sharedTaskSchemas[contract], input) || !validTaskRelations(contract, input)) invalid();
  if (Array.isArray(input.labels) && new Set(input.labels.map(v => String(v).trim())).size !== input.labels.length) invalid();
  if (new TextEncoder().encode(JSON.stringify(input)).byteLength > 256 * 1024) invalid();
  return input as SharedTaskInputs[C];
}

/** Closed subset emitted by Protocol; unsupported schema keywords fail export. */
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
}
function invalid(): never { throw new TypeError('Task input is invalid or unavailable.'); }
