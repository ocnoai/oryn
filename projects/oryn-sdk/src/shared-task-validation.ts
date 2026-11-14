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
function matches(raw: unknown, value: unknown): boolean {
  if (!record(raw)) return false;
  if ('const' in raw && value !== raw.const) return false;
  if (Array.isArray(raw.enum) && !raw.enum.includes(value)) return false;
  if (Array.isArray(raw.anyOf)) return raw.anyOf.some(schema => matches(schema, value));
  switch (raw.type) {
    case 'object': {
      if (!record(value) || (raw.properties !== undefined && !record(raw.properties))) return false;
      const properties = (raw.properties ?? {}) as Record<string, unknown>;
      if (raw.additionalProperties === false && Object.keys(value).some(key => !(Object.hasOwn(properties, key)))) return false;
      if (Array.isArray(raw.required) && raw.required.some(key => typeof key !== 'string' || !Object.hasOwn(value, key))) return false;
      return Object.entries(value).every(([key, item]) => !Object.hasOwn(properties, key) || matches(properties[key], item));
    }
    case 'array': {
      if (!Array.isArray(value) || value.length < Number(raw.minItems ?? 0) || value.length > Number(raw.maxItems ?? Infinity)) return false;
      if (raw.uniqueItems && new Set(value.map(item => JSON.stringify(item))).size !== value.length) return false;
      if (raw.contains && value.filter(item => matches(raw.contains, item)).length < Number(raw.minContains ?? 1)) return false;
      return value.every(item => matches(raw.items, item));
    }
    case 'string':
      return typeof value === 'string' && [...value].length >= Number(raw.minLength ?? 0) &&
        [...value].length <= Number(raw.maxLength ?? Infinity) &&
        (typeof raw.pattern !== 'string' || new RegExp(raw.pattern, 'u').test(value));
    case 'integer':
    case 'number':
      return typeof value === 'number' && Number.isFinite(value) &&
        (raw.type !== 'integer' || Number.isInteger(value)) &&
        value >= Number(raw.minimum ?? -Infinity) && value <= Number(raw.maximum ?? Infinity) &&
        value > Number(raw.exclusiveMinimum ?? -Infinity) &&
        (raw.multipleOf === undefined || value % Number(raw.multipleOf) === 0);
    case 'boolean': return typeof value === 'boolean';
    case 'null': return value === null;
    case undefined: return 'const' in raw || Array.isArray(raw.enum);
    default: return false;
  }
}
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
}
function invalid(): never { throw new TypeError('Task input is invalid or unavailable.'); }
