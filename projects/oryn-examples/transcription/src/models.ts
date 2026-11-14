import type { ModelRequest } from '@ocno/oryn-sdk';

export interface TranscriptionLanguage {
  readonly id: string;
  readonly label: string;
}

export interface TranscriptionModel {
  readonly id: string;
  readonly label: string;
  readonly repository: string;
  readonly description: string;
  readonly revision?: string;
}

export interface TranscriptionSelection {
  language: TranscriptionLanguage;
  model: TranscriptionModel;
  inputLanguage: string;
}

// Language tokens, rather than an accuracy guarantee:
// https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
// https://huggingface.co/openai/whisper-small/blob/main/generation_config.json
// https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/generation_config.json
// Cantonese has a dedicated token only in the large-v3 models offered here.
const LANGUAGE_LABELS: Record<string, string> = {
  af: 'Afrikaans', am: 'Amharic', ar: 'Arabic', as: 'Assamese', az: 'Azerbaijani',
  ba: 'Bashkir', be: 'Belarusian', bg: 'Bulgarian', bn: 'Bengali', bo: 'Tibetan',
  br: 'Breton', bs: 'Bosnian', ca: 'Catalan', cs: 'Czech', cy: 'Welsh',
  da: 'Danish', de: 'German', el: 'Greek', en: 'English', es: 'Spanish',
  et: 'Estonian', eu: 'Basque', fa: 'Persian', fi: 'Finnish', fo: 'Faroese',
  fr: 'French', gl: 'Galician', gu: 'Gujarati', ha: 'Hausa', haw: 'Hawaiian',
  he: 'Hebrew', hi: 'Hindi', hr: 'Croatian', ht: 'Haitian Creole', hu: 'Hungarian',
  hy: 'Armenian', id: 'Indonesian', is: 'Icelandic', it: 'Italian', ja: 'Japanese',
  jw: 'Javanese', ka: 'Georgian', kk: 'Kazakh', km: 'Khmer', kn: 'Kannada',
  ko: 'Korean', la: 'Latin', lb: 'Luxembourgish', ln: 'Lingala', lo: 'Lao',
  lt: 'Lithuanian', lv: 'Latvian', mg: 'Malagasy', mi: 'Maori', mk: 'Macedonian',
  ml: 'Malayalam', mn: 'Mongolian', mr: 'Marathi', ms: 'Malay', mt: 'Maltese',
  my: 'Myanmar (Burmese)', ne: 'Nepali', nl: 'Dutch', nn: 'Norwegian Nynorsk', no: 'Norwegian',
  oc: 'Occitan', pa: 'Punjabi', pl: 'Polish', ps: 'Pashto', pt: 'Portuguese',
  ro: 'Romanian', ru: 'Russian', sa: 'Sanskrit', sd: 'Sindhi', si: 'Sinhala',
  sk: 'Slovak', sl: 'Slovenian', sn: 'Shona', so: 'Somali', sq: 'Albanian',
  sr: 'Serbian', su: 'Sundanese', sv: 'Swedish', sw: 'Swahili', ta: 'Tamil',
  te: 'Telugu', tg: 'Tajik', th: 'Thai', tk: 'Turkmen', tl: 'Filipino (Tagalog)',
  tr: 'Turkish', tt: 'Tatar', uk: 'Ukrainian', ur: 'Urdu', uz: 'Uzbek',
  vi: 'Vietnamese', yi: 'Yiddish', yo: 'Yoruba', yue: 'Cantonese', zh: 'Chinese (Mandarin)',
};

export const LANGUAGES: readonly TranscriptionLanguage[] = Object.freeze([
  Object.freeze({ id: 'auto', label: 'Detect automatically' }),
  ...Object.entries(LANGUAGE_LABELS)
    .map(([id, label]) => Object.freeze({ id, label }))
    .sort((left, right) => left.label.localeCompare(right.label, 'en')),
]);

// The named language list is shared by both official Qwen3-ASR checkpoints:
// https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf#available-checkpoints
// https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf#available-checkpoints
// Use the processor's official names; Whisper uses its own language token codes.
const QWEN_LANGUAGE_NAMES: Record<string, string> = {
  zh: 'Chinese', en: 'English', yue: 'Cantonese', ar: 'Arabic', de: 'German',
  fr: 'French', es: 'Spanish', pt: 'Portuguese', id: 'Indonesian', it: 'Italian',
  ko: 'Korean', ru: 'Russian', th: 'Thai', vi: 'Vietnamese', ja: 'Japanese',
  tr: 'Turkish', hi: 'Hindi', ms: 'Malay', nl: 'Dutch', sv: 'Swedish',
  da: 'Danish', fi: 'Finnish', pl: 'Polish', cs: 'Czech', tl: 'Filipino',
  fa: 'Persian', el: 'Greek', hu: 'Hungarian', mk: 'Macedonian', ro: 'Romanian',
};

const MODELS: readonly TranscriptionModel[] = [
  {
    id: 'whisper-small', label: 'Whisper Small · 244M', repository: 'openai/whisper-small',
    description: 'Default multilingual model with 244 million parameters.',
  },
  {
    id: 'whisper-tiny', label: 'Whisper Tiny · 39M', repository: 'openai/whisper-tiny',
    description: 'Smallest model in this list, with 39 million parameters.',
  },
  {
    id: 'whisper-large-v3-turbo', label: 'Whisper Large v3 Turbo · 809M', repository: 'openai/whisper-large-v3-turbo',
    description: 'Multilingual large-v3 model with a reduced decoder and 809 million parameters.',
  },
  {
    id: 'whisper-large-v3', label: 'Whisper Large v3 · 1.55B', repository: 'openai/whisper-large-v3',
    description: 'Full multilingual large-v3 model with 1.55 billion parameters.',
  },
  {
    id: 'whisper-small.en', label: 'Whisper Small English · 244M', repository: 'openai/whisper-small.en',
    description: 'English-only model with 244 million parameters.',
  },
  {
    id: 'qwen3-asr-0.6b', label: 'Qwen3 ASR · 0.6B', repository: 'Qwen/Qwen3-ASR-0.6B-hf',
    description: 'Qwen speech recognition model with 0.6 billion parameters, for 30 named languages.',
  },
  {
    id: 'qwen3-asr-1.7b', label: 'Qwen3 ASR · 1.7B', repository: 'Qwen/Qwen3-ASR-1.7B-hf',
    revision: 'bcd2b5b7f32b480ab5790554cfa8347f246a14f3',
    description: 'Qwen speech recognition model with 1.7 billion parameters, for 30 named languages.',
  },
];

function languageForId(languageId: string): TranscriptionLanguage {
  const language = LANGUAGES.find(candidate => candidate.id === languageId);
  if (!language) throw new Error('Choose a language from the recording language list.');
  return language;
}

export function modelsForLanguage(languageId: string): TranscriptionModel[] {
  languageForId(languageId);
  return MODELS.filter(model => {
    if (model.id === 'whisper-small.en') return languageId === 'en';
    if (model.repository.startsWith('Qwen/')) {
      // Unknown or mixed-language recordings use the broader Whisper language set.
      return languageId !== 'auto' && Object.hasOwn(QWEN_LANGUAGE_NAMES, languageId);
    }
    if (languageId === 'yue') return model.id === 'whisper-large-v3' || model.id === 'whisper-large-v3-turbo';
    return true;
  });
}

export function defaultModelForLanguage(languageId: string): TranscriptionModel {
  const defaultId = languageId === 'yue' ? 'whisper-large-v3-turbo' : 'whisper-small';
  return modelsForLanguage(languageId).find(model => model.id === defaultId)!;
}

export function getSelection(languageId: string, modelId: string): TranscriptionSelection {
  const language = languageForId(languageId);
  const model = modelsForLanguage(languageId).find(candidate => candidate.id === modelId);
  if (!model) throw new Error('Choose a transcription model available for the selected recording language.');
  return {
    language,
    model,
    inputLanguage: model.id === 'whisper-small.en' ? 'auto'
      : model.repository.startsWith('Qwen/') ? QWEN_LANGUAGE_NAMES[languageId]!
      : languageId,
  };
}

export function buildModelRequest(
  selection: TranscriptionSelection,
  taskProfiles: boolean,
  modelSelection = false,
): ModelRequest {
  // Recheck the selected IDs so unsupported pairs never become a model request.
  const checked = getSelection(selection.language.id, selection.model.id);
  return {
    role: 'transcription',
    ...(modelSelection ? { selection: { version: 1, intent: 'preferred-variant' } as const } : {}),
    preferredModel: {
      provider: 'huggingface',
      repository: checked.model.repository,
      ...(checked.model.revision ? { revision: checked.model.revision } : {}),
    },
    substitution: { automatic: 'none' },
    requirements: {
      task: 'automatic-speech-recognition',
      inputModalities: ['audio'],
      outputModalities: ['text'],
      ...(taskProfiles ? {
        contract: { id: 'transcription-v1', revision: 1 },
        requiredFeatures: ['speechRecognition'],
        parameterValues: { language: checked.inputLanguage, timestamps: false },
      } : {}),
    },
  };
}
