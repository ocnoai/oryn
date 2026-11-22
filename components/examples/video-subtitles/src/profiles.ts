export interface ModelIdentity {
  provider: "huggingface";
  repository: string;
  revision: string;
}

export interface TranscriptionProfile {
  role: string;
  preferredModel: ModelIdentity;
  certifiedModels: ModelIdentity[];
  windowSeconds: number;
  alignmentModel: ModelIdentity;
  nativeTimingFallbackModel?: ModelIdentity;
}

const QWEN_ASR: ModelIdentity = {
  provider: "huggingface",
  repository: "Qwen/Qwen3-ASR-1.7B-hf",
  revision: "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
};

export const WHISPER_TIMED_ASR: ModelIdentity = {
  provider: "huggingface",
  repository: "openai/whisper-large-v3",
  revision: "06f233fe06e710322aca913c1bc4249a0d71fce1",
};

const QWEN_FORCED_ALIGNER: ModelIdentity = {
  provider: "huggingface",
  repository: "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  revision: "c07281df297b9905d24a508279258cccf987a064",
};

export function transcriptionProfile(
  intent: "quality" | "faster",
): TranscriptionProfile {
  if (intent === "quality") {
    return {
      role: "video-subtitle-timed-transcription",
      preferredModel: QWEN_ASR,
      certifiedModels: [WHISPER_TIMED_ASR],
      windowSeconds: 12,
      alignmentModel: QWEN_FORCED_ALIGNER,
      nativeTimingFallbackModel: WHISPER_TIMED_ASR,
    };
  }
  return {
    role: "video-subtitle-fast-transcription",
    preferredModel: QWEN_ASR,
    certifiedModels: [WHISPER_TIMED_ASR],
    windowSeconds: 15,
    alignmentModel: QWEN_FORCED_ALIGNER,
  };
}
