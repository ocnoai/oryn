// Generated from oryn-protocol; do not hand-edit.
import type { ArtifactMetadata } from "./types.js";
/** Closed additive task contracts. Unknown tasks remain unavailable on older peers. */
export interface SharedTaskInputs {
  "unconditional-image-generation-v1": { "width"?: number; "height"?: number; "steps"?: number; "seed"?: number };
  "audio-question-answering-v1": { "audioArtifactId": string; "instruction": string; "maxTokens"?: number };
  "audio-transformation-v1": { "operation": "enhance" | "separate" | "voice-conversion" | "encode" | "decode" | "vocode"; "audioArtifactId"?: string; "referenceAudioArtifactId"?: string; "codes"?: ((number)[])[]; "spectrogram"?: ((number)[])[]; "sampleRate"?: number };
  "multimodal-generation-v1": { "instruction": string; "imageArtifactId"?: string; "audioArtifactId"?: string; "inputMode": "image" | "audio" | "image-and-audio"; "outputMode"?: "text" | "audio" | "text-and-audio"; "maxTokens"?: number };
  "image-text-to-video-v1": { "imageArtifactId": string; "prompt": string; "width"?: number; "height"?: number; "frames"?: number; "fps"?: number; "steps"?: number; "guidanceScale"?: number; "seed"?: number };
  "video-editing-v1": { "videoArtifactId": string; "prompt"?: string; "maskVideoArtifactId"?: string; "operation"?: "edit" | "restore"; "startSeconds"?: number; "durationSeconds"?: number; "width"?: number; "height"?: number; "frames"?: number; "fps"?: number; "steps"?: number; "guidanceScale"?: number; "seed"?: number };
  "tabular-classification-v1": { "columns": ({ "name": string; "type": "numeric" | "categorical" })[]; "trainingRows": ((number | string | null)[])[]; "queryRows": ((number | string | null)[])[]; "targets": (string)[] };
  "tabular-regression-v1": { "columns": ({ "name": string; "type": "numeric" | "categorical" })[]; "trainingRows": ((number | string | null)[])[]; "queryRows": ((number | string | null)[])[]; "targets": (number)[] };
  "time-series-forecasting-v1": { "context": (number)[]; "horizon"?: number; "quantiles"?: (number)[]; "timestamps"?: (number)[]; "frequencyMilliseconds"?: number; "pastCovariates"?: ((number)[])[]; "futureCovariates"?: ((number)[])[] };
  "pose-estimation-v1": { "imageArtifactId": string; "boxes": ((number)[])[] };
  "visual-document-retrieval-v1": { imageArtifactId: string; queries: string[] };
  "table-question-answering-v1": { question: string; columns: string[]; rows: string[][] };
  "document-question-answering-v1": { imageArtifactId: string; question: string; words: string[]; boxes: [number, number, number, number][] };
  "keypoint-detection-v1": { imageArtifactId: string; threshold?: number; maxKeypoints?: number };
  "video-classification-v1": { videoArtifactId: string; startSeconds?: number; durationSeconds?: number; topK?: number };
  "video-question-answering-v1": { videoArtifactId: string; question: string; startSeconds?: number; durationSeconds?: number; maxTokens?: number };
  "voice-activity-detection-v1": { audioArtifactId: string; threshold?: number; minSpeechSeconds?: number; minSilenceSeconds?: number };
  "image-editing-v1": { imageArtifactId: string; referenceImageArtifactIds?: string[]; prompt: string; negativePrompt?: string; width?: number; height?: number; steps?: number; guidanceScale?: number; seed?: number };
  "image-layers-v1": { imageArtifactId: string; prompt: string; negativePrompt?: string; resolution?: 640 | 1024; layers?: number; steps?: number; guidanceScale?: number; seed?: number };
  "zero-shot-classification-v1": { text: string; labels: string[]; multiLabel?: boolean };
  "image-embeddings-v1": { imageArtifactId: string; normalize?: boolean };
  "image-segmentation-v1": { imageArtifactId: string };
  "mask-generation-v1": { imageArtifactId: string; points: { x: number; y: number; label: 0 | 1 }[] };
  "zero-shot-object-detection-v1": { imageArtifactId: string; labels: string[]; threshold?: number };
  "image-to-text-v1": { imageArtifactId: string; maxTokens?: number };
  "visual-question-answering-v1": { imageArtifactId: string; question: string; maxTokens?: number };
}
export type SharedTaskContract = keyof SharedTaskInputs;
export const SHARED_TASK_CONTRACTS = [
  "unconditional-image-generation-v1",
  "audio-question-answering-v1",
  "audio-transformation-v1",
  "multimodal-generation-v1",
  "image-text-to-video-v1",
  "video-editing-v1",
  "tabular-classification-v1",
  "tabular-regression-v1",
  "time-series-forecasting-v1",
  "pose-estimation-v1",
  "visual-document-retrieval-v1",
  "table-question-answering-v1",
  "document-question-answering-v1",
  "keypoint-detection-v1",
  "video-classification-v1",
  "video-question-answering-v1",
  "voice-activity-detection-v1",
  "image-editing-v1",
  "image-layers-v1",
  "zero-shot-classification-v1",
  "image-embeddings-v1",
  "image-segmentation-v1",
  "mask-generation-v1",
  "zero-shot-object-detection-v1",
  "image-to-text-v1",
  "visual-question-answering-v1",
] as const;

export type SpatialMaskResult = { kind: "artifact"; artifact: ArtifactMetadata } | {
  kind: "multi";
  outputs: [{ kind: "artifact"; artifact: ArtifactMetadata }, { kind: "structured"; value: {
    width: number; height: number; encoding: "class-id-16bit" | "binary-8bit";
    segments: { id: number; label: string; pixelCount: number }[];
  } }];
};
/** Structured values, or the existing artifact/multi wrapper for spatial masks. */
export interface SharedTaskOutputs {
  "unconditional-image-generation-v1": { "kind": "artifact"; "artifact": ArtifactMetadata };
  "audio-question-answering-v1": { "text": string };
  "audio-transformation-v1": { "kind": "artifact"; "artifact": ArtifactMetadata } | { "kind": "multi"; "outputs": ({ "kind": "artifact"; "artifact": ArtifactMetadata })[] } | { "kind": "structured"; "value": { "codes": ((number)[])[]; "sampleRate": number; "durationSeconds": number } };
  "multimodal-generation-v1": { "kind": "structured"; "value": { "text": string } } | { "kind": "artifact"; "artifact": ArtifactMetadata } | { "kind": "multi"; "outputs": [{ "kind": "structured"; "value": { "text": string } }, { "kind": "artifact"; "artifact": ArtifactMetadata }] };
  "image-text-to-video-v1": { "kind": "artifact"; "artifact": ArtifactMetadata };
  "video-editing-v1": { "kind": "artifact"; "artifact": ArtifactMetadata };
  "tabular-classification-v1": { "classes": (string)[]; "predictions": ({ "label": string; "probabilities": (number)[] })[] };
  "tabular-regression-v1": { "predictions": (number)[] };
  "time-series-forecasting-v1": { "predictions": (number)[]; "quantiles": ({ "level": number; "values": (number)[] })[]; "timestamps"?: (number)[] };
  "pose-estimation-v1": { "width": number; "height": number; "poses": ({ "box": (number)[]; "keypoints": ({ "id": number; "label": string; "x": number; "y": number; "confidence": number; "visible": boolean })[] })[] };
  "visual-document-retrieval-v1": { dimensions: 128 | 320 | 4096; normalized: true; scoring: "sum-max-cosine" | "mean-max-cosine"; documentVectors: number[][]; queries: { vectors: number[][]; score: number }[] };
  "table-question-answering-v1": { answer: string; cells: { row: number; column: number; text: string }[]; aggregation: "none" | "sum" | "average" | "count" | "generated" };
  "document-question-answering-v1": { answer: string; score: number; wordStart: number; wordEnd: number; box: [number, number, number, number] };
  "keypoint-detection-v1": { width: number; height: number; keypoints: { x: number; y: number; score: number }[] };
  "video-classification-v1": { predictions: { label: string; score: number }[]; sampledTimestamps: number[]; durationSeconds: number };
  "video-question-answering-v1": { text: string; sampledTimestamps: number[]; durationSeconds: number };
  "voice-activity-detection-v1": { segments: { start: number; end: number }[]; durationSeconds: number };
  "image-editing-v1": { kind: "artifact"; artifact: ArtifactMetadata };
  "image-layers-v1": { kind: "multi"; outputs: { kind: "artifact"; artifact: ArtifactMetadata }[] };
  "image-segmentation-v1": SpatialMaskResult;
  "mask-generation-v1": SpatialMaskResult;
  "zero-shot-classification-v1": { predictions: { label: string; score: number }[]; multiLabel: boolean };
  "image-embeddings-v1": { vector: number[]; dimensions: number; normalized: boolean; pooling: "cls" };
  "zero-shot-object-detection-v1": { detections: { label: string; score: number; box: [number, number, number, number] }[]; width: number; height: number };
  "image-to-text-v1": { text: string };
  "visual-question-answering-v1": { text: string };
}
