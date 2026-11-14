/** Cross-field semantics shared by Protocol and generated SDK data validation. */
import { validSprint61Input, validSprint61Output } from "./sprint61-relations.js";

export function validTaskRelations(contract: string, input: Record<string, unknown>): boolean {
  if (!validSprint61Input(contract, input)) return false;
  if (contract === "embeddings-v1" && input.inputType === "query" && input.contextual === true)
    return false;
  if (contract === "table-question-answering-v1") {
    const columns = input.columns as string[];
    return new Set(columns.map((v) => v.trim())).size === columns.length &&
      (input.rows as string[][]).every((row) => row.length === columns.length);
  }
  if (contract === "document-question-answering-v1") {
    const boxes = input.boxes as [number, number, number, number][];
    return boxes.length === (input.words as string[]).length &&
      boxes.every(([x0, y0, x1, y1]) => x0 <= x1 && y0 <= y1);
  }
  if (contract === "video-classification-v1" || contract === "video-question-answering-v1") {
    return Number(input.startSeconds ?? 0) < 30 &&
      Number(input.startSeconds ?? 0) + Number(input.durationSeconds ?? 0) <= 30;
  }
  return true;
}

export function validTaskOutputRelations(contract: string, output: Record<string, unknown>): boolean {
  if (!validSprint61Output(contract, output)) return false;
  if (contract === "visual-document-retrieval-v1") {
    const document = output.documentVectors as number[][];
    const queries = output.queries as {vectors: number[][]; score: number}[];
    if (![...document, ...queries.flatMap(q => q.vectors)].every(v => v.length === output.dimensions)) return false;
    if (![...document, ...queries.flatMap(q => q.vectors)].every(v => Math.abs(v.reduce((sum, x) => sum + x*x, 0) - 1) <= 0.0001)) return false;
    return queries.every(q => {
      const sum = q.vectors.reduce((total, v) => total + Math.max(...document.map(d => d.reduce((dot, x, i) => dot + x*v[i]!, 0))), 0);
      const score = output.scoring === "mean-max-cosine" ? sum / q.vectors.length : sum;
      return Math.abs(q.score - score) <= 0.001;
    });
  }
  if ((contract === "mask-generation-v1" || contract === "image-segmentation-v1") && output.kind === "multi") {
    const metadata = (output.outputs as [unknown, {value: {width: number; height: number; segments: {id: number; pixelCount: number}[]}}])[1].value;
    return new Set(metadata.segments.map(s => s.id)).size === metadata.segments.length &&
      metadata.segments.reduce((sum, s) => sum + s.pixelCount, 0) === metadata.width * metadata.height &&
      (contract !== "mask-generation-v1" || metadata.segments.every(s => s.id === 0 || s.id === 255));
  }
  if (contract === "document-question-answering-v1") {
    const [x0, y0, x1, y1] = output.box as [number, number, number, number];
    return Number(output.wordStart) <= Number(output.wordEnd) && x0 <= x1 && y0 <= y1;
  }
  if (contract === "keypoint-detection-v1") {
    return (output.keypoints as Array<{x: number; y: number}>).every(p => p.x <= Number(output.width) && p.y <= Number(output.height));
  }
  if (contract === "video-classification-v1" || contract === "video-question-answering-v1") {
    const times = output.sampledTimestamps as number[];
    return new Set(times).size >= 2 && times.every((t, i) => t < Number(output.durationSeconds) && (i === 0 || t >= times[i - 1]!));
  }
  if (contract === "voice-activity-detection-v1") {
    const segments = output.segments as Array<{start: number; end: number}>;
    return segments.every((s, i) => s.start < s.end && s.end <= Number(output.durationSeconds) && (i === 0 || segments[i - 1]!.end <= s.start));
  }
  return true;
}
