import {type Cue, type TranscriptSegment, serializeSubRip} from './subtitles.ts';
import {hasWordTiming} from './video-renderer.ts';

/** Validate supplied timings without inventing or redistributing word boundaries. */
export function withWordBounds(cue: Cue, bounds: Array<{startMs: number; endMs: number}>): Cue {
  serializeSubRip([cue]);
  const words = cue.text.trim().split(/\s+/u);
  if (words.length > 256 || words.length !== bounds.length) throw new Error('Supply one start and end for every word (maximum 256 words).');
  const {wordTimingHints: _hints, ...base} = cue;
  const repaired = {...base, spans: words.map((text, index) => ({text, ...bounds[index]!}))};
  if (!hasWordTiming(repaired) || bounds.some(span => !Number.isSafeInteger(span.startMs) || !Number.isSafeInteger(span.endMs)))
    throw new Error('Use whole milliseconds: each word must end after it starts, stay inside this caption, and not overlap the next word.');
  return repaired;
}

export interface WordBounds { startMs: number; endMs: number }
export type PartialWordTimings = Array<WordBounds | undefined>;
const wordKey = (text: string) => text.normalize('NFC').toLocaleLowerCase().replace(/[^\p{L}\p{N}]/gu, '');

/** Reconcile corrected words with measured word spans. Keep partial matches when
 * insertions/splits have no measured boundary; never divide a phrase evenly. */
export function recoverWordTimings(cue: Cue, measured: NonNullable<Cue['spans']>): PartialWordTimings {
  const words = cue.text.trim().split(/\s+/u);
  const result: PartialWordTimings = Array.from({length: words.length});
  if (words.length > 256) return result;
  const spans = measured.filter(span => span.text.trim() && !/\s/u.test(span.text.trim()) &&
    Number.isSafeInteger(span.startMs) && Number.isSafeInteger(span.endMs) &&
    span.startMs >= cue.startMs && span.endMs <= cue.endMs && span.endMs > span.startMs);
  if (!spans.length || spans.length > 512 || spans.some((span, index) => index > 0 && span.startMs < spans[index - 1]!.endMs)) return result;
  const old = spans.map(span => wordKey(span.text)), current = words.map(wordKey);
  const width = words.length + 1, scores = new Uint16Array((spans.length + 1) * width);
  for (let i = spans.length - 1; i >= 0; i--) for (let j = words.length - 1; j >= 0; j--) {
    scores[i * width + j] = old[i] && old[i] === current[j]
      ? 1 + scores[(i + 1) * width + j + 1]!
      : Math.max(scores[(i + 1) * width + j]!, scores[i * width + j + 1]!);
  }
  const anchors: Array<[number, number]> = [[-1, -1]];
  let i = 0, j = 0;
  while (i < spans.length && j < words.length) {
    if (old[i] && old[i] === current[j]) { anchors.push([i++, j++]); }
    else if (scores[(i + 1) * width + j]! >= scores[i * width + j + 1]!) i++;
    else j++;
  }
  anchors.push([spans.length, words.length]);
  const bounds = (first: number, last = first): WordBounds => ({startMs: spans[first]!.startMs, endMs: spans[last]!.endMs});
  for (let n = 1; n < anchors.length; n++) {
    const [before, targetBefore] = anchors[n - 1]!, [after, targetAfter] = anchors[n]!;
    const oldCount = after - before - 1, newCount = targetAfter - targetBefore - 1;
    // A short correction next to matching words can retain measured slots.
    // Completely unrelated recognition must not silently become a full repair.
    const anchored = anchors.length > 2 && scores[0]! >= Math.min(2, Math.min(words.length, spans.length) - 1);
    if (anchored && oldCount === newCount && newCount <= 3) {
      for (let k = 1; k <= newCount; k++) result[targetBefore + k] = bounds(before + k);
    } else if (anchored && newCount === 1 && oldCount > 1 && oldCount <= 3) {
      result[targetBefore + 1] = bounds(before + 1, after - 1);
    }
    if (after < spans.length) result[targetAfter] = bounds(after);
  }
  return result;
}

/** Keep known slots and add only candidates that fit between those neighbors. */
export function mergeWordTimings(cue: Cue, known: PartialWordTimings, recovered: PartialWordTimings): PartialWordTimings {
  const result = known.map(span => span && {...span});
  for (let i = 0; i < result.length; i++) {
    if (result[i] || !recovered[i]) continue;
    const candidate = recovered[i]!;
    const before = result.slice(0, i).reverse().find(Boolean)?.endMs ?? cue.startMs;
    const after = result.slice(i + 1).find(Boolean)?.startMs ?? cue.endMs;
    if (candidate.startMs >= before && candidate.endMs <= after && candidate.endMs > candidate.startMs) result[i] = {...candidate};
  }
  return result;
}

/** Convert relative model seconds without requiring identical recognition text. */
export function recoveredSegments(cue: Cue, segments: TranscriptSegment[], cropStartMs: number): PartialWordTimings {
  return recoverWordTimings(cue, segments.map(span => ({
    text: span.text, startMs: cropStartMs + Math.round(span.start * 1000), endMs: cropStartMs + Math.round(span.end * 1000),
  })));
}
