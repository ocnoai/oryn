export interface TimedText {
  startMs: number;
  endMs: number;
  text: string;
}
export interface Cue extends TimedText {
  /** Validated model/alignment or user-entered spans, never interpolated word timings. */
  spans?: TimedText[];
  /** Optional measured timings for only the words already recovered. */
  wordTimingHints?: TimedText[];
}

export interface TranscriptSegment {
  text: string;
  start: number;
  end: number;
}

export interface TranscriptValue {
  text: string;
  language: string | null;
  segments?: TranscriptSegment[];
}

export interface AudioWindow {
  index: number;
  total: number;
  startSample: number;
  endSample: number;
  startMs: number;
  endMs: number;
}

export interface WindowCues {
  cues: Cue[];
  timing: "native" | "window-estimated" | "empty";
}

const MAX_CUE_CHARACTERS = 84;
const MAX_CUE_DURATION_MS = 5_000;
const CAPTION_GAP_MS = 800;

export function planAudioWindows(
  sampleCount: number,
  sampleRate: number,
  windowSeconds: number,
): AudioWindow[] {
  if (
    !Number.isSafeInteger(sampleCount) ||
    sampleCount < 1 ||
    !Number.isSafeInteger(sampleRate) ||
    sampleRate < 1 ||
    !Number.isFinite(windowSeconds) ||
    windowSeconds <= 0
  ) {
    throw new Error("Audio window parameters are invalid.");
  }
  const samplesPerWindow = Math.max(1, Math.floor(sampleRate * windowSeconds));
  const total = Math.ceil(sampleCount / samplesPerWindow);
  return Array.from({ length: total }, (_, index) => {
    const startSample = index * samplesPerWindow;
    const endSample = Math.min(sampleCount, startSample + samplesPerWindow);
    return {
      index,
      total,
      startSample,
      endSample,
      startMs: Math.round((startSample / sampleRate) * 1_000),
      endMs: Math.round((endSample / sampleRate) * 1_000),
    };
  });
}

export function cuesForWindow(
  value: TranscriptValue,
  window: AudioWindow,
): WindowCues {
  const native = (value.segments ?? [])
    .map((segment) => ({
      startMs: window.startMs + Math.max(0, Math.round(segment.start * 1_000)),
      endMs: window.startMs + Math.max(1, Math.round(segment.end * 1_000)),
      text: normalizeText(segment.text),
    }))
    .map((cue) => ({
      ...cue,
      startMs: Math.min(window.endMs - 1, cue.startMs),
      endMs: Math.min(window.endMs, cue.endMs),
    }))
    .filter((cue) => cue.text && cue.endMs > cue.startMs);
  if (native.length)
    return {
      cues: captionizeNativeSegments(makeNonOverlapping(native)),
      timing: "native",
    };

  const pieces = splitCaptionText(value.text);
  if (!pieces.length) return { cues: [], timing: "empty" };
  const duration = window.endMs - window.startMs;
  return {
    timing: "window-estimated",
    cues: pieces.map((text, index) => ({
      startMs: window.startMs + Math.floor((duration * index) / pieces.length),
      endMs:
        window.startMs + Math.floor((duration * (index + 1)) / pieces.length),
      text,
    })),
  };
}

export function serializeSubRip(values: Cue[]): string {
  let lastEnd = 0;
  return values
    .map((cue, index) => {
      const text = normalizeText(cue.text);
      if (
        !Number.isInteger(cue.startMs) ||
        !Number.isInteger(cue.endMs) ||
        cue.startMs < lastEnd ||
        cue.endMs <= cue.startMs ||
        !text ||
        text.includes("-->")
      ) {
        throw new Error(`Cue ${index + 1} has invalid timing or text.`);
      }
      lastEnd = cue.endMs;
      return `${index + 1}\r\n${formatTimestamp(cue.startMs)} --> ${formatTimestamp(cue.endMs)}\r\n${text}\r\n`;
    })
    .join("\r\n");
}

export function formatTimestamp(value: number): string {
  const bounded = Math.max(0, Math.round(value));
  const hours = Math.floor(bounded / 3_600_000);
  const minutes = Math.floor((bounded % 3_600_000) / 60_000);
  const seconds = Math.floor((bounded % 60_000) / 1_000);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")},${String(bounded % 1_000).padStart(3, "0")}`;
}

function splitCaptionText(value: string): string[] {
  const words = normalizeText(value).split(" ").filter(Boolean);
  const result: string[] = [];
  let pending = "";
  for (const word of words) {
    if (word.length > MAX_CUE_CHARACTERS) {
      if (pending) result.push(pending);
      pending = "";
      for (let offset = 0; offset < word.length; offset += MAX_CUE_CHARACTERS) {
        result.push(word.slice(offset, offset + MAX_CUE_CHARACTERS));
      }
      continue;
    }
    const candidate = pending ? `${pending} ${word}` : word;
    if (candidate.length > MAX_CUE_CHARACTERS) {
      result.push(pending);
      pending = word;
    } else {
      pending = candidate;
    }
  }
  if (pending) result.push(pending);
  return result;
}

function makeNonOverlapping(values: Cue[]): Cue[] {
  const result: Cue[] = [];
  let lastEnd = 0;
  for (const value of values.sort(
    (left, right) => left.startMs - right.startMs,
  )) {
    const startMs = Math.max(lastEnd, value.startMs);
    if (value.endMs <= startMs) continue;
    result.push({ ...value, startMs });
    lastEnd = value.endMs;
  }
  return result;
}

function captionizeNativeSegments(values: Cue[]): Cue[] {
  const result: Cue[] = [];
  let pending: Cue | undefined;
  for (const value of values) {
    if (!pending) {
      pending = { ...value, spans: [{ ...value }] };
      continue;
    }
    const combinedText = `${pending.text} ${value.text}`;
    const mustFlush =
      value.startMs - pending.endMs > CAPTION_GAP_MS ||
      value.endMs - pending.startMs > MAX_CUE_DURATION_MS ||
      combinedText.length > MAX_CUE_CHARACTERS ||
      /[.!?]$/u.test(pending.text);
    if (mustFlush) {
      result.push(pending);
      pending = { ...value, spans: [{ ...value }] };
    } else {
      pending.endMs = value.endMs;
      pending.text = combinedText;
      pending.spans!.push({ ...value });
    }
  }
  if (pending) result.push(pending);
  return result;
}

function normalizeText(value: string): string {
  return value.replace(/[\r\n\t ]+/gu, " ").trim();
}

/** Import plain SRT as whole-caption timing, without fabricating word alignment. */
export function parseSubRip(value: string): Cue[] {
  const blocks = value.replace(/^\uFEFF/u, '').trim().split(/\r?\n\s*\r?\n/u);
  if (blocks.length > 10_000) throw new Error('Import at most 10,000 captions.');
  const timestamp = (text: string) => {
    const match = /^(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})$/u.exec(text);
    if (!match) throw new Error('SRT timestamps must use HH:MM:SS,mmm.');
    return Number(match[1]) * 3_600_000 + Number(match[2]) * 60_000 + Number(match[3]) * 1000 + Number(match[4]);
  };
  const cues = blocks.map((block, index) => {
    const lines = block.split(/\r?\n/u);
    if (lines[0]?.trim() !== String(index + 1)) throw new Error('SRT captions must be numbered consecutively from 1.');
    const times = lines[1]?.trim().split(/\s+-->\s+/u);
    if (!times || times.length !== 2) throw new Error('Invalid SRT time range.');
    return { startMs: timestamp(times[0]!), endMs: timestamp(times[1]!), text: normalizeText(lines.slice(2).join(' ')) };
  });
  serializeSubRip(cues);
  return cues;
}

/** Data-only companion format: SRT cannot preserve within-caption timing spans. */
export function parseCaptionTiming(value: string): Cue[] {
  const data: unknown = JSON.parse(value);
  if (!data || typeof data !== 'object' || !('schema' in data) ||
      data.schema !== 'oryn-caption-timing/1' || !('cues' in data) ||
      !Array.isArray(data.cues) || !data.cues.length || data.cues.length > 10_000)
    throw new Error('Invalid Oryn caption timing file.');
  let totalSpans = 0;
  const read = (value: unknown): TimedText => {
    if (!value || typeof value !== 'object' || !('text' in value) ||
        typeof value.text !== 'string' || !value.text.trim() || value.text.length > 10_000 ||
        !('startMs' in value) || !Number.isSafeInteger(value.startMs) ||
        !('endMs' in value) || !Number.isSafeInteger(value.endMs))
      throw new Error('Invalid caption timing entry.');
    const startMs = value.startMs as number, endMs = value.endMs as number;
    if (startMs < 0 || endMs <= startMs || endMs > 14_400_000)
      throw new Error('Invalid caption timing range.');
    return {text: normalizeText(value.text), startMs, endMs};
  };
  const cues: Cue[] = data.cues.map(value => {
    const cue: Cue = read(value);
    if (value.spans !== undefined) {
      if (!Array.isArray(value.spans) || !value.spans.length) throw new Error('Invalid timing spans.');
      totalSpans += value.spans.length;
      if (totalSpans > 100_000) throw new Error('Too many timing spans.');
      let previousEnd = cue.startMs;
      cue.spans = value.spans.map((value: unknown) => {
        const span = read(value);
        if (span.startMs < previousEnd || span.endMs > cue.endMs) throw new Error('Timing spans overlap or leave their caption.');
        previousEnd = span.endMs;
        return span;
      });
      if (cue.spans!.map(span => span.text).join(' ') !== cue.text)
        throw new Error('Timing spans do not match the caption text.');
    }
    if (value.wordTimingHints !== undefined) {
      if (!Array.isArray(value.wordTimingHints) || value.wordTimingHints.length > 256) throw new Error('Invalid partial word timings.');
      totalSpans += value.wordTimingHints.length;
      if (totalSpans > 100_000) throw new Error('Too many timing spans.');
      let previousEnd = cue.startMs, wordIndex = 0;
      const words = cue.text.split(/\s+/u);
      cue.wordTimingHints = value.wordTimingHints.map((value: unknown) => {
        const span = read(value);
        while (wordIndex < words.length && words[wordIndex] !== span.text) wordIndex++;
        if (wordIndex >= words.length || /\s/u.test(span.text) || span.startMs < previousEnd || span.endMs > cue.endMs)
          throw new Error('Partial word timings must follow caption words and stay within its bounds.');
        wordIndex++; previousEnd = span.endMs;
        return span;
      });
    }
    return cue;
  });
  serializeSubRip(cues);
  return cues;
}

export function serializeCaptionTiming(cues: Cue[]): string {
  const value = JSON.stringify({schema: 'oryn-caption-timing/1', cues}, null, 2);
  parseCaptionTiming(value);
  if (new TextEncoder().encode(value).byteLength > 4 * 1024 * 1024)
    throw new Error('The timing file exceeds the 4 MB limit.');
  return value;
}
