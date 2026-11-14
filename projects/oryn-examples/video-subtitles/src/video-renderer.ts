import { DEFAULT_DESIGN, normalizeDesign, captionBox, type CaptionDesign } from './caption-design.ts';
import { serializeSubRip, type Cue } from './subtitles.ts';
import { drawVideoEnding, endingAudioLevel, validateEnding, type VideoEnding } from './video-ending.ts';

export type CaptionStyle = 'highlight' | 'plain' | 'fade' | 'pop' | 'word-box' | 'word-reveal' | 'word-pop';
export function requiresWordTiming(style: string): boolean {
  return ['highlight', 'word-box', 'word-reveal', 'word-pop'].includes(style);
}

export const MAX_RENDER_BYTES = 256 * 1024 * 1024;

export function hasWordTiming(cue: Cue): boolean {
  if (!cue.spans?.length) return false;
  let end = cue.startMs;
  for (const span of cue.spans) {
    if (!span.text.trim() || /\s/u.test(span.text.trim()) ||
        !Number.isFinite(span.startMs) || !Number.isFinite(span.endMs) ||
        span.startMs < end || span.endMs <= span.startMs || span.endMs > cue.endMs) return false;
    end = span.endMs;
  }
  return cue.spans.map(span => span.text.trim()).join(' ') === cue.text.trim().replace(/\s+/gu, ' ');
}

export function requireWordTiming(cues: Cue[]): void {
  const missing = cues.filter(cue => !hasWordTiming(cue)).length;
  if (missing) throw new Error(`${missing} caption(s) have no individual word timings. Generate subtitles from the video on this page, or import a saved timing file. SRT alone cannot animate spoken words. Choose No animation, Fade, or Pop in to render without word timings.`);
}

/** Keep corrections on original timing slots; never invent sub-word boundaries. */
export function retainTiming(edited: Cue, original?: Cue): Cue {
  if (!original || edited.startMs !== original.startMs || edited.endMs !== original.endMs) return edited;
  if (edited.text === original.text) return { ...edited, spans: original.spans };
  const words = edited.text.trim().split(/\s+/u);
  if (!hasWordTiming(original) || words.length > 256 || original.spans!.length > 256) return edited;
  const spans = original.spans!;
  const key = (text: string) => text.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  const oldKeys = spans.map(span => key(span.text)), newKeys = words.map(key);
  const width = words.length + 1;
  const lengths = new Uint16Array((spans.length + 1) * width);
  for (let i = spans.length - 1; i >= 0; i--) {
    for (let j = words.length - 1; j >= 0; j--) {
      lengths[i * width + j] = oldKeys[i] && oldKeys[i] === newKeys[j]
        ? 1 + lengths[(i + 1) * width + j + 1]!
        : Math.max(lengths[(i + 1) * width + j]!, lengths[i * width + j + 1]!);
    }
  }
  const anchors: Array<[number, number]> = [[-1, -1]];
  let i = 0, j = 0;
  while (i < spans.length && j < words.length) {
    if (oldKeys[i] && oldKeys[i] === newKeys[j]) { anchors.push([i++, j++]); }
    else if (lengths[(i + 1) * width + j]! >= lengths[i * width + j + 1]!) i++;
    else j++;
  }
  anchors.push([spans.length, words.length]);
  const corrected: NonNullable<Cue['spans']> = [];
  for (let index = 1; index < anchors.length; index++) {
    const [oldBefore, newBefore] = anchors[index - 1]!;
    const [oldAfter, newAfter] = anchors[index]!;
    const oldCount = oldAfter - oldBefore - 1, newCount = newAfter - newBefore - 1;
    if (newCount) {
      if (!oldCount || (newCount !== oldCount && newCount !== 1)) return edited;
      for (let offset = 0; offset < newCount; offset++) {
        const first = spans[oldBefore + 1 + offset]!;
        const last = spans[newCount === 1 ? oldAfter - 1 : oldBefore + 1 + offset]!;
        corrected.push({text: words[newBefore + 1 + offset]!, startMs: first.startMs, endMs: last.endMs});
      }
    }
    if (oldAfter < spans.length) corrected.push({...spans[oldAfter]!, text: words[newAfter]!});
  }
  return { ...edited, spans: corrected };
}

export function drawCaptions(
  context: CanvasRenderingContext2D, cues: Cue[], timeMs: number,
  width: number, height: number, style: CaptionStyle, appearance: CaptionDesign = DEFAULT_DESIGN,
): void {
  const cue = cues.find(value => timeMs >= value.startMs && timeMs < value.endMs);
  if (!cue) return;
  const design = normalizeDesign(appearance);
  const size = Math.max(8, Math.min(width, height) * design.size / 100);
  const fonts = {sans: 'Arial, sans-serif', serif: 'Georgia, serif', mono: 'Menlo, monospace'};
  context.font = `${design.bold ? 700 : 400} ${size}px ${fonts[design.font]}`;
  context.textBaseline = 'middle';
  const wordTimed = hasWordTiming(cue);
  const spans = wordTimed ? cue.spans! : [cue];
  const tokens = spans.flatMap(span => span.text.split(/\s+/u).map(text => ({
    startMs: span.startMs, endMs: span.endMs,
    text: design.uppercase ? text.toLocaleUpperCase() : text, active: timeMs >= span.startMs && timeMs < span.endMs,
  })));
  const space = context.measureText(' ').width;
  const maxWidth = width * design.width / 100 - size * 0.8;
  const lines: Array<Array<typeof tokens[number] & { width: number }>> = [[]];
  let lineWidth = 0;
  for (const token of tokens) {
    // Split unusually long tokens so unbroken text cannot leave the frame.
    let part = '';
    const parts: string[] = [];
    for (const char of token.text) {
      if (part && context.measureText(part + char).width > maxWidth) {
        parts.push(part); part = '';
      }
      part += char;
    }
    if (part) parts.push(part);
    for (const text of parts) {
      const measured = context.measureText(text).width;
      if (lineWidth && lineWidth + space + measured > maxWidth) {
        lines.push([]); lineWidth = 0;
      }
      lines.at(-1)!.push({ ...token, text, width: measured });
      lineWidth += (lineWidth ? space : 0) + measured;
    }
  }
  const lineHeight = size * 1.4;
  if (lines.length * lineHeight > height * 0.7)
    throw new Error('A caption is too large to fit. Reduce text size, widen the caption box, or split it into shorter cues.');
  const padding = size * 0.4;
  const box = captionBox(width, height, maxWidth + padding * 2, lines.length * lineHeight + padding * 2, design);
  const age = timeMs - cue.startMs;
  const remaining = cue.endMs - timeMs;
  const ramp = Math.min(180, (cue.endMs - cue.startMs) / 3);
  context.save();
  try {
    if (style === 'fade') context.globalAlpha = Math.max(0, Math.min(1, age / ramp, remaining / ramp));
    if (style === 'pop') {
      const scale = 0.85 + 0.15 * Math.min(1, age / ramp);
      context.translate(box.x + box.width / 2, box.y + box.height / 2);
      context.scale(scale, scale);
      context.translate(-box.x - box.width / 2, -box.y - box.height / 2);
    }
    if (design.opacity > 0) {
      const opacity = Math.round(design.opacity * 255).toString(16).padStart(2, '0');
      context.fillStyle = design.background + opacity;
      context.beginPath();
      context.roundRect(box.x, box.y, box.width, box.height, size * 0.2);
      context.fill();
    }
    context.lineJoin = 'round';
    context.strokeStyle = '#000000';
    context.lineWidth = size * design.outline / 20;
    if (design.shadow) {
      context.shadowColor = 'rgba(0,0,0,0.9)';
      context.shadowBlur = size * 0.15;
      context.shadowOffsetY = size * 0.06;
    }
    lines.forEach((line, index) => {
      const total = line.reduce((sum, token) => sum + token.width, 0) + Math.max(0, line.length - 1) * space;
      let x = box.x + padding + (design.align === 'left' ? 0 : design.align === 'right' ? maxWidth - total : (maxWidth - total) / 2);
      const y = box.y + padding + (index + 0.5) * lineHeight;
      for (const token of line) {
        // Reserve every word's final position, even before a reveal, to avoid reflow.
        if (style === 'word-reveal' && wordTimed && timeMs < token.startMs) {
          x += token.width + space;
          continue;
        }
        context.save();
        try {
          const active = wordTimed && token.active;
          context.fillStyle = active && (style === 'highlight' || style === 'word-pop') ? design.highlight : design.color;
          if (style === 'word-box' && active) {
            context.shadowBlur = 0;
            context.shadowOffsetY = 0;
            context.fillStyle = design.highlight;
            context.beginPath();
            context.roundRect(x - space * 0.35, y - size * 0.62, token.width + space * 0.7, size * 1.24, size * 0.16);
            context.fill();
            const channels = [1, 3, 5].map(offset => parseInt(design.highlight.slice(offset, offset + 2), 16) / 255)
              .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
            const luminance = channels[0]! * 0.2126 + channels[1]! * 0.7152 + channels[2]! * 0.0722;
            context.fillStyle = luminance > 0.179 ? '#000000' : '#ffffff';
          }
          if (style === 'word-pop' && active) {
            const progress = Math.min(1, (timeMs - token.startMs) / Math.min(220, token.endMs - token.startMs));
            const scale = 1 + 0.14 * Math.sin(Math.PI * progress);
            context.translate(x + token.width / 2, y);
            context.scale(scale, scale);
            context.translate(-x - token.width / 2, -y);
          }
          if (style === 'word-reveal' && wordTimed) {
            const progress = Math.min(1, (timeMs - token.startMs) / Math.min(120, token.endMs - token.startMs));
            context.globalAlpha *= progress;
            context.translate(0, size * 0.15 * (1 - progress));
          }
          if (design.outline > 0 && !(style === 'word-box' && active)) context.strokeText(token.text, x, y);
          context.fillText(token.text, x, y);
        } finally { context.restore(); }
        x += token.width + space;
      }
    });
  } finally { context.restore(); }

}

export type VideoRenderOptions = {
  file: File; cues: Cue[]; canvas: HTMLCanvasElement; style: CaptionStyle; design?: CaptionDesign;
  ending?: VideoEnding;
  signal: AbortSignal; onProgress: (fraction: number) => void;
};

export async function renderCaptionedVideo(options: VideoRenderOptions): Promise<Blob> {
  const { cues, style, signal, ending } = options;
  if (ending) validateEnding(ending);
  serializeSubRip(cues);
  if (!cues.length) throw new Error('Create or import captions first.');
  if (requiresWordTiming(style)) requireWordTiming(cues);
  if (cues.some(cue => cue.text.length > 500))
    throw new Error('Split captions longer than 500 characters before rendering.');
  signal.throwIfAborted();
  const { renderVideoFrames } = await import('./video-export.ts');
  signal.throwIfAborted();
  return renderVideoFrames(options, {drawCaptions, drawVideoEnding, endingAudioLevel, maxBytes: MAX_RENDER_BYTES});
}
