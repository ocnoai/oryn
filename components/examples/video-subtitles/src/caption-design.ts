export interface CaptionDesign {
  font: 'sans' | 'serif' | 'mono';
  size: number;
  bold: boolean;
  uppercase: boolean;
  color: string;
  highlight: string;
  background: string;
  opacity: number;
  outline: number;
  shadow: boolean;
  x: number;
  y: number;
  width: number;
  align: 'left' | 'center' | 'right';
}

export const DEFAULT_DESIGN: CaptionDesign = {
  font: 'sans', size: 5.2, bold: true, uppercase: false,
  color: '#ffffff', highlight: '#ffe34d', background: '#000000',
  opacity: 0.7, outline: 0, shadow: false,
  x: 50, y: 82, width: 86, align: 'center',
};
export const CAPTION_PRESETS: Record<string, {label: string; design: CaptionDesign}> = {
  classic: {label: 'Classic', design: {...DEFAULT_DESIGN}},
  clean: {label: 'Clean outline', design: {...DEFAULT_DESIGN, opacity: 0, outline: 2, shadow: true}},
  bold: {label: 'Bold social', design: {...DEFAULT_DESIGN, size: 7, uppercase: true, opacity: 0, outline: 4, highlight: '#b8ff42', shadow: true}},
  card: {label: 'Light card', design: {...DEFAULT_DESIGN, color: '#182032', highlight: '#5138cc', background: '#fff6e8', opacity: 0.96, size: 4.8}},
  neon: {label: 'Neon', design: {...DEFAULT_DESIGN, color: '#c7f9ff', highlight: '#ff7edb', background: '#17102c', opacity: 0.9, shadow: true}},
  editorial: {label: 'Editorial', design: {...DEFAULT_DESIGN, font: 'serif', color: '#fff2cf', highlight: '#f6c871', bold: false, opacity: 0.25, size: 5.7}},
};

export function normalizeDesign(value: Partial<CaptionDesign>): CaptionDesign {
  const bounded = (value: unknown, min: number, max: number, fallback: number) =>
    typeof value === 'number' && Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback;
  const color = (value: unknown, fallback: string) => typeof value === 'string' && /^#[0-9a-f]{6}$/iu.test(value) ? value : fallback;
  return {
    font: value.font === 'serif' || value.font === 'mono' ? value.font : 'sans',
    size: bounded(value.size, 2, 12, DEFAULT_DESIGN.size),
    bold: typeof value.bold === 'boolean' ? value.bold : DEFAULT_DESIGN.bold,
    uppercase: value.uppercase === true,
    color: color(value.color, DEFAULT_DESIGN.color),
    highlight: color(value.highlight, DEFAULT_DESIGN.highlight),
    background: color(value.background, DEFAULT_DESIGN.background),
    opacity: bounded(value.opacity, 0, 1, DEFAULT_DESIGN.opacity),
    outline: bounded(value.outline, 0, 6, 0), shadow: value.shadow === true,
    x: bounded(value.x, 0, 100, 50), y: bounded(value.y, 0, 100, 82),
    width: bounded(value.width, 25, 96, 86),
    align: value.align === 'left' || value.align === 'right' ? value.align : 'center',
  };
}

export function captionBox(width: number, height: number, boxWidth: number, boxHeight: number, design: CaptionDesign) {
  const margin = Math.min(width, height) * 0.02;
  const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
  return {
    x: clamp(width * design.x / 100 - boxWidth / 2, margin, width - margin - boxWidth),
    y: clamp(height * design.y / 100 - boxHeight / 2, margin, height - margin - boxHeight),
    width: boxWidth, height: boxHeight,
  };
}
