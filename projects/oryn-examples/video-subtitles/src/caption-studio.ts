import { CAPTION_PRESETS, DEFAULT_DESIGN, normalizeDesign, type CaptionDesign } from './caption-design';
import { drawCaptions, type CaptionStyle } from './video-renderer';
import type { Cue } from './subtitles';

export function createCaptionStudio(getCues: () => Cue[], onChange: () => void, onPlay: () => void = () => {}) {
  const get = <T extends Element>(id: string) => document.querySelector<T>(id)!;
  const root = get<HTMLElement>('#caption-studio');
  const canvas = get<HTMLCanvasElement>('#design-preview');
  const context = canvas.getContext('2d')!;
  const video = document.createElement('video');
  video.playsInline = true;
  video.preload = 'auto';
  const play = get<HTMLButtonElement>('#design-play');
  const seek = get<HTMLInputElement>('#design-seek');
  const time = get<HTMLElement>('#design-time');
  const notice = get<HTMLElement>('#design-notice');
  let design = {...DEFAULT_DESIGN};
  let cachedCues: Cue[] = [];
  let frame = 0;
  let busy = false;
  let dragging = false;
  let dragOrigin: {x: number; y: number; clientX: number; clientY: number} | undefined;
  try {
    const saved = localStorage.getItem('oryn-caption-design-v1');
    if (saved && saved.length < 4096) design = normalizeDesign(JSON.parse(saved));
  } catch { /* Preferences are optional. */ }
  const fields = [...root.querySelectorAll<HTMLInputElement | HTMLSelectElement>('[data-design]')];
  const effect = () => get<HTMLSelectElement>('#caption-style').value as CaptionStyle;
  function sync() {
    for (const field of fields) {
      const key = field.dataset.design as keyof CaptionDesign;
      if (field instanceof HTMLInputElement && field.type === 'checkbox') field.checked = design[key] as boolean;
      else field.value = String(design[key]);
      const output = root.querySelector<HTMLOutputElement>(`[data-value="${key}"]`);
      if (output) output.value = key === 'opacity' ? `${Math.round(design.opacity * 100)}%` : `${Math.round(Number(design[key]) * 10) / 10}${['size','x','y','width'].includes(key) ? '%' : ''}`;
    }
  }
  function persist() {
    try { localStorage.setItem('oryn-caption-design-v1', JSON.stringify(design)); } catch { /* Private mode/full storage. */ }
  }
  function changed() {
    sync(); persist(); onChange(); draw();
    root.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach(button => button.setAttribute('aria-pressed', 'false'));
  }
  for (const field of fields) field.addEventListener('input', () => {
    const key = field.dataset.design as keyof CaptionDesign;
    const value = field instanceof HTMLInputElement && field.type === 'checkbox' ? field.checked
      : field instanceof HTMLInputElement && field.type === 'range' ? Number(field.value) : field.value;
    design = normalizeDesign({...design, [key]: value}); changed();
  });
  const presets = get<HTMLElement>('#design-presets');
  for (const [id, preset] of Object.entries(CAPTION_PRESETS)) {
    const button = document.createElement('button');
    button.type = 'button'; button.dataset.preset = id;
    button.className = `preset preset-${id}`;
    button.textContent = preset.label;
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => {
      design = {...preset.design}; changed(); button.setAttribute('aria-pressed', 'true');
    });
    presets.append(button);
  }
  root.querySelectorAll<HTMLButtonElement>('[data-position]').forEach(button => button.addEventListener('click', () => {
    const [x,y] = button.dataset.position!.split(',').map(Number);
    design = normalizeDesign({...design,x,y}); changed();
  }));
  get<HTMLButtonElement>('#design-reset').addEventListener('click', () => { design = {...DEFAULT_DESIGN}; changed(); });
  function draw() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (video.readyState < 2 || !video.videoWidth) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    try {
      const cues = cachedCues;
      const active = cues.some(cue => video.currentTime * 1000 >= cue.startMs && video.currentTime * 1000 < cue.endMs);
      drawCaptions(context, cues, video.currentTime * 1000, canvas.width, canvas.height, effect(), design);
      notice.textContent = active ? 'Drag to reposition captions. Use arrow keys for precise adjustments.' : 'No caption at this time. Scrub to a caption or press First caption.';
    } catch (error) { notice.textContent = error instanceof Error ? error.message : 'Could not preview captions.'; }
    seek.value = String(video.currentTime);
    time.textContent = `${clock(video.currentTime)} / ${clock(video.duration)}`;
  }
  function tick() { draw(); if (!video.paused && !busy) frame = requestAnimationFrame(tick); }
  video.addEventListener('loadedmetadata', () => {
    const scale = Math.min(1, 1920 / video.videoWidth, 1080 / video.videoHeight);
    canvas.width = Math.max(2, Math.floor(video.videoWidth * scale / 2) * 2);
    canvas.height = Math.max(2, Math.floor(video.videoHeight * scale / 2) * 2);
    canvas.style.maxWidth = `${560 * canvas.width / canvas.height}px`;
    seek.max = String(Number.isFinite(video.duration) ? video.duration : 0);
  });
  for (const event of ['loadeddata','seeked','timeupdate']) video.addEventListener(event, draw);
  video.addEventListener('play', () => { onPlay(); play.textContent = 'Pause'; cancelAnimationFrame(frame); tick(); });
  video.addEventListener('pause', () => { play.textContent = 'Play'; cancelAnimationFrame(frame); draw(); });
  video.addEventListener('error', () => { notice.textContent = 'This video could not be loaded for preview.'; });
  play.addEventListener('click', () => {
    if (video.paused) void video.play().catch(() => {notice.textContent = 'Could not play the preview.';}); else video.pause();
  });
  seek.addEventListener('input', () => { if (Number.isFinite(video.duration)) video.currentTime = Number(seek.value); });
  get<HTMLButtonElement>('#design-first').addEventListener('click', () => {
    const cue = cachedCues[0]; if (cue && video.readyState >= 1) video.currentTime = Math.min(video.duration, (cue.startMs + Math.min(250, (cue.endMs-cue.startMs)/2)) / 1000);
  });
  get<HTMLInputElement>('#design-muted').addEventListener('change', event => {video.muted = (event.target as HTMLInputElement).checked;});
  canvas.addEventListener('pointerdown', event => {
    if (busy || video.readyState < 2) return;
    dragging = true;
    dragOrigin = {x:design.x,y:design.y,clientX:event.clientX,clientY:event.clientY};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!dragging || !dragOrigin || busy) return;
    const rect = canvas.getBoundingClientRect();
    design = normalizeDesign({...design,x:dragOrigin.x+(event.clientX-dragOrigin.clientX)/rect.width*100,y:dragOrigin.y+(event.clientY-dragOrigin.clientY)/rect.height*100});
    changed();
  });
  for (const event of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(event, () => {dragging=false;dragOrigin=undefined;});
  canvas.addEventListener('keydown', event => {
    if (busy || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
    event.preventDefault(); const step = event.shiftKey ? 5 : 1;
    design = normalizeDesign({...design,x:design.x+(event.key==='ArrowRight'?step:event.key==='ArrowLeft'?-step:0),y:design.y+(event.key==='ArrowDown'?step:event.key==='ArrowUp'?-step:0)}); changed();
  });
  sync();
  return {
    getDesign: () => ({...design}),
    pause: () => video.pause(),
    refresh: () => { cachedCues = getCues(); draw(); },
    setSource(url?: string) {
      cachedCues = [];
      video.pause();
      if (url) video.src = url; else {video.removeAttribute('src');video.load();}
      draw();
    },
    setBusy(value: boolean) { busy = value; if (value) video.pause(); },
  };
}
function clock(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00';
  return `${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,'0')}`;
}
