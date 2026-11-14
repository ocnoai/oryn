export type EndingTransition = 'crossfade' | 'fade-black' | 'slide' | 'none';
type TransitionOptions = { transition: EndingTransition; transitionSeconds: number };
export type VideoEnding = TransitionOptions & (
  | { kind: 'video'; file: File; includeAudio: boolean }
  | { kind: 'animation'; title: string; subtitle: string; color: string; duration: number }
);

export function validateEnding(ending: VideoEnding): void {
  if (!['crossfade', 'fade-black', 'slide', 'none'].includes(ending.transition) ||
      !Number.isFinite(ending.transitionSeconds) || ending.transitionSeconds < 0.1 || ending.transitionSeconds > 2)
    throw new Error('Choose a transition duration between 0.1 and 2 seconds.');
  if (ending.kind === 'video') {
    if (!ending.file?.size) throw new Error('Choose a playable ending video.');
  } else if (ending.kind === 'animation') {
    if (!ending.title.trim() || ending.title.length > 100 || ending.subtitle.length > 160)
      throw new Error('Add an ending title (up to 100 characters) and an optional subtitle (up to 160).');
    if (!Number.isFinite(ending.duration) || ending.duration < 1 || ending.duration > 15)
      throw new Error('Choose an animation duration between 1 and 15 seconds.');
    if (!/^#[\da-f]{6}$/i.test(ending.color)) throw new Error('Choose a valid ending accent color.');
  } else throw new Error('Choose an ending video or animation.');
}

export function endingTransitionDuration(ending: VideoEnding, duration: number): number {
  return ending.transition === 'none' ? 0 : Math.min(ending.transitionSeconds, duration / 2);
}

export function endingProgress(ending: VideoEnding, time: number, duration: number): number {
  const transition = endingTransitionDuration(ending, duration);
  return transition ? Math.max(0, Math.min(1, time / transition)) : 1;
}

export function endingAudioLevel(ending: VideoEnding, time: number, duration: number): number {
  if (ending.kind !== 'video' || !ending.includeAudio) return 0;
  const progress = endingProgress(ending, time, duration);
  return ending.transition === 'fade-black' ? Math.max(0, progress * 2 - 1) : progress;
}

type EndingFrame = HTMLVideoElement | HTMLCanvasElement | OffscreenCanvas;

export function drawContainedVideo(context: CanvasRenderingContext2D, video: EndingFrame, width: number, height: number): void {
  context.fillStyle = '#000000';
  context.fillRect(0, 0, width, height);
  const frameWidth = 'videoWidth' in video ? video.videoWidth : video.width;
  const frameHeight = 'videoHeight' in video ? video.videoHeight : video.height;
  const scale = Math.min(width / frameWidth, height / frameHeight);
  const w = frameWidth * scale, h = frameHeight * scale;
  context.drawImage(video, (width - w) / 2, (height - h) / 2, w, h);
}

export function drawEndingCard(context: CanvasRenderingContext2D, ending: Extract<VideoEnding, {kind: 'animation'}>, time: number, width: number, height: number): void {
  context.save();
  try {
    context.fillStyle = '#10121e';
    context.fillRect(0, 0, width, height);
    const size = Math.min(width, height);
    const glow = context.createRadialGradient(width * 0.5, height * 0.45, 0, width * 0.5, height * 0.45, size * (0.7 + Math.sin(time * 1.4) * 0.06));
    glow.addColorStop(0, ending.color + '55');
    glow.addColorStop(1, ending.color + '00');
    context.fillStyle = glow;
    context.fillRect(0, 0, width, height);
    const enter = 1 - (1 - Math.max(0, Math.min(1, time / 0.7))) ** 3;
    context.globalAlpha *= enter;
    context.translate(0, size * 0.05 * (1 - enter));
    context.fillStyle = ending.color;
    context.fillRect(width * 0.5 - size * 0.07, height * 0.35, size * 0.14 * enter, Math.max(2, size * 0.006));
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    const text = (value: string, y: number, fontSize: number, bold: boolean, color: string) => {
      const font = (s: number) => `${bold ? 700 : 400} ${s}px Arial, sans-serif`;
      context.font = font(fontSize);
      const words = value.trim().split(/\s+/u);
      const lines: string[] = [];
      for (const word of words) {
        const last = lines.at(-1);
        if (last && context.measureText(last + ' ' + word).width <= width * 0.82) lines[lines.length - 1] += ' ' + word;
        else lines.push(word);
      }
      // Keep long titles inside portrait and landscape frames without clipping.
      const scale = Math.min(1, 3 / lines.length, ...lines.map(line => width * 0.82 / Math.max(1, context.measureText(line).width)));
      context.font = font(fontSize * scale);
      context.fillStyle = color;
      lines.forEach((line, i) => context.fillText(line, width / 2, y + (i - (lines.length - 1) / 2) * fontSize * scale * 1.25));
    };
    text(ending.title, height * 0.49, size * 0.09, true, '#ffffff');
    if (ending.subtitle.trim()) text(ending.subtitle, height * 0.7, size * 0.037, false, '#d5d8e8');
  } finally { context.restore(); }
}

const transitionLayers = new WeakMap<CanvasRenderingContext2D, HTMLCanvasElement>();

/** Transition from the held final source frame, preserving all source speech. */
export function drawVideoEnding(context: CanvasRenderingContext2D, outgoing: CanvasImageSource, ending: VideoEnding,
  time: number, duration: number, width: number, height: number, video?: EndingFrame): void {
  const progress = endingProgress(ending, time, duration);
  const paintIncoming = (target: CanvasRenderingContext2D) => {
    if (ending.kind === 'animation') drawEndingCard(target, ending, time, width, height);
    else if (video) drawContainedVideo(target, video, width, height);
  };
  context.save();
  try {
    context.fillStyle = '#000000';
    context.fillRect(0, 0, width, height);
    if (progress >= 1) { paintIncoming(context); return; }
    // Composite the complete incoming frame once, so its background, glow and text
    // do not each apply the crossfade alpha to the outgoing frame independently.
    let layer = transitionLayers.get(context);
    if (!layer) { layer = document.createElement('canvas'); transitionLayers.set(context, layer); }
    if (layer.width !== width) layer.width = width;
    if (layer.height !== height) layer.height = height;
    paintIncoming(layer.getContext('2d')!);
    const incoming = () => context.drawImage(layer, 0, 0);
    if (ending.transition === 'fade-black') {
      context.globalAlpha = Math.abs(progress * 2 - 1);
      if (progress < 0.5) context.drawImage(outgoing, 0, 0, width, height);
      else incoming();
    } else if (ending.transition === 'slide') {
      const ease = progress * progress * (3 - 2 * progress);
      context.drawImage(outgoing, -width * ease, 0, width, height);
      context.translate(width * (1 - ease), 0);
      incoming();
    } else {
      context.drawImage(outgoing, 0, 0, width, height);
      context.globalAlpha = progress;
      incoming();
    }
  } finally { context.restore(); }
}

export function loadVideo(video: HTMLVideoElement, url: string, signal: AbortSignal, label: string): Promise<void> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timeout);
      video.removeEventListener('loadeddata', loaded);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
    };
    const loaded = () => {
      cleanup();
      if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth || !video.videoHeight)
        reject(new Error(`${label} has no playable video track or duration.`));
      else resolve();
    };
    const failed = () => { cleanup(); reject(new Error(`${label} cannot be decoded by the browser. Choose another video.`)); };
    const aborted = () => { cleanup(); reject(signal.reason); };
    const timeout = setTimeout(failed, 30_000);
    video.addEventListener('loadeddata', loaded, {once: true});
    video.addEventListener('error', failed, {once: true});
    signal.addEventListener('abort', aborted, {once: true});
    video.src = url;
  });
}
