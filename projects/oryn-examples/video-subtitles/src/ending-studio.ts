import { drawEndingCard, drawVideoEnding, endingAudioLevel, loadVideo, validateEnding, type EndingTransition, type VideoEnding } from './video-ending';

export function createEndingStudio(getSource: () => File | undefined, onChange: () => void, onPlay: () => void,
  paintCaptions: (context: CanvasRenderingContext2D, time: number, width: number, height: number) => void) {
  const get = <T extends Element>(id: string) => document.querySelector<T>(id)!;
  const root = get<HTMLElement>('#ending-studio');
  const kind = get<HTMLSelectElement>('#ending-kind');
  const canvas = get<HTMLCanvasElement>('#ending-preview');
  const context = canvas.getContext('2d')!;
  const play = get<HTMLButtonElement>('#ending-play');
  const status = get<HTMLElement>('#ending-status');
  const value = (id: string) => get<HTMLInputElement | HTMLSelectElement>(id).value;
  let controller: AbortController | undefined;
  let stopPlayback = () => {};

  function settings(): VideoEnding | undefined {
    if (kind.value === 'none') return undefined;
    const transition = value('#ending-transition') as EndingTransition;
    const transitionSeconds = transition === 'none' ? 0.6 : Number(value('#ending-transition-duration'));
    let ending: VideoEnding;
    if (kind.value === 'video') {
      const file = get<HTMLInputElement>('#ending-file').files?.[0];
      if (!file) throw new Error('Choose a video to add as your ending.');
      ending = {kind: 'video', file, includeAudio: get<HTMLInputElement>('#ending-audio').checked, transition, transitionSeconds};
    } else {
      ending = {kind: 'animation', title: value('#ending-text'), subtitle: value('#ending-subtitle'), color: value('#ending-color'),
        duration: Number(value('#ending-duration')), transition, transitionSeconds};
    }
    validateEnding(ending);
    return ending;
  }
  function pause() {
    controller?.abort();
    controller = undefined;
    stopPlayback();
    play.textContent = 'Preview ending & transition';
  }
  function refresh() {
    pause();
    const enabled = kind.value !== 'none';
    get<HTMLElement>('#ending-options').hidden = !enabled;
    get<HTMLElement>('#ending-video-fields').hidden = kind.value !== 'video';
    get<HTMLElement>('#ending-animation-fields').hidden = kind.value !== 'animation';
    get<HTMLInputElement>('#ending-transition-duration').disabled = value('#ending-transition') === 'none';
    context.fillStyle = '#10121e'; context.fillRect(0, 0, canvas.width, canvas.height);
    try {
      const ending = settings();
      play.disabled = !ending;
      status.textContent = ending?.kind === 'video' ? `${ending.file.name} · Preview to check the transition and audio.`
        : ending ? `${ending.duration}-second animated ending · Preview before rendering.` : 'Your video will end with the original footage.';
      if (ending?.kind === 'animation') drawEndingCard(context, ending, 1, canvas.width, canvas.height);
      else if (ending) {
        context.fillStyle = '#aeb5ff'; context.textAlign = 'center'; context.font = '20px Arial, sans-serif';
        context.fillText('Preview your ending clip', canvas.width / 2, canvas.height / 2, canvas.width * 0.8);
      }
    } catch (error) { play.disabled = true; status.textContent = (error as Error).message; }
  }
  for (const input of root.querySelectorAll('input, select')) input.addEventListener(input instanceof HTMLInputElement && input.type === 'file' ? 'change' : 'input', () => { refresh(); onChange(); });
  // Select changes also support clients that dispatch change without input.
  root.querySelectorAll('select').forEach(select => select.addEventListener('change', () => { refresh(); onChange(); }));

  async function preview() {
    if (controller) { pause(); status.textContent = 'Preview stopped.'; return; }
    const source = getSource();
    if (!source) { status.textContent = 'Choose your source video first.'; return; }
    const abort = new AbortController();
    controller = abort;
    const videos: HTMLVideoElement[] = [];
    const urls: string[] = [];
    let frame = 0;
    const cleanup = () => {
      cancelAnimationFrame(frame);
      videos.forEach(video => { video.pause(); video.removeAttribute('src'); video.load(); });
      urls.splice(0).forEach(url => URL.revokeObjectURL(url));
    };
    stopPlayback = cleanup;
    try {
      const ending = settings();
      if (!ending) return;
      onPlay();
      play.textContent = 'Stop preview';
      status.textContent = 'Loading ending preview…';
      const load = async (file: File, label: string) => {
        const video = document.createElement('video'); video.playsInline = true; video.preload = 'auto';
        videos.push(video);
        const url = URL.createObjectURL(file); urls.push(url);
        await loadVideo(video, url, abort.signal, label);
        return video;
      };
      const sourceVideo = await load(source, 'The source video');
      const clip = ending.kind === 'video' ? await load(ending.file, 'The ending video') : undefined;
      const duration = ending.kind === 'animation' ? ending.duration : clip!.duration;
      if (sourceVideo.duration + duration > 1800) throw new Error('Video rendering is limited to 30 minutes per export, including the ending.');
      await new Promise<void>((resolve, reject) => {
        const cleanupSeek = () => { clearTimeout(timeout); sourceVideo.removeEventListener('seeked', done); abort.signal.removeEventListener('abort', aborted); };
        const done = () => { cleanupSeek(); resolve(); };
        const aborted = () => { cleanupSeek(); reject(abort.signal.reason); };
        const timeout = setTimeout(() => { cleanupSeek(); reject(new Error('Could not seek to the end of the source video.')); }, 10_000);
        sourceVideo.addEventListener('seeked', done, {once: true});
        abort.signal.addEventListener('abort', aborted, {once: true});
        sourceVideo.currentTime = Math.max(0, sourceVideo.duration - 0.001);
      });
      abort.signal.throwIfAborted();
      const scale = Math.min(1, 1920 / sourceVideo.videoWidth, 1080 / sourceVideo.videoHeight);
      canvas.width = Math.max(2, Math.floor(sourceVideo.videoWidth * scale / 2) * 2);
      canvas.height = Math.max(2, Math.floor(sourceVideo.videoHeight * scale / 2) * 2);
      const outgoing = document.createElement('canvas'); outgoing.width = canvas.width; outgoing.height = canvas.height;
      const outgoingContext = outgoing.getContext('2d')!;
      outgoingContext.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);
      paintCaptions(outgoingContext, sourceVideo.currentTime * 1000, canvas.width, canvas.height);
      if (clip) { clip.volume = endingAudioLevel(ending, 0, duration); await clip.play(); }
      abort.signal.throwIfAborted();
      const start = performance.now();
      let lastTime = 0, lastAdvance = start;
      await new Promise<void>((resolve, reject) => {
        const aborted = () => { abort.signal.removeEventListener('abort', aborted); reject(abort.signal.reason); };
        abort.signal.addEventListener('abort', aborted, {once: true});
        const tick = () => {
          try {
            const time = clip ? clip.currentTime : Math.min(duration, (performance.now() - start) / 1000);
            if (time !== lastTime) { lastTime = time; lastAdvance = performance.now(); }
            if (clip?.error || performance.now() - lastAdvance > 30_000) throw new Error('Ending playback stalled. Choose another clip or retry.');
            drawVideoEnding(context, outgoing, ending, time, duration, canvas.width, canvas.height, clip);
            if (clip) clip.volume = endingAudioLevel(ending, time, duration);
            status.textContent = `Ending preview · ${time.toFixed(1)} / ${duration.toFixed(1)} seconds`;
            if (clip ? clip.ended : time >= duration) { abort.signal.removeEventListener('abort', aborted); resolve(); }
            else frame = requestAnimationFrame(tick);
          } catch (error) { abort.signal.removeEventListener('abort', aborted); reject(error); }
        };
        tick();
      });
      status.textContent = 'Preview complete. The ending will be included in your rendered video.';
    } catch (error) {
      if (!abort.signal.aborted) status.textContent = error instanceof Error ? error.message : 'Could not preview ending.';
    } finally {
      cleanup();
      if (controller === abort) { controller = undefined; play.textContent = 'Preview ending & transition'; }
    }
  }
  play.addEventListener('click', () => void preview());
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  refresh();
  return {getEnding: settings, pause};
}
