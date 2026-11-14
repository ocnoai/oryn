import {
  ALL_FORMATS, AudioSample, AudioSampleSink, AudioSampleSource, BlobSource, BufferTarget,
  CanvasSink, CanvasSource, Input, Mp4OutputFormat, Output, Quality, WebMOutputFormat,
  canEncodeAudio, canEncodeVideo, type InputAudioTrack, type InputVideoTrack,
} from 'mediabunny';
import type { VideoRenderOptions, drawCaptions } from './video-renderer.ts';
import type { drawVideoEnding, endingAudioLevel } from './video-ending.ts';

const FPS = 30;
const AUDIO_RATE = 48_000;
const AUDIO_CHANNELS = 2;
type Clip = { video: InputVideoTrack; audio: InputAudioTrack | null; duration: number };
type RenderHelpers = {drawCaptions: typeof drawCaptions; drawVideoEnding: typeof drawVideoEnding;
  endingAudioLevel: typeof endingAudioLevel; maxBytes: number};

/** Encode exact media timestamps, independently of playback, timers and page repainting. */
export async function renderVideoFrames(options: VideoRenderOptions, helpers: RenderHelpers): Promise<Blob> {
  const {file, cues, canvas, style, design, ending, signal, onProgress} = options;
  // Receive the shared painters without importing the application entry chunk back
  // into this lazily loaded encoder module.
  const {drawCaptions, drawVideoEnding, endingAudioLevel, maxBytes} = helpers;
  const inputs: Input[] = [];
  let output: Output | undefined;
  let cancellation: Promise<void> | undefined;
  // A posted task lets Cancel and other UI events run even when encoders finish synchronously.
  const tasks = new MessageChannel();
  const yieldTask = () => new Promise<void>(resolve => { tasks.port1.onmessage = () => resolve(); tasks.port2.postMessage(null); });
  const abort = () => {
    inputs.forEach(input => input.dispose());
    cancellation ??= output?.cancel().catch(() => undefined);
  };
  signal.addEventListener('abort', abort, {once: true});
  try {
    signal.throwIfAborted();
    const read = async (file: File, label: string, includeAudio: boolean): Promise<Clip> => {
      const input = new Input({source: new BlobSource(file), formats: ALL_FORMATS});
      inputs.push(input);
      try {
        const video = await input.getPrimaryVideoTrack();
        const audio = await input.getPrimaryAudioTrack();
        if (!video || !await video.canDecode()) throw new Error('No supported video track.');
        if (includeAudio && audio && !await audio.canDecode()) throw new Error('The audio codec is unsupported.');
        const duration = await input.computeDuration(audio ? [video, audio] : [video]);
        if (!Number.isFinite(duration) || duration <= 0) throw new Error('No playable duration.');
        signal.throwIfAborted();
        return {video, audio: includeAudio ? audio : null, duration};
      } catch (error) {
        signal.throwIfAborted();
        throw new Error(`${label} cannot be decoded by the browser. Choose another video. ${error instanceof Error ? error.message : ''}`);
      }
    };
    const source = await read(file, 'The source video', true);
    const endClip = ending?.kind === 'video' ? await read(ending.file, 'The ending video', ending.includeAudio) : undefined;
    const endingDuration = ending?.kind === 'animation' ? ending.duration : endClip?.duration ?? 0;
    const duration = source.duration + endingDuration;
    if (duration > 1800) throw new Error('Video rendering is limited to 30 minutes per export, including the ending.');
    if (cues.at(-1)!.endMs > source.duration * 1000 + 100)
      throw new Error('A caption ends after the video. Correct its timing before rendering.');
    const width = await source.video.getDisplayWidth(), height = await source.video.getDisplayHeight();
    if (!width || !height) throw new Error('The source has no playable video dimensions.');
    const scale = Math.min(1, 1920 / width, 1080 / height);
    canvas.width = Math.max(2, Math.floor(width * scale / 2) * 2);
    canvas.height = Math.max(2, Math.floor(height * scale / 2) * 2);
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas rendering is unavailable.');

    const videoQuality = new Quality({bitrate: 4_000_000});
    const audioQuality = new Quality({bitrate: 128_000});
    const videoConfig = {width: canvas.width, height: canvas.height, frameRate: FPS, quality: videoQuality};
    const audioConfig = {numberOfChannels: AUDIO_CHANNELS, sampleRate: AUDIO_RATE, quality: audioQuality};
    const mp4 = await canEncodeVideo('avc', videoConfig) && await canEncodeAudio('aac', audioConfig);
    if (!mp4 && !(await canEncodeVideo('vp8', videoConfig) && await canEncodeAudio('opus', audioConfig)))
      throw new Error('This browser cannot encode MP4 or WebM for background rendering. Open the demo in an up-to-date Oryn Browser.');
    signal.throwIfAborted();
    const target = new BufferTarget();
    target.on('write', ({end}) => {
      if (end > maxBytes) throw new Error('The rendered video exceeds the 256 MB demo limit. Use a shorter source.');
    });
    // Write packets as they arrive instead of buffering the whole movie in the muxer.
    output = new Output({target, format: mp4 ? new Mp4OutputFormat({fastStart: false}) : new WebMOutputFormat()});
    const frames = new CanvasSource(canvas, {codec: mp4 ? 'avc' : 'vp8', quality: videoQuality});
    const audio = new AudioSampleSource({codec: mp4 ? 'aac' : 'opus', quality: audioQuality});
    output.addVideoTrack(frames, {frameRate: FPS});
    output.addAudioTrack(audio);
    await output.start();
    signal.throwIfAborted();
    onProgress(0);

    const outgoing = document.createElement('canvas');
    outgoing.width = canvas.width; outgoing.height = canvas.height;
    const renderSegment = async (clip: Clip | undefined, length: number, offset: number, isEnding: boolean) => {
      // Use one frame grid across both clips so fractional clip durations never
      // create duplicate timestamps at the join. The final frame holds to 1/FPS.
      const firstIndex = Math.ceil(offset * FPS - 1e-8);
      const count = Math.ceil((offset + length) * FPS - 1e-8) - firstIndex;
      const firstFrame = clip ? await clip.video.getFirstTimestamp() : 0;
      function* timestamps() { for (let i = 0; i < count; i++) yield Math.max((firstIndex + i) / FPS - offset, firstFrame); }
      const pictures = clip ? new CanvasSink(clip.video, {width: canvas.width, height: canvas.height, fit: 'contain', poolSize: 2})
        .canvasesAtTimestamps(timestamps()) : undefined;
      const samples = segmentAudio(clip?.audio ?? null, length, offset, signal,
        isEnding && ending ? time => endingAudioLevel(ending, time, length) : undefined);
      let nextAudio: IteratorResult<AudioSample> | undefined;
      let lastPicture: HTMLCanvasElement | OffscreenCanvas | undefined;
      try {
        nextAudio = await samples.next();
        for (let index = 0; index < count; index++) {
          signal.throwIfAborted();
          const timestamp = (firstIndex + index) / FPS;
          const time = timestamp - offset;
          const frameDuration = 1 / FPS;
          if (pictures) {
            const picture = await pictures.next();
            if (picture.done || !picture.value) throw new Error('A video frame could not be decoded.');
            lastPicture = picture.value.canvas;
          }
          if (isEnding && ending) drawVideoEnding(context, outgoing, ending, time, length, canvas.width, canvas.height, lastPicture);
          else {
            context.drawImage(lastPicture!, 0, 0, canvas.width, canvas.height);
            drawCaptions(context, cues, time * 1000, canvas.width, canvas.height, style, design);
          }
          await frames.add(timestamp, frameDuration);
          // Interleave audio with video to bound the muxer's pending packet queue.
          while (!nextAudio.done && nextAudio.value.timestamp < offset + time + frameDuration) {
            signal.throwIfAborted();
            await audio.add(nextAudio.value);
            nextAudio = await samples.next();
          }
          onProgress(Math.min(0.99, (offset + time + frameDuration) / duration * 0.99));
          if (index % FPS === 0) await yieldTask();
        }
        while (!nextAudio.done) {
          signal.throwIfAborted();
          await audio.add(nextAudio.value);
          nextAudio = await samples.next();
        }
        if (!isEnding && ending && lastPicture) {
          const snapshot = outgoing.getContext('2d')!;
          snapshot.drawImage(lastPicture, 0, 0, canvas.width, canvas.height);
          drawCaptions(snapshot, cues, length * 1000, canvas.width, canvas.height, style, design);
        }
      } finally {
        await samples.return(undefined);
        await pictures?.return(undefined);
      }
    };
    await renderSegment(source, source.duration, 0, false);
    if (ending) await renderSegment(endClip, endingDuration, source.duration, true);
    signal.throwIfAborted();
    await output.finalize();
    signal.throwIfAborted();
    if (!target.buffer?.byteLength) throw new Error('The encoder produced an empty video.');
    const blob = new Blob([target.buffer], {type: mp4 ? 'video/mp4' : 'video/webm'});
    onProgress(1);
    return blob;
  } catch (error) {
    signal.throwIfAborted();
    throw error;
  } finally {
    signal.removeEventListener('abort', abort);
    inputs.forEach(input => input.dispose());
    if (output && output.state !== 'finalized' && output.state !== 'canceled') await output.cancel().catch(() => undefined);
    await cancellation;
    tasks.port1.close(); tasks.port2.close();
  }
}

/** Resample bounded windows to one audio format, preserving gaps and clip offsets. */
async function* segmentAudio(track: InputAudioTrack | null, duration: number, offset: number, signal: AbortSignal,
  gain?: (time: number) => number): AsyncGenerator<AudioSample, void> {
  const sampleRate = track ? await track.getSampleRate() : AUDIO_RATE;
  const channels = track ? await track.getNumberOfChannels() : AUDIO_CHANNELS;
  const reader = track ? new AudioSampleSink(track).samples(undefined, duration) : undefined;
  const retained: AudioSample[] = [];
  let done = !reader;
  const totalFrames = Math.round(duration * AUDIO_RATE);
  // Include neighboring samples so native resampling does not introduce clicks at window edges.
  const padding = 256;
  try {
    for (let frame = 0; frame < totalFrames; frame += AUDIO_RATE) {
      signal.throwIfAborted();
      const count = Math.min(AUDIO_RATE, totalFrames - frame);
      const startTime = frame / AUDIO_RATE;
      const endTime = (frame + count) / AUDIO_RATE;
      let normalized: AudioBuffer;
      if (track) {
        const start = Math.floor(startTime * sampleRate) - padding;
        const end = Math.ceil(endTime * sampleRate) + padding;
        const buffer = new AudioBuffer({numberOfChannels: channels, sampleRate, length: end - start});
        while (!done && (!retained.length || retained.at(-1)!.timestamp + retained.at(-1)!.duration < end / sampleRate)) {
          signal.throwIfAborted();
          const next = await reader!.next();
          if (next.done) done = true;
          else retained.push(next.value);
        }
        for (const sample of retained) {
          if (sample.sampleRate !== sampleRate || sample.numberOfChannels !== channels)
            throw new Error('Audio format changed within a clip. Convert it to a single audio format before rendering.');
          const position = Math.round(sample.timestamp * sampleRate);
          const first = Math.max(start, position), last = Math.min(end, position + sample.numberOfFrames);
          if (last <= first) continue;
          for (let channel = 0; channel < channels; channel++) {
            sample.copyTo(buffer.getChannelData(channel).subarray(first - start, last - start),
              {format: 'f32-planar', planeIndex: channel, frameOffset: first - position, frameCount: last - first});
          }
        }
        const mix = new OfflineAudioContext(AUDIO_CHANNELS, count, AUDIO_RATE);
        const source = mix.createBufferSource();
        source.buffer = buffer;
        source.connect(mix.destination);
        source.start(0, startTime - start / sampleRate);
        try { normalized = await mix.startRendering(); }
        finally { source.disconnect(); source.buffer = null; }
        const nextWindowStart = (Math.floor(endTime * sampleRate) - padding) / sampleRate;
        while (retained.length && retained[0]!.timestamp + retained[0]!.duration <= nextWindowStart) retained.shift()!.close();
      } else {
        normalized = new AudioBuffer({numberOfChannels: AUDIO_CHANNELS, sampleRate: AUDIO_RATE, length: count});
      }
      signal.throwIfAborted();
      if (gain) {
        for (let channel = 0; channel < AUDIO_CHANNELS; channel++) {
          const values = normalized.getChannelData(channel);
          for (let index = 0; index < values.length; index++) values[index] *= gain(startTime + index / AUDIO_RATE);
        }
      }
      const samples = AudioSample.fromAudioBuffer(normalized, offset + startTime);
      try { for (const sample of samples) yield sample; }
      finally { samples.forEach(sample => sample.close()); }
    }
  } finally {
    retained.forEach(sample => sample.close());
    await reader?.return(undefined);
  }
}
