import { AudioSampleSink, BlobSource, Input, MP3, WAVE, type AudioSample } from 'mediabunny';

export const SAMPLE_RATE = 16_000;
export const MAX_DURATION_SECONDS = 4 * 60 * 60;
export const MAX_FILE_BYTES = 1024 ** 3;
const WINDOW_SECONDS = 30;
const OVERLAP_SECONDS = 2;
const RESAMPLING_PADDING = 256;

export interface AudioWindow {
  index: number;
  startSeconds: number;
  endSeconds: number;
}

export interface OpenedAudio {
  durationSeconds: number;
  windows: AudioWindow[];
  /** Read windows in their planned order; decoded audio is retained only until the next window. */
  readWindow(window: AudioWindow): Promise<Blob>;
  dispose(): Promise<void>;
}

export function planAudioWindows(durationSeconds: number, maxWindowSeconds = WINDOW_SECONDS): AudioWindow[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    throw new Error('This recording has no playable audio.');
  }
  if (durationSeconds > MAX_DURATION_SECONDS) {
    throw new Error('Choose a recording lasting up to 4 hours.');
  }
  if (!Number.isFinite(maxWindowSeconds) || maxWindowSeconds <= 0 || maxWindowSeconds > WINDOW_SECONDS) {
    throw new Error('Audio sections must last between 0 and 30 seconds.');
  }
  const overlapSeconds = Math.min(OVERLAP_SECONDS, maxWindowSeconds / 10);
  const windows: AudioWindow[] = [];
  for (let index = 0; index * (maxWindowSeconds - overlapSeconds) < durationSeconds; index++) {
    const startSeconds = index * (maxWindowSeconds - overlapSeconds);
    const endSeconds = Math.min(startSeconds + maxWindowSeconds, durationSeconds);
    windows.push({ index: windows.length, startSeconds, endSeconds });
    if (endSeconds === durationSeconds) break;
  }
  return windows;
}

/** Read from the File in bounded slices; never expand a full recording into an AudioBuffer. */
export async function openAudio(file: File, signal: AbortSignal, maxWindowSeconds = WINDOW_SECONDS): Promise<OpenedAudio> {
  signal.throwIfAborted();
  if (!file.size) throw new Error('This recording is empty. Choose an MP3 or WAV file.');
  if (file.size > MAX_FILE_BYTES) throw new Error('Choose an MP3 or WAV file smaller than 1 GB.');
  const input = new Input({
    source: new BlobSource(file, { maxCacheSize: 8 * 1024 ** 2 }),
    formats: [MP3, WAVE],
  });
  const retained: AudioSample[] = [];
  let reader: AsyncGenerator<AudioSample, void, unknown> | undefined;
  let disposal: Promise<void> | undefined;
  let disposed = false;
  let ended = false;
  let nextIndex = 0;
  let reading = false;

  const dispose = (): Promise<void> => {
    if (disposal) return disposal;
    disposed = true;
    signal.removeEventListener('abort', abort);
    input.dispose();
    retained.splice(0).forEach(sample => sample.close());
    disposal = reader?.return(undefined).then(() => undefined).catch(() => undefined) ?? Promise.resolve();
    return disposal;
  };
  const abort = () => { void dispose(); };
  const checkpoint = () => {
    signal.throwIfAborted();
    if (disposed) throw new Error('This recording has been closed.');
  };
  signal.addEventListener('abort', abort, { once: true });

  try {
    // Format detection uses the actual file contents, not its extension or MIME type.
    try { await input.getFormat(); }
    catch {
      checkpoint();
      throw new Error('Could not read this recording. Choose a valid MP3 or WAV file.');
    }
    checkpoint();
    const track = await input.getPrimaryAudioTrack();
    checkpoint();
    if (!track) throw new Error('This recording has no audio track.');
    if (!await track.canDecode()) {
      throw new Error('This audio codec cannot be decoded here. Open the demo in an up-to-date Oryn Browser, or choose a PCM WAV recording.');
    }
    checkpoint();
    // This scans packet headers for variable-bitrate MP3s, without decoding their audio.
    const durationSeconds = await track.computeDuration();
    checkpoint();
    const windows = planAudioWindows(durationSeconds, maxWindowSeconds);
    const sampleRate = await track.getSampleRate();
    const channels = await track.getNumberOfChannels();
    checkpoint();
    if (!Number.isFinite(sampleRate) || sampleRate < 8_000 || sampleRate > 192_000 ||
      !Number.isInteger(channels) || channels < 1 || channels > 8) {
      throw new Error('Choose audio with 1–8 channels and a sample rate between 8 and 192 kHz.');
    }
    reader = new AudioSampleSink(track).samples();

    return {
      durationSeconds,
      windows,
      dispose,
      async readWindow(window): Promise<Blob> {
        checkpoint();
        if (reading || window !== windows[nextIndex]) {
          throw new Error('Audio sections must be read one at a time, in order.');
        }
        reading = true;
        try {
          // Retain neighboring source frames so resampling preserves the sound at section edges.
          const start = Math.floor(window.startSeconds * sampleRate) - RESAMPLING_PADDING;
          const end = Math.ceil(window.endSeconds * sampleRate) + RESAMPLING_PADDING;
          const buffer = new AudioBuffer({ numberOfChannels: channels, sampleRate, length: end - start });
          while (!ended && (!retained.length || retained.at(-1)!.timestamp + retained.at(-1)!.duration < end / sampleRate)) {
            checkpoint();
            const next = await reader!.next();
            // An abort can dispose the input while a decoder read is pending.
            if (signal.aborted || disposed) {
              if (!next.done) next.value.close();
              checkpoint();
            }
            if (next.done) ended = true;
            else retained.push(next.value);
          }
          for (const sample of retained) {
            if (sample.sampleRate !== sampleRate || sample.numberOfChannels !== channels) {
              throw new Error('The audio format changes within this recording. Convert it to a single MP3 or PCM WAV format first.');
            }
            const position = Math.round(sample.timestamp * sampleRate);
            const first = Math.max(start, position);
            const last = Math.min(end, position + sample.numberOfFrames);
            if (last <= first) continue;
            for (let channel = 0; channel < channels; channel++) {
              sample.copyTo(buffer.getChannelData(channel).subarray(first - start, last - start), {
                format: 'f32-planar', planeIndex: channel, frameOffset: first - position, frameCount: last - first,
              });
            }
          }
          checkpoint();
          const count = Math.max(1, Math.min(WINDOW_SECONDS * SAMPLE_RATE,
            Math.round((window.endSeconds - window.startSeconds) * SAMPLE_RATE)));
          const mix = new OfflineAudioContext(1, count, SAMPLE_RATE);
          const source = mix.createBufferSource();
          source.buffer = buffer;
          source.connect(mix.destination);
          source.start(0, window.startSeconds - start / sampleRate);
          let normalized: AudioBuffer;
          try { normalized = await mix.startRendering(); }
          finally { source.disconnect(); source.buffer = null; }
          checkpoint();
          const blob = pcmWave(normalized.getChannelData(0));
          nextIndex++;
          const nextStart = windows[nextIndex]?.startSeconds;
          if (nextStart === undefined) await dispose();
          else {
            const keepFrom = (Math.floor(nextStart * sampleRate) - RESAMPLING_PADDING) / sampleRate;
            while (retained.length && retained[0]!.timestamp + retained[0]!.duration <= keepFrom) {
              retained.shift()!.close();
            }
          }
          return blob;
        } catch (error) {
          await dispose();
          signal.throwIfAborted();
          throw error;
        } finally {
          reading = false;
        }
      },
    };
  } catch (error) {
    await dispose();
    signal.throwIfAborted();
    throw error;
  }
}

/** The model accepts only bounded 16 kHz mono PCM WAV sections. */
export function pcmWave(samples: Float32Array): Blob {
  if (!samples.length || samples.length > WINDOW_SECONDS * SAMPLE_RATE) {
    throw new Error('Audio sections must contain between 1 sample and 30 seconds of audio.');
  }
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const ascii = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index++) view.setUint8(offset + index, text.charCodeAt(index));
  };
  ascii(0, 'RIFF');
  view.setUint32(4, bytes.byteLength - 8, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index++) {
    const value = Math.max(-1, Math.min(1, Number.isFinite(samples[index]) ? samples[index]! : 0));
    view.setInt16(44 + index * 2, Math.round(value * (value < 0 ? 0x8000 : 0x7fff)), true);
  }
  return new Blob([bytes], { type: 'audio/wav' });
}
