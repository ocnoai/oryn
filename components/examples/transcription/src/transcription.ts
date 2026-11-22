import type { LocalAIClient, LocalAIEvent, ModelSession } from '@ocno/oryn-sdk';
import type { AudioWindow, OpenedAudio } from './audio.ts';

export interface TranscriptionCallbacks {
  onProgress?: (window: AudioWindow, index: number, total: number, stage: string) => void;
  onTranscript?: (text: string, completedSeconds: number) => void;
}

export interface TranscriptionResult {
  text: string;
  language: string | null;
  completedSeconds: number;
}

/** Processes one bounded audio artifact at a time, preserving completed text on interruption. */
export async function transcribeAudio(
  client: LocalAIClient,
  session: ModelSession,
  source: OpenedAudio,
  signal: AbortSignal,
  callbacks: TranscriptionCallbacks = {},
  language = 'auto',
): Promise<TranscriptionResult> {
  if (typeof language !== 'string' || !language.trim()) {
    throw new Error('Choose a recording language or automatic detection before transcribing.');
  }
  let text = '';
  let detectedLanguage: string | null = null;
  let completedSeconds = 0;
  const limit = session.taskProfile?.limits.audioDurationSeconds ?? 30;
  if (source.windows.some(window => window.endSeconds - window.startSeconds > limit + 1e-6)) {
    throw new Error(`This model accepts only ${limit} seconds of audio per section. Choose a model supporting 30-second sections.`);
  }
  for (const [index, window] of source.windows.entries()) {
    signal.throwIfAborted();
    const progress = (stage: string) => callbacks.onProgress?.(window, index, source.windows.length, stage);
    progress('decoding-audio');
    const wave = await source.readWindow(window);
    signal.throwIfAborted();
    let artifactId: string | undefined;
    try {
      progress('copying-audio');
      const artifact = await client.artifacts.create(wave, {
        lifetime: 'session',
        name: `transcription-${String(index + 1).padStart(4, '0')}.wav`,
      });
      artifactId = artifact.id;
      signal.throwIfAborted();
      progress('transcribing');
      const value = await invokeSection(client, session.sessionId, artifactId, signal, event => {
        if (event.event === 'job.progress') progress(event.data.stage ?? 'transcribing');
        else if (event.event === 'job.queue.updated') progress(`queued:${event.data.jobsAhead}`);
      }, language);
      signal.throwIfAborted();
      text = mergeTranscript(text, value.text);
      detectedLanguage ??= value.language;
      completedSeconds = Math.min(source.durationSeconds, window.endSeconds);
      callbacks.onTranscript?.(text, completedSeconds);
    } finally {
      if (artifactId) await client.artifacts.release(artifactId).catch(() => undefined);
    }
  }
  return { text, language: detectedLanguage, completedSeconds };
}

/** Drops only a matching phrase at adjacent overlapping audio windows. */
export function mergeTranscript(previous: string, next: string): string {
  const before = previous.trim();
  const after = next.trim();
  if (!before) return after;
  if (!after) return before;
  const left = [...before.matchAll(/\S+/gu)];
  const right = [...after.matchAll(/\S+/gu)];
  const normalize = (word: string) => word.toLocaleLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  // Two seconds of overlap cannot reasonably contain more than forty spoken words.
  const maximum = Math.min(40, left.length, right.length);
  for (let length = maximum; length >= 2; length--) {
    if (right.slice(0, length).every((word, index) => {
      const normalized = normalize(word[0]);
      return normalized.length > 0 && normalized === normalize(left[left.length - length + index]![0]);
    })) {
      const remainder = after.slice(right[length - 1]!.index! + right[length - 1]![0].length).trimStart();
      return remainder ? `${before} ${remainder}` : before;
    }
  }
  return `${before} ${after}`;
}

async function invokeSection(
  client: LocalAIClient,
  sessionId: string,
  artifactId: string,
  signal: AbortSignal,
  onEvent: (event: LocalAIEvent) => void,
  language: string,
): Promise<{ text: string; language: string | null }> {
  signal.throwIfAborted();
  let jobId: string | undefined;
  let cancelled = false;
  let settled = false;
  const buffered: LocalAIEvent[] = [];
  let settle!: (event: LocalAIEvent | undefined) => void;
  const terminal = new Promise<LocalAIEvent | undefined>(resolve => { settle = resolve; });
  const finish = (event?: LocalAIEvent) => {
    if (settled) return;
    settled = true;
    settle(event);
  };
  const cancel = () => {
    if (jobId && !cancelled) {
      cancelled = true;
      void client.cancel(jobId).catch(() => undefined);
    }
    finish();
  };
  const accept = (event: LocalAIEvent) => {
    if (event.sessionId !== sessionId || settled) return;
    if (!jobId) {
      // Progress events before invoke resolves are dispensable. Retain only terminal events.
      if (['job.completed', 'job.failed', 'job.cancelled'].includes(event.event)) buffered.push(event);
      return;
    }
    if (event.jobId !== jobId) return;
    onEvent(event);
    if (['job.completed', 'job.failed', 'job.cancelled'].includes(event.event)) finish(event);
  };
  const unsubscribe = client.subscribe(accept);
  signal.addEventListener('abort', cancel, { once: true });
  try {
    signal.throwIfAborted();
    const job = await client.invoke({
      sessionId,
      input: { audioArtifactId: artifactId, language, timestamps: false },
    });
    jobId = job.jobId;
    // Abort may have arrived while invoke was assigning its job identifier.
    if (signal.aborted) {
      cancel();
      signal.throwIfAborted();
    }
    for (const event of buffered) accept(event);
    const event = await terminal;
    signal.throwIfAborted();
    if (!event || event.event === 'job.cancelled') {
      throw new DOMException('Transcription cancelled.', 'AbortError');
    }
    if (event.event === 'job.failed') throw new Error(event.data.error.message);
    if (event.event !== 'job.completed') throw new Error('The runner returned an unexpected result.');
    const result = event.data.result;
    if (result.kind !== 'structured' || !isTranscript(result.value)) {
      throw new Error('The runner returned an unexpected transcript.');
    }
    return { text: result.value.text, language: result.value.language ?? null };
  } finally {
    signal.removeEventListener('abort', cancel);
    unsubscribe();
  }
}

function isTranscript(value: unknown): value is { text: string; language?: string | null } {
  return typeof value === 'object' && value !== null && 'text' in value && typeof value.text === 'string'
    && (!('language' in value) || value.language === null || typeof value.language === 'string');
}
