import { createLocalAIClient, LocalAIError } from '@ocno/oryn-sdk';
import { openAudio, type OpenedAudio } from './audio.ts';
import { transcribeAudio } from './transcription.ts';
import {
  LANGUAGES, modelsForLanguage, defaultModelForLanguage, getSelection, buildModelRequest,
  type TranscriptionSelection,
} from './models.ts';

import './style.css';

const localAI = createLocalAIClient({ showOrynBrowserAlert: true });
const form = required<HTMLFormElement>('#transcription-form');
const fileInput = required<HTMLInputElement>('#audio-file');
const languageInput = required<HTMLSelectElement>('#language');
const modelInput = required<HTMLSelectElement>('#model');
const modelFields = required<HTMLElement>('#model-fields');
const recordingFields = required<HTMLElement>('#recording-fields');
const modelDescription = required<HTMLElement>('#model-description');
const modelSource = required<HTMLAnchorElement>('#model-source');
const submit = required<HTMLButtonElement>('#transcribe');
const cancel = required<HTMLButtonElement>('#cancel');
const download = required<HTMLButtonElement>('#download');
const status = required<HTMLElement>('#status');
const progress = required<HTMLProgressElement>('#progress');
const transcript = required<HTMLElement>('#transcript');
const details = required<HTMLElement>('#details');
const elapsed = required<HTMLElement>('#elapsed');
let active: AbortController | undefined;
let latestText = '';
let transcriptName = 'transcript.txt';

for (const language of LANGUAGES) languageInput.add(new Option(language.label, language.id));
const capabilities = await localAI.getCapabilities();
status.textContent = capabilities.runtimeAvailable
  ? 'Ready. Choose the recording language to begin.'
  : 'Oryn Local AI is unavailable. Open this page in Oryn Browser.';
updateSubmit();

languageInput.addEventListener('change', () => {
  const models = languageInput.value ? modelsForLanguage(languageInput.value) : [];
  modelInput.replaceChildren(...models.map(model => new Option(model.label, model.id)));
  const hasLanguage = models.length > 0;
  modelFields.hidden = !hasLanguage;
  recordingFields.hidden = !hasLanguage;
  modelInput.disabled = !hasLanguage;
  fileInput.disabled = !hasLanguage;
  if (hasLanguage) modelInput.value = defaultModelForLanguage(languageInput.value).id;
  updateModelDescription();
  updateSubmit();
});
modelInput.addEventListener('change', () => {
  updateModelDescription();
  updateSubmit();
});
languageInput.disabled = false;

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  const selection = currentSelection();
  if (!file || !selection || active || !capabilities.runtimeAvailable) return;
  void transcribe(file, selection);
});
cancel.addEventListener('click', () => {
  if (!active) return;
  cancel.disabled = true;
  status.textContent = 'Cancelling transcription…';
  active.abort();
});
download.addEventListener('click', () => {
  if (!latestText) return;
  const url = URL.createObjectURL(new Blob([latestText + '\n'], { type: 'text/plain;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = transcriptName;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
});

async function transcribe(file: File, selection: TranscriptionSelection): Promise<void> {
  const controller = new AbortController();
  active = controller;
  submit.disabled = true;
  fileInput.disabled = true;
  languageInput.disabled = true;
  modelInput.disabled = true;
  cancel.hidden = false;
  cancel.disabled = false;
  download.disabled = true;
  latestText = '';
  transcriptName = `${file.name.replace(/\.[^.]+$/, '') || 'transcript'}.txt`;
  transcript.textContent = '';
  details.textContent = '';
  progress.hidden = false;
  progress.removeAttribute('value');
  elapsed.hidden = false;
  let source: OpenedAudio | undefined;
  let sessionId: string | undefined;
  let completedSeconds = 0;
  const started = performance.now();
  const updateElapsed = () => {
    elapsed.textContent = `Elapsed ${formatDuration((performance.now() - started) / 1_000)}${source ? ` · Processed ${formatDuration(completedSeconds)} / ${formatDuration(source.durationSeconds)}` : ''}`;
  };
  updateElapsed();
  const timer = setInterval(updateElapsed, 1_000);
  try {
    status.textContent = 'Reading recording…';
    source = await openAudio(file, controller.signal);
    controller.signal.throwIfAborted();
    details.textContent = `${file.name} · ${formatDuration(source.durationSeconds)} · ${formatBytes(file.size)} · Language: ${selection.language.label}`;
    progress.max = source.durationSeconds;
    progress.value = 0;
    status.textContent = `Requesting ${selection.model.label}…`;
    const session = await localAI.models.request(
      buildModelRequest(selection, capabilities.features.taskProfiles === true, capabilities.features.modelSelection === true),
      {
        signal: controller.signal,
        onProgress: (value) => {
          if (!controller.signal.aborted) status.textContent = modelProgress(value);
        },
      },
    );
    sessionId = session.sessionId;
    controller.signal.throwIfAborted();
    if (session.resolution.resolvedModel.repository !== selection.model.repository) {
      throw new Error(`Oryn selected ${session.resolution.resolvedModel.repository} instead of ${selection.model.repository}. Choose ${selection.model.label} in Oryn, or change the model on this page`);
    }
    const modelWindowSeconds = session.taskProfile?.limits.audioDurationSeconds;
    if (modelWindowSeconds !== undefined && modelWindowSeconds < 30) {
      await source.dispose();
      source = await openAudio(file, controller.signal, modelWindowSeconds);
    }
    details.textContent += ` · Model: ${session.resolution.resolvedModel.repository}`;
    const result = await transcribeAudio(localAI, session, source, controller.signal, {
      onProgress: (window, index, total, stage) => {
        if (controller.signal.aborted) return;
        status.textContent = `Section ${index + 1} of ${total} (${formatDuration(window.startSeconds)}–${formatDuration(window.endSeconds)}) · ${stageLabel(stage)}`;
      },
      onTranscript: (text, processedSeconds) => {
        latestText = text;
        completedSeconds = processedSeconds;
        transcript.textContent = text;
        download.disabled = !text;
        progress.value = processedSeconds;
        updateElapsed();
      },
    }, selection.inputLanguage);
    transcript.textContent = result.text || 'No speech was detected.';
    if (selection.language.id === 'auto' && result.language) details.textContent += ` · Detected: ${result.language}`;
    status.textContent = `Transcription complete. Processed ${formatDuration(source.durationSeconds)} of audio.`;
  } catch (error) {
    const partial = completedSeconds > 0
      ? ` Completed text through ${formatDuration(completedSeconds)} is still available.`
      : '';
    if (controller.signal.aborted || (error instanceof DOMException && error.name === 'AbortError') ||
      (error instanceof LocalAIError && error.code === 'REQUEST_ABORTED')) {
      status.textContent = `Transcription cancelled.${partial}`;
    } else {
      const message = error instanceof Error ? error.message : 'Unknown error';
      status.textContent = `Could not transcribe: ${message}${/[.!?]$/.test(message) ? '' : '.'}${partial}`;
    }
  } finally {
    clearInterval(timer);
    updateElapsed();
    await source?.dispose().catch(() => undefined);
    if (sessionId) await localAI.models.releaseSession(sessionId).catch(() => undefined);
    active = undefined;
    languageInput.disabled = false;
    modelInput.disabled = false;
    fileInput.disabled = false;
    updateSubmit();
    cancel.hidden = true;
    if (completedSeconds === 0) progress.hidden = true;
  }
}

function currentSelection(): TranscriptionSelection | undefined {
  try { return getSelection(languageInput.value, modelInput.value); }
  catch { return undefined; }
}

function updateSubmit(): void {
  submit.disabled = !capabilities.runtimeAvailable || Boolean(active) || !currentSelection();
}

function updateModelDescription(): void {
  const selection = currentSelection();
  modelDescription.textContent = selection?.model.description ?? '';
  modelSource.textContent = selection?.model.repository ?? '';
  if (selection) modelSource.href = `https://huggingface.co/${selection.model.repository}`;
  else modelSource.removeAttribute('href');
}

function modelProgress(progress: { stage: string; downloadedBytes?: number; totalBytes?: number }): string {
  if (progress.stage === 'awaiting-permission') return 'Waiting for your permission in Oryn…';
  if (progress.stage === 'awaiting-model-choice') return 'Choose a transcription model in Oryn…';
  if (progress.stage === 'downloading') {
    const downloaded = formatBytes(progress.downloadedBytes ?? 0);
    return `Downloading transcription model locally — ${downloaded}${progress.totalBytes ? ` / ${formatBytes(progress.totalBytes)}` : ''}`;
  }
  if (progress.stage === 'loading') return 'Loading transcription model into memory…';
  if (progress.stage === 'ready') return 'Transcription model is ready.';
  return 'Resolving transcription model…';
}

function stageLabel(stage: string): string {
  if (stage.startsWith('queued:')) return `Waiting in queue (${stage.slice(7)} ahead)…`;
  const labels: Record<string, string> = {
    'decoding-audio': 'Decoding audio…',
    'preparing-inputs': 'Preparing audio…',
    uploading: 'Preparing audio…',
    'copying-audio': 'Preparing audio…',
    queued: 'Waiting in queue…',
    transcribing: 'Transcribing locally…',
    'decoding-text': 'Preparing transcript…',
  };
  return labels[stage] ?? 'Working locally…';
}

function formatDuration(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3_600);
  const minutes = Math.floor(whole / 60) % 60;
  return `${hours ? `${hours}:${String(minutes).padStart(2, '0')}` : minutes}:${String(whole % 60).padStart(2, '0')}`;
}

function formatBytes(value: number): string {
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}
