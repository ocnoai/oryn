import {
  createLocalAIClient,
  invokeAndWait,
  LocalAIError,
  type ArtifactResult,
  type ModelRequest,
} from '@ocno/oryn-sdk';

import './style.css';

const localAI = createLocalAIClient({ showOrynBrowserAlert: true });
const form = required<HTMLFormElement>('#song-form');
const prompt = required<HTMLTextAreaElement>('#prompt');
const duration = required<HTMLSelectElement>('#duration');
const exactModel = required<HTMLInputElement>('#exact-model');
const generate = required<HTMLButtonElement>('#generate');
const cancel = required<HTMLButtonElement>('#cancel');
const status = required<HTMLElement>('#status');
const progress = required<HTMLProgressElement>('#progress');
const details = required<HTMLElement>('#details');
const result = required<HTMLElement>('#result');
const audio = required<HTMLAudioElement>('#audio');
const save = required<HTMLButtonElement>('#save');
const uploadUrl = required<HTMLInputElement>('#upload-url');
const upload = required<HTMLButtonElement>('#upload');
let active: AbortController | undefined;
let sessionId: string | undefined;
let artifactId: string | undefined;

const miniMaxMusic3 = {
  provider: 'huggingface' as const,
  repository: 'MiniMaxAI/MiniMax-Music3',
  revision: 'fbdf52fbaaca799592917417eb05f1899f1255ec',
};

const musicGenFallback = {
  provider: 'huggingface' as const,
  repository: 'hf-tiny-v2/tiny-random-MusicgenForConditionalGeneration',
  revision: '4606e560821ed4c10d76d6a6513c893ec3cf2ae0',
};

const capabilities = await localAI.getCapabilities();
status.textContent = capabilities.runtimeAvailable
  ? 'Ready. Oryn will choose a compatible local music model.'
  : 'Oryn Local AI is unavailable. Open this page in Oryn Browser.';
generate.disabled = !capabilities.runtimeAvailable;

form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!prompt.value.trim()) return;
  void generateSong();
});
cancel.addEventListener('click', () => active?.abort());
save.addEventListener('click', () => void saveArtifact());
upload.addEventListener('click', () => void uploadArtifact());
window.addEventListener('pagehide', () => {
  active?.abort();
  if (artifactId) void localAI.artifacts.release(artifactId);
  if (sessionId) void localAI.models.releaseSession(sessionId);
});

async function generateSong(): Promise<void> {
  active?.abort();
  active = new AbortController();
  generate.disabled = true;
  cancel.hidden = false;
  result.hidden = true;
  progress.hidden = false;
  progress.removeAttribute('value');
  details.textContent = '';
  if (artifactId) {
    await localAI.artifacts.release(artifactId).catch(() => undefined);
    artifactId = undefined;
    audio.removeAttribute('src');
  }
  try {
    status.textContent = exactModel.checked
      ? 'Requesting MiniMax Music 3…'
      : 'Requesting local text-to-music capability…';
    const modelRequest: ModelRequest = exactModel.checked
      ? {
          role: 'song-generation',
          preferredModel: miniMaxMusic3,
          certifiedModels: [musicGenFallback],
          requirements: {
            task: 'text-to-audio',
            ...(capabilities.features.taskProfiles ? {
              contract: { id: 'text-to-music-v1', revision: 1 },
              parameterValues: { durationSeconds: Number(duration.value) },
            } : {}),
            inputModalities: ['text'],
            outputModalities: ['audio'],
            capabilities: { audioOutput: true },
          },
          substitution: { automatic: 'compatible' as const },
        }
      : {
          role: 'song-generation',
          requirements: {
            task: 'text-to-audio',
            ...(capabilities.features.taskProfiles ? {
              contract: { id: 'text-to-music-v1', revision: 1 },
              parameterValues: { durationSeconds: Number(duration.value) },
            } : {}),
            inputModalities: ['text'],
            outputModalities: ['audio'],
            capabilities: { audioOutput: true },
          },
          substitution: { automatic: 'compatible' as const },
        };
    const session = await localAI.models.request(
      modelRequest,
      {
        signal: active.signal,
        onProgress: (event) => {
          status.textContent = modelProgress(event.stage);
        },
      },
    );
    sessionId = session.sessionId;
    result.dataset.model = session.resolution.resolvedModel.repository;
    details.textContent = `${session.resolution.resolvedModel.repository}`;
    const generated = await invokeAndWait<
      { prompt: string; durationSeconds: number },
      ArtifactResult
    >(
      localAI,
      {
        sessionId: session.sessionId,
        input: {
          prompt: prompt.value.trim(),
          durationSeconds: Number(duration.value),
        },
      },
      {
        signal: active.signal,
        onEvent: (event) => {
          if (event.event === 'job.queue.updated') {
            status.textContent = `Queued — ${event.data.jobsAhead} job(s) ahead.`;
          } else if (event.event === 'job.progress') {
            status.textContent = jobProgress(event.data.stage);
            if (typeof event.data.progress === 'number') progress.value = event.data.progress;
          }
        },
      },
    );
    if (generated.kind !== 'artifact' || generated.artifact.mimeType !== 'audio/wav') {
      throw new Error('The runner returned an unexpected result.');
    }
    artifactId = generated.artifact.id;
    result.dataset.artifactBytes = String(generated.artifact.sizeBytes);
    const capabilityUrl = await localAI.artifacts.getUrl(artifactId);
    audio.src = capabilityUrl;
    audio.load();
    result.hidden = false;
    progress.value = 1;
    status.textContent = 'Song generation complete.';
    details.textContent = [
      details.textContent,
      formatBytes(generated.artifact.sizeBytes),
    ].filter(Boolean).join(' · ');
  } catch (error) {
    status.textContent =
      error instanceof LocalAIError && error.code === 'REQUEST_ABORTED'
        ? 'Song generation cancelled.'
        : `Song generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`;
  } finally {
    active = undefined;
    cancel.hidden = true;
    generate.disabled = !capabilities.runtimeAvailable;
    progress.hidden = true;
  }
}

async function saveArtifact(): Promise<void> {
  if (!artifactId) return;
  try {
    await localAI.artifacts.saveAs(artifactId, { suggestedFilename: 'generated-song.wav' });
    status.textContent = 'Trusted Save As flow completed.';
  } catch (error) {
    status.textContent = `Could not save: ${error instanceof Error ? error.message : 'Unknown error'}`;
  }
}

async function uploadArtifact(): Promise<void> {
  if (!artifactId) return;
  try {
    const destination = new URL(uploadUrl.value);
    if (destination.protocol !== 'https:') throw new Error('Enter an HTTPS destination.');
    upload.disabled = true;
    status.textContent = 'Reading the authorized artifact for explicit upload…';
    const blob = await localAI.artifacts.readBlob(artifactId);
    const body = new FormData();
    body.append('file', blob, 'generated-song.wav');
    const response = await fetch(destination, { method: 'POST', body });
    if (!response.ok) throw new Error(`Upload failed with HTTP ${response.status}.`);
    status.textContent = 'Explicit HTTPS upload complete.';
  } catch (error) {
    status.textContent = `Could not upload: ${error instanceof Error ? error.message : 'Unknown error'}`;
  } finally {
    upload.disabled = false;
  }
}

function modelProgress(stage: string): string {
  const labels: Record<string, string> = {
    resolving: exactModel.checked
      ? 'Resolving MiniMax Music 3 and compatible local alternatives…'
      : 'Resolving compatible local music models…',
    'awaiting-permission': 'Waiting for your permission in Oryn…',
    'awaiting-model-choice': 'Choose a compatible music model in Oryn…',
    downloading: 'Downloading the selected music model…',
    loading: 'Loading the music model…',
    ready: 'Music model ready.',
  };
  return labels[stage] ?? `Preparing music model — ${stage}…`;
}

function jobProgress(stage?: string): string {
  const labels: Record<string, string> = {
    'encoding-prompt': 'Encoding your music prompt…',
    'generating-audio': 'Generating audio locally…',
    'writing-audio': 'Writing the audio artifact…',
    'publishing-artifact': 'Publishing the private audio capability…',
  };
  return stage ? (labels[stage] ?? `Generating — ${stage}…`) : 'Generating locally…';
}

function formatBytes(value: number): string {
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}
