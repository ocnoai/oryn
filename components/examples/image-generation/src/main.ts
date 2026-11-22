import {
  createLocalAIClient,
  invokeAndWait,
  LocalAIError,
  type ArtifactResult,
  type LocalAIEvent,
  type ModelRequestProgress,
  type ModelSession,
} from '@ocno/oryn-sdk';

import {
  createImageModelRequest,
  IMAGE_MODEL_PROFILE,
} from './model-config';
import { initializeIcons } from './icons';
import './style.css';

initializeIcons();

interface ImageGenerationInput {
  prompt: string;
  width: number;
  height: number;
  steps: number;
  seed?: number;
}

const localAI = createLocalAIClient({ showOrynBrowserAlert: true });
const form = required<HTMLFormElement>('#image-form');
const prompt = required<HTMLTextAreaElement>('#prompt');
const promptCount = required<HTMLElement>('#prompt-count');
const steps = required<HTMLSelectElement>('#steps');
const useSeed = required<HTMLInputElement>('#use-seed');
const seed = required<HTMLInputElement>('#seed');
const generate = required<HTMLButtonElement>('#generate');
const cancel = required<HTMLButtonElement>('#cancel');
const retry = required<HTMLButtonElement>('#retry');
const status = required<HTMLElement>('#status');
const statusDot = required<HTMLElement>('#status-dot');
const progress = required<HTMLProgressElement>('#progress');
const details = required<HTMLElement>('#details');
const emptyState = required<HTMLElement>('#empty-state');
const previewFrame = required<HTMLElement>('#preview-frame');
const imagePreview = required<HTMLImageElement>('#image-preview');
const previewCaption = required<HTMLElement>('#preview-caption');
const resultActions = required<HTMLElement>('#result-actions');
const save = required<HTMLButtonElement>('#save');
const read = required<HTMLButtonElement>('#read');
const upload = required<HTMLButtonElement>('#upload');
const uploadUrl = required<HTMLInputElement>('#upload-url');
const transferStatus = required<HTMLElement>('#transfer-status');
const errorCard = required<HTMLElement>('#error-card');
const errorTitle = required<HTMLElement>('#error-title');
const errorMessage = required<HTMLElement>('#error-message');
const errorRecovery = required<HTMLElement>('#error-recovery');
const errorCode = required<HTMLElement>('#error-code');
const requestedModel = required<HTMLElement>('#requested-model');
const requestedRevision = required<HTMLElement>('#requested-revision');
const resolvedModel = required<HTMLElement>('#resolved-model');
const substitution = required<HTMLElement>('#substitution');
const runtime = required<HTMLElement>('#runtime');
const backend = required<HTMLElement>('#backend');
const modeLabel = required<HTMLElement>('#mode-label');
const modelAccessNote = required<HTMLElement>('#model-access-note');

let active: AbortController | undefined;
let sessionId: string | undefined;
let artifactId: string | undefined;
let previewArtifactId: string | undefined;
let previewSequence = 0;

const modelRequest = createImageModelRequest();
const selectedRequest = modelRequest.preferredModel;
requestedModel.textContent = selectedRequest.repository;
requestedRevision.textContent =
  selectedRequest.revision ?? 'Browser resolves the revision';
modeLabel.textContent =
  IMAGE_MODEL_PROFILE === 'fixture'
    ? 'Conformance fixture mode'
    : 'Production FLUX request';
modelAccessNote.hidden = IMAGE_MODEL_PROFILE === 'fixture';
document.documentElement.dataset.modelProfile = IMAGE_MODEL_PROFILE;
updatePromptCount();

const capabilities = await localAI.getCapabilities();
if (capabilities.runtimeAvailable) {
  setStatus(
    'ready',
    'Ready. Oryn will resolve a compatible local image model.',
  );
} else {
  setStatus(
    'error',
    'Oryn Local AI is unavailable. Open this page in Oryn Browser.',
  );
}
generate.disabled = !capabilities.runtimeAvailable;

prompt.addEventListener('input', updatePromptCount);
useSeed.addEventListener('change', () => {
  seed.disabled = !useSeed.checked;
});
form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (form.reportValidity()) void generateImage();
});
cancel.addEventListener('click', () => active?.abort());
retry.addEventListener('click', () => {
  if (form.reportValidity()) void generateImage();
});
save.addEventListener('click', () => void saveArtifact());
read.addEventListener('click', () => void readArtifact());
upload.addEventListener('click', () => void uploadArtifact());
uploadUrl.addEventListener('input', () => {
  upload.disabled = !artifactId || !isHttpsUrl(uploadUrl.value);
});
window.addEventListener('pagehide', () => {
  active?.abort();
  void releaseOwnedResources();
});

async function generateImage(): Promise<void> {
  active?.abort();
  await releaseOwnedResources();
  active = new AbortController();
  previewSequence += 1;
  const generationSequence = previewSequence;
  setBusy(true);
  clearResult();
  clearError();
  resetResolution();
  details.textContent = '';
  progress.hidden = false;
  progress.removeAttribute('value');

  try {
    setStatus(
      'working',
      IMAGE_MODEL_PROFILE === 'fixture'
        ? 'Requesting the exact small image conformance model…'
        : 'Requesting exact FLUX.1-dev with compatible substitution enabled…',
    );
    const session = await localAI.models.request(modelRequest, {
      signal: active.signal,
      onProgress: (event) => {
        setStatus('working', modelProgress(event));
        if (event.stage === 'loading' && typeof event.progress === 'number') {
          progress.value = event.progress;
        } else {
          progress.removeAttribute('value');
        }
      },
    });
    sessionId = session.sessionId;
    showResolution(session);
    const [width, height] = selectedSize();
    const input: ImageGenerationInput = {
      prompt: prompt.value.trim(),
      width,
      height,
      steps: Number(steps.value),
      ...(useSeed.checked ? { seed: Number(seed.value) } : {}),
    };
    const generated = await invokeAndWait<ImageGenerationInput, ArtifactResult>(
      localAI,
      { sessionId: session.sessionId, input },
      {
        signal: active.signal,
        onEvent: (event) => handleJobEvent(event, generationSequence),
      },
    );
    if (
      generated.kind !== 'artifact' ||
      generated.artifact.mimeType !== 'image/png'
    ) {
      throw new Error('The image runner returned an unexpected result.');
    }
    artifactId = generated.artifact.id;
    await releasePreview();
    imagePreview.src = await localAI.artifacts.getUrl(artifactId);
    imagePreview.alt = `Generated image: ${prompt.value.trim()}`;
    previewCaption.textContent = `${width} × ${height} PNG · ${formatBytes(generated.artifact.sizeBytes)}`;
    emptyState.hidden = true;
    previewFrame.hidden = false;
    resultActions.hidden = false;
    upload.disabled = !isHttpsUrl(uploadUrl.value);
    progress.value = 1;
    setStatus('complete', 'Image generation complete. The PNG remains local.');
    details.textContent = `${session.resolution.resolvedModel.repository}`;
  } catch (error) {
    await releasePreview();
    const code = localErrorCode(error);
    if (code === 'REQUEST_ABORTED' || code === 'JOB_CANCELLED') {
      setStatus(
        'cancelled',
        'Image generation cancelled. No final artifact was published.',
      );
    } else {
      const presentation = explainError(error);
      setStatus('error', presentation.message);
      showError(presentation);
    }
  } finally {
    active = undefined;
    setBusy(false);
    progress.hidden = true;
  }
}

function handleJobEvent(event: LocalAIEvent, generationSequence: number): void {
  if (event.event === 'job.queue.updated') {
    setStatus('working', `Queued — ${event.data.jobsAhead} job(s) ahead.`);
    details.textContent = `Generation lane position ${event.data.position}`;
  } else if (event.event === 'job.state') {
    if (event.data.state === 'queued' && event.data.queue) {
      setStatus(
        'working',
        `Queued — ${event.data.queue.jobsAhead} job(s) ahead.`,
      );
    } else {
      setStatus('working', jobStateLabel(event.data.state));
    }
  } else if (event.event === 'job.progress') {
    setStatus('working', jobProgress(event.data.stage));
    if (typeof event.data.progress === 'number')
      progress.value = event.data.progress;
  } else if (event.event === 'job.preview') {
    void showPreview(event.data.artifactId, generationSequence);
  }
}

async function showPreview(
  id: string,
  generationSequence: number,
): Promise<void> {
  try {
    const url = await localAI.artifacts.getUrl(id);
    if (generationSequence !== previewSequence || !active) {
      await localAI.artifacts.release(id).catch(() => undefined);
      return;
    }
    if (previewArtifactId && previewArtifactId !== id) {
      await localAI.artifacts.release(previewArtifactId).catch(() => undefined);
    }
    previewArtifactId = id;
    imagePreview.src = url;
    imagePreview.alt = 'Intermediate local image preview';
    previewCaption.textContent = 'Intermediate preview · still generating';
    emptyState.hidden = true;
    previewFrame.hidden = false;
  } catch {
    // A superseded or short-lived preview may disappear before the page resolves its URL.
  }
}

async function saveArtifact(): Promise<void> {
  if (!artifactId) return;
  try {
    await localAI.artifacts.saveAs(artifactId, {
      suggestedFilename: 'oryn-generated-image.png',
    });
    setStatus('complete', 'Trusted Save As flow closed.');
  } catch (error) {
    setStatus('error', `Could not save: ${errorMessageOf(error)}`);
  }
}

async function readArtifact(): Promise<Blob | undefined> {
  if (!artifactId) return undefined;
  try {
    read.disabled = true;
    transferStatus.textContent =
      'Reading the authorized artifact into this page…';
    const blob = await localAI.artifacts.readBlob(artifactId);
    const digest = await crypto.subtle.digest(
      'SHA-256',
      await blob.arrayBuffer(),
    );
    const shortHash = Array.from(new Uint8Array(digest).slice(0, 6), (value) =>
      value.toString(16).padStart(2, '0'),
    ).join('');
    transferStatus.textContent = `Authorized Blob read: ${formatBytes(blob.size)} · image/png · SHA-256 ${shortHash}…`;
    return blob;
  } catch (error) {
    transferStatus.textContent = `Could not read artifact: ${errorMessageOf(error)}`;
    return undefined;
  } finally {
    read.disabled = false;
  }
}

async function uploadArtifact(): Promise<void> {
  if (!artifactId) return;
  try {
    const destination = new URL(uploadUrl.value);
    if (destination.protocol !== 'https:')
      throw new Error('Enter an HTTPS destination.');
    upload.disabled = true;
    transferStatus.textContent =
      'Reading the artifact before the explicit website upload…';
    const blob = await localAI.artifacts.readBlob(artifactId);
    const body = new FormData();
    body.append('file', blob, 'oryn-generated-image.png');
    const response = await fetch(destination, { method: 'POST', body });
    if (!response.ok)
      throw new Error(`Upload failed with HTTP ${response.status}.`);
    transferStatus.textContent = 'Explicit HTTPS upload complete.';
  } catch (error) {
    transferStatus.textContent = `Could not upload: ${errorMessageOf(error)}`;
  } finally {
    upload.disabled = !artifactId || !isHttpsUrl(uploadUrl.value);
  }
}

async function releaseOwnedResources(): Promise<void> {
  previewSequence += 1;
  await releasePreview();
  if (artifactId) {
    await localAI.artifacts.release(artifactId).catch(() => undefined);
    artifactId = undefined;
  }
  if (sessionId) {
    await localAI.models.releaseSession(sessionId).catch(() => undefined);
    sessionId = undefined;
  }
}

async function releasePreview(): Promise<void> {
  if (!previewArtifactId) return;
  const id = previewArtifactId;
  previewArtifactId = undefined;
  await localAI.artifacts.release(id).catch(() => undefined);
}

function showResolution(session: ModelSession): void {
  const resolution = session.resolution;
  resolvedModel.textContent = `${resolution.resolvedModel.repository}@${shortRevision(resolution.resolvedModel.revision)}`;
  substitution.textContent = resolution.substituted
    ? `Yes · ${reasonLabel(resolution.resolutionReason)}`
    : `No · ${reasonLabel(resolution.resolutionReason)}`;
  runtime.textContent = "Local AI session ready";
  backend.textContent = "Managed by Oryn";
  resolvedModel.dataset.substituted = String(resolution.substituted);
}

function resetResolution(): void {
  resolvedModel.textContent = 'Resolving…';
  resolvedModel.removeAttribute('data-substituted');
  substitution.textContent = 'Pending';
  runtime.textContent = 'Pending';
  backend.textContent = 'Pending';
}

function clearResult(): void {
  imagePreview.removeAttribute('src');
  emptyState.hidden = false;
  previewFrame.hidden = true;
  resultActions.hidden = true;
  transferStatus.textContent = '';
  upload.disabled = true;
}

function showError(presentation: ErrorPresentation): void {
  errorTitle.textContent = presentation.title;
  errorMessage.textContent = presentation.message;
  errorRecovery.textContent = presentation.recovery;
  errorCode.textContent = presentation.code
    ? `Error: ${presentation.code}`
    : '';
  errorCard.hidden = false;
}

function clearError(): void {
  errorCard.hidden = true;
  errorCode.textContent = '';
}

function setBusy(busy: boolean): void {
  generate.disabled = busy || !capabilities.runtimeAvailable;
  cancel.hidden = !busy;
  prompt.disabled = busy;
  steps.disabled = busy;
  useSeed.disabled = busy;
  seed.disabled = busy || !useSeed.checked;
  form
    .querySelectorAll<HTMLInputElement>('input[name="size"]')
    .forEach((input) => {
      input.disabled = busy;
    });
}

function setStatus(
  state: 'ready' | 'working' | 'complete' | 'cancelled' | 'error',
  message: string,
): void {
  status.textContent = message;
  statusDot.dataset.state = state;
}

function selectedSize(): [number, number] {
  const selected = required<HTMLInputElement>(
    'input[name="size"]:checked',
  ).value;
  const [width, height] = selected.split('x').map(Number);
  return [width, height];
}

function updatePromptCount(): void {
  promptCount.textContent = `${prompt.value.length} / ${prompt.maxLength}`;
}

function modelProgress(event: ModelRequestProgress): string {
  const { stage } = event;
  const labels: Record<string, string> = {
    resolving: 'Resolving the requested model and compatible alternatives…',
    'awaiting-permission': 'Waiting for your decision in Oryn…',
    'awaiting-model-choice': 'Choose an image model in Oryn…',
    loading: 'Loading the selected image model into memory…',
    ready: 'Image model ready.',
  };
  if (stage === 'downloading') {
    const downloaded = formatBytes(event.downloadedBytes);
    return `Downloading the selected model — ${downloaded}${event.totalBytes ? ` / ${formatBytes(event.totalBytes)}` : ''}`;
  }
  return labels[stage] ?? `Preparing image model — ${stage}…`;
}

function jobStateLabel(state: string): string {
  const labels: Record<string, string> = {
    queued: 'Queued for the global generation lane…',
    loading: 'Loading the selected image model…',
    running: 'Generating the image locally…',
    streaming: 'Receiving a local preview…',
    saving: 'Publishing the private PNG artifact…',
  };
  return labels[state] ?? `Generation state: ${state}`;
}

function jobProgress(stage?: string): string {
  const labels: Record<string, string> = {
    'encoding-prompt': 'Encoding your prompt locally…',
    'preparing-latents': 'Preparing the image canvas…',
    denoising: 'Generating image detail locally…',
    'decoding-image': 'Decoding the final image…',
    'writing-image': 'Writing the PNG artifact…',
    'publishing-artifact': 'Publishing the private image capability…',
  };
  return stage
    ? (labels[stage] ?? `Generating — ${humanize(stage)}…`)
    : 'Generating locally…';
}

interface ErrorPresentation {
  code: string;
  title: string;
  message: string;
  recovery: string;
}

function explainError(error: unknown): ErrorPresentation {
  const code = localErrorCode(error);
  const fallback = errorMessageOf(error);
  if (
    [
      'MODEL_SECURITY_BLOCKED',
      'MODEL_ACCESS_RESTRICTED',
      'MODEL_NOT_FOUND',
    ].includes(code)
  ) {
    return {
      code,
      title: 'FLUX access is not available through Oryn',
      message:
        'The publisher gates this exact FLUX revision, and Oryn did not resolve an allowed compatible alternative.',
      recovery:
        'Review the publisher terms on the linked model page, or choose an available compatible model in Oryn. This website cannot accept terms or handle credentials for you.',
    };
  }
  if (
    [
      'MODEL_MEMORY_INSUFFICIENT',
      'MODEL_OUT_OF_MEMORY',
      'RESOURCE_REQUIREMENTS_UNSATISFIABLE',
      'RESOURCE_LIMIT_EXCEEDED',
    ].includes(code)
  ) {
    return {
      code,
      title: 'Not enough safe memory for this model',
      message:
        'Oryn kept its memory safety headroom and did not start a model that could destabilize the computer.',
      recovery:
        'Close memory-heavy applications, select a smaller compatible model in Oryn, then retry.',
    };
  }
  if (code === 'USER_DENIED' || code === 'PERMISSION_DENIED') {
    return {
      code,
      title: 'Local generation was not allowed',
      message: 'Oryn did not grant this website a model session.',
      recovery:
        'Retry when you are ready, then review the trusted Oryn permission panel.',
    };
  }
  if (code === 'JOB_INPUT_INVALID') {
    return {
      code,
      title: 'The image settings were rejected',
      message:
        'The selected model runtime did not accept one or more prompt, canvas, step, or seed values.',
      recovery: 'Choose one of the bounded presets shown above and retry.',
    };
  }
  return {
    code,
    title: 'Local image generation failed',
    message: fallback,
    recovery:
      'Review the model choice in Oryn and retry. No failed output is treated as a completed image.',
  };
}

function localErrorCode(error: unknown): string {
  return error instanceof LocalAIError ? String(error.code) : '';
}

function errorMessageOf(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error';
}

function reasonLabel(reason: string): string {
  return humanize(reason).replace(/^./, (letter) => letter.toUpperCase());
}

function humanize(value: string): string {
  return value.replaceAll('-', ' ');
}

function shortRevision(revision: string): string {
  return revision.slice(0, 10);
}

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function isHttpsUrl(value: string): boolean {
  try {
    return new URL(value).protocol === 'https:';
  } catch {
    return false;
  }
}

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}
