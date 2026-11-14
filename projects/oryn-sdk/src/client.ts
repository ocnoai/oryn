import { SHARED_MODEL_TASK_CONTRACTS, validateSharedTaskInput } from "./shared-task-validation.js";
import { SHARED_TASK_CONTRACTS, type SharedTaskContract } from "./shared-tasks.js";
import { TASK_CONTRACT_REVISIONS } from "./task-contract-revisions.js";
import type {
  BridgeResult,
  InjectedLocalAI,
  LocalAIEvent,
  LocalAIClient,
  LocalAIErrorCode,
  InferenceResult,
  RuntimeCapabilities,
  SerializedLocalAIError,
  ModelInvocationRequest,
  LocalAIClientOptions,
} from "./types.js";

export const SDK_VERSION = "0.1.0";

export class LocalAIError extends Error {
  readonly code: LocalAIErrorCode;
  readonly details?: Record<string, unknown>;

  constructor(error: SerializedLocalAIError) {
    super(error.message);
    this.name = "LocalAIError";
    this.code = error.code;
    this.details = error.details;
  }
}

function unavailableCapabilities(): RuntimeCapabilities {
  return {
    runtimeAvailable: false,
    runtimeState: "unavailable",
    sdkVersion: SDK_VERSION,
    supportedModelProviders: [],
    features: {
      modelRequests: false,
      modelSubstitution: false,
      streaming: false,
      artifacts: false,
      artifactUrls: false,
      cancellation: false,
      queueing: false,
      multipleSubmittedJobs: false,
    },
  };
}

function injectedClient(): InjectedLocalAI | undefined {
  if (typeof window === "undefined") {
    return undefined;
  }
  return (window as Window & { readonly localAI?: InjectedLocalAI }).localAI;
}

export function createLocalAIClient(options: LocalAIClientOptions = {}): LocalAIClient {
  if (options.showOrynBrowserAlert === true) scheduleOrynBrowserAlert();
  const sessionTasks = new Map<string, string>();
  return {
    sdkVersion: SDK_VERSION,
    async getCapabilities() {
      const client = injectedClient();
      if (!client) {
        return unavailableCapabilities();
      }
      return client.getCapabilities();
    },
    models: {
      async request(request, options) {
        const client = injectedClient();
        if (!client?.requestModel) throw unavailableError();
        if (options?.signal?.aborted) throw abortedError();
        if (request.selection || request.preferredModel?.files) {
          if (!(await client.getCapabilities()).features.modelSelection) {
            throw new LocalAIError({ code: "MODEL_NOT_EXECUTABLE", message: "This Oryn browser does not support model selection intents or exact file pins." });
          }
        }
        if (request.requirements?.contract || request.requirements?.requiredFeatures?.length || Object.keys(request.requirements?.parameterValues ?? {}).length) {
          if (!(await client.getCapabilities()).features.taskProfiles) {
            throw new LocalAIError({ code: "MODEL_NOT_EXECUTABLE", message: "This Oryn runtime does not support task contract negotiation." });
          }
        }
        const operationId = createOperationId();
        const stopProgress = client.onModelRequestProgress(
          operationId,
          options?.onProgress ?? (() => undefined),
        );
        const abort = () => void client.cancelModelRequest(operationId);
        options?.signal?.addEventListener("abort", abort, { once: true });
        try {
          const result = await client.requestModel(operationId, request);
          const session = unwrap(result);
          if (session.capabilities.task) sessionTasks.set(session.sessionId, session.capabilities.task);
          const selected = session.taskProfile?.contract;
          if (selected && TASK_CONTRACT_REVISIONS[selected.id] !== selected.revision) {
            await client.releaseModelSession(session.sessionId);
            sessionTasks.delete(session.sessionId);
            throw new LocalAIError({code: "MODEL_NOT_EXECUTABLE", message: "This SDK does not support the negotiated task contract revision."});
          }
          if (selected && (SHARED_TASK_CONTRACTS as readonly string[]).includes(selected.id))
            sessionTasks.set(session.sessionId, selected.id);
          return session;
        } finally {
          stopProgress();
          options?.signal?.removeEventListener("abort", abort);
        }
      },
      async releaseSession(sessionId) {
        const client = injectedClient();
        if (!client?.releaseModelSession) throw unavailableError();
        unwrap(await client.releaseModelSession(sessionId));
        sessionTasks.delete(sessionId);
      },
    },
    artifacts: {
      async create(input, options) {
        const client = injectedClient();
        if (!client?.beginArtifactCreate) throw unavailableError();
        if (!(input instanceof Blob)) {
          throw new LocalAIError({
            code: "ARTIFACT_INVALID",
            message: "Artifact input must be a Blob or File.",
          });
        }
        const fileName =
          options?.name ??
          (typeof File !== "undefined" && input instanceof File
            ? input.name
            : undefined);
        const begin = unwrap(
          await client.beginArtifactCreate({
            mimeType: input.type || "application/octet-stream",
            sizeBytes: input.size,
            name: fileName,
            lifetime: options?.lifetime,
          }),
        );
        try {
          for (
            let offset = 0;
            offset < input.size;
            offset += ARTIFACT_CHUNK_BYTES
          ) {
            const chunk = await input
              .slice(
                offset,
                Math.min(input.size, offset + ARTIFACT_CHUNK_BYTES),
              )
              .arrayBuffer();
            unwrap(await client.appendArtifactChunk(begin.uploadId, chunk));
          }
          return unwrap(await client.finishArtifactCreate(begin.uploadId));
        } catch (error) {
          await client
            .abortArtifactCreate(begin.uploadId)
            .catch(() => undefined);
          throw error;
        }
      },
      async get(artifactId) {
        const client = injectedClient();
        if (!client?.getArtifact) throw unavailableError();
        return unwrap(await client.getArtifact(artifactId));
      },
      async getUrl(artifactId) {
        const client = injectedClient();
        if (!client?.getArtifactUrl) throw unavailableError();
        return unwrap(await client.getArtifactUrl(artifactId));
      },
      async readBlob(artifactId) {
        const client = injectedClient();
        if (!client?.getArtifactUrl) throw unavailableError();
        const url = unwrap(await client.getArtifactUrl(artifactId));
        const response = await fetch(url, {
          credentials: "omit",
          cache: "no-store",
        });
        if (!response.ok) {
          throw new LocalAIError({
            code:
              response.status === 404
                ? "ARTIFACT_NOT_FOUND"
                : "ARTIFACT_ACCESS_DENIED",
            message: "The artifact could not be read.",
          });
        }
        return response.blob();
      },
      async saveAs(artifactId, options) {
        const client = injectedClient();
        if (!client?.saveArtifactAs) throw unavailableError();
        unwrap(await client.saveArtifactAs(artifactId, options));
      },
      async release(artifactId) {
        const client = injectedClient();
        if (!client?.releaseArtifact) throw unavailableError();
        unwrap(await client.releaseArtifact(artifactId));
      },
    },
    async invoke(request) {
      const client = injectedClient();
      if (!client?.invokeJob) throw unavailableError();
      const task = sessionTasks.get(request.sessionId) ?? '';
      const contract = (SHARED_TASK_CONTRACTS as readonly string[]).includes(task) ? task as SharedTaskContract : SHARED_MODEL_TASK_CONTRACTS[task];
      if (contract) validateSharedTaskInput(contract, request.input);
      return unwrap(await client.invokeJob(request));
    },
    async cancel(jobId) {
      const client = injectedClient();
      if (!client?.cancelJob) throw unavailableError();
      unwrap(await client.cancelJob(jobId));
    },
    subscribe(listener) {
      const client = injectedClient();
      return client?.onJobEvent ? client.onJobEvent(listener) : () => undefined;
    },
  };
}

const alertedDocuments = new WeakSet<Document>();
const ORYN_URL = "https://oryn.ocno.ai/";

function scheduleOrynBrowserAlert(): void {
  if (typeof window === "undefined" || typeof document === "undefined") return;
  if (hasCallableOrynBridge(window)) return;
  const owner = document;
  if (alertedDocuments.has(owner)) return;
  alertedDocuments.add(owner);
  const mount = () => {
    if (owner !== document || !owner.body || hasCallableOrynBridge(window)) return;
    mountOrynBrowserAlert(owner);
  };
  if (owner.readyState === "loading") {
    owner.addEventListener("DOMContentLoaded", mount, { once: true });
  } else {
    queueMicrotask(mount);
  }
}

function hasCallableOrynBridge(value: Window): boolean {
  const bridge = (value as Window & { readonly localAI?: unknown }).localAI;
  if (typeof bridge !== "object" || bridge === null) return false;
  const candidate = bridge as Partial<InjectedLocalAI>;
  return (
    typeof candidate.getCapabilities === "function" &&
    typeof candidate.requestModel === "function" &&
    typeof candidate.invokeJob === "function"
  );
}

function mountOrynBrowserAlert(owner: Document): void {
  const previousFocus = owner.activeElement instanceof HTMLElement ? owner.activeElement : null;
  const host = owner.createElement("div");
  host.dataset.orynBrowserAlert = "";
  const shadow = host.attachShadow({ mode: "open" });
  shadow.innerHTML = `<style>
    :host{all:initial;position:fixed;inset:0;z-index:2147483647;display:grid;place-items:center;padding:20px;box-sizing:border-box;background:rgba(8,10,16,.66);color-scheme:light dark;font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    *,*:before,*:after{box-sizing:border-box}
    .card{width:min(430px,100%);padding:24px;border:1px solid #3d4350;border-radius:18px;background:#20232b;color:#f5f6fa;box-shadow:0 24px 70px rgba(0,0,0,.45)}
    .top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:18px}.brand{font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#aeb7ff}
    h2{margin:0 0 10px;font-size:22px;line-height:1.2;letter-spacing:-.02em}p{margin:0;color:#c7cad3;font-size:15px;line-height:1.55}
    .actions{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:10px;margin-top:24px}button,a{min-height:40px;border-radius:10px;padding:9px 14px;font:600 14px/1 inherit;cursor:pointer}
    button{border:1px solid #454b59;background:#292d36;color:#f5f6fa}.close{min-width:40px;padding:7px;font-size:20px;line-height:1}
    a{display:inline-flex;align-items:center;text-decoration:none;border:1px solid #8994ff;background:#8994ff;color:#111522}
    button:focus-visible,a:focus-visible{outline:3px solid #cbd0ff;outline-offset:2px}
    @media (prefers-color-scheme:light){:host{background:rgba(20,23,32,.45)}.card{border-color:#d7dae4;background:#fff;color:#171923}.brand{color:#4c57bd}p{color:#555967}button{border-color:#d4d7e0;background:#f4f5f8;color:#171923}a{color:#111522}}
    @media (forced-colors:active){:host{background:Canvas}.card,button,a{border:1px solid CanvasText;background:Canvas;color:CanvasText}}
    @media (prefers-reduced-motion:no-preference){.card{animation:enter .14s ease-out}@keyframes enter{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}}
    @media (max-width:480px){:host{align-items:end;padding:12px}.card{padding:20px;border-radius:16px}.actions{flex-direction:column-reverse}.actions>*{justify-content:center;width:100%}}
  </style><section class="card" role="dialog" aria-modal="true" aria-labelledby="oryn-alert-title" aria-describedby="oryn-alert-copy">
    <div class="top"><span class="brand">Oryn Browser</span><button class="close" type="button" aria-label="Close">&#215;</button></div>
    <h2 id="oryn-alert-title">This feature needs Oryn Browser</h2>
    <p id="oryn-alert-copy">Oryn runs compatible AI models locally and keeps model choice, downloads, and hardware use under your control.</p>
    <div class="actions"><button class="dismiss" type="button">Not now</button><a href="${ORYN_URL}" target="_blank" rel="noopener noreferrer">Learn about Oryn</a></div>
  </section>`;
  const dismissButton = shadow.querySelector<HTMLButtonElement>(".dismiss");
  const closeButton = shadow.querySelector<HTMLButtonElement>(".close");
  const link = shadow.querySelector<HTMLAnchorElement>("a");
  if (!dismissButton || !closeButton || !link) return;
  const controls: HTMLElement[] = [closeButton, dismissButton, link];
  const dismiss = () => {
    owner.removeEventListener("keydown", onKeyDown, true);
    host.remove();
    previousFocus?.focus({ preventScroll: true });
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      dismiss();
      return;
    }
    if (event.key !== "Tab") return;
    const activeIndex = controls.indexOf(shadow.activeElement as HTMLElement);
    const direction = event.shiftKey ? -1 : 1;
    const next = (activeIndex + direction + controls.length) % controls.length;
    event.preventDefault();
    controls[next]?.focus();
  };
  closeButton.addEventListener("click", dismiss);
  dismissButton.addEventListener("click", dismiss);
  owner.addEventListener("keydown", onKeyDown, true);
  owner.body.append(host);
  dismissButton.focus({ preventScroll: true });
}

export async function invokeAndWait<
  TInput = unknown,
  TResult extends InferenceResult = InferenceResult,
>(
  client: LocalAIClient,
  request: ModelInvocationRequest<TInput>,
  options: {
    signal?: AbortSignal;
    onEvent?: (event: LocalAIEvent) => void;
  } = {},
): Promise<TResult> {
  if (options.signal?.aborted) throw abortedJobError();
  let jobId: string | undefined;
  const buffered: LocalAIEvent[] = [];
  let settle: ((event: LocalAIEvent) => void) | undefined;
  const terminal = new Promise<LocalAIEvent>((resolve) => {
    settle = resolve;
  });
  const accept = (event: LocalAIEvent) => {
    if (event.sessionId !== request.sessionId) return;
    if (!jobId) {
      buffered.push(event);
      return;
    }
    if (event.jobId !== jobId) return;
    options.onEvent?.(event);
    if (isTerminalEvent(event)) settle?.(event);
  };
  const unsubscribe = client.subscribe(accept);
  let abort: () => void = () => undefined;
  try {
    const job = await client.invoke(request);
    jobId = job.jobId;
    for (const event of buffered) accept(event);
    abort = () => void client.cancel(job.jobId);
    options.signal?.addEventListener("abort", abort, { once: true });
    const event = await terminal;
    if (event.event === "job.completed") return event.data.result as TResult;
    if (event.event === "job.failed") throw new LocalAIError(event.data.error);
    throw abortedJobError();
  } finally {
    options.signal?.removeEventListener("abort", abort);
    unsubscribe();
  }
}

export async function* invokeStream<TInput = unknown>(
  client: LocalAIClient,
  request: ModelInvocationRequest<TInput>,
  options: { signal?: AbortSignal } = {},
): AsyncIterable<LocalAIEvent> {
  if (options.signal?.aborted) throw abortedJobError();
  const pending: LocalAIEvent[] = [];
  let wake: (() => void) | undefined;
  let jobId: string | undefined;
  const listener = (event: LocalAIEvent) => {
    if (event.sessionId !== request.sessionId) return;
    if (jobId && event.jobId !== jobId) return;
    pending.push(event);
    wake?.();
    wake = undefined;
  };
  const unsubscribe = client.subscribe(listener);
  let abort: () => void = () => undefined;
  try {
    const job = await client.invoke(request);
    jobId = job.jobId;
    for (let index = pending.length - 1; index >= 0; index -= 1) {
      if (pending[index]?.jobId !== jobId) pending.splice(index, 1);
    }
    abort = () => void client.cancel(job.jobId);
    options.signal?.addEventListener("abort", abort, { once: true });
    while (true) {
      if (pending.length === 0) {
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
      }
      const event = pending.shift();
      if (!event) continue;
      yield event;
      if (isTerminalEvent(event)) return;
    }
  } finally {
    options.signal?.removeEventListener("abort", abort);
    unsubscribe();
  }
}

function unwrap<T>(result: BridgeResult<T>): T {
  if (result.ok) return result.value;
  throw new LocalAIError(result.error);
}

function unavailableError(): LocalAIError {
  return new LocalAIError({
    code: "RUNTIME_NOT_AVAILABLE",
    message: "Local AI is not available in this browser.",
  });
}

function abortedError(): LocalAIError {
  return new LocalAIError({
    code: "REQUEST_ABORTED",
    message: "The model request was aborted.",
  });
}

function abortedJobError(): LocalAIError {
  return new LocalAIError({
    code: "REQUEST_ABORTED",
    message: "The Local AI job was cancelled.",
  });
}

function isTerminalEvent(event: LocalAIEvent): boolean {
  return ["job.completed", "job.failed", "job.cancelled"].includes(event.event);
}

let operationSequence = 0;
const ARTIFACT_CHUNK_BYTES = 256 * 1024;
function createOperationId(): string {
  operationSequence += 1;
  return `model_request_${Date.now().toString(36)}_${operationSequence.toString(36)}`;
}
