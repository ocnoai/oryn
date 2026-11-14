import {
  createLocalAIClient,
  invokeStream,
  LocalAIError,
  type ModelSession,
} from "@ocno/oryn-sdk";

import "./style.css";

const localAI = createLocalAIClient({ showOrynBrowserAlert: true });
const form = required<HTMLFormElement>("#chat-form");
const prompt = required<HTMLTextAreaElement>("#prompt");
const maxTokens = required<HTMLSelectElement>("#max-tokens");
const submit = required<HTMLButtonElement>("#send");
const status = required<HTMLElement>("#status");
const conversations = required<HTMLElement>("#conversations");
const active = new Map<string, AbortController>();
let sessionPromise: Promise<ModelSession> | undefined;

const qwen35 = {
  provider: "huggingface" as const,
  repository: "Qwen/Qwen3.5-0.8B",
  revision: "2fc06364715b967f1860aea9cf38778875588b17",
};

const capabilities = await localAI.getCapabilities();
status.textContent = capabilities.runtimeAvailable
  ? "Ready. Messages are generated locally; submit more than one to see the global queue."
  : "Oryn Local AI is unavailable. Open this page in Oryn Browser.";
submit.disabled = !capabilities.runtimeAvailable;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = prompt.value.trim();
  if (!content) return;
  prompt.value = "";
  void runChat(content);
});

window.addEventListener("pagehide", () => {
  for (const controller of active.values()) controller.abort();
  void sessionPromise?.then((session) =>
    localAI.models.releaseSession(session.sessionId),
  );
});

async function getSession(): Promise<ModelSession> {
  sessionPromise ??= localAI.models.request(
    {
      role: "chat",
      preferredModel: qwen35,
      certifiedModels: [
        qwen35,
        { provider: "huggingface", repository: "google/gemma-3-1b-it" },
      ],
      requirements: {
        task: "text-generation",
        ...(capabilities.features.taskProfiles ? {
          contract: { id: "chat-v1", revision: 1 },
          requiredFeatures: ["streaming"],
        } : {}),
        inputModalities: ["text"],
        outputModalities: ["text"],
        capabilities: { chat: true, streaming: true },
      },
      substitution: { automatic: "compatible" },
    },
    {
      onProgress: (progress) => {
        status.textContent = progressLabel(progress.stage);
      },
    },
  );
  try {
    return await sessionPromise;
  } catch (error) {
    sessionPromise = undefined;
    throw error;
  }
}

async function runChat(content: string): Promise<void> {
  const localId = crypto.randomUUID();
  const controller = new AbortController();
  active.set(localId, controller);
  const card = document.createElement("article");
  card.className = "conversation";
  card.dataset.chatJob = localId;
  card.dataset.deltaCount = "0";
  card.innerHTML = `
    <p class="user"></p>
    <p class="assistant" aria-live="polite"></p>
    <p class="job-status">Waiting for model…</p>
    <button type="button">Cancel</button>
  `;
  requiredWithin<HTMLElement>(card, ".user").textContent = content;
  const answer = requiredWithin<HTMLElement>(card, ".assistant");
  const jobStatus = requiredWithin<HTMLElement>(card, ".job-status");
  requiredWithin<HTMLButtonElement>(card, "button").addEventListener(
    "click",
    () => controller.abort(),
  );
  conversations.prepend(card);

  try {
    const session = await getSession();
    card.dataset.model = session.resolution.resolvedModel.repository;
    status.textContent = `Warm session: ${session.resolution.resolvedModel.repository}`;
    for await (const event of invokeStream(
      localAI,
      {
        sessionId: session.sessionId,
        input: {
          messages: [{ role: "user", content }],
          maxTokens: Number(maxTokens.value),
          temperature: 0.7,
          topP: 0.9,
        },
      },
      { signal: controller.signal },
    )) {
      if (event.event === "job.output.delta" && event.data.channel === "text") {
        answer.textContent += String(event.data.delta);
        card.dataset.deltaCount = String(
          Number(card.dataset.deltaCount ?? "0") + 1,
        );
        jobStatus.textContent = "Streaming locally…";
      } else if (event.event === "job.queue.updated") {
        jobStatus.textContent = `Queued — ${event.data.jobsAhead} job(s) ahead.`;
      } else if (event.event === "job.completed") {
        jobStatus.textContent = "Chat complete.";
      } else if (event.event === "job.failed") {
        throw new Error(event.data.error.message);
      } else if (event.event === "job.cancelled") {
        jobStatus.textContent = "Chat cancelled.";
      }
    }
  } catch (error) {
    jobStatus.textContent =
      error instanceof LocalAIError && error.code === "REQUEST_ABORTED"
        ? "Chat cancelled."
        : `Chat failed: ${error instanceof Error ? error.message : "Unknown error"}`;
  } finally {
    active.delete(localId);
    requiredWithin<HTMLButtonElement>(card, "button").hidden = true;
  }
}

function progressLabel(stage: string): string {
  const labels: Record<string, string> = {
    resolving: "Resolving a local chat model…",
    "awaiting-permission": "Waiting for permission in Oryn…",
    "awaiting-model-choice": "Choose a compatible model in Oryn…",
    downloading: "Downloading the selected chat model…",
    loading: "Loading the chat model…",
    ready: "Chat model ready.",
  };
  return labels[stage] ?? `Preparing chat — ${stage}…`;
}

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}

function requiredWithin<T extends Element>(root: Element, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}
