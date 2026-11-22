import {
  createLocalAIClient,
  invokeAndWait,
  LocalAIError,
} from "@ocno/oryn-sdk";
import "./style.css";
import {
  cuesForWindow,
  formatTimestamp,
  planAudioWindows,
  serializeSubRip,
  parseSubRip,
  parseCaptionTiming,
  serializeCaptionTiming,
  type Cue,
  type TranscriptValue,
} from "./subtitles";
import { withWordBounds, recoverWordTimings, mergeWordTimings, recoveredSegments, type PartialWordTimings } from "./caption-repair";
import { createCaptionStudio } from "./caption-studio";
import { createEndingStudio } from "./ending-studio";
import type { CaptionStyle } from "./video-renderer";
import { WHISPER_TIMED_ASR, transcriptionProfile } from "./profiles";
import { drawCaptions, renderCaptionedVideo, retainTiming, hasWordTiming, requireWordTiming, requiresWordTiming } from "./video-renderer";
import { createVideoArchive } from "./video-archive";

interface TranscriptResult {
  kind: "structured";
  value: TranscriptValue;
}
interface AlignmentResult {
  kind: "structured";
  value: { segments: NonNullable<TranscriptValue["segments"]> };
}
interface ExtractedPcm {
  samples: Float32Array;
  sampleRate: number;
  durationMs: number;
}

const SAMPLE_RATE = 16_000;
const localAI = createLocalAIClient({ showOrynBrowserAlert: true });
const form = required<HTMLFormElement>("#workflow");
const fileInput = required<HTMLInputElement>("#video-file");
const preview = required<HTMLVideoElement>("#preview");
const language = required<HTMLSelectElement>("#language");
const quality = required<HTMLSelectElement>("#quality");
const start = required<HTMLButtonElement>("#start");
const cancel = required<HTMLButtonElement>("#cancel");
const status = required<HTMLElement>("#status");
const stages = required<HTMLOListElement>("#stages");
const disclosure = required<HTMLElement>("#disclosure");
const editor = required<HTMLElement>("#editor");
const cuesRoot = required<HTMLElement>("#cues");
const summary = required<HTMLElement>("#summary");
const srtPreview = required<HTMLElement>("#srt");
let controller: AbortController | undefined;
let generated: Cue[] = [];
let objectURL: string | undefined;
let captionSource: File | undefined;
let renderController: AbortController | undefined;
let repairController: AbortController | undefined;
const repairDrafts = new Map<number, Cue>();
let renderedBlob: Blob | undefined;
let renderedURL: string | undefined;
interface VideoItem {
  file: File;
  generated: Cue[];
  cues: Cue[];
  summary: string;
  reviewed: boolean;
  rendered?: Blob;
  error?: string;
}
const videos: VideoItem[] = [];
let activeVideo: VideoItem | undefined;
let restoringVideo = false;
let batchOperation: "transcribe" | "render" | "download" | undefined;
let stopQueue = false;
let archiveController: AbortController | undefined;
const batch = required<HTMLElement>("#batch");
const queue = required<HTMLOListElement>("#video-queue");
const batchStatus = required<HTMLElement>("#batch-status");
const renderAll = required<HTMLButtonElement>("#render-all");
const downloadAll = required<HTMLButtonElement>("#download-all");
const addVideos = required<HTMLInputElement>("#add-videos");
const reviewVideo = required<HTMLButtonElement>("#review-video");
const cancelBatch = required<HTMLButtonElement>("#cancel-batch");
const renderButton = required<HTMLButtonElement>("#render-video");
const renderCancel = required<HTMLButtonElement>("#cancel-render");
const saveVideo = required<HTMLButtonElement>("#save-video");
const renderStatus = required<HTMLElement>("#render-status");
const renderCanvas = required<HTMLCanvasElement>("#render-canvas");
const renderedPreview = required<HTMLVideoElement>("#rendered-preview");
const renderProgress = required<HTMLProgressElement>("#render-progress");
const subtitleInput = required<HTMLInputElement>("#subtitle-file");
subtitleInput.addEventListener("change", () => void importSubtitles());
const captionStyle = required<HTMLSelectElement>("#caption-style");
captionStyle.addEventListener("change", () => { clearAllRendered(); updateTimingStatus(); studio.refresh(); });
const studio = createCaptionStudio(readCues, () => { clearAllRendered(); updateTimingStatus(); }, () => { preview.pause(); endingStudio.pause(); });
const endingStudio = createEndingStudio(() => activeVideo?.file, () => { clearAllRendered(); updateTimingStatus(); },
  () => { preview.pause(); studio.pause(); renderedPreview.pause(); },
  (context, time, width, height) => drawCaptions(context, readCues(), time, width, height, captionStyle.value as CaptionStyle, studio.getDesign()));
preview.addEventListener("play", () => { studio.pause(); endingStudio.pause(); });
renderedPreview.addEventListener("play", () => { preview.pause(); studio.pause(); endingStudio.pause(); });
required<HTMLButtonElement>("#save-timing").addEventListener("click", () => void saveTiming());

async function importSubtitles(): Promise<void> {
  const file = subtitleInput.files?.[0];
  const video = activeVideo?.file;
  try {
    if (!video) throw new Error("Choose the source video first.");
    if (!file) return;
    if (file.size > 4 * 1024 * 1024) throw new Error("The subtitle file exceeds the 4 MB import limit.");
    const isTimingFile = file.name.toLowerCase().endsWith(".json");
    let imported = isTimingFile ? parseCaptionTiming(await file.text()) : parseSubRip(await file.text());
    if (activeVideo?.file !== video || busy()) return;
    if (!isTimingFile && captionSource === video) {
      const originals = new Map(generated.map(cue => [`${cue.startMs}:${cue.endMs}`, cue]));
      imported = imported.map(cue => retainTiming(cue, originals.get(`${cue.startMs}:${cue.endMs}`)));
    }
    repairDrafts.clear();
    generated = imported;
    captionSource = video;
    renderCues(generated);
    const timedCount = generated.filter(hasWordTiming).length;
    summary.textContent = `${generated.length} imported captions · ${isTimingFile ? "Saved timing spans restored." : timedCount ? `Existing word timings reused for ${timedCount} captions with matching boundaries.` : "SRT has no individual word timings. Choose Plain captions, or generate subtitles from the video for word highlighting."}`;
    editor.hidden = false;
    status.textContent = "Imported subtitles are ready to review and render.";
    if (activeVideo) activeVideo.error = undefined;
    snapshotVideo(); updateQueue();
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not import subtitles.";
  } finally {
    subtitleInput.value = "";
  }
}

function clearRendered(): void {
  if (!restoringVideo && activeVideo) { activeVideo.rendered = undefined; activeVideo.error = undefined; }
  endingStudio.pause();
  renderedBlob = undefined;
  if (renderedURL) URL.revokeObjectURL(renderedURL);
  renderedURL = undefined;
  renderedPreview.removeAttribute("src");
  renderedPreview.load();
  renderedPreview.hidden = true;
  saveVideo.disabled = true;
}
function clearAllRendered(): void {
  videos.forEach(item => { item.rendered = undefined; });
  clearRendered(); updateQueue();
}
renderButton.addEventListener("click", () => void renderVideo());
renderCancel.addEventListener("click", () => { stopQueue = true; renderController?.abort(); });
saveVideo.addEventListener("click", () => void saveRenderedVideo());

const capabilities = await localAI.getCapabilities();
start.disabled = true;
status.textContent = capabilities.runtimeAvailable
  ? "Ready. Choose a video."
  : "Open this page in Oryn Browser to run the local pipeline.";

fileInput.addEventListener("change", () => addFiles(fileInput.files, true));
addVideos.addEventListener("change", () => { addFiles(addVideos.files); addVideos.value = ""; });
renderAll.addEventListener("click", () => void renderQueue());
downloadAll.addEventListener("click", () => void downloadQueue());
cancelBatch.addEventListener("click", () => {
  stopQueue = true; controller?.abort(); renderController?.abort(); archiveController?.abort();
});
reviewVideo.addEventListener("click", () => {
  if (!activeVideo || busy()) return;
  try { serializeSubRip(readCues()); }
  catch (error) { batchStatus.textContent = `Correct this video's captions first: ${(error as Error).message}`; return; }
  activeVideo.reviewed = true;
  snapshotVideo();
  const index = videos.indexOf(activeVideo);
  const next = [...videos.slice(index + 1), ...videos.slice(0, index)].find(item => item.cues.length && !item.reviewed);
  if (next) selectVideo(next);
  updateQueue();
});

function busy(): boolean { return Boolean(batchOperation || controller || renderController || repairController); }
function snapshotVideo(): void {
  if (!activeVideo || restoringVideo) return;
  activeVideo.generated = generated;
  activeVideo.cues = readCues();
  activeVideo.summary = summary.textContent ?? "";
  activeVideo.rendered = renderedBlob;
}
function addFiles(files: FileList | null, replace = false): void {
  if (!files?.length || busy()) return;
  snapshotVideo();
  const added = [...files].map(file => ({file, generated: [], cues: [], summary: "", reviewed: false} satisfies VideoItem));
  // A single selection keeps the familiar replace-video flow. Multiple files start a queue.
  if (videos.length <= 1 && replace) videos.splice(0, videos.length, ...added);
  else videos.push(...added);
  selectVideo(added[0]!);
  updateQueue();
}
function selectVideo(item?: VideoItem): void {
  snapshotVideo();
  restoringVideo = true;
  activeVideo = item;
  const file = item?.file;
  captionSource = undefined;
  preview.pause(); renderedPreview.pause();
  endingStudio.pause();
  repairDrafts.clear();
  generated = [];
  editor.hidden = true;
  clearRendered();
  studio.setSource();
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = undefined;
  preview.removeAttribute("src");
  preview.load();
  preview.hidden = true;
  cuesRoot.textContent = ""; summary.textContent = ""; stages.textContent = "";
  if (file && item) {
    objectURL = URL.createObjectURL(file);
    preview.src = objectURL;
    studio.setSource(objectURL);
    preview.hidden = false;
    disclosure.textContent = `${file.name} · ${formatBytes(file.size)} · previewed from a temporary local URL`;
    generated = item.generated;
    // Use the saved edits as the timing baseline while rebuilding the editor.
    const original = generated;
    generated = item.cues;
    captionSource = item.cues.length ? file : undefined;
    renderCues(item.cues);
    generated = original;
    repairDrafts.clear();
    item.cues.forEach((cue, index) => repairDrafts.set(index, {...cue, spans: cue.spans ?? cue.wordTimingHints}));
    summary.textContent = item.summary;
    editor.hidden = !item.cues.length;
    if (videos.length > 1 && item.cues.length && !item.reviewed) required<HTMLDetailsElement>(".caption-transcript").open = true;
    renderedBlob = item.rendered;
    if (renderedBlob) {
      renderedURL = URL.createObjectURL(renderedBlob);
      renderedPreview.src = renderedURL; renderedPreview.hidden = false; saveVideo.disabled = false;
      renderStatus.textContent = "Rendered video ready to preview or save.";
    }
    status.textContent = item.error ?? (item.cues.length ? "Subtitles are ready for review." : "Ready to create subtitles, or import an SRT or timing JSON for this video.");
  } else { disclosure.textContent = ""; status.textContent = "Ready. Choose a video."; }
  restoringVideo = false;
  updateQueue();
}
cancel.addEventListener("click", () => { stopQueue = true; controller?.abort(); });
form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy()) return;
  if (videos.length > 1) void transcribeQueue();
  else if (activeVideo) void run(activeVideo.file);
});
required<HTMLButtonElement>("#reset").addEventListener("click", () =>
  renderCues(generated),
);
required<HTMLButtonElement>("#save").addEventListener(
  "click",
  () => void save(),
);

function updateQueue(): void {
  const multiple = videos.length > 1;
  const locked = busy();
  batch.hidden = !multiple;
  required<HTMLElement>("#add-video-control").hidden = !videos.length;
  required<HTMLElement>("#review-bar").hidden = !multiple || editor.hidden;
  fileInput.disabled = locked; addVideos.disabled = locked;
  subtitleInput.disabled = locked;
  language.disabled = locked; quality.disabled = locked;
  const pending = videos.filter(item => !item.cues.length).length;
  start.textContent = multiple ? `Create subtitles for ${pending} video${pending === 1 ? "" : "s"}` : "Create subtitles locally";
  start.disabled = locked || !activeVideo || !capabilities.runtimeAvailable || (multiple && !pending);
  const reviewed = videos.filter(item => item.reviewed).length;
  const rendered = videos.filter(item => item.rendered).length;
  renderAll.disabled = locked || !multiple || reviewed !== videos.length || rendered === videos.length;
  renderAll.textContent = rendered && rendered < videos.length ? `Render remaining ${videos.length - rendered} videos` : "Render all videos";
  downloadAll.disabled = locked || !multiple || rendered !== videos.length;
  reviewVideo.disabled = locked || !activeVideo?.cues.length || Boolean(activeVideo.reviewed);
  reviewVideo.textContent = activeVideo?.reviewed ? "Reviewed" : "Mark reviewed & next";
  required<HTMLElement>("#review-label").textContent = activeVideo ? `${videos.indexOf(activeVideo) + 1} of ${videos.length} · ${activeVideo.file.name}${activeVideo.reviewed ? " · Captions reviewed" : " · Check the text and timing below"}` : "";
  cancelBatch.hidden = !batchOperation;
  cancelBatch.disabled = false;
  if (!locked) batchStatus.textContent = `${reviewed} of ${videos.length} reviewed · ${rendered} of ${videos.length} rendered. ${rendered === videos.length ? "Download all videos together as a ZIP." : reviewed === videos.length ? "Ready to render the batch." : "Open each video and mark its captions reviewed before rendering all."}`;
  queue.replaceChildren(...videos.map((item, index) => {
    const row = document.createElement("li");
    row.className = "video-queue-item";
    const select = document.createElement("button"); select.type = "button"; select.className = "queue-select secondary";
    select.disabled = locked;
    select.setAttribute("aria-current", String(item === activeVideo));
    const title = document.createElement("strong"); title.textContent = `${index + 1}. ${item.file.name}`;
    const detail = document.createElement("span");
    detail.textContent = item === activeVideo && controller ? "Creating subtitles…" : item === activeVideo && renderController ? "Rendering…"
      : item.error ? item.error : item.rendered ? "Rendered · ready to download" : item.reviewed ? "Reviewed · ready to render" : item.cues.length ? "Needs review" : "Waiting for subtitles";
    detail.className = item.error ? "queue-error" : item.rendered || item.reviewed ? "queue-ready" : "";
    select.append(title, detail); select.addEventListener("click", () => selectVideo(item));
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "secondary queue-remove";
    remove.textContent = "Remove"; remove.disabled = locked; remove.setAttribute("aria-label", `Remove ${item.file.name}`);
    remove.addEventListener("click", () => {
      if (busy()) return;
      videos.splice(videos.indexOf(item), 1);
      if (item === activeVideo) selectVideo(videos[Math.min(index, videos.length - 1)]);
      updateQueue();
    });
    row.append(select, remove); return row;
  }));
}

async function transcribeQueue(): Promise<void> {
  if (busy()) return;
  snapshotVideo();
  batchOperation = "transcribe"; stopQueue = false;
  const pending = videos.filter(item => !item.cues.length);
  try {
    for (const [index, item] of pending.entries()) {
      if (stopQueue) break;
      selectVideo(item);
      batchStatus.textContent = `Creating subtitles · ${index + 1} of ${pending.length} · ${item.file.name}`;
      await run(item.file);
    }
  } finally {
    batchOperation = undefined;
    selectVideo(videos.find(item => item.cues.length && !item.reviewed) ?? activeVideo);
    updateQueue();
    if (stopQueue) batchStatus.textContent = "Queue stopped. Completed transcripts are kept. Create subtitles again to continue the remaining videos.";
  }
}

async function renderQueue(): Promise<void> {
  if (busy()) return;
  snapshotVideo();
  if (!videos.length || videos.some(item => !item.reviewed || !item.cues.length)) return;
  try {
    endingStudio.getEnding();
    for (const item of videos) {
      try {
        serializeSubRip(item.cues);
        if (requiresWordTiming(captionStyle.value)) requireWordTiming(item.cues);
      } catch (error) { selectVideo(item); throw new Error(`${item.file.name}: ${(error as Error).message}`); }
    }
  } catch (error) { batchStatus.textContent = (error as Error).message; return; }
  batchOperation = "render"; stopQueue = false;
  const pending = videos.filter(item => !item.rendered);
  try {
    for (const [index, item] of pending.entries()) {
      if (stopQueue) break;
      selectVideo(item);
      batchStatus.textContent = `Rendering · ${index + 1} of ${pending.length} · ${item.file.name}`;
      await renderVideo();
    }
  } finally {
    batchOperation = undefined; updateQueue();
    if (stopQueue) batchStatus.textContent = "Queue stopped. Finished videos are kept. Render remaining videos to continue, or save finished videos individually.";
    else if (videos.some(item => item.error)) batchStatus.textContent = "Some videos could not render. Finished videos are kept. Check the errors above, then retry the remaining videos.";
  }
}

async function downloadQueue(): Promise<void> {
  if (busy() || videos.some(item => !item.rendered)) return;
  const controls = [...document.querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement | HTMLTextAreaElement>("input, button, select, textarea")];
  const disabled = controls.map(control => control.disabled);
  controls.forEach(control => { control.disabled = true; });
  studio.setBusy(true); preview.pause(); renderedPreview.pause(); endingStudio.pause();
  batchOperation = "download"; archiveController = new AbortController(); updateQueue();
  let id: string | undefined;
  let message = "";
  try {
    const archive = await createVideoArchive(videos.map(item => ({name: videoFilename(item.file, item.rendered!), blob: item.rendered!})), archiveController.signal,
      count => { batchStatus.textContent = `Preparing download · ${count} of ${videos.length} videos`; });
    archiveController.signal.throwIfAborted();
    const name = "captioned-videos.zip";
    batchStatus.textContent = "Preparing ZIP for the save dialog…";
    const artifact = await localAI.artifacts.create(archive, {lifetime: "session", name});
    id = artifact.id;
    archiveController.signal.throwIfAborted();
    await localAI.artifacts.saveAs(id, {suggestedFilename: name});
    message = "ZIP save dialog closed. All rendered videos remain available to preview or download again.";
  } catch (error) {
    message = archiveController.signal.aborted ? "Download cancelled. Your rendered videos are kept." : `Could not download ZIP: ${(error as Error).message}. Your videos are kept; retry or save them individually.`;
  } finally {
    if (id) await localAI.artifacts.release(id).catch(() => undefined);
    controls.forEach((control, index) => { control.disabled = disabled[index]!; });
    studio.setBusy(false);
    archiveController = undefined; batchOperation = undefined; updateQueue(); batchStatus.textContent = message;
  }
}
function videoFilename(file: File, blob: Blob): string {
  return `${baseName(file.name)}-captioned.${blob.type.startsWith("video/mp4") ? "mp4" : "webm"}`;
}

async function run(file: File): Promise<void> {
  controller?.abort();
  controller = new AbortController();
  start.disabled = true;
  cancel.hidden = false;
  editor.hidden = true;
  studio.setBusy(true);
  endingStudio.pause();
  fileInput.disabled = true;
  subtitleInput.disabled = true;
  clearRendered();
  if (activeVideo) activeVideo.reviewed = false;
  updateQueue();
  stages.textContent = "";
  let sessionId: string | undefined;
  let alignmentSessionId: string | undefined;
  let nativeTimingFallbackSessionId: string | undefined;
  let inputId: string | undefined;
  try {
    stage("Video ingestion", "running");
    const videoArtifact = await localAI.artifacts.create(file, {
      lifetime: "session",
      name: file.name,
    });
    inputId = videoArtifact.id;
    stage("Video ingestion", "done");
    stage("Audio extraction", "running");
    const extracted = await extractPcm(file, controller.signal);
    stage(
      "Audio extraction",
      "done",
      `${formatTimestamp(extracted.durationMs)} mono PCM`,
    );
    stage(
      "Vocal isolation",
      "warning",
      "No conformant isolation adapter is installed; using the complete audio mix as an explicit fallback.",
    );
    stage("Model resolution", "running");
    const profile = transcriptionProfile(
      quality.value === "quality" ? "quality" : "faster",
    );
    const session = await localAI.models.request(
      {
        role: profile.role,
        preferredModel: profile.preferredModel,
        certifiedModels: profile.certifiedModels,
        requirements: {
          task: "automatic-speech-recognition",
          ...(capabilities.features.taskProfiles ? { contract: { id: "transcription-v1", revision: 1 }, requiredFeatures: ["speechRecognition"] } : {}),
          inputModalities: ["audio"],
          outputModalities: ["text"],
          capabilities: { speechRecognition: true },
        },
        substitution: { automatic: "certified-only" },
      },
      {
        signal: controller.signal,
        onProgress: (progress) =>
          (status.textContent =
            progress.stage === "downloading"
              ? `Downloading model · ${formatBytes(progress.downloadedBytes)}${progress.totalBytes ? ` / ${formatBytes(progress.totalBytes)}` : ""}`
              : `Model · ${progress.stage}`),
      },
    );
    sessionId = session.sessionId;
    stage(
      "Model resolution",
      "done",
      session.resolution.resolvedModel.repository,
    );
    const windows = planAudioWindows(
      extracted.samples.length,
      extracted.sampleRate,
      profile.windowSeconds,
    );
    const collected: Cue[] = [];
    let detectedLanguage: string | null = null;
    let estimatedWindows = 0;
    let alignedWindows = 0;
    let nativeTimingFallbackWindows = 0;
    let nativeTimingFallbackRepository: string | undefined;
    const unsupportedAlignmentLanguages = new Set<string>();
    stage("Transcription", "running", `0 / ${windows.length} windows`);
    for (const window of windows) {
      if (controller.signal.aborted)
        throw new DOMException("Aborted", "AbortError");
      const windowLabel = `${formatTimestamp(window.startMs)}–${formatTimestamp(window.endMs)}`;
      stage(
        "Transcription",
        "running",
        `${window.index + 1} / ${windows.length} · ${windowLabel}`,
      );
      status.textContent = `Transcribing window ${window.index + 1} of ${windows.length} · ${windowLabel}`;
      const wave = pcmWave(
        extracted.samples.subarray(window.startSample, window.endSample),
        extracted.sampleRate,
      );
      const audio = await localAI.artifacts.create(wave, {
        lifetime: "session",
        name: `${baseName(file.name)}.${String(window.index + 1).padStart(4, "0")}.wav`,
      });
      let activeWindowStage = "Transcription";
      try {
        const result = await invokeAndWait<
          { audioArtifactId: string; language: string; timestamps: boolean },
          TranscriptResult
        >(
          localAI,
          {
            sessionId,
            input: {
              audioArtifactId: audio.id,
              language: language.value,
              timestamps: session.taskProfile?.features.wordTimestamps === "unsupported" ? false : true,
            },
          },
          {
            signal: controller.signal,
            onEvent: (event) => {
              if (event.event === "job.progress") {
                status.textContent = `Window ${window.index + 1} of ${windows.length} · ${event.data.stage ?? "transcribing locally"}`;
              }
            },
          },
        );
        detectedLanguage ??= result.value.language;
        let timed = cuesForWindow(result.value, window);
        if (timed.timing === "window-estimated") {
          activeWindowStage = "Timestamp alignment";
          stage("Timestamp alignment", "running", windowLabel);
          const alignmentLanguage =
            result.value.language ?? language.value ?? "auto";
          if (!unsupportedAlignmentLanguages.has(alignmentLanguage)) {
            try {
              if (!alignmentSessionId) {
                const alignment = await localAI.models.request(
                  {
                    role: "video-subtitle-forced-alignment",
                    preferredModel: profile.alignmentModel,
                    certifiedModels: [],
                    requirements: {
                      task: "forced-alignment",
                      ...(capabilities.features.taskProfiles ? {
                        contract: { id: "forced-alignment-v1", revision: 1 },
                        requiredFeatures: ["forcedAlignment", "wordTimestamps"],
                      } : {}),
                      inputModalities: ["audio", "text"],
                      outputModalities: ["structured"],
                    },
                    substitution: { automatic: "none" },
                  },
                  { signal: controller.signal },
                );
                alignmentSessionId = alignment.sessionId;
                stage(
                  "Timestamp alignment",
                  "running",
                  alignment.resolution.resolvedModel.repository,
                );
              }
              const aligned = await invokeAndWait<
                {
                  audioArtifactId: string;
                  transcript: string;
                  language: string;
                  timestamps: "word";
                },
                AlignmentResult
              >(
                localAI,
                {
                  sessionId: alignmentSessionId,
                  input: {
                    audioArtifactId: audio.id,
                    transcript: result.value.text,
                    language: alignmentLanguage,
                    timestamps: "word",
                  },
                },
                { signal: controller.signal },
              );
              timed = cuesForWindow(
                {
                  ...result.value,
                  segments: aligned.value.segments,
                },
                window,
              );
              if (timed.timing === "native") alignedWindows += 1;
            } catch (error) {
              if (
                !(error instanceof LocalAIError) ||
                String(error.code) !== "ALIGNMENT_LANGUAGE_UNSUPPORTED"
              ) {
                throw error;
              }
              unsupportedAlignmentLanguages.add(alignmentLanguage);
              if (alignmentSessionId) {
                const unsupportedAlignmentSessionId = alignmentSessionId;
                alignmentSessionId = undefined;
                await localAI.models
                  .releaseSession(unsupportedAlignmentSessionId)
                  .catch(() => undefined);
              }
              stage(
                "Timestamp alignment",
                "warning",
                `${alignmentLanguage} is not supported by ${profile.alignmentModel.repository}`,
              );
            }
          }
          if (
            timed.timing === "window-estimated" &&
            unsupportedAlignmentLanguages.has(alignmentLanguage) &&
            profile.nativeTimingFallbackModel
          ) {
            activeWindowStage = "Timed-caption fallback";
            stage(
              "Timed-caption fallback",
              "running",
              `${window.index + 1} / ${windows.length} · ${windowLabel}`,
            );
            if (!nativeTimingFallbackSessionId) {
              const fallback = await localAI.models.request(
                {
                  role: "video-subtitle-native-timing-fallback",
                  preferredModel: profile.nativeTimingFallbackModel,
                  certifiedModels: [profile.nativeTimingFallbackModel],
                  requirements: {
                    task: "automatic-speech-recognition",
                    ...(capabilities.features.taskProfiles ? {
                      contract: { id: "transcription-v1", revision: 1 },
                      requiredFeatures: ["wordTimestamps"],
                      parameterValues: { timestamps: true },
                    } : {}),
                    inputModalities: ["audio"],
                    outputModalities: ["text"],
                    capabilities: { speechRecognition: true },
                  },
                  substitution: { automatic: "none" },
                },
                { signal: controller.signal },
              );
              nativeTimingFallbackSessionId = fallback.sessionId;
              nativeTimingFallbackRepository =
                fallback.resolution.resolvedModel.repository;
            }
            const fallbackResult = await invokeAndWait<
              {
                audioArtifactId: string;
                language: string;
                timestamps: boolean;
              },
              TranscriptResult
            >(
              localAI,
              {
                sessionId: nativeTimingFallbackSessionId,
                input: {
                  audioArtifactId: audio.id,
                  language: language.value,
                  timestamps: true,
                },
              },
              { signal: controller.signal },
            );
            const fallbackTimed = cuesForWindow(fallbackResult.value, window);
            if (fallbackTimed.timing === "native") {
              timed = fallbackTimed;
              detectedLanguage =
                fallbackResult.value.language ?? detectedLanguage;
              nativeTimingFallbackWindows += 1;
            }
          }
        }
        if (timed.timing === "window-estimated") estimatedWindows += 1;
        collected.push(...timed.cues);
      } catch (error) {
        if (controller.signal.aborted) throw error;
        const detail =
          error instanceof Error ? error.message : "Unknown Local AI error";
        throw new Error(
          `${activeWindowStage} failed for ${windowLabel}: ${detail}`,
          { cause: error },
        );
      } finally {
        await localAI.artifacts.release(audio.id).catch(() => undefined);
      }
    }
    stage(
      "Transcription",
      "done",
      `${windows.length} / ${windows.length} windows`,
    );
    if (!collected.length)
      throw new Error("No speech was detected in the selected video.");
    stage(
      "Cue timing",
      estimatedWindows ? "warning" : "done",
      estimatedWindows
        ? `${estimatedWindows} transcription window${estimatedWindows === 1 ? "" : "s"} had no native ASR timestamps; their cue times are estimated inside the displayed window and must be reviewed.`
        : "Model-native or declared alignment timestamps were preserved on the original video timeline.",
    );
    if (alignedWindows) {
      stage(
        "Timestamp alignment",
        "done",
        `${alignedWindows} / ${windows.length} windows aligned by ${profile.alignmentModel.repository}`,
      );
    } else if (unsupportedAlignmentLanguages.size) {
      stage(
        "Timestamp alignment",
        "warning",
        `${[...unsupportedAlignmentLanguages].join(", ")} is outside the aligner's declared language set`,
      );
    }
    if (nativeTimingFallbackWindows) {
      stage(
        "Timed-caption fallback",
        "done",
        `${nativeTimingFallbackWindows} / ${windows.length} windows transcribed with native timing by ${nativeTimingFallbackRepository}`,
      );
    }
    generated = collected;
    captionSource = file;
    stage("Cue creation", "done");
    renderCues(generated);
    summary.textContent = `${generated.length} timed cues · ${formatTimestamp(extracted.durationMs)} · ${detectedLanguage ?? "auto-detected language"} · preferred transcription: ${session.resolution.resolvedModel.repository}${nativeTimingFallbackRepository ? ` · final timed captions: ${nativeTimingFallbackRepository}` : ""}${estimatedWindows ? " · estimated window timing requires review" : ""}`;
    editor.hidden = false;
    status.textContent = "Subtitles are ready for review.";
  } catch (error) {
    const cancelled =
      controller.signal.aborted ||
      (error instanceof LocalAIError && error.code === "REQUEST_ABORTED") ||
      (error instanceof DOMException && error.name === "AbortError");
    status.textContent = cancelled
      ? "Pipeline cancelled. No final SRT was published."
      : `Could not create subtitles: ${error instanceof Error ? error.message : "Unknown error"}`;
    if (activeVideo) activeVideo.error = status.textContent;
  } finally {
    if (inputId)
      await localAI.artifacts.release(inputId).catch(() => undefined);
    if (sessionId)
      await localAI.models.releaseSession(sessionId).catch(() => undefined);
    if (alignmentSessionId)
      await localAI.models
        .releaseSession(alignmentSessionId)
        .catch(() => undefined);
    if (nativeTimingFallbackSessionId)
      await localAI.models
        .releaseSession(nativeTimingFallbackSessionId)
        .catch(() => undefined);
    controller = undefined;
    studio.setBusy(false);
    fileInput.disabled = false;
    subtitleInput.disabled = false;
    start.disabled = !capabilities.runtimeAvailable;
    cancel.hidden = true;
    editor.hidden = !generated.length;
    snapshotVideo(); updateQueue();
  }
}

async function extractPcm(
  file: File,
  signal: AbortSignal,
): Promise<ExtractedPcm> {
  if (file.size > 8 * 1024 * 1024 * 1024)
    throw new Error("The video exceeds the 8 GB demo limit.");
  const context = new AudioContext({ sampleRate: SAMPLE_RATE });
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");
    if (!Number.isFinite(decoded.duration) || decoded.duration <= 0)
      throw new Error("The video has no decodable audio track.");
    if (decoded.duration > 14_400)
      throw new Error("The video exceeds the four-hour duration limit.");
    const frames = Math.ceil(decoded.duration * SAMPLE_RATE);
    const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");
    const samples = rendered.getChannelData(0).slice();
    return {
      samples,
      sampleRate: SAMPLE_RATE,
      durationMs: Math.round((samples.length / SAMPLE_RATE) * 1_000),
    };
  } finally {
    await context.close();
  }
}

function pcmWave(samples: Float32Array, rate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) =>
    [...value].forEach((character, index) =>
      view.setUint8(offset + index, character.charCodeAt(0)),
    );
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) =>
    view.setInt16(
      44 + index * 2,
      Math.max(-1, Math.min(1, sample)) * 0x7fff,
      true,
    ),
  );
  return new Blob([buffer], { type: "audio/wav" });
}

function renderCues(values: Cue[]): void {
  cuesRoot.textContent = "";
  repairDrafts.clear();
  values.forEach((cue, index) => {
    const row = document.createElement("div");
    row.className = "cue";
    row.innerHTML = `<span>${index + 1}</span><input type="number" min="0" value="${cue.startMs}" aria-label="Cue ${index + 1} start milliseconds"><input type="number" min="1" value="${cue.endMs}" aria-label="Cue ${index + 1} end milliseconds"><textarea aria-label="Cue ${index + 1} text"></textarea><small class="cue-timing"></small>`;
    row.querySelector("textarea")!.value = cue.text;
    const repair = document.createElement("button");
    repair.type = "button";
    repair.className = "repair-caption secondary";
    repair.textContent = "Repair word timings";
    repair.addEventListener("click", () => openRepair(row, index));
    row.append(repair);
    row.addEventListener("input", event => {
      if ((event.target as Element).closest(".caption-repair")) return;
      row.querySelector(".caption-repair")?.remove();
      refreshPreview();
    });
    cuesRoot.append(row);
  });
  refreshPreview();
}
function openRepair(row: HTMLElement, index: number): void {
  if (row.querySelector(".caption-repair") || repairController) return;
  const cue = readCues()[index]!;
  const panel = document.createElement("div");
  panel.className = "caption-repair";
  const message = document.createElement("p");
  message.setAttribute("role", "status");
  message.textContent = "Regenerate timings for this caption using the model selected in Oryn. Your corrected text stays intact. Known word timings are already filled in below.";
  const languageLabel = document.createElement("label");
  languageLabel.textContent = "Caption language ";
  const repairLanguage = language.cloneNode(true) as HTMLSelectElement;
  repairLanguage.removeAttribute("id");
  repairLanguage.value = language.value;
  repairLanguage.setAttribute("aria-label", `Caption ${index + 1} repair language`);
  languageLabel.append(repairLanguage);
  const auto = document.createElement("button");
  auto.type = "button";
  auto.textContent = "Repair this caption locally";
  auto.disabled = !capabilities.runtimeAvailable;
  const stop = document.createElement("button");
  stop.type = "button";
  stop.textContent = "Cancel repair";
  stop.hidden = true;
  stop.addEventListener("click", () => repairController?.abort());
  auto.addEventListener("click", () => void repairCaption(row, index, cue, message, stop, repairLanguage.value, readBounds(), fillBounds));
  const manual = document.createElement("details");
  const title = document.createElement("summary");
  title.textContent = "Set word timings manually";
  manual.append(title);
  const hint = document.createElement("p");
  hint.textContent = `Times are milliseconds from the start of the video, within ${cue.startMs}–${cue.endMs} ms. Play this section, then enter each word’s start and end. Known timings are prefilled. Only words without a measured timing need attention.`;
  manual.append(hint);
  const play = document.createElement("button");
  play.type = "button";
  play.textContent = "Play this caption";
  let stopPlayback = () => {};
  play.addEventListener("click", () => {
    stopPlayback();
    studio.pause();
    preview.currentTime = cue.startMs / 1000;
    const ended = () => {
      if (preview.currentTime * 1000 >= cue.endMs || !panel.isConnected) { preview.pause(); stopPlayback(); }
    };
    stopPlayback = () => preview.removeEventListener("timeupdate", ended);
    preview.addEventListener("timeupdate", ended);
    void preview.play().catch(() => { stopPlayback(); message.textContent = "Use the video preview controls to listen to this caption."; });
  });
  manual.append(play);
  const headings = document.createElement("div");
  headings.className = "word-timing-row";
  for (const text of ["Word", "Start ms", "End ms"]) {
    const heading = document.createElement("small"); heading.textContent = text; headings.append(heading);
  }
  manual.append(headings);
  const inputs: Array<[HTMLInputElement, HTMLInputElement]> = [];
  const words = cue.text.trim().split(/\s+/u);
  const initialBounds = mergeWordTimings(cue,
    recoverWordTimings(cue, repairDrafts.get(index)?.spans ?? []),
    recoverWordTimings(cue, cue.wordTimingHints ?? generated[index]?.spans ?? []));
  const readBounds = (): PartialWordTimings => inputs.map(([start, end]) => {
    const startMs = Number(start.value), endMs = Number(end.value);
    return start.value !== "" && end.value !== "" && Number.isSafeInteger(startMs) && Number.isSafeInteger(endMs) && startMs >= cue.startMs && endMs > startMs && endMs <= cue.endMs ? {startMs, endMs} : undefined;
  });
  const rememberBounds = (bounds: PartialWordTimings) => repairDrafts.set(index, {...cue, spans: bounds.flatMap((span, n) => span ? [{...span, text: words[n]!}] : [])});
  const fillBounds = (bounds: PartialWordTimings) => {
    bounds.forEach((span, n) => {
      if (inputs[n]) { inputs[n]![0].value = span ? String(span.startMs) : ""; inputs[n]![1].value = span ? String(span.endMs) : ""; }
    });
    rememberBounds(bounds);
  };
  manual.addEventListener("input", () => rememberBounds(readBounds()));
  if (words.length > 256) { message.textContent = "Split this caption into fewer than 257 words before repairing it."; auto.disabled = true; }
  for (const [wordIndex, word] of words.slice(0, 256).entries()) {
    const item = document.createElement("div");
    item.className = "word-timing-row";
    const label = document.createElement("span");
    label.textContent = word;
    const fields = ["start", "end"].map(bound => {
      const field = document.createElement("input");
      field.type = "number"; field.step = "1"; field.min = String(cue.startMs); field.max = String(cue.endMs);
      field.placeholder = `${bound} ms`;
      field.setAttribute("aria-label", `Caption ${index + 1} word ${wordIndex + 1} ${bound} milliseconds`);
      return field;
    }) as [HTMLInputElement, HTMLInputElement];
    inputs.push(fields);
    item.append(label, ...fields);
    manual.append(item);
  }
  fillBounds(initialBounds);
  const apply = document.createElement("button");
  apply.type = "button"; apply.textContent = "Apply word timings";
  apply.addEventListener("click", () => {
    try {
      const repaired = withWordBounds(cue, inputs.map(([start, end]) => ({startMs: start.value === "" ? NaN : Number(start.value), endMs: end.value === "" ? NaN : Number(end.value)})));
      commitRepair(row, index, cue, repaired);
      stopPlayback(); preview.pause();
    } catch (error) { message.textContent = error instanceof Error ? error.message : "Invalid word timings."; }
  });
  manual.append(apply);
  panel.append(message, languageLabel, auto, stop, manual);
  row.append(panel);
}

function commitRepair(row: HTMLElement, index: number, original: Cue, repaired: Cue): void {
  const current = readCues()[index];
  if (!row.isConnected || !current || current.text !== original.text || current.startMs !== original.startMs || current.endMs !== original.endMs)
    throw new Error("Caption changed during repair. Reopen repair for the current text.");
  generated[index] = repaired;
  repairDrafts.delete(index);
  row.querySelector(".caption-repair")?.remove();
  refreshPreview();
}

async function repairCaption(row: HTMLElement, index: number, cue: Cue, message: HTMLElement, stop: HTMLButtonElement, selectedLanguage: string, known: PartialWordTimings, onPartial: (bounds: PartialWordTimings) => void): Promise<void> {
  if (repairController || controller || renderController) return;
  const video = captionSource;
  if (!video) { message.textContent = "Choose the original video first."; return; }
  const operation = new AbortController();
  repairController = operation;
  const controls = [...document.querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement | HTMLTextAreaElement>("input, button, select, textarea")];
  const disabled = controls.map(control => control.disabled);
  controls.forEach(control => { control.disabled = true; });
  stop.hidden = false; stop.disabled = false;
  studio.setBusy(true); preview.pause(); endingStudio.pause();
  let artifactId: string | undefined, sessionId: string | undefined;
  try {
    serializeSubRip([cue]);
    if (cue.endMs - cue.startMs > 30_000 || cue.text.length > 2000) throw new Error("Split this caption before repair: maximum 30 seconds and 2,000 characters.");
    message.textContent = "Reading source audio locally…";
    const pcm = await extractPcm(video, operation.signal);
    if (cue.endMs > pcm.durationMs) throw new Error("This caption ends after the source audio. Correct its bounds first.");
    message.textContent = "Requesting the speech model selected in Oryn…";
    const session = await localAI.models.request({
      role: "video-subtitle-native-timing-fallback",
      preferredModel: WHISPER_TIMED_ASR,
      certifiedModels: [WHISPER_TIMED_ASR],
      requirements: {...(capabilities.features.taskProfiles ? { contract: {id: "transcription-v1", revision: 1}, requiredFeatures: ["wordTimestamps"], parameterValues: {language: selectedLanguage, timestamps: true} } : {}), task: "automatic-speech-recognition", inputModalities: ["audio"], outputModalities: ["text"], capabilities: {speechRecognition: true}},
      substitution: {automatic: "none"},
    }, {signal: operation.signal});
    sessionId = session.sessionId;
    operation.signal.throwIfAborted();
    if (session.capabilities.task !== "automatic-speech-recognition") throw new Error("Select a speech-recognition model with word timestamps in Oryn.");
    let recovered = known;
    // Try the cue, then retry with a small amount of neighboring audio to avoid
    // cutting speech at edited boundaries. Only this cue's timings are changed.
    for (const paddingMs of [0, Math.min(750, Math.max(0, (30_000 - (cue.endMs - cue.startMs)) / 2))]) {
      const cropStartMs = Math.max(0, cue.startMs - paddingMs);
      const cropEndMs = Math.min(pcm.durationMs, cue.endMs + paddingMs);
      const samples = pcm.samples.slice(Math.round(cropStartMs * SAMPLE_RATE / 1000), Math.round(cropEndMs * SAMPLE_RATE / 1000));
      const artifact = await localAI.artifacts.create(pcmWave(samples, SAMPLE_RATE), {lifetime: "session", name: `caption-${index + 1}-repair.wav`});
      artifactId = artifact.id;
      operation.signal.throwIfAborted();
      message.textContent = paddingMs ? "Recovering remaining word timings with surrounding audio…" : "Regenerating this caption’s word timings…";
      const result = await invokeAndWait<{audioArtifactId: string; language: string; timestamps: boolean}, TranscriptResult>(localAI, {
        sessionId, input: {audioArtifactId: artifactId, language: selectedLanguage, timestamps: true},
      }, {signal: operation.signal});
      operation.signal.throwIfAborted();
      if (captionSource !== video || !row.isConnected) throw new Error("Source changed. Reopen repair.");
      const candidate = recoveredSegments(cue, result.value.segments ?? [], cropStartMs);
      // A complete fresh alignment takes precedence over old slots which may
      // cover a word that has since been split. Otherwise retain useful partials.
      recovered = candidate.every(Boolean) ? candidate : mergeWordTimings(cue, recovered, candidate);
      onPartial(recovered);
      await localAI.artifacts.release(artifactId).catch(() => undefined);
      artifactId = undefined;
      if (recovered.every(Boolean)) {
        commitRepair(row, index, cue, withWordBounds(cue, recovered as Array<{startMs: number; endMs: number}>));
        return;
      }
    }
    const count = recovered.filter(Boolean).length;
    message.textContent = `Recovered ${count} of ${recovered.length} word timings after two attempts. Known timings are filled in below; only ${recovered.length - count} word(s) still need timing. Your corrected text is unchanged.`;
    const manual = row.querySelector<HTMLDetailsElement>(".caption-repair details");
    if (manual) manual.open = true;
  } catch (error) {
    message.textContent = operation.signal.aborted ? "Repair cancelled. Your corrected text and other captions are unchanged." : `Could not repair this caption${error instanceof LocalAIError ? ` (${error.code})` : ""}: ${error instanceof Error ? error.message : "Unknown error"}. Known timings remain filled in below. Your corrected text is preserved.`;
  } finally {
    if (artifactId) await localAI.artifacts.release(artifactId).catch(() => undefined);
    if (sessionId) await localAI.models.releaseSession(sessionId).catch(() => undefined);
    controls.forEach((control, position) => { control.disabled = disabled[position]!; });
    stop.hidden = true; repairController = undefined; saveVideo.disabled = !renderedBlob; studio.setBusy(false); updateTimingStatus(); updateQueue();
  }
}

function readCues(): Cue[] {
  return [...cuesRoot.querySelectorAll<HTMLElement>(".cue")].map((row, index) => {
    const inputs = row.querySelectorAll<HTMLInputElement>("input");
    const cue = retainTiming({
      startMs: Number(inputs[0]?.value),
      endMs: Number(inputs[1]?.value),
      text: row.querySelector("textarea")?.value.trim() ?? "",
    }, generated[index]);
    if (!hasWordTiming(cue)) {
      const known = mergeWordTimings(cue,
        recoverWordTimings(cue, repairDrafts.get(index)?.spans ?? []),
        recoverWordTimings(cue, generated[index]?.wordTimingHints ?? generated[index]?.spans ?? []));
      const words = cue.text.split(/\s+/u);
      cue.wordTimingHints = known.flatMap((span, n) => span ? [{...span, text: words[n]!}] : []);
    }
    return cue;
  });
}
function refreshPreview(): void {
  clearRendered();
  if (!restoringVideo && activeVideo) activeVideo.reviewed = false;
  updateTimingStatus();
  studio.refresh();
  try {
    srtPreview.textContent = serializeSubRip(readCues());
  } catch (error) {
    srtPreview.textContent =
      error instanceof Error ? error.message : "Invalid cues";
  }
  if (!restoringVideo) { snapshotVideo(); updateQueue(); }
}
async function save(): Promise<void> {
  const button = required<HTMLButtonElement>("#save");
  button.disabled = true;
  let id: string | undefined;
  try {
    const name = `${baseName(activeVideo?.file.name ?? "subtitles")}.srt`;
    const blob = new Blob([serializeSubRip(readCues())], {type: "application/x-subrip;charset=utf-8"});
    const artifact = await localAI.artifacts.create(blob, {lifetime: "session", name});
    id = artifact.id;
    await localAI.artifacts.saveAs(id, {suggestedFilename: name});
    stage("SRT publication", "done");
  } catch (error) {
    status.textContent = `Could not save subtitles: ${(error as Error).message}`;
  } finally {
    if (id) await localAI.artifacts.release(id).catch(() => undefined);
    button.disabled = busy();
  }
}
function stage(
  label: string,
  state: "running" | "done" | "warning",
  detail = "",
): void {
  let item = [...stages.children].find(
    (value) => value.getAttribute("data-label") === label,
  ) as HTMLElement | undefined;
  if (!item) {
    item = document.createElement("li");
    item.dataset.label = label;
    stages.append(item);
  }
  item.className = state;
  item.textContent = `${label}${detail ? ` · ${detail}` : ""}`;
}
function formatBytes(value: number): string {
  return value >= 1024 ** 3
    ? `${(value / 1024 ** 3).toFixed(1)} GB`
    : `${Math.round(value / 1024 ** 2)} MB`;
}
function baseName(value: string): string {
  return (
    value
      .replace(/\.[^.]+$/u, "")
      .replace(/[^\p{L}\p{N}._-]+/gu, "-")
      .slice(0, 80) || "subtitles"
  );
}
function required<T extends Element>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (!value) throw new Error(`Missing ${selector}`);
  return value;
}

async function renderVideo(): Promise<void> {
  if (!captionSource || renderController || controller || repairController || batchOperation === "download") return;
  let ending;
  try { ending = endingStudio.getEnding(); }
  catch (error) { renderStatus.textContent = (error as Error).message; return; }
  if (requiresWordTiming(captionStyle.value)) {
    try { requireWordTiming(readCues()); }
    catch (error) { renderStatus.textContent = (error as Error).message; return; }
  }
  renderController = new AbortController();
  clearRendered();
  preview.pause();
  endingStudio.pause();
  studio.setBusy(true);
  const controls = [...document.querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement | HTMLTextAreaElement>("input, button, select, textarea")];
  const disabled = controls.map(control => control.disabled);
  controls.forEach(control => { control.disabled = true; });
  renderCancel.disabled = false;
  renderCancel.hidden = false;
  renderCanvas.hidden = false;
  renderProgress.hidden = false;
  renderProgress.value = 0;
  renderStatus.textContent = "Preparing video for export… You can switch tabs; keep this page open.";
  const signal = renderController.signal;
  updateQueue();
  try {
    renderedBlob = await renderCaptionedVideo({
      file: captionSource,
      cues: readCues(),
      canvas: renderCanvas,
      style: captionStyle.value as CaptionStyle,
      design: studio.getDesign(),
      ending,
      signal,
      onProgress: fraction => {
        renderProgress.value = fraction;
        renderStatus.textContent = `Rendering locally · ${Math.floor(fraction * 100)}% · you can switch tabs; keep this page open`;
      },
    });
    renderedURL = URL.createObjectURL(renderedBlob);
    renderedPreview.src = renderedURL;
    renderedPreview.hidden = false;
    renderStatus.textContent = `Video ready · ${renderedBlob.type.startsWith("video/mp4") ? "MP4" : "WebM"} · ${formatBytes(renderedBlob.size)}${ending ? " · ending included" : ""}. Preview it, then save. Captions are permanently drawn into the video.`;
  } catch (error) {
    renderStatus.textContent = signal.aborted
      ? "Rendering cancelled. No partial video was saved."
      : `Could not render video: ${error instanceof Error ? error.message : "Unknown error"}`;
    if (activeVideo) activeVideo.error = renderStatus.textContent;
  } finally {
    controls.forEach((control, index) => { control.disabled = disabled[index]!; });
    saveVideo.disabled = !renderedBlob;
    renderCancel.hidden = true;
    renderCanvas.hidden = true;
    renderProgress.hidden = true;
    renderController = undefined;
    studio.setBusy(false);
    snapshotVideo(); updateQueue();
  }
}

async function saveRenderedVideo(): Promise<void> {
  if (!renderedBlob || !captionSource) return;
  saveVideo.disabled = true;
  let id: string | undefined;
  try {
    const name = videoFilename(captionSource, renderedBlob);
    renderStatus.textContent = "Preparing video for the save dialog…";
    const artifact = await localAI.artifacts.create(renderedBlob, { lifetime: "session", name });
    id = artifact.id;
    await localAI.artifacts.saveAs(id, { suggestedFilename: name });
    renderStatus.textContent = "Save dialog closed. The rendered video remains available to preview or save again.";
  } catch (error) {
    renderStatus.textContent = `Could not save video: ${error instanceof Error ? error.message : "Unknown error"}`;
  } finally {
    if (id) await localAI.artifacts.release(id).catch(() => undefined);
    saveVideo.disabled = !renderedBlob;
  }
}

function updateTimingStatus(): void {
  const cues = readCues();
  const count = cues.filter(hasWordTiming).length;
  const complete = cues.length > 0 && count === cues.length;
  cuesRoot.querySelectorAll<HTMLElement>(".cue-timing").forEach((label, index) => {
    const ready = hasWordTiming(cues[index]!);
    const button = label.parentElement?.querySelector<HTMLButtonElement>(".repair-caption");
    if (button) button.hidden = ready;
    label.textContent = ready ? "Word timings available" : "No usable word timings — word highlighting needs alignment for this caption";
  });
  required<HTMLElement>("#timing-status").textContent = complete
    ? `Word timing ready: ${count} / ${cues.length} captions. Corrections use original word timing bounds; review their synchronization.`
    : `Word timing missing: ${cues.length - count} / ${cues.length} captions. SRT alone, phrase-only timing, inserted/split words, and changed boundaries cannot drive word highlighting. Generate subtitles from the video, then optionally import your corrected SRT with matching cue boundaries. Or import saved Oryn timing JSON. Plain captions remain available.`;
  renderButton.disabled = !complete && requiresWordTiming(captionStyle.value);
  renderStatus.textContent = renderButton.disabled ? "This word animation needs individual word timings. Generate subtitles or import Oryn timing JSON; otherwise choose No animation, Fade, or Pop in." : "Ready to render.";
  try { endingStudio.getEnding(); }
  catch (error) { renderButton.disabled = true; renderStatus.textContent = (error as Error).message; }
}

async function saveTiming(): Promise<void> {
  const button = required<HTMLButtonElement>("#save-timing");
  button.disabled = true;
  let id: string | undefined;
  try {
    const blob = new Blob([serializeCaptionTiming(readCues())], {type: "application/json"});
    const name = `${baseName(captionSource?.name ?? "subtitles")}.oryn-captions.json`;
    const artifact = await localAI.artifacts.create(blob, {lifetime: "session", name});
    id = artifact.id;
    await localAI.artifacts.saveAs(id, {suggestedFilename: name});
    renderStatus.textContent = "Timing-file save dialog closed. Import this JSON with the same source video to restore available word timings.";
  } catch (error) {
    renderStatus.textContent = `Could not save timings: ${error instanceof Error ? error.message : "Unknown error"}`;
  } finally {
    if (id) await localAI.artifacts.release(id).catch(() => undefined);
    button.disabled = false;
  }
}
