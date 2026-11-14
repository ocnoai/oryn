import { StrictMode, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';

import type { ModelWorkspaceState, ModelWorkspaceTaskContract } from '../../shared/ipc';
import { formatStorageSize as formatBytes } from '../../shared/format-storage-size';
import { Badge, Button, Card, EmptyState, Icon, ProgressBar } from './components/ui';
import './components/ui/ui.css';
import {
  generationDefaults,
  generationProfileSource,
  isGenerationProfileDirty,
} from './model-workspace-generation-profile';
import './model-workspace.css';
import {
  forecastInput,
  numericControl,
  poseBoxes,
  tabularInput,
  numericMatrix,
  choiceValues,
} from './model-workspace-sprint61';

type FormValue = Record<string, string | number | boolean>;
type Artifact = { id: string; mimeType: string; sizeBytes: number; name?: string };
type JobEvent = { event: string; jobId: string; data?: Record<string, unknown> };
type ChatMessage = { role: 'user' | 'assistant'; content: string };
type WorkspaceFailure = { code?: string; message: string };

const initialForm: FormValue = {
  prompt: '',
  lyrics: '',
  text: '',
  query: '',
  documents: '',
  instruction: '',
  question: '',
  context: '',
  labels: '',
  multiLabel: false,
  pointX: 50,
  pointY: 50,
  topK: 5,
  maxTokens: 256,
  maxAnswerLength: 64,
  threshold: 0.5,
  width: 512,
  height: 512,
  steps: 20,
  guidanceScale: 7.5,
  seed: 0,
  durationSeconds: 10,
  frames: 16,
  fps: 8,
  strength: 0.8,
  voice: 'af_heart',
  language: 'en-us',
  speed: 1,
  normalize: true,
  timestamps: false,
  table: 'City\tCountry\nParis\tFrance\nRome\tItaly',
  ocr: '',
  startSeconds: 0,
  keypointThreshold: 0.005,
  maxKeypoints: 256,
  minSpeechSeconds: 0.25,
  minSilenceSeconds: 0.1,
  horizon: 16,
  quantiles: '',
  personBoxes: '0, 0, 100, 100',
  featureColumns: 'temperature\tnumeric',
  trainingRows: '1\n2',
  trainingTargets: 'cold\nwarm',
  queryRows: '1.5',
  audioOperation: 'enhance',
  sourceGrid: '',
  outputMode: 'text',
  omniSource: 'image',
  videoOperation: 'restore',
};

function ModelWorkspace(): React.JSX.Element {
  const [workspace, setWorkspace] = useState<ModelWorkspaceState>({
    status: 'error',
    message: 'Opening model workspace…',
  });
  const [form, setForm] = useState<FormValue>(initialForm);
  const [inputFile, setInputFile] = useState<File>();
  const [secondaryFile, setSecondaryFile] = useState<File>();
  const [referenceFiles, setReferenceFiles] = useState<(File | undefined)[]>([]);
  const [jobId, setJobId] = useState<string>();
  const activeJob = useRef<string | undefined>(undefined);
  const [stage, setStage] = useState('');
  const [progress, setProgress] = useState<number>();
  const [stream, setStream] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const taskContractRef = useRef<ModelWorkspaceTaskContract | undefined>(undefined);
  const [result, setResult] = useState<unknown>();
  const [artifact, setArtifact] = useState<Artifact>();
  const [artifactURL, setArtifactURL] = useState<string>();
  const [error, setError] = useState<WorkspaceFailure>();

  useEffect(() => {
    void window.orynWorkspace.getState().then(setWorkspace);
  }, []);
  useEffect(() => {
    setReferenceFiles([]);
  }, [workspace.taskProfile?.profileId]);
  useEffect(() => {
    if (!workspace.appearance) return;
    document.documentElement.dataset.theme = workspace.appearance.theme;
    if (workspace.appearance.density === 'default') delete document.documentElement.dataset.density;
    else document.documentElement.dataset.density = workspace.appearance.density;
  }, [workspace.appearance]);
  useEffect(() => {
    if (!workspace.generationProfile) return;
    setForm((value) => ({ ...value, ...generationDefaults(workspace.generationProfile!) }));
  }, [workspace.generationProfile]);
  useEffect(() => {
    taskContractRef.current = workspace.taskContract;
  }, [workspace.taskContract]);
  useEffect(() => {
    if (!workspace.taskProfile) return;
    const defaults = Object.fromEntries(
      Object.entries(workspace.taskProfile.parameters)
        .filter(([, parameter]) =>
          ['string', 'number', 'boolean'].includes(typeof parameter.default),
        )
        .map(([key, parameter]) => [
          key === 'inputMode'
            ? 'omniSource'
            : key === 'operation'
              ? workspace.taskContract === 'video-editing-v1'
                ? 'videoOperation'
                : 'audioOperation'
              : key,
          parameter.default,
        ]),
    ) as FormValue;
    if (
      workspace.taskContract === 'audio-transformation-v1' ||
      workspace.taskContract === 'video-editing-v1'
    ) {
      const key =
        workspace.taskContract === 'video-editing-v1' ? 'videoOperation' : 'audioOperation';
      const values = choiceValues(
        workspace.taskProfile.parameters.operation?.valueSchema,
        key === 'videoOperation'
          ? ['restore', 'edit']
          : ['enhance', 'separate', 'voice-conversion', 'encode', 'decode', 'vocode'],
      );
      defaults[key] = values.includes(String(defaults[key]))
        ? String(defaults[key])
        : (values[0] ?? '');
    }
    if (workspace.taskContract === 'multimodal-generation-v1') {
      const modes = choiceValues(workspace.taskProfile.parameters.inputMode?.valueSchema, [
        'image',
        'audio',
        'image-and-audio',
      ]).filter(
        (mode) =>
          (mode === 'audio' || workspace.taskProfile!.inputModalities.includes('image')) &&
          (mode === 'image' || workspace.taskProfile!.inputModalities.includes('audio')),
      );
      defaults.omniSource = modes.includes(String(defaults.omniSource))
        ? String(defaults.omniSource)
        : (modes[0] ?? '');
      const outputs = choiceValues(workspace.taskProfile.parameters.outputMode?.valueSchema, [
        'text',
        'audio',
        'text-and-audio',
      ]).filter(
        (mode) =>
          (mode === 'audio' || workspace.taskProfile!.outputModalities.includes('text')) &&
          (mode === 'text' || workspace.taskProfile!.outputModalities.includes('audio')),
      );
      defaults.outputMode = outputs.includes(String(defaults.outputMode))
        ? String(defaults.outputMode)
        : (outputs[0] ?? '');
    }
    setForm((value) => ({
      ...value,
      ...defaults,
    }));
  }, [workspace.taskProfile]);
  useEffect(
    () =>
      window.orynWorkspace.onJobEvent((unknownEvent) => {
        const event = unknownEvent as JobEvent;
        if (event.jobId !== activeJob.current) return;
        if (event.event === 'job.output.delta' && typeof event.data?.delta === 'string')
          setStream((value) => value + event.data!.delta);
        if (event.event === 'job.progress') {
          setStage(typeof event.data?.stage === 'string' ? event.data.stage : 'Working locally');
          setProgress(typeof event.data?.progress === 'number' ? event.data.progress : undefined);
        }
        if (event.event === 'job.preview' && event.data?.output)
          void showOutput({ kind: 'artifact', artifact: event.data.output });
        if (event.event === 'job.completed') {
          void showOutput(event.data?.result);
          setJobId(undefined);
          activeJob.current = undefined;
          setStage('Complete');
          setProgress(1);
        }
        if (event.event === 'job.failed' || event.event === 'job.cancelled') {
          const failure = event.data?.error as { code?: string; message?: string } | undefined;
          setError(
            event.event === 'job.cancelled'
              ? { code: 'JOB_CANCELLED', message: 'Generation cancelled.' }
              : {
                  code: failure?.code,
                  message: failure?.message ?? 'Local generation failed.',
                },
          );
          setJobId(undefined);
          activeJob.current = undefined;
          setStage('');
          setProgress(undefined);
        }
      }),
    [],
  );

  const showOutput = async (output: unknown) => {
    const resolved = output as
      { kind?: string; value?: unknown; artifact?: Artifact; outputs?: unknown[] } | undefined;
    if (resolved?.kind === 'structured') {
      setResult(resolved.value);
      const value = resolved.value as { text?: unknown } | undefined;
      if (taskContractRef.current === 'chat-v1' && typeof value?.text === 'string') {
        setChatMessages((messages) => [
          ...messages,
          { role: 'assistant', content: value.text as string },
        ]);
      }
      return;
    }
    if (resolved?.kind === 'artifact' && resolved.artifact) {
      setArtifact(resolved.artifact);
      const url = await window.orynWorkspace.getArtifactUrl(resolved.artifact.id);
      if (url.ok) setArtifactURL(url.value);
      return;
    }
    if (resolved?.kind === 'multi') {
      setArtifact(undefined);
      setArtifactURL(undefined);
      setResult(resolved);
      return;
    }
    setResult(output);
  };

  const contract = workspace.taskContract;
  const submit = async () => {
    if (!contract || workspace.status !== 'ready' || jobId) return;
    setError(undefined);
    setResult(undefined);
    setArtifact(undefined);
    setArtifactURL(undefined);
    setStream('');
    setProgress(undefined);
    setStage('Validating input');
    const uploaded: string[] = [];
    const uploadInput = async (file?: File) => {
      const id = await upload(file);
      uploaded.push(id);
      return id;
    };
    try {
      const references =
        contract === 'image-editing-v1' ? referenceFiles.filter((f): f is File => !!f) : [];
      if (contract === 'image-editing-v1') {
        const minimum = workspace.taskProfile?.limits.minReferenceImages ?? 1;
        const maximum = workspace.taskProfile?.limits.maxReferenceImages ?? 1;
        if (1 + references.length < minimum || 1 + references.length > maximum)
          throw new Error(
            `Choose ${minimum === maximum ? minimum : `${minimum} to ${maximum}`} reference images in order.`,
          );
      }
      const artifactId =
        ((needsFile(contract) ||
          (contract === 'speech-synthesis-v1' &&
            workspace.taskProfile?.features.referenceAudio === 'supported')) &&
          !(
            contract === 'audio-transformation-v1' &&
            ['decode', 'vocode'].includes(String(form.audioOperation))
          )) ||
        ((contract === 'chat-v1' || contract === 'text-to-image-v1') &&
          workspace.capabilities?.imageInput === true &&
          inputFile)
          ? await uploadInput(inputFile)
          : undefined;
      const secondaryArtifactId =
        (contract === 'audio-transformation-v1' && form.audioOperation === 'voice-conversion') ||
        (contract === 'multimodal-generation-v1' && form.omniSource === 'image-and-audio') ||
        (contract === 'video-editing-v1' && form.videoOperation === 'edit' && secondaryFile)
          ? await uploadInput(secondaryFile)
          : undefined;
      const referenceIds: string[] = [];
      for (const file of references) referenceIds.push(await uploadInput(file));
      const input = buildInput(
        contract,
        form,
        artifactId,
        workspace.capabilities,
        chatMessages,
        secondaryArtifactId,
        referenceIds,
      );
      const response = await window.orynWorkspace.invoke(input);
      if (!response.ok) {
        setError({ code: response.error.code, message: response.error.message });
        setStage('');
        return;
      }
      activeJob.current = response.value.jobId;
      setJobId(response.value.jobId);
      if (contract === 'chat-v1') {
        setChatMessages((messages) => [
          ...messages,
          { role: 'user', content: String(form.prompt).trim() },
        ]);
        setForm({ ...form, prompt: '' });
      }
      setStage(response.value.state === 'queued' ? 'Queued' : 'Starting locally');
    } catch (cause) {
      setError({
        message:
          cause instanceof Error ? cause.message : 'The workspace input could not be submitted.',
      });
      setStage('');
    } finally {
      // Job admission acquires its own input leases before acknowledging the
      // invocation. Release the workspace's upload ownership on every outcome.
      await Promise.all(uploaded.map((id) => window.orynWorkspace.releaseArtifact(id)));
    }
  };

  const cancel = async () => {
    if (!jobId) return;
    await window.orynWorkspace.cancel(jobId);
  };

  if (workspace.status !== 'ready' || !contract) {
    return (
      <main className="workspace recovery">
        <EmptyState
          icon="alert"
          title="Model workspace unavailable"
          description={workspace.message ?? 'This model cannot currently be opened.'}
          action={
            <Button onClick={() => void window.orynWorkspace.openSource()}>Open model page</Button>
          }
        />
      </main>
    );
  }

  return (
    <main className="workspace">
      <header className="workspace-header">
        <div className="model-mark">
          <Icon name="sparkles" />
        </div>
        <div>
          <span className="ui-eyebrow">Oryn model workspace</span>
          <h1>{workspace.repository}</h1>
          <p>{surfaceTitle(contract)} · exact local revision</p>
          {workspace.runtimeSelection && (
            <p className="runtime-selection">
              {workspace.runtimeSelection.automatic ? 'Automatic' : 'User selected'} ·{' '}
              {workspace.runtimeSelection.foundationId.replace('oryn.runtime.', '')} ·{' '}
              {workspace.runtimeSelection.backend} · {workspace.runtimeSelection.precision}
              {workspace.runtimeSelection.quantization
                ? ` · ${workspace.runtimeSelection.quantization}`
                : ''}
              {' · '}
              {workspace.runtimeSelection.policy.replace('-', ' ')} · ~
              {formatBytes(workspace.runtimeSelection.estimatedMemoryBytes)} peak local use
            </p>
          )}
        </div>
        <Badge tone={workspace.trustMode === 'user-trusted' ? 'warning' : 'success'}>
          {workspace.trustMode === 'user-trusted'
            ? 'Local · user trusted'
            : 'Local · Oryn verified'}
        </Badge>
        <Button size="sm" onClick={() => void window.orynWorkspace.openSource()}>
          Model page
        </Button>
      </header>
      <div className="workspace-grid">
        <Card className="input-card">
          <section className="surface-heading">
            <span className="step">01</span>
            <div>
              <h2>{surfaceTitle(contract)}</h2>
              <p>{surfaceDescription(contract)}</p>
            </div>
          </section>
          <TaskForm
            contract={contract}
            capabilities={workspace.capabilities}
            generationProfile={workspace.generationProfile}
            taskProfile={workspace.taskProfile}
            form={form}
            setForm={setForm}
            file={inputFile}
            setFile={setInputFile}
            secondaryFile={secondaryFile}
            setSecondaryFile={setSecondaryFile}
            referenceFiles={referenceFiles}
            setReferenceFiles={setReferenceFiles}
            chatMessages={chatMessages}
            clearChat={() => {
              setChatMessages([]);
              setStream('');
              setResult(undefined);
            }}
          />
          {error && (
            <div className="workspace-error" role="alert">
              <strong>{failureTitle(error.code)}</strong>
              <p>{error.message}</p>
              {error.code === 'MODEL_MEMORY_INSUFFICIENT' &&
                workspace.runtimeSelection?.estimatedMemoryBytes && (
                  <p>
                    Estimated peak local use:{' '}
                    {formatBytes(workspace.runtimeSelection.estimatedMemoryBytes)}.
                  </p>
                )}
              {error.code && (
                <details>
                  <summary>Technical details</summary>
                  <code>{error.code}</code>
                </details>
              )}
            </div>
          )}
          {jobId ? (
            <Button variant="danger" onClick={() => void cancel()}>
              <Icon name="close" /> Stop
            </Button>
          ) : (
            <Button variant="primary" onClick={() => void submit()}>
              <Icon name="sparkles" /> {error ? 'Try again' : actionLabel(contract)}
            </Button>
          )}
          <p className="local-note">
            This task runs through Oryn’s signed runtime, global scheduler, and cancellable job
            system.
          </p>
        </Card>
        <Card className="output-card">
          <section className="surface-heading">
            <span className="step">02</span>
            <div>
              <h2>Result</h2>
              <p>Outputs remain temporary until you save them.</p>
            </div>
          </section>
          {(jobId || stage) && (
            <div className="job-status">
              <strong>{stage || 'Working locally'}</strong>
              {progress !== undefined && (
                <ProgressBar label="Generation progress" value={progress} max={1} />
              )}
            </div>
          )}
          <Output
            contract={contract}
            stream={stream}
            result={result}
            artifact={artifact}
            artifactURL={artifactURL}
          />
          {artifact && (
            <div className="result-actions">
              <Button
                variant="primary"
                onClick={() => void window.orynWorkspace.saveArtifactAs(artifact.id, artifact.name)}
              >
                Save As…
              </Button>
              <span>
                {artifact.mimeType} · {formatBytes(artifact.sizeBytes)}
              </span>
            </div>
          )}
          {result !== undefined && !artifact && (
            <div className="result-actions">
              <Button
                variant="primary"
                onClick={() =>
                  void exportJSON(
                    contract === 'chat-v1' ? { messages: chatMessages } : result,
                    contract === 'chat-v1'
                      ? 'oryn-conversation.json'
                      : `oryn-${contract}-result.json`,
                  ).catch((error: Error) =>
                    setError({ code: 'ARTIFACT_IO_ERROR', message: error.message }),
                  )
                }
              >
                {contract === 'embeddings-v1'
                  ? 'Export vectors as JSON…'
                  : contract === 'chat-v1'
                    ? 'Export conversation…'
                    : 'Export result as JSON…'}
              </Button>
            </div>
          )}
        </Card>
      </div>
    </main>
  );
}

function failureTitle(code?: string): string {
  const titles: Record<string, string> = {
    MODEL_MEMORY_INSUFFICIENT: 'Not enough available memory',
    MODEL_FORMAT_UNSUPPORTED: 'Model format is not supported',
    MODEL_RUNTIME_INCOMPATIBLE: 'Model and runtime are not compatible',
    MODEL_GENERATION_FAILED: 'Image generation failed',
    MODEL_INPUT_INVALID: 'Check the generation settings',
    RUNTIME_DEPENDENCY_UNAVAILABLE: 'Required local runtime is unavailable',
    ARTIFACT_IO_ERROR: 'The result could not be saved',
    PACK_TIMEOUT: 'Generation took too long',
    PACK_CRASHED: 'The local runtime stopped',
  };
  return code ? (titles[code] ?? 'The Local AI job could not be completed') : 'Local AI failed';
}

function TaskForm({
  contract,
  capabilities,
  generationProfile,
  taskProfile,
  form,
  setForm,
  file,
  setFile,
  secondaryFile,
  setSecondaryFile,
  referenceFiles,
  setReferenceFiles,
  chatMessages,
  clearChat,
}: {
  contract: ModelWorkspaceTaskContract;
  capabilities?: Record<string, unknown>;
  generationProfile?: ModelWorkspaceState['generationProfile'];
  taskProfile?: ModelWorkspaceState['taskProfile'];
  form: FormValue;
  setForm: (value: FormValue) => void;
  file?: File;
  setFile: (file?: File) => void;
  secondaryFile?: File;
  setSecondaryFile: (file?: File) => void;
  referenceFiles: (File | undefined)[];
  setReferenceFiles: (files: (File | undefined)[]) => void;
  chatMessages: ChatMessage[];
  clearChat: () => void;
}) {
  const set = (key: string, value: string | number | boolean) => setForm({ ...form, [key]: value });
  const limitsFor = (key: string, min: number, max: number, step = 1) =>
    numericControl(taskProfile?.parameters[key]?.valueSchema, { min, max, step });
  const choicesFor = (key: string, choices: string[]) =>
    choiceValues(taskProfile?.parameters[key]?.valueSchema, choices);
  if (contract === 'tabular-classification-v1' || contract === 'tabular-regression-v1')
    return (
      <>
        <Field label="Features · name and numeric or categorical type, separated by a tab">
          <textarea
            maxLength={4096}
            value={String(form.featureColumns)}
            onChange={(e) => set('featureColumns', e.target.value)}
          />
        </Field>
        <Field label="Training features · tab-separated cells, one row per line; empty cells mean missing values">
          <textarea
            maxLength={80000}
            value={String(form.trainingRows)}
            onChange={(e) => set('trainingRows', e.target.value)}
          />
        </Field>
        <Field label="Known training targets · one per line, in the same row order">
          <textarea
            maxLength={80000}
            value={String(form.trainingTargets)}
            onChange={(e) => set('trainingTargets', e.target.value)}
          />
        </Field>
        <Field label="Rows to predict · features only, in the same column order">
          <textarea
            maxLength={80000}
            value={String(form.queryRows)}
            onChange={(e) => set('queryRows', e.target.value)}
          />
        </Field>
      </>
    );
  if (contract === 'audio-transformation-v1')
    return (
      <>
        <Field label="Operation">
          <select
            value={String(form.audioOperation)}
            onChange={(e) => set('audioOperation', e.target.value)}
          >
            {choicesFor('operation', [
              'enhance',
              'separate',
              'voice-conversion',
              'encode',
              'decode',
              'vocode',
            ]).map((operation) => (
              <option key={operation}>{operation}</option>
            ))}
          </select>
        </Field>
        {['decode', 'vocode'].includes(String(form.audioOperation)) ? (
          <Field
            label={
              form.audioOperation === 'decode'
                ? 'Code indices · one frame per line'
                : 'Spectrogram · one frequency bin per line'
            }
          >
            <textarea
              maxLength={80000}
              value={String(form.sourceGrid)}
              onChange={(e) => set('sourceGrid', e.target.value)}
            />
          </Field>
        ) : (
          <FileField accept="audio/wav,audio/x-wav" file={file} onChange={setFile} />
        )}
        {form.audioOperation === 'voice-conversion' && (
          <Field label="Reference voice">
            <FileField
              accept="audio/wav,audio/x-wav"
              file={secondaryFile}
              onChange={setSecondaryFile}
            />
          </Field>
        )}
      </>
    );
  if (contract === 'multimodal-generation-v1')
    return (
      <>
        <Field label="Input">
          <select
            value={String(form.omniSource)}
            onChange={(e) => set('omniSource', e.target.value)}
          >
            {choicesFor('inputMode', ['image', 'audio', 'image-and-audio']).includes('image') &&
              taskProfile?.inputModalities.includes('image') && (
                <option value="image">Image</option>
              )}
            {choicesFor('inputMode', ['image', 'audio', 'image-and-audio']).includes('audio') &&
              taskProfile?.inputModalities.includes('audio') && (
                <option value="audio">Audio</option>
              )}
            {choicesFor('inputMode', ['image', 'audio', 'image-and-audio']).includes(
              'image-and-audio',
            ) &&
              taskProfile?.inputModalities.includes('image') &&
              taskProfile.inputModalities.includes('audio') && (
                <option value="image-and-audio">Image and audio</option>
              )}
          </select>
        </Field>
        <FileField
          accept={
            form.omniSource === 'audio'
              ? 'audio/wav,audio/x-wav'
              : 'image/png,image/jpeg,image/webp'
          }
          file={file}
          onChange={setFile}
        />
        {form.omniSource === 'image-and-audio' && (
          <FileField
            accept="audio/wav,audio/x-wav"
            file={secondaryFile}
            onChange={setSecondaryFile}
          />
        )}
        <Field label="Instruction">
          <textarea
            maxLength={2048}
            value={String(form.instruction)}
            onChange={(e) => set('instruction', e.target.value)}
          />
        </Field>
        <Field label="Output">
          <select
            value={String(form.outputMode)}
            onChange={(e) => set('outputMode', e.target.value)}
          >
            {choicesFor('outputMode', ['text', 'audio', 'text-and-audio'])
              .filter(
                (mode) =>
                  (mode === 'audio' || taskProfile?.outputModalities.includes('text')) &&
                  (mode === 'text' || taskProfile?.outputModalities.includes('audio')),
              )
              .map((mode) => (
                <option key={mode}>{mode}</option>
              ))}
          </select>
        </Field>
        <NumberField
          label="Maximum answer tokens"
          value={Number(form.maxTokens)}
          {...limitsFor('maxTokens', 1, 256)}
          onChange={(v) => set('maxTokens', v)}
        />
      </>
    );
  if (contract === 'video-editing-v1')
    return (
      <>
        <FileField accept="video/mp4,video/webm" file={file} onChange={setFile} />
        <Field label="Operation">
          <select
            value={String(form.videoOperation)}
            onChange={(e) => set('videoOperation', e.target.value)}
          >
            {choicesFor('operation', ['restore', 'edit']).map((operation) => (
              <option key={operation}>{operation}</option>
            ))}
          </select>
        </Field>
        {form.videoOperation === 'edit' && (
          <Field label="Editing instruction">
            <textarea
              maxLength={2048}
              value={String(form.prompt)}
              onChange={(e) => set('prompt', e.target.value)}
            />
          </Field>
        )}
        {form.videoOperation === 'edit' && (
          <Field label="Mask video · optional, aligned with the source clip">
            <FileField
              accept="video/mp4,video/webm"
              file={secondaryFile}
              onChange={setSecondaryFile}
            />
          </Field>
        )}
        <NumberField
          label="Clip start in seconds"
          value={Number(form.startSeconds)}
          {...limitsFor('startSeconds', 0, 29, 0.1)}
          onChange={(v) => set('startSeconds', v)}
        />
        <NumberField
          label="Clip duration in seconds"
          value={Number(form.durationSeconds)}
          {...limitsFor('durationSeconds', 1, 30, 0.1)}
          onChange={(v) => set('durationSeconds', v)}
        />
      </>
    );
  if (contract === 'pose-estimation-v1')
    return (
      <>
        <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
        <Field label="Person boxes · left, top, right, bottom percentages, one per line">
          <textarea
            maxLength={1024}
            value={String(form.personBoxes)}
            onChange={(e) => set('personBoxes', e.target.value)}
          />
        </Field>
      </>
    );
  if (contract === 'time-series-forecasting-v1')
    return (
      <>
        <Field label="Historical values · oldest to newest, separated by commas or spaces">
          <textarea
            maxLength={80000}
            value={String(form.context)}
            onChange={(e) => set('context', e.target.value)}
          />
        </Field>
        <NumberField
          label="Future values to predict"
          value={Number(form.horizon)}
          min={1}
          max={taskProfile?.limits.forecastHorizon ?? 512}
          onChange={(v) => set('horizon', v)}
        />
        {taskProfile?.features.quantiles === 'supported' && (
          <Field label="Quantiles · optional increasing levels, for example 0.1, 0.5, 0.9">
            <input
              maxLength={200}
              value={String(form.quantiles)}
              onChange={(e) => set('quantiles', e.target.value)}
            />
          </Field>
        )}
      </>
    );
  if (contract === 'audio-question-answering-v1')
    return (
      <>
        <FileField accept="audio/wav,audio/x-wav" file={file} onChange={setFile} />
        <Field label="Question or instruction about the audio">
          <textarea
            maxLength={2048}
            value={String(form.instruction)}
            onChange={(e) => set('instruction', e.target.value)}
          />
        </Field>
        <NumberField
          label="Maximum answer tokens"
          value={Number(form.maxTokens)}
          min={1}
          max={256}
          onChange={(v) => set('maxTokens', v)}
        />
      </>
    );
  if (contract === 'unconditional-image-generation-v1' || contract === 'image-text-to-video-v1')
    return (
      <>
        {contract === 'image-text-to-video-v1' && (
          <>
            <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
            <Field label="Describe the motion">
              <textarea
                maxLength={2048}
                value={String(form.prompt)}
                onChange={(e) => set('prompt', e.target.value)}
              />
            </Field>
          </>
        )}
        <div className="field-row">
          {(['width', 'height'] as const).map((key) => (
            <NumberField
              key={key}
              label={key === 'width' ? 'Width' : 'Height'}
              value={Number(form[key])}
              {...limitsFor(
                key,
                contract === 'unconditional-image-generation-v1' ? 8 : 64,
                1024,
                8,
              )}
              disabled={taskProfile?.parameters[key]?.handling === 'fixed'}
              onChange={(v) => set(key, v)}
            />
          ))}
        </div>
        <NumberField
          label="Inference steps"
          value={Number(form.steps)}
          {...limitsFor('steps', 1, contract === 'unconditional-image-generation-v1' ? 1000 : 100)}
          onChange={(v) => set('steps', v)}
        />
        <NumberField
          label="Seed"
          value={Number(form.seed)}
          min={0}
          max={2147483647}
          onChange={(v) => set('seed', v)}
        />
        {contract === 'image-text-to-video-v1' && (
          <>
            <NumberField
              label="Frames"
              value={Number(form.frames)}
              {...limitsFor('frames', 1, 129)}
              onChange={(v) => set('frames', v)}
            />
            <NumberField
              label="Frames per second"
              value={Number(form.fps)}
              {...limitsFor('fps', 1, 30)}
              onChange={(v) => set('fps', v)}
            />
            <NumberField
              label="Guidance scale"
              value={Number(form.guidanceScale)}
              {...limitsFor('guidanceScale', 0, 30, 0.1)}
              onChange={(v) => set('guidanceScale', v)}
            />
          </>
        )}
      </>
    );
  if (contract === 'zero-shot-classification-v1')
    return (
      <>
        <Field label="Text">
          <textarea
            maxLength={8192}
            value={String(form.text)}
            onChange={(e) => set('text', e.target.value)}
          />
        </Field>
        <Field label="Candidate labels · one per line">
          <textarea
            maxLength={4128}
            value={String(form.labels)}
            onChange={(e) => set('labels', e.target.value)}
          />
        </Field>
        <label>
          <input
            type="checkbox"
            checked={Boolean(form.multiLabel)}
            onChange={(e) => set('multiLabel', e.target.checked)}
          />{' '}
          Score each label independently
        </label>
      </>
    );
  if (contract === 'visual-document-retrieval-v1')
    return (
      <>
        <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
        <Field label="Search queries · one per line, up to two">
          <textarea
            maxLength={513}
            value={String(form.question)}
            onChange={(e) => set('question', e.target.value)}
          />
        </Field>
      </>
    );
  if (contract === 'table-question-answering-v1' || contract === 'document-question-answering-v1')
    return (
      <>
        {contract === 'document-question-answering-v1' && (
          <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
        )}
        <Field label="Question">
          <input
            maxLength={2048}
            value={String(form.question)}
            onChange={(e) => set('question', e.target.value)}
          />
        </Field>
        {contract === 'table-question-answering-v1' ? (
          <Field label="Table · paste tab-separated columns, with headings in the first row">
            <textarea
              maxLength={100000}
              value={String(form.table)}
              onChange={(e) => set('table', e.target.value)}
            />
          </Field>
        ) : (
          <Field label="Recognized words · one word per row, followed by tab-separated left, top, right and bottom percentages">
            <textarea
              maxLength={100000}
              value={String(form.ocr)}
              onChange={(e) => set('ocr', e.target.value)}
            />
          </Field>
        )}
      </>
    );
  if (contract === 'video-classification-v1' || contract === 'video-question-answering-v1')
    return (
      <>
        <FileField accept="video/mp4,video/webm" file={file} onChange={setFile} />
        <p>Use a clip up to 30 seconds and 64 MB. The model samples frames in temporal order.</p>
        <NumberField
          label="Start (seconds)"
          value={Number(form.startSeconds)}
          min={0}
          max={29}
          step={0.1}
          onChange={(v) => set('startSeconds', v)}
        />
        <NumberField
          label="Duration (seconds)"
          value={Number(form.durationSeconds)}
          min={0.1}
          max={30}
          step={0.1}
          onChange={(v) => set('durationSeconds', v)}
        />
        {contract === 'video-question-answering-v1' ? (
          <Field label="Question">
            <input
              maxLength={2048}
              value={String(form.question)}
              onChange={(e) => set('question', e.target.value)}
            />
          </Field>
        ) : (
          <NumberField
            label="Results"
            value={Number(form.topK)}
            min={1}
            max={20}
            onChange={(v) => set('topK', v)}
          />
        )}
      </>
    );
  if (contract === 'voice-activity-detection-v1')
    return (
      <>
        <FileField accept="audio/wav,audio/x-wav" file={file} onChange={setFile} />
        <NumberField
          label="Speech confidence"
          value={Number(form.threshold)}
          min={0.01}
          max={0.99}
          step={0.01}
          onChange={(v) => set('threshold', v)}
        />
        <NumberField
          label="Minimum speech (seconds)"
          value={Number(form.minSpeechSeconds)}
          min={0.032}
          max={2}
          step={0.01}
          onChange={(v) => set('minSpeechSeconds', v)}
        />
        <NumberField
          label="Minimum silence (seconds)"
          value={Number(form.minSilenceSeconds)}
          min={0.032}
          max={2}
          step={0.01}
          onChange={(v) => set('minSilenceSeconds', v)}
        />
      </>
    );
  if (
    [
      'keypoint-detection-v1',
      'image-embeddings-v1',
      'image-segmentation-v1',
      'mask-generation-v1',
      'zero-shot-object-detection-v1',
      'image-to-text-v1',
      'visual-question-answering-v1',
    ].includes(contract)
  )
    return (
      <>
        <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
        {contract === 'keypoint-detection-v1' && (
          <>
            <NumberField
              label="Point confidence"
              value={Number(form.keypointThreshold)}
              min={0}
              max={1}
              step={0.001}
              onChange={(v) => set('keypointThreshold', v)}
            />
            <NumberField
              label="Maximum points"
              value={Number(form.maxKeypoints)}
              min={1}
              max={1024}
              onChange={(v) => set('maxKeypoints', v)}
            />
          </>
        )}
        {contract === 'zero-shot-object-detection-v1' && (
          <>
            <Field label="Objects to find · one per line">
              <textarea
                maxLength={4128}
                value={String(form.labels)}
                onChange={(e) => set('labels', e.target.value)}
              />
            </Field>
            <NumberField
              label="Confidence threshold"
              value={Number(form.threshold)}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => set('threshold', v)}
            />
          </>
        )}
        {contract === 'image-embeddings-v1' && (
          <label>
            <input
              type="checkbox"
              checked={Boolean(form.normalize)}
              onChange={(e) => set('normalize', e.target.checked)}
            />{' '}
            Normalize the vector
          </label>
        )}
        {contract === 'visual-question-answering-v1' && (
          <Field label="Question">
            <input
              maxLength={2048}
              value={String(form.question)}
              onChange={(e) => set('question', e.target.value)}
            />
          </Field>
        )}
        {(contract === 'image-to-text-v1' || contract === 'visual-question-answering-v1') && (
          <NumberField
            label="Maximum answer tokens"
            value={Number(form.maxTokens)}
            min={1}
            max={256}
            onChange={(v) => set('maxTokens', v)}
          />
        )}
        {contract === 'mask-generation-v1' && (
          <>
            <p>
              Choose a point inside the object. Position is measured from the image’s top-left
              corner.
            </p>
            <NumberField
              label="Horizontal position (%)"
              value={Number(form.pointX)}
              min={0}
              max={100}
              onChange={(v) => set('pointX', v)}
            />
            <NumberField
              label="Vertical position (%)"
              value={Number(form.pointY)}
              min={0}
              max={100}
              onChange={(v) => set('pointY', v)}
            />
          </>
        )}
      </>
    );
  if (contract === 'chat-v1')
    return (
      <>
        {capabilities?.imageInput === true && (
          <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
        )}
        <div className="chat-history">
          {chatMessages.length === 0 ? (
            <span>No messages yet.</span>
          ) : (
            chatMessages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`chat-${message.role}`}>
                <small>{message.role === 'user' ? 'You' : 'Model'}</small>
                <p>{message.content}</p>
              </article>
            ))
          )}
        </div>
        <div className="chat-field-heading">
          <strong>Message</strong>
          <Button size="sm" disabled={chatMessages.length === 0} onClick={clearChat}>
            Clear conversation
          </Button>
        </div>
        <Field label="">
          <textarea
            value={String(form.prompt)}
            maxLength={32768}
            onChange={(e) => set('prompt', e.target.value)}
            placeholder="Ask this model anything…"
          />
        </Field>
      </>
    );
  if (contract === 'embeddings-v1')
    return (
      <>
        <Field label="Texts · one per line">
          <textarea
            value={String(form.text)}
            onChange={(e) => set('text', e.target.value)}
            placeholder={'First text\nSecond text'}
          />
        </Field>
        <Check
          label="Normalize vectors"
          checked={Boolean(form.normalize)}
          onChange={(value) => set('normalize', value)}
        />
      </>
    );
  if (contract === 'reranking-v1')
    return (
      <>
        <Field label="Query">
          <input value={String(form.query)} onChange={(e) => set('query', e.target.value)} />
        </Field>
        <Field label="Documents · one per line">
          <textarea
            value={String(form.documents)}
            onChange={(e) => set('documents', e.target.value)}
          />
        </Field>
        {taskProfile?.parameters.instruction?.handling !== 'unsupported' && (
          <Field label="Instruction">
            <input
              value={String(form.instruction)}
              placeholder="Rank by relevance"
              onChange={(e) => set('instruction', e.target.value)}
            />
          </Field>
        )}
      </>
    );
  if (contract === 'speech-synthesis-v1')
    return (
      <>
        <Field label="Text">
          <textarea
            maxLength={taskProfile?.limits.textCharacters ?? 350}
            value={String(form.text)}
            onChange={(e) => set('text', e.target.value)}
          />
        </Field>
        {taskProfile?.features.referenceAudio === 'supported' && (
          <>
            <Field label="Reference voice · 0.4–10 seconds of WAV audio">
              <FileField accept="audio/wav,audio/x-wav" file={file} onChange={setFile} />
            </Field>
          </>
        )}
        {taskProfile?.parameters.language?.valueSchema && (
          <Field label="Language">
            <select
              value={String(form.language)}
              disabled={taskProfile.parameters.language.handling === 'fixed'}
              onChange={(e) => set('language', e.target.value)}
            >
              {choicesFor('language', ['en-us', 'en-gb']).map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </Field>
        )}
        <div className="field-row">
          <Field label="Voice">
            <select
              value={String(form.voice)}
              disabled={taskProfile?.parameters.voice?.handling === 'fixed'}
              onChange={(e) => {
                setForm({
                  ...form,
                  voice: e.target.value,
                  language: e.target.value.startsWith('b') ? 'en-gb' : 'en-us',
                });
              }}
            >
              {choicesFor(
                'voice',
                taskProfile?.features.referenceAudio === 'supported'
                  ? ['reference']
                  : [
                      'af_heart',
                      'af_bella',
                      'af_nicole',
                      'af_sarah',
                      'am_adam',
                      'am_michael',
                      'bf_emma',
                      'bf_isabella',
                      'bm_george',
                      'bm_lewis',
                    ],
              ).map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </Field>
          {taskProfile?.parameters.speed?.handling !== 'fixed' && (
            <NumberField
              label="Speed"
              value={Number(form.speed)}
              min={0.5}
              max={2}
              step={0.1}
              onChange={(v) => set('speed', v)}
            />
          )}
        </div>
      </>
    );
  if (contract === 'transcription-v1')
    return (
      <>
        <FileField accept="audio/wav,audio/x-wav" file={file} onChange={setFile} />
        <div className="field-row">
          <Field label="Language">
            <input
              value={String(form.language === 'en-us' ? 'auto' : form.language)}
              onChange={(e) => set('language', e.target.value)}
            />
          </Field>
          <Check
            label="Include timestamps"
            checked={Boolean(form.timestamps)}
            onChange={(v) => set('timestamps', v)}
          />
        </div>
      </>
    );
  if (
    contract === 'text-classification-v1' ||
    contract === 'token-classification-v1' ||
    contract === 'fill-mask-v1' ||
    contract === 'translation-v1' ||
    contract === 'summarization-v1'
  )
    return (
      <>
        <Field label={contract === 'fill-mask-v1' ? 'Text with one mask token' : 'Text'}>
          <textarea
            maxLength={32768}
            value={String(form.text)}
            onChange={(e) => set('text', e.target.value)}
          />
        </Field>
        {(contract === 'text-classification-v1' || contract === 'fill-mask-v1') && (
          <NumberField
            label="Results"
            value={Number(form.topK)}
            min={1}
            max={20}
            onChange={(v) => set('topK', v)}
          />
        )}
        {(contract === 'translation-v1' || contract === 'summarization-v1') && (
          <NumberField
            label="Maximum output tokens"
            value={Number(form.maxTokens)}
            min={1}
            max={512}
            onChange={(v) => set('maxTokens', v)}
          />
        )}
      </>
    );
  if (contract === 'question-answering-v1')
    return (
      <>
        <Field label="Question">
          <input value={String(form.question)} onChange={(e) => set('question', e.target.value)} />
        </Field>
        <Field label="Context">
          <textarea
            maxLength={65536}
            value={String(form.context)}
            onChange={(e) => set('context', e.target.value)}
          />
        </Field>
        <NumberField
          label="Maximum answer tokens"
          value={Number(form.maxAnswerLength)}
          min={1}
          max={128}
          onChange={(v) => set('maxAnswerLength', v)}
        />
      </>
    );
  if (
    contract === 'image-classification-v1' ||
    contract === 'audio-classification-v1' ||
    contract === 'object-detection-v1' ||
    contract === 'depth-estimation-v1' ||
    contract === 'zero-shot-image-classification-v1'
  )
    return (
      <>
        <FileField
          accept={
            contract === 'audio-classification-v1'
              ? 'audio/wav,audio/x-wav'
              : 'image/png,image/jpeg,image/webp'
          }
          file={file}
          onChange={setFile}
        />
        {(contract === 'image-classification-v1' || contract === 'audio-classification-v1') && (
          <NumberField
            label="Results"
            value={Number(form.topK)}
            min={1}
            max={20}
            onChange={(v) => set('topK', v)}
          />
        )}
        {contract === 'object-detection-v1' && (
          <NumberField
            label="Confidence threshold"
            value={Number(form.threshold)}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => set('threshold', v)}
          />
        )}
        {contract === 'zero-shot-image-classification-v1' && (
          <Field label="Candidate labels · one per line">
            <textarea value={String(form.labels)} onChange={(e) => set('labels', e.target.value)} />
          </Field>
        )}
      </>
    );
  if (contract === 'image-editing-v1' || contract === 'image-layers-v1') {
    const layered = contract === 'image-layers-v1';
    const maximum = layered ? 1 : Math.min(3, taskProfile?.limits.maxReferenceImages ?? 1);
    const minimum = taskProfile?.limits.minReferenceImages ?? 1;
    return (
      <>
        <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
        {maximum > 1 && (
          <p className="inference-help">
            The input above is image 1. Add references in the order used by your instruction.{' '}
            {minimum > 1
              ? `${minimum} images are required.`
              : 'Additional references are optional.'}
          </p>
        )}
        {Array.from(
          { length: Math.min(maximum - 1, 1 + referenceFiles.filter(Boolean).length) },
          (_, index) => (
            <Field
              key={index}
              label={`Reference image ${index + 2}${index + 2 > minimum ? ' · optional' : ''}`}
            >
              <FileField
                accept="image/png,image/jpeg,image/webp"
                file={referenceFiles[index]}
                onChange={(next) => {
                  const files = [...referenceFiles];
                  files[index] = next;
                  setReferenceFiles(files);
                }}
              />
              {referenceFiles[index] && (
                <button
                  type="button"
                  onClick={() => setReferenceFiles(referenceFiles.filter((_, i) => i !== index))}
                >
                  Remove reference
                </button>
              )}
            </Field>
          ),
        )}
        <Field label={layered ? 'Describe the layers to separate' : 'Editing instruction'}>
          <textarea
            maxLength={4096}
            value={String(form.prompt)}
            onChange={(e) => set('prompt', e.target.value)}
          />
        </Field>
        {layered ? (
          <>
            <NumberField
              label="Layers"
              value={Number(form.layers ?? 4)}
              min={1}
              max={4}
              onChange={(v) => set('layers', v)}
            />
            <Field label="Resolution">
              <select
                value={Number(form.resolution ?? 640)}
                onChange={(e) => set('resolution', Number(e.target.value))}
              >
                <option value={640}>640 pixels</option>
                <option value={1024}>1024 pixels</option>
              </select>
            </Field>
            <p className="inference-help">
              Returns separate transparent PNG layers. Use an image with an aspect ratio between 1:2
              and 2:1.
            </p>
          </>
        ) : (
          <div className="field-row">
            <NumberField
              label="Width"
              value={Number(form.width)}
              min={64}
              max={1024}
              step={16}
              onChange={(v) => set('width', v)}
            />
            <NumberField
              label="Height"
              value={Number(form.height)}
              min={64}
              max={1024}
              step={16}
              onChange={(v) => set('height', v)}
            />
          </div>
        )}
        <NumberField
          label="Steps"
          value={Number(form.steps)}
          min={1}
          max={50}
          onChange={(v) => set('steps', v)}
        />
        <NumberField
          label="Guidance"
          value={Number(form.guidanceScale)}
          min={0}
          max={10}
          step={0.1}
          onChange={(v) => set('guidanceScale', v)}
        />
        <NumberField
          label="Seed"
          value={Number(form.seed)}
          min={0}
          max={2 ** 32 - 1}
          onChange={(v) => set('seed', v)}
        />
      </>
    );
  }
  const music = contract === 'text-to-music-v1';
  const profileParameters = generationProfile?.parameters;
  const profileDirty = generationProfile
    ? isGenerationProfileDirty(
        {
          width: Number(form.width),
          height: Number(form.height),
          steps: Number(form.steps),
          guidanceScale: Number(form.guidanceScale),
        },
        generationProfile,
      )
    : false;
  return (
    <>
      {(needsFile(contract) ||
        (contract === 'text-to-image-v1' && capabilities?.imageInput === true)) && (
        <FileField accept="image/png,image/jpeg,image/webp" file={file} onChange={setFile} />
      )}
      <Field label={music ? 'Music description' : 'Prompt'}>
        <textarea
          maxLength={music ? 2000 : 4096}
          value={String(form.prompt)}
          onChange={(e) => set('prompt', e.target.value)}
          placeholder={
            music
              ? 'Describe the mood, instruments, and structure…'
              : 'Describe the result in detail…'
          }
        />
      </Field>
      {music && capabilities?.lyrics === true && (
        <Field label="Lyrics · optional">
          <textarea
            maxLength={12000}
            value={String(form.lyrics)}
            onChange={(e) => set('lyrics', e.target.value)}
            placeholder="[Instrumental]"
          />
        </Field>
      )}
      <div className="field-row canvas-controls">
        {music ? (
          <>
            <NumberField
              label="Duration · seconds"
              value={Number(form.durationSeconds)}
              min={1}
              max={180}
              onChange={(v) => set('durationSeconds', v)}
            />
            {capabilities?.deterministicSeed === true && (
              <NumberField
                label="Seed"
                value={Number(form.seed)}
                min={-1}
                max={2 ** 32 - 1}
                onChange={(v) => set('seed', v)}
              />
            )}
          </>
        ) : (
          <>
            <NumberField
              label="Width"
              value={Number(form.width)}
              min={profileParameters?.width.minimum ?? 64}
              max={profileParameters?.width.maximum ?? 2048}
              step={profileParameters?.width.step ?? 8}
              disabled={profileParameters?.width.editable === false}
              onChange={(v) => set('width', v)}
            />
            <NumberField
              label="Height"
              value={Number(form.height)}
              min={profileParameters?.height.minimum ?? 64}
              max={profileParameters?.height.maximum ?? 2048}
              step={profileParameters?.height.step ?? 8}
              disabled={profileParameters?.height.editable === false}
              onChange={(v) => set('height', v)}
            />
          </>
        )}
      </div>
      {!music && (
        <fieldset className="inference-settings">
          <legend>Inference settings</legend>
          {generationProfile ? (
            <div
              className={`generation-profile ${generationProfile.recommended ? 'is-recommended' : ''}`}
            >
              <div>
                <strong>
                  {generationProfile.recommended
                    ? 'Recommended settings for this model'
                    : 'Runtime starting settings'}
                </strong>
                <span>{generationProfileSource(generationProfile)}</span>
              </div>
              {profileDirty && (
                <Button
                  size="sm"
                  onClick={() => setForm({ ...form, ...generationDefaults(generationProfile) })}
                >
                  Restore profile
                </Button>
              )}
            </div>
          ) : (
            <p className="generation-profile-missing">
              This runtime did not publish a model generation profile. Values remain editable, but
              Oryn cannot claim a model-specific recommendation.
            </p>
          )}
          <div className="field-row inference-controls">
            <NumberField
              label="Inference steps"
              value={Number(form.steps)}
              min={profileParameters?.steps.minimum ?? 1}
              max={profileParameters?.steps.maximum ?? 100}
              step={profileParameters?.steps.step ?? 1}
              disabled={profileParameters?.steps.editable === false}
              onChange={(v) => set('steps', v)}
            />
            {profileParameters?.guidanceScale.supported === false ? (
              <div className="guidance-disabled" aria-label="Guidance scale disabled">
                <span>Guidance scale</span>
                <strong>Not used by this model profile</strong>
              </div>
            ) : (
              <NumberField
                label="Guidance scale"
                value={Number(form.guidanceScale)}
                min={profileParameters?.guidanceScale.minimum ?? 0}
                max={profileParameters?.guidanceScale.maximum ?? 30}
                step={profileParameters?.guidanceScale.step ?? 0.1}
                disabled={profileParameters?.guidanceScale.editable === false}
                onChange={(v) => set('guidanceScale', v)}
              />
            )}
          </div>
          <p className="inference-help">
            {profileDirty
              ? 'You changed the runtime profile for this generation.'
              : 'These controls and limits are supplied by the selected runtime adapter.'}
          </p>
        </fieldset>
      )}
      {contract.includes('video') && (
        <div className="field-row">
          <NumberField
            label="Frames"
            value={Number(form.frames)}
            min={1}
            max={120}
            onChange={(v) => set('frames', v)}
          />
          <NumberField
            label="FPS"
            value={Number(form.fps)}
            min={1}
            max={60}
            onChange={(v) => set('fps', v)}
          />
        </div>
      )}
      {contract === 'image-to-image-v1' && (
        <NumberField
          label="Transformation strength"
          value={Number(form.strength)}
          min={0.05}
          max={1}
          step={0.05}
          onChange={(v) => set('strength', v)}
        />
      )}
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  disabled = false,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </Field>
  );
}
function Check({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}
function FileField({
  accept,
  file,
  onChange,
}: {
  accept: string;
  file?: File;
  onChange: (file?: File) => void;
}) {
  return (
    <label className="file-field">
      <Icon name="download" />
      <span>
        <strong>{file?.name ?? 'Choose an input file'}</strong>
        <small>
          {file
            ? formatBytes(file.size)
            : accept.includes('audio')
              ? 'PCM WAV'
              : 'PNG, JPEG, or WebP'}
        </small>
      </span>
      <input type="file" accept={accept} onChange={(e) => onChange(e.target.files?.[0])} />
    </label>
  );
}

function Output({
  contract,
  stream,
  result,
  artifact,
  artifactURL,
}: {
  contract: ModelWorkspaceTaskContract;
  stream: string;
  result: unknown;
  artifact?: Artifact;
  artifactURL?: string;
}) {
  const multi = result as { kind?: string; outputs?: unknown[] } | undefined;
  if (multi?.kind === 'multi' && Array.isArray(multi.outputs))
    return (
      <div className="multi-result">
        {multi.outputs.map((output, index) => (
          <OutputPart key={index} contract={contract} output={output} index={index} />
        ))}
      </div>
    );
  if (artifactURL && artifact) {
    if (artifact.mimeType.startsWith('image/'))
      return <img className="media-result" src={artifactURL} alt="Generated result" />;
    if (artifact.mimeType.startsWith('audio/'))
      return <audio className="audio-result" src={artifactURL} controls />;
    if (artifact.mimeType.startsWith('video/'))
      return <video className="media-result" src={artifactURL} controls />;
  }
  const value = result as
    | {
        text?: string;
        embeddings?: number[][];
        dimensions?: number;
        documentVectors?: number[][];
        queries?: Array<{ vectors: number[][]; score: number }>;
        results?: Array<{ index: number; score: number }>;
      }
    | undefined;
  if (contract === 'chat-v1' && (stream || value?.text))
    return <div className="text-result">{stream || value?.text}</div>;
  if (contract === 'transcription-v1' && value?.text)
    return <div className="text-result">{value.text}</div>;
  if (contract === 'embeddings-v1' && value?.embeddings)
    return (
      <div className="structured-result">
        <strong>
          {value.embeddings.length} vectors · {value.dimensions} dimensions
        </strong>
        <pre>
          {JSON.stringify(
            value.embeddings.map((row) => row.slice(0, 8)),
            null,
            2,
          )}
        </pre>
      </div>
    );
  if (contract === 'reranking-v1' && value?.results)
    return (
      <ol className="ranking">
        {value.results.map((item) => (
          <li key={item.index}>
            <span>Document {item.index + 1}</span>
            <strong>{item.score.toFixed(4)}</strong>
          </li>
        ))}
      </ol>
    );
  if (contract === 'visual-document-retrieval-v1' && value?.queries && value.documentVectors)
    return (
      <div className="structured-result">
        <strong>Visual relevance to this page</strong>
        <ol className="ranking">
          {value.queries.map((item, index) => (
            <li key={index}>
              <span>Query {index + 1}</span>
              <strong>{item.score.toFixed(4)}</strong>
            </li>
          ))}
        </ol>
        <p>Higher scores indicate a closer match. Compare queries made with this model and page.</p>
        <details>
          <summary>Embedding details</summary>
          <p>
            {value.documentVectors.length} page vectors · {value.dimensions} dimensions
          </p>
          <p>
            {value.queries.map((q, i) => `Query ${i + 1}: ${q.vectors.length} vectors`).join(' · ')}
          </p>
        </details>
      </div>
    );
  if (result !== undefined)
    return <pre className="structured-result">{JSON.stringify(result, null, 2)}</pre>;
  return (
    <div className="output-placeholder">
      <Icon name="sparkles" />
      <strong>Your local result will appear here</strong>
      <span>You can switch tabs while generation runs.</span>
    </div>
  );
}

function OutputPart({
  contract,
  output,
  index,
}: {
  contract: ModelWorkspaceTaskContract;
  output: unknown;
  index: number;
}) {
  const node = output as { kind?: string; value?: unknown; artifact?: Artifact };
  const artifact = node.kind === 'artifact' ? node.artifact : undefined;
  const [url, setURL] = useState<string>();
  const [failure, setFailure] = useState<string>();
  useEffect(() => {
    let active = true;
    setURL(undefined);
    setFailure(undefined);
    if (artifact)
      void window.orynWorkspace
        .getArtifactUrl(artifact.id)
        .then((response) => {
          if (!active) return;
          if (response.ok) setURL(response.value);
          else setFailure(response.error.message);
        })
        .catch(() => {
          if (active) setFailure('The output preview is unavailable.');
        });
    return () => {
      active = false;
    };
  }, [artifact?.id]);
  return (
    <section className="result-part">
      <p>{artifact?.name ?? `Output ${index + 1}`}</p>
      <Output
        contract={contract}
        stream=""
        result={node.kind === 'structured' ? node.value : artifact ? undefined : output}
        artifact={artifact}
        artifactURL={url}
      />
      {failure && <p role="alert">{failure}</p>}
      {artifact && (
        <div className="result-actions">
          <Button
            onClick={() => void window.orynWorkspace.saveArtifactAs(artifact.id, artifact.name)}
          >
            Save As…
          </Button>
          <span>
            {artifact.mimeType} · {formatBytes(artifact.sizeBytes)}
          </span>
        </div>
      )}
    </section>
  );
}

async function upload(file?: File): Promise<string> {
  if (!file) throw new Error('Choose an input file first.');
  const begin = await window.orynWorkspace.beginArtifactCreate({
    mimeType: file.type || 'application/octet-stream',
    sizeBytes: file.size,
    name: file.name,
  });
  if (!begin.ok) throw new Error(begin.error.message);
  try {
    for (let offset = 0; offset < file.size; offset += 1024 * 1024) {
      const chunk = await file.slice(offset, offset + 1024 * 1024).arrayBuffer();
      const appended = await window.orynWorkspace.appendArtifactChunk(begin.value.uploadId, chunk);
      if (!appended.ok) throw new Error(appended.error.message);
    }
    const complete = await window.orynWorkspace.finishArtifactCreate(begin.value.uploadId);
    if (!complete.ok) throw new Error(complete.error.message);
    const artifact = complete.value as { id?: string };
    if (!artifact.id) throw new Error('Oryn did not publish the input file.');
    return artifact.id;
  } catch (error) {
    await window.orynWorkspace.abortArtifactCreate(begin.value.uploadId);
    throw error;
  }
}

async function exportJSON(value: unknown, name: string): Promise<void> {
  const file = new File([`${JSON.stringify(value, null, 2)}\n`], name, {
    type: 'application/json',
  });
  const artifactId = await upload(file);
  try {
    const saved = await window.orynWorkspace.saveArtifactAs(artifactId, file.name);
    if (!saved.ok) throw new Error(saved.error.message);
  } finally {
    await window.orynWorkspace.releaseArtifact(artifactId);
  }
}

function buildInput(
  contract: ModelWorkspaceTaskContract,
  form: FormValue,
  artifactId?: string,
  capabilities?: Record<string, unknown>,
  chatMessages: ChatMessage[] = [],
  secondaryArtifactId?: string,
  referenceImageArtifactIds: string[] = [],
): unknown {
  switch (contract) {
    case 'tabular-classification-v1':
    case 'tabular-regression-v1':
      return tabularInput(
        String(form.featureColumns),
        String(form.trainingRows),
        String(form.trainingTargets),
        String(form.queryRows),
        contract === 'tabular-classification-v1',
      );
    case 'audio-transformation-v1': {
      const operation = String(form.audioOperation);
      return {
        operation,
        ...(operation === 'decode'
          ? { codes: numericMatrix(String(form.sourceGrid), true) }
          : operation === 'vocode'
            ? { spectrogram: numericMatrix(String(form.sourceGrid)) }
            : { audioArtifactId: artifactId }),
        ...(operation === 'voice-conversion'
          ? { referenceAudioArtifactId: secondaryArtifactId }
          : {}),
      };
    }
    case 'multimodal-generation-v1':
      return {
        inputMode: String(form.omniSource),
        instruction: String(form.instruction),
        outputMode: String(form.outputMode),
        maxTokens: Number(form.maxTokens),
        ...(form.omniSource === 'audio'
          ? { audioArtifactId: artifactId }
          : { imageArtifactId: artifactId }),
        ...(form.omniSource === 'image-and-audio' ? { audioArtifactId: secondaryArtifactId } : {}),
      };
    case 'video-editing-v1':
      return {
        operation: String(form.videoOperation),
        videoArtifactId: artifactId,
        startSeconds: Number(form.startSeconds),
        durationSeconds: Number(form.durationSeconds),
        ...(form.videoOperation === 'edit' ? { prompt: String(form.prompt) } : {}),
        ...(form.videoOperation === 'edit' && secondaryArtifactId
          ? { maskVideoArtifactId: secondaryArtifactId }
          : {}),
      };
    case 'pose-estimation-v1':
      return { imageArtifactId: artifactId, boxes: poseBoxes(String(form.personBoxes)) };
    case 'unconditional-image-generation-v1':
      return {
        width: Number(form.width),
        height: Number(form.height),
        steps: Number(form.steps),
        seed: Number(form.seed),
      };
    case 'audio-question-answering-v1':
      return {
        audioArtifactId: artifactId,
        instruction: String(form.instruction),
        maxTokens: Number(form.maxTokens),
      };
    case 'time-series-forecasting-v1':
      return forecastInput(String(form.context), Number(form.horizon), String(form.quantiles));
    case 'image-text-to-video-v1':
      return {
        imageArtifactId: artifactId,
        prompt: String(form.prompt),
        width: Number(form.width),
        height: Number(form.height),
        frames: Number(form.frames),
        fps: Number(form.fps),
        steps: Number(form.steps),
        guidanceScale: Number(form.guidanceScale),
        seed: Number(form.seed),
      };
    case 'visual-document-retrieval-v1':
      return { imageArtifactId: artifactId, queries: lines(String(form.question)) };
    case 'table-question-answering-v1': {
      const rows = String(form.table)
        .trim()
        .split('\n')
        .map((row) => row.split('\t'));
      return { question: String(form.question), columns: rows[0], rows: rows.slice(1) };
    }
    case 'document-question-answering-v1': {
      const rows = String(form.ocr)
        .trim()
        .split('\n')
        .map((row) => row.split('\t'));
      return {
        imageArtifactId: artifactId,
        question: String(form.question),
        words: rows.map((row) => row[0]),
        boxes: rows.map((row) => row.slice(1).map((value) => Number(value) / 100)),
      };
    }
    case 'keypoint-detection-v1':
      return {
        imageArtifactId: artifactId,
        threshold: Number(form.keypointThreshold),
        maxKeypoints: Number(form.maxKeypoints),
      };
    case 'video-classification-v1':
    case 'video-question-answering-v1':
      return {
        videoArtifactId: artifactId,
        startSeconds: Number(form.startSeconds),
        durationSeconds: Number(form.durationSeconds),
        ...(contract === 'video-classification-v1'
          ? { topK: Number(form.topK) }
          : { question: String(form.question), maxTokens: Number(form.maxTokens) }),
      };
    case 'voice-activity-detection-v1':
      return {
        audioArtifactId: artifactId,
        threshold: Number(form.threshold),
        minSpeechSeconds: Number(form.minSpeechSeconds),
        minSilenceSeconds: Number(form.minSilenceSeconds),
      };
    case 'chat-v1':
      return {
        ...(artifactId ? { imageArtifactId: artifactId } : {}),
        messages: [...chatMessages, { role: 'user', content: String(form.prompt) }],
        maxTokens: 256,
        temperature: 0.7,
        topP: 0.9,
      };
    case 'text-to-music-v1':
      return {
        prompt: String(form.prompt),
        ...(capabilities?.lyrics === true && String(form.lyrics).trim()
          ? { lyrics: String(form.lyrics) }
          : {}),
        durationSeconds: Number(form.durationSeconds),
        ...(capabilities?.deterministicSeed === true ? { seed: Number(form.seed) } : {}),
      };
    case 'speech-synthesis-v1':
      return {
        text: String(form.text),
        voice: String(form.voice),
        language: String(form.language),
        speed: Number(form.speed),
        ...(form.voice === 'reference' ? { audioArtifactId: artifactId } : {}),
      };
    case 'transcription-v1':
      return {
        audioArtifactId: artifactId,
        language: String(form.language === 'en-us' ? 'auto' : form.language),
        timestamps: Boolean(form.timestamps),
      };
    case 'embeddings-v1':
      return { texts: lines(String(form.text)), normalize: Boolean(form.normalize) };
    case 'reranking-v1': {
      const documents = lines(String(form.documents));
      return {
        query: String(form.query),
        documents,
        ...(String(form.instruction).trim()
          ? { instruction: String(form.instruction).trim() }
          : {}),
        topK: documents.length,
      };
    }
    case 'zero-shot-classification-v1':
      return {
        text: String(form.text),
        labels: lines(String(form.labels)),
        multiLabel: Boolean(form.multiLabel),
      };
    case 'image-embeddings-v1':
      return { imageArtifactId: artifactId, normalize: Boolean(form.normalize) };
    case 'image-segmentation-v1':
      return { imageArtifactId: artifactId };
    case 'mask-generation-v1':
      return {
        imageArtifactId: artifactId,
        points: [{ x: Number(form.pointX) / 100, y: Number(form.pointY) / 100, label: 1 }],
      };
    case 'zero-shot-object-detection-v1':
      return {
        imageArtifactId: artifactId,
        labels: lines(String(form.labels)),
        threshold: Number(form.threshold),
      };
    case 'image-to-text-v1':
      return { imageArtifactId: artifactId, maxTokens: Number(form.maxTokens) };
    case 'visual-question-answering-v1':
      return {
        imageArtifactId: artifactId,
        question: String(form.question),
        maxTokens: Number(form.maxTokens),
      };
    case 'text-classification-v1':
    case 'fill-mask-v1':
      return { text: String(form.text), topK: Number(form.topK) };
    case 'token-classification-v1':
      return { text: String(form.text) };
    case 'question-answering-v1':
      return {
        question: String(form.question),
        context: String(form.context),
        maxAnswerLength: Number(form.maxAnswerLength),
      };
    case 'translation-v1':
    case 'summarization-v1':
      return { text: String(form.text), maxTokens: Number(form.maxTokens) };
    case 'image-classification-v1':
      return { imageArtifactId: artifactId, topK: Number(form.topK) };
    case 'audio-classification-v1':
      return { audioArtifactId: artifactId, topK: Number(form.topK) };
    case 'object-detection-v1':
      return { imageArtifactId: artifactId, threshold: Number(form.threshold) };
    case 'depth-estimation-v1':
      return { imageArtifactId: artifactId };
    case 'image-editing-v1':
    case 'image-layers-v1':
      return {
        imageArtifactId: artifactId,
        ...(contract === 'image-editing-v1' && referenceImageArtifactIds.length
          ? { referenceImageArtifactIds }
          : {}),
        prompt: String(form.prompt),
        steps: Number(form.steps),
        guidanceScale: Number(form.guidanceScale),
        seed: Number(form.seed),
        ...(contract === 'image-editing-v1'
          ? { width: Number(form.width), height: Number(form.height) }
          : { resolution: Number(form.resolution ?? 640), layers: Number(form.layers ?? 4) }),
      };
    case 'zero-shot-image-classification-v1':
      return { imageArtifactId: artifactId, labels: lines(String(form.labels)) };
    default:
      return {
        prompt: String(form.prompt),
        negativePrompt: '',
        width: Number(form.width),
        height: Number(form.height),
        steps: Number(form.steps),
        guidanceScale: Number(form.guidanceScale),
        seed: Number(form.seed),
        ...(contract.includes('video')
          ? { frames: Number(form.frames), fps: Number(form.fps) }
          : {}),
        ...(contract.startsWith('image-to-') || artifactId ? { imageArtifactId: artifactId } : {}),
        ...(contract === 'image-to-image-v1' ||
        (contract === 'text-to-image-v1' && artifactId && capabilities?.instructionEditing !== true)
          ? { strength: Number(form.strength ?? 0.8) }
          : {}),
      };
  }
}

function needsFile(contract: ModelWorkspaceTaskContract) {
  return (
    [
      'visual-document-retrieval-v1',
      'audio-question-answering-v1',
      'audio-transformation-v1',
      'multimodal-generation-v1',
      'video-editing-v1',
      'pose-estimation-v1',
      'image-text-to-video-v1',
      'document-question-answering-v1',
      'keypoint-detection-v1',
      'video-classification-v1',
      'video-question-answering-v1',
      'voice-activity-detection-v1',
      'image-editing-v1',
      'image-layers-v1',
      'image-embeddings-v1',
      'image-segmentation-v1',
      'mask-generation-v1',
      'zero-shot-object-detection-v1',
      'visual-question-answering-v1',
    ].includes(contract) ||
    contract === 'transcription-v1' ||
    contract === 'audio-classification-v1' ||
    contract === 'image-classification-v1' ||
    contract === 'object-detection-v1' ||
    contract === 'depth-estimation-v1' ||
    contract === 'zero-shot-image-classification-v1' ||
    contract.startsWith('image-to-')
  );
}
function lines(value: string) {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter(Boolean);
}
function surfaceTitle(contract: ModelWorkspaceTaskContract) {
  return (
    {
      'unconditional-image-generation-v1': 'Sample an image',
      'audio-question-answering-v1': 'Ask about audio',
      'pose-estimation-v1': 'Estimate labeled body joints',
      'audio-transformation-v1': 'Transform audio',
      'multimodal-generation-v1': 'Generate from image or audio',
      'image-text-to-video-v1': 'Animate an image with an instruction',
      'video-editing-v1': 'Edit or restore video',
      'tabular-classification-v1': 'Classify table rows',
      'tabular-regression-v1': 'Predict values from a table',
      'time-series-forecasting-v1': 'Forecast a time series',
      'visual-document-retrieval-v1': 'Search a document page',
      'table-question-answering-v1': 'Ask about a table',
      'document-question-answering-v1': 'Ask about a document page',
      'keypoint-detection-v1': 'Find image interest points',
      'video-classification-v1': 'Classify a video clip',
      'video-question-answering-v1': 'Ask about a video clip',
      'voice-activity-detection-v1': 'Find speech intervals',
      'zero-shot-classification-v1': 'Classify text with your labels',
      'image-embeddings-v1': 'Create an image embedding',
      'image-segmentation-v1': 'Label image regions',
      'mask-generation-v1': 'Create an object mask',
      'zero-shot-object-detection-v1': 'Find objects by name',
      'image-editing-v1': 'Edit an image',
      'image-layers-v1': 'Separate image layers',
      'image-to-text-v1': 'Read or describe an image',
      'visual-question-answering-v1': 'Ask about an image',
      'chat-v1': 'Chat',
      'text-to-image-v1': 'Generate an image',
      'image-to-image-v1': 'Transform an image',
      'text-to-music-v1': 'Generate music',
      'speech-synthesis-v1': 'Synthesize speech',
      'transcription-v1': 'Transcribe audio',
      'media-audio-extract-v1': 'Extract video audio',
      'audio-source-separation-v1': 'Separate vocals',
      'forced-alignment-v1': 'Align transcript timing',
      'subtitle-export-v1': 'Export subtitles',
      'embeddings-v1': 'Create embeddings',
      'reranking-v1': 'Rerank documents',
      'text-to-video-v1': 'Generate a video',
      'image-to-video-v1': 'Animate an image',
      'text-classification-v1': 'Classify text',
      'token-classification-v1': 'Label text tokens',
      'question-answering-v1': 'Answer from context',
      'fill-mask-v1': 'Complete a masked token',
      'translation-v1': 'Translate text',
      'summarization-v1': 'Summarize text',
      'image-classification-v1': 'Classify an image',
      'audio-classification-v1': 'Classify audio',
      'object-detection-v1': 'Detect objects',
      'depth-estimation-v1': 'Estimate image depth',
      'zero-shot-image-classification-v1': 'Classify an image with your labels',
    } as const
  )[contract];
}
function surfaceDescription(contract: ModelWorkspaceTaskContract) {
  return (
    {
      'unconditional-image-generation-v1':
        'Draw a native image sample using geometry and seed, with no text prompt.',
      'audio-question-answering-v1': 'Answer an instruction using the selected local audio.',
      'pose-estimation-v1': 'Locate named body joints inside the person boxes you provide.',
      'audio-transformation-v1': 'Run an explicitly supported audio transformation.',
      'multimodal-generation-v1': 'Use a supported image or audio input and output combination.',
      'image-text-to-video-v1': 'Condition a short video on both the image and your instruction.',
      'video-editing-v1': 'Transform a bounded source clip while preserving temporal order.',
      'tabular-classification-v1': 'Use labeled training rows to classify separate query rows.',
      'tabular-regression-v1': 'Use training examples to predict separate numeric targets.',
      'time-series-forecasting-v1': 'Predict future values using historical context only.',
      'visual-document-retrieval-v1':
        'Compare search queries with a page image using visual relevance scores.',
      'table-question-answering-v1': 'Answer questions using a bounded table of text cells.',
      'document-question-answering-v1':
        'Ground an answer in supplied page text and word positions.',
      'keypoint-detection-v1':
        'Return interest points and confidence in original image coordinates.',
      'video-classification-v1': 'Classify a bounded clip using its ordered frame sequence.',
      'video-question-answering-v1': 'Answer a question using a bounded temporal clip.',
      'voice-activity-detection-v1':
        'Return speech intervals in seconds relative to the input audio.',
      'zero-shot-classification-v1': 'Score text against up to 32 candidate labels.',
      'image-embeddings-v1': 'Encode an image for similarity and visual search.',
      'image-segmentation-v1': 'Create a PNG label map; each pixel records the model’s class ID.',
      'mask-generation-v1': 'Create a black-and-white mask for the object at your chosen point.',
      'zero-shot-object-detection-v1':
        'Find named objects and return confidence scores and image coordinates.',
      'image-editing-v1':
        'Edit images using an instruction and the references supported by this model.',
      'image-layers-v1': 'Separate an image into transparent PNG layers.',
      'image-to-text-v1': 'Read text or generate a caption, according to the selected model.',
      'visual-question-answering-v1': 'Answer a question using the selected image.',
      'chat-v1': 'A streaming conversation with the exact installed model.',
      'text-to-image-v1': 'Create a local image from a bounded prompt and options.',
      'image-to-image-v1': 'Use a local image as the starting point.',
      'text-to-music-v1': 'Describe a song, add optional lyrics, and generate locally.',
      'speech-synthesis-v1': 'Convert bounded text into playable speech.',
      'transcription-v1': 'Upload PCM WAV audio and receive a transcript.',
      'media-audio-extract-v1': 'Extract a bounded audio track from a local video.',
      'audio-source-separation-v1': 'Isolate a bounded vocal stem from local audio.',
      'forced-alignment-v1': 'Align bounded transcript text to local audio.',
      'subtitle-export-v1': 'Validate and serialize timed cues as SubRip.',
      'embeddings-v1': 'Encode a bounded batch and inspect vector summaries.',
      'reranking-v1': 'Score documents against a query.',
      'text-to-video-v1': 'Create a short bounded local video.',
      'image-to-video-v1': 'Turn a local image into a short video.',
      'text-classification-v1': 'Score bounded text against the model’s labels.',
      'token-classification-v1': 'Assign the model’s labels to bounded text tokens.',
      'question-answering-v1': 'Extract a bounded answer from supplied context.',
      'fill-mask-v1': 'Predict one masked token using a sealed masked-language model.',
      'translation-v1': 'Translate bounded text with the installed seq2seq model.',
      'summarization-v1': 'Summarize bounded text with the installed seq2seq model.',
      'image-classification-v1': 'Score a local image against the model’s labels.',
      'audio-classification-v1': 'Score local PCM WAV audio against the model’s labels.',
      'object-detection-v1': 'Return bounded labels, confidence scores, and boxes.',
      'depth-estimation-v1': 'Return bounded summary statistics for the inferred depth map.',
      'zero-shot-image-classification-v1': 'Compare a local image with bounded candidate labels.',
    } as const
  )[contract];
}
function actionLabel(contract: ModelWorkspaceTaskContract) {
  return contract === 'chat-v1'
    ? 'Send message'
    : contract === 'transcription-v1'
      ? 'Transcribe locally'
      : contract === 'reranking-v1'
        ? 'Rank locally'
        : contract === 'embeddings-v1'
          ? 'Embed locally'
          : 'Generate locally';
}
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ModelWorkspace />
  </StrictMode>,
);
