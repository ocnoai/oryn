import { contextBridge, ipcRenderer, webUtils } from 'electron';

const unavailable = () =>
  Promise.resolve({
    ok: false,
    error: {
      code: 'RUNTIME_NOT_AVAILABLE',
      message: 'Local AI is unavailable in private windows.',
      details: { reason: 'PRIVATE_CONTEXT' },
    },
  });
const noop = () => Promise.resolve();
const subscribe = () => () => undefined;

// Capability detection still recognizes Oryn. This preload has no Local AI IPC authority.
contextBridge.exposeInMainWorld(
  'localAI',
  Object.freeze({
    sdkVersion: '0.1.0',
    getCapabilities: () =>
      Promise.resolve({
        runtimeAvailable: false,
        runtimeState: 'unavailable',
        sdkVersion: '0.1.0',
        unavailableReason: 'PRIVATE_CONTEXT',
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
      }),
    requestModel: unavailable,
    cancelModelRequest: noop,
    releaseModelSession: unavailable,
    onModelRequestProgress: subscribe,
    beginArtifactCreate: unavailable,
    appendArtifactChunk: unavailable,
    finishArtifactCreate: unavailable,
    abortArtifactCreate: noop,
    getArtifact: unavailable,
    getArtifactUrl: unavailable,
    saveArtifactAs: unavailable,
    releaseArtifact: unavailable,
    invokeJob: unavailable,
    cancelJob: unavailable,
    onJobEvent: subscribe,
  }),
);

window.addEventListener('drop', (event) => {
  if (!event.isTrusted || event.defaultPrevented || !process.isMainFrame) return;
  const paths = [...(event.dataTransfer?.files ?? [])]
    .filter((file) => /\.(html?|xhtml|shtml)$/iu.test(file.name))
    .map((file) => webUtils.getPathForFile(file))
    .filter(Boolean)
    .slice(0, 16);
  if (!paths.length) return;
  event.preventDefault();
  void ipcRenderer.invoke('browser:native-file-drop', paths);
});
