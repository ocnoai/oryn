import { contextBridge, ipcRenderer, webUtils } from 'electron';

import type { InjectedLocalAI, WebsiteBridgeResult } from '../shared/ipc';

const localAI: InjectedLocalAI = Object.freeze({
  sdkVersion: '0.1.0',
  getCapabilities: () =>
    ipcRenderer.invoke('local-ai:get-capabilities') as ReturnType<
      InjectedLocalAI['getCapabilities']
    >,
  requestModel: (operationId: string, request: unknown) =>
    ipcRenderer.invoke('local-ai:model-request', operationId, request) as Promise<
      WebsiteBridgeResult<unknown>
    >,
  cancelModelRequest: (operationId: string) =>
    ipcRenderer.invoke('local-ai:model-request-cancel', operationId) as Promise<void>,
  releaseModelSession: (sessionId: string) =>
    ipcRenderer.invoke('local-ai:model-session-release', sessionId) as Promise<
      WebsiteBridgeResult<void>
    >,
  onModelRequestProgress(operationId: string, listener: (progress: unknown) => void) {
    const handler = (
      _event: Electron.IpcRendererEvent,
      value: { operationId?: unknown; progress?: unknown },
    ) => {
      if (value?.operationId === operationId) listener(value.progress);
    };
    ipcRenderer.on('local-ai:model-request-progress', handler);
    return () => ipcRenderer.removeListener('local-ai:model-request-progress', handler);
  },
  beginArtifactCreate: (metadata: unknown) =>
    ipcRenderer.invoke('local-ai:artifact-create-begin', metadata) as Promise<
      WebsiteBridgeResult<{ uploadId: string }>
    >,
  appendArtifactChunk: (uploadId: string, chunk: ArrayBuffer) =>
    ipcRenderer.invoke('local-ai:artifact-create-chunk', uploadId, chunk) as Promise<
      WebsiteBridgeResult<void>
    >,
  finishArtifactCreate: (uploadId: string) =>
    ipcRenderer.invoke('local-ai:artifact-create-finish', uploadId) as Promise<
      WebsiteBridgeResult<unknown>
    >,
  abortArtifactCreate: (uploadId: string) =>
    ipcRenderer.invoke('local-ai:artifact-create-abort', uploadId) as Promise<void>,
  getArtifact: (artifactId: string) =>
    ipcRenderer.invoke('local-ai:artifact-get', artifactId) as Promise<
      WebsiteBridgeResult<unknown>
    >,
  getArtifactUrl: (artifactId: string) =>
    ipcRenderer.invoke('local-ai:artifact-get-url', artifactId) as Promise<
      WebsiteBridgeResult<string>
    >,
  saveArtifactAs: (artifactId: string, options?: { suggestedFilename?: string }) =>
    ipcRenderer.invoke('local-ai:artifact-save-as', artifactId, options) as Promise<
      WebsiteBridgeResult<void>
    >,
  releaseArtifact: (artifactId: string) =>
    ipcRenderer.invoke('local-ai:artifact-release', artifactId) as Promise<
      WebsiteBridgeResult<void>
    >,
  invokeJob: (request: unknown) =>
    ipcRenderer.invoke('local-ai:job-invoke', request) as Promise<WebsiteBridgeResult<unknown>>,
  cancelJob: (jobId: string) =>
    ipcRenderer.invoke('local-ai:job-cancel', jobId) as Promise<WebsiteBridgeResult<void>>,
  onJobEvent(listener: (event: unknown) => void) {
    const handler = (_event: Electron.IpcRendererEvent, value: unknown) => listener(value);
    ipcRenderer.on('local-ai:job-event', handler);
    return () => ipcRenderer.removeListener('local-ai:job-event', handler);
  },
});

contextBridge.exposeInMainWorld('localAI', localAI);

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
