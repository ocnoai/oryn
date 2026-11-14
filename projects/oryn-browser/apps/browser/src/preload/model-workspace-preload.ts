import { contextBridge, ipcRenderer } from 'electron';

import {
  IPC,
  type ModelWorkspaceAPI,
  type ModelWorkspaceState,
  type WebsiteBridgeResult,
} from '../shared/ipc';

const api: ModelWorkspaceAPI = Object.freeze({
  getState: () => ipcRenderer.invoke(IPC.workspaceGetState) as Promise<ModelWorkspaceState>,
  invoke: (input: unknown) =>
    ipcRenderer.invoke(IPC.workspaceInvoke, input) as Promise<
      WebsiteBridgeResult<{ jobId: string; sessionId: string; state: string }>
    >,
  cancel: (jobId: string) =>
    ipcRenderer.invoke(IPC.workspaceCancel, jobId) as Promise<WebsiteBridgeResult<void>>,
  onJobEvent(listener: (event: unknown) => void) {
    const handler = (_event: Electron.IpcRendererEvent, value: unknown) => listener(value);
    ipcRenderer.on(IPC.workspaceJobEvent, handler);
    return () => ipcRenderer.removeListener(IPC.workspaceJobEvent, handler);
  },
  beginArtifactCreate: (metadata: { mimeType: string; sizeBytes: number; name?: string }) =>
    ipcRenderer.invoke(IPC.workspaceArtifactCreateBegin, metadata),
  appendArtifactChunk: (uploadId: string, chunk: ArrayBuffer) =>
    ipcRenderer.invoke(IPC.workspaceArtifactCreateChunk, uploadId, chunk),
  finishArtifactCreate: (uploadId: string) =>
    ipcRenderer.invoke(IPC.workspaceArtifactCreateFinish, uploadId),
  abortArtifactCreate: (uploadId: string) =>
    ipcRenderer.invoke(IPC.workspaceArtifactCreateAbort, uploadId),
  getArtifactUrl: (artifactId: string) =>
    ipcRenderer.invoke(IPC.workspaceArtifactGetUrl, artifactId),
  saveArtifactAs: (artifactId: string, suggestedFilename?: string) =>
    ipcRenderer.invoke(IPC.workspaceArtifactSaveAs, artifactId, { suggestedFilename }),
  releaseArtifact: (artifactId: string) =>
    ipcRenderer.invoke(IPC.workspaceArtifactRelease, artifactId),
  openSource: () => ipcRenderer.invoke(IPC.workspaceOpenSource),
});

contextBridge.exposeInMainWorld('orynWorkspace', api);
