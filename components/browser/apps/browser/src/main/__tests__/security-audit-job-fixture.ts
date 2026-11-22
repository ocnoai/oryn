// Deterministic Engine and artifact boundary fixtures for Browser orchestration tests.
import { EventEmitter } from 'node:events';

import { vi } from 'vitest';
import type { EngineJobReference, TaskProfile } from '@ocno/oryn-protocol';

import type { BrowserCallerContextService } from '../browser-caller-context';
import type { ArtifactManager } from '../artifact-manager/artifact-manager';
import type { BrowserModelSessionRegistry } from '../model-resolution/browser-model-sessions';
import type { AIActivityStore } from '../data/ai-activity-store';
import { BrowserJobRegistry } from '../jobs/browser-job-registry';

const owner = {
  origin: 'https://site.test',
  tabId: 'tab_one',
  webContentsId: 10,
  frameId: 1,
  frameUrl: 'https://site.test/app',
  documentId: 'document_one',
};

class FakeEngine extends EventEmitter {
  readonly engineInstanceId = 'engine_test';
  readonly createJob = vi.fn(
    async (
      jobId: string,
      sessionId: string,
      input: unknown,
      artifacts: unknown[],
    ): Promise<EngineJobReference> => {
      void input;
      void artifacts;
      return {
        jobId,
        sessionId,
        state: 'running' as const,
      };
    },
  );
  readonly cancelJob = vi.fn(async (jobId: string) => ({
    jobId,
    sessionId: 'session_one',
    state: 'cancelled' as const,
    sequence: 2,
  }));
}

function setup(task = 'automatic-speech-recognition', taskProfile?: TaskProfile) {
  const engine = new FakeEngine();
  const release = vi.fn(async () => undefined);
  const acquireEngineInputLeases = vi.fn(() => ({
    bindings: [
      {
        artifactId: 'artifact_abcdefghijklmnopqrstuvwxyz',
        internalPath: '/controlled/input/object',
        mimeType: 'audio/wav',
        sizeBytes: 3,
        access: 'read' as const,
      },
    ],
    release,
  }));
  const publishEngineResult = vi.fn(
    async (_owner: unknown, _engineId: string, _jobId: string, value: unknown, preview = false) => {
      if (preview) {
        return {
          kind: 'artifact',
          artifact: {
            id: 'artifact_preview_abcdefghijklmnopqrstuvwxyz',
            mimeType: 'image/png',
            sizeBytes: 4,
            createdAt: 1,
          },
        };
      }
      void value;
      return {
        kind: 'artifact',
        artifact: {
          id: 'artifact_result_abcdefghijklmnopqrstuvwxyz',
          mimeType: 'video/mp4',
          sizeBytes: 64,
          createdAt: 2,
        },
      };
    },
  );
  const cleanupEngineJob = vi.fn();
  const sessions = {
    requireOwned: (sessionId: string) => ({
      sessionId,
      role: 'transcription',
      origin: owner.origin,
      tabId: owner.tabId,
      webContentsId: owner.webContentsId,
      documentId: owner.documentId,
      resolution: {
        preferredModel: {
          provider: 'huggingface',
          repository: 'openai/whisper-small',
          revision: 'a'.repeat(40),
        },
        resolvedModel: {
          provider: 'huggingface',
          repository: 'openai/whisper-small',
          revision: 'a'.repeat(40),
        },
        compatibility: 'compatible',
        executability: 'executable',
        security: 'allowed',
        developerCertified: false,
        substituted: false,
        overriddenByUser: false,
        resolutionReason: 'preferred-model',
        compatibilityReasons: [],
        downloadRequired: false,
      },
      runtime: { runner: 'test', backend: 'cpu', deviceClass: 'cpu' },
      capabilities: { task },
      taskProfile,
      request: {},
    }),
  };
  const callerContexts = {
    hasDocumentAuthority: (_tabId: string, _webContentsId: number, documentId: string) =>
      documentId.startsWith('document_'),
  };
  const activity = {
    create: vi.fn(() => 'activity_abcdefghijklmnopqrstuvwxyz'),
    complete: vi.fn(),
  };
  const registry = new BrowserJobRegistry(
    engine,
    sessions as unknown as BrowserModelSessionRegistry,
    {
      acquireEngineInputLeases,
      publishEngineResult,
      cleanupEngineJob,
      cancelPublication: vi.fn(),
    } as unknown as ArtifactManager,
    callerContexts as unknown as BrowserCallerContextService,
    activity as unknown as AIActivityStore,
  );
  return {
    engine,
    registry,
    sessions,
    callerContexts,
    artifacts: { acquireEngineInputLeases, publishEngineResult, cleanupEngineJob },
    release,
    acquireEngineInputLeases,
    publishEngineResult,
    cleanupEngineJob,
    activity,
  };
}

export { owner, setup };
