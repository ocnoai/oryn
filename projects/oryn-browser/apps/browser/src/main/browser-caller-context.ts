import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';

import type { IpcMainInvokeEvent, WebContents } from 'electron';

import { readOrynModelWorkspaceReference } from '../shared/browser-urls';

export interface BrowserCallerContext {
  tabId: string;
  webContentsId: number;
  frameId: number;
  frameUrl: string;
  origin: string;
  documentId: string;
}

interface ManagedDocument {
  tabId: string;
  documentId: string;
  webContents: WebContents;
  provisional?: boolean;
  crashed?: boolean;
  committedURL?: string;
  processId?: number;
  routingId?: number;
  securityClass: 'website' | 'model-workspace' | 'local-file' | 'private';
}

export function isAllowedLocalAIUrl(value: string): boolean {
  try {
    const url = new URL(value);
    if (url.protocol === 'https:') {
      return true;
    }
    return (
      url.protocol === 'http:' &&
      (url.hostname === 'localhost' || url.hostname === '127.0.0.1' || url.hostname === '[::1]')
    );
  } catch {
    return false;
  }
}

export class BrowserCallerContextService extends EventEmitter {
  private readonly managed = new Map<number, ManagedDocument>();

  constructor() {
    super();
  }

  register(
    tabId: string,
    webContents: WebContents,
    securityClass: 'website' | 'model-workspace' | 'local-file' | 'private' = 'website',
  ): () => void {
    const refreshDocument = (crashed = false) => {
      const previous = this.managed.get(webContents.id);
      if (previous) {
        this.emit('document-invalidated', {
          tabId: previous.tabId,
          webContentsId: webContents.id,
          documentId: previous.documentId,
        });
      }
      // A crashed renderer has no usable frame; querying it can throw before recovery UI publishes.
      const frame = crashed ? undefined : webContents.mainFrame;
      this.managed.set(webContents.id, {
        tabId,
        webContents,
        securityClass,
        documentId: `document_${randomUUID().replaceAll('-', '')}`,
        committedURL: frame?.url,
        processId: frame?.processId,
        routingId: frame?.routingId,
        crashed,
        provisional: crashed,
      });
    };

    refreshDocument();
    const rotateForNavigation = (
      _event: Electron.Event,
      _url: string,
      isSameDocument: boolean,
      isMainFrame: boolean,
    ) => {
      if (isMainFrame && !isSameDocument) {
        const current = this.managed.get(webContents.id);
        if (current) current.provisional = true;
      }
    };
    webContents.on('did-start-navigation', rotateForNavigation);
    const committed = () => refreshDocument();
    const stopped = () => {
      const current = this.managed.get(webContents.id);
      if (!current?.provisional || current.crashed) return;
      const frame = webContents.mainFrame;
      if (
        current?.provisional &&
        !current.crashed &&
        current.committedURL === frame.url &&
        current.processId === frame.processId &&
        current.routingId === frame.routingId
      )
        current.provisional = false;
    };
    const crashed = () => {
      refreshDocument(true);
    };
    const sameDocument = (_event: Electron.Event, url: string, isMainFrame: boolean) => {
      const current = this.managed.get(webContents.id);
      if (current && isMainFrame) current.committedURL = url;
    };
    webContents.on('did-navigate-in-page', sameDocument);
    webContents.on('did-navigate', committed);
    webContents.on('did-stop-loading', stopped);
    webContents.on('render-process-gone', crashed);
    const destroyed = () => {
      const previous = this.managed.get(webContents.id);
      if (previous) {
        this.emit('document-invalidated', {
          tabId,
          webContentsId: webContents.id,
          documentId: previous.documentId,
        });
        previous.provisional = true;
      }
    };
    webContents.on('destroyed', destroyed);

    return () => {
      webContents.removeListener('did-start-navigation', rotateForNavigation);
      webContents.removeListener('did-navigate-in-page', sameDocument);
      webContents.removeListener('did-navigate', committed);
      webContents.removeListener('did-stop-loading', stopped);
      webContents.removeListener('render-process-gone', crashed);
      webContents.removeListener('destroyed', destroyed);
      const previous = this.managed.get(webContents.id);
      if (previous) {
        this.emit('document-invalidated', {
          tabId: previous.tabId,
          webContentsId: webContents.id,
          documentId: previous.documentId,
        });
      }
      this.managed.delete(webContents.id);
    };
  }

  getDocumentId(webContents: WebContents): string {
    const managed = this.managed.get(webContents.id);
    if (!managed || managed.webContents !== webContents) {
      throw new Error('LOCAL_AI_CALLER_NOT_MANAGED');
    }
    return managed.documentId;
  }

  hasDocumentAuthority(tabId: string, webContentsId: number, documentId: string): boolean {
    const managed = this.managed.get(webContentsId);
    return (
      managed?.tabId === tabId &&
      managed.documentId === documentId &&
      !managed.provisional &&
      !managed.webContents.isDestroyed()
    );
  }

  derive(event: IpcMainInvokeEvent, activeWebContents: WebContents): BrowserCallerContext {
    if (event.sender !== activeWebContents) {
      throw new Error('LOCAL_AI_CALLER_NOT_ACTIVE');
    }
    return this.deriveManaged(event);
  }

  deriveManaged(event: IpcMainInvokeEvent): BrowserCallerContext {
    const sender = event.sender;
    const senderFrame = event.senderFrame;
    const managed = this.managed.get(sender.id);

    if (!managed || managed.webContents !== sender) {
      throw new Error('LOCAL_AI_CALLER_NOT_MANAGED');
    }

    if (managed.securityClass !== 'website') throw new Error('LOCAL_AI_CALLER_NOT_MANAGED');
    if (managed.provisional) throw new Error('LOCAL_AI_DOCUMENT_TRANSITION');
    const mainFrame = sender.mainFrame;
    if (
      !senderFrame ||
      senderFrame.processId !== mainFrame.processId ||
      senderFrame.routingId !== mainFrame.routingId
    ) {
      throw new Error('LOCAL_AI_FRAME_NOT_ALLOWED');
    }

    const frameUrl = senderFrame.url;
    if (!isAllowedLocalAIUrl(frameUrl)) {
      throw new Error('LOCAL_AI_SECURE_CONTEXT_REQUIRED');
    }

    const parsed = new URL(frameUrl);
    return {
      webContentsId: sender.id,
      tabId: managed.tabId,
      frameId: senderFrame.routingId,
      frameUrl,
      origin: parsed.origin,
      documentId: managed.documentId,
    };
  }

  deriveModelWorkspace(event: IpcMainInvokeEvent): BrowserCallerContext & { reference: string } {
    const sender = event.sender;
    const senderFrame = event.senderFrame;
    const managed = this.managed.get(sender.id);
    if (!managed || managed.webContents !== sender || managed.securityClass !== 'model-workspace')
      throw new Error('WORKSPACE_CALLER_NOT_MANAGED');
    if (managed.provisional) throw new Error('LOCAL_AI_DOCUMENT_TRANSITION');
    const mainFrame = sender.mainFrame;
    if (
      !senderFrame ||
      senderFrame.processId !== mainFrame.processId ||
      senderFrame.routingId !== mainFrame.routingId
    ) {
      throw new Error('WORKSPACE_FRAME_NOT_ALLOWED');
    }
    const reference = readOrynModelWorkspaceReference(senderFrame.url);
    if (!reference) throw new Error('WORKSPACE_ROUTE_REQUIRED');
    return {
      tabId: managed.tabId,
      webContentsId: sender.id,
      frameId: senderFrame.routingId,
      frameUrl: senderFrame.url,
      origin: 'oryn://model-workspace',
      documentId: managed.documentId,
      reference,
    };
  }
}
