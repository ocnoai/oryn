import { X509Certificate } from 'node:crypto';
import type { WebContents, Certificate } from 'electron';
import { readOrynNetworkErrorURL } from '../shared/browser-urls';
import { OBSERVATION_READY_TIMEOUT_MS } from './observation-readiness';

export type ConnectionSecurityState =
  | 'secure'
  | 'not-fully-secure'
  | 'not-secure'
  | 'local'
  | 'local-file'
  | 'oryn-page'
  | 'unavailable';
export interface CertificateDetails {
  subject: string;
  issuer: string;
  serialNumber: string;
  validFrom: string;
  validTo: string;
  subjectAltName?: string;
  keyType?: string;
  keySize?: number;
  namedCurve?: string;
  signatureAlgorithm: string;
  fingerprint256: string;
  fingerprint: string;
  keyUsage: string[];
  pem: string;
}
export interface MixedContentIssue {
  url: string;
  resourceType: string;
  resolution: 'blocked' | 'upgraded' | 'loaded-insecure' | 'warning';
}
export interface ConnectionSecuritySnapshot {
  state: ConnectionSecurityState;
  url: string;
  reason?: string;
  failedURL?: string;
  tls?: {
    protocol: string;
    cipher: string;
    keyExchange?: string;
    subject: string;
    issuer: string;
    validFrom: number;
    validTo: number;
    certificateTransparency?: string;
    signedCertificateTimestamps: { log: string; timestamp: number; signatureAlgorithm: string }[];
  };
  certificates: CertificateDetails[];
  chainIncludesRoot: boolean;
  issues: MixedContentIssue[];
}
export interface ConnectionSecurityOptions {
  beforeChildResume?: (
    sessionId: string,
    sendCommand: (method: string, params?: Record<string, unknown>) => Promise<unknown>,
  ) => Promise<void>;
  onCdpMessage?: (method: string, params: Record<string, unknown>, sessionId?: string) => void;
}
interface DocumentResponse {
  loaderId: string;
  frameId: string;
  url: string;
  response: Record<string, unknown>;
}
interface MixedAudit {
  epoch: number;
  sessionId?: string;
  requestId?: string;
  frameId?: string;
  mainURL?: string;
  issue: MixedContentIssue;
  checking?: boolean;
}
interface FreshFrame {
  epoch: number;
  parentId: string;
  bornAt: number;
  committed: boolean;
}
interface OwnedChildSession {
  epoch: number;
  mainLoaderId?: string;
  mainFrameId?: string;
  frameId?: string;
  parentSessionId?: string;
  validated: boolean;
  cancelled: boolean;
  initializing: boolean;
}
const MAX_REQUESTS = 2048;
const MAX_ISSUES = 100;
const FRESH_FRAME_MILLISECONDS = 2000;
const PENDING_ISSUE_REASON = 'Connection details are not yet fully verified';
const MAX_CERTIFICATES = 10;
const MAX_CERTIFICATE_BYTES = 64 * 1024;

/** Read-only, navigation-bound observer. No Security-domain guesses or body capture. */
export class ConnectionSecurityObserver {
  readonly ready: Promise<void>;
  private owned = false;
  private available = false;
  private disposed = false;
  private initializationCancelled = false;
  private epoch = 0;
  private certificateFailure?: {
    failedURL: string;
    certificates: CertificateDetails[];
    epoch: number;
  };
  private mainFrameId?: string;
  private loaderId?: string;
  private committed = false;
  private observationComplete = true;
  private readonly frameIds = new Set<string>();
  private readonly childSessions = new Map<string, OwnedChildSession>();
  private readonly frameLoaders = new Map<string, string>();
  private readonly frameURLs = new Map<string, string>();
  private readonly freshFrames = new Map<string, FreshFrame>();
  private readonly pendingAudits = new Map<string, MixedAudit>();
  private readonly corroboratedIssues: {
    audit: MixedAudit;
    frameId: string;
    loaderId: string;
    fresh?: FreshFrame;
  }[] = [];
  private readonly requests = new Map<
    string,
    { url: string; mixed: boolean; epoch: number; frameId: string; loaderId: string }
  >();
  private readonly responses = new Map<string, DocumentResponse>();
  private value: ConnectionSecuritySnapshot;

  constructor(
    private readonly contents: WebContents,
    private readonly onChanged: (snapshot: ConnectionSecuritySnapshot) => void,
    private readonly options: ConnectionSecurityOptions = {},
  ) {
    this.value = emptySnapshot(contents.getURL());
    contents.on('did-start-navigation', this.onNavigationStart);
    contents.on('did-navigate-in-page', this.onSameDocumentNavigation);
    contents.once('destroyed', this.onDestroyed);
    contents.debugger.on('message', this.onMessage);
    contents.debugger.on('detach', this.onDetach);
    this.ready = withDeadline(this.attach(), OBSERVATION_READY_TIMEOUT_MS).catch(() => {
      this.initializationCancelled = true;
      this.available = false;
      this.observationComplete = false;
      const owned = this.owned;
      this.owned = false;
      if (owned) {
        try {
          if (!contents.isDestroyed() && contents.debugger.isAttached()) contents.debugger.detach();
        } catch {
          /* Native teardown may already have released the target. */
        }
      }
      if (!this.disposed)
        this.unavailable('Connection observation could not be initialized in time');
    });
  }

  get isAvailable(): boolean {
    if (this.disposed || !this.available || !this.owned) return false;
    try {
      return !this.contents.isDestroyed() && this.contents.debugger.isAttached();
    } catch {
      return false;
    }
  }

  snapshot(): ConnectionSecuritySnapshot {
    return structuredClone(this.value);
  }

  sendCommand(
    method: string,
    params: Record<string, unknown> = {},
    sessionId?: string,
  ): Promise<unknown> {
    if (!this.isAvailable) return Promise.reject(new Error('CONNECTION_OBSERVER_UNAVAILABLE'));
    if (sessionId && !this.isCurrentChildSession(sessionId))
      return Promise.reject(new Error('UNKNOWN_DEBUGGER_SESSION'));
    return this.contents.debugger.sendCommand(method, params, sessionId);
  }

  /** Called only for Electron's native main-document certificate-error event. Never trusts it. */
  recordCertificateError(url: string, certificate: Certificate): void {
    if (this.disposed || this.committed || !url.startsWith('https:') || this.value.url !== url)
      return;
    try {
      const encoded: string[] = [];
      const seen = new Set<string>();
      let current: Certificate | undefined = certificate;
      while (current) {
        if (typeof current.data !== 'string' || current.data.length > 96 * 1024)
          throw new Error('CERTIFICATE_SIZE_OR_ENCODING');
        const parsed = new X509Certificate(current.data);
        if (seen.has(parsed.fingerprint256)) break;
        if (encoded.length >= MAX_CERTIFICATES) throw new Error('CERTIFICATE_CHAIN_LIMIT');
        seen.add(parsed.fingerprint256);
        encoded.push(parsed.raw.toString('base64'));
        current = current.issuerCert;
      }
      const certificates = parseCertificateChain(encoded);
      this.certificateFailure = { failedURL: url, certificates, epoch: this.epoch };
      this.value = {
        ...emptySnapshot(url),
        failedURL: url,
        certificates,
        reason: 'Certificate validation failed; the connection remains blocked',
      };
      this.publish();
    } catch {
      this.certificateFailure = undefined;
      this.unavailable('The failed connection certificate could not be inspected');
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.epoch++;
    const owned = this.owned;
    this.owned = false;
    this.available = false;
    try {
      this.contents.removeListener('did-start-navigation', this.onNavigationStart);
      this.contents.removeListener('did-navigate-in-page', this.onSameDocumentNavigation);
      this.contents.removeListener('destroyed', this.onDestroyed);
    } catch {
      // Teardown must never interrupt the remaining native destroyed listeners.
    }
    try {
      // Electron's debugger native methods can throw once its WebContents dies.
      // No detach/query is necessary for a target Chromium already destroyed.
      if (!this.contents.isDestroyed()) {
        this.contents.debugger.removeListener('message', this.onMessage);
        this.contents.debugger.removeListener('detach', this.onDetach);
        if (owned && this.contents.debugger.isAttached()) this.contents.debugger.detach();
      }
    } catch {
      // Native teardown can race this cleanup even before isDestroyed reports it.
    }
    this.requests.clear();
    this.responses.clear();
    this.frameIds.clear();
    this.childSessions.clear();
    this.frameLoaders.clear();
    this.frameURLs.clear();
    this.freshFrames.clear();
    this.pendingAudits.clear();
    this.corroboratedIssues.length = 0;
  }

  private async attach(): Promise<void> {
    try {
      if (this.contents.debugger.isAttached()) {
        this.unavailable('Another debugger is attached');
        return;
      }
      this.contents.debugger.attach('1.3');
      this.owned = true;
      await this.enableDomains();
      if (this.disposed || this.initializationCancelled || !this.owned) return;
      await this.contents.debugger.sendCommand('Target.setAutoAttach', {
        autoAttach: true,
        waitForDebuggerOnStart: true,
        flatten: true,
        filter: [{ type: 'iframe', exclude: false }, { exclude: true }],
      });
      if (this.disposed || this.initializationCancelled || !this.owned) return;
      const tree = record(await this.contents.debugger.sendCommand('Page.getFrameTree'));
      if (this.disposed || this.initializationCancelled || !this.owned) return;
      const frameTree = record(tree?.frameTree);
      const frame = record(frameTree?.frame);
      if (typeof frame?.id === 'string') {
        this.mainFrameId = frame.id;
        this.frameIds.add(frame.id);
      }
      this.available = true;
      // Initialization can overlap the first renderer load. Bind only a frame-tree
      // document matching the browser's current URL; cached responses still need its loader.
      if (!this.committed && frame && string(frame.url) === this.contents.getURL()) {
        this.loaderId = string(frame.loaderId);
        this.committed = true;
        this.value = emptySnapshot(this.contents.getURL());
      }
      if (this.committed && !this.value.failedURL) this.classifyNonTLS(this.value.url);
      const response = this.loaderId ? this.responses.get(this.loaderId) : undefined;
      if (response && response.frameId === this.mainFrameId && response.url === this.value.url)
        void this.applyResponse(response, this.epoch);
      this.publish();
    } catch {
      if (this.disposed || this.initializationCancelled || !this.owned) return;
      this.unavailable('Connection observation could not be initialized');
    }
  }

  private async enableDomains(sessionId?: string): Promise<void> {
    const active = () =>
      !this.disposed &&
      this.owned &&
      !this.initializationCancelled &&
      (!sessionId || this.isCurrentChildSession(sessionId));
    if (!active()) throw new Error('CONNECTION_OBSERVER_UNAVAILABLE');
    await this.contents.debugger.sendCommand('Page.enable', {}, sessionId);
    if (!active()) throw new Error('CONNECTION_OBSERVER_UNAVAILABLE');
    await this.contents.debugger.sendCommand(
      'Network.enable',
      {
        maxTotalBufferSize: 0,
        maxResourceBufferSize: 0,
        maxPostDataSize: 0,
      },
      sessionId,
    );
    if (!active()) throw new Error('CONNECTION_OBSERVER_UNAVAILABLE');
    await this.contents.debugger.sendCommand('Audits.enable', {}, sessionId);
  }

  private readonly onNavigationStart = (
    _event: unknown,
    url: string,
    isInPlace: boolean,
    isMainFrame: boolean,
  ): void => {
    if (!isMainFrame || isInPlace) return;
    const failure = this.certificateFailure;
    const errorPage = readOrynNetworkErrorURL(url);
    const preserveFailure =
      failure &&
      failure.epoch === this.epoch &&
      errorPage?.url === failure.failedURL &&
      errorPage.code <= -200 &&
      errorPage.code >= -299;
    this.epoch++;
    this.certificateFailure = preserveFailure ? { ...failure, epoch: this.epoch } : undefined;
    this.loaderId = undefined;
    this.committed = false;
    this.observationComplete = true;
    this.requests.clear();
    this.responses.clear();
    this.frameIds.clear();
    this.frameLoaders.clear();
    this.frameURLs.clear();
    this.freshFrames.clear();
    this.pendingAudits.clear();
    this.corroboratedIssues.length = 0;
    if (this.mainFrameId) this.frameIds.add(this.mainFrameId);
    this.value = emptySnapshot(url);
    this.publish();
  };

  private readonly onSameDocumentNavigation = (
    _event: unknown,
    url: string,
    isMainFrame: boolean,
  ): void => {
    if (isMainFrame && this.committed) {
      this.value.url = url;
      this.publish();
    }
  };
  private readonly onDestroyed = (): void => this.dispose();
  private readonly onDetach = (): void => {
    this.owned = false;
    this.available = false;
    this.unavailable('Connection observation detached; details are unavailable');
  };

  private readonly onMessage = (
    _event: unknown,
    method: string,
    raw: unknown,
    sessionId?: string,
  ): void => {
    if (this.disposed || !this.owned) return;
    const params = record(raw);
    if (!params) return;
    if (sessionId && !this.childSessions.has(sessionId)) return;
    // Ownership/lifecycle messages still need cleanup for obsolete paused
    // targets. Their Page/Network/Audits data must never seed a new document.
    const lifecycle =
      method === 'Target.attachedToTarget' || method === 'Target.detachedFromTarget';
    if (sessionId && !lifecycle && !this.isCurrentChildSession(sessionId)) return;
    try {
      this.handleMessage(method, params, sessionId);
      this.options.onCdpMessage?.(method, params, sessionId || undefined);
    } catch {
      this.observationComplete = false;
      this.unavailable('Connection observation was incomplete');
    }
  };

  private handleMessage(method: string, params: Record<string, unknown>, sessionId?: string): void {
    if (method === 'Target.attachedToTarget') {
      const child = string(params.sessionId);
      if (!child || record(params.targetInfo)?.type !== 'iframe') return;
      if (this.childSessions.size >= 128) {
        this.observationComplete = false;
        this.publish();
        this.resumeOwnedChild(child);
        return;
      }
      const parent = sessionId ? this.childSessions.get(sessionId) : undefined;
      const owned: OwnedChildSession = {
        epoch: parent?.epoch ?? this.epoch,
        mainLoaderId: parent?.mainLoaderId ?? this.loaderId,
        mainFrameId: parent?.mainFrameId ?? this.mainFrameId,
        frameId: string(record(params.targetInfo)?.targetId),
        parentSessionId: sessionId || undefined,
        validated: false,
        cancelled: false,
        initializing: true,
      };
      this.childSessions.set(child, owned);
      this.publish();
      // A running pre-existing OOPIF may already have loaded unobserved resources.
      if (params.waitingForDebugger !== true) this.observationComplete = false;
      void withDeadline(
        (async () => {
          // Validate while the target is still paused and before enabling any
          // child data domain. The authoritative tree binds its target ID and
          // complete parent chain to this exact committed main loader.
          if (!(await this.validateChildSession(child, owned))) {
            if (!this.disposed && owned.epoch === this.epoch) {
              this.observationComplete = false;
              this.publish();
            }
            return;
          }
          await this.enableDomains(child);
          if (!this.isCurrentChildSession(child)) return;
          await this.contents.debugger.sendCommand(
            'Target.setAutoAttach',
            {
              autoAttach: true,
              waitForDebuggerOnStart: true,
              flatten: true,
              filter: [{ type: 'iframe', exclude: false }, { exclude: true }],
            },
            child,
          );
          await this.options.beforeChildResume?.(child, (method, params = {}) => {
            if (this.disposed || !this.owned || !this.isCurrentChildSession(child))
              return Promise.reject(new Error('UNKNOWN_DEBUGGER_SESSION'));
            return this.contents.debugger.sendCommand(method, params, child);
          });
        })(),
        2000,
      )
        .catch(() => {
          owned.cancelled = true;
          owned.validated = false;
          if (!this.disposed && owned.epoch === this.epoch) {
            this.observationComplete = false;
            this.unavailable('An embedded frame could not be observed');
          }
        })
        .finally(() => {
          owned.initializing = false;
          // Never leave a renderer paused if observation setup fails.
          if (this.childSessions.get(child) === owned) this.resumeOwnedChild(child);
          if (!this.disposed && owned.epoch === this.epoch) this.publish();
        });
      return;
    }
    if (method === 'Target.detachedFromTarget') {
      const child = string(params.sessionId);
      if (child) this.childSessions.delete(child);
      return;
    }
    if (method === 'Page.frameNavigated') {
      const frame = record(params.frame);
      const id = string(frame?.id);
      if (!frame || !id) return;
      this.frameIds.add(id);
      const frameLoader = string(frame.loaderId);
      const frameURL = string(frame.url);
      if (frameLoader) this.frameLoaders.set(id, frameLoader);
      if (frameURL) this.frameURLs.set(id, frameURL);
      const fresh = this.freshFrames.get(id);
      if (fresh) fresh.committed = true;
      if (!sessionId && !frame.parentId) {
        this.mainFrameId = id;
        this.loaderId = string(frame.loaderId);
        this.committed = true;
        const url = string(frame.url) ?? '';
        this.value = emptySnapshot(url);
        this.classifyNonTLS(url);
        const failure = this.certificateFailure;
        if (
          failure?.epoch === this.epoch &&
          readOrynNetworkErrorURL(url)?.url === failure.failedURL
        ) {
          this.value = {
            ...emptySnapshot(url),
            failedURL: failure.failedURL,
            certificates: failure.certificates,
            reason: 'Certificate validation failed; the connection remains blocked',
          };
        }
        const response = this.loaderId ? this.responses.get(this.loaderId) : undefined;
        if (response && response.frameId === id && response.url === url)
          void this.applyResponse(response, this.epoch);
        this.publish();
      }
      this.resolvePendingAudits();
      return;
    }
    if (method === 'Page.frameAttached' || method === 'Page.frameDetached') {
      const id = string(params.frameId);
      if (id) {
        if (method === 'Page.frameAttached') {
          this.frameIds.add(id);
          const parentId = string(params.parentFrameId);
          if (this.committed && parentId && this.frameIds.has(parentId)) {
            if (this.freshFrames.size >= MAX_REQUESTS) this.observationComplete = false;
            else
              this.freshFrames.set(id, {
                epoch: this.epoch,
                parentId,
                bornAt: Date.now(),
                committed: false,
              });
          }
        } else {
          this.frameIds.delete(id);
          this.frameLoaders.delete(id);
          this.frameURLs.delete(id);
          this.freshFrames.delete(id);
        }
        this.resolvePendingAudits();
      }
      return;
    }
    if (method === 'Network.requestWillBeSent') {
      const request = record(params.request);
      const id = string(params.requestId);
      const frameId = string(params.frameId);
      const url = string(request?.url);
      const loaderId = string(params.loaderId);
      if (!id || !frameId || !url || !loaderId || !this.frameIds.has(frameId)) return;
      if (this.requests.size >= MAX_REQUESTS) {
        this.observationComplete = false;
        this.publish();
        return;
      }
      this.requests.set(`${sessionId ?? ''}:${id}`, {
        url,
        frameId,
        loaderId,
        epoch: this.epoch,
        mixed:
          request?.mixedContentType === 'blockable' ||
          request?.mixedContentType === 'optionally-blockable',
      });
      this.resolvePendingAudits();
      return;
    }
    if (method === 'Network.responseReceived') {
      const response = record(params.response);
      const frameId = string(params.frameId);
      const loaderId = string(params.loaderId);
      const url = string(response?.url);
      if (!response || !frameId || !url) return;
      if (!sessionId && params.type === 'Document' && loaderId) {
        const candidate = { frameId, loaderId, url, response };
        if (this.responses.size >= 4) this.responses.delete(this.responses.keys().next().value!);
        this.responses.set(loaderId, candidate);
        if (this.committed && this.loaderId === loaderId && this.value.url === url)
          void this.applyResponse(candidate, this.epoch);
      }
      const request = this.requests.get(`${sessionId ?? ''}:${string(params.requestId) ?? ''}`);
      if (request?.epoch === this.epoch && request.mixed && url.startsWith('http:'))
        this.addIssue({
          url: request.url,
          resourceType: string(params.type) ?? 'Resource',
          resolution: 'loaded-insecure',
        });
      return;
    }
    if (method === 'Network.loadingFailed') {
      const request = this.requests.get(`${sessionId ?? ''}:${string(params.requestId) ?? ''}`);
      if (request?.epoch === this.epoch && params.blockedReason === 'mixed-content')
        this.addIssue({
          url: request.url,
          resourceType: string(params.type) ?? 'Resource',
          resolution: 'blocked',
        });
      return;
    }
    if (method === 'Audits.issueAdded') {
      const details = record(record(record(params.issue)?.details)?.mixedContentIssueDetails);
      if (!details || !this.committed) return;
      const resolution =
        details.resolutionStatus === 'MixedContentBlocked'
          ? 'blocked'
          : details.resolutionStatus === 'MixedContentAutomaticallyUpgraded'
            ? 'upgraded'
            : details.resolutionStatus === 'MixedContentWarning'
              ? 'warning'
              : undefined;
      const url = string(details.insecureURL);
      if (!resolution || !url) return;
      const audit: MixedAudit = {
        epoch: this.epoch,
        sessionId,
        requestId: string(record(details.request)?.requestId),
        frameId: string(record(details.frame)?.frameId),
        mainURL: string(details.mainResourceURL),
        issue: { url, resourceType: string(details.resourceType) ?? 'Resource', resolution },
      };
      if (this.isCorroboratedDuplicate(audit)) return;
      const key = JSON.stringify([
        sessionId ?? '',
        audit.requestId,
        audit.frameId,
        url,
        resolution,
        audit.issue.resourceType,
      ]);
      if (!this.pendingAudits.has(key)) {
        if (this.pendingAudits.size >= MAX_ISSUES) this.observationComplete = false;
        else this.pendingAudits.set(key, audit);
      }
      this.resolvePendingAudits();
    }
  }

  private isCurrentChildSession(sessionId: string): boolean {
    let id: string | undefined = sessionId;
    const seen = new Set<string>();
    while (id) {
      if (seen.has(id) || seen.size >= 128) return false;
      seen.add(id);
      const child = this.childSessions.get(id);
      if (
        !child?.validated ||
        child.cancelled ||
        child.epoch !== this.epoch ||
        !this.committed ||
        child.mainLoaderId !== this.loaderId ||
        child.mainFrameId !== this.mainFrameId ||
        !child.frameId ||
        !this.frameIds.has(child.frameId)
      )
        return false;
      id = child.parentSessionId;
    }
    return true;
  }

  private resumeOwnedChild(sessionId: string): void {
    if (!this.owned || this.disposed) return;
    try {
      void this.contents.debugger
        .sendCommand('Runtime.runIfWaitingForDebugger', {}, sessionId)
        .catch(() => undefined);
    } catch {
      // Native destruction can race cleanup; detached targets resume.
    }
  }

  private async validateChildSession(id: string, child: OwnedChildSession): Promise<boolean> {
    const current = () =>
      !this.disposed &&
      this.owned &&
      this.committed &&
      this.childSessions.get(id) === child &&
      !child.cancelled &&
      child.epoch === this.epoch &&
      !!child.mainLoaderId &&
      child.mainLoaderId === this.loaderId &&
      child.mainFrameId === this.mainFrameId &&
      !!child.frameId &&
      (!child.parentSessionId || this.isCurrentChildSession(child.parentSessionId));
    if (!current()) return false;
    const result = record(await this.sendCommand('Page.getFrameTree'));
    if (!current()) return false;
    const tree = record(result?.frameTree);
    const root = record(tree?.frame);
    if (
      !tree ||
      root?.id !== child.mainFrameId ||
      root?.loaderId !== child.mainLoaderId ||
      root?.url !== this.value.url
    )
      return false;
    const path = frameTreePath(tree, child.frameId!);
    if (
      !path ||
      path.length < 2 ||
      !path.every((frame, index) => index === 0 || frame.parentId === path[index - 1]?.id)
    )
      return false;
    if (child.parentSessionId) {
      const parent = this.childSessions.get(child.parentSessionId);
      if (!path.slice(0, -1).some((frame) => frame.id === parent?.frameId)) return false;
    }
    if (this.frameIds.size + path.length > MAX_REQUESTS) return false;
    for (const frame of path) {
      const frameId = string(frame.id);
      if (!frameId) return false;
      this.frameIds.add(frameId);
      const loaderId = string(frame.loaderId);
      const url = string(frame.url);
      if (loaderId) this.frameLoaders.set(frameId, loaderId);
      if (url) this.frameURLs.set(frameId, url);
      const fresh = this.freshFrames.get(frameId);
      if (fresh && url && url !== 'about:blank') fresh.committed = true;
    }
    child.validated = true;
    return true;
  }

  private isCurrentRequest(request: { epoch: number; frameId: string; loaderId: string }): boolean {
    return (
      request.epoch === this.epoch &&
      this.frameIds.has(request.frameId) &&
      this.frameLoaders.get(request.frameId) === request.loaderId
    );
  }

  private isCorroboratedDuplicate(audit: MixedAudit): boolean {
    if (audit.sessionId && !this.isCurrentChildSession(audit.sessionId)) return false;
    if (audit.epoch !== this.epoch || !audit.frameId || !this.frameIds.has(audit.frameId))
      return false;
    const request = audit.requestId
      ? this.requests.get(`${audit.sessionId ?? ''}:${audit.requestId}`)
      : undefined;
    if (request && (!this.isCurrentRequest(request) || request.frameId !== audit.frameId))
      return false;
    return this.corroboratedIssues.some(
      (proof) =>
        proof.audit.epoch === this.epoch &&
        proof.audit.sessionId === audit.sessionId &&
        proof.frameId === audit.frameId &&
        proof.audit.issue.url === audit.issue.url &&
        proof.audit.issue.resolution === audit.issue.resolution &&
        proof.audit.issue.resourceType === audit.issue.resourceType &&
        (proof.fresh
          ? this.freshFrames.get(proof.frameId) === proof.fresh &&
            !proof.fresh.committed &&
            this.loaderId === proof.loaderId &&
            Date.now() - proof.fresh.bornAt <= FRESH_FRAME_MILLISECONDS
          : this.frameLoaders.get(proof.frameId) === proof.loaderId),
    );
  }

  private corroborateAudit(
    key: string,
    audit: MixedAudit,
    frameId: string,
    loaderId: string,
    fresh?: FreshFrame,
  ): void {
    this.pendingAudits.delete(key);
    if (this.corroboratedIssues.length >= MAX_ISSUES) {
      this.observationComplete = false;
    } else {
      this.corroboratedIssues.push({ audit, frameId, loaderId, fresh });
      this.addIssue(audit.issue);
    }
  }

  private resolvePendingAudits(): void {
    for (const [key, audit] of this.pendingAudits) {
      if (audit.epoch !== this.epoch) {
        this.pendingAudits.delete(key);
        continue;
      }
      if (audit.sessionId && !this.isCurrentChildSession(audit.sessionId)) continue;
      if (this.isCorroboratedDuplicate(audit)) {
        this.pendingAudits.delete(key);
        continue;
      }
      const request = audit.requestId
        ? this.requests.get(`${audit.sessionId ?? ''}:${audit.requestId}`)
        : undefined;
      if (
        request &&
        this.isCurrentRequest(request) &&
        (!audit.frameId || audit.frameId === request.frameId) &&
        (audit.mainURL === this.value.url || audit.mainURL === this.frameURLs.get(request.frameId))
      ) {
        this.corroborateAudit(key, audit, request.frameId, request.loaderId);
      } else if (
        !audit.requestId &&
        audit.issue.resolution === 'blocked' &&
        audit.issue.resourceType === 'Frame' &&
        audit.frameId &&
        !audit.checking
      ) {
        const fresh = this.freshFrames.get(audit.frameId);
        if (
          fresh &&
          fresh.epoch === this.epoch &&
          !fresh.committed &&
          Date.now() - fresh.bornAt <= FRESH_FRAME_MILLISECONDS &&
          audit.mainURL === this.value.url
        ) {
          audit.checking = true;
          void this.corroborateBlockedFrame(key, audit, fresh);
        }
      } else if (
        !audit.requestId &&
        audit.issue.resolution === 'warning' &&
        audit.issue.resourceType === 'Form' &&
        audit.frameId &&
        !audit.checking
      ) {
        const loaderId = this.frameLoaders.get(audit.frameId);
        if (loaderId && this.frameIds.has(audit.frameId)) {
          audit.checking = true;
          void this.corroborateFormWarning(key, audit, loaderId);
        }
      }
    }
    // A speculative audit can precede the later matching request's audit in
    // insertion order. Recheck only native-frame/current-loader proven duplicates
    // after the pass establishes that proof; unresolved distinct evidence stays.
    for (const [key, audit] of this.pendingAudits) {
      if (this.isCorroboratedDuplicate(audit)) this.pendingAudits.delete(key);
    }
    this.publish();
  }

  private async corroborateFormWarning(
    key: string,
    audit: MixedAudit,
    frameLoaderId: string,
  ): Promise<void> {
    const epoch = this.epoch;
    const mainLoaderId = this.loaderId;
    const mainFrameId = this.mainFrameId;
    const session = audit.sessionId ? this.childSessions.get(audit.sessionId) : undefined;
    try {
      const result = record(
        await withDeadline(this.sendCommand('Page.getFrameTree'), FRESH_FRAME_MILLISECONDS),
      );
      if (
        this.disposed ||
        !this.committed ||
        epoch !== this.epoch ||
        audit.epoch !== epoch ||
        this.pendingAudits.get(key) !== audit ||
        !mainLoaderId ||
        mainLoaderId !== this.loaderId ||
        !audit.frameId ||
        this.frameLoaders.get(audit.frameId) !== frameLoaderId ||
        (audit.sessionId &&
          (this.childSessions.get(audit.sessionId) !== session ||
            !this.isCurrentChildSession(audit.sessionId)))
      )
        return;
      const tree = record(result?.frameTree);
      const root = record(tree?.frame);
      if (
        !tree ||
        root?.id !== mainFrameId ||
        root?.loaderId !== mainLoaderId ||
        root?.url !== this.value.url
      )
        return;
      const path = frameTreePath(tree, audit.frameId);
      if (
        !path ||
        path.at(-1)?.loaderId !== frameLoaderId ||
        !path.every(
          (frame, index) =>
            typeof frame.id === 'string' &&
            this.frameIds.has(frame.id) &&
            this.frameLoaders.get(frame.id) === frame.loaderId &&
            (index === 0 || frame.parentId === path[index - 1]?.id),
        ) ||
        (session && !path.some((frame) => frame.id === session.frameId))
      )
        return;
      // Chromium coarsens Form mainResourceURL to its top-level origin. This
      // cross-check supplements the exact native loader/session/tree proof.
      if (!audit.mainURL || new URL(audit.mainURL).origin !== new URL(this.value.url).origin)
        return;
      this.corroborateAudit(key, audit, audit.frameId, frameLoaderId);
      this.resolvePendingAudits();
    } catch {
      // Unknown, replaced or timed-out frame evidence remains Unavailable.
    } finally {
      if (!this.disposed && epoch === this.epoch) this.publish();
    }
  }

  private async corroborateBlockedFrame(
    key: string,
    audit: MixedAudit,
    fresh: FreshFrame,
  ): Promise<void> {
    const epoch = this.epoch;
    const loaderId = this.loaderId;
    const mainFrameId = this.mainFrameId;
    try {
      const result = record(
        await withDeadline(this.sendCommand('Page.getFrameTree'), FRESH_FRAME_MILLISECONDS),
      );
      if (
        this.disposed ||
        audit.epoch !== this.epoch ||
        epoch !== this.epoch ||
        !this.committed ||
        this.pendingAudits.get(key) !== audit ||
        this.loaderId !== loaderId ||
        !loaderId ||
        !audit.frameId ||
        this.freshFrames.get(audit.frameId) !== fresh ||
        fresh.committed ||
        !this.frameIds.has(audit.frameId) ||
        Date.now() - fresh.bornAt > FRESH_FRAME_MILLISECONDS
      )
        return;
      const tree = record(result?.frameTree);
      const root = record(tree?.frame);
      if (
        !tree ||
        root?.id !== mainFrameId ||
        root?.loaderId !== loaderId ||
        root?.url !== this.value.url
      )
        return;
      const path = frameTreePath(tree, audit.frameId);
      const target = path?.at(-1);
      const parent = path?.at(-2);
      if (
        !path ||
        !target ||
        !parent ||
        parent.id !== fresh.parentId ||
        !['', 'about:blank'].includes(typeof target.url === 'string' ? target.url : '') ||
        !path.every(
          (frame, index) =>
            typeof frame.id === 'string' &&
            this.frameIds.has(frame.id) &&
            (index === 0 || frame.parentId === path[index - 1]?.id),
        )
      )
        return;
      this.corroborateAudit(key, audit, audit.frameId, loaderId, fresh);
      this.resolvePendingAudits();
    } catch {
      // A timed-out, detached or stale frame never supplies evidence. The bounded
      // unresolved audit keeps connection state Unavailable until navigation.
    } finally {
      if (!this.disposed && epoch === this.epoch) this.publish();
    }
  }

  private async applyResponse(candidate: DocumentResponse, epoch: number): Promise<void> {
    if (!this.current(candidate, epoch) || !candidate.url.startsWith('https:')) return;
    const details = parseTLS(record(candidate.response.securityDetails));
    if (candidate.response.securityState !== 'secure' || !details) {
      this.unavailable('The committed connection could not be verified');
      return;
    }
    try {
      const result = record(
        await this.sendCommand('Network.getCertificate', { origin: new URL(candidate.url).origin }),
      );
      const certificates = parseCertificateChain(result?.tableNames);
      if (!certificates.length) throw new Error('CERTIFICATE_CHAIN_UNAVAILABLE');
      const leaf = new X509Certificate(certificates[0]!.pem);
      const legacy = leaf.toLegacyObject();
      const hostname = new URL(candidate.url).hostname.replace(/^\[|\]$/g, '');
      if (
        (!leaf.checkHost(hostname) && !leaf.checkIP(hostname)) ||
        Math.floor(Date.parse(leaf.validFrom) / 1000) !== details.validFrom ||
        Math.floor(Date.parse(leaf.validTo) / 1000) !== details.validTo ||
        (legacy.subject.CN && legacy.subject.CN !== details.subject) ||
        (legacy.issuer.CN && legacy.issuer.CN !== details.issuer)
      )
        throw new Error('CERTIFICATE_ATTRIBUTION_MISMATCH');
      if (!this.current(candidate, epoch)) return;
      this.value.tls = details;
      this.value.certificates = certificates;
      const last = new X509Certificate(certificates.at(-1)!.pem);
      this.value.chainIncludesRoot = last.subject === last.issuer && last.verify(last.publicKey);
      this.value.reason = undefined;
      this.publish();
    } catch {
      if (this.current(candidate, epoch))
        this.unavailable('Certificate details could not be attributed to this document');
    }
  }

  private current(candidate: DocumentResponse, epoch: number): boolean {
    return (
      this.isAvailable &&
      epoch === this.epoch &&
      this.committed &&
      this.loaderId === candidate.loaderId &&
      this.mainFrameId === candidate.frameId &&
      this.value.url === candidate.url
    );
  }

  private classifyNonTLS(url: string): void {
    try {
      const parsed = new URL(url);
      if (parsed.protocol === 'http:')
        this.value.state = ['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname)
          ? 'local'
          : 'not-secure';
      else if (parsed.protocol === 'file:') this.value.state = 'local-file';
      else if (parsed.protocol === 'oryn:' || url === 'about:blank') this.value.state = 'oryn-page';
    } catch {
      /* Unrecognized committed documents remain unavailable. */
    }
  }

  private addIssue(issue: MixedContentIssue): void {
    if (!this.committed || !this.value.url.startsWith('https:')) return;
    if (
      this.value.issues.some(
        (existing) => existing.url === issue.url && existing.resolution === issue.resolution,
      )
    )
      return;
    if (this.value.issues.length >= MAX_ISSUES) {
      this.observationComplete = false;
      this.publish();
      return;
    }
    this.value.issues.push(issue);
    this.publish();
  }

  private unavailable(reason: string): void {
    this.value = { ...emptySnapshot(this.value.url), reason };
    this.publish();
  }

  private publish(): void {
    if (this.disposed) return;
    if (!this.available || !this.observationComplete) {
      this.value.state = 'unavailable';
    } else if (
      this.pendingAudits.size > 0 ||
      [...this.childSessions.values()].some(
        (child) => child.epoch === this.epoch && child.initializing,
      )
    ) {
      this.value.state = 'unavailable';
      this.value.reason = PENDING_ISSUE_REASON;
    } else if (this.value.tls && this.value.certificates.length) {
      if (this.value.reason === PENDING_ISSUE_REASON) this.value.reason = undefined;
      this.value.state = this.value.issues.length > 0 ? 'not-fully-secure' : 'secure';
    }
    this.onChanged(this.snapshot());
  }
}

function frameTreePath(
  tree: Record<string, unknown>,
  target: string,
): Record<string, unknown>[] | undefined {
  const pending: { tree: Record<string, unknown>; path: Record<string, unknown>[] }[] = [
    { tree, path: [] },
  ];
  const seen = new Set<string>();
  while (pending.length && seen.size < MAX_REQUESTS) {
    const item = pending.pop()!;
    const frame = record(item.tree.frame);
    const id = string(frame?.id);
    if (!frame || !id || seen.has(id) || item.path.length >= 64) return undefined;
    seen.add(id);
    const path = [...item.path, frame];
    if (id === target) return path;
    const children = item.tree.childFrames;
    if (Array.isArray(children)) {
      if (children.length + pending.length > MAX_REQUESTS) return undefined;
      for (const child of children) {
        const value = record(child);
        if (value) pending.push({ tree: value, path });
      }
    }
  }
  return undefined;
}

export function parseCertificateChain(value: unknown): CertificateDetails[] {
  if (!Array.isArray(value) || value.length > MAX_CERTIFICATES)
    throw new Error('CERTIFICATE_CHAIN_LIMIT');
  const certificates = value.map((entry): CertificateDetails => {
    if (
      typeof entry !== 'string' ||
      entry.length > Math.ceil(MAX_CERTIFICATE_BYTES / 3) * 4 ||
      !/^[A-Za-z0-9+/]+={0,2}$/.test(entry)
    )
      throw new Error('CERTIFICATE_SIZE_OR_ENCODING');
    const der = Buffer.from(entry, 'base64');
    if (!der.length || der.length > MAX_CERTIFICATE_BYTES || der.toString('base64') !== entry)
      throw new Error('CERTIFICATE_SIZE_OR_ENCODING');
    const cert = new X509Certificate(der);
    const key = cert.publicKey;
    return {
      subject: cert.subject,
      issuer: cert.issuer,
      serialNumber: cert.serialNumber,
      validFrom: cert.validFrom,
      validTo: cert.validTo,
      subjectAltName: cert.subjectAltName,
      keyType: key.asymmetricKeyType,
      keySize: key.asymmetricKeyDetails?.modulusLength,
      namedCurve: key.asymmetricKeyDetails?.namedCurve,
      signatureAlgorithm: certificateSignatureOID(der),
      fingerprint256: cert.fingerprint256,
      fingerprint: cert.fingerprint,
      keyUsage: cert.keyUsage ?? [],
      pem: cert.toString(),
    };
  });
  for (let index = 0; index + 1 < certificates.length; index++) {
    const child = new X509Certificate(certificates[index]!.pem);
    const issuer = new X509Certificate(certificates[index + 1]!.pem);
    if (!child.checkIssued(issuer) || !child.verify(issuer.publicKey))
      throw new Error('CERTIFICATE_CHAIN_ORDER');
  }
  return certificates;
}

/** Export only an already parsed bounded chain, never a path or a trust mutation. */
export function exportCertificateChain(certificates: CertificateDetails[]): string {
  if (
    certificates.length > MAX_CERTIFICATES ||
    certificates.some(
      (certificate) => typeof certificate.pem !== 'string' || certificate.pem.length > 96 * 1024,
    )
  )
    throw new Error('CERTIFICATE_CHAIN_LIMIT');
  return parseCertificateChain(
    certificates.map((certificate) => new X509Certificate(certificate.pem).raw.toString('base64')),
  )
    .map((certificate) => certificate.pem)
    .join('\n');
}

function certificateSignatureOID(der: Buffer): string {
  const outer = derElement(der, 0, der.length);
  if (outer.tag !== 0x30 || outer.end !== der.length) throw new Error('INVALID_CERTIFICATE_DER');
  const tbs = derElement(der, outer.start, outer.end);
  const algorithm = derElement(der, tbs.end, outer.end);
  const oid = derElement(der, algorithm.start, algorithm.end);
  if (
    algorithm.tag !== 0x30 ||
    oid.tag !== 6 ||
    oid.end - oid.start < 1 ||
    oid.end - oid.start > 32
  )
    throw new Error('INVALID_CERTIFICATE_SIGNATURE');
  const bytes = der.subarray(oid.start, oid.end);
  const components: number[] = [];
  let current = 0;
  for (let index = 0; index < bytes.length; index++) {
    current = current * 128 + (bytes[index]! & 127);
    if (!Number.isSafeInteger(current)) throw new Error('INVALID_CERTIFICATE_SIGNATURE');
    if (!(bytes[index]! & 128)) {
      components.push(current);
      current = 0;
    }
  }
  if (bytes.at(-1)! & 128) throw new Error('INVALID_CERTIFICATE_SIGNATURE');
  const first = components.shift()!;
  const parts = first < 40 ? [0, first] : first < 80 ? [1, first - 40] : [2, first - 80];
  parts.push(...components);
  return parts.join('.');
}
function derElement(bytes: Buffer, offset: number, limit: number) {
  if (offset + 2 > limit) throw new Error('INVALID_CERTIFICATE_DER');
  const tag = bytes[offset++]!;
  let length = bytes[offset++]!;
  if (length & 128) {
    const count = length & 127;
    if (!count || count > 4 || offset + count > limit) throw new Error('INVALID_CERTIFICATE_DER');
    length = 0;
    for (let index = 0; index < count; index++) length = length * 256 + bytes[offset++]!;
  }
  if (offset + length > limit) throw new Error('INVALID_CERTIFICATE_DER');
  return { tag, start: offset, end: offset + length };
}
function parseTLS(value: Record<string, unknown> | undefined): ConnectionSecuritySnapshot['tls'] {
  if (!value) return undefined;
  const protocol = string(value.protocol);
  const cipher = string(value.cipher);
  const subject = string(value.subjectName);
  const issuer = string(value.issuer);
  if (
    !protocol ||
    !cipher ||
    !subject ||
    !issuer ||
    typeof value.validFrom !== 'number' ||
    typeof value.validTo !== 'number' ||
    !Number.isFinite(value.validFrom) ||
    !Number.isFinite(value.validTo)
  )
    return undefined;
  const timestamps = Array.isArray(value.signedCertificateTimestampList)
    ? value.signedCertificateTimestampList.slice(0, 64)
    : [];
  return {
    protocol,
    cipher,
    subject,
    issuer,
    validFrom: value.validFrom,
    validTo: value.validTo,
    keyExchange: string(value.keyExchange),
    certificateTransparency: string(value.certificateTransparencyCompliance),
    signedCertificateTimestamps: timestamps.flatMap((entry) => {
      const timestamp = record(entry);
      const log = string(timestamp?.logDescription);
      const signatureAlgorithm = string(timestamp?.signatureAlgorithm);
      return log &&
        signatureAlgorithm &&
        typeof timestamp?.timestamp === 'number' &&
        Number.isFinite(timestamp.timestamp) &&
        // Finite remote SCT values can still exceed Date's representable range.
        // Only publish timestamps that trusted chrome can format without throwing.
        Number.isFinite(new Date(timestamp.timestamp).getTime())
        ? [{ log, signatureAlgorithm, timestamp: timestamp.timestamp }]
        : [];
    }),
  };
}
function emptySnapshot(url: string): ConnectionSecuritySnapshot {
  return { state: 'unavailable', url, certificates: [], chainIncludesRoot: false, issues: [] };
}
function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
function string(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 && value.length <= 8192 ? value : undefined;
}

function withDeadline<T>(operation: Promise<T>, milliseconds: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('CONNECTION_OBSERVATION_TIMEOUT')),
      milliseconds,
    );
    operation.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}
