import { EventEmitter } from 'node:events';
import { X509Certificate } from 'node:crypto';
import { readFileSync } from 'node:fs';
import type { WebContents, Certificate } from 'electron';
import { describe, expect, it, vi } from 'vitest';
import {
  ConnectionSecurityObserver,
  parseCertificateChain,
  exportCertificateChain,
} from '../connection-security';

const pem = readFileSync(
  new URL('./fixtures/connection-security-cert.pem', import.meta.url),
  'utf8',
);
const certificate = new X509Certificate(pem);
const der = certificate.raw.toString('base64');
class FakeDebugger extends EventEmitter {
  attached = false;
  attach = vi.fn(() => {
    this.attached = true;
  });
  detach = vi.fn(() => {
    this.attached = false;
  });
  isAttached() {
    return this.attached;
  }
  sendCommand = vi.fn(async (method: string): Promise<unknown> => {
    if (method === 'Page.getFrameTree') return { frameTree: { frame: { id: 'main' } } };
    if (method === 'Network.getCertificate') return { tableNames: [der] };
    return {};
  });
  message(method: string, params: object, sessionId = '') {
    this.emit('message', {}, method, params, sessionId);
  }
}
async function setup(external = false) {
  const debug = new FakeDebugger();
  debug.attached = external;
  const contents = Object.assign(new EventEmitter(), {
    debugger: debug,
    isDestroyed: () => false,
    getURL: () => 'about:blank',
  }) as unknown as WebContents;
  const changed = vi.fn();
  const cdp = vi.fn();
  const observer = new ConnectionSecurityObserver(contents, changed, { onCdpMessage: cdp });
  await observer.ready;
  return { debug, contents, observer, changed, cdp };
}
function navigate(
  contents: WebContents,
  debug: FakeDebugger,
  url = 'https://security.example/',
  loader = 'load-1',
) {
  contents.emit('did-start-navigation', {}, url, false, true);
  debug.message('Page.frameNavigated', { frame: { id: 'main', loaderId: loader, url } });
}
function response(debug: FakeDebugger, overrides: Record<string, unknown> = {}, loader = 'load-1') {
  debug.message('Network.responseReceived', {
    frameId: 'main',
    loaderId: loader,
    type: 'Document',
    requestId: 'document',
    response: {
      url: 'https://security.example/',
      securityState: 'secure',
      securityDetails: {
        protocol: 'TLS 1.3',
        cipher: 'AES_128_GCM',
        subjectName: 'security.example',
        issuer: 'security.example',
        validFrom: Date.parse(certificate.validFrom) / 1000,
        validTo: Date.parse(certificate.validTo) / 1000,
        signedCertificateTimestampList: [],
      },
      ...overrides,
    },
  });
}
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));
function mixedAudit(
  debug: FakeDebugger,
  {
    requestId,
    frameId = 'main',
    resourceType = 'Image',
    resolutionStatus = 'MixedContentAutomaticallyUpgraded',
  }: {
    requestId?: string;
    frameId?: string;
    resourceType?: string;
    resolutionStatus?: string;
  } = {},
) {
  debug.message('Audits.issueAdded', {
    issue: {
      details: {
        mixedContentIssueDetails: {
          mainResourceURL: 'https://security.example/',
          frame: { frameId },
          ...(requestId ? { request: { requestId } } : {}),
          insecureURL: 'http://insecure.example/resource',
          resourceType,
          resolutionStatus,
        },
      },
    },
  });
}
function liveFrameTree(id = 'fresh') {
  return {
    frameTree: {
      frame: { id: 'main', loaderId: 'load-1', url: 'https://security.example/' },
      childFrames: [{ frame: { id, parentId: 'main', loaderId: '', url: 'about:blank' } }],
    },
  };
}

function nativeTree(debug: FakeDebugger, tree = liveFrameTree()) {
  debug.sendCommand.mockImplementation(async (method) =>
    method === 'Page.getFrameTree'
      ? tree
      : method === 'Network.getCertificate'
        ? { tableNames: [der] }
        : {},
  );
}

describe('bounded certificate data', () => {
  it('parses real DER fields and exports the same read-only chain', () => {
    const chain = parseCertificateChain([der]);
    expect(chain[0]).toMatchObject({
      subject: certificate.subject,
      issuer: certificate.issuer,
      fingerprint256: certificate.fingerprint256,
      fingerprint: certificate.fingerprint,
      signatureAlgorithm: '1.2.840.113549.1.1.11',
      keyType: 'rsa',
      keySize: 2048,
    });
    expect(new X509Certificate(exportCertificateChain(chain)).raw).toEqual(certificate.raw);
  });
  it.each([
    Array(11).fill(der),
    ['x'.repeat(90000)],
    ['!invalid'],
    [Buffer.alloc(65537).toString('base64')],
    ['AA=='],
  ])('rejects excessive chains, oversized members and malformed DER', (input) => {
    expect(() => parseCertificateChain(input)).toThrow();
  });
});
describe('committed-document connection evidence', () => {
  it('never infers secure from a URL and observes with no response-body buffering', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    expect(observer.snapshot().state).toBe('unavailable');
    expect(debug.sendCommand).toHaveBeenCalledWith(
      'Network.enable',
      {
        maxTotalBufferSize: 0,
        maxResourceBufferSize: 0,
        maxPostDataSize: 0,
      },
      undefined,
    );
    response(debug);
    await flush();
    expect(observer.snapshot()).toMatchObject({
      state: 'secure',
      chainIncludesRoot: true,
      tls: { protocol: 'TLS 1.3' },
    });
    observer.dispose();
  });
  it('drops unrenderable remote SCT dates while retaining valid TLS and timestamp evidence', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    const representable = [-8_640_000_000_000_000, 0, 1_793_793_600_000, 8_640_000_000_000_000];
    const unrenderable = [
      8_640_000_000_000_001,
      -8_640_000_000_000_001,
      Number.MAX_VALUE,
      Number.POSITIVE_INFINITY,
      Number.NaN,
      '1793793600000',
    ];
    response(debug, {
      securityDetails: {
        protocol: 'TLS 1.3',
        cipher: 'AES_128_GCM',
        subjectName: 'security.example',
        issuer: 'security.example',
        validFrom: Date.parse(certificate.validFrom) / 1000,
        validTo: Date.parse(certificate.validTo) / 1000,
        certificateTransparencyCompliance: 'unknown',
        signedCertificateTimestampList: [...unrenderable, ...representable].map(
          (timestamp, index) => ({
            logDescription: `Remote log ${index}`,
            signatureAlgorithm: 'ECDSA',
            timestamp,
          }),
        ),
      },
    });
    await flush();
    const snapshot = observer.snapshot();
    expect(snapshot).toMatchObject({
      state: 'secure',
      tls: { protocol: 'TLS 1.3', cipher: 'AES_128_GCM', certificateTransparency: 'unknown' },
      certificates: [expect.objectContaining({ fingerprint256: certificate.fingerprint256 })],
    });
    expect(snapshot.tls!.signedCertificateTimestamps).toEqual(
      representable.map((timestamp, index) => ({
        log: `Remote log ${unrenderable.length + index}`,
        signatureAlgorithm: 'ECDSA',
        timestamp,
      })),
    );
    // Site controls uses this conversion during render. Untrusted metadata must
    // not leave an invalid Date that can throw and take down trusted chrome.
    expect(() =>
      snapshot.tls!.signedCertificateTimestamps.map((stamp) =>
        new Date(stamp.timestamp).toISOString(),
      ),
    ).not.toThrow();
    observer.dispose();
  });
  it('accepts a main-document response before commit only for its matching loader', async () => {
    const { observer, contents, debug } = await setup();
    contents.emit('did-start-navigation', {}, 'https://security.example/', false, true);
    response(debug);
    expect(observer.snapshot().state).toBe('unavailable');
    debug.message('Page.frameNavigated', {
      frame: { id: 'main', loaderId: 'load-1', url: 'https://security.example/' },
    });
    await flush();
    expect(observer.snapshot().state).toBe('secure');
    observer.dispose();
  });
  it('does not reveal the previous certificate after navigation or late certificate response', async () => {
    const { observer, contents, debug } = await setup();
    let resolve!: (value: unknown) => void;
    debug.sendCommand.mockImplementation(async (method) =>
      method === 'Network.getCertificate'
        ? new Promise((done) => {
            resolve = done;
          })
        : {},
    );
    navigate(contents, debug);
    response(debug);
    navigate(contents, debug, 'http://other.example/', 'load-2');
    resolve({ tableNames: [der] });
    await flush();
    expect(observer.snapshot()).toMatchObject({
      url: 'http://other.example/',
      state: 'not-secure',
      certificates: [],
    });
    observer.dispose();
  });
  it('refuses a certificate chain that does not match committed response details', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug, {
      securityDetails: {
        protocol: 'TLS 1.3',
        cipher: 'AES',
        subjectName: 'another.example',
        issuer: 'security.example',
        validFrom: Date.parse(certificate.validFrom) / 1000,
        validTo: Date.parse(certificate.validTo) / 1000,
      },
    });
    await flush();
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', certificates: [] });
    observer.dispose();
  });
  it('reports mixed-content disposition and never treats a blocked resource as loaded', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    debug.message('Network.requestWillBeSent', {
      requestId: 'blocked-script',
      frameId: 'main',
      loaderId: 'load-1',
      request: { url: 'http://insecure.example/script.js', mixedContentType: 'blockable' },
    });
    debug.message('Audits.issueAdded', {
      issue: {
        details: {
          mixedContentIssueDetails: {
            mainResourceURL: 'https://security.example/',
            request: { requestId: 'blocked-script' },
            insecureURL: 'http://insecure.example/script.js',
            resourceType: 'Script',
            resolutionStatus: 'MixedContentBlocked',
          },
        },
      },
    });
    expect(observer.snapshot()).toMatchObject({
      state: 'not-fully-secure',
      issues: [{ resolution: 'blocked' }],
    });
    debug.message('Network.requestWillBeSent', {
      requestId: 'mixed',
      frameId: 'main',
      loaderId: 'load-1',
      request: { url: 'http://insecure.example/image', mixedContentType: 'optionally-blockable' },
    });
    debug.message('Network.responseReceived', {
      requestId: 'mixed',
      frameId: 'main',
      type: 'Image',
      response: { url: 'http://insecure.example/image' },
    });
    expect(observer.snapshot()).toMatchObject({ state: 'not-fully-secure' });
    observer.dispose();
  });
  it.each([
    ['MixedContentAutomaticallyUpgraded', 'upgraded'],
    ['MixedContentWarning', 'warning'],
  ])(
    'keeps a valid TLS document with %s evidence Not fully secure',
    async (resolutionStatus, resolution) => {
      const { observer, contents, debug } = await setup();
      navigate(contents, debug);
      response(debug);
      await flush();
      debug.message('Network.requestWillBeSent', {
        requestId: 'mixed-image',
        frameId: 'main',
        loaderId: 'load-1',
        request: { url: 'http://insecure.example/image', mixedContentType: 'optionally-blockable' },
      });
      debug.message('Audits.issueAdded', {
        issue: {
          details: {
            mixedContentIssueDetails: {
              mainResourceURL: 'https://security.example/',
              request: { requestId: 'mixed-image' },
              insecureURL: 'http://insecure.example/image',
              resourceType: 'Image',
              resolutionStatus,
            },
          },
        },
      });
      expect(observer.snapshot()).toMatchObject({
        state: 'not-fully-secure',
        tls: { protocol: 'TLS 1.3' },
        issues: [{ resolution, url: 'http://insecure.example/image', resourceType: 'Image' }],
      });
      observer.dispose();
    },
  );
  it('invalidates all details on debugger detach and does not steal user DevTools', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    debug.attached = false;
    debug.emit('detach', {}, 'replaced_with_devtools');
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', certificates: [] });
    await expect(observer.sendCommand('Fetch.enable')).rejects.toThrow('UNAVAILABLE');
    observer.dispose();
    const external = await setup(true);
    expect(external.debug.attach).not.toHaveBeenCalled();
    external.observer.dispose();
    expect(external.debug.detach).not.toHaveBeenCalled();
  });
  it('shares only owned iframe debugger sessions with the form observer', async () => {
    const { observer, contents, debug, cdp } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    nativeTree(debug);
    debug.message('Target.attachedToTarget', {
      sessionId: 'child',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'fresh' },
    });
    await flush();
    debug.message('Page.frameRequestedNavigation', { reason: 'formSubmissionPost' }, 'child');
    expect(cdp).toHaveBeenCalledWith(
      'Page.frameRequestedNavigation',
      { reason: 'formSubmissionPost' },
      'child',
    );
    await observer.sendCommand('Fetch.enable', {}, 'child');
    await expect(observer.sendCommand('Fetch.enable', {}, 'foreign')).rejects.toThrow(
      'UNKNOWN_DEBUGGER_SESSION',
    );
    observer.dispose();
  });
  it('shows failed TLS details only on the matching trusted error document without trusting them', async () => {
    const { observer, contents, debug } = await setup();
    contents.emit('did-start-navigation', {}, 'https://security.example/', false, true);
    const native = { data: pem } as Certificate;
    observer.recordCertificateError('https://security.example/', native);
    expect(observer.snapshot()).toMatchObject({
      state: 'unavailable',
      failedURL: 'https://security.example/',
      certificates: [expect.objectContaining({ subject: certificate.subject })],
    });
    const errorURL =
      'oryn://network-error/?url=https%3A%2F%2Fsecurity.example%2F&code=-202&description=ERR_CERT_AUTHORITY_INVALID';
    navigate(contents, debug, errorURL, 'error-loader');
    expect(observer.snapshot()).toMatchObject({
      state: 'unavailable',
      failedURL: 'https://security.example/',
      certificates: [expect.objectContaining({ subject: certificate.subject })],
    });
    navigate(contents, debug, 'https://different.example/', 'other-loader');
    expect(observer.snapshot().certificates).toEqual([]);
    observer.recordCertificateError('https://security.example/', native);
    expect(observer.snapshot().certificates).toEqual([]);
    observer.dispose();
  });
  it('does not retain a failed certificate for a forged or unrelated error-page target', async () => {
    const { observer, contents, debug } = await setup();
    contents.emit('did-start-navigation', {}, 'https://security.example/', false, true);
    observer.recordCertificateError('https://security.example/', { data: pem } as Certificate);
    navigate(
      contents,
      debug,
      'oryn://network-error/?url=https%3A%2F%2Fforeign.example%2F&code=-202&description=ERR_CERT',
      'error-loader',
    );
    expect(observer.snapshot().certificates).toEqual([]);
    observer.dispose();
  });
  it('recovers a matching first-navigation response while observer initialization is still pending', async () => {
    const debug = new FakeDebugger();
    const contents = Object.assign(new EventEmitter(), {
      debugger: debug,
      isDestroyed: () => false,
      getURL: () => 'https://security.example/',
    }) as unknown as WebContents;
    const observer = new ConnectionSecurityObserver(contents, vi.fn());
    navigate(contents, debug);
    response(debug);
    await observer.ready;
    await flush();
    expect(observer.snapshot().state).toBe('secure');
    observer.dispose();
  });
  it('does not attach stale same-URL audit details to a replacement document', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    debug.message('Network.requestWillBeSent', {
      requestId: 'old-request',
      frameId: 'main',
      loaderId: 'load-1',
      request: { url: 'http://old.example/script', mixedContentType: 'blockable' },
    });
    navigate(contents, debug, 'https://security.example/', 'replacement');
    response(debug, {}, 'replacement');
    await flush();
    debug.message('Audits.issueAdded', {
      issue: {
        details: {
          mixedContentIssueDetails: {
            mainResourceURL: 'https://security.example/',
            request: { requestId: 'old-request' },
            insecureURL: 'http://old.example/script',
            resolutionStatus: 'MixedContentBlocked',
          },
        },
      },
    });
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
    observer.dispose();
  });
  it('finishes the optional child guard hook before resuming a paused iframe', async () => {
    const debug = new FakeDebugger();
    const contents = Object.assign(new EventEmitter(), {
      debugger: debug,
      isDestroyed: () => false,
      getURL: () => 'about:blank',
    }) as unknown as WebContents;
    const beforeChildResume = vi.fn(
      async (
        sessionId: string,
        send: (method: string, params?: Record<string, unknown>) => Promise<unknown>,
      ) => {
        expect(sessionId).toBe('child');
        await send('Fetch.enable', { patterns: [] });
      },
    );
    const observer = new ConnectionSecurityObserver(contents, vi.fn(), { beforeChildResume });
    await observer.ready;
    navigate(contents, debug);
    response(debug);
    await flush();
    nativeTree(debug);
    debug.message('Target.attachedToTarget', {
      sessionId: 'child',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'fresh' },
    });
    await flush();
    expect(beforeChildResume).toHaveBeenCalledOnce();
    const methods = debug.sendCommand.mock.calls.map((call) => call[0]);
    expect(methods.indexOf('Fetch.enable')).toBeLessThan(
      methods.indexOf('Runtime.runIfWaitingForDebugger'),
    );
    expect(debug.sendCommand).toHaveBeenCalledWith('Fetch.enable', { patterns: [] }, 'child');
    observer.dispose();
  });
  it('rejects late old OOPIF data before it can seed a replacement document frame, request or audit', async () => {
    const { observer, contents, debug, cdp } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    nativeTree(debug);
    debug.message('Target.attachedToTarget', {
      sessionId: 'old-child',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'fresh' },
    });
    await flush();
    navigate(contents, debug, 'https://security.example/', 'replacement');
    response(debug, {}, 'replacement');
    await flush();
    cdp.mockClear();
    debug.message(
      'Page.frameNavigated',
      {
        frame: {
          id: 'fresh',
          parentId: 'main',
          loaderId: 'old-child-loader',
          url: 'https://frame.example/',
        },
      },
      'old-child',
    );
    debug.message(
      'Network.requestWillBeSent',
      {
        requestId: 'retired-request',
        frameId: 'fresh',
        loaderId: 'old-child-loader',
        request: {
          url: 'http://insecure.example/stale-secret-path',
          mixedContentType: 'blockable',
        },
      },
      'old-child',
    );
    debug.message(
      'Audits.issueAdded',
      {
        issue: {
          details: {
            mixedContentIssueDetails: {
              mainResourceURL: 'https://security.example/',
              frame: { frameId: 'fresh' },
              request: { requestId: 'retired-request' },
              insecureURL: 'http://insecure.example/stale-secret-path',
              resourceType: 'Script',
              resolutionStatus: 'MixedContentBlocked',
            },
          },
        },
      },
      'old-child',
    );
    expect(observer.snapshot()).toMatchObject({ state: 'secure', issues: [] });
    expect(cdp).not.toHaveBeenCalled();
    await expect(observer.sendCommand('Fetch.enable', {}, 'old-child')).rejects.toThrow(
      'UNKNOWN_DEBUGGER_SESSION',
    );
    observer.dispose();
  });
  it.each(['missing-target', 'wrong-loader', 'wrong-parent'])(
    'does not activate an owned target with a %s native main-frame tree',
    async (tamper) => {
      const { observer, contents, debug, cdp } = await setup();
      navigate(contents, debug);
      response(debug);
      await flush();
      const tree = liveFrameTree();
      if (tamper === 'missing-target') tree.frameTree.childFrames = [];
      if (tamper === 'wrong-loader') tree.frameTree.frame.loaderId = 'retired-loader';
      if (tamper === 'wrong-parent') tree.frameTree.childFrames[0]!.frame.parentId = 'foreign';
      nativeTree(debug, tree);
      debug.message('Target.attachedToTarget', {
        sessionId: 'ambiguous-child',
        waitingForDebugger: true,
        targetInfo: { type: 'iframe', targetId: 'fresh' },
      });
      await flush();
      expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
      expect(debug.sendCommand).toHaveBeenCalledWith(
        'Runtime.runIfWaitingForDebugger',
        {},
        'ambiguous-child',
      );
      cdp.mockClear();
      debug.message(
        'Page.frameNavigated',
        {
          frame: {
            id: 'fresh',
            parentId: 'main',
            loaderId: 'unverified-loader',
            url: 'https://frame.example/',
          },
        },
        'ambiguous-child',
      );
      expect(cdp).not.toHaveBeenCalled();
      await expect(observer.sendCommand('Fetch.enable', {}, 'ambiguous-child')).rejects.toThrow(
        'UNKNOWN_DEBUGGER_SESSION',
      );
      observer.dispose();
    },
  );
  it('resumes an owned paused target after navigation invalidates its asynchronous membership proof', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    let resolveTree!: (value: unknown) => void;
    debug.sendCommand.mockImplementation(async (method) =>
      method === 'Page.getFrameTree'
        ? new Promise((done) => {
            resolveTree = done;
          })
        : method === 'Network.getCertificate'
          ? { tableNames: [der] }
          : {},
    );
    debug.message('Target.attachedToTarget', {
      sessionId: 'paused-old-child',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'fresh' },
    });
    expect(observer.snapshot().state).toBe('unavailable');
    navigate(contents, debug, 'https://security.example/', 'replacement');
    response(debug, {}, 'replacement');
    await flush();
    resolveTree(liveFrameTree());
    await flush();
    expect(observer.snapshot()).toMatchObject({ state: 'secure', issues: [] });
    expect(debug.sendCommand).toHaveBeenCalledWith(
      'Runtime.runIfWaitingForDebugger',
      {},
      'paused-old-child',
    );
    await expect(observer.sendCommand('Fetch.enable', {}, 'paused-old-child')).rejects.toThrow(
      'UNKNOWN_DEBUGGER_SESSION',
    );
    observer.dispose();
  });
  it('resumes timed-out targets and prevents their late native proof from enabling data domains', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    let resolveTree!: (value: unknown) => void;
    debug.sendCommand.mockImplementation(async (method) =>
      method === 'Page.getFrameTree'
        ? new Promise((done) => {
            resolveTree = done;
          })
        : {},
    );
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      debug.message('Target.attachedToTarget', {
        sessionId: 'timed-out-child',
        waitingForDebugger: true,
        targetInfo: { type: 'iframe', targetId: 'fresh' },
      });
      await vi.advanceTimersByTimeAsync(2001);
      expect(debug.sendCommand).toHaveBeenCalledWith(
        'Runtime.runIfWaitingForDebugger',
        {},
        'timed-out-child',
      );
      resolveTree(liveFrameTree());
      await flush();
      expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
      expect(debug.sendCommand).not.toHaveBeenCalledWith(
        'Network.enable',
        expect.anything(),
        'timed-out-child',
      );
      await expect(observer.sendCommand('Fetch.enable', {}, 'timed-out-child')).rejects.toThrow(
        'UNKNOWN_DEBUGGER_SESSION',
      );
    } finally {
      vi.useRealTimers();
      observer.dispose();
    }
  });
  it('resumes a paused target rejected by the owned-session limit without accepting its data', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    nativeTree(debug);
    for (let index = 0; index < 129; index++)
      debug.message('Target.attachedToTarget', {
        sessionId: `child-${index}`,
        waitingForDebugger: true,
        targetInfo: { type: 'iframe', targetId: `unverified-${index}` },
      });
    await flush();
    expect(debug.sendCommand).toHaveBeenCalledWith(
      'Runtime.runIfWaitingForDebugger',
      {},
      'child-128',
    );
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
    await expect(observer.sendCommand('Fetch.enable', {}, 'child-128')).rejects.toThrow(
      'UNKNOWN_DEBUGGER_SESSION',
    );
    observer.dispose();
  });
  it('does not accept a late root attach for a former child absent from the replacement native tree', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    nativeTree(debug);
    debug.message('Target.attachedToTarget', {
      sessionId: 'old-child',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'fresh' },
    });
    await flush();
    navigate(contents, debug, 'https://security.example/', 'replacement');
    response(debug, {}, 'replacement');
    await flush();
    const tree = liveFrameTree();
    tree.frameTree.frame.loaderId = 'replacement';
    tree.frameTree.childFrames = [];
    nativeTree(debug, tree);
    debug.message('Target.attachedToTarget', {
      sessionId: 'late-former-child',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'fresh' },
    });
    await flush();
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
    expect(debug.sendCommand).toHaveBeenCalledWith(
      'Runtime.runIfWaitingForDebugger',
      {},
      'late-former-child',
    );
    await expect(observer.sendCommand('Fetch.enable', {}, 'late-former-child')).rejects.toThrow(
      'UNKNOWN_DEBUGGER_SESSION',
    );
    observer.dispose();
  });
  it('corroborates out-of-order audits by exact current request/loader and deduplicates only proven current issues', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    mixedAudit(debug, { requestId: 'speculative-before-matched-audit' });
    mixedAudit(debug, { requestId: 'image-2' });
    expect(observer.snapshot()).toMatchObject({
      state: 'unavailable',
      tls: { protocol: 'TLS 1.3' },
      certificates: [expect.objectContaining({ fingerprint256: certificate.fingerprint256 })],
      issues: [],
    });
    debug.message('Network.requestWillBeSent', {
      requestId: 'image-2',
      frameId: 'main',
      loaderId: 'load-1',
      request: { url: 'https://insecure.example/resource', mixedContentType: 'none' },
    });
    expect(observer.snapshot()).toMatchObject({
      state: 'not-fully-secure',
      issues: [{ resolution: 'upgraded' }],
    });
    mixedAudit(debug, { requestId: 'preload-without-network-event' });
    expect(observer.snapshot()).toMatchObject({
      state: 'not-fully-secure',
      issues: [{ resolution: 'upgraded' }],
    });
    mixedAudit(debug, { requestId: 'unknown-frame', frameId: 'foreign' });
    expect(observer.snapshot().state).toBe('unavailable');
    observer.dispose();
  });
  it('does not match pending audit IDs against another loader or retain them across same-URL replacement', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    mixedAudit(debug, { requestId: 'old-loader-request' });
    debug.message('Network.requestWillBeSent', {
      requestId: 'old-loader-request',
      frameId: 'main',
      loaderId: 'prior-loader',
      request: { url: 'https://insecure.example/resource' },
    });
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
    navigate(contents, debug, 'https://security.example/', 'replacement');
    response(debug, {}, 'replacement');
    await flush();
    expect(observer.snapshot()).toMatchObject({ state: 'secure', issues: [] });
    debug.message('Network.requestWillBeSent', {
      requestId: 'old-loader-request',
      frameId: 'main',
      loaderId: 'prior-loader',
      request: { url: 'https://insecure.example/resource' },
    });
    mixedAudit(debug, { requestId: 'old-loader-request' });
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
    observer.dispose();
  });
  it('cannot satisfy an iframe audit with an identically named request from another debugger session', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    debug.message('Page.frameAttached', { frameId: 'child-frame', parentFrameId: 'main' });
    nativeTree(debug, liveFrameTree('child-frame'));
    debug.message('Target.attachedToTarget', {
      sessionId: 'child-session',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'child-frame' },
    });
    await flush();
    debug.message(
      'Page.frameNavigated',
      {
        frame: {
          id: 'child-frame',
          parentId: 'main',
          loaderId: 'child-loader',
          url: 'https://frame.example/',
        },
      },
      'child-session',
    );
    debug.message(
      'Audits.issueAdded',
      {
        issue: {
          details: {
            mixedContentIssueDetails: {
              mainResourceURL: 'https://frame.example/',
              frame: { frameId: 'child-frame' },
              request: { requestId: 'shared-id' },
              insecureURL: 'http://insecure.example/resource',
              resourceType: 'Image',
              resolutionStatus: 'MixedContentAutomaticallyUpgraded',
            },
          },
        },
      },
      'child-session',
    );
    const request = {
      requestId: 'shared-id',
      frameId: 'child-frame',
      loaderId: 'child-loader',
      request: { url: 'https://insecure.example/resource' },
    };
    debug.message('Network.requestWillBeSent', request);
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
    debug.message('Network.requestWillBeSent', request, 'child-session');
    expect(observer.snapshot()).toMatchObject({
      state: 'not-fully-secure',
      issues: [{ resolution: 'upgraded' }],
    });
    observer.dispose();
  });
  it('corroborates a requestless blocked frame only through its fresh live native parent chain and current main loader', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    debug.sendCommand.mockImplementation(async (method) =>
      method === 'Page.getFrameTree' ? liveFrameTree() : {},
    );
    debug.message('Page.frameAttached', { frameId: 'fresh', parentFrameId: 'main' });
    mixedAudit(debug, {
      frameId: 'fresh',
      resourceType: 'Frame',
      resolutionStatus: 'MixedContentBlocked',
    });
    expect(observer.snapshot().state).toBe('unavailable');
    await flush();
    expect(observer.snapshot()).toMatchObject({
      state: 'not-fully-secure',
      issues: [{ resourceType: 'Frame', resolution: 'blocked' }],
    });
    observer.dispose();
  });
  it.each(['wrong-main-loader', 'wrong-parent', 'detached', 'committed', 'expired', 'unknown'])(
    'keeps requestless %s frame evidence unavailable without erasing the verified document certificate',
    async (tamper) => {
      const { observer, contents, debug } = await setup();
      navigate(contents, debug);
      response(debug);
      await flush();
      let now = 10_000;
      const clock = vi.spyOn(Date, 'now').mockImplementation(() => now);
      try {
        let resolveTree!: (value: unknown) => void;
        debug.sendCommand.mockImplementation(async (method) =>
          method === 'Page.getFrameTree'
            ? new Promise((done) => {
                resolveTree = done;
              })
            : {},
        );
        if (tamper !== 'unknown')
          debug.message('Page.frameAttached', { frameId: 'fresh', parentFrameId: 'main' });
        if (tamper === 'committed')
          debug.message('Page.frameNavigated', {
            frame: {
              id: 'fresh',
              parentId: 'main',
              loaderId: 'child-loader',
              url: 'https://frame.example/',
            },
          });
        if (tamper === 'expired') now += 2001;
        mixedAudit(debug, {
          frameId: 'fresh',
          resourceType: 'Frame',
          resolutionStatus: 'MixedContentBlocked',
        });
        if (resolveTree) {
          const tree = liveFrameTree();
          if (tamper === 'wrong-main-loader') tree.frameTree.frame.loaderId = 'old-loader';
          if (tamper === 'wrong-parent') tree.frameTree.childFrames[0]!.frame.parentId = 'foreign';
          if (tamper === 'detached') debug.message('Page.frameDetached', { frameId: 'fresh' });
          resolveTree(tree);
        }
        await flush();
        expect(observer.snapshot()).toMatchObject({
          state: 'unavailable',
          certificates: [expect.objectContaining({ fingerprint256: certificate.fingerprint256 })],
          issues: [],
        });
      } finally {
        clock.mockRestore();
        observer.dispose();
      }
    },
  );
  it('cannot apply an asynchronous old frame-tree proof to a replacement document', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    let resolveTree!: (value: unknown) => void;
    debug.sendCommand.mockImplementation(async (method) =>
      method === 'Page.getFrameTree'
        ? new Promise((done) => {
            resolveTree = done;
          })
        : method === 'Network.getCertificate'
          ? { tableNames: [der] }
          : {},
    );
    debug.message('Page.frameAttached', { frameId: 'fresh', parentFrameId: 'main' });
    mixedAudit(debug, {
      frameId: 'fresh',
      resourceType: 'Frame',
      resolutionStatus: 'MixedContentBlocked',
    });
    navigate(contents, debug, 'https://security.example/', 'replacement');
    response(debug, {}, 'replacement');
    await flush();
    resolveTree(liveFrameTree());
    await flush();
    expect(observer.snapshot()).toMatchObject({ state: 'secure', issues: [] });
    observer.dispose();
  });
  it('keeps an excessive unresolved native issue burst unavailable', async () => {
    const { observer, contents, debug } = await setup();
    navigate(contents, debug);
    response(debug);
    await flush();
    for (let index = 0; index < 101; index++)
      mixedAudit(debug, { requestId: `unmatched-${index}` });
    expect(observer.snapshot()).toMatchObject({ state: 'unavailable', issues: [] });
    observer.dispose();
  });
  it('attributes a requestless Form warning only to a native current target/session/main loader', async () => {
    const { observer, contents, debug } = await setup();
    const url = 'https://security.example/top-document';
    navigate(contents, debug, url);
    response(debug, { url });
    await flush();
    const tree = liveFrameTree();
    tree.frameTree.frame.url = url;
    nativeTree(debug, tree);
    debug.message('Target.attachedToTarget', {
      sessionId: 'form-child',
      waitingForDebugger: true,
      targetInfo: { type: 'iframe', targetId: 'fresh' },
    });
    await flush();
    Object.assign(tree.frameTree.childFrames[0]!.frame, {
      loaderId: 'form-loader',
      url: 'https://frame.example/form',
    });
    debug.message(
      'Page.frameNavigated',
      { frame: tree.frameTree.childFrames[0]!.frame },
      'form-child',
    );
    debug.message(
      'Audits.issueAdded',
      {
        issue: {
          details: {
            mixedContentIssueDetails: {
              mainResourceURL: 'https://security.example/',
              frame: { frameId: 'fresh' },
              insecureURL: 'http://insecure.example/action',
              resourceType: 'Form',
              resolutionStatus: 'MixedContentWarning',
            },
          },
        },
      },
      'form-child',
    );
    expect(observer.snapshot().state).toBe('unavailable');
    await flush();
    expect(observer.snapshot()).toMatchObject({
      state: 'not-fully-secure',
      url,
      tls: { protocol: 'TLS 1.3' },
      issues: [
        { resourceType: 'Form', resolution: 'warning', url: 'http://insecure.example/action' },
      ],
    });
    observer.dispose();
  });
  it.each(['main-loader', 'target-loader', 'session-detach', 'main-replacement'])(
    'does not attribute requestless Form metadata after %s changes during native verification',
    async (tamper) => {
      const { observer, contents, debug } = await setup();
      navigate(contents, debug);
      response(debug);
      await flush();
      const tree = liveFrameTree();
      nativeTree(debug, tree);
      debug.message('Target.attachedToTarget', {
        sessionId: 'form-child',
        waitingForDebugger: true,
        targetInfo: { type: 'iframe', targetId: 'fresh' },
      });
      await flush();
      Object.assign(tree.frameTree.childFrames[0]!.frame, {
        loaderId: 'form-loader',
        url: 'https://frame.example/form',
      });
      debug.message(
        'Page.frameNavigated',
        { frame: tree.frameTree.childFrames[0]!.frame },
        'form-child',
      );
      let resolveTree!: (value: unknown) => void;
      debug.sendCommand.mockImplementation(async (method) =>
        method === 'Page.getFrameTree'
          ? new Promise((done) => {
              resolveTree = done;
            })
          : method === 'Network.getCertificate'
            ? { tableNames: [der] }
            : {},
      );
      debug.message(
        'Audits.issueAdded',
        {
          issue: {
            details: {
              mixedContentIssueDetails: {
                mainResourceURL: 'https://security.example/',
                frame: { frameId: 'fresh' },
                insecureURL: 'http://insecure.example/action',
                resourceType: 'Form',
                resolutionStatus: 'MixedContentWarning',
              },
            },
          },
        },
        'form-child',
      );
      if (tamper === 'main-loader') tree.frameTree.frame.loaderId = 'different-main';
      if (tamper === 'target-loader')
        tree.frameTree.childFrames[0]!.frame.loaderId = 'different-child';
      if (tamper === 'session-detach')
        debug.message('Target.detachedFromTarget', { sessionId: 'form-child' });
      if (tamper === 'main-replacement') {
        navigate(contents, debug, 'https://security.example/', 'replacement');
        response(debug, {}, 'replacement');
        await flush();
      }
      resolveTree(tree);
      await flush();
      expect(observer.snapshot()).toMatchObject({
        state: tamper === 'main-replacement' ? 'secure' : 'unavailable',
        issues: [],
      });
      observer.dispose();
    },
  );
  it('settles stalled initial observation without reviving or enabling later domains after its deadline', async () => {
    const debug = new FakeDebugger();
    let resolvePage!: (value: unknown) => void;
    debug.sendCommand.mockImplementation(async (method) =>
      method === 'Page.enable'
        ? new Promise((done) => {
            resolvePage = done;
          })
        : {},
    );
    const contents = Object.assign(new EventEmitter(), {
      debugger: debug,
      isDestroyed: () => false,
      getURL: () => 'about:blank',
    }) as unknown as WebContents;
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const observer = new ConnectionSecurityObserver(contents, vi.fn());
    try {
      await vi.advanceTimersByTimeAsync(2001);
      await observer.ready;
      expect(observer.snapshot().state).toBe('unavailable');
      expect(debug.detach).toHaveBeenCalledOnce();
      resolvePage({});
      await flush();
      expect(observer.isAvailable).toBe(false);
      expect(debug.sendCommand.mock.calls.map((call) => call[0])).toEqual(['Page.enable']);
    } finally {
      vi.useRealTimers();
      observer.dispose();
    }
  });
  it.each(['late-tree', 'late-error'])(
    'does not process %s after initial observation was revoked',
    async (result) => {
      const debug = new FakeDebugger();
      let resolveTree!: (value: unknown) => void;
      let rejectTree!: (error: Error) => void;
      debug.sendCommand.mockImplementation(async (method) =>
        method === 'Page.getFrameTree'
          ? new Promise((done, reject) => {
              resolveTree = done;
              rejectTree = reject;
            })
          : {},
      );
      const contents = Object.assign(new EventEmitter(), {
        debugger: debug,
        isDestroyed: () => false,
        getURL: () => 'about:blank',
      }) as unknown as WebContents;
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      const observer = new ConnectionSecurityObserver(contents, vi.fn());
      try {
        await flush();
        await vi.advanceTimersByTimeAsync(2001);
        await observer.ready;
        const before = observer.snapshot();
        const readTree = vi.fn(() => ({
          frame: { id: 'stale-main', loaderId: 'retired', url: 'about:blank' },
        }));
        if (result === 'late-error') rejectTree(new Error('LATE_NATIVE_ERROR'));
        else
          resolveTree({
            get frameTree() {
              return readTree();
            },
          });
        await flush();
        expect(readTree).not.toHaveBeenCalled();
        expect(observer.isAvailable).toBe(false);
        expect(observer.snapshot().state).toBe('unavailable');
        expect(observer.snapshot()).toEqual(before);
      } finally {
        vi.useRealTimers();
        observer.dispose();
      }
    },
  );
  it('never queries a destroyed native debugger or interrupts later destroyed listeners', async () => {
    const { observer, contents } = await setup();
    const debuggerGetter = vi.fn(() => {
      throw new Error('Object has been destroyed');
    });
    Object.assign(contents, { isDestroyed: () => true });
    Object.defineProperty(contents, 'debugger', { get: debuggerGetter, configurable: true });
    const closed = vi.fn();
    contents.once('destroyed', closed);
    expect(observer.isAvailable).toBe(false);
    expect(() => contents.emit('destroyed')).not.toThrow();
    expect(closed).toHaveBeenCalledOnce();
    expect(debuggerGetter).not.toHaveBeenCalled();
    expect(() => observer.dispose()).not.toThrow();
  });
  it('contains native teardown races even before isDestroyed becomes true', async () => {
    const { observer, contents, debug } = await setup();
    vi.spyOn(debug, 'isAttached').mockImplementation(() => {
      throw new Error('Object has been destroyed');
    });
    const closed = vi.fn();
    contents.once('destroyed', closed);
    expect(() => contents.emit('destroyed')).not.toThrow();
    expect(closed).toHaveBeenCalledOnce();
    expect(observer.isAvailable).toBe(false);
    expect(() => observer.dispose()).not.toThrow();
  });
});
