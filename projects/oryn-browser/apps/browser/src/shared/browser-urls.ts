export const ORYN_SCHEME = 'oryn';
export const ORYN_NEW_TAB_URL = `${ORYN_SCHEME}://new-tab/`;
export const ORYN_NETWORK_ERROR_HOST = 'network-error';
export const ORYN_MODEL_WORKSPACE_HOST = 'model';
const OPAQUE_WORKSPACE_REFERENCE = /^[A-Za-z0-9_-]{32,64}$/u;

export interface OrynNetworkErrorDetails {
  url: string;
  code: number;
  description: string;
}

const LEGACY_DEVELOPMENT_START_URLS = new Set([
  'https://example.com/',
  'https://www.example.com/',
  'https://www.iana.org/help/example-domains',
  'https://www.iana.org/help/example-domains/',
]);

export function isOrynNewTabURL(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === `${ORYN_SCHEME}:` && url.hostname === 'new-tab';
  } catch {
    return false;
  }
}

export function createOrynModelWorkspaceURL(reference: string): string {
  if (!OPAQUE_WORKSPACE_REFERENCE.test(reference)) throw new Error('INVALID_WORKSPACE_REFERENCE');
  return `${ORYN_SCHEME}://${ORYN_MODEL_WORKSPACE_HOST}/${reference}`;
}

export function readOrynModelWorkspaceReference(value: string): string | undefined {
  try {
    const url = new URL(value);
    if (
      url.protocol !== `${ORYN_SCHEME}:` ||
      url.hostname !== ORYN_MODEL_WORKSPACE_HOST ||
      url.username ||
      url.password ||
      url.port ||
      url.search ||
      url.hash
    )
      return undefined;
    const segments = url.pathname.split('/').filter(Boolean);
    return segments.length === 1 && OPAQUE_WORKSPACE_REFERENCE.test(segments[0] as string)
      ? segments[0]
      : undefined;
  } catch {
    return undefined;
  }
}

export function createOrynNetworkErrorURL(details: OrynNetworkErrorDetails): string {
  const url = new URL(`${ORYN_SCHEME}://${ORYN_NETWORK_ERROR_HOST}/`);
  url.searchParams.set('url', details.url);
  url.searchParams.set('code', String(details.code));
  url.searchParams.set('description', details.description);
  return url.toString();
}

export function readOrynNetworkErrorURL(value: string): OrynNetworkErrorDetails | undefined {
  try {
    const url = new URL(value);
    if (url.protocol !== `${ORYN_SCHEME}:` || url.hostname !== ORYN_NETWORK_ERROR_HOST) {
      return undefined;
    }
    const failedURL = url.searchParams.get('url');
    const code = Number(url.searchParams.get('code'));
    const description = url.searchParams.get('description');
    if (!failedURL || !Number.isInteger(code) || !description) return undefined;
    const parsedFailedURL = new URL(failedURL);
    if (!['http:', 'https:', 'file:'].includes(parsedFailedURL.protocol)) return undefined;
    return { url: parsedFailedURL.toString(), code, description };
  } catch {
    return undefined;
  }
}

export function migrateLegacyDevelopmentStartURL(value: string): string {
  try {
    const url = new URL(value);
    url.hash = '';
    return LEGACY_DEVELOPMENT_START_URLS.has(url.toString()) ? ORYN_NEW_TAB_URL : value;
  } catch {
    return value;
  }
}
