const MAX_IDENTITY_SEGMENT_LENGTH = 96;
const IDENTITY_SEGMENT = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/u;
const REVISION_SEGMENT = /^[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,198}[A-Za-z0-9])?$/u;

const RESERVED_NAMESPACES = new Set([
  'api',
  'datasets',
  'docs',
  'enterprise',
  'join',
  'login',
  'models',
  'organizations',
  'pricing',
  'search',
  'settings',
  'spaces',
]);

export interface HuggingFaceModelPage {
  repository: string;
  revision?: string;
  sourceURL: string;
  route: 'root' | 'tree' | 'blob';
}

/**
 * Parses only browser-committed, canonical Hugging Face model repository URLs.
 * This intentionally does not attempt to infer a repository from arbitrary HF pages.
 */
export function parseHuggingFaceModelPage(value: string): HuggingFaceModelPage | undefined {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return undefined;
  }
  if (
    url.protocol !== 'https:' ||
    url.hostname !== 'huggingface.co' ||
    url.username ||
    url.password ||
    url.port
  ) {
    return undefined;
  }

  const rawSegments = url.pathname.split('/').slice(1);
  if (rawSegments.at(-1) === '') rawSegments.pop();
  const segments = rawSegments.map(decodeSegment);
  if (segments.some((segment) => segment === undefined)) return undefined;
  const [owner, name, route, revision, ...tail] = segments as string[];
  if (!owner || !name || RESERVED_NAMESPACES.has(owner.toLowerCase())) return undefined;
  if (!validIdentity(owner) || !validIdentity(name)) return undefined;

  const repository = `${owner}/${name}`;
  if (!route) return { repository, sourceURL: canonicalSourceURL(repository), route: 'root' };
  if (route === 'tree' && revision && tail.length === 0 && validRevision(revision)) {
    return {
      repository,
      revision,
      sourceURL: canonicalSourceURL(repository, 'tree', revision),
      route,
    };
  }
  if (route === 'blob' && revision && tail.length > 0 && validRevision(revision)) {
    return {
      repository,
      revision,
      sourceURL: canonicalSourceURL(repository, 'blob', revision, tail),
      route,
    };
  }
  return undefined;
}

function decodeSegment(value: string): string | undefined {
  try {
    const decoded = decodeURIComponent(value);
    if (
      !decoded ||
      decoded === '.' ||
      decoded === '..' ||
      decoded.includes('/') ||
      decoded.includes('\\') ||
      [...decoded].some((character) => {
        const codePoint = character.codePointAt(0) ?? 0;
        return codePoint <= 0x1f || codePoint === 0x7f;
      })
    ) {
      return undefined;
    }
    return decoded;
  } catch {
    return undefined;
  }
}

function validIdentity(value: string): boolean {
  return value.length <= MAX_IDENTITY_SEGMENT_LENGTH && IDENTITY_SEGMENT.test(value);
}

function validRevision(value: string): boolean {
  return value.length <= 200 && REVISION_SEGMENT.test(value) && !value.includes('..');
}

function canonicalSourceURL(
  repository: string,
  route?: 'tree' | 'blob',
  revision?: string,
  tail: string[] = [],
): string {
  const parts = repository.split('/').map(encodeURIComponent);
  if (route && revision)
    parts.push(route, encodeURIComponent(revision), ...tail.map(encodeURIComponent));
  return `https://huggingface.co/${parts.join('/')}`;
}
