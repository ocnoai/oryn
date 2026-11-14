import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import specification from '../../../../../packaging/runtime-spec.json';
import type { EngineVersion, EngineVersionStatus } from '../../shared/engine-versions';

type RecordValue = Record<string, unknown>;
const record = (value: unknown): RecordValue =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as RecordValue)
    : {};
const string = (value: unknown): string | undefined =>
  typeof value === 'string' && value.length <= 160 ? value : undefined;

// Only final releases (plus PyPI post releases) are comparable. Do not compare
// prereleases, arbitrary tags, or revision hashes lexicographically.
function versionParts(value: string): number[] | undefined {
  const match = /^v?(\d+(?:\.\d+){1,3})(?:\.post(\d+))?$/u.exec(value);
  if (!match) return undefined;
  const parts = match[1]!.split('.').map(Number);
  while (parts.length < 4) parts.push(0);
  parts.push(Number(match[2] ?? 0));
  return parts.every(Number.isSafeInteger) ? parts : undefined;
}

export function compareVersions(current: string, latest: string): number | undefined {
  const a = versionParts(current.replace(/-electron\.\d+$/u, ''));
  const b = versionParts(latest);
  if (!a || !b) return undefined;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return Math.sign(a[i]! - b[i]!);
  }
  return 0;
}

function greatest(versions: string[]): string {
  const latest = versions.filter((v) => versionParts(v)).sort((a, b) => compareVersions(b, a)!)[0];
  if (!latest) throw new Error('No stable release was returned by the publisher.');
  return latest;
}

function comparisonStatus(current: string | null, latest: string): EngineVersionStatus {
  const comparison = current === null ? undefined : compareVersions(current, latest);
  return comparison === undefined
    ? 'unknown'
    : comparison < 0
      ? 'update'
      : comparison > 0
        ? 'ahead'
        : 'current';
}

const PYPI = new Map([
  ['torch', 'PyTorch'],
  ['transformers', 'Transformers'],
  ['onnxruntime', 'ONNX Runtime'],
  ['mlx', 'MLX'],
  ['mlx-lm', 'MLX-LM'],
  ['mlx-vlm', 'MLX-VLM'],
  ['diffusers', 'Diffusers'],
  ['mflux', 'MFLUX'],
  ['qwen-tts', 'Qwen-TTS'],
]);
const ELECTRON_URL = 'https://releases.electronjs.org/releases.json';
const PYTHON_URL = 'https://www.python.org/api/v2/downloads/release/?is_published=true';
const ACE_REPO = 'https://api.github.com/repos/ace-step/ACE-Step-1.5';
const LLAMA_REPO = 'https://api.github.com/repos/ggml-org/llama.cpp';
const MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

export class EngineVersionService {
  private inventory?: Promise<EngineVersion[]>;
  private manifest: RecordValue = {};
  private readonly results = new Map<string, EngineVersion>();
  private readonly pending = new Map<string, Promise<EngineVersion>>();
  private readonly responses = new Map<string, { at: number; value: Promise<unknown> }>();

  constructor(
    private readonly options: {
      packaged: boolean;
      resourcesPath: string;
      browserVersions: Record<string, string | undefined>;
      fetch?: typeof fetch;
      now?: () => number;
      readManifest?: () => Promise<unknown>;
    },
  ) {}

  private now() {
    return this.options.now?.() ?? Date.now();
  }

  async list(): Promise<EngineVersion[]> {
    this.inventory ??= this.loadInventory().catch((error: unknown) => {
      this.inventory = undefined;
      throw error;
    });
    return (await this.inventory).map((entry) => ({ ...entry, ...this.results.get(entry.id) }));
  }

  async check(id: unknown): Promise<EngineVersion> {
    if (typeof id !== 'string') throw new Error('INVALID_ENGINE_ID');
    const entry = (await this.list()).find((item) => item.id === id);
    if (!entry) throw new Error('INVALID_ENGINE_ID');
    if (!entry.checkable) return entry;
    const existing = this.pending.get(id);
    if (existing) return existing;
    const task = this.checkEntry(entry)
      .catch((error: unknown): EngineVersion => ({
        ...entry,
        status: 'error',
        message:
          error instanceof Error && error.message.startsWith('Publisher ')
            ? error.message
            : 'Could not check the release source. Check your connection and try again.',
      }))
      .then((result) => {
        this.results.set(id, result);
        return result;
      })
      .finally(() => this.pending.delete(id));
    this.pending.set(id, task);
    return task;
  }

  private async json(url: string): Promise<unknown> {
    const cached = this.responses.get(url);
    if (cached && this.now() - cached.at < 60_000) return cached.value;
    const value = (async () => {
      const response = await (this.options.fetch ?? fetch)(url, {
        headers: { Accept: 'application/json', 'User-Agent': 'Oryn-Engine-Version-Check' },
        signal: AbortSignal.timeout(12_000),
        redirect: 'error',
        credentials: 'omit',
      });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(
          response.status === 429 || response.status === 403
            ? 'Publisher request limit reached. Try again later.'
            : 'Publisher release information is temporarily unavailable.',
        );
      }
      const reader = response.body?.getReader();
      if (!reader) throw new Error('Missing release response');
      const chunks: Uint8Array[] = [];
      let size = 0;
      try {
        while (true) {
          const { done, value: chunk } = await reader.read();
          if (done) break;
          size += chunk.byteLength;
          if (size > MAX_RESPONSE_BYTES) throw new Error('Release response too large');
          chunks.push(chunk);
        }
        return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
      } finally {
        await reader.cancel();
      }
    })();
    this.responses.set(url, { at: this.now(), value });
    void value.catch(() => this.responses.delete(url));
    return value;
  }

  private async loadInventory(): Promise<EngineVersion[]> {
    let pinned = false;
    if (this.options.readManifest) this.manifest = record(await this.options.readManifest());
    else if (this.options.packaged) {
      this.manifest = record(
        JSON.parse(
          await readFile(join(this.options.resourcesPath, 'engine/release-manifest.json'), 'utf8'),
        ),
      );
    } else {
      // A source checkout has no bundled runtime. Explicitly label these as pins.
      pinned = true;
      const packs: Record<string, unknown> = {};
      const keys: Record<string, string> = {
        aceStep: 'ace-step',
        mlxLm: 'mlx-lm',
        mlxVlm: 'mlx-vlm',
        qwenTts: 'qwen-tts',
      };
      for (const [id, pack] of Object.entries(specification.runtimePacks)) {
        packs[id] = {
          ...pack,
          versions: Object.fromEntries(Object.entries(pack).map(([k, v]) => [keys[k] ?? k, v])),
        };
      }
      this.manifest = {
        versions: { ...specification, 'oryn-engine': specification.engine },
        artifacts: {
          runtimePacks: packs,
          runtimeFoundations: {
            'oryn.runtime.mlx': {
              versions: { mlx: specification.runtimeFoundations['oryn.runtime.mlx'].mlx },
            },
          },
        },
      };
    }
    const base = record(this.manifest.versions);
    if (!string(base['oryn-engine']))
      throw new Error('Bundled engine version information is unavailable.');
    const artifacts = record(this.manifest.artifacts);
    const packs = record(artifacts.runtimePacks);
    const pack = (id: string) => record(packs[`oryn.${id}`]);
    const packVersion = (id: string, key: string) => string(record(pack(id).versions)[key]);
    const entries: EngineVersion[] = [];
    const add = (
      id: string,
      name: string,
      group: EngineVersion['group'],
      version: string | undefined,
      source: string,
      detail = '',
      checkable = true,
    ) =>
      entries.push({
        id,
        name,
        group,
        currentVersion: version ?? null,
        source,
        detail,
        checkable,
        versionOrigin: group === 'Browser runtime' ? 'running' : pinned ? 'pinned' : 'bundled',
        status: checkable ? 'unchecked' : 'managed',
      });
    add(
      'oryn-engine',
      'Oryn Engine',
      'Oryn runtime',
      string(base['oryn-engine']),
      'Oryn',
      'Included with Oryn app updates; no separate public release feed.',
      false,
    );
    add(
      'python',
      'Python',
      'Oryn runtime',
      string(base.python),
      'python.org',
      'Python runtime for local AI.',
    );
    const locations: Record<string, string> = {
      'mlx-lm': 'mlx-lm',
      'mlx-vlm': 'mlx-lm',
      diffusers: 'diffusers-pytorch',
      mflux: 'mflux-mlx',
      'qwen-tts': 'qwen-tts-pytorch',
    };
    for (const [id, name] of PYPI) {
      const version =
        id === 'mlx'
          ? string(
              record(record(record(artifacts.runtimeFoundations)['oryn.runtime.mlx']).versions).mlx,
            )
          : locations[id]
            ? packVersion(locations[id]!, id)
            : string(base[id]);
      let detail = '';
      if (id === 'qwen-tts')
        detail = `Separate runtime · PyTorch ${packVersion('qwen-tts-pytorch', 'torch') ?? 'unknown'} · Transformers ${packVersion('qwen-tts-pytorch', 'transformers') ?? 'unknown'}`;
      if (id === 'diffusers' || id === 'mflux') detail = 'Includes Oryn compatibility changes.';
      add(id, name, 'AI engines', version, 'PyPI', detail);
    }
    add(
      'ace-step',
      'ACE-Step',
      'AI engines',
      packVersion('ace-step-mlx', 'ace-step'),
      'GitHub',
      `Separate runtime · PyTorch ${packVersion('ace-step-mlx', 'torch') ?? 'unknown'} · Transformers ${packVersion('ace-step-mlx', 'transformers') ?? 'unknown'} · MLX-LM ${packVersion('ace-step-mlx', 'mlx-lm') ?? 'unknown'}`,
    );
    const llama = pack('llama-cpp');
    const libraryVersion =
      string(llama.libraryVersion) ??
      Object.keys(record(llama.files))
        .map((file) => /^libllama\.(\d+\.\d+\.\d+)\.dylib$/u.exec(file)?.[1])
        .find(Boolean);
    add(
      'llama-cpp',
      'llama.cpp',
      'AI engines',
      libraryVersion ?? (string(llama.build) ? `b${string(llama.build)}` : undefined),
      'GitHub',
      `Build b${string(llama.build) ?? '?'} · ${string(llama.nativeBackend) ?? 'native'} backend`,
    );
    for (const [id, name, key] of [
      ['electron', 'Electron', 'electron'],
      ['chromium', 'Chromium', 'chrome'],
      ['v8', 'V8', 'v8'],
      ['node', 'Node.js', 'node'],
    ]) {
      add(
        id!,
        name!,
        'Browser runtime',
        this.options.browserVersions[key!],
        'Electron releases',
        id === 'electron'
          ? ''
          : 'Updated with Electron. Latest shows the version in stable Electron.',
      );
    }
    return entries;
  }

  private async checkEntry(entry: EngineVersion): Promise<EngineVersion> {
    let latest: string;
    let status: EngineVersionStatus | undefined;
    let message: string | undefined;
    if (PYPI.has(entry.id)) {
      const data = record(await this.json(`https://pypi.org/pypi/${entry.id}/json`));
      const name = string(record(data.info).name)
        ?.toLowerCase()
        .replace(/[-_.]+/gu, '-');
      if (name !== entry.id) throw new Error('Unexpected release source');
      latest = greatest(
        Object.entries(record(data.releases))
          .filter(
            ([, files]) =>
              Array.isArray(files) && files.some((file: unknown) => record(file).yanked === false),
          )
          .map(([version]) => version),
      );
    } else if (entry.group === 'Browser runtime') {
      const data = await this.json(ELECTRON_URL);
      if (!Array.isArray(data)) throw new Error('Invalid Electron releases');
      const version = greatest(data.map((item: unknown) => string(record(item).version) ?? ''));
      const release = record(data.find((item: unknown) => record(item).version === version));
      latest =
        string(
          release[{ electron: 'version', chromium: 'chrome', v8: 'v8', node: 'node' }[entry.id]!],
        ) ?? '';
      if (!versionParts(latest)) throw new Error('Invalid bundled version');
      message = entry.id === 'electron' ? undefined : `Included in Electron ${version}.`;
    } else if (entry.id === 'python') {
      const data = await this.json(PYTHON_URL);
      if (!Array.isArray(data)) throw new Error('Invalid Python releases');
      latest = greatest(
        data
          .filter((item: unknown) => {
            const release = record(item);
            return (
              release.is_published === true &&
              release.pre_release === false &&
              Date.parse(String(release.release_date)) <= this.now()
            );
          })
          .map((item: unknown) => (string(record(item).name) ?? '').replace(/^Python /u, '')),
      );
    } else {
      const repo = entry.id === 'ace-step' ? ACE_REPO : LLAMA_REPO;
      const release = record(await this.json(`${repo}/releases/latest`));
      const tag = string(release.tag_name);
      if (
        !tag ||
        release.draft !== false ||
        release.prerelease !== false ||
        !/^(?:v?\d+\.\d+\.\d+|b\d+)$/u.test(tag)
      )
        throw new Error('No stable release');
      latest = tag;
      if (entry.id === 'ace-step') {
        const packs = record(record(this.manifest.artifacts).runtimePacks);
        const revision = string(record(packs['oryn.ace-step-mlx']).sourceRevision);
        if (revision && /^[a-f0-9]{40}$/u.test(revision)) {
          const comparison = record(await this.json(`${repo}/compare/${revision}...${tag}`));
          status =
            (
              {
                ahead: 'update',
                behind: 'ahead',
                identical: 'current',
                diverged: 'unknown',
              } as const
            )[String(comparison.status) as 'ahead' | 'behind' | 'identical' | 'diverged'] ??
            'unknown';
          message =
            'Release tag compared with the bundled source revision; package and release numbers differ.';
        } else {
          status = 'unknown';
          message = 'The bundled source revision is unavailable for comparison.';
        }
      } else if (/^b\d+$/u.test(tag)) {
        const build = string(
          record(record(record(this.manifest.artifacts).runtimePacks)['oryn.llama-cpp']).build,
        );
        status =
          build && /^\d+$/u.test(build)
            ? Number(build) < Number(tag.slice(1))
              ? 'update'
              : Number(build) > Number(tag.slice(1))
                ? 'ahead'
                : 'current'
            : 'unknown';
      }
    }
    return {
      ...entry,
      latestVersion: latest,
      status: status ?? comparisonStatus(entry.currentVersion, latest),
      checkedAt: new Date(this.now()).toISOString(),
      message,
    };
  }
}
