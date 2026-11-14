import { describe, expect, it, vi } from 'vitest';
import { compareVersions, EngineVersionService } from '../engine-versions/engine-version-service';

const manifest = {
  versions: { 'oryn-engine': '0.2.0', python: '3.12.13', torch: '2.14.1' },
  artifacts: {
    runtimePacks: {
      'oryn.ace-step-mlx': {
        sourceRevision: 'a'.repeat(40),
        versions: {
          'ace-step': '1.5.0',
          torch: '2.11.0',
          transformers: '4.57.6',
          'mlx-lm': '0.29.1',
        },
      },
      'oryn.llama-cpp': {
        build: '11711',
        nativeBackend: 'metal',
        files: { 'libllama.0.7.0.dylib': 'hash' },
      },
    },
  },
};
const pythonPackage = (releases: Record<string, unknown>) => ({
  info: { name: 'torch' },
  releases,
});
const file = [{ yanked: false }];
const reply = (body: unknown) => new Response(JSON.stringify(body));
function fixture(body: unknown = pythonPackage({ '2.15.0': file })) {
  let now = Date.parse('2026-10-30T12:00:00Z');
  const fetcher = vi.fn<typeof fetch>(async () => reply(body));
  const service = new EngineVersionService({
    packaged: true,
    resourcesPath: '/unused',
    browserVersions: {
      electron: '45.1.0',
      chrome: '156.0.8078.49',
      v8: '15.6.75.19-electron.0',
      node: '24.22.1',
    },
    fetch: fetcher,
    now: () => now,
    readManifest: async () => manifest,
  });
  return {
    service,
    fetcher,
    advance: () => {
      now += 61_000;
    },
  };
}

describe('engine release checks', () => {
  it('reads bundled versions without a network request and keeps source checkout pins explicit', async () => {
    const { service, fetcher } = fixture();
    const entries = await service.list();
    expect(entries.find((entry) => entry.id === 'torch')).toMatchObject({
      currentVersion: '2.14.1',
      versionOrigin: 'bundled',
      status: 'unchecked',
    });
    expect(entries.find((entry) => entry.id === 'llama-cpp')).toMatchObject({
      currentVersion: '0.7.0',
      detail: 'Build b11711 · metal backend',
    });
    expect(entries.find((entry) => entry.id === 'transformers')?.currentVersion).toBeNull();
    expect((await service.check('oryn-engine')).status).toBe('managed');
    expect(fetcher).not.toHaveBeenCalled();
    const source = new EngineVersionService({
      packaged: false,
      resourcesPath: '',
      browserVersions: {},
    });
    expect((await source.list()).find((entry) => entry.id === 'torch')?.versionOrigin).toBe(
      'pinned',
    );
  });

  it('compares numeric versions and post releases instead of sorting strings', () => {
    expect(compareVersions('2.9.0', '2.15.0')).toBe(-1);
    expect(compareVersions('v1.0.0', '1.0')).toBe(0);
    expect(compareVersions('1.0.0', '1.0.0.post1')).toBe(-1);
    expect(compareVersions('15.6.75.19-electron.0', '15.6.75.19')).toBe(0);
    expect(compareVersions('1.0.0rc1', '1.0.0')).toBeUndefined();
  });

  it('ignores prereleases, yanked and empty releases while finding the latest stable', async () => {
    const { service } = fixture(
      pythonPackage({
        '2.9.0': file,
        '2.15.0': file,
        '3.0.0rc1': file,
        '3.0.0.dev1': file,
        '9.0.0': [{ yanked: true }],
        '8.0.0': [],
      }),
    );
    expect(await service.check('torch')).toMatchObject({
      latestVersion: '2.15.0',
      status: 'update',
      checkedAt: '2026-10-30T12:00:00.000Z',
    });
  });

  it.each([
    ['2.14.1', 'current'],
    ['2.13.0', 'ahead'],
  ])('handles stable %s without claiming an upgrade', async (version, status) => {
    const { service } = fixture(pythonPackage({ [version]: file }));
    expect((await service.check('torch')).status).toBe(status);
  });

  it('deduplicates simultaneous checks and reuses fresh metadata for one minute', async () => {
    const { service, fetcher, advance } = fixture();
    await Promise.all([service.check('torch'), service.check('torch')]);
    await service.check('torch');
    expect(fetcher).toHaveBeenCalledTimes(1);
    advance();
    await service.check('torch');
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({ credentials: 'omit', redirect: 'error' });
  });

  it('preserves last successful data on network failure and allows retry', async () => {
    const { service, fetcher, advance } = fixture();
    await service.check('torch');
    advance();
    fetcher.mockRejectedValueOnce(new Error('offline'));
    expect(await service.check('torch')).toMatchObject({
      status: 'error',
      latestVersion: '2.15.0',
      checkedAt: '2026-10-30T12:00:00.000Z',
    });
    expect((await service.check('torch')).status).toBe('update');
  });

  it('reports rate limits and malformed feeds without declaring the engine current', async () => {
    const { service, fetcher } = fixture();
    fetcher.mockResolvedValueOnce(new Response('', { status: 429 }));
    expect(await service.check('torch')).toMatchObject({
      status: 'error',
      message: 'Publisher request limit reached. Try again later.',
    });
    fetcher.mockResolvedValueOnce(
      reply({ info: { name: 'wrong-project' }, releases: { '2.14.1': file } }),
    );
    expect((await service.check('torch')).status).toBe('error');
  });

  it('rejects unknown engine identifiers before any network request', async () => {
    const { service, fetcher } = fixture();
    await expect(service.check('https://example.com')).rejects.toThrow('INVALID_ENGINE_ID');
    await expect(service.check({ id: 'torch' })).rejects.toThrow('INVALID_ENGINE_ID');
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('compares Chromium, V8 and Node against the newest stable Electron bundle', async () => {
    const { service, fetcher } = fixture([
      { version: '46.0.0-beta.1', chrome: '158.0.0.0' },
      { version: '45.1.0', chrome: '156.0.8078.49', v8: '15.6.75.19', node: '24.22.1' },
    ]);
    for (const id of ['electron', 'chromium', 'v8', 'node'])
      expect((await service.check(id)).status).toBe('current');
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('excludes unpublished, future and prerelease Python builds', async () => {
    const { service } = fixture([
      { name: 'Python 3.15.2', is_published: true, pre_release: false, release_date: '2026-10-25' },
      { name: 'Python 3.16.0', is_published: true, pre_release: false, release_date: '2027-10-25' },
      {
        name: 'Python 3.16.0rc1',
        is_published: true,
        pre_release: true,
        release_date: '2026-10-25',
      },
      {
        name: 'Python 3.17.0',
        is_published: false,
        pre_release: false,
        release_date: '2026-10-25',
      },
    ]);
    expect(await service.check('python')).toMatchObject({
      latestVersion: '3.15.2',
      status: 'update',
    });
  });

  it.each([
    ['identical', 'current'],
    ['ahead', 'update'],
    ['behind', 'ahead'],
    ['diverged', 'unknown'],
  ])(
    'uses ACE-Step commit ancestry (%s), not unrelated package/tag numbers',
    async (ancestry, status) => {
      const { service, fetcher } = fixture();
      fetcher
        .mockResolvedValueOnce(reply({ tag_name: 'v0.1.8', draft: false, prerelease: false }))
        .mockResolvedValueOnce(reply({ status: ancestry }));
      expect(await service.check('ace-step')).toMatchObject({
        currentVersion: '1.5.0',
        latestVersion: 'v0.1.8',
        status,
      });
    },
  );

  it('compares stable llama.cpp library releases and legacy build releases', async () => {
    const { service, fetcher, advance } = fixture({
      tag_name: 'v0.8.0',
      draft: false,
      prerelease: false,
    });
    expect(await service.check('llama-cpp')).toMatchObject({
      latestVersion: 'v0.8.0',
      status: 'update',
    });
    advance();
    fetcher.mockResolvedValueOnce(reply({ tag_name: 'b12001', draft: false, prerelease: false }));
    expect((await service.check('llama-cpp')).status).toBe('update');
  });

  it('fails visibly when a packaged manifest cannot be read instead of substituting project pins', async () => {
    const service = new EngineVersionService({
      packaged: true,
      resourcesPath: '/missing-oryn',
      browserVersions: {},
    });
    await expect(service.list()).rejects.toThrow();
  });
});
