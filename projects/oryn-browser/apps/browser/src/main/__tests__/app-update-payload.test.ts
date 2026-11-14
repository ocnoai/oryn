import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { constants, openSync, writeFileSync, unlinkSync } from 'node:fs';
import { mkdir, mkdtemp, readdir, realpath, rm, writeFile } from 'node:fs/promises';
import { Server, type IncomingMessage, type ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Writable } from 'node:stream';
import { deflateRawSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { VerifiedMacHandoff } from '../app-updates/verified-mac-handoff';
import {
  VerifiedUpdateArchive,
  validBundleVersion,
  verifyUpdateArchive,
  type NativeCommandRunner,
  type UpdatePayloadIdentity,
} from '../app-updates/verified-update-archive';

const directories: string[] = [];
const leases: VerifiedUpdateArchive[] = [];
type TestZipEntry = {
  name: string;
  body?: string;
  mode?: number;
  compress?: boolean;
  unicode?: string;
  unicodeCRC?: number;
};
afterEach(async () => {
  for (const lease of leases.splice(0)) lease.close();
  for (const directory of directories.splice(0))
    await rm(directory, { recursive: true, force: true });
});

async function scratch() {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'oryn-update-payload-')));
  directories.push(directory);
  return directory;
}

function crc32(data: Buffer) {
  let value = 0xffffffff;
  for (const byte of data) {
    value ^= byte;
    for (let index = 0; index < 8; index++) value = (value >>> 1) ^ (value & 1 ? 0xedb88320 : 0);
  }
  return (value ^ 0xffffffff) >>> 0;
}

function zip(entries: TestZipEntry[]) {
  const local: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name);
    const body = Buffer.from(entry.body ?? 'fixture');
    const stored = entry.compress ? deflateRawSync(body) : body;
    let extra = Buffer.alloc(0);
    if (entry.unicode !== undefined) {
      const target = Buffer.from(entry.unicode);
      extra = Buffer.alloc(9 + target.length);
      extra.writeUInt16LE(0x7075);
      extra.writeUInt16LE(5 + target.length, 2);
      extra[4] = 1;
      extra.writeUInt32LE(entry.unicodeCRC ?? crc32(name), 5);
      target.copy(extra, 9);
    }
    const header = Buffer.alloc(30);
    header.writeUInt32LE(0x04034b50);
    header.writeUInt16LE(20, 4);
    header.writeUInt16LE(entry.compress ? 8 : 0, 8);
    header.writeUInt32LE(crc32(body), 14);
    header.writeUInt32LE(stored.length, 18);
    header.writeUInt32LE(body.length, 22);
    header.writeUInt16LE(name.length, 26);
    header.writeUInt16LE(extra.length, 28);
    const record = Buffer.alloc(46);
    record.writeUInt32LE(0x02014b50);
    record.writeUInt16LE(0x0314, 4);
    record.writeUInt16LE(20, 6);
    record.writeUInt16LE(entry.compress ? 8 : 0, 10);
    record.writeUInt32LE(crc32(body), 16);
    record.writeUInt32LE(stored.length, 20);
    record.writeUInt32LE(body.length, 24);
    record.writeUInt16LE(name.length, 28);
    record.writeUInt16LE(extra.length, 30);
    record.writeUInt32LE(((entry.mode ?? 0x81a4) << 16) >>> 0, 38);
    record.writeUInt32LE(offset, 42);
    local.push(header, name, extra, stored);
    central.push(record, name, extra);
    offset += header.length + name.length + extra.length + stored.length;
  }
  const table = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50);
  end.writeUInt16LE(entries.length & 0xffff, 8);
  end.writeUInt16LE(entries.length & 0xffff, 10);
  end.writeUInt32LE(table.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, table, end]);
}

function executable(cpu = 0x0100000c, platform = 1) {
  const value = Buffer.alloc(56);
  value.writeUInt32LE(0xfeedfacf);
  value.writeUInt32LE(cpu, 4);
  value.writeUInt32LE(2, 12);
  value.writeUInt32LE(1, 16);
  value.writeUInt32LE(24, 20);
  value.writeUInt32LE(0x32, 32);
  value.writeUInt32LE(24, 36);
  value.writeUInt32LE(platform, 40);
  return value;
}

async function fixture(
  entries: TestZipEntry[] = [{ name: 'Oryn Browser.app/Contents/Info.plist' }],
  fields: Record<string, string> = {},
  cpu = 0x0100000c,
  platform = 1,
) {
  const directory = await scratch();
  const source = join(directory, 'download.zip');
  const data = zip(entries);
  await writeFile(source, data);
  const payload: UpdatePayloadIdentity = {
    version: '2.0.0',
    bundleVersion: '200',
    size: data.length,
    sha512: createHash('sha512').update(data).digest('base64'),
    unpackedSize: 1024 * 1024,
  };
  const policy = {
    expectedAppId: 'ai.ocno.oryn.browser',
    expectedTeamId: 'ABCDEFGHIJ',
    currentVersion: '1.0.0',
    currentSystemVersion: '26.6.2',
    verificationRoot: join(directory, 'verification'),
  };
  const runner = vi.fn<NativeCommandRunner>(async (command, args) => {
    if (command.endsWith('sandbox-exec')) {
      const destination = args.at(-1)!;
      await mkdir(join(destination, 'Oryn Browser.app/Contents/MacOS'), {
        recursive: true,
      });
      await writeFile(join(destination, 'Oryn Browser.app/Contents/Info.plist'), 'fixture');
      await writeFile(
        join(destination, 'Oryn Browser.app/Contents/MacOS/Oryn Browser'),
        executable(cpu, platform),
      );
      return '';
    }
    if (command.endsWith('codesign')) return 'flags=0x10000(runtime)';
    if (command.endsWith('spctl')) return 'accepted\nsource=Notarized Developer ID';
    const defaults: Record<string, string> = {
      CFBundleIdentifier: 'ai.ocno.oryn.browser',
      CFBundleShortVersionString: '2.0.0',
      CFBundleVersion: '200',
      CFBundleExecutable: 'Oryn Browser',
      LSMinimumSystemVersion: '14.0',
    };
    if (command.endsWith('plutil')) return fields[args[1]!] ?? defaults[args[1]!] ?? '';
    throw new Error('Unexpected fixture command');
  });
  return { directory, source, data, payload, policy, runner };
}

describe('signed application payload verification', () => {
  it.each([
    { name: '../outside', unicode: 'Oryn Browser.app/Contents/Info.plist' },
    { name: 'Oryn Browser.app/Contents/Info.plist', unicode: '../outside' },
    { name: 'Oryn Browser.app/Contents/Info.plist', unicode: '../outside', unicodeCRC: 0 },
  ])('rejects raw/UnicodePath disagreement before native extraction', async (entry) => {
    const value = await fixture([entry]);
    await expect(
      verifyUpdateArchive(
        value.source,
        value.payload,
        value.policy,
        new AbortController().signal,
        value.runner,
      ),
    ).rejects.toThrow();
    expect(value.runner).not.toHaveBeenCalled();
  });

  it('rejects a composed symlink escape and cycles, while retaining ordinary framework aliases', async () => {
    const info = { name: 'Oryn Browser.app/Contents/Info.plist' };
    for (const links of [
      [
        { name: 'Oryn Browser.app/Contents/a', body: 'deep/link/../../../outside', mode: 0xa1ff },
        { name: 'Oryn Browser.app/Contents/deep/link', body: '../../Contents', mode: 0xa1ff },
      ],
      [
        { name: 'Oryn Browser.app/Contents/a', body: 'b', mode: 0xa1ff },
        { name: 'Oryn Browser.app/Contents/b', body: 'a', mode: 0xa1ff },
      ],
    ]) {
      const value = await fixture([info, ...links]);
      await expect(
        verifyUpdateArchive(
          value.source,
          value.payload,
          value.policy,
          new AbortController().signal,
          value.runner,
        ),
      ).rejects.toThrow('APP_UPDATE_ARCHIVE_INVALID');
      expect(value.runner).not.toHaveBeenCalled();
    }
    const value = await fixture([
      info,
      {
        name: 'Oryn Browser.app/Contents/Frameworks/Test.framework/Versions/Current',
        body: 'A',
        mode: 0xa1ff,
      },
      {
        name: 'Oryn Browser.app/Contents/Frameworks/Test.framework/Resources',
        body: 'Versions/Current/Resources',
        mode: 0xa1ff,
      },
      {
        name: 'Oryn Browser.app/Contents/Frameworks/Test.framework/Versions/A/Resources/readme',
        body: '',
      },
      { name: 'Oryn Browser.app/Contents/é.txt', unicode: 'Oryn Browser.app/Contents/é.txt' },
    ]);
    const archive = await verifyUpdateArchive(
      value.source,
      value.payload,
      value.policy,
      new AbortController().signal,
      value.runner,
    );
    leases.push(archive);
    expect(value.runner).toHaveBeenCalled();
  });
  it('detects false deflate expansion sizes before allowing native extraction', async () => {
    const value = await fixture([
      { name: 'Oryn Browser.app/Contents/Info.plist', body: 'x'.repeat(100_000), compress: true },
    ]);
    const central = value.data.indexOf(Buffer.from([0x50, 0x4b, 0x01, 0x02]));
    value.data.writeUInt32LE(1, central + 24);
    value.data.writeUInt32LE(1, 22);
    value.payload.sha512 = createHash('sha512').update(value.data).digest('base64');
    await writeFile(value.source, value.data);
    await expect(
      verifyUpdateArchive(
        value.source,
        value.payload,
        value.policy,
        new AbortController().signal,
        value.runner,
      ),
    ).rejects.toThrow();
    expect(value.runner).not.toHaveBeenCalled();
  });

  it('walks every record in a ditto-style wrapped-count ZIP, including the required final Info.plist', async () => {
    const entries = Array.from({ length: 65_536 }, (_, index) => ({
      name: `Oryn Browser.app/Contents/f-${index}`,
      body: '',
    }));
    entries.push({ name: 'Oryn Browser.app/Contents/Info.plist', body: 'fixture' });
    const value = await fixture(entries);
    const archive = await verifyUpdateArchive(
      value.source,
      value.payload,
      value.policy,
      new AbortController().signal,
      value.runner,
    );
    leases.push(archive);
    expect(value.runner).toHaveBeenCalled();
  }, 60_000);

  it.each(['count', 'disk', 'local-name', 'overlap', 'trailing'])(
    'rejects inconsistent ZIP %s metadata without native extraction',
    async (mutation) => {
      const value = await fixture([
        { name: 'Oryn Browser.app/Contents/Info.plist' },
        { name: 'Oryn Browser.app/Contents/other' },
      ]);
      let data = value.data;
      const end = data.length - 22;
      if (mutation === 'count') {
        data.writeUInt16LE(1, end + 8);
        data.writeUInt16LE(1, end + 10);
      }
      if (mutation === 'disk') data.writeUInt16LE(1, end + 4);
      if (mutation === 'local-name') data[30] = 'X'.charCodeAt(0);
      if (mutation === 'overlap') {
        const first = data.indexOf(Buffer.from([0x50, 0x4b, 0x01, 0x02]));
        const second = data.indexOf(Buffer.from([0x50, 0x4b, 0x01, 0x02]), first + 4);
        data.writeUInt32LE(0, second + 42);
      }
      if (mutation === 'trailing') data = Buffer.concat([data, Buffer.from('unseen record')]);
      value.payload.size = data.length;
      value.payload.sha512 = createHash('sha512').update(data).digest('base64');
      await writeFile(value.source, data);
      await expect(
        verifyUpdateArchive(
          value.source,
          value.payload,
          value.policy,
          new AbortController().signal,
          value.runner,
        ),
      ).rejects.toThrow('APP_UPDATE_ARCHIVE_INVALID');
      expect(value.runner).not.toHaveBeenCalled();
    },
  );

  it('rejects a correctly signed release that requires a newer macOS', async () => {
    const value = await fixture(undefined, { LSMinimumSystemVersion: '27.0' });
    await expect(
      verifyUpdateArchive(
        value.source,
        value.payload,
        value.policy,
        new AbortController().signal,
        value.runner,
      ),
    ).rejects.toThrow('APP_UPDATE_OS_UNSUPPORTED');
  });
  it('authenticates an exact version/build, retains unlinked read authority and removes extraction scratch', async () => {
    const value = await fixture();
    const archive = await verifyUpdateArchive(
      value.source,
      value.payload,
      value.policy,
      new AbortController().signal,
      value.runner,
    );
    leases.push(archive);
    const extraction = value.runner.mock.calls.find(([command]) =>
      command.endsWith('sandbox-exec'),
    )!;
    const [, args, options] = extraction;
    const profile = args[1]!;
    const destination = args.at(-1)!;
    expect(args.slice(2, -1)).toEqual(['/usr/bin/ditto', '-x', '-k', '/dev/fd/3']);
    expect(options.archiveFd).toBe(archive.descriptor());
    expect(profile).toContain('(deny default)');
    expect(profile).toContain('(allow process-exec (literal "/usr/bin/ditto"))');
    expect(profile).toContain('(allow file-read* (literal "/") ');
    expect(profile).toContain(`(allow file-write* (subpath ${JSON.stringify(destination)}))`);
    expect(profile.match(/\(allow file-write\*/gu)).toHaveLength(1);
    expect(profile).not.toMatch(/\(allow (?:default|network|process-fork)/u);
    expect(profile).not.toContain('(subpath "/")');
    expect(profile).not.toContain('/Users/');
    expect(profile).not.toContain(value.source);
    expect(
      value.runner.mock.calls.find(([command]) => command.endsWith('codesign'))?.[1].join(' '),
    ).toContain('subject.OU] = "ABCDEFGHIJ"');
    expect(await readdir(value.policy.verificationRoot)).toEqual([]);
    await writeFile(value.source, Buffer.alloc(value.data.length, 0xff));
    await archive.revalidate();
    const chunks: Buffer[] = [];
    for await (const chunk of archive.stream()) chunks.push(Buffer.from(chunk));
    expect(Buffer.concat(chunks)).toEqual(value.data);
  });

  it.each([
    ['CFBundleShortVersionString', '3.0.0', 'APP_UPDATE_VERSION_MISMATCH'],
    ['CFBundleVersion', '201', 'APP_UPDATE_VERSION_MISMATCH'],
    ['CFBundleIdentifier', 'same.publisher.other.app', 'APP_UPDATE_IDENTITY_MISMATCH'],
  ])(
    'rejects authenticated %s mismatch rather than trusting its feed label',
    async (field, actual, code) => {
      const value = await fixture(undefined, { [field]: actual });
      await expect(
        verifyUpdateArchive(
          value.source,
          value.payload,
          value.policy,
          new AbortController().signal,
          value.runner,
        ),
      ).rejects.toThrow(code);
      expect(await readdir(value.policy.verificationRoot)).toEqual([]);
    },
  );

  it.each([
    [0x01000007, 1],
    [0x0100000c, 2],
  ])('rejects wrong executable architecture/platform', async (cpu, platform) => {
    const value = await fixture(undefined, {}, cpu, platform);
    await expect(
      verifyUpdateArchive(
        value.source,
        value.payload,
        value.policy,
        new AbortController().signal,
        value.runner,
      ),
    ).rejects.toThrow('APP_UPDATE_ARCH_MISMATCH');
  });

  it('denies a bad signature and notarization without reading unauthenticated bundle fields', async () => {
    for (const failing of ['codesign', 'spctl']) {
      const value = await fixture();
      const base = value.runner.getMockImplementation()!;
      value.runner.mockImplementation(async (...args) => {
        if (args[0].endsWith(failing)) throw new Error('private native stderr');
        return base(...args);
      });
      await expect(
        verifyUpdateArchive(
          value.source,
          value.payload,
          value.policy,
          new AbortController().signal,
          value.runner,
        ),
      ).rejects.toThrow(
        failing === 'codesign' ? 'APP_UPDATE_SIGNATURE_INVALID' : 'APP_UPDATE_NOTARIZATION_INVALID',
      );
      expect(value.runner.mock.calls.some(([command]) => command.endsWith('plutil'))).toBe(false);
    }
  });

  it.each(
    [
      [{ name: '../outside' }],
      [{ name: 'Other.app/Contents/Info.plist' }],
      [
        { name: 'Oryn Browser.app/Contents/Info.plist' },
        { name: 'Oryn Browser.app/contents/info.plist' },
      ],
      [
        { name: 'Oryn Browser.app/Contents/Info.plist' },
        {
          name: 'Oryn Browser.app/Contents/link',
          mode: 0xa1ff,
          body: '../../../outside',
        },
      ],
      [
        { name: 'Oryn Browser.app/Contents/Info.plist' },
        { name: 'Oryn Browser.app/Contents/link', mode: 0xa1ff, body: 'real' },
        { name: 'Oryn Browser.app/Contents/link/file' },
      ],
    ].map((entries) => [entries]),
  )('rejects unsafe archive layout before invoking any extractor', async (entries) => {
    const value = await fixture(entries);
    await expect(
      verifyUpdateArchive(
        value.source,
        value.payload,
        value.policy,
        new AbortController().signal,
        value.runner,
      ),
    ).rejects.toThrow();
    expect(value.runner).not.toHaveBeenCalled();
  });

  it('rejects tampered cache bytes, stale cancellation and an excessive expansion declaration', async () => {
    const value = await fixture();
    await writeFile(value.source, Buffer.alloc(value.data.length, 0xff));
    await expect(
      verifyUpdateArchive(
        value.source,
        value.payload,
        value.policy,
        new AbortController().signal,
        value.runner,
      ),
    ).rejects.toThrow('APP_UPDATE_PAYLOAD_CHANGED');
    const controller = new AbortController();
    controller.abort();
    await expect(
      verifyUpdateArchive(
        value.source,
        value.payload,
        value.policy,
        controller.signal,
        value.runner,
      ),
    ).rejects.toThrow();
    await expect(
      verifyUpdateArchive(
        value.source,
        { ...value.payload, unpackedSize: Number.MAX_SAFE_INTEGER },
        value.policy,
        new AbortController().signal,
        value.runner,
      ),
    ).rejects.toThrow('APP_UPDATE_VERIFICATION_UNAVAILABLE');
    expect(value.runner).not.toHaveBeenCalled();
  });

  it.each(['', ' ', '2 00', '200\n', 'é', 'a'.repeat(65)])(
    'rejects unbounded or whitespace build identity %j',
    (value) => {
      expect(validBundleVersion(value)).toBe(false);
    },
  );
});

class Response extends Writable {
  statusCode = 200;
  headersSent = false;
  chunks: Buffer[] = [];
  override _write(chunk: Buffer, _encoding: string, callback: () => void) {
    this.chunks.push(Buffer.from(chunk));
    callback();
  }
  writeHead(code: number) {
    this.statusCode = code;
    this.headersSent = true;
    return this;
  }
}

function request(server: Server, url: string, authorization?: string) {
  const response = new Response();
  const done = new Promise<{ status: number; body: Buffer }>((resolveResult, reject) => {
    response.once('finish', () =>
      resolveResult({
        status: response.statusCode,
        body: Buffer.concat(response.chunks),
      }),
    );
    response.once('error', reject);
  });
  const incoming = Object.assign(new EventEmitter(), {
    method: 'GET',
    url,
    headers: { authorization },
  });
  server.emit('request', incoming as IncomingMessage, response as unknown as ServerResponse);
  return done;
}

describe('verified-byte native loopback handoff', () => {
  it('keeps the maintained root authentication and serves the retained bytes after cache path replacement', async () => {
    const directory = await scratch();
    const path = join(directory, 'retained.zip');
    const bytes = Buffer.from('verified original update');
    writeFileSync(path, bytes);
    const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    unlinkSync(path);
    const archive = new VerifiedUpdateArchive(fd, {
      version: '2.0.0',
      bundleVersion: '200',
      size: bytes.length,
      sha512: createHash('sha512').update(bytes).digest('base64'),
      unpackedSize: 100,
    });
    leases.push(archive);
    const server = new Server();
    vi.spyOn(server, 'address').mockReturnValue({
      address: '127.0.0.1',
      family: 'IPv4',
      port: 12345,
    });
    vi.spyOn(server, 'closeAllConnections').mockImplementation(() => undefined);
    vi.spyOn(server, 'close').mockImplementation(() => server);
    const originalPayload = vi.fn();
    const route = `/${'a'.repeat(128)}.zip`;
    server.on('request', (incoming, response) => {
      if (incoming.url !== '/') {
        originalPayload();
        response.end('UNVERIFIED');
        return;
      }
      if (incoming.headers.authorization !== 'Basic fixture-secret') {
        response.statusCode = 401;
        response.end();
        return;
      }
      response.end(JSON.stringify({ url: `http://127.0.0.1:12345${route}` }));
    });
    let current = true;
    const handoff = new VerifiedMacHandoff({ server }, () => current);
    expect((await request(server, '/')).status).toBe(503);
    handoff.attach(archive);
    expect((await request(server, '/')).status).toBe(401);
    expect((await request(server, '/', 'Basic fixture-secret')).status).toBe(200);
    writeFileSync(path, 'replacement signed C payload');
    expect((await request(server, route)).body).toEqual(bytes);
    expect((await request(server, `${route}?changed=1`)).status).toBe(503);
    expect(originalPayload).not.toHaveBeenCalled();
    current = false;
    expect((await request(server, route)).status).toBe(503);
    await expect(handoff.revalidate()).rejects.toThrow('APP_UPDATE_STALE_PAYLOAD');
    handoff.close();
  });

  it('refuses a changed maintained-updater hook shape rather than falling back to path serving', () => {
    expect(() => new VerifiedMacHandoff({}, () => true)).toThrow(
      'APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE',
    );
    const server = new Server();
    vi.spyOn(server, 'address').mockReturnValue({
      address: '127.0.0.1',
      family: 'IPv4',
      port: 12345,
    });
    server.on('request', () => undefined);
    server.on('request', () => undefined);
    expect(() => new VerifiedMacHandoff({ server }, () => true)).toThrow(
      'APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE',
    );
  });
});
