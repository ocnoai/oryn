import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';

import { afterEach, expect, it, vi } from 'vitest';

const processes = vi.hoisted(() => ({ spawn: vi.fn(), spawnSync: vi.fn() }));
vi.mock('node:child_process', () => processes);

import { ArtifactFileService } from '../artifact-manager/artifact-file-service';

const runtime = {
  pythonExecutable: 'C:\\Oryn\\runtime\\python.exe',
  helperPath: 'C:\\Oryn\\helpers\\artifact_io_windows.py',
};
const services: ArtifactFileService[] = [];

function setup() {
  vi.stubGlobal(
    'process',
    Object.create(process, {
      platform: { value: 'win32' },
      env: {
        value: { SystemRoot: 'C:\\Windows', SECRET: 'must-not-inherit', PYTHONPATH: 'untrusted' },
      },
    }),
  );
  processes.spawnSync.mockImplementation((_exe, _args, options) => ({
    status: 0,
    stdout: JSON.parse(options.input).operation === 'initialize' ? '[100,104,108]' : '',
  }));
  const service = new ArtifactFileService('C:\\stage', 'C:\\temp', 'C:\\published', runtime);
  services.push(service);
  return service;
}

afterEach(() => {
  for (const service of services.splice(0)) service.close();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

it('pins roots before use and releases native handles once through the trusted helper', () => {
  const service = setup();
  const [executable, args, options] = processes.spawnSync.mock.calls[0]!;
  expect(executable).toBe(runtime.pythonExecutable);
  expect(args).toEqual(['-I', '-S', '-B', runtime.helperPath]);
  expect(options.env).toEqual({ SystemRoot: 'C:\\Windows', PATH: '' });
  expect(options.cwd).toBe('C:\\Oryn\\helpers');
  expect(options.stdio).toEqual(['pipe', 'pipe', 'ignore']);
  expect(JSON.parse(options.input)).toMatchObject({
    operation: 'initialize',
    roots: ['C:\\stage', 'C:\\temp', 'C:\\published'],
  });
  expect(service.cleanupAtStartup()).toBe(true);
  expect(JSON.parse(processes.spawnSync.mock.calls[1]![2].input)).toMatchObject({
    operation: 'cleanup',
    all: true,
    handles: [100, 104, 108],
  });
  service.close();
  service.close();
  expect(processes.spawnSync).toHaveBeenCalledTimes(3);
  expect(JSON.parse(processes.spawnSync.mock.calls[2]![2].input)).toMatchObject({
    operation: 'close',
    handles: [100, 104, 108],
  });
});

it('fails closed if Windows root initialization fails', () => {
  setup();
  processes.spawnSync.mockReturnValue({ status: 1, stdout: 'ARTIFACT_PATH_INVALID' });
  expect(() => new ArtifactFileService('C:\\stage', 'C:\\temp', 'C:\\published', runtime)).toThrow(
    'ARTIFACT_PATH_INVALID',
  );
});

it('kills a cancelled Windows copy and waits for helper close before settling', async () => {
  const service = setup();
  const child = Object.assign(new EventEmitter(), {
    stdin: new PassThrough(),
    stdout: new PassThrough(),
    kill: vi.fn(),
  });
  processes.spawn.mockReturnValue(child);
  const controller = new AbortController();
  const operation = service.run({ operation: 'copy', outputs: [] }, controller.signal);
  const rejected = expect(operation).rejects.toMatchObject({ code: 'ARTIFACT_ACCESS_DENIED' });
  await Promise.resolve();
  expect(processes.spawn).toHaveBeenCalledOnce();
  let settled = false;
  void operation.catch(() => {
    settled = true;
  });
  controller.abort();
  expect(child.kill).toHaveBeenCalledWith('SIGKILL');
  child.emit('exit', 1);
  await Promise.resolve();
  expect(settled).toBe(false);
  child.emit('close', 1);
  await rejected;
  expect(settled).toBe(true);
});
