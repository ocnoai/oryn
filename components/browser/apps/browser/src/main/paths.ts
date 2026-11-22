import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { app } from 'electron';
import { browserApplicationName, type BrowserBuildFlavor } from './browser-build-flavor';

export interface EngineLaunchPaths {
  engineDirectory: string;
  pythonExecutable: string;
  pythonArguments: string[];
  pythonPath?: string;
  engineSecurityProfile?: string;
  runtimePackSecurityProfile?: string;
  sandboxRequired: boolean;
  packaged: boolean;
}

interface EnginePathEnvironment {
  ORYN_ENGINE_DIR?: string;
  ORYN_ENGINE_PYTHON?: string;
}

export function resolveEngineLaunchPaths(options: {
  packaged: boolean;
  resourcesPath: string;
  platform: NodeJS.Platform;
  environment: EnginePathEnvironment;
  appPath?: string;
  pathExists?: (path: string) => boolean;
}): EngineLaunchPaths {
  const pathExists = options.pathExists ?? existsSync;
  if (options.packaged) {
    const engineDirectory = join(resolve(options.resourcesPath), 'engine');
    const pythonExecutable = join(
      engineDirectory,
      'runtime',
      'bin',
      options.platform === 'win32' ? 'python.exe' : 'python3.12',
    );
    if (!pathExists(pythonExecutable)) {
      throw new Error('The bundled Oryn Engine runtime is missing.');
    }
    const securityRoot = join(resolve(options.resourcesPath), 'security');
    const engineSecurityProfile = join(securityRoot, 'engine.sb');
    const runtimePackSecurityProfile = join(securityRoot, 'runtime-pack.sb');
    if (
      options.platform === 'darwin' &&
      (!pathExists(engineSecurityProfile) || !pathExists(runtimePackSecurityProfile))
    ) {
      throw new Error('The bundled Oryn Engine security profiles are missing.');
    }
    return {
      engineDirectory,
      pythonExecutable,
      pythonArguments: ['-I', '-B', '-m', 'oryn_engine'],
      ...(options.platform === 'darwin'
        ? { engineSecurityProfile, runtimePackSecurityProfile }
        : {}),
      sandboxRequired: options.platform === 'darwin',
      packaged: true,
    };
  }

  const configuredDirectory = options.environment.ORYN_ENGINE_DIR;
  if (!configuredDirectory) {
    throw new Error('ORYN_ENGINE_DIR is required for a source checkout of Oryn Browser.');
  }
  const engineDirectory = resolve(configuredDirectory);
  const virtualEnvironmentPython =
    options.platform === 'win32'
      ? join(engineDirectory, '.venv', 'Scripts', 'python.exe')
      : join(engineDirectory, '.venv', 'bin', 'python');
  const pythonExecutable =
    options.environment.ORYN_ENGINE_PYTHON ??
    (pathExists(virtualEnvironmentPython)
      ? virtualEnvironmentPython
      : options.platform === 'win32'
        ? 'python'
        : 'python3');
  return {
    engineDirectory,
    pythonExecutable,
    pythonArguments: ['-m', 'oryn_engine'],
    pythonPath: join(engineDirectory, 'src'),
    ...(options.platform === 'darwin' &&
    options.appPath &&
    pathExists(join(resolve(options.appPath), 'build', 'security', 'engine.sb')) &&
    pathExists(join(resolve(options.appPath), 'build', 'security', 'runtime-pack.sb'))
      ? {
          engineSecurityProfile: join(resolve(options.appPath), 'build', 'security', 'engine.sb'),
          runtimePackSecurityProfile: join(
            resolve(options.appPath),
            'build',
            'security',
            'runtime-pack.sb',
          ),
        }
      : {}),
    sandboxRequired: false,
    packaged: false,
  };
}

export function engineLaunchPaths(): EngineLaunchPaths {
  return resolveEngineLaunchPaths({
    packaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    platform: process.platform,
    environment: process.env,
    appPath: app.getAppPath(),
  });
}

export function packagedProfileRoot(
  appDataPath: string,
  flavor: BrowserBuildFlavor = 'release',
): string {
  return join(resolve(appDataPath), browserApplicationName(flavor));
}

/** Trusted Browser helper; no Engine/model packages are imported by this process. */
export function artifactIoRuntime(): { pythonExecutable: string; helperPath: string } {
  const helperName = artifactIoHelperName(process.platform);
  const helperPath = app.isPackaged
    ? join(process.resourcesPath, 'helpers', helperName)
    : join(app.getAppPath(), 'build', 'helpers', helperName);
  if (!existsSync(helperPath)) throw new Error('The Browser artifact I/O helper is missing.');
  return { pythonExecutable: engineLaunchPaths().pythonExecutable, helperPath };
}

export function artifactIoHelperName(platform: NodeJS.Platform): string {
  return platform === 'win32' ? 'artifact_io_windows.py' : 'artifact_io.py';
}
