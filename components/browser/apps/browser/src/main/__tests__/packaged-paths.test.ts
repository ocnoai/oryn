import { describe, expect, it } from 'vitest';

import { artifactIoHelperName, packagedProfileRoot, resolveEngineLaunchPaths } from '../paths';

describe('packaged engine paths', () => {
  it('selects the native Windows artifact helper and bundled runtime independently of host settings', () => {
    expect(artifactIoHelperName('win32')).toBe('artifact_io_windows.py');
    expect(artifactIoHelperName('darwin')).toBe('artifact_io.py');
    expect(artifactIoHelperName('linux')).toBe('artifact_io.py');
    const paths = resolveEngineLaunchPaths({
      packaged: true,
      resourcesPath: '/packaged/resources',
      platform: 'win32',
      environment: { ORYN_ENGINE_PYTHON: '/untrusted/python.exe' },
      pathExists: (path) => path.endsWith('/engine/runtime/bin/python.exe'),
    });
    expect(paths.pythonExecutable).toBe('/packaged/resources/engine/runtime/bin/python.exe');
    expect(paths.pythonArguments).toEqual(['-I', '-B', '-m', 'oryn_engine']);
    // The backend is not a substitute for the still-missing Windows Engine sandbox.
    expect(paths.sandboxRequired).toBe(false);
  });
  it('uses only the bundled isolated runtime in a packaged app', () => {
    const expected =
      '/Applications/Oryn Browser.app/Contents/Resources/engine/runtime/bin/python3.12';
    const engineProfile = '/Applications/Oryn Browser.app/Contents/Resources/security/engine.sb';
    const packProfile =
      '/Applications/Oryn Browser.app/Contents/Resources/security/runtime-pack.sb';
    const paths = resolveEngineLaunchPaths({
      packaged: true,
      resourcesPath: '/Applications/Oryn Browser.app/Contents/Resources',
      platform: 'darwin',
      environment: {
        ORYN_ENGINE_DIR: '/tmp/attacker-engine',
        ORYN_ENGINE_PYTHON: '/tmp/attacker-python',
      },
      pathExists: (path) => [expected, engineProfile, packProfile].includes(path),
    });

    expect(paths).toEqual({
      engineDirectory: '/Applications/Oryn Browser.app/Contents/Resources/engine',
      pythonExecutable: expected,
      pythonArguments: ['-I', '-B', '-m', 'oryn_engine'],
      engineSecurityProfile: engineProfile,
      runtimePackSecurityProfile: packProfile,
      sandboxRequired: true,
      packaged: true,
    });
  });

  it('fails closed when the packaged Python runtime is missing', () => {
    expect(() =>
      resolveEngineLaunchPaths({
        packaged: true,
        resourcesPath: '/Applications/Oryn Browser.app/Contents/Resources',
        platform: 'darwin',
        environment: {},
        pathExists: () => false,
      }),
    ).toThrow('The bundled Oryn Engine runtime is missing.');
  });

  it('retains explicit source-checkout configuration for development', () => {
    expect(
      resolveEngineLaunchPaths({
        packaged: false,
        resourcesPath: '/unused',
        platform: 'darwin',
        environment: { ORYN_ENGINE_DIR: '/checkout/oryn-engine' },
        pathExists: () => true,
      }),
    ).toEqual({
      engineDirectory: '/checkout/oryn-engine',
      pythonExecutable: '/checkout/oryn-engine/.venv/bin/python',
      pythonArguments: ['-m', 'oryn_engine'],
      pythonPath: '/checkout/oryn-engine/src',
      sandboxRequired: false,
      packaged: false,
    });
  });

  it('fails closed when packaged security profiles are missing', () => {
    expect(() =>
      resolveEngineLaunchPaths({
        packaged: true,
        resourcesPath: '/Applications/Oryn Browser.app/Contents/Resources',
        platform: 'darwin',
        environment: {},
        pathExists: (path) => path.endsWith('/engine/runtime/bin/python3.12'),
      }),
    ).toThrow('The bundled Oryn Engine security profiles are missing.');
  });

  it('uses the stable Oryn Browser application-support root', () => {
    expect(packagedProfileRoot('/Users/person/Library/Application Support')).toBe(
      '/Users/person/Library/Application Support/Oryn Browser',
    );
  });

  it('isolates packaged local development data from the release profile', () => {
    const appData = '/Users/person/Library/Application Support';
    expect(packagedProfileRoot(appData, 'local-development')).toBe(`${appData}/Oryn Browser Local`);
    expect(packagedProfileRoot(appData, 'local-development')).not.toBe(
      packagedProfileRoot(appData),
    );
  });
});
