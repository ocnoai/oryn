/* global console, process */
import { createHash } from 'node:crypto';
import { Buffer } from 'node:buffer';
import {
  copyFileSync,
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  readlinkSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { spawnSync } from 'node:child_process';

const browserRoot = resolve(import.meta.dirname, '..');
const stageRoot = join(browserRoot, 'packaging', '.stage');
const engineRoot = join(stageRoot, 'engine');
const runtimeRoot = join(engineRoot, 'runtime');
const specificationPath = join(browserRoot, 'packaging', 'runtime-spec.json');
const requirementsPath = join(
  browserRoot,
  'packaging',
  'engine-runtime-macos-arm64-py312.lock.txt',
);
const aceStepRequirementsPath = join(
  browserRoot,
  'packaging',
  'ace-step-macos-arm64-py312.lock.txt',
);
const mlxFoundationRequirementsPath = join(
  browserRoot,
  'packaging',
  'mlx-foundation-macos-arm64-py312.lock.txt',
);
const mfluxRequirementsPath = join(
  browserRoot,
  'packaging',
  'mflux-mlx-macos-arm64-py312.lock.txt',
);
const kokoroRequirementsPath = join(
  browserRoot,
  'packaging',
  'kokoro-onnx-macos-arm64-py312.lock.txt',
);
const diffusersRequirementsPath = join(
  browserRoot,
  'packaging',
  'diffusers-macos-arm64-py312.lock.txt',
);
const qwenTtsRequirementsPath = join(
  browserRoot,
  'packaging',
  'qwen-tts-pytorch-macos-arm64-py312.lock.txt',
);
const specification = JSON.parse(readFileSync(specificationPath, 'utf8'));

if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  fail('The initial packaged runtime must be assembled on macOS arm64.');
}
const engineWheel = requiredFile('ORYN_ENGINE_WHEEL');
const protocolWheel = requiredFile('ORYN_PROTOCOL_WHEEL');
const depthNativeWheel = join(browserRoot, 'vendor', specification.depthNativeWheel.file);
if (digest(depthNativeWheel) !== specification.depthNativeWheel.sha256) {
  fail('The reviewed depth native wheel digest changed.');
}
const engineRevision = process.env.ORYN_ENGINE_REVISION;
if (!engineRevision || !/^[a-f0-9]{7,40}$/u.test(engineRevision)) {
  fail('ORYN_ENGINE_REVISION must identify the tested Engine source revision.');
}

run('uv', ['python', 'install', specification.python]);
const pythonExecutable = capture('uv', [
  'python',
  'find',
  '--managed-python',
  specification.python,
]);
const pythonHome = dirname(dirname(realpathSync(pythonExecutable)));

// Finder can recreate .DS_Store after a recursive directory enumeration. Retry
// the whole traversal so that newly created entries are enumerated and removed.
for (let attempt = 0; ; attempt += 1) {
  try {
    rmSync(stageRoot, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
    break;
  } catch (error) {
    if (error.code !== 'ENOTEMPTY' || attempt >= 4) throw error;
  }
}
mkdirSync(engineRoot, { recursive: true, mode: 0o700 });
cpSync(pythonHome, runtimeRoot, { recursive: true, dereference: false, preserveTimestamps: true });
relativizeInternalSymlinks(runtimeRoot, pythonHome);
const sitePackages = join(runtimeRoot, 'lib', 'python3.12', 'site-packages');
const bundledPython = join(runtimeRoot, 'bin', 'python3.12');
rmSync(sitePackages, { recursive: true, force: true });
mkdirSync(sitePackages, { recursive: true, mode: 0o755 });

run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  sitePackages,
  '--only-binary',
  ':all:',
  '--requirements',
  requirementsPath,
]);
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  sitePackages,
  '--only-binary',
  ':all:',
  '--no-deps',
  engineWheel,
  protocolWheel,
  depthNativeWheel,
]);
// Advertise worker chat-v2 support only after installing a wheel that contains
// the reviewed implementation. Older separately installed packs keep legacy
// payloads and limits until rebuilt with these workers.
run(bundledPython, [
  '-I',
  '-B',
  '-c',
  'from oryn_engine.chat_v2 import DEFAULT_TOKENS, MAX_TOKENS; from oryn_protocol.task_contracts import TASK_CONTRACTS; assert DEFAULT_TOKENS == 4096 and MAX_TOKENS == 8192 and "chat-v2" in TASK_CONTRACTS',
]);
const mlxFoundation = specification.runtimeFoundations?.['oryn.runtime.mlx'];
if (!mlxFoundation) fail('runtime-spec.json must define the shared MLX foundation.');
const transformersPack = specification.runtimePacks['oryn.transformers-pytorch'];
const transformersSitePackages = join(
  engineRoot,
  'runtime-packs',
  'oryn.transformers-pytorch',
  'site-packages',
);
mkdirSync(transformersSitePackages, { recursive: true, mode: 0o755 });
writeFileSync(
  join(transformersSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.transformers-pytorch',
      version: transformersPack.version,
      dependencyClosure: transformersPack.dependencyClosure,
      inheritBaseRuntime: true,
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);

const mlxFoundationRoot = join(engineRoot, 'runtime-foundations', 'oryn.runtime.mlx');
const onnxPack = specification.runtimePacks['oryn.onnx-runtime'];
const onnxSitePackages = join(engineRoot, 'runtime-packs', 'oryn.onnx-runtime', 'site-packages');
mkdirSync(onnxSitePackages, { recursive: true, mode: 0o755 });
writeFileSync(
  join(onnxSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.onnx-runtime',
      version: onnxPack.version,
      dependencyClosure: onnxPack.dependencyClosure,
      inheritBaseRuntime: true,
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const mlxFoundationSitePackages = join(mlxFoundationRoot, 'site-packages');
mkdirSync(mlxFoundationSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  mlxFoundationSitePackages,
  '--requirements',
  mlxFoundationRequirementsPath,
]);
writeFileSync(
  join(mlxFoundationSitePackages, 'oryn-runtime-foundation.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      foundationId: 'oryn.runtime.mlx',
      version: mlxFoundation.version,
      dependencyClosure: mlxFoundation.dependencyClosure,
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const aceStepPack = specification.runtimePacks?.['oryn.ace-step-mlx'];
if (!aceStepPack) fail('runtime-spec.json must define the ACE-Step runtime pack.');
const aceStepPackRoot = join(engineRoot, 'runtime-packs', 'oryn.ace-step-mlx');
const aceStepSitePackages = join(aceStepPackRoot, 'site-packages');
mkdirSync(aceStepSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  aceStepSitePackages,
  '--no-deps',
  '--requirements',
  aceStepRequirementsPath,
]);
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  aceStepSitePackages,
  '--no-deps',
  engineWheel,
]);
writeFileSync(
  join(aceStepSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.ace-step-mlx',
      version: aceStepPack.version,
      dependencyClosure: aceStepPack.dependencyClosure,
      sourceRevision: aceStepPack.sourceRevision,
      requirementsSha256: digest(aceStepRequirementsPath),
      foundation: {
        id: 'oryn.runtime.mlx',
        version: mlxFoundation.version,
        dependencyClosure: mlxFoundation.dependencyClosure,
      },
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const mfluxPack = specification.runtimePacks?.['oryn.mflux-mlx'];
if (!mfluxPack) fail('runtime-spec.json must define the MFlux adapter pack.');
if (
  mfluxPack.foundationId !== 'oryn.runtime.mlx' ||
  mfluxPack.foundationVersion !== mlxFoundation.version
) {
  fail('The MFlux adapter must use the packaged shared MLX foundation.');
}
const mfluxPackRoot = join(engineRoot, 'runtime-packs', 'oryn.mflux-mlx');
const mfluxSitePackages = join(mfluxPackRoot, 'site-packages');
for (const artifact of [mfluxPack.wheel, mfluxPack.providerWheel]) {
  if (digest(join(browserRoot, 'vendor', artifact.file)) !== artifact.sha256) {
    fail('The MFLUX provider build differs from runtime-spec.json.');
  }
}
mkdirSync(mfluxSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  mfluxSitePackages,
  '--no-deps',
  '--require-hashes',
  '--find-links',
  join(browserRoot, 'vendor'),
  '--requirements',
  mfluxRequirementsPath,
]);
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  mfluxSitePackages,
  '--no-deps',
  join(browserRoot, 'vendor', mfluxPack.providerWheel.file),
]);
writeFileSync(
  join(mfluxSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.mflux-mlx',
      version: mfluxPack.version,
      dependencyClosure: mfluxPack.dependencyClosure,
      inheritBaseRuntime: true,
      requirementsSha256: digest(mfluxRequirementsPath),
      foundation: {
        id: 'oryn.runtime.mlx',
        version: mlxFoundation.version,
        dependencyClosure: mlxFoundation.dependencyClosure,
      },
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const kokoroPack = specification.runtimePacks?.['oryn.kokoro-onnx'];
if (!kokoroPack) fail('runtime-spec.json must define the Kokoro ONNX runtime pack.');
const kokoroPackRoot = join(engineRoot, 'runtime-packs', 'oryn.kokoro-onnx');
const kokoroSitePackages = join(kokoroPackRoot, 'site-packages');
mkdirSync(kokoroSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  kokoroSitePackages,
  '--requirements',
  kokoroRequirementsPath,
]);
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  kokoroSitePackages,
  '--no-deps',
  engineWheel,
]);
writeFileSync(
  join(kokoroSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.kokoro-onnx',
      version: kokoroPack.version,
      dependencyClosure: kokoroPack.dependencyClosure,
      requirementsSha256: digest(kokoroRequirementsPath),
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const qwenTtsPack = specification.runtimePacks?.['oryn.qwen-tts-pytorch'];
if (!qwenTtsPack || qwenTtsPack.inheritBaseRuntime !== false)
  fail('The Qwen TTS pack must have a separate dependency closure.');
const qwenTtsPackRoot = join(engineRoot, 'runtime-packs', 'oryn.qwen-tts-pytorch');
const qwenTtsSitePackages = join(qwenTtsPackRoot, 'site-packages');
mkdirSync(qwenTtsSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  qwenTtsSitePackages,
  '--require-hashes',
  '--requirements',
  qwenTtsRequirementsPath,
]);
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  qwenTtsSitePackages,
  '--no-deps',
  engineWheel,
]);
writeFileSync(
  join(qwenTtsSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.qwen-tts-pytorch',
      version: qwenTtsPack.version,
      dependencyClosure: qwenTtsPack.dependencyClosure,
      inheritBaseRuntime: false,
      requirementsSha256: digest(qwenTtsRequirementsPath),
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const diffusersPack = specification.runtimePacks?.['oryn.diffusers-pytorch'];
if (!diffusersPack) fail('runtime-spec.json must define the Diffusers runtime pack.');
if (diffusersPack.inheritBaseRuntime !== true) {
  fail('The Diffusers pack must explicitly inherit the compatible base runtime.');
}
const diffusersWheel = join(browserRoot, 'vendor', diffusersPack.wheel.file);
if (digest(diffusersWheel) !== diffusersPack.wheel.sha256) {
  fail('The pinned Diffusers source-build wheel changed; rebuild and review its provenance.');
}
const diffusersPackRoot = join(engineRoot, 'runtime-packs', 'oryn.diffusers-pytorch');
const diffusersSitePackages = join(diffusersPackRoot, 'site-packages');
mkdirSync(diffusersSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  diffusersSitePackages,
  '--no-deps',
  '--find-links',
  join(browserRoot, 'vendor'),
  '--require-hashes',
  '--requirements',
  diffusersRequirementsPath,
]);
writeFileSync(
  join(diffusersSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.diffusers-pytorch',
      version: diffusersPack.version,
      dependencyClosure: diffusersPack.dependencyClosure,
      inheritBaseRuntime: true,
      requirementsSha256: digest(diffusersRequirementsPath),
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const llamaCppPack = specification.runtimePacks?.['oryn.llama-cpp'];
if (!llamaCppPack) fail('runtime-spec.json must define the llama.cpp runtime pack.');
if (
  llamaCppPack.nativeBackend !== 'metal' ||
  !/^\d+$/u.test(llamaCppPack.build ?? '') ||
  !/^\d+\.\d+\.\d+$/u.test(llamaCppPack.libraryVersion ?? '') ||
  !/^\d+\.\d+\.\d+$/u.test(llamaCppPack.ggmlVersion ?? '') ||
  llamaCppPack.asset?.file !== `llama-b${llamaCppPack.build}-bin-macos-arm64.tar.gz` ||
  !/^https:\/\/github\.com\/ggml-org\/llama\.cpp\/releases\/download\//u.test(
    llamaCppPack.asset?.url ?? '',
  ) ||
  !/^[a-f0-9]{64}$/u.test(llamaCppPack.asset?.sha256 ?? '')
) {
  fail('The llama.cpp macOS native asset specification is invalid.');
}
const llamaCppPackRoot = join(engineRoot, 'runtime-packs', 'oryn.llama-cpp');
const llamaCppSitePackages = join(llamaCppPackRoot, 'site-packages');
const llamaCppNative = join(llamaCppPackRoot, 'native');
const llamaCppDownloads = join(stageRoot, 'downloads');
const llamaCppExtract = join(stageRoot, 'llama-cpp-extract');
const llamaCppArchive = join(llamaCppDownloads, llamaCppPack.asset.file);
mkdirSync(llamaCppSitePackages, { recursive: true, mode: 0o755 });
mkdirSync(llamaCppNative, { recursive: true, mode: 0o755 });
mkdirSync(llamaCppDownloads, { recursive: true, mode: 0o700 });
mkdirSync(llamaCppExtract, { recursive: true, mode: 0o700 });
run('curl', [
  '--fail',
  '--location',
  '--proto',
  '=https',
  '--tlsv1.2',
  '--output',
  llamaCppArchive,
  llamaCppPack.asset.url,
]);
if (digest(llamaCppArchive) !== llamaCppPack.asset.sha256) {
  fail('The downloaded llama.cpp native archive digest is invalid.');
}
const llamaCppDirectory = `llama-b${llamaCppPack.build}`;
const llamaLibrary = (name) => `${name}.${llamaCppPack.libraryVersion}.dylib`;
const ggmlLibrary = (name) => `${name}.${llamaCppPack.ggmlVersion}.dylib`;
const llamaLibraryMajor = llamaCppPack.libraryVersion.split('.')[0];
const ggmlLibraryMajor = llamaCppPack.ggmlVersion.split('.')[0];
const archiveEntries = capture('tar', ['-tzf', llamaCppArchive]).split('\n').filter(Boolean);
if (
  archiveEntries.length === 0 ||
  archiveEntries.some(
    (entry) =>
      !entry.startsWith(`${llamaCppDirectory}/`) ||
      isAbsolute(entry) ||
      entry.split('/').includes('..') ||
      entry.includes('\\'),
  )
) {
  fail('The llama.cpp native archive contains an invalid path.');
}
run('tar', ['-xzf', llamaCppArchive, '-C', llamaCppExtract]);
const llamaCppSource = join(llamaCppExtract, llamaCppDirectory);
const llamaCppNativeFiles = [
  'llama-completion',
  'llama-mtmd-cli',
  llamaLibrary('libmtmd'),
  'libllama-completion-impl.dylib',
  llamaLibrary('libllama-common'),
  llamaLibrary('libllama'),
  ...['libggml', 'libggml-cpu', 'libggml-blas', 'libggml-metal', 'libggml-rpc', 'libggml-base'].map(
    ggmlLibrary,
  ),
  'LICENSE',
];
for (const file of llamaCppNativeFiles) {
  const source = join(llamaCppSource, file);
  if (!existsSync(source)) fail(`The llama.cpp native archive omits ${file}.`);
  copyFileSync(source, join(llamaCppNative, file));
}
chmodSync(join(llamaCppNative, 'llama-completion'), 0o755);
chmodSync(join(llamaCppNative, 'llama-mtmd-cli'), 0o755);
for (const [link, target] of [
  [`libmtmd.${llamaLibraryMajor}.dylib`, llamaLibrary('libmtmd')],
  ['libmtmd.dylib', llamaLibrary('libmtmd')],
  [`libllama-common.${llamaLibraryMajor}.dylib`, llamaLibrary('libllama-common')],
  [`libllama.${llamaLibraryMajor}.dylib`, llamaLibrary('libllama')],
  ...['libggml', 'libggml-cpu', 'libggml-blas', 'libggml-metal', 'libggml-rpc', 'libggml-base'].map(
    (name) => [`${name}.${ggmlLibraryMajor}.dylib`, ggmlLibrary(name)],
  ),
]) {
  symlinkSync(target, join(llamaCppNative, link));
}
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  llamaCppSitePackages,
  '--no-deps',
  engineWheel,
  `pillow==${specification.pillow}`,
]);
writeFileSync(
  join(llamaCppSitePackages, 'oryn-runtime-pack.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.llama-cpp',
      version: llamaCppPack.version,
      chatV2: true,
      dependencyClosure: llamaCppPack.dependencyClosure,
      sourceRevision: llamaCppPack.sourceRevision,
      archiveSha256: llamaCppPack.asset.sha256,
      nativeBackend: llamaCppPack.nativeBackend,
    },
    null,
    2,
  )}\n`,
  { mode: 0o644 },
);
const mlxLmPack = specification.runtimePacks['oryn.mlx-lm'];
const mlxLmRoot = join(engineRoot, 'runtime-packs', 'oryn.mlx-lm');
const mlxLmSitePackages = join(mlxLmRoot, 'site-packages');
const mlxLmRequirements = join(browserRoot, 'packaging', 'mlx-lm-macos-arm64-py312.lock.txt');
mkdirSync(mlxLmSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  mlxLmSitePackages,
  '--no-deps',
  '--require-hashes',
  '--requirements',
  mlxLmRequirements,
]);
writeFileSync(
  join(mlxLmSitePackages, 'oryn-runtime-pack.json'),
  JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.mlx-lm',
      version: mlxLmPack.version,
      chatV2: true,
      dependencyClosure: mlxLmPack.dependencyClosure,
      inheritBaseRuntime: true,
      requirementsSha256: digest(mlxLmRequirements),
      foundation: {
        id: 'oryn.runtime.mlx',
        version: mlxFoundation.version,
        dependencyClosure: mlxFoundation.dependencyClosure,
      },
    },
    null,
    2,
  ) + '\n',
);
const mlxAudioPack = specification.runtimePacks['oryn.mlx-audio'];
const mlxAudioRoot = join(engineRoot, 'runtime-packs', 'oryn.mlx-audio');
const mlxAudioSitePackages = join(mlxAudioRoot, 'site-packages');
const mlxAudioRequirements = join(browserRoot, 'packaging', 'mlx-audio-macos-arm64-py312.lock.txt');
mkdirSync(mlxAudioSitePackages, { recursive: true, mode: 0o755 });
run('uv', [
  'pip',
  'install',
  '--python',
  bundledPython,
  '--target',
  mlxAudioSitePackages,
  '--no-deps',
  '--require-hashes',
  '--requirements',
  mlxAudioRequirements,
]);
writeFileSync(
  join(mlxAudioSitePackages, 'oryn-runtime-pack.json'),
  JSON.stringify(
    {
      schemaVersion: 1,
      packId: 'oryn.mlx-audio',
      version: mlxAudioPack.version,
      dependencyClosure: mlxAudioPack.dependencyClosure,
      inheritBaseRuntime: true,
      requirementsSha256: digest(mlxAudioRequirements),
      foundation: {
        id: 'oryn.runtime.mlx',
        version: mlxFoundation.version,
        dependencyClosure: mlxFoundation.dependencyClosure,
      },
    },
    null,
    2,
  ) + '\n',
);
removeBytecode(mlxAudioRoot);
removeBytecode(mlxLmRoot);
removeBytecode(runtimeRoot);
removeBytecode(mlxFoundationRoot);
removeBytecode(aceStepPackRoot);
removeBytecode(mfluxPackRoot);
removeBytecode(kokoroPackRoot);
removeBytecode(qwenTtsPackRoot);
removeBytecode(diffusersPackRoot);
removeBytecode(llamaCppPackRoot);
run(
  bundledPython,
  [
    '-I',
    '-m',
    'compileall',
    '-q',
    '-f',
    '-s',
    runtimeRoot,
    '-p',
    '/oryn-runtime',
    join(runtimeRoot, 'lib', 'python3.12'),
  ],
  isolatedPythonEnvironment(),
);
for (const [sitePackagesPath, prependDirectory] of [
  [mlxFoundationSitePackages, '/oryn-runtime-foundations/oryn.runtime.mlx'],
  [mfluxSitePackages, '/oryn-runtime-packs/oryn.mflux-mlx'],
  [mlxLmSitePackages, '/oryn-runtime-packs/oryn.mlx-lm'],
  [mlxAudioSitePackages, '/oryn-runtime-packs/oryn.mlx-audio'],
  [qwenTtsSitePackages, '/oryn-runtime-packs/oryn.qwen-tts-pytorch'],
]) {
  run(
    bundledPython,
    [
      '-I',
      '-S',
      '-B',
      '-c',
      [
        'import compileall,site',
        `site.addsitedir(${JSON.stringify(sitePackagesPath)})`,
        `raise SystemExit(0 if compileall.compile_dir(${JSON.stringify(
          sitePackagesPath,
        )},quiet=1,force=True,stripdir=${JSON.stringify(
          sitePackagesPath,
        )},prependdir=${JSON.stringify(prependDirectory)}) else 1)`,
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  );
}
run(
  bundledPython,
  [
    '-I',
    '-S',
    '-B',
    '-c',
    [
      'import compileall,site',
      `site.addsitedir(${JSON.stringify(llamaCppSitePackages)})`,
      `raise SystemExit(0 if compileall.compile_dir(${JSON.stringify(
        llamaCppSitePackages,
      )},quiet=1,force=True,stripdir=${JSON.stringify(
        llamaCppSitePackages,
      )},prependdir='/oryn-runtime-packs/oryn.llama-cpp') else 1)`,
    ].join(';'),
  ],
  isolatedPythonEnvironment(),
);
run(
  bundledPython,
  [
    '-I',
    '-S',
    '-B',
    '-c',
    [
      'import compileall,site',
      `site.addsitedir(${JSON.stringify(aceStepSitePackages)})`,
      `raise SystemExit(0 if compileall.compile_dir(${JSON.stringify(
        aceStepSitePackages,
      )},quiet=1,force=True,stripdir=${JSON.stringify(
        aceStepSitePackages,
      )},prependdir='/oryn-runtime-packs/oryn.ace-step-mlx') else 1)`,
    ].join(';'),
  ],
  isolatedPythonEnvironment(),
);
run(
  bundledPython,
  [
    '-I',
    '-S',
    '-B',
    '-c',
    [
      'import compileall,site',
      `site.addsitedir(${JSON.stringify(kokoroSitePackages)})`,
      `raise SystemExit(0 if compileall.compile_dir(${JSON.stringify(
        kokoroSitePackages,
      )},quiet=1,force=True,stripdir=${JSON.stringify(
        kokoroSitePackages,
      )},prependdir='/oryn-runtime-packs/oryn.kokoro-onnx') else 1)`,
    ].join(';'),
  ],
  isolatedPythonEnvironment(),
);
run(
  bundledPython,
  [
    '-I',
    '-B',
    '-c',
    [
      'import compileall,site',
      `site.addsitedir(${JSON.stringify(diffusersSitePackages)})`,
      `raise SystemExit(0 if compileall.compile_dir(${JSON.stringify(
        diffusersSitePackages,
      )},quiet=1,force=True,stripdir=${JSON.stringify(
        diffusersSitePackages,
      )},prependdir='/oryn-runtime-packs/oryn.diffusers-pytorch') else 1)`,
    ].join(';'),
  ],
  isolatedPythonEnvironment(),
);
assertBytecodeIsPathNeutral(runtimeRoot);
assertBytecodeIsPathNeutral(mlxFoundationRoot);
assertBytecodeIsPathNeutral(aceStepPackRoot);
assertBytecodeIsPathNeutral(mfluxPackRoot);
assertBytecodeIsPathNeutral(kokoroPackRoot);
assertBytecodeIsPathNeutral(qwenTtsPackRoot);
assertBytecodeIsPathNeutral(diffusersPackRoot);
assertBytecodeIsPathNeutral(llamaCppPackRoot);
assertBytecodeIsPathNeutral(mlxLmRoot);
assertBytecodeIsPathNeutral(mlxAudioRoot);

const versions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-B',
      '-c',
      [
        'import json,platform',
        'from importlib.metadata import version',
        'from transformers import AutoModelForImageTextToText,Qwen3_5ForConditionalGeneration',
        'import h5py,sacremoses,tiktoken,timesfm3,depth_anything_3,cv2,ctranslate2',
        "names=['oryn-engine','oryn-protocol','torch','torchvision','timm','open-clip-torch','ftfy','librosa','ctranslate2','chronos-forecasting','timesfm','tabpfn','autogluon.tabular','pillow','transformers','av','pandas','onnxruntime','h5py','sacremoses','tiktoken','oryn-depth-native','opencv-python-headless','addict','omegaconf']",
        "print(json.dumps({'python':platform.python_version(),**{name:version(name) for name in names}}))",
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const expected = {
  python: specification.python,
  'oryn-engine': specification.engine,
  'oryn-protocol': specification.protocol,
  torch: specification.torch,
  torchvision: specification.torchvision,
  timm: specification.timm,
  'open-clip-torch': specification.openClip,
  ftfy: specification.ftfy,
  librosa: specification.librosa,
  ctranslate2: specification.ctranslate2,
  'chronos-forecasting': specification.chronosForecasting,
  timesfm: specification.timesfm,
  tabpfn: specification.tabpfn,
  'autogluon.tabular': specification.autogluonTabular,
  pillow: specification.pillow,
  transformers: specification.transformers,
  av: specification.av,
  pandas: specification.pandas,
  onnxruntime: specification.onnxruntime,
  h5py: specification.h5py,
  sacremoses: specification.sacremoses,
  tiktoken: specification.tiktoken,
  'oryn-depth-native': specification.orynDepthNative,
  'opencv-python-headless': specification.opencvPythonHeadless,
  addict: specification.addict,
  omegaconf: specification.omegaconf,
};
if (JSON.stringify(versions) !== JSON.stringify(expected)) {
  fail(`Bundled runtime versions differ from runtime-spec.json: ${JSON.stringify(versions)}`);
}

const packCatalog = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-B',
      '-c',
      [
        'import json',
        'from oryn_engine.runtime_packs.builtin_pack_catalog import builtin_pack_catalog',
        'catalog=builtin_pack_catalog()',
        "print(json.dumps({'version':catalog.catalog_version,'digest':catalog.digest,'publisher':catalog.publisher,'packs':sorted(pack.pack_id for pack in catalog.packs)}))",
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const expectedPackIds = [
  'oryn.ace-step-mlx',
  'oryn.diffusers-pytorch',
  'oryn.kokoro-onnx',
  'oryn.llama-cpp',
  'oryn.mflux-mlx',
  'oryn.mlx-audio',
  'oryn.mlx-lm',
  'oryn.onnx-runtime',
  'oryn.qwen-tts-pytorch',
  'oryn.transformers-pytorch',
];
if (
  packCatalog.version !== 8 ||
  !/^[a-f0-9]{64}$/u.test(packCatalog.digest ?? '') ||
  packCatalog.publisher !== 'oryn.first-party' ||
  JSON.stringify(packCatalog.packs) !== JSON.stringify(expectedPackIds)
) {
  fail(`The signed first-party runtime-pack catalog is invalid: ${JSON.stringify(packCatalog)}`);
}

const mlxAudioVersions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-B',
      '-c',
      [
        'import json,sys',
        'from pathlib import Path',
        'from importlib.metadata import version',
        `sys.path[:0]=[${JSON.stringify(mlxFoundationSitePackages)},${JSON.stringify(mlxAudioSitePackages)}]`,
        'from oryn_engine.generated_capabilities import verify_audio_build',
        `verify_audio_build(Path(${JSON.stringify(join(mlxAudioSitePackages, 'mlx_audio'))}))`,
        'from mlx_audio.tts.utils import load_model as load_tts',
        'from mlx_audio.stt.utils import load_model as load_stt',
        "names=['mlx-audio','mlx-lm','pillow','mlx']",
        'print(json.dumps({name:version(name) for name in names}))',
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const expectedMlxAudioVersions = {
  'mlx-audio': mlxAudioPack.mlxAudio,
  'mlx-lm': mlxAudioPack.mlxLm,
  pillow: mlxAudioPack.pillow,
  mlx: mlxFoundation.mlx,
};
if (JSON.stringify(mlxAudioVersions) !== JSON.stringify(expectedMlxAudioVersions)) {
  fail(`MLX-Audio versions differ from runtime-spec.json: ${JSON.stringify(mlxAudioVersions)}`);
}

const aceStepVersions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-S',
      '-B',
      '-c',
      [
        'import json,site',
        'from importlib.metadata import version',
        `site.addsitedir(${JSON.stringify(mlxFoundationSitePackages)})`,
        `site.addsitedir(${JSON.stringify(aceStepSitePackages)})`,
        "names=['ace-step','torch','transformers','mlx','mlx-lm']",
        'print(json.dumps({name:version(name) for name in names}))',
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const mlxFoundationVersions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-S',
      '-B',
      '-c',
      [
        'import json,site',
        'from importlib.metadata import version',
        `site.addsitedir(${JSON.stringify(mlxFoundationSitePackages)})`,
        "print(json.dumps({'mlx':version('mlx')}))",
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
if (mlxFoundationVersions.mlx !== mlxFoundation.mlx) {
  fail(
    `MLX foundation version differs from runtime-spec.json: ${JSON.stringify(mlxFoundationVersions)}`,
  );
}
const mfluxVersions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-B',
      '-c',
      [
        'import json,site,sys',
        'from importlib.metadata import version',
        `sys.path[:0]=[${JSON.stringify(mlxFoundationSitePackages)},${JSON.stringify(mfluxSitePackages)}]`,
        `site.addsitedir(${JSON.stringify(mlxFoundationSitePackages)})`,
        `site.addsitedir(${JSON.stringify(mfluxSitePackages)})`,
        "names=['mflux','mlx','torch','transformers','pillow']",
        'print(json.dumps({name:version(name) for name in names}))',
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const expectedMfluxVersions = {
  mflux: mfluxPack.mflux,
  mlx: mlxFoundation.mlx,
  torch: mfluxPack.torch,
  transformers: mfluxPack.transformers,
  pillow: mfluxPack.pillow,
};
if (JSON.stringify(mfluxVersions) !== JSON.stringify(expectedMfluxVersions)) {
  fail(`MFlux adapter versions differ from runtime-spec.json: ${JSON.stringify(mfluxVersions)}`);
}
const expectedAceStepVersions = {
  'ace-step': aceStepPack.aceStep,
  torch: aceStepPack.torch,
  transformers: aceStepPack.transformers,
  mlx: aceStepPack.mlx,
  'mlx-lm': aceStepPack.mlxLm,
};
if (JSON.stringify(aceStepVersions) !== JSON.stringify(expectedAceStepVersions)) {
  fail(`ACE-Step pack versions differ from runtime-spec.json: ${JSON.stringify(aceStepVersions)}`);
}

const kokoroVersions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-S',
      '-B',
      '-c',
      [
        'import json,site',
        'from importlib.metadata import version',
        `site.addsitedir(${JSON.stringify(kokoroSitePackages)})`,
        "names=['onnxruntime','phonemizer-fork','espeakng-loader']",
        'print(json.dumps({name:version(name) for name in names}))',
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const expectedKokoroVersions = {
  onnxruntime: kokoroPack.onnxruntime,
  'phonemizer-fork': kokoroPack.phonemizerFork,
  'espeakng-loader': kokoroPack.espeakngLoader,
};
if (JSON.stringify(kokoroVersions) !== JSON.stringify(expectedKokoroVersions)) {
  fail(`Kokoro pack versions differ from runtime-spec.json: ${JSON.stringify(kokoroVersions)}`);
}

const qwenTtsVersions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-S',
      '-B',
      '-c',
      [
        'import json,site',
        `site.addsitedir(${JSON.stringify(qwenTtsSitePackages)})`,
        'from oryn_engine.qwen_tts import verify_runtime',
        'print(json.dumps(verify_runtime()))',
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const expectedQwenTtsVersions = {
  'qwen-tts': qwenTtsPack.qwenTts,
  transformers: qwenTtsPack.transformers,
  torch: qwenTtsPack.torch,
  torchaudio: qwenTtsPack.torchaudio,
  accelerate: qwenTtsPack.accelerate,
};
if (
  Object.entries(expectedQwenTtsVersions).some(
    ([name, version]) => qwenTtsVersions[name] !== version,
  )
) {
  fail(`Qwen TTS pack versions differ from runtime-spec.json: ${JSON.stringify(qwenTtsVersions)}`);
}
const diffusersVersions = JSON.parse(
  capture(
    bundledPython,
    [
      '-I',
      '-B',
      '-c',
      [
        'import json,site',
        'from importlib.metadata import version',
        `site.addsitedir(${JSON.stringify(diffusersSitePackages)})`,
        "names=['diffusers','accelerate','bitsandbytes','imageio','imageio-ffmpeg','torch']",
        'print(json.dumps({name:version(name) for name in names}))',
      ].join(';'),
    ],
    isolatedPythonEnvironment(),
  ),
);
const expectedDiffusersVersions = {
  diffusers: diffusersPack.diffusers,
  accelerate: diffusersPack.accelerate,
  bitsandbytes: diffusersPack.bitsandbytes,
  imageio: diffusersPack.imageio,
  'imageio-ffmpeg': diffusersPack.imageioFfmpeg,
  torch: diffusersPack.torch,
};
if (JSON.stringify(diffusersVersions) !== JSON.stringify(expectedDiffusersVersions)) {
  fail(
    `Diffusers pack versions differ from runtime-spec.json: ${JSON.stringify(diffusersVersions)}`,
  );
}
const llamaCppVersion = captureCombined(join(llamaCppNative, 'llama-completion'), ['--version'], {
  ...isolatedPythonEnvironment(),
  DYLD_LIBRARY_PATH: llamaCppNative,
  GGML_METAL_PATH_RESOURCES: llamaCppNative,
});
const llamaCppIdentity = /version:\s+\S+\s+\(build\s+(\d+),\s+commit\s+([0-9a-f]+)\)/u.exec(
  llamaCppVersion,
);
if (
  llamaCppIdentity?.[1] !== llamaCppPack.build ||
  !llamaCppPack.sourceRevision.startsWith(llamaCppIdentity?.[2] ?? 'invalid')
) {
  fail(`llama.cpp native identity differs from runtime-spec.json: ${llamaCppVersion}`);
}

capture(
  bundledPython,
  [
    '-I',
    '-B',
    '-c',
    [
      'from pathlib import Path',
      'from oryn_engine.generated_capabilities import verify_build_inputs',
      `verify_build_inputs(mlx_package=Path(${JSON.stringify(join(mlxLmSitePackages, 'mlx_lm'))}), diffusers_package=Path(${JSON.stringify(join(diffusersSitePackages, 'diffusers'))}), llama_binary=Path(${JSON.stringify(join(engineRoot, 'runtime-packs', 'oryn.llama-cpp', 'native', 'llama-completion'))}))`,
    ].join(';'),
  ],
  isolatedPythonEnvironment(),
);

const manifest = {
  schemaVersion: 1,
  platform: specification.platform,
  assembledAt: new Date().toISOString(),
  engineRevision,
  versions,
  artifacts: {
    packCatalog,
    engineWheel: { file: engineWheel.split('/').at(-1), sha256: digest(engineWheel) },
    protocolWheel: { file: protocolWheel.split('/').at(-1), sha256: digest(protocolWheel) },
    requirements: {
      file: requirementsPath.split('/').at(-1),
      sha256: digest(requirementsPath),
    },
    runtimePacks: {
      'oryn.onnx-runtime': {
        version: onnxPack.version,
        dependencyClosure: onnxPack.dependencyClosure,
        inheritsBaseRuntime: true,
        markerSha256: digest(join(onnxSitePackages, 'oryn-runtime-pack.json')),
      },
      'oryn.transformers-pytorch': {
        version: transformersPack.version,
        dependencyClosure: transformersPack.dependencyClosure,
        inheritsBaseRuntime: true,
        markerSha256: digest(join(transformersSitePackages, 'oryn-runtime-pack.json')),
      },
      'oryn.mlx-lm': {
        requirements: {
          file: 'mlx-lm-macos-arm64-py312.lock.txt',
          sha256: digest(mlxLmRequirements),
        },
        foundation: 'oryn.runtime.mlx',
        inheritsBaseRuntime: true,
        versions: {
          'mlx-lm': mlxLmPack.mlxLm,
          'mlx-vlm': mlxLmPack.mlxVlm,
          mlx: mlxFoundation.mlx,
        },
      },
      'oryn.mlx-audio': {
        requirements: {
          file: 'mlx-audio-macos-arm64-py312.lock.txt',
          sha256: digest(mlxAudioRequirements),
        },
        foundation: 'oryn.runtime.mlx',
        inheritsBaseRuntime: true,
        markerSha256: digest(join(mlxAudioSitePackages, 'oryn-runtime-pack.json')),
        versions: mlxAudioVersions,
      },
      'oryn.ace-step-mlx': {
        requirements: {
          file: aceStepRequirementsPath.split('/').at(-1),
          sha256: digest(aceStepRequirementsPath),
        },
        sourceRevision: aceStepPack.sourceRevision,
        versions: aceStepVersions,
      },
      'oryn.mflux-mlx': {
        requirements: {
          file: mfluxRequirementsPath.split('/').at(-1),
          sha256: digest(mfluxRequirementsPath),
        },
        foundation: 'oryn.runtime.mlx',
        sourceRevision: mfluxPack.sourceRevision,
        wheel: mfluxPack.wheel,
        providerWheel: mfluxPack.providerWheel,
        versions: mfluxVersions,
      },
      'oryn.kokoro-onnx': {
        requirements: {
          file: kokoroRequirementsPath.split('/').at(-1),
          sha256: digest(kokoroRequirementsPath),
        },
        versions: kokoroVersions,
      },
      'oryn.qwen-tts-pytorch': {
        requirements: {
          file: qwenTtsRequirementsPath.split('/').at(-1),
          sha256: digest(qwenTtsRequirementsPath),
        },
        inheritsBaseRuntime: false,
        markerSha256: digest(join(qwenTtsSitePackages, 'oryn-runtime-pack.json')),
        versions: qwenTtsVersions,
      },
      'oryn.diffusers-pytorch': {
        sourceRevision: diffusersPack.sourceRevision,
        wheel: diffusersPack.wheel,
        requirements: {
          file: diffusersRequirementsPath.split('/').at(-1),
          sha256: digest(diffusersRequirementsPath),
        },
        inheritsBaseRuntime: true,
        versions: diffusersVersions,
      },
      'oryn.llama-cpp': {
        sourceRevision: llamaCppPack.sourceRevision,
        build: llamaCppPack.build,
        nativeBackend: llamaCppPack.nativeBackend,
        archive: {
          file: llamaCppPack.asset.file,
          sha256: llamaCppPack.asset.sha256,
        },
        files: Object.fromEntries(
          llamaCppNativeFiles.map((file) => [file, digest(join(llamaCppNative, file))]),
        ),
      },
    },
    runtimeFoundations: {
      'oryn.runtime.mlx': {
        requirements: {
          file: mlxFoundationRequirementsPath.split('/').at(-1),
          sha256: digest(mlxFoundationRequirementsPath),
        },
        versions: mlxFoundationVersions,
      },
    },
  },
};
writeFileSync(join(engineRoot, 'release-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, {
  mode: 0o600,
});
copyFileSync(specificationPath, join(engineRoot, 'runtime-spec.json'));
console.log(`Assembled isolated Oryn Engine runtime at ${engineRoot}`);

function requiredFile(name) {
  const value = process.env[name];
  if (!value) fail(`${name} is required.`);
  const path = resolve(value);
  if (!existsSync(path)) fail(`${name} does not point to a file.`);
  return path;
}

function relativizeInternalSymlinks(destinationRoot, sourceRoot) {
  const pending = [destinationRoot];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        const target = readlinkSync(path);
        if (!isAbsolute(target)) continue;
        const sourceRelation = relative(sourceRoot, target);
        if (
          sourceRelation === '..' ||
          sourceRelation.startsWith(`..${sep}`) ||
          isAbsolute(sourceRelation)
        ) {
          fail(`Managed Python contains an external absolute symlink: ${path}`);
        }
        const packagedTarget = join(destinationRoot, sourceRelation);
        const packagedRelation = relative(dirname(path), packagedTarget) || '.';
        unlinkSync(path);
        symlinkSync(packagedRelation, path);
      } else if (entry.isDirectory()) {
        pending.push(path);
      }
    }
  }
}

function removeBytecode(root) {
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === '__pycache__') {
        rmSync(path, { recursive: true, force: true });
      } else {
        removeBytecode(path);
      }
    } else if (entry.isFile() && entry.name.endsWith('.pyc')) {
      rmSync(path, { force: true });
    }
  }
}

function assertBytecodeIsPathNeutral(root) {
  const forbidden = Buffer.from(browserRoot);
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) pending.push(path);
      if (entry.isFile() && entry.name.endsWith('.pyc') && readFileSync(path).includes(forbidden)) {
        fail(`Packaged bytecode contains a developer checkout path: ${path}`);
      }
    }
  }
}

function isolatedPythonEnvironment() {
  const environment = { ...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1' };
  delete environment.PYTHONHOME;
  delete environment.PYTHONPATH;
  return environment;
}

function digest(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function capture(command, arguments_, environment = process.env) {
  const result = spawnSync(command, arguments_, {
    cwd: browserRoot,
    env: environment,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  if (result.status !== 0) fail(`${command} failed with status ${result.status ?? 'unknown'}.`);
  return result.stdout.trim();
}

function captureCombined(command, arguments_, environment = process.env) {
  const result = spawnSync(command, arguments_, {
    cwd: browserRoot,
    env: environment,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (result.status !== 0) fail(`${command} failed with status ${result.status ?? 'unknown'}.`);
  return `${result.stdout}${result.stderr}`.trim();
}

function run(command, arguments_, environment = process.env) {
  const result = spawnSync(command, arguments_, {
    cwd: browserRoot,
    env: environment,
    stdio: 'inherit',
  });
  if (result.status !== 0) fail(`${command} failed with status ${result.status ?? 'unknown'}.`);
}

function fail(message) {
  throw new Error(message);
}
