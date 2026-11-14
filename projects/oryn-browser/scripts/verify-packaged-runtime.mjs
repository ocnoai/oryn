/* global console, process */
import { Buffer } from 'node:buffer';
import { createHash } from 'node:crypto';
import { lstatSync, readFileSync, readlinkSync, readdirSync, realpathSync } from 'node:fs';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { spawnSync } from 'node:child_process';

const input =
  process.argv.slice(2).find((value) => value !== '--') ?? process.env.ORYN_PACKAGED_APP;
if (!input) throw new Error('Pass an Oryn Browser.app path or set ORYN_PACKAGED_APP.');
const resolvedInput = resolve(input);
const engineRoot = resolvedInput.endsWith('.app')
  ? join(resolvedInput, 'Contents', 'Resources', 'engine')
  : resolvedInput;
if (resolvedInput.endsWith('.app')) {
  for (const name of ['artifact_io.py', 'artifact_io_windows.py']) {
    const helper = join(resolvedInput, 'Contents', 'Resources', 'helpers', name);
    const helperStat = lstatSync(helper);
    if (!helperStat.isFile() || helperStat.isSymbolicLink())
      throw new Error('Packaged Browser artifact helper is missing or invalid.');
  }
}
const canonicalRoot = realpathSync(engineRoot);
assertRelativeSymlinks(canonicalRoot);
assertRelocatableBytecode(canonicalRoot);

const manifest = JSON.parse(readFileSync(join(canonicalRoot, 'release-manifest.json'), 'utf8'));
const specification = JSON.parse(readFileSync(join(canonicalRoot, 'runtime-spec.json'), 'utf8'));
const python = join(canonicalRoot, 'runtime', 'bin', 'python3.12');
const environment = { ...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1' };
delete environment.PYTHONHOME;
delete environment.PYTHONPATH;
const probe = spawnSync(
  python,
  [
    '-I',
    '-B',
    '-c',
    [
      'import json,platform',
      'from importlib.metadata import version',
      'import oryn_engine,oryn_protocol,torch,torchvision,timm,open_clip,ftfy,librosa,ctranslate2,chronos,timesfm3,transformers,h5py,sacremoses,tiktoken,depth_anything_3,cv2',
      'from PIL import Image',
      'from transformers import AutoModelForImageTextToText,Qwen3_5ForConditionalGeneration',
      "names=['oryn-engine','oryn-protocol','torch','torchvision','timm','open-clip-torch','ftfy','librosa','ctranslate2','chronos-forecasting','timesfm','tabpfn','autogluon.tabular','pillow','transformers','av','pandas','onnxruntime','h5py','sacremoses','tiktoken','oryn-depth-native','opencv-python-headless','addict','omegaconf','scipy','sentencepiece','protobuf']",
      "print(json.dumps({'python':platform.python_version(),**{name:version(name) for name in names}}))",
    ].join(';'),
  ],
  { cwd: canonicalRoot, env: environment, encoding: 'utf8' },
);
if (probe.status !== 0) {
  process.stderr.write(probe.stderr);
  throw new Error('The bundled isolated Python runtime could not import Oryn and ML dependencies.');
}
const versions = JSON.parse(probe.stdout.trim());
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
if (
  Object.entries(expected).some(([name, value]) => versions[name] !== value) ||
  ['scipy', 'sentencepiece', 'protobuf'].some((name) => versions[name] !== specification[name])
) {
  throw new Error(`Packaged runtime version mismatch: ${JSON.stringify(versions)}`);
}
if (JSON.stringify(manifest.versions) !== JSON.stringify(expected)) {
  throw new Error('The release manifest does not match the packaged runtime.');
}
const catalogProbe = spawnSync(
  python,
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
  { cwd: canonicalRoot, env: environment, encoding: 'utf8' },
);
if (catalogProbe.status !== 0) {
  process.stderr.write(catalogProbe.stderr);
  throw new Error('The packaged signed runtime-pack catalog could not be verified.');
}
const packCatalog = JSON.parse(catalogProbe.stdout.trim());
const expectedPackIds = [
  'oryn.ace-step-mlx',
  'oryn.diffusers-pytorch',
  'oryn.kokoro-onnx',
  'oryn.llama-cpp',
  'oryn.mflux-mlx',
  'oryn.mlx-lm',
  'oryn.onnx-runtime',
  'oryn.qwen-tts-pytorch',
  'oryn.transformers-pytorch',
];
if (
  packCatalog.version !== 7 ||
  packCatalog.publisher !== 'oryn.first-party' ||
  !/^[a-f0-9]{64}$/u.test(packCatalog.digest ?? '') ||
  JSON.stringify(packCatalog.packs) !== JSON.stringify(expectedPackIds) ||
  JSON.stringify(manifest.artifacts.packCatalog) !== JSON.stringify(packCatalog)
) {
  throw new Error('The release manifest does not bind the signed runtime-pack catalog.');
}
const onnxSitePackages = join(canonicalRoot, 'runtime-packs', 'oryn.onnx-runtime', 'site-packages');
const onnxMarkerPath = join(onnxSitePackages, 'oryn-runtime-pack.json');
const onnxMarker = JSON.parse(readFileSync(onnxMarkerPath, 'utf8'));
if (
  onnxMarker.packId !== 'oryn.onnx-runtime' ||
  onnxMarker.version !== specification.runtimePacks['oryn.onnx-runtime'].version ||
  onnxMarker.dependencyClosure !==
    specification.runtimePacks['oryn.onnx-runtime'].dependencyClosure ||
  createHash('sha256').update(readFileSync(onnxMarkerPath)).digest('hex') !==
    manifest.artifacts.runtimePacks['oryn.onnx-runtime'].markerSha256
) {
  throw new Error('The ONNX closure marker does not match the signed pack.');
}
const onnxProbe = spawnSync(python, ['-I', '-B', '-m', 'oryn_engine.runtime_packs.onnx_worker'], {
  cwd: canonicalRoot,
  env: environment,
  encoding: 'utf8',
  input:
    JSON.stringify({
      abi: '1',
      packId: 'oryn.onnx-runtime',
      version: onnxMarker.version,
      command: 'probe',
      payload: {},
    }) + '\n',
});
const onnxResult = JSON.parse(onnxProbe.stdout.trim() || '{}');
if (
  onnxProbe.status !== 0 ||
  onnxResult.ok !== true ||
  onnxResult.result.onnxruntime !== specification.onnxruntime
) {
  throw new Error('The packaged ONNX worker ABI probe failed.');
}
const aceStepSitePackages = join(
  canonicalRoot,
  'runtime-packs',
  'oryn.ace-step-mlx',
  'site-packages',
);
const mlxFoundationSitePackages = join(
  canonicalRoot,
  'runtime-foundations',
  'oryn.runtime.mlx',
  'site-packages',
);
const mlxLmSitePackages = join(canonicalRoot, 'runtime-packs', 'oryn.mlx-lm', 'site-packages');
const mlxLmProbe = spawnSync(
  python,
  [
    '-I',
    '-B',
    '-c',
    [
      'import site,json,runpy',
      `site.addsitedir(${JSON.stringify(mlxFoundationSitePackages)})`,
      `site.addsitedir(${JSON.stringify(mlxLmSitePackages)})`,
      'from pathlib import Path',
      'from importlib.metadata import version',
      'from oryn_engine.generated_capabilities import verify_build_inputs',
      `verify_build_inputs(mlx_package=Path(${JSON.stringify(join(mlxLmSitePackages, 'mlx_lm'))}), diffusers_package=Path(${JSON.stringify(join(canonicalRoot, 'runtime-packs', 'oryn.diffusers-pytorch', 'site-packages', 'diffusers'))}), llama_binary=Path(${JSON.stringify(join(canonicalRoot, 'runtime-packs', 'oryn.llama-cpp', 'native', 'llama-completion'))}))`,
      'import mlx_lm,mlx_vlm,mlx.core as mx',
      'from oryn_engine.mlx_vision import verify_vision_build',
      'verify_vision_build()',
      'assert mx.metal.is_available()',
      'assert mx.array([1,2]).sum().item()==3',
      "print(json.dumps({'mlx-lm':version('mlx-lm'),'mlx-vlm':version('mlx-vlm'),'mlx':version('mlx')}))",
      "runpy.run_module('oryn_engine.runtime_packs.mlx_lm_worker',run_name='__main__')",
    ].join(';'),
  ],
  {
    cwd: canonicalRoot,
    env: environment,
    encoding: 'utf8',
    input: `${JSON.stringify({
      abi: '1',
      packId: 'oryn.mlx-lm',
      version: specification.runtimePacks['oryn.mlx-lm'].version,
      command: 'probe',
      payload: {},
    })}\n`,
  },
);
const mlxLmResults = mlxLmProbe.stdout
  .trim()
  .split('\n')
  .filter(Boolean)
  .map((line) => JSON.parse(line));
if (
  mlxLmProbe.status !== 0 ||
  mlxLmResults.length !== 2 ||
  JSON.stringify(mlxLmResults[0]) !==
    JSON.stringify(manifest.artifacts.runtimePacks['oryn.mlx-lm']?.versions) ||
  mlxLmResults[1]?.ok !== true ||
  mlxLmResults[1]?.result?.backend !== 'mlx' ||
  mlxLmResults[1]?.result?.available !== true
) {
  process.stderr.write(mlxLmProbe.stderr ?? '');
  throw new Error('The packaged MLX chat adapter or signed capability build is invalid.');
}
const mlxFoundationMarker = JSON.parse(
  readFileSync(join(mlxFoundationSitePackages, 'oryn-runtime-foundation.json'), 'utf8'),
);
if (
  JSON.stringify(mlxFoundationMarker) !==
  JSON.stringify({
    schemaVersion: 1,
    foundationId: 'oryn.runtime.mlx',
    version: specification.runtimeFoundations['oryn.runtime.mlx'].version,
    dependencyClosure: specification.runtimeFoundations['oryn.runtime.mlx'].dependencyClosure,
  })
) {
  throw new Error('The packaged shared MLX foundation marker is invalid.');
}
for (const adapterSite of [
  aceStepSitePackages,
  join(canonicalRoot, 'runtime-packs', 'oryn.mflux-mlx', 'site-packages'),
]) {
  const names = readdirSync(adapterSite);
  if (names.some((name) => name === 'mlx' || /^mlx(?:-|_metal).*\.dist-info$/u.test(name))) {
    throw new Error('An MLX adapter duplicated the shared MLX foundation.');
  }
}
const transformerMarkerPath = join(
  canonicalRoot,
  'runtime-packs',
  'oryn.transformers-pytorch',
  'site-packages',
  'oryn-runtime-pack.json',
);
const transformerMarker = JSON.parse(readFileSync(transformerMarkerPath, 'utf8'));
if (
  transformerMarker.version !== specification.runtimePacks['oryn.transformers-pytorch'].version ||
  transformerMarker.inheritBaseRuntime !== true ||
  createHash('sha256').update(readFileSync(transformerMarkerPath)).digest('hex') !==
    manifest.artifacts.runtimePacks['oryn.transformers-pytorch'].markerSha256
) {
  throw new Error('The shared Transformers worker marker does not match the release.');
}
const transformersProbe = spawnSync(
  python,
  ['-I', '-B', '-m', 'oryn_engine.runtime_packs.transformers_worker'],
  {
    cwd: canonicalRoot,
    env: { ...environment, ORYN_PACK_NETWORK_POLICY: 'deny' },
    encoding: 'utf8',
    timeout: 30000,
    input:
      JSON.stringify({
        abi: '1',
        packId: 'oryn.transformers-pytorch',
        version: transformerMarker.version,
        command: 'probe',
        payload: {},
      }) + '\n',
  },
);
const transformersResult = JSON.parse(transformersProbe.stdout.trim() || '{}');
if (
  transformersProbe.status !== 0 ||
  transformersResult.ok !== true ||
  !Array.isArray(transformersResult.result.taskContracts) ||
  transformersResult.result.taskContracts.length === 0 ||
  transformersResult.result.transformers !== specification.transformers
) {
  throw new Error('The shared Transformers worker ABI probe failed.');
}
const aceStepProbe = spawnSync(
  python,
  [
    '-I',
    '-S',
    '-B',
    '-c',
    [
      'import json,runpy,site',
      `site.addsitedir(${JSON.stringify(mlxFoundationSitePackages)})`,
      `site.addsitedir(${JSON.stringify(aceStepSitePackages)})`,
      "runpy.run_module('oryn_engine.runtime_packs.ace_step_worker',run_name='__main__')",
    ].join(';'),
  ],
  {
    cwd: canonicalRoot,
    env: { ...environment, ORYN_PACK_NETWORK_POLICY: 'deny' },
    encoding: 'utf8',
    input: `${JSON.stringify({
      abi: '1',
      packId: 'oryn.ace-step-mlx',
      version: specification.runtimePacks['oryn.ace-step-mlx'].version,
      command: 'probe',
      payload: {},
    })}\n`,
  },
);
if (aceStepProbe.status !== 0) {
  process.stderr.write(aceStepProbe.stderr);
  throw new Error('The packaged ACE-Step runtime pack failed its isolated ABI probe.');
}
const mfluxSitePackages = join(canonicalRoot, 'runtime-packs', 'oryn.mflux-mlx', 'site-packages');
const mfluxProbe = spawnSync(
  python,
  [
    '-I',
    '-B',
    '-c',
    [
      'import json,runpy,site,sys',
      `sys.path[:0]=[${JSON.stringify(mlxFoundationSitePackages)},${JSON.stringify(mfluxSitePackages)}]`,
      `site.addsitedir(${JSON.stringify(mlxFoundationSitePackages)})`,
      `site.addsitedir(${JSON.stringify(mfluxSitePackages)})`,
      'from importlib.metadata import version',
      `assert version('pillow') == ${JSON.stringify(specification.runtimePacks['oryn.mflux-mlx'].pillow)}`,
      "runpy.run_module('oryn_mflux_provider.worker',run_name='__main__')",
    ].join(';'),
  ],
  {
    cwd: canonicalRoot,
    env: {
      ...environment,
      ORYN_PACK_NETWORK_POLICY: 'deny',
      HF_HUB_OFFLINE: '1',
      TRANSFORMERS_OFFLINE: '1',
    },
    encoding: 'utf8',
    input: `${JSON.stringify({
      abi: '1',
      packId: 'oryn.mflux-mlx',
      version: specification.runtimePacks['oryn.mflux-mlx'].version,
      command: 'probe',
      payload: {},
    })}\n`,
  },
);
if (mfluxProbe.status !== 0) {
  process.stderr.write(mfluxProbe.stderr);
  throw new Error('The packaged MFlux adapter failed its shared-foundation ABI probe.');
}
const mfluxResult = JSON.parse(mfluxProbe.stdout.trim());
if (
  mfluxResult.type !== 'result' ||
  mfluxResult.ok !== true ||
  mfluxResult.result?.foundation !== 'oryn.runtime.mlx' ||
  mfluxResult.result?.adapter !== 'oryn.adapter.mflux-image' ||
  mfluxResult.result?.networkPolicy !== 'deny'
) {
  throw new Error('The packaged MFlux adapter returned an invalid ABI response.');
}
const aceStepResult = JSON.parse(aceStepProbe.stdout.trim());
if (
  aceStepResult.type !== 'result' ||
  aceStepResult.ok !== true ||
  aceStepResult.result?.networkPolicy !== 'deny'
) {
  throw new Error('The packaged ACE-Step runtime pack returned an invalid ABI response.');
}
if (
  manifest.artifacts.runtimePacks?.['oryn.ace-step-mlx']?.sourceRevision !==
  specification.runtimePacks['oryn.ace-step-mlx'].sourceRevision
) {
  throw new Error('The release manifest does not bind the ACE-Step source revision.');
}
const diffusersSitePackages = join(
  canonicalRoot,
  'runtime-packs',
  'oryn.diffusers-pytorch',
  'site-packages',
);
const diffusersProbe = spawnSync(
  python,
  [
    '-I',
    '-B',
    '-c',
    [
      'import json,runpy,site',
      `site.addsitedir(${JSON.stringify(diffusersSitePackages)})`,
      'from diffusers import QwenImage21Pipeline,QwenImage21Transformer2DModel,AutoencoderKLQwenImage21',
      'from transformers import Qwen3VLProcessor,Qwen3VLForConditionalGeneration',
      "runpy.run_module('oryn_engine.runtime_packs.diffusers_worker',run_name='__main__')",
    ].join(';'),
  ],
  {
    cwd: canonicalRoot,
    env: { ...environment, ORYN_PACK_NETWORK_POLICY: 'deny' },
    encoding: 'utf8',
    input: `${JSON.stringify({
      abi: '1',
      packId: 'oryn.diffusers-pytorch',
      version: specification.runtimePacks['oryn.diffusers-pytorch'].version,
      command: 'probe',
      payload: {},
    })}\n`,
  },
);
if (diffusersProbe.status !== 0) {
  process.stderr.write(diffusersProbe.stderr);
  throw new Error('The packaged Diffusers runtime pack failed its isolated ABI probe.');
}
const diffusersResult = JSON.parse(diffusersProbe.stdout.trim());
if (
  diffusersResult.type !== 'result' ||
  diffusersResult.ok !== true ||
  diffusersResult.result?.diffusers !==
    specification.runtimePacks['oryn.diffusers-pytorch'].diffusers ||
  diffusersResult.result?.bitsandbytes !==
    specification.runtimePacks['oryn.diffusers-pytorch'].bitsandbytes ||
  diffusersResult.result?.imageio !==
    specification.runtimePacks['oryn.diffusers-pytorch'].imageio ||
  !['mps', 'cuda', 'cpu'].includes(diffusersResult.result?.backend) ||
  diffusersResult.result?.networkPolicy !== 'deny'
) {
  throw new Error('The packaged Diffusers runtime pack returned an invalid ABI response.');
}
const kokoroSitePackages = join(
  canonicalRoot,
  'runtime-packs',
  'oryn.kokoro-onnx',
  'site-packages',
);
const kokoroProbe = spawnSync(
  python,
  [
    '-I',
    '-S',
    '-B',
    '-c',
    [
      'import json,runpy,site',
      `site.addsitedir(${JSON.stringify(kokoroSitePackages)})`,
      "runpy.run_module('oryn_engine.runtime_packs.kokoro_onnx_worker',run_name='__main__')",
    ].join(';'),
  ],
  {
    cwd: canonicalRoot,
    env: { ...environment, ORYN_PACK_NETWORK_POLICY: 'deny' },
    encoding: 'utf8',
    input: `${JSON.stringify({
      abi: '1',
      packId: 'oryn.kokoro-onnx',
      version: specification.runtimePacks['oryn.kokoro-onnx'].version,
      command: 'probe',
      payload: {},
    })}\n`,
  },
);
if (kokoroProbe.status !== 0) {
  process.stderr.write(kokoroProbe.stderr);
  throw new Error('The packaged Kokoro ONNX runtime pack failed its isolated ABI probe.');
}
const kokoroResult = JSON.parse(kokoroProbe.stdout.trim());
if (
  kokoroResult.type !== 'result' ||
  kokoroResult.ok !== true ||
  !Array.isArray(kokoroResult.result?.providers) ||
  !kokoroResult.result.providers.includes('CPUExecutionProvider')
) {
  throw new Error('The packaged Kokoro ONNX runtime pack returned an invalid ABI response.');
}
const qwenTtsSitePackages = join(
  canonicalRoot,
  'runtime-packs',
  'oryn.qwen-tts-pytorch',
  'site-packages',
);
const qwenTtsMarkerPath = join(qwenTtsSitePackages, 'oryn-runtime-pack.json');
const qwenTtsMarker = JSON.parse(readFileSync(qwenTtsMarkerPath, 'utf8'));
const qwenTtsSpec = specification.runtimePacks['oryn.qwen-tts-pytorch'];
const qwenTtsArtifact = manifest.artifacts.runtimePacks['oryn.qwen-tts-pytorch'];
if (
  qwenTtsMarker.packId !== 'oryn.qwen-tts-pytorch' ||
  qwenTtsMarker.version !== qwenTtsSpec.version ||
  qwenTtsMarker.dependencyClosure !== qwenTtsSpec.dependencyClosure ||
  qwenTtsMarker.inheritBaseRuntime !== false ||
  qwenTtsArtifact.inheritsBaseRuntime !== false ||
  createHash('sha256').update(readFileSync(qwenTtsMarkerPath)).digest('hex') !==
    qwenTtsArtifact.markerSha256
) {
  throw new Error(
    'The Qwen TTS dependency closure must be separate and bound by the release manifest.',
  );
}
const qwenTtsProbe = spawnSync(
  python,
  [
    '-I',
    '-S',
    '-B',
    '-c',
    [
      'import runpy,site',
      `site.addsitedir(${JSON.stringify(qwenTtsSitePackages)})`,
      "runpy.run_module('oryn_engine.runtime_packs.qwen_tts_worker',run_name='__main__')",
    ].join(';'),
  ],
  {
    cwd: canonicalRoot,
    env: { ...environment, ORYN_PACK_NETWORK_POLICY: 'deny' },
    encoding: 'utf8',
    input: `${JSON.stringify({
      abi: '1',
      packId: 'oryn.qwen-tts-pytorch',
      version: qwenTtsSpec.version,
      command: 'probe',
      payload: {},
    })}\n`,
  },
);
const qwenTtsResult = JSON.parse(qwenTtsProbe.stdout.trim() || '{}');
const expectedQwenTts = {
  'qwen-tts': qwenTtsSpec.qwenTts,
  transformers: qwenTtsSpec.transformers,
  torch: qwenTtsSpec.torch,
  torchaudio: qwenTtsSpec.torchaudio,
  accelerate: qwenTtsSpec.accelerate,
};
if (
  qwenTtsProbe.status !== 0 ||
  qwenTtsResult.type !== 'result' ||
  qwenTtsResult.ok !== true ||
  Object.entries(expectedQwenTts).some(
    ([name, version]) =>
      qwenTtsResult.result?.versions?.[name] !== version ||
      qwenTtsArtifact.versions?.[name] !== version,
  )
) {
  throw new Error('The packaged Qwen TTS native source or dependency identity does not match.');
}
const llamaCppRoot = join(canonicalRoot, 'runtime-packs', 'oryn.llama-cpp');
const llamaCppSitePackages = join(llamaCppRoot, 'site-packages');
const llamaCppProbe = spawnSync(
  python,
  [
    '-I',
    '-S',
    '-B',
    '-c',
    [
      'import json,runpy,site',
      `site.addsitedir(${JSON.stringify(llamaCppSitePackages)})`,
      "runpy.run_module('oryn_engine.runtime_packs.llama_cpp_worker',run_name='__main__')",
    ].join(';'),
  ],
  {
    cwd: canonicalRoot,
    env: {
      ...environment,
      ORYN_PACK_NETWORK_POLICY: 'deny',
      ORYN_PACK_ROOT: llamaCppRoot,
    },
    encoding: 'utf8',
    input: `${JSON.stringify({
      abi: '1',
      packId: 'oryn.llama-cpp',
      version: specification.runtimePacks['oryn.llama-cpp'].version,
      command: 'probe',
      payload: {},
    })}\n`,
  },
);
if (llamaCppProbe.status !== 0) {
  process.stderr.write(llamaCppProbe.stderr);
  throw new Error('The packaged llama.cpp runtime pack failed its isolated ABI probe.');
}
const llamaCppResult = JSON.parse(llamaCppProbe.stdout.trim());
const expectedLlamaCpp = specification.runtimePacks['oryn.llama-cpp'];
if (
  llamaCppResult.type !== 'result' ||
  llamaCppResult.ok !== true ||
  llamaCppResult.result?.build !== expectedLlamaCpp.build ||
  llamaCppResult.result?.revision !== expectedLlamaCpp.sourceRevision ||
  llamaCppResult.result?.backend !== expectedLlamaCpp.nativeBackend ||
  llamaCppResult.result?.networkPolicy !== 'deny' ||
  llamaCppResult.result?.vision?.pillow !== specification.pillow
) {
  throw new Error('The packaged llama.cpp runtime pack returned an invalid ABI response.');
}
const llamaCppArtifact = manifest.artifacts.runtimePacks?.['oryn.llama-cpp'];
if (
  llamaCppArtifact?.sourceRevision !== expectedLlamaCpp.sourceRevision ||
  llamaCppArtifact?.build !== expectedLlamaCpp.build ||
  llamaCppArtifact?.nativeBackend !== expectedLlamaCpp.nativeBackend ||
  llamaCppArtifact?.archive?.sha256 !== expectedLlamaCpp.asset.sha256
) {
  throw new Error('The release manifest does not bind the llama.cpp native closure.');
}
console.log(
  `PACKAGED_RUNTIME_RESULT ${JSON.stringify({
    passed: true,
    versions,
    packCatalog,
    runtimePacks: {
      'oryn.transformers-pytorch': transformersResult.result,
      'oryn.onnx-runtime': onnxResult.result,
      'oryn.qwen-tts-pytorch': qwenTtsResult.result,
      'oryn.ace-step-mlx': aceStepResult.result,
      'oryn.diffusers-pytorch': diffusersResult.result,
      'oryn.kokoro-onnx': kokoroResult.result,
      'oryn.llama-cpp': llamaCppResult.result,
      'oryn.mflux-mlx': mfluxResult.result,
      'oryn.mlx-lm': mlxLmResults[0],
    },
  })}`,
);

function assertRelativeSymlinks(root) {
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        const target = readlinkSync(path);
        if (isAbsolute(target)) throw new Error(`Absolute symlink in packaged runtime: ${path}`);
        const canonical = realpathSync(path);
        const relation = relative(root, canonical);
        if (relation === '..' || relation.startsWith(`..${sep}`) || isAbsolute(relation)) {
          throw new Error(`Escaping symlink in packaged runtime: ${path}`);
        }
      } else if (entry.isDirectory() && !lstatSync(path).isSymbolicLink()) {
        pending.push(path);
      }
    }
  }
}

function assertRelocatableBytecode(root) {
  const forbidden = Buffer.from(root);
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) pending.push(path);
      if (entry.isFile() && entry.name.endsWith('.pyc') && readFileSync(path).includes(forbidden)) {
        throw new Error(`Packaged bytecode contains its installation path: ${path}`);
      }
    }
  }
}
