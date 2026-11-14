#!/usr/bin/env node
// Check the new public guides without reading or packaging the private specifications.
import { createRequire } from 'node:module';
import { mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const sdk = path.join(root, 'projects/oryn-sdk');
const require = createRequire(path.join(sdk, 'package.json'));
const linksOnly = process.argv.includes('--links-only');
const allPublicDocs = process.argv.includes('--all');
const ts = linksOnly ? undefined : require('typescript');
const exampleFiles = [
  'README.md',
  'docs/architecture.md',
  'projects/oryn-sdk/README.md',
  'projects/oryn-sdk/docs/api-reference.md',
  'projects/oryn-sdk/docs/recipes.md',
];
const files = [...exampleFiles];
if (allPublicDocs) {
  const ignored = new Set(['node_modules', '.venv', '.git', 'dist', 'out', '.stage', '.cache', 'coverage', 'test-results', 'playwright-report']);
  async function markdown(directory) {
    const result = [];
    for (const item of await readdir(path.join(root, directory), { withFileTypes: true })) {
      if (ignored.has(item.name)) continue;
      const name = `${directory}/${item.name}`;
      if (item.isDirectory()) result.push(...await markdown(name));
      else if (item.isFile() && item.name.endsWith('.md')) result.push(name);
    }
    return result;
  }
  for (const directory of ['docs', 'projects/oryn-browser', 'projects/oryn-engine', 'projects/oryn-protocol', 'projects/oryn-sdk', 'projects/oryn-examples']) {
    files.push(...await markdown(directory));
  }
  files.push('CONTRIBUTING.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md');
}
const contents = new Map(await Promise.all([...new Set(files)].map(async (file) => [file, await readFile(path.join(root, file), 'utf8')])));
const failures = [];
let linkCount = 0;
const withoutCode = (text) => text.replace(/^```[^\n]*\n[\s\S]*?^```\s*$/gm, '');
const slug = (heading) => heading.trim().toLowerCase().replace(/[`*_]/g, '').replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
function anchors(text) {
  const seen = new Map();
  return new Set([...withoutCode(text).matchAll(/^#{1,6}\s+(.+)$/gm)].map((match) => {
    const key = slug(match[1]);
    const count = seen.get(key) ?? 0;
    seen.set(key, count + 1);
    return count ? `${key}-${count}` : key;
  }));
}
for (const [file, content] of contents) {
  const prose = withoutCode(content);
  const targets = [
    ...[...prose.matchAll(/!?\[[^\]]*\]\(([^\s)]+)\)/g)].map((m) => m[1]),
    ...[...prose.matchAll(/<img\b[^>]*\bsrc="([^"]+)"/g)].map((m) => m[1]),
  ];
  for (const target of targets) {
    if (/^(https?:|mailto:)/.test(target)) continue;
    const [rawFile, fragment] = target.split('#');
    const destination = rawFile ? path.resolve(root, path.dirname(file), decodeURIComponent(rawFile)) : path.join(root, file);
    const relative = path.relative(root, destination).replaceAll(path.sep, '/');
    if (relative.startsWith('../') || /^(planning|reports|\.git)\//.test(relative) || relative.startsWith('projects/oryn-specs/') || (allPublicDocs && /^projects\/oryn-(private-examples|registry|web)\//.test(relative))) {
      failures.push(`${file}: private or outside-workspace link: ${target}`);
      continue;
    }
    try {
      const info = await stat(destination);
      if (fragment && info.isFile() && destination.endsWith('.md')) {
        const text = await readFile(destination, 'utf8');
        if (!anchors(text).has(decodeURIComponent(fragment))) failures.push(`${file}: missing anchor: ${target}`);
      }
      linkCount++;
    } catch {
      failures.push(`${file}: missing link or image: ${target}`);
    }
  }
}

const temporary = await mkdtemp(path.join(tmpdir(), 'oryn-doc-types-'));
let snippetCount = 0;
try {
  const sources = [];
  const labels = new Map();
  for (const [file, content] of linksOnly ? [] : contents) {
    if (!exampleFiles.includes(file)) continue;
    for (const match of content.matchAll(/^```(?:ts|typescript)\n([\s\S]*?)^```/gm)) {
      snippetCount++;
      const source = path.join(temporary, `example-${snippetCount}.ts`);
      await writeFile(source, `${match[1]}\nexport {};\n`);
      sources.push(source);
      labels.set(source, `${file}:${content.slice(0, match.index).split('\n').length + 1}`);
    }
  }
  const options = ts ? {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    strict: true,
    noUncheckedIndexedAccess: true,
    noEmit: true,
    skipLibCheck: true,
    types: [],
    lib: ['lib.es2022.d.ts', 'lib.dom.d.ts', 'lib.dom.iterable.d.ts'],
    baseUrl: root,
    paths: { '@ocno/oryn-sdk': ['projects/oryn-sdk/src/index.ts'] },
  } : undefined;
  const program = ts?.createProgram(sources, options);
  for (const diagnostic of program ? ts.getPreEmitDiagnostics(program) : []) {
    const source = diagnostic.file;
    const line = source && diagnostic.start !== undefined ? source.getLineAndCharacterOfPosition(diagnostic.start).line + 1 : 0;
    failures.push(`${labels.get(source?.fileName) ?? source?.fileName ?? 'TypeScript'} (snippet line ${line}): ${ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`);
  }
} finally {
  await rm(temporary, { recursive: true, force: true });
}
if (failures.length) {
  console.error(failures.join('\n'));
  process.exitCode = 1;
} else {
  console.log(`Public docs: ${contents.size} guides, ${linkCount} local links/images and ${snippetCount} SDK TypeScript examples passed.`);
}
