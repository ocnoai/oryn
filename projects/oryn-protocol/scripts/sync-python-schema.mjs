import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(repositoryRoot, 'schemas/v1/protocol-v1.schema.json');
const destination = resolve(
  repositoryRoot,
  'python/src/oryn_protocol/schemas/v1/protocol-v1.schema.json',
);

await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
