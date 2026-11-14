import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, it } from 'vitest';

it('allows owned artifact data reads while excluding network connections from the workspace', () => {
  const html = readFileSync(resolve('src/renderer/model-workspace.html'), 'utf8');
  const policy = html.match(/content="(default-src[^"]+)"/u)?.[1];
  expect(policy).toBeTruthy();
  const directives = new Map(
    policy!.split(';').map((part) => {
      const [name, ...sources] = part.trim().split(/\s+/u);
      return [name, sources];
    }),
  );
  expect(directives.get('default-src')).toEqual(["'none'"]);
  expect(directives.get('connect-src')).toEqual(['oryn-artifact:']);
  expect(directives.get('script-src')).toEqual(["'self'"]);
});
