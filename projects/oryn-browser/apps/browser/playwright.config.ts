import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/electron',
  testMatch: '**/*.spec.ts',
  workers: 1,
  fullyParallel: false,
  timeout: 45_000,
  expect: { timeout: 10_000 },
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'test-results/electron-results.json' }]],
  use: { trace: 'retain-on-failure' },
});
