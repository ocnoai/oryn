import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  timeout: 45000,
  expect: { timeout: 10000 },
  use: {
    baseURL: "http://127.0.0.1:5192",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: {
    command: "pnpm build && pnpm preview --strictPort",
    url: "http://127.0.0.1:5192",
    reuseExistingServer: !process.env.CI,
  },
});
