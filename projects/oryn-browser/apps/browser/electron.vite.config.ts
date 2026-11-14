import { resolve } from 'node:path';

import react from '@vitejs/plugin-react';
import { defineConfig, externalizeDepsPlugin } from 'electron-vite';

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin({ exclude: ['@ocno/oryn-protocol', '@ocno/oryn-registry'] })],
    build: {
      rollupOptions: {
        input: {
          index: resolve('src/main/index.ts'),
          'browser-import-worker': resolve('src/main/data/browser-import-worker.ts'),
        },
        external: ['original-fs'],
      },
    },
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: {
          trusted: resolve('src/preload/trusted-ui-preload.ts'),
          website: resolve('src/preload/website-preload.ts'),
          private: resolve('src/preload/private-website-preload.ts'),
          workspace: resolve('src/preload/model-workspace-preload.ts'),
        },
        output: {
          entryFileNames: '[name].cjs',
          format: 'cjs',
        },
      },
    },
  },
  renderer: {
    root: resolve('src/renderer'),
    plugins: [react()],
    server: {
      // Keep the browser's private development renderer away from the stable
      // loopback ports used by the three ordinary reference websites.
      port: 5180,
      strictPort: true,
      fs: {
        allow: [resolve('../..')],
      },
    },
    build: {
      rollupOptions: {
        input: {
          index: resolve('src/renderer/index.html'),
          workspace: resolve('src/renderer/model-workspace.html'),
        },
      },
    },
  },
});
