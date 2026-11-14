import { defineConfig } from 'vite';

export default defineConfig(({ mode }) => {
  const profile = mode === 'fixture' ? 'fixture' : 'production';

  return {
    define: {
      __ORYN_IMAGE_MODEL_PROFILE__: JSON.stringify(profile),
    },
    plugins: [
      {
        name: 'oryn-image-model-profile',
        transformIndexHtml(html) {
          return html.replaceAll('%ORYN_IMAGE_MODEL_PROFILE%', profile);
        },
      },
    ],
  };
});
