import type { ExplicitModelRequest, ModelRef } from '@ocno/oryn-sdk';

export const FLUX_MODEL: ModelRef = {
  provider: 'huggingface',
  repository: 'black-forest-labs/FLUX.1-dev',
  revision: '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21',
};

export const FIXTURE_MODEL: ModelRef = {
  provider: 'huggingface',
  repository: 'dg845/tiny-random-stable-diffusion-xl',
  revision: '7ae769dd7f63b06308f2f23881d329a0cb1c89d2',
};

export const PRODUCTION_FALLBACK_MODEL: ModelRef = {
  provider: 'huggingface',
  repository: 'segmind/Segmind-Vega',
  revision: '7714c4363e5856ff974a4f4b068e8691f26d0b40',
};

export type ImageModelProfile = 'production' | 'fixture';

export const IMAGE_MODEL_PROFILE: ImageModelProfile = __ORYN_IMAGE_MODEL_PROFILE__;

export function createImageModelRequest(
  profile: ImageModelProfile = IMAGE_MODEL_PROFILE,
): ExplicitModelRequest {
  const fixture = profile === 'fixture';

  return {
    role: 'image-generation',
    preferredModel: { ...(fixture ? FIXTURE_MODEL : FLUX_MODEL) },
    certifiedModels: [{ ...(fixture ? FIXTURE_MODEL : PRODUCTION_FALLBACK_MODEL) }],
    requirements: {
      task: 'text-to-image',
      inputModalities: ['text'],
      outputModalities: ['image'],
      capabilities: { deterministicSeed: true },
    },
    substitution: { automatic: fixture ? 'none' : 'compatible' },
  };
}

declare const __ORYN_IMAGE_MODEL_PROFILE__: ImageModelProfile;
