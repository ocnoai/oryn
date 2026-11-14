import type { ModelGenerationProfile } from '@ocno/oryn-protocol';
import { describe, expect, it } from 'vitest';

import {
  generationDefaults,
  generationProfileSource,
  isGenerationProfileDirty,
} from '../../renderer/src/model-workspace-generation-profile';

const profile: ModelGenerationProfile = {
  profileId: 'diffusers.example.distilled',
  provenance: 'adapter-derived',
  recommended: true,
  parameters: {
    width: {
      default: 768,
      minimum: 64,
      maximum: 2048,
      step: 8,
      supported: true,
      editable: true,
    },
    height: {
      default: 512,
      minimum: 64,
      maximum: 2048,
      step: 8,
      supported: true,
      editable: true,
    },
    steps: {
      default: 6,
      minimum: 1,
      maximum: 100,
      step: 1,
      supported: true,
      editable: true,
    },
    guidanceScale: {
      default: 0,
      minimum: 0,
      maximum: 30,
      step: 0.1,
      supported: true,
      editable: true,
    },
  },
};

describe('runtime-owned model generation profiles', () => {
  it('uses the complete runtime profile without renderer-owned preset values', () => {
    expect(generationDefaults(profile)).toEqual({
      width: 768,
      height: 512,
      steps: 6,
      guidanceScale: 0,
    });
  });

  it('detects customization relative to the exact model profile', () => {
    expect(isGenerationProfileDirty(generationDefaults(profile), profile)).toBe(false);
    expect(isGenerationProfileDirty({ ...generationDefaults(profile), steps: 10 }, profile)).toBe(
      true,
    );
  });

  it('does not present fallback values as a model recommendation', () => {
    expect(generationProfileSource({ ...profile, provenance: 'runtime-fallback' })).toContain(
      'not a model-specific recommendation',
    );
  });
});
