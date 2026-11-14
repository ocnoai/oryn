import type { ModelGenerationProfile } from '@ocno/oryn-protocol';

export interface GenerationValues {
  width: number;
  height: number;
  steps: number;
  guidanceScale: number;
}

export function generationDefaults(profile: ModelGenerationProfile): GenerationValues {
  return {
    width: profile.parameters.width.default,
    height: profile.parameters.height.default,
    steps: profile.parameters.steps.default,
    guidanceScale: profile.parameters.guidanceScale.default,
  };
}

export function isGenerationProfileDirty(
  values: GenerationValues,
  profile: ModelGenerationProfile,
): boolean {
  const defaults = generationDefaults(profile);
  return (Object.keys(defaults) as Array<keyof GenerationValues>).some(
    (key) => values[key] !== defaults[key],
  );
}

export function generationProfileSource(profile: ModelGenerationProfile): string {
  switch (profile.provenance) {
    case 'runtime-tested':
      return 'Tested by the selected Oryn runtime';
    case 'runtime-declared':
      return 'Declared by the selected signed runtime';
    case 'adapter-derived':
      return 'Derived by the runtime adapter from model metadata';
    case 'runtime-fallback':
      return 'Generic starting values — not a model-specific recommendation';
  }
}
