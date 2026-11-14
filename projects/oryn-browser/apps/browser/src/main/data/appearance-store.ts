import { JsonFileStore, isRecord } from './json-file-store';

export interface BrowserAppearanceData {
  version: 1;
  theme: 'light' | 'dark' | 'system';
  density: 'default' | 'compact' | 'comfortable';
  onboardingComplete: boolean;
  searchProvider?: 'google' | 'duckduckgo' | 'bing';
  localSuggestions?: boolean;
}

export class BrowserAppearanceStore {
  private readonly store: JsonFileStore<BrowserAppearanceData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore<BrowserAppearanceData>(
      filePath,
      () => ({
        version: 1,
        theme: 'system',
        density: 'default',
        onboardingComplete: false,
        localSuggestions: true,
      }),
      isAppearance,
    );
  }

  read(): BrowserAppearanceData {
    return { localSuggestions: true, ...structuredClone(this.store.read()) };
  }

  update(value: Partial<Omit<BrowserAppearanceData, 'version'>>): BrowserAppearanceData {
    if (
      !isRecord(value) ||
      Object.keys(value).some(
        (key) =>
          ![
            'theme',
            'density',
            'onboardingComplete',
            'searchProvider',
            'localSuggestions',
          ].includes(key),
      ) ||
      (value.searchProvider !== undefined &&
        !['google', 'duckduckgo', 'bing'].includes(value.searchProvider)) ||
      (value.localSuggestions !== undefined && typeof value.localSuggestions !== 'boolean') ||
      (value.theme !== undefined && !['light', 'dark', 'system'].includes(value.theme)) ||
      (value.density !== undefined &&
        !['default', 'compact', 'comfortable'].includes(value.density)) ||
      (value.onboardingComplete !== undefined && typeof value.onboardingComplete !== 'boolean')
    ) {
      throw new Error('INVALID_APPEARANCE_SETTINGS');
    }
    return this.store.update((current) => ({ ...current, ...value }));
  }
}

function isAppearance(value: unknown): value is BrowserAppearanceData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    ['light', 'dark', 'system'].includes(String(value.theme)) &&
    ['default', 'compact', 'comfortable'].includes(String(value.density)) &&
    typeof value.onboardingComplete === 'boolean' &&
    (value.searchProvider === undefined ||
      ['google', 'duckduckgo', 'bing'].includes(String(value.searchProvider))) &&
    (value.localSuggestions === undefined || typeof value.localSuggestions === 'boolean')
  );
}
