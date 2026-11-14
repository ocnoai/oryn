import { JsonFileStore, isRecord } from './json-file-store';
export class SiteZoomStore {
  private readonly store: JsonFileStore<Record<string, number>>;
  constructor(path: string) {
    this.store = new JsonFileStore<Record<string, number>>(
      path,
      () => ({}),
      (value): value is Record<string, number> =>
        isRecord(value) &&
        Object.entries(value).every(
          ([origin, factor]) =>
            /^https?:\/\//u.test(origin) &&
            typeof factor === 'number' &&
            factor >= 0.5 &&
            factor <= 2,
        ),
    );
  }
  get(url: string): number {
    try {
      return this.store.read()[new URL(url).origin] ?? 1;
    } catch {
      return 1;
    }
  }
  set(url: string, factor: number): void {
    const parsed = new URL(url);
    if (!['http:', 'https:'].includes(parsed.protocol)) return;
    this.store.update((value) => ({ ...value, [parsed.origin]: factor }));
  }
}
