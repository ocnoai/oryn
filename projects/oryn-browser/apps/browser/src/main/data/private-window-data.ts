import { randomUUID } from 'node:crypto';
import { BrowserDataService } from './browser-data-service';

/** Every mutable browsing store is memory-backed; explicit bookmark saves use the shared store. */
export function dataForPrivateWindow(shared: BrowserDataService): BrowserDataService {
  const data = new BrowserDataService(`memory:private-${randomUUID()}`);
  Object.defineProperty(data, 'bookmarks', { value: shared.bookmarks });
  const appearance = shared.appearance.read();
  data.appearance.update({
    theme: appearance.theme,
    density: appearance.density,
    searchProvider: appearance.searchProvider,
    localSuggestions: appearance.localSuggestions,
    onboardingComplete: true,
  });
  data.experience.updateSidebar({ visible: false, activeAppId: null });
  const privateHistory = data.history;
  Object.defineProperty(data, 'history', {
    value: new Proxy(privateHistory, {
      get(target, property) {
        if (property === 'search') return (query = '') => shared.history.search(query);
        const value = Reflect.get(target, property, target);
        return typeof value === 'function' ? value.bind(target) : value;
      },
    }),
  });
  return data;
}
