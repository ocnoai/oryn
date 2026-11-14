import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import type { BrowserDataService } from './browser-data-service';
import { BrowserSessionStore } from './session-store';
import { JsonFileStore, isRecord } from './json-file-store';
import { DEFAULT_WORKSPACE_ID } from './browser-experience-store';

export class WindowRegistryStore {
  private readonly store: JsonFileStore<{ version: 1; ids: string[] }>;
  constructor(root: string) {
    this.store = new JsonFileStore<{ version: 1; ids: string[] }>(
      join(root, 'windows.json'),
      () => ({ version: 1, ids: ['main'] }),
      (value): value is { version: 1; ids: string[] } =>
        isRecord(value) &&
        value.version === 1 &&
        Array.isArray(value.ids) &&
        value.ids.length <= 32 &&
        value.ids.every(
          (id) => typeof id === 'string' && /^(main|window_[a-f0-9]{32})$/u.test(id),
        ) &&
        new Set(value.ids).size === value.ids.length,
    );
  }
  list() {
    return this.store.read().ids;
  }
  add() {
    const id = `window_${randomUUID().replaceAll('-', '')}`;
    this.store.update((data) => ({ ...data, ids: [...data.ids, id] }));
    return id;
  }
  remove(id: string) {
    this.store.update((data) => ({ ...data, ids: data.ids.filter((item) => item !== id) }));
  }
}

/** Shared profile data; session snapshot and currently selected organization belong to one window. */
export function dataForWindow(shared: BrowserDataService, id: string): BrowserDataService {
  let activeWorkspaceId = shared.experience.read().activeWorkspaceId;
  let activeAppId: string | null = null;
  const experience = new Proxy(shared.experience, {
    get(target, property) {
      if (property === 'read')
        return () => {
          const value = target.read();
          if (!value.workspaces.some((workspace) => workspace.workspaceId === activeWorkspaceId))
            activeWorkspaceId = DEFAULT_WORKSPACE_ID;
          return { ...value, activeWorkspaceId, sidebar: { ...value.sidebar, activeAppId } };
        };
      if (property === 'setActiveWorkspace')
        return (workspaceId: string) => {
          if (!target.read().workspaces.some((workspace) => workspace.workspaceId === workspaceId))
            throw new Error('WORKSPACE_NOT_FOUND');
          activeWorkspaceId = workspaceId;
        };
      if (property === 'updateSidebar')
        return (patch: Parameters<typeof target.updateSidebar>[0]) => {
          if ('activeAppId' in patch) activeAppId = patch.activeAppId ?? null;
          const persistent = { ...patch };
          delete persistent.activeAppId;
          if (Object.keys(persistent).length) target.updateSidebar(persistent);
        };
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  return Object.assign(Object.create(shared) as BrowserDataService, {
    experience,
    sessions: new BrowserSessionStore(
      id === 'main'
        ? shared.paths.session
        : join(shared.paths.browserCoreRoot, 'windows', `${id}.json`),
    ),
  });
}
