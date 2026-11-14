import { randomUUID } from 'node:crypto';
import { renameSync } from 'node:fs';

import { JsonFileStore, isRecord } from './json-file-store';
import { isTabIslandColor, TAB_ISLAND_COLORS } from '../../shared/tab-island-colors';
import { TAB_SEARCH_SHORTCUT, clampSidebarPanelWidth } from '../../shared/ipc';

export const DEFAULT_WORKSPACE_ID = 'workspace_default';
const MAX_ORGANIZATION_NAME_LENGTH = 64;
const SIDEBAR_TOOL_IDS = new Set([
  'tab-search',
  'bookmarks',
  'history',
  'downloads',
  'resources',
  'settings',
]);

export interface WorkspaceRecord {
  workspaceId: string;
  name: string;
  accent: string;
  order: number;
  createdAt: string;
}

export interface TabIslandRecord {
  islandId: string;
  workspaceId: string;
  name: string;
  accent: string;
  collapsed: boolean;
  order: number;
  createdAt: string;
}

export interface SidebarWebAppRecord {
  appId: string;
  name: string;
  url: string;
  allowedHosts: string[];
  enabled: boolean;
  order: number;
  muted: boolean;
}

export interface SidebarToolRecord {
  toolId: 'tab-search' | 'bookmarks' | 'history' | 'downloads' | 'resources' | 'settings';
  name: string;
  enabled: boolean;
  order: number;
}

interface BrowserExperienceData {
  version: 1;
  activeWorkspaceId: string;
  workspaces: WorkspaceRecord[];
  islands: TabIslandRecord[];
  sidebar: {
    visible: boolean;
    width: number;
    panelWidth: number;
    panelMode: 'overlay' | 'pinned';
    activeAppId: string | null;
    tools: SidebarToolRecord[];
    apps: SidebarWebAppRecord[];
  };
  tabSearchShortcut: typeof TAB_SEARCH_SHORTCUT;
  autoGroupOpenerTabs: boolean;
}

const now = () => new Date().toISOString();

function defaults(): BrowserExperienceData {
  return {
    version: 1,
    activeWorkspaceId: DEFAULT_WORKSPACE_ID,
    workspaces: [
      {
        workspaceId: DEFAULT_WORKSPACE_ID,
        name: 'Main',
        accent: '#8b8cff',
        order: 0,
        createdAt: now(),
      },
    ],
    islands: [],
    sidebar: {
      visible: true,
      width: 52,
      panelWidth: 390,
      panelMode: 'overlay',
      activeAppId: null,
      tools: defaultSidebarTools(),
      apps: [
        {
          appId: 'whatsapp',
          name: 'WhatsApp',
          url: 'https://web.whatsapp.com/',
          allowedHosts: ['web.whatsapp.com', 'whatsapp.com'],
          enabled: true,
          order: 0,
          muted: false,
        },
        {
          appId: 'messenger',
          name: 'Messenger',
          url: 'https://www.messenger.com/',
          allowedHosts: ['www.messenger.com', 'messenger.com', 'facebook.com'],
          enabled: true,
          order: 1,
          muted: false,
        },
        {
          appId: 'instagram',
          name: 'Instagram',
          url: 'https://www.instagram.com/',
          allowedHosts: ['www.instagram.com', 'instagram.com', 'facebook.com'],
          enabled: true,
          order: 2,
          muted: false,
        },
      ],
    },
    tabSearchShortcut: TAB_SEARCH_SHORTCUT,
    autoGroupOpenerTabs: true,
  };
}

export class BrowserExperienceStore {
  private readonly store: JsonFileStore<BrowserExperienceData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(filePath, defaults, isBrowserExperienceData);
    try {
      const data = this.store.read();
      const hasLegacyResourcesTool = data.sidebar.tools?.some(
        (tool) => tool.toolId === 'resources',
      );
      const tools = !Array.isArray(data.sidebar.tools)
        ? defaultSidebarTools()
        : hasLegacyResourcesTool
          ? data.sidebar.tools
              .slice()
              .sort((left, right) => left.order - right.order)
              .filter((tool) => tool.toolId !== 'resources')
              .map((tool, order) => ({ ...tool, order }))
          : data.sidebar.tools;
      if (
        !Array.isArray(data.sidebar.tools) ||
        hasLegacyResourcesTool ||
        data.tabSearchShortcut !== TAB_SEARCH_SHORTCUT
      ) {
        this.store.write({
          ...data,
          sidebar: { ...data.sidebar, tools },
          tabSearchShortcut: TAB_SEARCH_SHORTCUT,
        });
      }
    } catch {
      try {
        renameSync(filePath, `${filePath}.corrupt-${Date.now()}`);
      } catch {
        // A failed quarantine (for example a racing deletion) still permits a clean rewrite.
      }
      this.store.write(defaults());
    }
  }

  read(): BrowserExperienceData {
    return this.store.read();
  }

  createWorkspace(name: string): WorkspaceRecord {
    const workspace: WorkspaceRecord = {
      workspaceId: `workspace_${randomUUID().replaceAll('-', '')}`,
      name: boundedName(name, 'Workspace'),
      accent: workspaceAccent(this.store.read().workspaces.length),
      order: this.store.read().workspaces.length,
      createdAt: now(),
    };
    this.store.update((data) => ({ ...data, workspaces: [...data.workspaces, workspace] }));
    return workspace;
  }

  renameWorkspace(workspaceId: string, name: string): void {
    const clean = boundedName(name, '');
    if (!clean) throw new Error('WORKSPACE_NAME_REQUIRED');
    this.store.update((data) => ({
      ...data,
      workspaces: data.workspaces.map((item) =>
        item.workspaceId === workspaceId ? { ...item, name: clean } : item,
      ),
    }));
  }

  deleteWorkspace(workspaceId: string): void {
    if (workspaceId === DEFAULT_WORKSPACE_ID) throw new Error('DEFAULT_WORKSPACE_REQUIRED');
    this.store.update((data) => {
      if (!data.workspaces.some((item) => item.workspaceId === workspaceId)) {
        throw new Error('WORKSPACE_NOT_FOUND');
      }
      return {
        ...data,
        activeWorkspaceId:
          data.activeWorkspaceId === workspaceId ? DEFAULT_WORKSPACE_ID : data.activeWorkspaceId,
        workspaces: data.workspaces.filter((item) => item.workspaceId !== workspaceId),
        islands: data.islands.filter((item) => item.workspaceId !== workspaceId),
      };
    });
  }

  setActiveWorkspace(workspaceId: string): void {
    this.store.update((data) => {
      if (!data.workspaces.some((item) => item.workspaceId === workspaceId)) {
        throw new Error('WORKSPACE_NOT_FOUND');
      }
      return { ...data, activeWorkspaceId: workspaceId };
    });
  }

  createIsland(workspaceId: string, name: string, order: number): TabIslandRecord {
    const island: TabIslandRecord = {
      islandId: `island_${randomUUID().replaceAll('-', '')}`,
      workspaceId,
      name: boundedName(name, 'Tab island'),
      accent: workspaceAccent(order + 2),
      collapsed: false,
      order,
      createdAt: now(),
    };
    this.store.update((data) => ({ ...data, islands: [...data.islands, island] }));
    return island;
  }

  updateIsland(
    islandId: string,
    patch: Partial<
      Pick<TabIslandRecord, 'workspaceId' | 'name' | 'accent' | 'collapsed' | 'order'>
    >,
  ): void {
    if (patch.accent !== undefined && !isTabIslandColor(patch.accent)) {
      throw new Error('TAB_ISLAND_ACCENT_INVALID');
    }
    this.store.update((data) => ({
      ...data,
      islands: data.islands.map((item) =>
        item.islandId === islandId ? { ...item, ...patch } : item,
      ),
    }));
  }

  deleteIsland(islandId: string): void {
    this.store.update((data) => ({
      ...data,
      islands: data.islands.filter((item) => item.islandId !== islandId),
    }));
  }

  updateSidebar(patch: Partial<BrowserExperienceData['sidebar']>): void {
    this.store.update((data) => ({
      ...data,
      sidebar: {
        ...data.sidebar,
        ...patch,
        width: Math.min(76, Math.max(44, Math.round(patch.width ?? data.sidebar.width))),
        panelWidth: clampSidebarPanelWidth(patch.panelWidth ?? data.sidebar.panelWidth),
      },
    }));
  }

  updateSidebarApp(
    appId: string,
    patch: Partial<Pick<SidebarWebAppRecord, 'enabled' | 'order' | 'muted'>>,
  ): void {
    this.store.update((data) => ({
      ...data,
      sidebar: {
        ...data.sidebar,
        apps: data.sidebar.apps.map((item) =>
          item.appId === appId ? { ...item, ...patch } : item,
        ),
      },
    }));
  }

  updateSidebarTool(
    toolId: SidebarToolRecord['toolId'],
    patch: Partial<Pick<SidebarToolRecord, 'enabled' | 'order'>>,
  ): void {
    this.store.update((data) => ({
      ...data,
      sidebar: {
        ...data.sidebar,
        tools: data.sidebar.tools.map((item) =>
          item.toolId === toolId ? { ...item, ...patch } : item,
        ),
      },
    }));
  }

  updateTabPreferences(patch: {
    tabSearchShortcut?: BrowserExperienceData['tabSearchShortcut'];
    autoGroupOpenerTabs?: boolean;
  }): void {
    this.store.update((data) => ({ ...data, ...patch }));
  }
}

function defaultSidebarTools(): SidebarToolRecord[] {
  return [
    { toolId: 'tab-search', name: 'Search Tabs', enabled: true, order: 0 },
    { toolId: 'bookmarks', name: 'Favorites & Bookmarks', enabled: true, order: 1 },
    { toolId: 'history', name: 'History', enabled: true, order: 2 },
    { toolId: 'downloads', name: 'Downloads', enabled: true, order: 3 },
    { toolId: 'settings', name: 'Settings', enabled: true, order: 4 },
  ];
}

function workspaceAccent(index: number): string {
  return TAB_ISLAND_COLORS[index % TAB_ISLAND_COLORS.length]?.value ?? '#8b8cff';
}

function boundedName(value: string, fallback: string): string {
  return (value.trim() || fallback).slice(0, MAX_ORGANIZATION_NAME_LENGTH);
}

function isBrowserExperienceData(value: unknown): value is BrowserExperienceData {
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.workspaces)) return false;
  return (
    typeof value.activeWorkspaceId === 'string' &&
    value.workspaces.some(
      (workspace) => isRecord(workspace) && workspace.workspaceId === DEFAULT_WORKSPACE_ID,
    ) &&
    Array.isArray(value.islands) &&
    isRecord(value.sidebar) &&
    typeof value.sidebar.visible === 'boolean' &&
    typeof value.sidebar.width === 'number' &&
    typeof value.sidebar.panelWidth === 'number' &&
    (value.sidebar.panelMode === 'overlay' || value.sidebar.panelMode === 'pinned') &&
    Array.isArray(value.sidebar.apps) &&
    (value.sidebar.tools === undefined ||
      (Array.isArray(value.sidebar.tools) &&
        value.sidebar.tools.every(
          (tool) =>
            isRecord(tool) &&
            typeof tool.toolId === 'string' &&
            SIDEBAR_TOOL_IDS.has(tool.toolId) &&
            typeof tool.name === 'string' &&
            typeof tool.enabled === 'boolean' &&
            typeof tool.order === 'number',
        ))) &&
    typeof value.autoGroupOpenerTabs === 'boolean'
  );
}
