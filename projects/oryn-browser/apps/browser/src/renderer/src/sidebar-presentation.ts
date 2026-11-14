import type { BrowserWindowState } from '../../shared/ipc';

type SidebarToolId = BrowserWindowState['experience']['sidebar']['tools'][number]['toolId'];

const toolDescriptions: Record<SidebarToolId, string> = {
  'tab-search': 'Search tabs — find and switch between open tabs',
  bookmarks: 'Bookmarks — open saved pages and favorites',
  history: 'History — view recently visited pages',
  downloads: 'Downloads — view and manage downloaded files',
  resources: 'Local AI resources — view jobs, memory, and installed models',
  settings: 'Settings — customize Oryn Browser',
};

export function sidebarToolTooltip(toolId: SidebarToolId): string {
  return toolDescriptions[toolId];
}

export function nextWorkspaceName(workspaces: ReadonlyArray<{ name: string }>): string {
  const names = new Set(workspaces.map((workspace) => workspace.name.trim().toLocaleLowerCase()));
  let ordinal = 2;
  while (names.has(`workspace ${ordinal}`)) ordinal += 1;
  return `Workspace ${ordinal}`;
}
