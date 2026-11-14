import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { BookmarkStore } from '../data/bookmark-store';
import { DownloadStore } from '../data/download-store';
import { BrowsingHistoryStore } from '../data/history-store';
import { ProfileDataPaths } from '../data/profile-data-paths';
import { BrowserSessionStore } from '../data/session-store';
import { SitePermissionStore } from '../data/site-permission-store';
import { IPC, TAB_SEARCH_SHORTCUT } from '../../shared/ipc';
import { BrowserAppearanceStore } from '../data/appearance-store';
import { fitWindowBoundsToWorkArea, WindowPlacementStore } from '../data/window-placement-store';
import { BrowserExperienceStore, DEFAULT_WORKSPACE_ID } from '../data/browser-experience-store';

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

function profile(): ProfileDataPaths {
  const directory = mkdtempSync(join(tmpdir(), 'oryn-browser-data-test-'));
  temporaryDirectories.push(directory);
  return new ProfileDataPaths(directory);
}

describe('BrowsingHistoryStore', () => {
  it('records multi-tab visits and supports search, deletion, clearing, and reload', () => {
    const paths = profile();
    const history = new BrowsingHistoryStore(paths.history);
    history.recordVisit('https://one.example/page', 'First page', new Date('2026-01-01'));
    history.recordVisit('https://two.example/page', 'Second page', new Date('2026-01-02'));
    history.recordVisit('https://three.example/page', 'Third page', new Date('2026-01-03'));

    expect(history.search()).toHaveLength(3);
    expect(history.search('second')[0]?.url).toBe('https://two.example/page');

    history.flush();
    const persisted = new BrowsingHistoryStore(paths.history);
    const first = persisted.search('first')[0];
    expect(first).toBeDefined();
    expect(persisted.delete(first?.historyId ?? '')).toBe(true);
    expect(persisted.search()).toHaveLength(2);
    persisted.clear();
    expect(persisted.search()).toEqual([]);
  });
});

describe('BookmarkStore', () => {
  it('persists the canonical graph and supports recoverable recursive trash', () => {
    const paths = profile();
    const bookmarks = new BookmarkStore(paths.bookmarks);
    const folder = bookmarks.createFolder('Research');
    bookmarks.createBookmark('Oryn', 'https://oryn.ocno.ai', folder.bookmarkId);
    bookmarks.setBarVisible(false);

    const persisted = new BookmarkStore(paths.bookmarks);
    expect(persisted.list()).toHaveLength(2);
    expect(persisted.isBarVisible()).toBe(false);
    expect(persisted.delete(folder.bookmarkId)).toBe(true);
    expect(new BookmarkStore(paths.bookmarks).list().every((item) => item.trashedAt)).toBe(true);
    persisted.restore([folder.bookmarkId]);
    expect(persisted.list().every((item) => !item.trashedAt)).toBe(true);
    persisted.setFavorite(folder.bookmarkId, true);
    expect(persisted.list().find((item) => item.bookmarkId === folder.bookmarkId)?.favorite).toBe(
      true,
    );
    persisted.deletePermanently([folder.bookmarkId]);
    expect(new BookmarkStore(paths.bookmarks).list()).toEqual([]);
  });

  it('bounds nesting, rejects descendant cycles, and stores only inert bounded thumbnails', () => {
    const paths = profile();
    const bookmarks = new BookmarkStore(paths.bookmarks);
    const parent = bookmarks.createFolder('Parent');
    const child = bookmarks.createFolder('Child', parent.bookmarkId);
    const page = bookmarks.createBookmark('Oryn', 'https://oryn.ocno.ai', child.bookmarkId);

    expect(() => bookmarks.move([parent.bookmarkId], child.bookmarkId)).toThrow('BOOKMARK_CYCLE');
    expect(() =>
      bookmarks.setThumbnail(page.bookmarkId, 'data:image/svg+xml;base64,PHN2Zz4='),
    ).toThrow('BOOKMARK_THUMBNAIL_UNSAFE');

    const thumbnail = 'data:image/png;base64,aGVsbG8=';
    bookmarks.setThumbnail(page.bookmarkId, thumbnail);
    expect(bookmarks.list().find((item) => item.bookmarkId === page.bookmarkId)).toMatchObject({
      thumbnail,
    });
    bookmarks.setThumbnail(page.bookmarkId);
    expect(bookmarks.list().find((item) => item.bookmarkId === page.bookmarkId)).not.toHaveProperty(
      'thumbnail',
    );

    let deepest = child;
    for (let depth = 3; depth <= 8; depth += 1) {
      deepest = bookmarks.createFolder(`Depth ${depth}`, deepest.bookmarkId);
    }
    expect(() => bookmarks.createFolder('Too deep', deepest.bookmarkId)).toThrow(
      'BOOKMARK_DEPTH_EXCEEDED',
    );
  });
});

describe('BrowserExperienceStore', () => {
  it('preserves the non-deletable default workspace and isolated sidebar configuration', () => {
    const paths = profile();
    const experience = new BrowserExperienceStore(paths.browserExperience);
    const research = experience.createWorkspace('Research');
    experience.setActiveWorkspace(research.workspaceId);
    const island = experience.createIsland(research.workspaceId, 'Sources', 0);
    experience.updateIsland(island.islandId, { collapsed: true });
    experience.updateIsland(island.islandId, { accent: '#67a9ff' });
    expect(() => experience.updateIsland(island.islandId, { accent: '#ffffff' })).toThrow(
      'TAB_ISLAND_ACCENT_INVALID',
    );
    experience.updateSidebar({ panelMode: 'pinned', panelWidth: 444, activeAppId: 'whatsapp' });
    experience.updateSidebarTool('history', { enabled: false, order: 0 });
    experience.updateTabPreferences({
      autoGroupOpenerTabs: false,
      tabSearchShortcut: TAB_SEARCH_SHORTCUT,
    });

    const restored = new BrowserExperienceStore(paths.browserExperience).read();
    expect(restored.activeWorkspaceId).toBe(research.workspaceId);
    expect(restored.islands[0]).toMatchObject({
      name: 'Sources',
      accent: '#67a9ff',
      collapsed: true,
    });
    expect(restored.sidebar).toMatchObject({ panelMode: 'pinned', panelWidth: 444 });
    expect(restored.sidebar.tools.find((tool) => tool.toolId === 'history')).toMatchObject({
      enabled: false,
      order: 0,
    });
    expect(restored.sidebar.tools.some((tool) => tool.toolId === 'resources')).toBe(false);
    expect(restored).toMatchObject({
      autoGroupOpenerTabs: false,
      tabSearchShortcut: TAB_SEARCH_SHORTCUT,
    });
    expect(() => experience.deleteWorkspace(DEFAULT_WORKSPACE_ID)).toThrow(
      'DEFAULT_WORKSPACE_REQUIRED',
    );
  });

  it('removes the legacy Local AI resources shortcut without resetting sidebar preferences', () => {
    const paths = profile();
    const experience = new BrowserExperienceStore(paths.browserExperience);
    experience.updateSidebar({ panelMode: 'pinned', panelWidth: 444 });
    const persisted = JSON.parse(readFileSync(paths.browserExperience, 'utf8')) as {
      tabSearchShortcut: string;
      sidebar: {
        panelMode: string;
        panelWidth: number;
        tools: Array<{ toolId: string; name: string; enabled: boolean; order: number }>;
      };
    };
    persisted.sidebar.tools.splice(4, 0, {
      toolId: 'resources',
      name: 'Oryn Resources',
      enabled: true,
      order: 4,
    });
    persisted.tabSearchShortcut = 'Control+Space';
    writeFileSync(paths.browserExperience, JSON.stringify(persisted), 'utf8');

    const restored = new BrowserExperienceStore(paths.browserExperience).read();

    expect(restored.sidebar.tools.some((tool) => tool.toolId === 'resources')).toBe(false);
    expect(restored.sidebar.tools.map((tool) => tool.order)).toEqual([0, 1, 2, 3, 4]);
    expect(restored.sidebar).toMatchObject({ panelMode: 'pinned', panelWidth: 444 });
    expect(restored.tabSearchShortcut).toBe(TAB_SEARCH_SHORTCUT);
  });

  it('quarantines corrupt experience state and recovers the default workspace', () => {
    const paths = profile();
    new BrowserExperienceStore(paths.browserExperience).createWorkspace('Seed');
    writeFileSync(paths.browserExperience, '{"version":1,"workspaces":[]}', 'utf8');
    const restored = new BrowserExperienceStore(paths.browserExperience).read();
    expect(restored.activeWorkspaceId).toBe(DEFAULT_WORKSPACE_ID);
    expect(restored.workspaces).toHaveLength(1);
  });
});

describe('BrowserSessionStore', () => {
  it('keeps cached favicon bytes only in open tabs, not recently closed history', () => {
    const paths = profile();
    const sessions = new BrowserSessionStore(paths.session);
    const tab = { url: 'https://one.example/', favicon: 'data:image/png;base64,iVBORw0KGgo=' };
    sessions.saveOpenTabs([tab], 0);
    expect(new BrowserSessionStore(paths.session).getRestoreState().openTabs).toEqual([tab]);
    sessions.pushRecentlyClosed(tab);
    sessions.saveOpenTabs([], 0);
    expect(new BrowserSessionStore(paths.session).listRecentlyClosed()).toEqual([{ url: tab.url }]);
    expect(readFileSync(paths.session, 'utf8')).not.toContain(tab.favicon);
  });

  it('persists open tabs, active selection, and recently closed state for restart', () => {
    const paths = profile();
    const sessions = new BrowserSessionStore(paths.session);
    sessions.saveOpenTabs([{ url: 'https://one.example/' }, { url: 'https://two.example/' }], 1);
    sessions.pushRecentlyClosed({ url: 'https://closed.example/' });

    const restored = new BrowserSessionStore(paths.session);
    expect(restored.getRestoreState()).toEqual({
      openTabs: [{ url: 'https://one.example/' }, { url: 'https://two.example/' }],
      activeIndex: 1,
    });
    expect(restored.popRecentlyClosed()).toEqual({ url: 'https://closed.example/' });
    expect(new BrowserSessionStore(paths.session).listRecentlyClosed()).toEqual([]);
  });
});

describe('WindowPlacementStore', () => {
  it('persists the last normal window bounds and maximized state', () => {
    const paths = profile();
    const placement = new WindowPlacementStore(paths.windowPlacement);
    expect(placement.read()).toEqual({ version: 1, maximized: false });

    placement.write({ x: 120, y: 80, width: 1440, height: 900 }, true);

    expect(new WindowPlacementStore(paths.windowPlacement).read()).toEqual({
      version: 1,
      bounds: { x: 120, y: 80, width: 1440, height: 900 },
      maximized: true,
    });
  });

  it('keeps restored bounds visible when the saved display is no longer available', () => {
    expect(
      fitWindowBoundsToWorkArea(
        { x: 2600, y: -800, width: 1800, height: 1200 },
        { x: 0, y: 25, width: 1512, height: 957 },
      ),
    ).toEqual({ x: 0, y: 25, width: 1512, height: 957 });
  });
});

describe('normal browser data boundaries', () => {
  it('enables local suggestions for new and older profiles while preserving an explicit opt-out', () => {
    const paths = profile();
    const freshAppearance = new BrowserAppearanceStore(paths.appearance);
    expect(freshAppearance.read().localSuggestions).toBe(true);
    freshAppearance.update({ onboardingComplete: true });
    writeFileSync(
      paths.appearance,
      JSON.stringify({ version: 1, theme: 'system', density: 'default', onboardingComplete: true }),
    );
    const appearance = new BrowserAppearanceStore(paths.appearance);
    expect(appearance.read().localSuggestions).toBe(true);
    appearance.update({ localSuggestions: false });
    expect(new BrowserAppearanceStore(paths.appearance).read().localSuggestions).toBe(false);
  });

  it('persists first-run completion and complete theme choices', () => {
    const paths = profile();
    const appearance = new BrowserAppearanceStore(paths.appearance);
    expect(appearance.read()).toMatchObject({ theme: 'system', onboardingComplete: false });
    appearance.update({ theme: 'dark', density: 'comfortable', onboardingComplete: true });
    expect(new BrowserAppearanceStore(paths.appearance).read()).toMatchObject({
      theme: 'dark',
      density: 'comfortable',
      onboardingComplete: true,
    });
    expect(() => appearance.update({ theme: 'sepia' } as never)).toThrow(
      'INVALID_APPEARANCE_SETTINGS',
    );
  });

  it('persists downloads and Chromium site permissions as Browser Core state', () => {
    const paths = profile();
    const downloads = new DownloadStore(paths.downloads);
    const download = downloads.start({
      url: 'https://files.example/report.pdf',
      filename: 'report.pdf',
      totalBytes: 100,
    });
    downloads.update(download.downloadId, { receivedBytes: 100, state: 'completed' });
    expect(new DownloadStore(paths.downloads).list()[0]?.state).toBe('completed');

    const permissions = new SitePermissionStore(paths.sitePermissions);
    permissions.set('https://camera.example', 'media', 'allow');
    expect(
      new SitePermissionStore(paths.sitePermissions).get('https://camera.example', 'media'),
    ).toBe('allow');
  });

  it('physically separates Browser Core from Local AI data and exposes no data-store IPC', () => {
    const paths = profile();
    expect(paths.history.startsWith(paths.browserCoreRoot)).toBe(true);
    expect(paths.windowPlacement.startsWith(paths.browserCoreRoot)).toBe(true);
    expect(paths.aiActivity.startsWith(paths.localAIRoot)).toBe(true);
    expect(paths.browserCoreRoot).not.toBe(paths.localAIRoot);
    expect(paths.history).not.toBe(paths.aiActivity);

    const localAIChannels = Object.values(IPC).filter((channel) => channel.startsWith('local-ai:'));
    expect(localAIChannels).toContain('local-ai:get-capabilities');
    expect(localAIChannels.join(' ')).not.toMatch(/history|bookmark|download|cookie|site-data/u);
    const websitePreload = readFileSync(
      join(import.meta.dirname, '../../preload/website-preload.ts'),
      'utf8',
    );
    expect(websitePreload).toContain('local-ai:get-capabilities');
    expect(websitePreload).not.toContain('permission-respond');
  });
});
