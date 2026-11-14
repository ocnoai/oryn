import { useEffect, useRef, useState, type ButtonHTMLAttributes, type CSSProperties } from 'react';

import {
  SIDEBAR_PANEL_MAX_WIDTH,
  SIDEBAR_PANEL_MIN_WIDTH,
  TAB_SEARCH_SHORTCUT,
  clampSidebarPanelWidth,
  type BrowserWindowState,
} from '../../shared/ipc';
import { Icon, type IconName } from './components/ui';
import { SidebarAppIcon } from './components/ui/SidebarAppIcon';
import { nextWorkspaceName, sidebarToolTooltip } from './sidebar-presentation';

const emptyState: BrowserWindowState = {
  activeTabId: '',
  tabs: [],
  engineState: 'starting',
  zoomFactor: 1,
  memoryPressure: 'normal',
  experience: {
    activeWorkspaceId: 'workspace_default',
    workspaces: [],
    islands: [],
    recentlyClosed: [],
    sidebar: {
      visible: true,
      width: 52,
      panelWidth: 390,
      panelMode: 'overlay',
      activeAppId: null,
      tools: [],
      apps: [],
    },
    tabSearchShortcut: TAB_SEARCH_SHORTCUT,
    autoGroupOpenerTabs: true,
  },
};

export function BrowserSidebar(): React.JSX.Element {
  const [state, setState] = useState(emptyState);
  useEffect(() => {
    void window.browser.getState().then(setState);
    return window.browser.onStateChanged(setState);
  }, []);
  const experience = state.experience;
  return (
    <aside className="browser-sidebar" aria-label="Browser sidebar">
      <div
        className="sidebar-brand"
        aria-label="Oryn sidebar"
        title="Oryn sidebar — workspaces, browser tools, and web apps"
      >
        O
      </div>
      <nav className="workspace-switcher" aria-label="Workspaces">
        {experience.workspaces.map((workspace) => (
          <SidebarIconButton
            key={workspace.workspaceId}
            className={workspace.workspaceId === experience.activeWorkspaceId ? 'active' : ''}
            style={{ '--workspace-accent': workspace.accent } as CSSProperties}
            label={`${workspace.name} workspace`}
            tooltip={`${workspace.name} workspace — switch to its tabs`}
            onClick={() =>
              void window.browser.updateBrowserExperience({
                type: 'workspace-switch',
                workspaceId: workspace.workspaceId,
              })
            }
          >
            {workspace.name.slice(0, 1).toUpperCase()}
          </SidebarIconButton>
        ))}
        <SidebarIconButton
          className="sidebar-add"
          label="Create workspace"
          tooltip="New workspace — create and switch to a separate tab space"
          onClick={() =>
            void window.browser.updateBrowserExperience({
              type: 'workspace-create',
              name: nextWorkspaceName(experience.workspaces),
            })
          }
        >
          <Icon name="plus" />
        </SidebarIconButton>
      </nav>
      <nav className="sidebar-tools" aria-label="Browser tools">
        {experience.sidebar.tools
          .filter((tool) => tool.enabled)
          .sort((left, right) => left.order - right.order)
          .map((tool) => (
            <SidebarIconButton
              key={tool.toolId}
              label={tool.name}
              tooltip={sidebarToolTooltip(tool.toolId)}
              onClick={() =>
                void window.browser.updateBrowserExperience({
                  type: 'sidebar-tool-open',
                  toolId: tool.toolId,
                })
              }
            >
              <Icon name={sidebarToolIcon(tool.toolId)} />
            </SidebarIconButton>
          ))}
      </nav>
      <div className="sidebar-spacer" />
      <nav className="sidebar-apps" aria-label="Sidebar apps">
        {experience.sidebar.apps
          .filter((app) => app.enabled)
          .sort((left, right) => left.order - right.order)
          .map((app) => (
            <SidebarIconButton
              key={app.appId}
              className={app.appId === experience.sidebar.activeAppId ? 'active' : ''}
              label={app.name}
              tooltip={`${app.name} — ${app.appId === experience.sidebar.activeAppId ? 'close' : 'open'} its isolated sidebar panel`}
              onClick={() =>
                void window.browser.updateBrowserExperience({
                  type: 'sidebar-app-open',
                  appId: app.appId === experience.sidebar.activeAppId ? undefined : app.appId,
                })
              }
            >
              <SidebarAppIcon appId={app.appId} name={app.name} />
            </SidebarIconButton>
          ))}
      </nav>
      {experience.sidebar.activeAppId && (
        <div className="sidebar-panel-controls" aria-label="Sidebar panel controls">
          {(() => {
            const app = experience.sidebar.apps.find(
              (item) => item.appId === experience.sidebar.activeAppId,
            );
            return app ? (
              <SidebarIconButton
                label={app.muted ? `Unmute ${app.name}` : `Mute ${app.name}`}
                tooltip={`${app.muted ? 'Unmute' : 'Mute'} ${app.name} — control audio from this sidebar app`}
                onClick={() =>
                  void window.browser.updateBrowserExperience({
                    type: 'sidebar-app-mute',
                    appId: app.appId,
                    muted: !app.muted,
                  })
                }
              >
                <Icon name={app.muted ? 'volume-muted' : 'volume'} />
              </SidebarIconButton>
            ) : null;
          })()}
        </div>
      )}
      <SidebarIconButton
        className="sidebar-mode"
        label={`Use ${experience.sidebar.panelMode === 'overlay' ? 'pinned' : 'overlay'} sidebar panels`}
        tooltip={
          experience.sidebar.panelMode === 'overlay'
            ? 'Pin sidebar panel — keep it beside the current page'
            : 'Overlay sidebar panel — show it above the current page'
        }
        onClick={() =>
          void window.browser.updateBrowserExperience({
            type: 'sidebar-mode',
            mode: experience.sidebar.panelMode === 'overlay' ? 'pinned' : 'overlay',
          })
        }
      >
        <Icon name={experience.sidebar.panelMode === 'overlay' ? 'pin-panel' : 'overlay-panel'} />
      </SidebarIconButton>
    </aside>
  );
}

function SidebarIconButton({
  label,
  tooltip,
  children,
  ...props
}: Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'aria-label' | 'title' | 'type'> & {
  label: string;
  tooltip: string;
}) {
  return (
    <button type="button" aria-label={label} title={tooltip} {...props}>
      {children}
    </button>
  );
}

interface SidebarResizeDrag {
  pointerId: number;
  startScreenX: number;
  startWidth: number;
  currentWidth: number;
}

export function BrowserSidebarResizeHandle(): React.JSX.Element {
  const [panelWidth, setPanelWidth] = useState(emptyState.experience.sidebar.panelWidth);
  const [dragging, setDragging] = useState(false);
  const drag = useRef<SidebarResizeDrag | null>(null);

  useEffect(() => {
    void window.browser
      .getState()
      .then((state) => setPanelWidth(state.experience.sidebar.panelWidth));
    return window.browser.onStateChanged((state) => {
      if (!drag.current) setPanelWidth(state.experience.sidebar.panelWidth);
    });
  }, []);

  const persistWidth = (width: number) => {
    const bounded = clampSidebarPanelWidth(width);
    setPanelWidth(bounded);
    void window.browser.updateBrowserExperience({
      type: 'sidebar-resize',
      width: bounded,
    });
  };

  return (
    <div
      className={`sidebar-resize-handle${dragging ? ' dragging' : ''}`}
      role="separator"
      aria-label="Resize sidebar panel"
      aria-orientation="vertical"
      aria-valuemin={SIDEBAR_PANEL_MIN_WIDTH}
      aria-valuemax={SIDEBAR_PANEL_MAX_WIDTH}
      aria-valuenow={panelWidth}
      tabIndex={0}
      title="Drag to resize sidebar panel"
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = {
          pointerId: event.pointerId,
          startScreenX: event.screenX,
          startWidth: panelWidth,
          currentWidth: panelWidth,
        };
        setDragging(true);
      }}
      onPointerMove={(event) => {
        const current = drag.current;
        if (!current || current.pointerId !== event.pointerId) return;
        const width = clampSidebarPanelWidth(
          current.startWidth + event.screenX - current.startScreenX,
        );
        if (width === current.currentWidth) return;
        current.currentWidth = width;
        setPanelWidth(width);
        void window.browser.updateBrowserExperience({
          type: 'sidebar-resize',
          width,
          persist: false,
        });
      }}
      onPointerUp={(event) => {
        const current = drag.current;
        if (!current || current.pointerId !== event.pointerId) return;
        event.currentTarget.releasePointerCapture(event.pointerId);
        drag.current = null;
        setDragging(false);
        persistWidth(current.currentWidth);
      }}
      onPointerCancel={(event) => {
        const current = drag.current;
        if (!current || current.pointerId !== event.pointerId) return;
        drag.current = null;
        setDragging(false);
        persistWidth(current.currentWidth);
      }}
      onKeyDown={(event) => {
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
        event.preventDefault();
        persistWidth(panelWidth + (event.key === 'ArrowLeft' ? -20 : 20));
      }}
    />
  );
}

function sidebarToolIcon(
  toolId: BrowserWindowState['experience']['sidebar']['tools'][number]['toolId'],
): IconName {
  const icons = {
    'tab-search': 'search',
    bookmarks: 'bookmark',
    history: 'history',
    downloads: 'download',
    resources: 'resources',
    settings: 'settings',
  } satisfies Record<typeof toolId, IconName>;
  return icons[toolId];
}
