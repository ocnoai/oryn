import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const rendererRoot = join(import.meta.dirname, '../../renderer/src');
const chrome = readFileSync(join(rendererRoot, 'BrowserChrome.tsx'), 'utf8');
const browserWindow = readFileSync(join(import.meta.dirname, '../browser-window.ts'), 'utf8');
const ipcHandlers = readFileSync(join(import.meta.dirname, '../ipc-handlers.ts'), 'utf8');
const componentStyles = readFileSync(join(rendererRoot, 'styles.css'), 'utf8');
const componentLibrary = readFileSync(join(rendererRoot, 'components/ui/index.tsx'), 'utf8');
const componentLibraryStyles = readFileSync(join(rendererRoot, 'components/ui/ui.css'), 'utf8');
const workspace = readFileSync(join(rendererRoot, 'model-workspace.tsx'), 'utf8');
const workspaceStyles = readFileSync(join(rendererRoot, 'model-workspace.css'), 'utf8');
const theme = readFileSync(
  join(import.meta.dirname, '../../../../../packages/theme/src/theme.css'),
  'utf8',
);

describe('trusted UI accessibility and theme contract', () => {
  it('uses semantic tokens without component-level literal colors', () => {
    expect(componentStyles).not.toMatch(/#[0-9a-f]{3,8}\b/iu);
    expect(componentStyles).not.toMatch(/\b(?:rgb|hsl)a?\(/iu);
    expect(componentStyles).toContain("@import '../../../../../packages/theme/src/theme.css'");
  });

  it('never references an undefined design token', () => {
    const styles = [theme, componentStyles, componentLibraryStyles, workspaceStyles].join('\n');
    const references = new Set(
      [...styles.matchAll(/var\((--[a-z0-9-]+)/gu)].map((match) => match[1]!),
    );
    const definitions = new Set(
      [...styles.matchAll(/(--[a-z0-9-]+)\s*:/gu)].map((match) => match[1]!),
    );
    definitions.add('--browser-content-height');
    definitions.add('--ui-progress-value');
    expect([...references].filter((token) => !definitions.has(token)).sort()).toEqual([]);
  });

  it('provides light, dark, system, focus, and reduced-motion behavior', () => {
    expect(theme).toContain("[data-theme='light']");
    expect(theme).toContain("[data-theme='dark']");
    expect(theme).toContain("[data-theme='system']");
    expect(theme).toContain('@media (prefers-color-scheme: dark)');
    expect(theme).toContain('@media (prefers-reduced-motion: reduce)');
    expect(theme).toContain(':focus-visible');
    expect(workspaceStyles).toContain('@media (prefers-reduced-motion: reduce)');
  });

  it('keeps every workspace surface labelled and keyboard-native', () => {
    for (const contract of [
      'chat-v1',
      'text-to-image-v1',
      'image-to-image-v1',
      'text-to-music-v1',
      'speech-synthesis-v1',
      'transcription-v1',
      'embeddings-v1',
      'reranking-v1',
      'text-to-video-v1',
      'image-to-video-v1',
    ])
      expect(workspace).toContain(contract);
    expect(workspace).toContain('role="alert"');
    expect(workspace).toContain('label="Generation progress"');
    expect(workspace).toContain('<label className="field">');
    expect(workspaceStyles).not.toMatch(/#[0-9a-f]{3,8}\b/iu);
  });

  it('keeps critical dialogs, progress, errors, and notifications semantically labelled', () => {
    expect(chrome).toContain('aria-modal="true"');
    expect(chrome).toContain('aria-labelledby="oryn-onboarding-title"');
    expect(chrome).toContain('label={`Download progress');
    expect(componentLibrary).toContain('role="progressbar"');
    expect(componentLibrary).toContain('aria-valuenow={boundedValue}');
    expect(chrome).toContain('role="alert"');
    expect(chrome).toContain('role="status"');
    expect(chrome).toContain('useFocusTrap');
    expect(chrome).toContain('Access restricted by model host · Set up access from its model page');
    expect(chrome).toContain('Connect the Engine with a read token');
    expect(chrome.replace(/\s+/gu, ' ')).toContain('never exposed to websites or model code');
    expect(chrome).toContain('is safely available right now');
    expect(chrome).toContain('Close memory-heavy apps or choose a smaller compatible model');
    expect(chrome).toContain('Trust and load this model?');
    expect(chrome).toContain('This exact model has not been verified by Oryn.');
    expect(chrome).toContain('Trust and download exact model');
    expect(workspace).toContain('Local · user trusted');
    expect(workspace).toContain('<summary>Technical details</summary>');
    expect(workspace).toContain('<code>{error.code}</code>');
    expect(workspace).toContain('failureTitle(error.code)');
    expect(workspace).toContain("{error ? 'Try again' : actionLabel(contract)}");
    expect(ipcHandlers).toContain(
      'The installed model could not be loaded into its local runtime.',
    );
    expect(ipcHandlers).toContain('The model loaded, but local text generation failed.');
    expect(chrome).toContain('window.browser.showHuggingFacePageActionDetails()');
    expect(chrome).toContain('Cancel download');
    expect(chrome).toContain(
      'You can close this panel or switch tabs while the download continues.',
    );
    expect(chrome).not.toContain(
      'void window.browser.cancelHuggingFacePageAction();\n                if',
    );
  });

  it('dismisses the browser menu without changing website layout ownership', () => {
    expect(chrome).toContain('resolveBrowserChromeLayout');
    expect(chrome).toContain('className="popover-dismiss-layer"');
    expect(chrome).toContain(
      '<PopoverDismissLayer onDismiss={() => setWorkspaceMenuOpen(false)} />',
    );
    expect(chrome).toContain('<PopoverDismissLayer onDismiss={() => setTabMenu(null)} />');
    expect(chrome).toContain('style={{ left: position.left, top: position.top }}');
    expect(componentStyles).not.toMatch(/\.tab-context-menu\s*\{[^}]*right:/su);
    expect(chrome).toContain("event.key === 'Escape'");
    expect(componentStyles).toMatch(
      /\.popover-dismiss-layer\s*\{[^}]*background: var\(--color-background-canvas\);[^}]*opacity: 0\.01;/su,
    );
  });

  it('supports dragging an unpinned tab into an existing tab island', () => {
    expect(chrome).toContain("closest<HTMLElement>('[data-island-id]')");
    expect(chrome).toContain('onPointerMove={moveDrag}');
    expect(chrome).toContain('onPointerUp={finishDrag}');
    expect(chrome).toContain("type: 'tab-move-island'");
    expect(chrome).toContain(
      "updateBrowserExperience({ type: 'tab-move-island', tabId: drag.tabId })",
    );
    expect(componentStyles).toContain('.tab-island.drag-over');
    expect(componentStyles).toContain('.tab-strip.ungroup-drop-active');
  });

  it('keeps primary-click tab activation reliable while a popover or drag tracker is active', () => {
    expect(chrome).toContain('activateTab(drag.tabId);');
    expect(chrome).toContain('setWorkspaceMenuOpen(false);');
    expect(chrome).toContain('setTabMenu(null);');
    expect(chrome).toContain('onClick={() => activateTab(tab.tabId)}');
    expect(componentStyles).toMatch(
      /\.popover-dismiss-layer\s*\{[^}]*top: calc\(var\(--browser-tab-height\) \+ var\(--space-2\)\);/su,
    );
    expect(componentStyles).toMatch(
      /\.workspace-menu\s*\{[^}]*top: calc\(var\(--browser-tab-height\) \+ var\(--space-2\)\);/su,
    );
    expect(componentStyles).not.toMatch(/\.popover-dismiss-layer\s*\{[^}]*inset: 0;/su);
  });

  it('uses the platform-standard Chrome shortcut for tab search', () => {
    expect(chrome).toContain("event.code === 'KeyA'");
    expect(chrome).toContain("return isMacOS() ? '⌘ Shift A' : 'Ctrl Shift A'");
    expect(chrome).not.toContain("event.code === 'Space'");
  });

  it('uses the square shared icon-button geometry for notification dismissal', () => {
    expect(chrome).toMatch(/icon="close"\s+label="Dismiss notification"\s+variant="surface"/u);
    expect(componentStyles).not.toContain('.browser-notification button');
  });

  it('keeps system capacity and pressure compact in Local AI resources', () => {
    for (const label of [
      'System memory for Local AI',
      'System memory',
      'Available now',
      'Browser core',
      'Oryn AI runtime',
    ]) {
      expect(chrome).toContain(label);
    }
    expect(chrome).toContain('pressure={state.memoryPressure}');
    expect(chrome).toContain('`${titleCase(pressure)} pressure`');
    expect(chrome).not.toContain('Memory is available for Local AI work');
    expect(chrome).not.toContain('shared Chromium pages more than once');
    for (const caption of [
      'Measured total',
      'Measured by Engine',
      'Measured private memory',
      'Measured supervisor',
    ]) {
      expect(chrome).not.toContain(caption);
    }
    expect(componentStyles).not.toContain('.resource-decision');
    expect(componentStyles).not.toContain('.resource-control-card footer');
    expect(chrome).not.toContain('Local compute overview');
    expect(browserWindow).toContain('systemTotalBytes: engineResources.systemMemory?.totalBytes');
    expect(browserWindow).toContain(
      'systemAvailableBytes: engineResources.systemMemory?.availableBytes',
    );
    expect(browserWindow).toMatch(/engineResources\?\.workers\?\.reduce/u);
    expect(browserWindow).toContain('process.getProcessMemoryInfo()');
    expect(browserWindow).not.toContain('getAppMetrics()');
    expect(browserWindow).not.toContain("from 'node:os'");
  });

  it('meets WCAG AA contrast for primary and secondary text in both themes', () => {
    expect(contrast('#111318', '#ffffff')).toBeGreaterThanOrEqual(4.5);
    expect(contrast('#475467', '#ffffff')).toBeGreaterThanOrEqual(4.5);
    expect(contrast('#f5f7fa', '#191c22')).toBeGreaterThanOrEqual(4.5);
    expect(contrast('#c8ced8', '#191c22')).toBeGreaterThanOrEqual(4.5);
  });

  it('keeps warning copy readable on subtle warning surfaces in both themes', () => {
    expect(componentStyles).toMatch(
      /\.compatibility-warning\s*\{[^}]*color: var\(--color-text-primary\);/su,
    );
    expect(componentStyles).toMatch(
      /\.compatibility-warning strong\s*\{[^}]*color: var\(--color-warning\);/su,
    );
    expect(componentStyles).toMatch(
      /\.compatibility-warning p\s*\{[^}]*color: var\(--color-text-secondary\);/su,
    );
    expect(componentStyles).not.toMatch(
      /\.compatibility-warning\s*\{[^}]*color: var\(--color-warning-on\);/su,
    );
    expect(contrast('#111318', '#fffaeb')).toBeGreaterThanOrEqual(4.5);
    expect(contrast('#475467', '#fffaeb')).toBeGreaterThanOrEqual(4.5);
    expect(contrast('#f5f7fa', '#3a2a0f')).toBeGreaterThanOrEqual(4.5);
    expect(contrast('#c8ced8', '#3a2a0f')).toBeGreaterThanOrEqual(4.5);
  });
});

function contrast(foreground: string, background: string): number {
  const lighter = Math.max(luminance(foreground), luminance(background));
  const darker = Math.min(luminance(foreground), luminance(background));
  return (lighter + 0.05) / (darker + 0.05);
}

function luminance(value: string): number {
  const channels = value
    .slice(1)
    .match(/.{2}/gu)!
    .map((item) => Number.parseInt(item, 16) / 255)
    .map((item) => (item <= 0.04045 ? item / 12.92 : ((item + 0.055) / 1.055) ** 2.4));
  return channels[0]! * 0.2126 + channels[1]! * 0.7152 + channels[2]! * 0.0722;
}
