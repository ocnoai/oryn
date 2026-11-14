import { EventEmitter } from 'node:events';
import type { ContextMenuParams, MenuItemConstructorOptions, WebContents } from 'electron';
import { describe, expect, it, vi } from 'vitest';
import { browserContextMenu } from '../browser-context-menu';

function fixture() {
  const frame = { url: 'https://website.test/page', frameToken: 'main', detached: false };
  const contents = Object.assign(new EventEmitter(), {
    mainFrame: frame,
    focusedFrame: frame,
    getURL: () => frame.url,
    isDestroyed: () => false,
    session: {
      isPersistent: () => true,
      addWordToSpellCheckerDictionary: vi.fn().mockReturnValue(true),
    },
    navigationHistory: {
      canGoBack: () => true,
      canGoForward: () => false,
      goBack: vi.fn(),
      goForward: vi.fn(),
    },
    undo: vi.fn(),
    redo: vi.fn(),
    cut: vi.fn(),
    copy: vi.fn(),
    paste: vi.fn(),
    pasteAndMatchStyle: vi.fn(),
    selectAll: vi.fn(),
    reload: vi.fn(),
    replaceMisspelling: vi.fn(),
    inspectElement: vi.fn(),
    copyImageAt: vi.fn(),
    downloadURL: vi.fn(),
  }) as unknown as WebContents;
  const params = {
    frame,
    linkURL: '',
    srcURL: '',
    selectionText: '',
    dictionarySuggestions: [],
    isEditable: true,
    mediaType: 'none',
    misspelledWord: '',
    spellcheckEnabled: true,
    editFlags: {
      canUndo: true,
      canRedo: false,
      canCut: true,
      canCopy: true,
      canPaste: true,
      canSelectAll: true,
    },
  } as unknown as ContextMenuParams;
  const actions = {
    owned: () => true,
    openTab: vi.fn(),
    openWindow: vi.fn(),
    openPrivateWindow: vi.fn(),
    copyText: vi.fn().mockResolvedValue(undefined),
    searchURL: (value: string) => `https://search.test/?q=${encodeURIComponent(value)}`,
    savePage: vi.fn().mockResolvedValue(undefined),
    printPage: vi.fn().mockResolvedValue(undefined),
    failed: vi.fn().mockResolvedValue(undefined),
  };
  return { contents, params, actions, frame };
}

function click(template: MenuItemConstructorOptions[], label: string) {
  const item = template.find((entry) => entry.label === label)!;
  expect(item, `native menu contains ${label}`).toBeDefined();
  (item.click as () => void)();
}

describe('source-bound native context menu commands', () => {
  it('uses native edit flags and commands on the originating contents', () => {
    const { contents, params, actions } = fixture();
    const menu = browserContextMenu(contents, params, actions);
    expect(menu.template.find((item) => item.label === 'Redo')?.enabled).toBe(false);
    click(menu.template, 'Undo');
    click(menu.template, 'Paste and Match Style');
    click(menu.template, 'Select All');
    expect(contents.undo).toHaveBeenCalledOnce();
    expect(contents.pasteAndMatchStyle).toHaveBeenCalledOnce();
    expect(contents.selectAll).toHaveBeenCalledOnce();
    menu.dispose();
    expect(contents.listenerCount('did-start-navigation')).toBe(0);
  });

  it('does not retarget old menu actions after same-URL navigation or loss of tab ownership', () => {
    const reload = fixture();
    const menu = browserContextMenu(reload.contents, reload.params, reload.actions);
    reload.contents.emit('did-start-navigation', {}, reload.contents.getURL(), false, true);
    click(menu.template, 'Paste');
    click(menu.template, 'Back');
    click(menu.template, 'Save Page As…');
    expect(reload.contents.paste).not.toHaveBeenCalled();
    expect(reload.contents.navigationHistory.goBack).not.toHaveBeenCalled();
    expect(reload.actions.savePage).not.toHaveBeenCalled();
    menu.dispose();
    const moved = fixture();
    let owned = true;
    moved.actions.owned = () => owned;
    const oldMenu = browserContextMenu(moved.contents, moved.params, moved.actions);
    owned = false;
    click(oldMenu.template, 'Reload');
    expect(moved.contents.reload).not.toHaveBeenCalled();
    oldMenu.dispose();
  });

  it('adds source-bound image, spelling, Save and existing Print actions', () => {
    const { contents, params, actions } = fixture();
    params.mediaType = 'image';
    params.srcURL = 'https://website.test/image.png';
    params.misspelledWord = 'Oryn';
    params.dictionarySuggestions = ['Oryn suggestion'];
    const menu = browserContextMenu(contents, params, actions);
    click(menu.template, 'Copy image address');
    click(menu.template, 'Open image in new tab');
    click(menu.template, 'Add to dictionary');
    click(menu.template, 'Oryn suggestion');
    click(menu.template, 'Save Page As…');
    click(menu.template, 'Print…');
    expect(actions.copyText).toHaveBeenCalledWith(params.srcURL);
    expect(actions.openTab).toHaveBeenCalledWith(params.srcURL, true);
    expect(contents.session.addWordToSpellCheckerDictionary).toHaveBeenCalledWith('Oryn');
    expect(contents.replaceMisspelling).toHaveBeenCalledWith('Oryn suggestion');
    expect(actions.savePage).toHaveBeenCalledOnce();
    expect(actions.printPage).toHaveBeenCalledOnce();
    menu.dispose();
  });

  it('ignores a menu whose requesting frame has navigated or detached', () => {
    const { contents, params, actions, frame } = fixture();
    const menu = browserContextMenu(contents, params, actions);
    frame.detached = true;
    click(menu.template, 'Copy');
    expect(contents.copy).not.toHaveBeenCalled();
    menu.dispose();
  });

  it('does not send editing commands to a different newly focused frame', () => {
    const { contents, params, actions } = fixture();
    const menu = browserContextMenu(contents, params, actions);
    Object.defineProperty(contents, 'focusedFrame', { value: { frameToken: 'other-frame' } });
    click(menu.template, 'Paste');
    click(menu.template, 'Copy');
    expect(contents.paste).not.toHaveBeenCalled();
    expect(contents.copy).not.toHaveBeenCalled();
    menu.dispose();
  });

  it('does not replace spelling in another newly focused frame', () => {
    const { contents, params, actions } = fixture();
    params.dictionarySuggestions = ['Expected spelling'];
    const menu = browserContextMenu(contents, params, actions);
    Object.defineProperty(contents, 'focusedFrame', { value: { frameToken: 'other-frame' } });
    click(menu.template, 'Expected spelling');
    expect(contents.replaceMisspelling).not.toHaveBeenCalled();
    menu.dispose();
  });

  it('offers no persistent dictionary mutation in an in-memory private session', () => {
    const { contents, params, actions } = fixture();
    contents.session.isPersistent = () => false;
    params.misspelledWord = 'private-fixture-word';
    const menu = browserContextMenu(contents, params, actions);
    expect(menu.template.some((item) => item.label === 'Add to dictionary')).toBe(false);
    expect(contents.session.addWordToSpellCheckerDictionary).not.toHaveBeenCalled();
    menu.dispose();
  });
});
