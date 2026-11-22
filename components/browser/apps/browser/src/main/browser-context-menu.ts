import type { ContextMenuParams, MenuItemConstructorOptions, WebContents } from 'electron';
import { browserDocumentAction } from './browser-document-action';
import type { ExternalApplicationHandoff } from './external-application-handoff';

export function browserContextMenu(
  contents: WebContents,
  params: ContextMenuParams,
  actions: {
    owned: () => boolean;
    openTab(url: string, activate?: boolean): void;
    openWindow(url: string): void;
    openPrivateWindow(url: string): void;
    copyText(text: string): Promise<void>;
    searchURL(text: string): string;
    savePage(): Promise<unknown>;
    printPage(): Promise<unknown>;
    failed(): Promise<void>;
    externalApplications?: ExternalApplicationHandoff;
  },
) {
  const document = browserDocumentAction(contents, actions.owned);
  const frame = params.frame;
  const frameURL = frame?.url;
  const frameToken = frame?.frameToken;
  const current = () => {
    try {
      return Boolean(
        document.current() &&
        frame &&
        !frame.detached &&
        frame.url === frameURL &&
        frame.frameToken === frameToken,
      );
    } catch {
      return false;
    }
  };
  const run = (action: () => unknown) => () => {
    if (!current()) return;
    try {
      Promise.resolve(action()).catch(() => void actions.failed().catch(() => {}));
    } catch {
      void actions.failed().catch(() => {});
    }
  };
  const runInFocusedFrame = (action: () => unknown) =>
    run(() => {
      if (contents.focusedFrame?.frameToken === frameToken) return action();
    });
  const template: MenuItemConstructorOptions[] = [];
  if (/^https?:/u.test(params.linkURL)) {
    template.push(
      { label: 'Open link in new tab', click: run(() => actions.openTab(params.linkURL)) },
      { label: 'Open link in new window', click: run(() => actions.openWindow(params.linkURL)) },
      {
        label: 'Open link in private window',
        click: run(() => actions.openPrivateWindow(params.linkURL)),
      },
      { label: 'Save link as…', click: run(() => contents.downloadURL(params.linkURL)) },
    );
  } else if (actions.externalApplications?.canRequestMenu(contents, params.linkURL, frame)) {
    template.push({
      label: 'Open link in application…',
      click: run(() => actions.externalApplications!.requestMenu(contents, params.linkURL, frame)),
    });
  }
  if (params.linkURL)
    template.push(
      { label: 'Copy link address', click: run(() => actions.copyText(params.linkURL)) },
      { type: 'separator' },
    );
  if (params.mediaType === 'image') {
    template.push({
      label: 'Copy image',
      click: run(() => contents.copyImageAt(params.x, params.y)),
    });
    if (/^https?:/u.test(params.srcURL))
      template.push(
        { label: 'Copy image address', click: run(() => actions.copyText(params.srcURL)) },
        { label: 'Open image in new tab', click: run(() => actions.openTab(params.srcURL, true)) },
        { label: 'Save image as…', click: run(() => contents.downloadURL(params.srcURL)) },
      );
  }
  if (params.selectionText)
    template.push({
      label: 'Search selected text',
      click: run(() => actions.openTab(actions.searchURL(params.selectionText), true)),
    });
  for (const word of params.dictionarySuggestions.slice(0, 5))
    template.push({
      label: word,
      click: runInFocusedFrame(() => contents.replaceMisspelling(word)),
    });
  if (params.misspelledWord && params.spellcheckEnabled && contents.session.isPersistent())
    template.push({
      label: 'Add to dictionary',
      click: run(() => {
        if (!contents.session.addWordToSpellCheckerDictionary(params.misspelledWord))
          throw new Error('DICTIONARY_UPDATE_FAILED');
      }),
    });

  // Role-only menu items operate on whichever WebContents is focused at click time.
  // Use Chromium's native editing commands with the original contents and edit flags instead.
  const edit = (
    label: string,
    enabled: boolean,
    action: () => void,
  ): MenuItemConstructorOptions => ({
    label,
    enabled,
    click: runInFocusedFrame(action),
  });
  if (params.isEditable) {
    template.push(
      { type: 'separator' },
      edit('Undo', params.editFlags.canUndo, () => contents.undo()),
      edit('Redo', params.editFlags.canRedo, () => contents.redo()),
      { type: 'separator' },
      edit('Cut', params.editFlags.canCut, () => contents.cut()),
      edit('Copy', params.editFlags.canCopy, () => contents.copy()),
      edit('Paste', params.editFlags.canPaste, () => contents.paste()),
      edit('Paste and Match Style', params.editFlags.canPaste, () => contents.pasteAndMatchStyle()),
      edit('Select All', params.editFlags.canSelectAll, () => contents.selectAll()),
    );
  } else {
    if (params.selectionText)
      template.push(edit('Copy', params.editFlags.canCopy, () => contents.copy()));
    template.push(edit('Select All', params.editFlags.canSelectAll, () => contents.selectAll()));
  }
  template.push(
    { type: 'separator' },
    {
      label: 'Back',
      enabled: contents.navigationHistory.canGoBack(),
      click: run(() => contents.navigationHistory.goBack()),
    },
    {
      label: 'Forward',
      enabled: contents.navigationHistory.canGoForward(),
      click: run(() => contents.navigationHistory.goForward()),
    },
    { label: 'Reload', click: run(() => contents.reload()) },
    { type: 'separator' },
    {
      label: 'Save Page As…',
      enabled: /^(https?:|file:)/u.test(contents.getURL()),
      click: run(actions.savePage),
    },
    { label: 'Print…', click: run(actions.printPage) },
    { label: 'Inspect', click: run(() => contents.inspectElement(params.x, params.y)) },
  );
  return { template, dispose: document.dispose };
}
