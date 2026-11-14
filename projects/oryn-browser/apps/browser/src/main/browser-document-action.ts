import type { WebContents } from 'electron';

/** A native action belongs to the original live document, even across a same-URL reload. */
export function browserDocumentAction(contents: WebContents, owned: () => boolean = () => true) {
  let invalidated = false;
  const frame = contents.mainFrame;
  const url = contents.getURL();
  const invalidate = () => {
    invalidated = true;
  };
  contents.on('did-start-navigation', invalidate);
  contents.on('render-process-gone', invalidate);
  contents.on('destroyed', invalidate);
  return {
    current() {
      try {
        return (
          !invalidated &&
          owned() &&
          !contents.isDestroyed() &&
          Boolean(frame) &&
          !frame?.detached &&
          contents.mainFrame === frame &&
          contents.getURL() === url
        );
      } catch {
        return false;
      }
    },
    dispose() {
      contents.removeListener('did-start-navigation', invalidate);
      contents.removeListener('render-process-gone', invalidate);
      contents.removeListener('destroyed', invalidate);
    },
  };
}
