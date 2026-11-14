// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BrowserDialog, requestChoice, requestText } from '../BrowserDialog';

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function (this: HTMLDialogElement) {
    this.open = true;
  });
});
afterEach(cleanup);
describe('mounted browser dialogs', () => {
  it('focuses the input, rejects empty text, saves once and restores focus', async () => {
    render(
      <>
        <button>Rename</button>
        <BrowserDialog />
      </>,
    );
    const trigger = screen.getByText('Rename');
    trigger.focus();
    let result!: Promise<string | null>;
    act(() => {
      result = requestText('Workspace name', 'Draft');
    });
    const input = screen.getByRole('textbox', { name: 'Workspace name' });
    expect(document.activeElement).toBe(input);
    fireEvent.change(input, { target: { value: ' ' } });
    expect((screen.getByText('Save') as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(input, { target: { value: 'Research' } });
    fireEvent.click(screen.getByText('Save'));
    expect(await result).toBe('Research');
    expect(document.activeElement).toBe(trigger);
  });
  it('B19 Cancel does not select either workspace deletion action', async () => {
    render(<BrowserDialog />);
    let result!: Promise<string | null>;
    act(() => {
      result = requestChoice('Delete workspace?', [
        { label: 'Move tabs to Main', value: 'move' },
        { label: 'Close tabs', value: 'close' },
      ]);
    });
    fireEvent.click(screen.getByText('Cancel'));
    expect(await result).toBeNull();
  });
  it('native Escape cancellation dismisses without submitting', async () => {
    render(<BrowserDialog />);
    let result!: Promise<string | null>;
    act(() => {
      result = requestText('Folder name', 'New folder');
    });
    fireEvent(
      screen.getByRole('dialog'),
      new Event('cancel', { bubbles: false, cancelable: true }),
    );
    expect(await result).toBeNull();
  });
});
