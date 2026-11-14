// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createRef, type ComponentProps } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AddressBar } from '../AddressBar';

afterEach(cleanup);

function mount(overrides: Partial<ComponentProps<typeof AddressBar>> = {}) {
  const props: ComponentProps<typeof AddressBar> = {
    activeTabId: 'tab-1',
    currentURL: 'oryn://new-tab/',
    inputRef: createRef<HTMLInputElement>(),
    suggestionsEnabled: true,
    tabs: [],
    bookmarks: [],
    history: [
      {
        historyId: 'history-1',
        url: 'https://oryn.example/guide',
        title: 'Oryn guide',
        visitedAt: '2026-01-02T00:00:00Z',
      },
      {
        historyId: 'history-2',
        url: 'https://oryn.example/reference',
        title: 'Oryn reference',
        visitedAt: '2026-01-01T00:00:00Z',
      },
    ],
    leading: <button type="button">Site controls</button>,
    trailing: <button type="button">Bookmark</button>,
    onNavigate: vi.fn(),
    onSuggestionsOpenChange: vi.fn(),
    ...overrides,
  };
  const result = render(<AddressBar {...props} />);
  const input = screen.getByRole('combobox', { name: 'Address and search' }) as HTMLInputElement;
  return { ...result, props, input, user: userEvent.setup() };
}

describe('mounted address bar suggestions', () => {
  it('shows local matches while typing and publishes popup visibility', async () => {
    const { props, input, user } = mount();
    expect(input.value).toBe('');
    expect(screen.queryByRole('listbox')).toBeNull();
    await user.type(input, 'Oryn');
    expect(screen.getByRole('listbox', { name: 'Address suggestions' })).toBeTruthy();
    expect(screen.getAllByRole('option')).toHaveLength(2);
    expect(input.getAttribute('aria-expanded')).toBe('true');
    expect(input.getAttribute('aria-activedescendant')).toBeNull();
    expect(props.onSuggestionsOpenChange).toHaveBeenLastCalledWith(true);
    expect(props.onNavigate).not.toHaveBeenCalled();
  });

  it('submits the typed query on Enter without automatically selecting a suggestion', async () => {
    const { props, input, user } = mount();
    await user.type(input, 'oryn');
    await user.keyboard('{Enter}');
    expect(props.onNavigate).toHaveBeenCalledExactlyOnceWith('oryn');
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(props.onSuggestionsOpenChange).toHaveBeenLastCalledWith(false);
  });

  it('selects a suggested address with ArrowDown and navigates to it on Enter', async () => {
    const { props, input, user } = mount();
    await user.type(input, 'oryn');
    await user.keyboard('{ArrowDown}');
    expect(input.getAttribute('aria-activedescendant')).toBe('address-suggestion-0');
    expect(screen.getAllByRole('option')[0]?.getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(input);
    await user.keyboard('{Enter}');
    expect(props.onNavigate).toHaveBeenCalledExactlyOnceWith('https://oryn.example/guide');
  });

  it('wraps upward from the first suggestion without losing the keyboard selection', async () => {
    const { props, input, user } = mount();
    await user.type(input, 'oryn');
    await user.keyboard('{ArrowDown}{ArrowUp}');
    expect(input.getAttribute('aria-activedescendant')).toBe('address-suggestion-1');
    await user.keyboard('{Enter}');
    expect(props.onNavigate).toHaveBeenCalledExactlyOnceWith('https://oryn.example/reference');
  });

  it('keeps the input focused through mouse down so clicking submits the chosen address', async () => {
    const { props, input, user } = mount();
    await user.type(input, 'oryn');
    const option = screen.getAllByRole('option')[1]!;
    expect(fireEvent.mouseDown(option)).toBe(false);
    expect(document.activeElement).toBe(input);
    await user.click(option);
    expect(props.onNavigate).toHaveBeenCalledExactlyOnceWith('https://oryn.example/reference');
    expect(screen.queryByRole('listbox')).toBeNull();
  });

  it('dismisses on Escape and resumes matching after further typing', async () => {
    const { props, input, user } = mount();
    await user.type(input, 'ory');
    await user.keyboard('{ArrowDown}{Escape}');
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(input.getAttribute('aria-expanded')).toBe('false');
    expect(input.value).toBe('ory');
    expect(document.activeElement).toBe(input);
    expect(props.onNavigate).not.toHaveBeenCalled();
    await user.type(input, 'n');
    expect(screen.getAllByRole('option')).toHaveLength(2);
    expect(input.getAttribute('aria-activedescendant')).toBeNull();
  });

  it('respects an explicitly disabled suggestions preference while allowing typed navigation', async () => {
    const { props, input, user } = mount({ suggestionsEnabled: false });
    await user.type(input, 'oryn');
    await user.keyboard('{ArrowDown}');
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(input.getAttribute('aria-expanded')).toBe('false');
    await user.keyboard('{Enter}');
    expect(props.onNavigate).toHaveBeenCalledExactlyOnceWith('oryn');
  });

  it('resets the draft and closes the popup when the active tab changes', async () => {
    const { props, input, user, rerender } = mount();
    await user.type(input, 'oryn');
    await user.keyboard('{ArrowDown}');
    rerender(
      <AddressBar {...props} activeTabId="tab-2" currentURL="https://different.example/page" />,
    );
    expect(input.value).toBe('https://different.example/page');
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(input.getAttribute('aria-activedescendant')).toBeNull();
    expect(props.onSuggestionsOpenChange).toHaveBeenLastCalledWith(false);
    expect(props.onNavigate).not.toHaveBeenCalled();
  });

  it('does not submit or change selection when Enter confirms an IME composition', async () => {
    const { props, input, user } = mount();
    await user.type(input, 'oryn');
    fireEvent.compositionStart(input);
    expect(fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', isComposing: true })).toBe(
      false,
    );
    expect(props.onNavigate).not.toHaveBeenCalled();
    expect(input.getAttribute('aria-activedescendant')).toBeNull();
    expect(screen.getByRole('listbox')).toBeTruthy();
    fireEvent.compositionEnd(input);
    await user.keyboard('{Enter}');
    expect(props.onNavigate).toHaveBeenCalledExactlyOnceWith('oryn');
  });
});
