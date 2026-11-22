import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
  type RefObject,
} from 'react';
import type { BrowserBookmarkItem, BrowserHistoryItem, BrowserTabState } from '../../../shared/ipc';
import { isOrynNewTabURL } from '../../../shared/browser-urls';
import { getNavigationSuggestions } from '../navigation-suggestions';
import { Icon } from './ui';

interface AddressBarProps {
  activeTabId: string;
  currentURL: string;
  inputRef: RefObject<HTMLInputElement | null>;
  suggestionsEnabled: boolean;
  tabs: BrowserTabState[];
  bookmarks: BrowserBookmarkItem[];
  history: BrowserHistoryItem[];
  leading: ReactNode;
  trailing: ReactNode;
  onNavigate: (value: string) => void;
  onSuggestionsOpenChange: (open: boolean) => void;
}

const sourceLabels = { tab: 'Open tab', bookmark: 'Bookmark', history: 'History' };
const sourceIcons = { tab: 'globe', bookmark: 'star', history: 'history' } as const;

export function AddressBar({
  activeTabId,
  currentURL,
  inputRef,
  suggestionsEnabled,
  tabs,
  bookmarks,
  history,
  leading,
  trailing,
  onNavigate,
  onSuggestionsOpenChange,
}: AddressBarProps): React.JSX.Element {
  const displayURL = isOrynNewTabURL(currentURL) ? '' : currentURL;
  const [value, setValue] = useState(displayURL);
  const [focused, setFocused] = useState(false);
  const [dismissed, setDismissed] = useState(true);
  const [selectedURL, setSelectedURL] = useState<string | null>(null);
  const suggestions = useMemo(
    () => (suggestionsEnabled ? getNavigationSuggestions(value, tabs, bookmarks, history) : []),
    [suggestionsEnabled, value, tabs, bookmarks, history],
  );
  const open = focused && !dismissed && suggestions.length > 0;
  const selectedIndex = suggestions.findIndex((item) => item.url === selectedURL);

  useEffect(() => {
    if (!focused) setValue(displayURL);
  }, [displayURL, focused]);

  useEffect(() => {
    setValue(displayURL);
    setDismissed(true);
    setSelectedURL(null);
  }, [activeTabId]);

  // Publish visibility before paint so Electron expands the trusted native view.
  useLayoutEffect(() => {
    onSuggestionsOpenChange(open);
  }, [open, onSuggestionsOpenChange]);

  useEffect(() => {
    if (open && selectedIndex >= 0) {
      document.getElementById(`address-suggestion-${selectedIndex}`)?.scrollIntoView?.({
        block: 'nearest',
      });
    }
  }, [open, selectedIndex]);

  const navigate = (destination: string) => {
    setDismissed(true);
    setSelectedURL(null);
    inputRef.current?.blur();
    onNavigate(destination);
  };

  return (
    <>
      {open && (
        <div
          className="popover-dismiss-layer"
          aria-hidden="true"
          onPointerDown={() => {
            setDismissed(true);
            inputRef.current?.blur();
          }}
        />
      )}
      <div className={`address-bar${open ? ' address-bar-open' : ''}`}>
        <form
          className="omnibox-form"
          onSubmit={(event) => {
            event.preventDefault();
            navigate(open && selectedIndex >= 0 ? suggestions[selectedIndex]!.url : value);
          }}
        >
          {leading}
          <input
            ref={inputRef}
            className="omnibox"
            role="combobox"
            aria-label="Address and search"
            aria-autocomplete="list"
            aria-expanded={open}
            aria-controls={open ? 'address-suggestions' : undefined}
            aria-activedescendant={
              open && selectedIndex >= 0 ? `address-suggestion-${selectedIndex}` : undefined
            }
            autoComplete="off"
            spellCheck={false}
            value={value}
            onFocus={() => setFocused(true)}
            onBlur={() => {
              setFocused(false);
              setDismissed(true);
              setSelectedURL(null);
            }}
            onChange={(event) => {
              setValue(event.target.value);
              setDismissed(false);
              setSelectedURL(null);
            }}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing) {
                if (event.key === 'Enter') event.preventDefault();
                return;
              }
              if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && suggestions.length) {
                event.preventDefault();
                const nextIndex =
                  event.key === 'ArrowDown'
                    ? (selectedIndex + 1) % suggestions.length
                    : (selectedIndex <= 0 ? suggestions.length : selectedIndex) - 1;
                setDismissed(false);
                setSelectedURL(suggestions[nextIndex]!.url);
              }
              if (event.key === 'Escape') {
                event.preventDefault();
                setDismissed(true);
                setSelectedURL(null);
              }
            }}
          />
          {trailing}
        </form>
        {open && (
          <ul
            id="address-suggestions"
            className="address-suggestions"
            role="listbox"
            aria-label="Address suggestions"
            data-trusted-overlay="true"
          >
            {suggestions.map((suggestion, index) => (
              <li
                key={suggestion.url}
                id={`address-suggestion-${index}`}
                role="option"
                aria-selected={index === selectedIndex}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => navigate(suggestion.url)}
              >
                <Icon name={sourceIcons[suggestion.source]} />
                <span className="address-suggestion-text">
                  <strong>{suggestion.title || suggestion.url}</strong>
                  <span>{suggestion.url}</span>
                </span>
                <small>{sourceLabels[suggestion.source]}</small>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}
