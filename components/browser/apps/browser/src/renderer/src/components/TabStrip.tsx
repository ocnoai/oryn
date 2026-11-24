import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { IconButton } from './ui';

const OVERFLOW_TOLERANCE = 1;
const REVEAL_PADDING = 8;
const TAB_LIST_ID = 'open-tabs';

function revealSelectedTab(viewport: HTMLDivElement) {
  const selected = viewport.querySelector('.tab.active, [data-active-island]');
  if (!selected) return;

  const tab = selected.getBoundingClientRect();
  const bounds = viewport.getBoundingClientRect();
  if (tab.left < bounds.left + REVEAL_PADDING) {
    viewport.scrollLeft += tab.left - bounds.left - REVEAL_PADDING;
  } else if (tab.right > bounds.right - REVEAL_PADDING) {
    viewport.scrollLeft += tab.right - bounds.right + REVEAL_PADDING;
  }
}

export function TabStrip({
  activeTabId,
  activeCollapsedIslandId,
  ungroupDropActive,
  unpinnedTabCount,
  pinnedTabCount,
  islandCount,
  children,
}: {
  activeTabId: string;
  activeCollapsedIslandId?: string;
  ungroupDropActive: boolean;
  unpinnedTabCount: number;
  pinnedTabCount: number;
  islandCount: number;
  children: ReactNode;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  const [scroll, setScroll] = useState({ overflowing: false, left: false, right: false });

  useLayoutEffect(() => {
    const frame = frameRef.current!;
    const viewport = viewportRef.current!;
    const strip = stripRef.current!;
    let viewportWidth = viewport.clientWidth;

    const updateScroll = () => {
      // Measure against the frame so showing the buttons can't cause overflow by itself.
      const overflowing = strip.offsetWidth > frame.clientWidth + OVERFLOW_TOLERANCE;
      const left = overflowing && viewport.scrollLeft > OVERFLOW_TOLERANCE;
      const right =
        overflowing &&
        viewport.scrollWidth - viewport.clientWidth - viewport.scrollLeft > OVERFLOW_TOLERANCE;
      setScroll((current) =>
        current.overflowing === overflowing && current.left === left && current.right === right
          ? current
          : { overflowing, left, right },
      );
    };
    const resize = new ResizeObserver(() => {
      if (viewportWidth !== viewport.clientWidth) {
        viewportWidth = viewport.clientWidth;
        revealSelectedTab(viewport);
      }
      updateScroll();
    });
    const wheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey || event.deltaX || !event.deltaY) return;
      let delta = event.deltaY;
      if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) delta *= 16;
      if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) delta *= viewport.clientWidth;
      const before = viewport.scrollLeft;
      viewport.scrollLeft += delta;
      if (viewport.scrollLeft !== before) event.preventDefault();
    };

    resize.observe(frame);
    resize.observe(viewport);
    resize.observe(strip);
    viewport.addEventListener('scroll', updateScroll, { passive: true });
    viewport.addEventListener('wheel', wheel, { passive: false });
    updateScroll();
    return () => {
      resize.disconnect();
      viewport.removeEventListener('scroll', updateScroll);
      viewport.removeEventListener('wheel', wheel);
    };
  }, []);

  useLayoutEffect(() => {
    revealSelectedTab(viewportRef.current!);
  }, [activeTabId, activeCollapsedIslandId]);

  const scrollTabs = (direction: -1 | 1) => {
    const viewport = viewportRef.current!;
    viewport.scrollBy({
      left: direction * viewport.clientWidth * 0.8,
      behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    });
  };

  return (
    <div
      className="tab-scroll-frame"
      ref={frameRef}
      style={
        {
          '--unpinned-tab-count': unpinnedTabCount,
          '--pinned-tab-count': pinnedTabCount,
          '--tab-island-count': islandCount,
        } as CSSProperties
      }
    >
      <div
        className="tab-scroll-viewport"
        ref={viewportRef}
        data-scroll-left={scroll.left || undefined}
        data-scroll-right={scroll.right || undefined}
      >
        <div
          className={`tab-strip ${ungroupDropActive ? 'ungroup-drop-active' : ''}`}
          id={TAB_LIST_ID}
          ref={stripRef}
          role="tablist"
          aria-label="Open tabs"
        >
          {children}
        </div>
      </div>
      {scroll.overflowing && (
        <div className="tab-scroll-buttons">
          <IconButton
            className="tab-scroll-button tab-scroll-left"
            icon="chevron-right"
            label="Scroll tabs left"
            aria-controls={TAB_LIST_ID}
            disabled={!scroll.left}
            onClick={() => scrollTabs(-1)}
          />
          <IconButton
            className="tab-scroll-button"
            icon="chevron-right"
            label="Scroll tabs right"
            aria-controls={TAB_LIST_ID}
            disabled={!scroll.right}
            onClick={() => scrollTabs(1)}
          />
        </div>
      )}
    </div>
  );
}
