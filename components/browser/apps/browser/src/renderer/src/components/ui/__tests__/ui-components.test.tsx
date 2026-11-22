import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { Badge, Button, EmptyState, Icon, ProgressBar, SegmentedTabs, Sheet } from '..';

describe('trusted UI component library', () => {
  it('renders vendor icons through the semantic Oryn adapter', () => {
    const markup = renderToStaticMarkup(<Icon name="search" className="test-icon" />);

    expect(markup).toContain('ui-icon test-icon');
    expect(markup).toContain('lucide-search');
    expect(markup).toContain('aria-hidden="true"');
    expect(markup).toContain('focusable="false"');
  });

  it('applies shared button variants, sizes, and icon semantics', () => {
    const markup = renderToStaticMarkup(
      <Button variant="danger" size="sm" icon="trash">
        Delete model
      </Button>,
    );

    expect(markup).toContain('ui-button-danger');
    expect(markup).toContain('ui-button-sm');
    expect(markup).toContain('Delete model');
    expect(markup).toContain('aria-hidden="true"');
  });

  it('renders a consistent accessible sheet shell', () => {
    const markup = renderToStaticMarkup(
      <Sheet title="Local AI" icon="sparkles" close={() => undefined}>
        <p>Content</p>
      </Sheet>,
    );

    expect(markup).toContain('ui-sheet-layer');
    expect(markup).toContain('role="dialog"');
    expect(markup).toContain('aria-modal="true"');
    expect(markup).toContain('aria-label="Close Local AI"');
  });

  it('shares badges, empty states, and segmented navigation', () => {
    const markup = renderToStaticMarkup(
      <>
        <Badge tone="success">Compatible</Badge>
        <EmptyState title="Nothing here" description="Run a model to get started." />
        <SegmentedTabs
          label="Views"
          value="activity"
          onChange={() => undefined}
          items={[
            { value: 'activity', label: 'Activity', icon: 'activity' },
            { value: 'models', label: 'Models', icon: 'models', count: 2 },
          ]}
        />
      </>,
    );

    expect(markup).toContain('ui-badge-success');
    expect(markup).toContain('ui-empty-state');
    expect(markup).toContain('ui-segmented-tabs');
    expect(markup).toContain('aria-pressed="true"');
  });

  it('renders smooth progress with bounded accessible values', () => {
    const markup = renderToStaticMarkup(
      <ProgressBar label="Download progress for org/model" value={125} max={100} />,
    );

    expect(markup).toContain('class="ui-progress"');
    expect(markup).toContain('role="progressbar"');
    expect(markup).toContain('aria-valuenow="100"');
    expect(markup).toContain('--ui-progress-value:100%');
  });
});
