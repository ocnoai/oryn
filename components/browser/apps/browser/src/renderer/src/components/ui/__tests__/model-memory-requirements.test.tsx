import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ModelMemoryRequirements } from '../../ModelMemoryRequirements';

describe('model memory requirements', () => {
  it('compares available memory against the estimate and shows the shortfall', () => {
    const markup = renderToStaticMarkup(
      <ModelMemoryRequirements requiredBytes={50e9} availableBytes={20e9} />,
    );
    expect(markup).toContain('~50 GB');
    expect(markup).toContain('20 GB');
    expect(markup).toContain('30 GB more needed');
    expect(markup).toContain('width:40%');
    expect(markup).toContain('role="meter"');
    expect(markup).toContain('20 GB available; approximately 50 GB needed');
  });

  it('bounds the bar while keeping actual available memory visible', () => {
    const markup = renderToStaticMarkup(
      <ModelMemoryRequirements requiredBytes={20e9} availableBytes={50e9} />,
    );
    expect(markup).toContain('width:100%');
    expect(markup).toContain('50 GB');
    expect(markup).toContain('Enough memory is available now');
    expect(markup).toContain('aria-valuenow="20000000000"');
  });

  it('distinguishes unavailable measurements from zero memory', () => {
    const unknown = renderToStaticMarkup(<ModelMemoryRequirements requiredBytes={50e9} />);
    expect(unknown).toContain('Unavailable');
    expect(unknown).not.toContain('role="meter"');
    const zero = renderToStaticMarkup(
      <ModelMemoryRequirements requiredBytes={50e9} availableBytes={0} />,
    );
    expect(zero).toContain('width:0%');
    expect(zero).toContain('50 GB more needed');
  });
});
