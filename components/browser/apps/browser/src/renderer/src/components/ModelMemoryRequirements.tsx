import { formatStorageSize as formatBytes } from '../../../shared/format-storage-size';

export function ModelMemoryRequirements({
  requiredBytes,
  availableBytes,
}: {
  requiredBytes?: number;
  availableBytes?: number;
}) {
  const required = validBytes(requiredBytes);
  const available = validBytes(availableBytes);
  const known = required !== undefined && required > 0 && available !== undefined;
  const shortfall = known ? Math.max(0, required - available) : undefined;
  return (
    <section
      className="hf-download-details hf-memory-details"
      aria-label="Model memory requirements"
    >
      <strong>Memory</strong>
      <p className="hf-memory-note">
        System model memory, shared with the accelerator on unified-memory devices. Availability is
        a live estimate, not total capacity.
      </p>
      <dl>
        <div>
          <dt>Model needs (estimated)</dt>
          <dd>{required === undefined ? 'Unavailable' : `~${formatBytes(required)}`}</dd>
        </div>
        <div>
          <dt>Available to Oryn</dt>
          <dd>{available === undefined ? 'Unavailable' : formatBytes(available)}</dd>
        </div>
      </dl>
      {known && (
        <div
          className="hf-memory-bar"
          data-sufficient={shortfall === 0}
          role="meter"
          aria-label="Available memory compared with model requirement"
          aria-valuemin={0}
          aria-valuemax={required}
          aria-valuenow={Math.min(available, required)}
          aria-valuetext={`${formatBytes(available)} available; approximately ${formatBytes(required)} needed`}
        >
          <span style={{ width: `${Math.min(100, (available / required) * 100)}%` }} />
        </div>
      )}
      <p className="hf-memory-note">
        {shortfall === undefined
          ? 'Memory availability could not be read. Check again to refresh.'
          : shortfall > 0
            ? `${formatBytes(shortfall)} more needed. Available memory reflects your Oryn memory budget.`
            : 'Enough memory is available now. Check again to retry model inspection.'}
      </p>
    </section>
  );
}

function validBytes(value: number | undefined): number | undefined {
  return value !== undefined && Number.isFinite(value) && value >= 0 ? value : undefined;
}
