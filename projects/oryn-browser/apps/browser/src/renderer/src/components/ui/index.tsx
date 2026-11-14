import { useRef } from 'react';

import type { ButtonHTMLAttributes, CSSProperties, HTMLAttributes, ReactNode } from 'react';

import { Icon, type IconName } from './Icon';

export { Icon, type IconName } from './Icon';

function classes(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(' ');
}

export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md';
  icon?: IconName;
}) {
  return (
    <button
      className={classes('ui-button', `ui-button-${variant}`, `ui-button-${size}`, className)}
      {...props}
    >
      {icon && <Icon name={icon} />}
      {children}
    </button>
  );
}

export function IconButton({
  icon,
  label,
  variant = 'ghost',
  className,
  ...props
}: Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
  icon: IconName;
  label: string;
  variant?: 'ghost' | 'surface';
}) {
  return (
    <button
      className={classes('ui-icon-button', `ui-icon-button-${variant}`, className)}
      aria-label={label}
      title={props.title ?? label}
      {...props}
    >
      <Icon name={icon} />
    </button>
  );
}

export function Badge({
  tone = 'neutral',
  children,
}: {
  tone?: 'neutral' | 'brand' | 'success' | 'warning' | 'danger';
  children: ReactNode;
}) {
  return <span className={`ui-badge ui-badge-${tone}`}>{children}</span>;
}

export function Card({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return <article className={classes('ui-card', className)} {...props} />;
}

export function EmptyState({
  icon = 'sparkles',
  title,
  description,
  action,
}: {
  icon?: IconName;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="ui-empty-state">
      <span className="ui-empty-state-icon">
        <Icon name={icon} />
      </span>
      <strong>{title}</strong>
      {description && <p>{description}</p>}
      {action}
    </div>
  );
}

export function SegmentedTabs<T extends string>({
  items,
  value,
  onChange,
  label,
}: {
  items: ReadonlyArray<{ value: T; label: string; icon?: IconName; count?: number }>;
  value: T;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <nav className="ui-segmented-tabs" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.value}
          aria-pressed={value === item.value}
          onClick={() => onChange(item.value)}
        >
          {item.icon && <Icon name={item.icon} />}
          <span>{item.label}</span>
          {item.count !== undefined && item.count > 0 && <Badge tone="brand">{item.count}</Badge>}
        </button>
      ))}
    </nav>
  );
}

export function MenuItem({
  icon,
  shortcut,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  icon: IconName;
  shortcut?: ReactNode;
}) {
  return (
    <button className="ui-menu-item" role="menuitem" {...props}>
      <Icon name={icon} />
      <span>{children}</span>
      {shortcut && <kbd>{shortcut}</kbd>}
    </button>
  );
}

export function Sheet({
  title,
  eyebrow = 'Oryn Browser',
  icon,
  wide = false,
  close,
  children,
  ariaLabel,
}: {
  title: string;
  eyebrow?: string;
  icon: IconName;
  wide?: boolean;
  close: () => void;
  children: ReactNode;
  ariaLabel?: string;
}) {
  const backdropPointer = useRef<number | null>(null);
  return (
    <div
      className="ui-sheet-layer"
      data-trusted-overlay="true"
      role="presentation"
      onPointerDown={(event) => {
        backdropPointer.current = event.target === event.currentTarget ? event.pointerId : null;
      }}
      onPointerCancel={() => {
        backdropPointer.current = null;
      }}
      onPointerUp={(event) => {
        const startedOnBackdrop = backdropPointer.current === event.pointerId;
        backdropPointer.current = null;
        if (!startedOnBackdrop || event.target !== event.currentTarget) return;
        const button = event.button === 1 ? 'middle' : event.button === 2 ? 'right' : 'left';
        void window.browser
          .dismissPanelAndClickPage(event.clientX, event.clientY, button)
          .then(close);
      }}
    >
      <section
        className={classes('ui-sheet', wide && 'ui-sheet-wide')}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel ?? title}
      >
        <header className="ui-sheet-header">
          <span className="ui-sheet-mark">
            <Icon name={icon} />
          </span>
          <div>
            <span className="ui-eyebrow">{eyebrow}</span>
            <h2>{title}</h2>
          </div>
          <IconButton icon="close" label={`Close ${title}`} onClick={close} />
        </header>
        <div className="ui-sheet-body">{children}</div>
      </section>
    </div>
  );
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="ui-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function ProgressBar({ label, value, max }: { label: string; value: number; max: number }) {
  const boundedMax = Math.max(1, max);
  const boundedValue = Math.min(boundedMax, Math.max(0, value));
  const percentage = (boundedValue / boundedMax) * 100;

  return (
    <div
      className="ui-progress"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={boundedMax}
      aria-valuenow={boundedValue}
    >
      <span style={{ '--ui-progress-value': `${percentage}%` } as CSSProperties} />
    </div>
  );
}
