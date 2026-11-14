import type { SVGProps } from 'react';

import { activeIconPack } from './icon-packs/active';
import type { IconName } from './icon-types';

export type { IconName } from './icon-types';

export interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
}

/**
 * Product-level icon adapter. Callers depend on semantic Oryn names rather than
 * importing a vendor pack; changing packs only requires replacing activeIconPack.
 */
export function Icon({ name, className, ...props }: IconProps): React.JSX.Element {
  const Glyph = activeIconPack[name];
  const classes = ['ui-icon', className].filter(Boolean).join(' ');

  return (
    <Glyph className={classes} aria-hidden="true" focusable="false" strokeWidth={1.8} {...props} />
  );
}
