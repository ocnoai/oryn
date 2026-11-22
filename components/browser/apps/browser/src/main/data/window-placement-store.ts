import type { Rectangle } from 'electron';

import { JsonFileStore, isRecord } from './json-file-store';

export const DEFAULT_WINDOW_SIZE = { width: 1280, height: 820 } as const;
export const MINIMUM_WINDOW_SIZE = { width: 720, height: 520 } as const;

export interface WindowPlacementData {
  version: 1;
  bounds?: Rectangle;
  maximized: boolean;
}

export class WindowPlacementStore {
  private readonly store: JsonFileStore<WindowPlacementData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore<WindowPlacementData>(
      filePath,
      () => ({ version: 1, maximized: false }),
      isWindowPlacementData,
    );
  }

  read(): WindowPlacementData {
    return this.store.read();
  }

  write(bounds: Rectangle, maximized: boolean): void {
    this.store.write({ version: 1, bounds: roundRectangle(bounds), maximized });
  }
}

export function fitWindowBoundsToWorkArea(bounds: Rectangle, workArea: Rectangle): Rectangle {
  const width = Math.min(
    Math.max(Math.round(bounds.width), MINIMUM_WINDOW_SIZE.width),
    workArea.width,
  );
  const height = Math.min(
    Math.max(Math.round(bounds.height), MINIMUM_WINDOW_SIZE.height),
    workArea.height,
  );
  const maximumX = workArea.x + workArea.width - width;
  const maximumY = workArea.y + workArea.height - height;

  return {
    x: Math.min(Math.max(Math.round(bounds.x), workArea.x), maximumX),
    y: Math.min(Math.max(Math.round(bounds.y), workArea.y), maximumY),
    width,
    height,
  };
}

function isWindowPlacementData(value: unknown): value is WindowPlacementData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    (value.bounds === undefined ||
      (isRectangle(value.bounds) &&
        value.bounds.width >= MINIMUM_WINDOW_SIZE.width &&
        value.bounds.height >= MINIMUM_WINDOW_SIZE.height)) &&
    typeof value.maximized === 'boolean'
  );
}

function isRectangle(value: unknown): value is Rectangle {
  return (
    isRecord(value) &&
    isFiniteNumber(value.x) &&
    isFiniteNumber(value.y) &&
    isFiniteNumber(value.width) &&
    isFiniteNumber(value.height)
  );
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function roundRectangle(bounds: Rectangle): Rectangle {
  return {
    x: Math.round(bounds.x),
    y: Math.round(bounds.y),
    width: Math.round(bounds.width),
    height: Math.round(bounds.height),
  };
}
