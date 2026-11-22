// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { EngineVersion } from '../../../../shared/engine-versions';
import { EngineVersions } from '../EngineVersions';

const torch: EngineVersion = {
  id: 'torch',
  name: 'PyTorch',
  group: 'AI engines',
  currentVersion: '2.14.1',
  versionOrigin: 'bundled',
  detail: '',
  source: 'PyPI',
  checkable: true,
  status: 'unchecked',
};
const get = vi.fn(async () => [
  torch,
  { ...torch, id: 'mlx', name: 'MLX', currentVersion: '0.32.4' },
]);
const check = vi.fn<(id: string) => Promise<EngineVersion>>();
beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(window, 'browser', {
    configurable: true,
    value: { getEngineVersions: get, checkEngineVersion: check },
  });
});
afterEach(cleanup);

it('loads local versions without checking upstream and updates only the requested row', async () => {
  let finish!: (entry: EngineVersion) => void;
  check.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  render(<EngineVersions />);
  const button = await screen.findByRole('button', { name: 'Check version for PyTorch' });
  expect(check).not.toHaveBeenCalled();
  fireEvent.click(button);
  expect(check).toHaveBeenCalledExactlyOnceWith('torch');
  expect((button as HTMLButtonElement).disabled).toBe(true);
  expect(
    (screen.getByRole('button', { name: 'Check version for MLX' }) as HTMLButtonElement).disabled,
  ).toBe(false);
  await act(async () =>
    finish({
      ...torch,
      latestVersion: '2.15.0',
      status: 'update',
      checkedAt: '2026-10-30T12:00:00Z',
    }),
  );
  expect(screen.getByText('2.15.0')).toBeTruthy();
  expect(screen.getByText('Update available')).toBeTruthy();
  expect(
    within(screen.getByRole('article', { name: 'MLX' })).getByText('Not checked'),
  ).toBeTruthy();
  expect((button as HTMLButtonElement).disabled).toBe(false);
});

it('allows a failed check to be retried without losing the bundled version', async () => {
  check
    .mockRejectedValueOnce(new Error('offline'))
    .mockResolvedValueOnce({ ...torch, latestVersion: '2.14.1', status: 'current' });
  render(<EngineVersions />);
  const button = await screen.findByRole('button', { name: 'Check version for PyTorch' });
  fireEvent.click(button);
  expect(await screen.findByText('Check failed')).toBeTruthy();
  expect(screen.getByText('2.14.1')).toBeTruthy();
  fireEvent.click(button);
  expect(await screen.findByText('Up to date')).toBeTruthy();
});

it('offers retry when local metadata cannot be loaded', async () => {
  get.mockRejectedValueOnce(new Error('no manifest'));
  render(<EngineVersions />);
  fireEvent.click(await screen.findByRole('button', { name: 'Retry loading versions' }));
  expect(await screen.findByRole('article', { name: 'PyTorch' })).toBeTruthy();
});
