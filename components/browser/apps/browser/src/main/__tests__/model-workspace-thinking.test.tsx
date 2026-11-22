// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ModelWorkspace } from '../../renderer/src/model-workspace';

afterEach(cleanup);

it('sends an Off preference, keeps reasoning separate and labels an incomplete final answer', async () => {
  let listener: (event: unknown) => void = () => undefined;
  const invoke = vi.fn(async (_input: unknown) => ({
    ok: true,
    value: { jobId: 'job_test', state: 'running' },
  }));
  Object.defineProperty(window, 'orynWorkspace', {
    configurable: true,
    value: {
      getState: async () => ({
        status: 'ready',
        repository: 'test/local',
        taskContract: 'chat-v2',
      }),
      onJobEvent: (callback: typeof listener) => {
        listener = callback;
        return () => undefined;
      },
      invoke,
      releaseArtifact: vi.fn(),
      openSource: vi.fn(),
    },
  });
  const { container } = render(<ModelWorkspace />);
  await screen.findByRole('heading', { name: 'test/local' });
  fireEvent.change(screen.getByRole('combobox', { name: /Thinking/ }), {
    target: { value: 'off' },
  });
  fireEvent.change(screen.getByPlaceholderText('Ask this model anything…'), {
    target: { value: 'Write an article.' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
  await waitFor(() =>
    expect(invoke).toHaveBeenCalledWith(expect.objectContaining({ thinking: false })),
  );
  await act(async () => undefined);
  act(() =>
    listener({
      event: 'job.output.delta',
      jobId: 'job_test',
      data: { channel: 'reasoning', delta: 'Reasoning only.' },
    }),
  );
  expect(container.querySelector('details')?.textContent).toContain('Reasoning only.');
  expect(container.querySelector('details')?.open).toBe(false);
  const result = {
    kind: 'structured',
    value: {
      text: 'Partial answer.',
      reasoning: 'Reasoning only.',
      finishReason: 'length',
      metadata: { thinking: { requested: false, applied: false, effective: true } },
    },
  };
  act(() => listener({ event: 'job.completed', jobId: 'job_test', data: { result } }));
  expect(screen.getByText('Partial answer.')).toBeTruthy();
  expect(screen.getByText(/Request could not be applied/)).toBeTruthy();
  expect(screen.getByText(/This answer is incomplete/)).toBeTruthy();
  expect(container.querySelectorAll('.chat-assistant')).toHaveLength(0);
  expect(container.querySelector('.text-result')?.textContent).toBe('Partial answer.');
});

it('keeps a reasoning-only stop out of assistant history and the next request', async () => {
  let listener: (event: unknown) => void = () => undefined;
  const invoke = vi.fn(async (_input: unknown) => ({
    ok: true,
    value: { jobId: 'job_empty', state: 'running' },
  }));
  Object.defineProperty(window, 'orynWorkspace', {
    configurable: true,
    value: {
      getState: async () => ({
        status: 'ready',
        repository: 'test/local',
        taskContract: 'chat-v2',
      }),
      onJobEvent: (callback: typeof listener) => {
        listener = callback;
        return () => undefined;
      },
      invoke,
      releaseArtifact: vi.fn(),
      openSource: vi.fn(),
    },
  });
  const { container } = render(<ModelWorkspace />);
  await screen.findByRole('heading', { name: 'test/local' });
  fireEvent.change(screen.getByPlaceholderText('Ask this model anything…'), {
    target: { value: 'First request.' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
  await waitFor(() => expect(invoke).toHaveBeenCalledTimes(1));
  await act(async () => undefined);
  act(() =>
    listener({
      event: 'job.completed',
      jobId: 'job_empty',
      data: {
        result: {
          kind: 'structured',
          value: { text: '  ', reasoning: 'Reasoning without an answer.', finishReason: 'stop' },
        },
      },
    }),
  );
  expect(screen.getByText('No final answer was generated.')).toBeTruthy();
  expect(container.querySelectorAll('.chat-assistant')).toHaveLength(0);
  fireEvent.change(screen.getByPlaceholderText('Ask this model anything…'), {
    target: { value: 'Second request.' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
  await waitFor(() => expect(invoke).toHaveBeenCalledTimes(2));
  expect(invoke.mock.calls[1]?.[0]).toMatchObject({
    messages: [
      { role: 'user', content: 'First request.' },
      { role: 'user', content: 'Second request.' },
    ],
  });
});
