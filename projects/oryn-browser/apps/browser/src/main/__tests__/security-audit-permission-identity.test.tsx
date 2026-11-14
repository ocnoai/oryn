// @vitest-environment jsdom
import { IPC, TAB_SEARCH_SHORTCUT } from '../../shared/ipc';
import type {
  BrowserWindowState,
  LocalAIPermissionPrompt,
  LocalAIPermissionPromptResponse,
} from '../../shared/ipc';
import React from 'react';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { fixture, owner, exact, request, profile } from './security-audit-model-fixture';
import { installFixture, page, FakeSession } from './security-audit-ipc-fixture';
import { EventEmitter } from 'node:events';
import { BrowserChrome } from '../../renderer/src/BrowserChrome';

it.each(['distinct', 'same-exact', 'single', 'file-set', 'choose-other-exact', 'invalid-choice'])(
  'permission display and granted artifact agree: %s',
  async (mode) => {
    const a = {
      ...exact('validation/model', 'a'.repeat(40)),
      ...(mode === 'file-set' ? { files: ['first.safetensors'] } : {}),
    };
    const b =
      mode === 'same-exact'
        ? a
        : mode === 'file-set'
          ? { ...a, files: ['second.safetensors'] }
          : exact('validation/model', 'b'.repeat(40));
    const f = fixture(mode === 'single' ? [] : [a]);
    f.engine.inspectModel = vi.fn(async (model) => ({
      ...profile(model),
      estimatedDiskBytes: model.revision === a.revision ? 64 : 128,
    }));
    const ipc = installFixture({}, new EventEmitter(), f.prompts, [
      page(owner.webContentsId, new FakeSession(), owner.frameUrl),
    ]);
    const subscriptions = new Map<string, (prompt: LocalAIPermissionPrompt) => void>();
    let prompt!: LocalAIPermissionPrompt;
    let response!: LocalAIPermissionPromptResponse;
    const never = new Promise(() => {});
    const normalState: BrowserWindowState = {
      privateMode: false,
      activeTabId: owner.tabId,
      tabs: [
        {
          tabId: owner.tabId,
          webContentsId: owner.webContentsId,
          title: 'Permission fixture',
          url: owner.frameUrl,
          canGoBack: false,
          canGoForward: false,
          isLoading: false,
          workspaceId: 'workspace_default',
          pinned: false,
        },
      ],
      engineState: 'ready',
      zoomFactor: 1,
      memoryPressure: 'normal',
      experience: {
        activeWorkspaceId: 'workspace_default',
        workspaces: [
          { workspaceId: 'workspace_default', name: 'Main', accent: 'indigo', order: 0 },
        ],
        islands: [],
        recentlyClosed: [],
        sidebar: {
          visible: true,
          width: 52,
          panelWidth: 390,
          panelMode: 'overlay',
          activeAppId: null,
          tools: [],
          apps: [],
        },
        tabSearchShortcut: TAB_SEARCH_SHORTCUT,
        autoGroupOpenerTabs: true,
      },
    };
    const browser = new Proxy(
      {},
      {
        get(_target, key) {
          if (key === 'getState') return () => Promise.resolve(normalState);
          if (key === 'respondToLocalAIPermission')
            return async (value: LocalAIPermissionPromptResponse) => {
              response = value;
              return ipc.invoke(IPC.localAIPermissionRespond, value);
            };
          if (typeof key === 'string' && key.startsWith('on'))
            return (f: (prompt: LocalAIPermissionPrompt) => void) => {
              subscriptions.set(key, f);
              return () => subscriptions.delete(key);
            };
          if (typeof key === 'string' && key.startsWith('get')) return () => never;
          return () => Promise.resolve();
        },
      },
    );
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
    Object.defineProperty(window, 'browser', { configurable: true, value: browser });
    ipc.trusted.send.mockImplementation((channel: string, value: LocalAIPermissionPrompt) => {
      if (channel === IPC.localAIPermissionPrompt) {
        if (mode === 'invalid-choice') value = { ...value, candidates: [] };
        prompt = value;
        subscriptions.get('onLocalAIPermissionPrompt')?.(value);
      }
    });
    try {
      // Resolve the real normal-window state before delivering a permission
      // event; unknown/private contexts deliberately fail closed in production.
      await act(async () => {
        render(<BrowserChrome />);
      });
      let pending!: ReturnType<typeof f.sessions.requestSession>;
      await act(async () => {
        pending = f.sessions.requestSession(
          { ...request(b), ...(mode === 'single' ? {} : { certifiedModels: [a] }) },
          owner,
        );
        // Hard bound; no background waiting or Engine work is created by the fixture.
        for (let i = 0; i < 30 && !prompt; i++) await Promise.resolve();
      });
      expect(prompt).toBeDefined();
      expect(prompt.exactRevision).toBe(b.revision);
      const dialog = screen.getByRole('dialog', { name: 'Local AI permission' });
      const displayedRevision = dialog.querySelector('.revision-value')?.textContent;
      const displayText = dialog.textContent ?? '';
      if (mode === 'invalid-choice') {
        const allow = within(dialog).getByRole('button', {
          name: 'Allow once',
        }) as HTMLButtonElement;
        expect(allow.disabled).toBe(true);
        fireEvent.click(allow);
        expect(response).toBeUndefined();
        const denied = pending.then(
          () => false,
          (error: { code: string }) => error.code === 'USER_DENIED',
        );
        ipc.invoke(IPC.localAIPermissionRespond, { promptId: prompt.promptId, action: 'deny' });
        expect(await denied).toBe(true);
        expect(f.engine.installModel).not.toHaveBeenCalled();
        return;
      }
      if (['distinct', 'file-set', 'choose-other-exact'].includes(mode)) {
        expect(prompt.candidates).toHaveLength(2);
        expect(prompt.candidates[0]!.model).toEqual(a);
        expect(displayedRevision).toBe(b.revision);
        expect(displayText).not.toContain('Already installed');
        expect(prompt.installed).toBe(false);
      } else {
        expect(prompt.candidates).toHaveLength(1);
        expect(displayedRevision).toBe(b.revision);
      }
      if (mode === 'choose-other-exact' || mode === 'file-set') {
        await act(async () => {
          fireEvent.click(within(dialog).getByRole('button', { name: 'Choose model' }));
        });
        const choices = within(dialog).getAllByRole('radio') as HTMLInputElement[];
        expect(choices).toHaveLength(2);
        expect(choices.filter((c) => c.checked)).toHaveLength(1);
        expect(choices[0]!.value).not.toBe(choices[1]!.value);
        if (mode === 'choose-other-exact') {
          await act(async () => {
            fireEvent.click(choices[0]!);
          });
          expect(dialog.querySelector('.revision-value')?.textContent).toBe(a.revision);
        }
      }
      await act(async () => {
        fireEvent.click(
          within(dialog).getByRole('button', {
            name: /^(Allow once|Trust and allow once|Use anyway once)$/,
          }),
        );
        await pending;
      });
      expect(response.action).toBe('allow-once');
      if (mode === 'choose-other-exact') expect(response.selectedModel).toEqual(a);
      else expect(response).not.toHaveProperty('selectedModel');
      expect(f.engine.attachModelSession.mock.calls[0]![1]).toEqual(
        mode === 'choose-other-exact' ? a : b,
      );
      if (!['same-exact', 'choose-other-exact'].includes(mode))
        expect(f.engine.installModel.mock.calls[0]![1]).toEqual(b);
      else expect(f.engine.installModel).not.toHaveBeenCalled();
      console.log(
        JSON.stringify({
          id: 'NV-16',
          mode,
          scope:
            'actual Browser request/resolver/prompt/renderer/trusted IPC/session, simulated Engine profiles',
          promptRevision: prompt.exactRevision,
          displayedRevision,
          installedTarget: f.engine.installModel.mock.calls[0]?.[1],
          trustDifferenceClaimed: false,
        }),
      );
    } finally {
      cleanup();
      ipc.cleanup();
      f.cleanup();
      vi.unstubAllGlobals();
    }
  },
);
