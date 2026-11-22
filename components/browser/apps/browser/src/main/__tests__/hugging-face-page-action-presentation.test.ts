import { describe, expect, it } from 'vitest';

import type { HuggingFacePageActionState } from '../../shared/ipc';
import {
  downloadPercentage,
  formatModelBytes,
  formatResourceMemory,
  huggingFacePageActionLabel,
} from '../../renderer/src/BrowserChrome';

function downloading(
  downloadedBytes: number | undefined,
  totalBytes: number | undefined,
): HuggingFacePageActionState {
  return {
    status: 'downloading',
    repository: 'Qwen/Qwen3.5-4B',
    sourceURL: 'https://huggingface.co/Qwen/Qwen3.5-4B',
    selectedFiles: ['model.safetensors'],
    downloadedBytes,
    totalBytes,
  };
}

describe('Hugging Face page-action download presentation', () => {
  it('shows bounded percentage progress in compact browser chrome', () => {
    expect(huggingFacePageActionLabel(downloading(375, 1_000))).toBe('Downloading · 37%');
    expect(huggingFacePageActionLabel(downloading(1_500, 1_000))).toBe('Downloading · 100%');
    expect(downloadPercentage(-10, 1_000)).toBe(0);
  });

  it('reports received bytes while total size is still being resolved', () => {
    expect(huggingFacePageActionLabel(downloading(2_500_000, undefined))).toBe(
      'Downloading · 2.5 MB',
    );
    expect(huggingFacePageActionLabel(downloading(0, undefined))).toBe('Starting download…');
  });
});

describe('Local AI resource memory presentation', () => {
  it('uses the shared decimal storage-size presentation', () => {
    expect(formatResourceMemory(0)).toBe('0 B');
    expect(formatResourceMemory(85 * 1024 ** 2)).toBe('89.1 MB');
    expect(formatResourceMemory(250_000_000)).toBe('250 MB');
    expect(formatResourceMemory(3_705_300_000)).toBe('3.7 GB');
    expect(formatResourceMemory(64 * 1024 ** 3)).toBe('68.7 GB');
  });

  it('uses the same MB and GB presentation in the Models tab', () => {
    expect(formatModelBytes(250_000_000)).toBe('250 MB');
    expect(formatModelBytes(3_705_300_000)).toBe('3.7 GB');
  });
});
