import type { BrowserWindowController } from '../browser-window';
import type { ModelManagerService } from './model-manager-service';

/** Runs only in the isolated, explicitly requested Electron smoke profile. */
export async function runModelManagerSmoke(
  controller: BrowserWindowController,
  manager: ModelManagerService,
) {
  const ui = controller.trustedWebContents;
  const snapshot = manager.snapshot.bind(manager);
  const runtimePacks = manager.runtimePacks.bind(manager);
  const substitutes = manager.modelSubstitutes.bind(manager);
  const installed = await snapshot();
  if (!installed.models.length) throw new Error('Model Manager smoke requires a saved inventory');
  const check = async (expression: string, label: string) => {
    for (let attempt = 0; attempt < 100; attempt++) {
      if (await ui.executeJavaScript(expression)) return;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    throw new Error(`Model Manager smoke failed: ${label}`);
  };
  const text = `document.querySelector('[aria-label="Model Manager"]')?.textContent ?? ''`;
  const count = `document.querySelectorAll('.model-card').length`;
  let releaseInventory!: () => void;
  let failRuntime!: (error: Error) => void;
  let failSubstitutes!: (error: Error) => void;
  let inspectionRequests = 0;
  manager.snapshot = () =>
    new Promise((resolve) => {
      releaseInventory = () => resolve(installed);
    });
  manager.runtimePacks = () =>
    new Promise((_resolve, reject) => {
      failRuntime = reject;
    });
  manager.modelSubstitutes = () => {
    inspectionRequests++;
    return new Promise((_resolve, reject) => {
      failSubstitutes = reject;
    });
  };
  try {
    await ui.executeJavaScript(
      `document.querySelector('[aria-label="Local AI activity"]')?.click()`,
    );
    await check(`Boolean(document.querySelector('[aria-label="Oryn Local AI"]'))`, 'panel opens');
    await ui.executeJavaScript(
      `Array.from(document.querySelectorAll('[aria-label="Oryn Local AI"] button')).find(button => button.textContent === 'Models')?.click()`,
    );
    await check(
      `(${text}).includes('Loading installed models') && !(${text}).includes('No local AI models installed')`,
      'loading is distinct from empty',
    );
    releaseInventory();
    await check(
      `${count} === ${installed.models.length}`,
      'inventory visible while runtime details are pending',
    );
    if (inspectionRequests !== 0) throw new Error('Inventory unexpectedly requested substitutes');
    failRuntime(new Error('Simulated runtime failure'));
    await check(
      `(${text}).includes('Could not refresh runtime packs') && ${count} === ${installed.models.length}`,
      'runtime failure preserves models',
    );
    manager.runtimePacks = runtimePacks;
    await ui.executeJavaScript(
      `Array.from(document.querySelectorAll('[aria-label="Model Manager"] button')).find(button => button.textContent === 'Retry runtime packs')?.click()`,
    );
    await check(
      `!(${text}).includes('Could not refresh runtime packs') && !(${text}).includes('Loading runtime packs')`,
      'runtime retry succeeds',
    );
    await ui.executeJavaScript(`document.querySelector('.model-card details').open = true`);
    await check(
      `(${text}).includes('Checking installed substitutes') && ${count} === ${installed.models.length}`,
      'optional checks do not hide inventory',
    );
    if (Number(inspectionRequests) !== 1) throw new Error('Expected one lazy substitute request');
    failSubstitutes(new Error('Simulated inspection timeout'));
    await check(
      `(${text}).includes('Compatibility checks could not be completed')`,
      'compatibility error is visible',
    );
    manager.modelSubstitutes = async () => [];
    await ui.executeJavaScript(
      `Array.from(document.querySelectorAll('.model-card button')).find(button => button.textContent === 'Retry compatibility checks')?.click()`,
    );
    await check(
      `(${text}).includes('No installed substitutes for this model')`,
      'compatibility retry succeeds',
    );
    manager.snapshot = async () => {
      throw new Error('Simulated inventory failure');
    };
    await ui.executeJavaScript(`document.querySelector('[aria-label="Refresh models"]')?.click()`);
    await check(
      `(${text}).includes('Showing the last successful list') && ${count} === ${installed.models.length}`,
      'failed refresh retains inventory',
    );
    manager.snapshot = snapshot;
    await ui.executeJavaScript(`document.querySelector('[aria-label="Refresh models"]')?.click()`);
    await check(
      `!(${text}).includes('Could not refresh installed models') && !(${text}).includes('Loading installed models') && ${count} === ${installed.models.length}`,
      'inventory retry succeeds',
    );
    return {
      passed: true,
      installedModels: installed.models.length,
      automaticCompatibilityRequests: 0,
      loadingState: true,
      isolatedRuntimeFailure: true,
      isolatedCompatibilityFailure: true,
      preservedInventoryOnError: true,
      retries: true,
    };
  } finally {
    manager.snapshot = snapshot;
    manager.runtimePacks = runtimePacks;
    manager.modelSubstitutes = substitutes;
  }
}
