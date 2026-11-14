import { join, resolve } from 'node:path';

export class ProfileDataPaths {
  readonly browserCoreRoot: string;
  readonly localAIRoot: string;

  readonly history: string;
  readonly bookmarks: string;
  readonly downloads: string;
  readonly session: string;
  readonly sitePermissions: string;
  readonly appearance: string;
  readonly windowPlacement: string;
  readonly browserExperience: string;
  readonly appUpdates: string;

  readonly localAIPermissions: string;
  readonly localAIPreferences: string;
  readonly aiActivity: string;
  readonly artifactRegistry: string;
  readonly localAISettings: string;
  readonly modelWorkspaces: string;
  readonly huggingFaceCredential: string;

  constructor(readonly profileRoot: string) {
    const root = profileRoot.startsWith('memory:') ? profileRoot : resolve(profileRoot);
    this.browserCoreRoot = join(root, 'browser-core');
    this.localAIRoot = join(root, 'local-ai');
    this.history = join(this.browserCoreRoot, 'history.json');
    this.bookmarks = join(this.browserCoreRoot, 'bookmarks.json');
    this.downloads = join(this.browserCoreRoot, 'downloads.json');
    this.session = join(this.browserCoreRoot, 'session.json');
    this.sitePermissions = join(this.browserCoreRoot, 'site-permissions.json');
    this.appearance = join(this.browserCoreRoot, 'appearance.json');
    this.windowPlacement = join(this.browserCoreRoot, 'window-placement.json');
    this.browserExperience = join(this.browserCoreRoot, 'experience.json');
    this.appUpdates = join(this.browserCoreRoot, 'app-updates.json');
    this.localAIPermissions = join(this.localAIRoot, 'permissions.json');
    this.localAIPreferences = join(this.localAIRoot, 'preferences.json');
    this.aiActivity = join(this.localAIRoot, 'ai-activity.json');
    this.artifactRegistry = join(this.localAIRoot, 'artifacts.json');
    this.localAISettings = join(this.localAIRoot, 'settings.json');
    this.modelWorkspaces = join(this.localAIRoot, 'model-workspaces.json');
    this.huggingFaceCredential = join(this.localAIRoot, 'credentials', 'hugging-face.json');
  }
}
