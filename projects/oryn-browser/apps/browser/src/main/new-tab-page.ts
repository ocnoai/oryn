import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

import themeCSS from '../../../../packages/theme/src/theme.css?raw';
import { Icon, type IconName } from '../renderer/src/components/ui/Icon';
import chatPreview from './assets/new-tab/chat.jpg?inline';
import imagePreview from './assets/new-tab/image.jpg?inline';
import musicPreview from './assets/new-tab/music.jpg?inline';
import subtitlesPreview from './assets/new-tab/subtitles.jpg?inline';
import transcriptionPreview from './assets/new-tab/transcription.jpg?inline';
import pageCSS from './new-tab-page.css?raw';

interface Favorite {
  title: string;
  url: string;
  thumbnail?: string;
}

const DEMO_SITES = [
  {
    id: 'chat',
    label: 'Chat',
    title: 'Chat',
    description: 'Ask a question, draft something, or explore an idea.',
    url: 'https://chat.oryn.ocno.ai',
    icon: 'chat',
    preview: chatPreview,
  },
  {
    id: 'music',
    label: 'Music',
    title: 'Song generation',
    description: 'Describe a sound, choose a duration, and make a track.',
    url: 'https://music.oryn.ocno.ai',
    icon: 'music',
    preview: musicPreview,
  },
  {
    id: 'transcription',
    label: 'Transcribe',
    title: 'Transcription',
    description: 'Open a recording and turn spoken words into text.',
    url: 'https://transcription.oryn.ocno.ai',
    icon: 'microphone',
    preview: transcriptionPreview,
  },
  {
    id: 'image',
    label: 'Images',
    title: 'Image generation',
    description: 'Write a prompt, create an image, and save the result.',
    url: 'https://image.oryn.ocno.ai',
    icon: 'image',
    preview: imagePreview,
  },
  {
    id: 'subtitles',
    label: 'Subtitles',
    title: 'Video subtitles',
    description: 'Add a video, create captions, and make them your own.',
    url: 'https://subtitles.oryn.ocno.ai',
    icon: 'subtitles',
    preview: subtitlesPreview,
  },
] as const;

function icon(name: IconName): string {
  return renderToStaticMarkup(createElement(Icon, { name }));
}

export function newTabHTML(favorites: Favorite[], privateMode = false): string {
  const favoriteCards = favorites
    .slice(0, 12)
    .map((favorite) => {
      const title = escapeHTML(favorite.title);
      const initial = escapeHTML(
        (favorite.title || new URL(favorite.url).hostname).slice(0, 1).toUpperCase(),
      );
      const artwork = favorite.thumbnail
        ? `<img src="${escapeHTML(favorite.thumbnail)}" alt="" />`
        : initial;
      return `<a class="favorite" href="${escapeHTML(favorite.url)}"><span class="favorite-art" aria-hidden="true">${artwork}</span><strong>${title}</strong></a>`;
    })
    .join('');
  const demos = DEMO_SITES.map(
    (demo) => `<li class="app-item app-${demo.id}">
      <a class="app-link" href="${demo.url}" interestfor="app-info-${demo.id}" aria-label="Open ${demo.label}" aria-describedby="app-description-${demo.id}" style="anchor-name: --app-${demo.id}">
        <span class="app-icon" aria-hidden="true">${icon(demo.icon)}<span class="oryn-dot">O</span></span>
        <span class="app-name">${demo.label}</span>
      </a>
      <div id="app-info-${demo.id}" class="app-details" popover="hint" aria-labelledby="app-title-${demo.id}" style="position-anchor: --app-${demo.id}">
        <div class="preview-screen" aria-hidden="true"><img src="${demo.preview}" alt="" decoding="async" /></div>
        <div class="preview-content"><h3 id="app-title-${demo.id}">${demo.title}</h3><p id="app-description-${demo.id}">${demo.description}</p>
        <div class="preview-footer"><span class="preview-domain">${new URL(demo.url).hostname}</span><a href="${demo.url}" class="preview-open" aria-label="Open ${demo.title}">Open ${icon('arrow-right')}</a></div></div>
      </div>
    </li>`,
  ).join('');

  return `<!doctype html>
<html lang="en" data-theme="system">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'none'; base-uri 'none'" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${privateMode ? 'Private browsing' : 'New Tab'}</title>
    <style>${themeCSS}\n${pageCSS}</style>
  </head>
  <body${privateMode ? ' class="private-page"' : ''}>
    <main>
      <header class="start-header">
        <div class="brand"><span class="mark" aria-hidden="true">O</span><span>oryn</span>${privateMode ? '<span class="private-label">Private</span>' : ''}</div>
        <h1 class="${privateMode ? 'private-title' : 'sr-only'}">${privateMode ? 'Private browsing' : 'New tab'}</h1>
        ${privateMode ? '<p class="private-description">Browsing data is temporary. Temporary files are removed when this window closes; interrupted cleanup is retried on the next startup. Bookmarks and files you save remain. Websites, your network, employer or internet provider can still see your activity. Local AI is unavailable in private windows.</p>' : ''}
      </header>
      ${
        privateMode
          ? ''
          : `<section class="launcher-section" aria-labelledby="launcher-heading">
        <div class="section-heading"><h2 id="launcher-heading">Quick launch</h2><span class="supported-badge"><span class="badge-mark" aria-hidden="true">O</span>Oryn supported</span></div>
        <nav aria-label="Oryn apps"><ul class="app-grid">${demos}</ul></nav>
        <p class="menu-hint">Hover for details · Click to open</p>
      </section>`
      }
      ${favoriteCards ? `<section class="favorites-section" aria-labelledby="favorites-heading"><div class="section-heading"><h2 id="favorites-heading">Favorites</h2></div><nav class="favorites" aria-label="Favorites">${favoriteCards}</nav></section>` : ''}
      <footer class="page-footer"><span class="shortcut"><kbd>${process.platform === 'darwin' ? '⌘' : 'Ctrl'}</kbd><kbd>L</kbd><span>Search or enter an address</span></span></footer>
    </main>
  </body>
</html>`;
}

function escapeHTML(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
