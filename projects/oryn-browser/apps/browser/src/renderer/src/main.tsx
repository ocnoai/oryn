import React from 'react';
import ReactDOM from 'react-dom/client';

import { applyThemeMode } from '@ocno/oryn-theme';

import { BrowserChrome, BrowserPassiveOverlay } from './BrowserChrome';
import { BrowserSidebar, BrowserSidebarResizeHandle } from './BrowserSidebar';
import './styles.css';

applyThemeMode('system');

const root = document.getElementById('root');
if (!root) {
  throw new Error('Browser chrome root was not found.');
}

const surface = new URLSearchParams(window.location.search).get('surface');

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    {surface === 'sidebar' ? (
      <BrowserSidebar />
    ) : surface === 'sidebar-resize' ? (
      <BrowserSidebarResizeHandle />
    ) : surface === 'passive-overlay' ? (
      <BrowserPassiveOverlay />
    ) : (
      <BrowserChrome />
    )}
  </React.StrictMode>,
);
