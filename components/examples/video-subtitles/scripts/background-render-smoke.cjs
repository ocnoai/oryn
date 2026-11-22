// Run with Electron and the Vite server. Uses the same detached WebContentsView
// tab switch as Oryn, with Chromium's normal background throttling enabled.
const {app, BrowserWindow, WebContentsView} = require('electron');
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
app.whenReady().then(async () => {
  const window = new BrowserWindow({width:900,height:650,webPreferences:{sandbox:true,contextIsolation:true,nodeIntegration:false}});
  await window.loadURL('data:text/html,<h1>Another tab is active</h1>');
  const results = [];
  for (const mode of ['source','animation','video','hidden-start','cancel','webm']) {
    const view = new WebContentsView({webPreferences:{sandbox:true,contextIsolation:true,nodeIntegration:false}});
    view.setBounds({x:0,y:0,width:900,height:650});
    window.contentView.addChildView(view);
    let attached = true;
    const hide = () => { if (attached) {window.contentView.removeChildView(view);attached=false;} };
    const show = () => { if (!attached) {window.contentView.addChildView(view);attached=true;} };
    view.webContents.on('console-message', event => {
      if (event.message === 'SMOKE_HIDE_RENDER_TAB') hide();
      if (event.message === 'SMOKE_RENDER_FINISHED') show();
    });
    const url = new URL(process.env.ORYN_BACKGROUND_SMOKE_URL || 'http://127.0.0.1:5188/tests/background-render-smoke.html');
    url.searchParams.set('mode', mode);
    await view.webContents.loadURL(url.href);
    if (mode === 'hidden-start' || mode === 'webm') hide();
    let deadline;
    try {
      const result = await Promise.race([
        view.webContents.executeJavaScript('window.runBackgroundRender()'),
        new Promise((_,reject) => {deadline=setTimeout(()=>reject(new Error(`Background render timed out: ${mode}`)),45000);}),
      ]);
      console.log(JSON.stringify(result));
      if (!result.ok) throw new Error(result.error);
      results.push(result);
    } finally {
      clearTimeout(deadline);
      if (attached) window.contentView.removeChildView(view);
      view.webContents.close();
    }
  }
  if (process.env.ORYN_BACKGROUND_SMOKE_REPORT) require('node:fs').writeFileSync(process.env.ORYN_BACKGROUND_SMOKE_REPORT, JSON.stringify({ok:true,results},null,2)+'\n');
  app.exit(0);
}).catch(error=>{console.error(error);app.exit(1);});
