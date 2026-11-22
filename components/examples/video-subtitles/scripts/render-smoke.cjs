// Invoke with the existing Electron executable and a running local Vite demo server.
const { app, BrowserWindow } = require('electron');
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
app.whenReady().then(async () => {
  const window = new BrowserWindow({width:1100,height:950,webPreferences:{sandbox:true,contextIsolation:true,nodeIntegration:false,backgroundThrottling:false}});
  await window.loadURL(process.env.ORYN_RENDER_SMOKE_URL || 'http://localhost:5188/tests/render-smoke.html');
  const focusSelector = process.env.ORYN_SMOKE_FOCUS || '#caption-studio';
  const focusScript = `document.querySelector(${JSON.stringify(focusSelector)})?.scrollIntoView()`;
  let dragged = false;
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    if (!dragged) {
      const drag = await window.webContents.executeJavaScript('window.smokeDrag');
      if (drag) {
        dragged = true;
        window.webContents.sendInputEvent({type:'mouseMove',x:drag.x,y:drag.y});
        window.webContents.sendInputEvent({type:'mouseDown',x:drag.x,y:drag.y,button:'left',clickCount:1});
        window.webContents.sendInputEvent({type:'mouseMove',x:drag.x+30,y:drag.y,button:'left'});
        window.webContents.sendInputEvent({type:'mouseUp',x:drag.x+30,y:drag.y,button:'left',clickCount:1});
      }
    }
    const result = await window.webContents.executeJavaScript('window.smokeResult');
    if (result) {
      if (process.env.ORYN_RENDER_SMOKE_REPORT) {
        const fs = require('node:fs');
        const path = require('node:path');
        fs.mkdirSync(process.env.ORYN_RENDER_SMOKE_REPORT, {recursive:true});
        fs.writeFileSync(path.join(process.env.ORYN_RENDER_SMOKE_REPORT, 'render-smoke.json'), JSON.stringify(result, null, 2) + '\n');
        await window.webContents.executeJavaScript(focusScript);
        await window.webContents.executeJavaScript('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
        fs.writeFileSync(path.join(process.env.ORYN_RENDER_SMOKE_REPORT, 'test-window.png'), (await window.webContents.capturePage()).toPNG());
        if (await window.webContents.executeJavaScript('Boolean(document.querySelector("#caption-studio"))')) {
          window.setSize(390, 844);
          await window.webContents.executeJavaScript(focusScript + '; new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
          const overflow = await window.webContents.executeJavaScript('document.documentElement.scrollWidth > innerWidth');
          if (overflow) {
            fs.writeFileSync(path.join(process.env.ORYN_RENDER_SMOKE_REPORT, 'mobile-window.png'), (await window.webContents.capturePage()).toPNG());
            console.error(await window.webContents.executeJavaScript("JSON.stringify([...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().right > innerWidth + 1).slice(0,20).map(e=>({tag:e.tagName,id:e.id,cls:e.className,width:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right})))"));
            app.exit(1); return;
          }
          fs.writeFileSync(path.join(process.env.ORYN_RENDER_SMOKE_REPORT, 'mobile-window.png'), (await window.webContents.capturePage()).toPNG());
        }
        const frame = await window.webContents.executeJavaScript('window.smokeFrame');
        if (frame) fs.writeFileSync(path.join(process.env.ORYN_RENDER_SMOKE_REPORT, 'rendered-frame.png'), Buffer.from(frame.split(',')[1], 'base64'));
      }
      console.log(JSON.stringify(result)); app.exit(result.ok ? 0 : 1); return;
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  console.error('Render smoke timed out'); app.exit(1);
}).catch(error => { console.error(error); app.exit(1); });
