import { defineConfig } from "vite";
import { createHash } from "node:crypto";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

// Precache application assets only. Document data never passes through the worker.
export default defineConfig({
  base: "./",
  build: { target: "es2022", sourcemap: false },
  worker: { format: "es" },
  plugins: [
    {
      name: "editor-offline-assets",
      async closeBundle() {
        const root = fileURLToPath(new URL("./dist/", import.meta.url));
        const paths: string[] = [];
        const hash = createHash("sha256");
        async function visit(directory: string) {
          for (const file of await readdir(directory, {
            withFileTypes: true,
          })) {
            const path = join(directory, file.name);
            if (file.isDirectory()) await visit(path);
            else if (!["sw.js", "_headers"].includes(file.name)) {
              const name = relative(root, path).replaceAll("\\", "/");
              paths.push(`./${name}`);
              hash.update(name);
              hash.update(await readFile(path));
            }
          }
        }
        await visit(root);
        const cache = `oryn-editor-${hash.digest("hex").slice(0, 16)}`;
        await writeFile(
          join(root, "sw.js"),
          `const CACHE=${JSON.stringify(cache)};\nconst ASSETS=${JSON.stringify(["./", ...paths])};\nself.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS.map(url=>new Request(url,{cache:'reload'}))))));\nself.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key.startsWith('oryn-editor-')&&key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));\nself.addEventListener('fetch',event=>{if(event.request.method!=='GET')return;const url=new URL(event.request.url);if(url.origin!==self.location.origin)return;const allowed=ASSETS.some(path=>new URL(path,self.registration.scope).href===url.href);if(!allowed)return;event.respondWith(caches.open(CACHE).then(async cache=>(await cache.match(event.request,{ignoreVary:true}))||fetch(event.request)));});\n`,
        );
      },
    },
  ],
});
