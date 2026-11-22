import { createRoot } from "react-dom/client";
import { EditorStorage } from "./storage";
import { App } from "./App";
import "./style.css";

async function start() {
  try {
    const storage = await EditorStorage.open();
    createRoot(document.getElementById("root")!).render(
      <App storage={storage} />,
    );
    if (import.meta.env.PROD && "serviceWorker" in navigator) {
      void navigator.serviceWorker.register("./sw.js").catch(() => {
        /* Editing remains available without offline caching. */
      });
    }
  } catch {
    const root = document.getElementById("root")!;
    const title = document.createElement("h1");
    title.textContent = "Editor could not open local storage";
    const message = document.createElement("p");
    message.textContent =
      "Use a secure HTTPS or localhost address and allow this site to store data. Your existing data has not been cleared.";
    root.append(title, message);
  }
}
void start();
