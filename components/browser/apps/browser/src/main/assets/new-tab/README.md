These JPEGs are screenshots of the public demo apps, captured on 6 November 2026:

- `chat.jpg`: https://chat.oryn.ocno.ai — message form
- `music.jpg`: https://music.oryn.ocno.ai — track prompt and duration controls
- `transcription.jpg`: https://transcription.oryn.ocno.ai — recording upload form
- `image.jpg`: https://image.oryn.ocno.ai — prompt controls and image canvas
- `subtitles.jpg`: https://subtitles.oryn.ocno.ai — video upload and caption controls

Captured with Chromium at a 1280 × 1400 viewport, in the apps' default idle states,
cropped to the primary controls. The browser-required dialog was dismissed. No
generation was run. Refresh these images when the corresponding app UI changes.

`new-tab-page.ts` imports them with Vite's `?inline` asset query. They are bundled
as data URLs so the new-tab page can show previews offline without contacting
the demo sites or changing its image content security policy.
