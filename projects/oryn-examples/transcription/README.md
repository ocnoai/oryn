# Local audio transcription

Choose the recording language in Oryn Browser, select one of its available transcription models, then choose an MP3 or WAV recording and select **Transcribe locally**. The demo accepts recordings up to 4 hours and 1 GiB, including a 1 hour 27 minute MP3. Keep the page open during processing. Text appears as sections finish; **Download text** saves the completed transcript, including partial text after cancellation or a later error.

The language picker includes 100 language tokens plus automatic detection. Whisper Small (`openai/whisper-small`) remains the default for most languages. Other choices are Whisper Tiny, Whisper Large v3 Turbo, Whisper Large v3, English-only Whisper Small, and Qwen3 ASR 0.6B / 1.7B. Choices are filtered by their published language coverage: Slovak shows multilingual Whisper, Czech also shows Qwen, and Cantonese omits older Whisper checkpoints. Language tokens describe model coverage, not an accuracy guarantee. Coverage comes from the [official Whisper tokenizer](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) and [Qwen model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf).

Changing the language or model does not start a download. On submission, the site requests the selected model repository with automatic substitution disabled and binds the chosen language when runtime task profiles are available. It checks the resolved repository before submitting any transcription jobs; an unexpected browser preference override is reported instead of silently using a different model. The selected language is passed into every audio section. Whisper receives language codes and Qwen receives its processor's language names. English-only Whisper receives `auto`, since the current runner must omit forced language/task generation arguments for that checkpoint. Automatic detection offers multilingual Whisper models.

Audio stays on the computer. Mediabunny reads the selected file in bounded slices and decodes it progressively using browser media APIs. Each section becomes mono, 16 kHz, 16-bit PCM WAV before entering Oryn's private artifact store. The original MP3 is never submitted to a WAV-only runner.

Transcription uses one model session and sequential sections of at most 30 seconds, with 2 seconds of overlap to retain speech at boundaries. Matching phrases are removed when joining neighboring results. An 87-minute recording takes 187 sections; each WAV is at most 960,044 bytes and is released after its job. Lower duration limits advertised by the selected model reduce the section size. There is no overall transcription timeout.

```bash
pnpm --dir projects/oryn-examples install --frozen-lockfile
pnpm --dir projects/oryn-examples/transcription dev
pnpm --dir projects/oryn-examples/transcription test
```

The focused tests cover language/model filtering, request construction, language propagation through all 87 minutes, tail coverage, WAV encoding, overlapping transcript merging, cancellation, invalid input, and artifact cleanup. Browser verification also decoded a generated stereo 44.1 kHz, variable-bitrate 87-minute MP3 through all 187 sections, and checked the page workflow, selected model and language, unexpected model overrides, text download, cancellation, and cleanup with a mock transcription model. This verifies long-file handling; it does not measure speech recognition accuracy or full-recording inference time on a real model.
