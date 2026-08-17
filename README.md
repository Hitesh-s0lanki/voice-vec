# voice-vec

A light-theme voice interface wired to [Sarvam AI](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview) speech-to-text.

Tap the orb to record, tap again to transcribe. The halo and waveform are driven by the live
analyser, so the orb reacts to what the mic actually hears.

## Setup

```bash
cp .env.example .env.local   # then paste your key from dashboard.sarvam.ai
npm run dev
```

`SARVAM_API_KEY` is read server-side only — the browser never sees it.

## How it works

| Piece | File |
| --- | --- |
| Mic capture, analyser levels, upload | [`src/hooks/use-voice-capture.ts`](src/hooks/use-voice-capture.ts) |
| Server proxy to Sarvam | [`src/app/api/transcribe/route.ts`](src/app/api/transcribe/route.ts) |
| States and copy | [`src/components/voice-app.tsx`](src/components/voice-app.tsx) |
| Orb, waveform, theme | [`src/components/aurora-orb.tsx`](src/components/aurora-orb.tsx), [`src/app/globals.css`](src/app/globals.css) |

Requests go to `POST https://api.sarvam.ai/speech-to-text` with `model=saaras:v3`,
`mode=transcribe` and `language_code=unknown` so Saaras detects the language itself.

### Constraints worth knowing

- The REST endpoint accepts **30s of audio per request**; recording auto-stops at 29s.
- Sarvam matches content types **exactly**. `MediaRecorder` reports
  `audio/webm;codecs=opus`, which is rejected — the codec parameter is stripped on both the
  client and the server before upload.
- Recording requires a secure context: `localhost` works, any other host needs HTTPS.
