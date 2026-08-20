import { NextResponse } from "next/server";

const SARVAM_ENDPOINT = "https://api.sarvam.ai/speech-to-text";

/** The REST endpoint caps a request at 30s of audio; ~6 MB is a generous ceiling for that. */
const MAX_BYTES = 6 * 1024 * 1024;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const key = process.env.SARVAM_API_KEY;

  if (!key) {
    return NextResponse.json(
      { error: "SARVAM_API_KEY is not set on the server." },
      { status: 500 },
    );
  }

  let incoming: FormData;
  try {
    incoming = await request.formData();
  } catch {
    return NextResponse.json({ error: "Malformed upload." }, { status: 400 });
  }

  const file = incoming.get("file");
  if (!(file instanceof File) || file.size === 0) {
    return NextResponse.json({ error: "No audio was received." }, { status: 400 });
  }

  if (file.size > MAX_BYTES) {
    return NextResponse.json(
      { error: "Recording is too long — keep it under 30 seconds." },
      { status: 413 },
    );
  }

  const language = incoming.get("language_code");

  // Sarvam matches content types exactly: `audio/webm` passes, the
  // `audio/webm;codecs=opus` that MediaRecorder emits does not.
  const contentType = (file.type || "audio/webm").split(";")[0].trim();
  const audio = new Blob([await file.arrayBuffer()], { type: contentType });

  const payload = new FormData();
  payload.append("file", audio, file.name || "speech.webm");
  payload.append("model", "saaras:v3");
  payload.append("mode", "transcribe");
  // "unknown" lets Saaras detect the language itself.
  payload.append(
    "language_code",
    typeof language === "string" && language ? language : "unknown",
  );

  let response: Response;
  try {
    response = await fetch(SARVAM_ENDPOINT, {
      method: "POST",
      headers: { "api-subscription-key": key },
      body: payload,
    });
  } catch {
    return NextResponse.json(
      { error: "Could not reach Sarvam. Check your connection." },
      { status: 502 },
    );
  }

  const raw = await response.text();
  let body: unknown = null;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    // Sarvam returned something that isn't JSON — surface it as an upstream failure.
  }

  if (!response.ok) {
    const detail =
      (body as { error?: { message?: string }; message?: string } | null)?.error
        ?.message ??
      (body as { message?: string } | null)?.message ??
      raw.slice(0, 200);

    return NextResponse.json(
      { error: detail || `Sarvam responded with ${response.status}.` },
      { status: response.status === 401 || response.status === 403 ? 401 : 502 },
    );
  }

  const result = body as {
    request_id?: string;
    transcript?: string;
    language_code?: string | null;
    language_probability?: number;
  } | null;

  const transcript = result?.transcript?.trim() ?? "";

  if (!transcript) {
    return NextResponse.json(
      { error: "Nothing was picked up — try speaking a little closer." },
      { status: 422 },
    );
  }

  return NextResponse.json({
    transcript,
    languageCode: result?.language_code ?? null,
    confidence: result?.language_probability ?? null,
    requestId: result?.request_id ?? null,
  });
}
