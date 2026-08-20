"""Speech in, speech out.

Four small pieces, each doing one thing to a stream:

  languages  which language was heard, and who can speak it back
  stt        audio → transcript (Sarvam Saaras, OpenAI as a fallback)
  llm        transcript → reply tokens (any OpenAI-compatible chat API)
  segment    reply tokens → speakable segments, as early as one can be spoken
  tts        a segment → PCM, streamed (Sarvam Bulbul, OpenAI off-Indic)

`src/services/voice_service.py` is what threads them together into a turn.
"""
