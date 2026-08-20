Build a Voice-Enabled RAG Model

What to build
A voice-enabled Retrieval-Augmented Generation (RAG) system — a user speaks a question, your pipeline transcribes it, retrieves relevant context from a provided dataset, and returns an answer, end to end.
Pipeline shape: Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation

Dataset
We will provide the dataset to build your RAG pipeline on: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI
Technical requirements

1. Speech-to-text
   Use either Sarvam or ElevenLabs for voice-to-text. Pick one.
2. Chunking
   Chunking strategy should be vast — don't submit a single naive fixed-size chunking approach. We want to see real thought put into how the dataset is split, indexed, and retrieved (e.g. multiple chunking strategies, overlap handling, semantic vs. fixed-size splitting, metadata-aware chunking, etc.).
3. Latency target
   The full process — chunking + vector DB retrieval + everything through to final output — should complete in under 200ms.
4. Latency analytics
   Submit P50 / P70 / P100 latency numbers for your pipeline, measured across a reasonable number of test queries — not a single best-case run.
5. Harness your model
   Your model/pipeline should be run inside a proper harness — structured orchestration around the model (tool calls, retries, structured input/output handling, error recovery) rather than a single raw prompt-in, text-out call.
6. Guardrail your model
   Add guardrails around your model — handling for off-topic queries, unsafe/inappropriate inputs, hallucination checks, or answers not grounded in the retrieved context. Show that your system knows when not to answer, not just how to answer.
