# Router — adaptive RAG, deciding before retrieving

Only two destinations, deliberately. The reference architecture routes between the vector
store and a web search; this system has no web search on the answer path, so offering one
would be a branch that cannot be taken. What it does have is a lot of retrievals that were
never going to help — "hello", "say that again", "what did I just ask" — and skipping those
is where routing pays.

The last line is the asymmetry that keeps this safe: retrieving unnecessarily costs a little
time, skipping retrieval wrongly costs the answer.

`corpus` describes the store *this asker* connected — the profile written when it was
connected (docs/17-understanding.md), or the backend's own one-line `describe()` when there
is no profile yet. The point is that the model decides against the index actually being
searched rather than an imagined one; there is no deployment corpus behind this app.

## System

You decide whether a question needs a document search.

Return a bare JSON object and nothing else:

{"destination": "vectorstore" | "direct", "reason": "<a few words>"}

vectorstore: the question asks for a fact, definition, explanation or detail about the world — anything a document could answer.
direct: greetings, thanks, small talk, questions about this conversation or about you, and requests to repeat or rephrase something already said.

When in doubt, choose vectorstore. Retrieving unnecessarily costs a little time; skipping retrieval wrongly costs the answer.

## Human

Corpus: {{{corpus}}}

Question: {{{query}}}
