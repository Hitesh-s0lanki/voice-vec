# Query rewriter — a repair, never a pre-processing step

This runs only after retrieval has already been graded bad, so it is paid for by a query that
was going to be abstained on anyway. A rewrite on the happy path is a round trip in front of
every question, which is what puts query enhancement outside this system's latency budget.

"Keep the user's language" matters more than it looks: the index is Hindi, the question may be
Tamil, and a rewrite that quietly translates to English throws away the cross-lingual path the
retrieval actually uses (docs/13a-cross-lingual.md).

The caller rejects a rewrite that came back identical, empty, or longer than 400 characters —
those are not second attempts at anything.

## System

Rewrite the question as a better search query for a passage index.

Keep the user's language. Keep every proper noun, number and unit.
Prefer the vocabulary a written article would use over the phrasing of speech.
Return only the rewritten query, on one line, with no quotes or preamble.

## Human

{{{query}}}
