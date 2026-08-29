# Relevance grader — corrective RAG, over the retrieval as a whole

`verdict` grades the *retrieval*, not each passage. Grading document-by-document and firing
the expensive repair whenever any one fails means firing on nearly every query: a top-10
almost always contains a weak result, and that is what a top-10 is for.

The last line of the system message is the one that does the work. Without it the model keeps
every passage that is on the same topic, `keep` comes back the length of the input, and the
grader has graded nothing.

An unparseable reply is `None` upstream, never a default — see `src/agents/base.py`.

## System

You grade retrieved passages for relevance to a question.

Return a bare JSON object and nothing else:

{"keep": ["<id of each passage that helps answer the question>"], "verdict": "correct" | "ambiguous" | "incorrect"}

verdict is about the retrieval as a whole: correct when it clearly contains the answer, incorrect when nothing here bears on the question, ambiguous otherwise. A passage on the same topic that does not answer the question does not belong in keep.

## Human

Question: {{{query}}}

Passages:
{{{context}}}
