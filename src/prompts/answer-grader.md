# Answer grader — Gate 4's model half

Two independent bits from one call. They could be two calls with tighter prompts, and the
paper's formulation uses separate graders — but each is a full round trip on a rung that
already makes several, and both questions are answerable from exactly the same material.

They are separate *bits* because they have different repairs: not supported means regenerate
from the same context; not useful means rewrite the query and retrieve again. Collapsing them
into one "bad answer" signal sends half the repairs at the wrong problem.

## System

You check an answer against the passages it was written from.

Return a bare JSON object and nothing else:

{"supported": true | false, "useful": true | false}

supported: every claim in the answer is stated in the passages.
useful: the answer actually addresses the question that was asked.

These are independent. An answer can be faithful to the passages and still not answer the question.

## Human

Question: {{{query}}}

Passages:
{{{context}}}

Answer: {{{answer}}}
