# src/prompts/

One file per agent, named for the agent's `name` in [`src/agents/`](../agents/):
`router.md` is what `RouterAgent` says, `dataset-sql.md` is what `DatasetAgent` says. They
are loaded by [`prompts.py`](../agents/prompts.py) and rendered as LangChain
`ChatPromptTemplate`s.

They live here rather than in triple-quoted strings for one reason: a prompt is the part of
an agent most likely to be edited by somebody who is not editing Python that day, and the
diff of a prompt change should be a diff of the prompt, not of the file that happens to call
the model.

## The file format

```markdown
# Router

Anything before the first `##` is a note for whoever edits this file.
It is NOT sent to the model.

## System

The system message. Everything until the next `##`.

## Human

The user message. Usually where the variables go.
```

- `## System` is required. `## Human` is optional — an agent driving a tool loop
  (`dataset-sql.md`) supplies its own user turn.
- Everything above the first `##` is documentation. Say *why* a rule is there; the rules
  themselves belong below.

## Variables

Templates are **mustache**, and the interpolation is always **triple-braced**:

```markdown
Question: {{{query}}}
```

`{{query}}` — two braces — HTML-escapes what it substitutes, so a passage containing `<` or
`&` reaches the model as `&lt;` and `&amp;`. That is silent and it is wrong: it changes the
text the model is grading. Triple braces substitute raw. Two braces are never correct here.

Mustache rather than f-string formatting because these prompts are *full* of JSON braces —
`{"supported": true}` is a literal in three of these files, and under f-string templating
every one of them would have to be doubled.

## Editing one

The agent loads its file once, when it is constructed, and a missing file or a missing
`## System` section raises at startup rather than degrading a turn later. Restart the server
after editing; there is no hot reload.

Behaviour changes that belong in *code* — a budget, a timeout, a fallback, what happens to
an unparseable reply — are not in here. This directory is only what the model is told.
