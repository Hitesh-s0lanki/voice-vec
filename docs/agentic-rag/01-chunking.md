# 01 — Chunking

Source modules:
[`data-ingestion/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/data-ingestion) ·
[`hybrid-search-strategies/src/semanti_chunking.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/hybrid-search-strategies/src/semanti_chunking.ipynb)

Chunking is the highest-leverage decision in the whole pipeline and the one most often made
by copying a default. A chunk is the atomic unit of retrieval: nothing smaller can ever be
returned, and nothing larger can ever be ranked. Every downstream metric is bounded by it.

The two forces pulling against each other:

- **Small chunks** → precise embeddings, tight matches, but answers get truncated mid-thought
  and the LLM loses the surrounding context that made the passage meaningful.
- **Large chunks** → self-contained context, but the embedding averages several topics into
  one vector, so it matches everything weakly and nothing strongly.

## The unit of measurement

Before choosing a strategy, choose what "size" means. The repo uses all three.

| Unit | Splitter | Pros | Cons |
| --- | --- | --- | --- |
| Characters | `CharacterTextSplitter`, `RecursiveCharacterTextSplitter` | Fast, no tokeniser, deterministic | Character count ≠ token count; ratio varies wildly by script |
| Tokens | `TokenTextSplitter`, `.from_tiktoken_encoder()` | Matches the model's real context limit | Slower; tokeniser must match the embedding model |
| Sentences | Custom, semantic chunkers | Never breaks mid-sentence | Needs a sentence segmenter that handles your language |

The character-vs-token gap is not cosmetic. For English, ~4 characters ≈ 1 token. For Indic
scripts in Devanagari, the same character count can be **2–4× more tokens** under a
byte-pair tokeniser, because the script is under-represented in the vocabulary. A
`chunk_size=500` tuned on English silently becomes a chunk that overflows the encoder when
you feed it Hindi. This matters directly for Vec — see [../03-chunking.md](../03-chunking.md).

The repo uses `.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)` in
[`adaptive-rag`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/adaptive-rag/src/adaptive_rag.ipynb)
and plain character counts everywhere else, without noting that the two `500`s mean
different things.

---

## Strategies in this repo

### 1. Fixed-size character splitting

```python
CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=20, length_function=len)
```

Splits on **one** separator, then packs the pieces up to `chunk_size`. The gotcha the repo
demonstrates by accident: if the separator never appears, or appears too rarely, the
splitter cannot split and returns chunks **larger than `chunk_size`**. It does not fall back.
`core.ipynb` runs this twice — once with `separator=" "` and once with `separator="\n"` —
producing different chunk counts on identical text, which is the whole lesson.

**Use when:** the text has one reliable structural delimiter you actually trust (log lines,
`\n\n`-separated records).

### 2. Recursive character splitting — the default

```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

Tries an ordered separator list — by default `["\n\n", "\n", " ", ""]` — descending from
coarsest to finest. It splits on paragraphs; if a piece is still too large it splits that
piece on newlines; then spaces; then, as a last resort, mid-character. This is why it
respects structure: it only degrades to a cruder boundary when the finer one fails.

This is the right default and the repo uses it in **13 of 16 modules**. `500/50` is the
house setting, appearing identically in `adaptive-rag`, `corrective-rag`, `autonomus-rag`
(all four notebooks), `rag-implementation`, `multi-model-openai`, and
`e2e-project/src/config/config.py`.

One misuse worth calling out: several notebooks pass `separators=[" "]`, which collapses the
recursive splitter into a fixed-size word splitter and throws away the entire benefit.
`core.ipynb` does this deliberately to demonstrate the difference;
`SmartPDFProcessor` in `data_parsing_pdf.ipynb` does it *non*-deliberately and inherits the
degraded behaviour.

**Overlap** is the repair mechanism for boundary damage. A fact that straddles a chunk
boundary is lost to both chunks unless they overlap. The 10% ratio (`50/500`) is
conventional; the cost is a ~10% larger index and duplicate passages in the result set,
which is what MMR ([03-retrieval.md](03-retrieval.md#mmr)) exists to clean up.

### 3. Token splitting

```python
TokenTextSplitter(chunk_size=50, chunk_overlap=10)
```

Same packing logic, measured in tokens. Use it whenever the chunk must fit a hard model
limit — CLIP's 77-token text encoder, an embedding model's max sequence length, or a
context budget you're accounting for exactly.

### 4. Semantic chunking — threshold method

Implemented twice, identically, in `data-ingestion` and `hybrid-search-strategies`:

```python
sentences  = [s.strip() for s in text.split('.') if s.strip()]
embeddings = model.encode(sentences)          # all-MiniLM-L6-v2

for i in range(1, len(sentences)):
    sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
    if sim >= 0.7: current_chunk.append(sentences[i])   # same topic, keep going
    else:          chunks.append(current_chunk); current_chunk = [sentences[i]]
```

Boundaries are placed where **consecutive-sentence similarity drops** below a threshold —
i.e. where the topic changes. On the demo text this correctly separates the three LangChain
sentences from the two France sentences.

Three real weaknesses in this implementation:

- **`text.split('.')` is not a sentence segmenter.** It breaks on `Dr.`, `3.14`, `etc.`,
  URLs, and abbreviations, and it does not work at all for scripts that use `।` (Devanagari
  danda) or `。` (CJK). For Vec's Hindi corpus this splitter would produce garbage.
- **Adjacent-pair comparison only.** It compares sentence *i* to *i−1*, so a single
  off-topic sentence severs the chunk permanently even if sentence *i+1* returns to the
  topic. The standard fix is a rolling window or a percentile threshold over the whole
  document's distance distribution.
- **No size bound.** A uniformly on-topic document yields exactly one chunk, however long.
  Production semantic chunkers clamp with `min_chunk_size` / `max_chunk_size`.

`threshold=0.7` is also not portable — it is specific to `all-MiniLM-L6-v2`'s similarity
distribution. Swap the embedding model and the threshold must be re-tuned.

### 5. Semantic chunking — LangChain's `SemanticChunker`

```python
from langchain_experimental.text_splitter import SemanticChunker
chunker = SemanticChunker(OpenAIEmbeddings())
```

The library version, which fixes the second weakness above: it uses a **breakpoint
percentile** over the document's own distance distribution rather than a fixed threshold,
and compares combined sentence groups rather than bare adjacent pairs. Still `experimental`,
still one embedding call per sentence — expensive at index time and unusable at query time.

### 6. Document-aware chunking — PDF

`SmartPDFProcessor` in
[`data_parsing_pdf.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/data-ingestion/src/data_parsing_pdf.ipynb)
is the most production-shaped code in the ingestion module. It chunks **per page**, not
across the whole document, which means:

- page numbers survive into metadata (`page`, `total_pages`) and become citable
- a chunk can never span a page boundary — no headers/footers spliced into body text
- near-empty pages are dropped (`len(cleaned) < 50`)

It also cleans the two classic PDF extraction artefacts:

```python
text = " ".join(text.split())        # collapse the whitespace PDFs emit everywhere
text = text.replace("ﬁ", "fi")       # decompose ligatures — U+FB01, U+FB02
```

Ligatures matter more than they look. `ﬁ` is a single codepoint; leave it in and
`"efficiency"` tokenises differently from every query the user types, so BM25 misses it
entirely and the dense embedding is subtly off. The general fix is Unicode NFKC
normalisation, which handles the whole ligature block plus a lot more; the repo hardcodes
the two most common.

### 7. Structured-data chunking

Tables and records don't have paragraphs, so "chunking" becomes "what is one row worth."
The repo shows both ends of the trade-off:

| Source | Naive | Intelligent (repo's own term) |
| --- | --- | --- |
| CSV | `CSVLoader` — one document per row, raw `col: val` dump | Hand-built prose per row + typed metadata (`price`, `category`) for filtering |
| Excel | — | One document per **sheet**, with column names and row count in the header line |
| JSON | `JSONLoader(jq_schema='.employees[]')` | Flatten nested children (projects) into the parent's prose so relationships survive |
| SQL | `SQLDatabase.get_table_info()` — DDL only | One doc per table (schema + 5 sample rows) **plus** a doc per JOIN, so cross-table relationships are retrievable |

The SQL JOIN document is the sharpest idea in the module. A vector store cannot perform a
join at query time, so any relationship you want to retrieve has to be **materialised as
text at index time**. "Jane Smith (Data Scientist) leads ML Platform" is retrievable;
`employees` and `projects` sitting in separate chunks are not.

The general principle across all four: **metadata is a first-class product of chunking, not
an afterthought.** A `price` float in metadata supports a pre-filter that no embedding can
express.

---

## Not in this repo

Six strategies that are standard and absent. Each is worth knowing about because they solve
failure modes the repo's five cannot.

### Parent-document / small-to-big

Embed small chunks (precision), but return their **larger parent** to the LLM (context).
Two stores: a vector index of children, a docstore of parents keyed by ID. LangChain ships
`ParentDocumentRetriever`. This is the standard resolution of the size trade-off at the top
of this document, and its absence is the biggest single gap in the repo's chunking coverage.

### Contextual retrieval

Prepend an LLM-generated, document-aware blurb to each chunk before embedding it —
"This chunk is from ACME's Q2 2024 report; it discusses revenue growth" — so that a chunk
saying "revenue rose 3%" is retrievable without the reader knowing whose revenue. Costs one
LLM call per chunk at index time and nothing at query time. Anthropic reported a large
reduction in retrieval failures from this plus BM25.

### Late chunking

Embed the **whole document** through a long-context embedding model, then mean-pool the
token embeddings per chunk. Each chunk vector is contextualised by the full document, so
pronouns and back-references resolve. Requires a long-context embedder; no LLM calls.

### Structure-aware splitting

`MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter`, `RecursiveCharacterTextSplitter.from_language()`
for source code, and layout-aware PDF parsers. These split on the document's *declared*
structure rather than guessing it from whitespace, and carry the heading hierarchy into
metadata. Free — no model calls at all. For any Markdown, HTML, or code corpus this beats
recursive character splitting outright.

### Sentence-window retrieval

Embed **one sentence** per chunk; at retrieval, expand the hit to include the *k* sentences
either side. Maximum embedding precision with restored reading context. Cheap, and it
sidesteps the "how big should a chunk be" question by answering it differently at index time
and at read time.

### Agentic / LLM chunking

Hand the document to an LLM and ask it where the semantic boundaries are. Highest quality,
highest cost, and non-deterministic — the same document chunked twice can differ. Reserve it
for small, high-value corpora.

---

## Choosing

| Situation | Strategy |
| --- | --- |
| Default, unknown text | Recursive character, 500/50, token-measured |
| Markdown / HTML / code | Structure-aware splitter — always, it's free |
| PDF | Page-scoped recursive + NFKC normalisation |
| CSV / SQL / JSON | Row- or record-level, with materialised relationships and typed metadata |
| Chunks lack context to be understood alone | Parent-document, or contextual retrieval |
| Topic boundaries matter more than size | Semantic (`SemanticChunker`, percentile mode) |
| Hard model limit (CLIP, encoder max) | Token splitter, sized to the limit |
| Small corpus, quality is everything | Agentic chunking |

Two rules that survive every situation:

1. **Measure in the tokeniser your embedding model uses.** A character budget tuned on
   English is a different budget in Devanagari.
2. **Chunking is an index-time cost.** Anything you do here is free at query time. If you
   are latency-bound, this is the only place you can spend freely — which is exactly the
   argument [../03-chunking.md](../03-chunking.md) makes for Vec.

Next: [02-embeddings-and-stores.md](02-embeddings-and-stores.md).
