# 01 — The dataset

Source: [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) ·
paper [arXiv:2506.01615](https://arxiv.org/abs/2506.01615)

MS MARCO v2.1 (QnA) machine-translated into 14 Indic languages, with the original English
kept alongside every translation. It is a **parallel** corpus, not merely a translated one.

## The reframe that matters

The obvious reading is "here is my document corpus." That is a third of what is in the file.
It is three datasets braided together:

**1. A corpus.** `passages.Translated_passages[]` flattened across rows. This is what goes
into the vector DB.

**2. A labelled retrieval benchmark.** `is_selected[]` marks *which* passage answers the
query. So recall@k and MRR are computable with no human labelling and no LLM judge. We can
report "recall@5 = 0.87" instead of "retrieval seems good."

**3. A labelled abstention set.** ~39% of rows have `sum(is_selected) == 0` and
`Eng_Answer == "No Answer Present."` — queries the corpus genuinely cannot answer.

Requirement 6 asks us to *"show that your system knows when not to answer."* The dataset
hands us thousands of labelled unanswerable queries. That turns the softest requirement in
the brief into a hard metric. See [06-guardrails.md](06-guardrails.md).

## Why this dataset was chosen for us

It is Indic, and the brief names Sarvam — the Indic-specialised STT vendor. Indic dataset +
Indic STT means the unstated ask is **Indic voice RAG**: someone speaks Hindi or Tamil and
the entire retrieval path has to work in that language.

The parallel English (`Eng_Query`, `Eng_Answer`, `English_passages`) is deliberate. It
unlocks cross-lingual indexing, translate-then-retrieve as a comparison baseline, and
cross-lingual evaluation via `query_id`, which is stable across all 14 language files.

One consequence is load-bearing and is covered in [04-latency.md](04-latency.md):
**translate-then-retrieve cannot be on the query path**, because a translation call is
another network round trip. The English text is an *index-time and eval-time* asset, not a
query-time one.

## Schema

Read from the parquet footer of `validation/hinval.parquet`.

| Field | Type | Contents |
| --- | --- | --- |
| `source_lang` | `string` | always `eng_Latn` |
| `target_lang` | `string` | FLORES code — `hin_Deva`, `asm_Beng`, … |
| `meta` | `struct` | `model_name`, `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty` — the MT decode config |
| `query_id` | `int64` | original MS MARCO id; **stable across languages** |
| `query_type` | `string` | `DESCRIPTION` / `NUMERIC` / `ENTITY` / `PERSON` / `LOCATION` |
| `Eng_Query` | `string` | original English query |
| `query` | `string` | translated query |
| `Eng_Answer` | `string` | original English answer |
| `Answer` | `string` | translated answer |
| `passages` | `struct` of 3 parallel lists | `English_passages[]`, `Translated_passages[]`, `is_selected[]` (0/1) |

The three lists inside `passages` are index-aligned: `is_selected[i]` labels both
`English_passages[i]` and `Translated_passages[i]`.

> The repo's loader script advertises an `answers` list field. It does not exist. The real
> fields are the singular `Answer` / `Eng_Answer`.

## Sample record

First row of Hindi validation, abridged in the passage lists only.

```json
{
  "source_lang": "eng_Latn",
  "target_lang": "hin_Deva",
  "meta": {
    "model_name": "ckpt-3epochs-sft-then-400k-kd",
    "temperature": 0, "top_p": 1, "max_tokens": 4096,
    "frequency_penalty": 0, "presence_penalty": 0
  },
  "query_id": 1102432,
  "query_type": "DESCRIPTION",

  "Eng_Query": ". what is a corporation?",
  "query":     "कॉर्पोरेशन क्या है?",

  "Eng_Answer": "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
  "Answer":     "निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है और कानून में इस प्रकार से मान्यता प्राप्त होती है।",

  "passages": {
    "is_selected": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "English_passages": [
      "A company is incorporated in a specific nation, often within the bounds of a smaller subset of that nation, such as a state or province. …",
      "Today, there is a growing community of more than 2,100 Certified B Corps from 50 countries …",
      "Corporation definition, an association of individuals, created by law or under authority of law …",
      "Examples of corporation in a Sentence. …",
      "1: a government-owned corporation (as a utility or railroad) …",
      "McDonald's Corporation is one of the most recognizable corporations in the world. A corporation is a company or group of people authorized to act as a single entity (legally a person) and recognized as such in law. …",
      "… 10 passages total"
    ],
    "Translated_passages": [
      "एक कंपनी एक विशिष्ट देश में निगमित होती है, अक्सर उस देश के एक छोटे उपसमूह, जैसे कि एक राज्य या प्रांत, की सीमाओं के भीतर। …",
      "आज, 50 देशों से 2,100 से अधिक प्रमाणित बी कोर का एक बढ़ता हुआ समुदाय है …",
      "निगम की परिभाषा, व्यक्तियों का एक समूह, जो कानून द्वारा या कानून के अधिकार के तहत बनाया गया है …",
      "वाक्य में निगम के उदाहरण। …",
      "1: एक सरकारी स्वामित्व वाला निगम (एक उपयोगिता या रेलमार्ग के रूप में) …",
      "मैकडॉनल्ड कॉर्पोरेशन दुनिया के सबसे पहचानने योग्य निगमों में से एक है। एक निगम एक कंपनी या लोगों का समूह है जो एक एकल इकाई (कानूनी रूप से एक व्यक्ति) के रूप में कार्य करने के लिए अधिकृत है …",
      "… 10 passages total"
    ]
  }
}
```

`is_selected[5] == 1` → passage index 5 is gold. Note how closely `Answer` tracks the second
sentence of that passage. **MS MARCO answers are largely extractive**, and that survives
translation. This is the single most important property for hitting 200 ms — see
[02-architecture.md](02-architecture.md).

## Measured statistics

Row counts and file sizes from the HF dataset info endpoint. Distributions measured over the
**first 2,000 rows** of `validation/hinval.parquet` — a sample, not the full split.

| | |
| --- | --- |
| train | 10,080,140 rows across **13** language files |
| validation | 1,371,174 rows across **14** language files |
| total on disk | 55.6 GB parquet |
| per language | ≈ 775k train rows, ≈ 98k validation rows |
| passages per row | 6–10, mean 9.99 |
| rows with no gold passage | 781 / 2000 (39.1%) |
| rows with `"No Answer Present."` | 780 / 2000 (39.0%) |
| MT model | single checkpoint, `ckpt-3epochs-sft-then-400k-kd`, greedy (`temperature: 0`) |

Query type distribution over the same 2,000 rows:

| Type | Count | Share |
| --- | --- | --- |
| `DESCRIPTION` | 1437 | 71.9% |
| `NUMERIC` | 408 | 20.4% |
| `ENTITY` | 103 | 5.2% |
| `PERSON` | 43 | 2.2% |
| `LOCATION` | 9 | 0.5% |

The no-gold count (781) and the no-answer-string count (780) differ by one. Reconcile the
two signals when building the abstention labels rather than assuming they are identical —
[07-evaluation.md](07-evaluation.md) treats `sum(is_selected) == 0` as authoritative.

## Three gotchas

**Telugu has no train file.** 13 train files, 14 validation files — `teltrain.parquet` does
not exist despite the README table listing it. Pick Telugu and you are validation-only.

**The translations are noisy.** In the sampled row, *"a sucker for all-you-can-eat buffets"*
came back as *"सभी संभव भोज्य-प्रेमी"*, which is mangled. Consequences: do not score
generation with exact match against gold `Answer` (use embedding similarity or an LLM
judge), and accept that the hallucination check is grading against imperfect context. Say so
in the writeup rather than pretending the gold is clean.

**The HF viewer and `load_dataset` are both broken for this repo.** The viewer fails with
`ArrowNotImplementedError: Nested data conversions not implemented for chunked array
outputs`, and the README's `load_dataset("ai4bharat/MSMARCO-XI", "hi")` depends on a loader
script that modern `datasets` no longer executes. Only a `default` config resolves.

## How to actually read it

Point straight at the parquet files.

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={
        "train":      "hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet",
        "validation": "hf://datasets/ai4bharat/MSMARCO-XI/validation/hinval.parquet",
    },
    streaming=True,
)
```

Or DuckDB, which range-reads over HTTPS so you only pull the row groups you touch — this is
how every measured number above was produced:

```sql
INSTALL httpfs; LOAD httpfs;

SELECT query, passages.Translated_passages[6] AS positive
FROM read_parquet('https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet')
WHERE list_sum(passages.is_selected) > 0
LIMIT 10;
```

DuckDB lists are **1-indexed**; the JSON above is 0-indexed. `is_selected[5]` in JSON is
`[6]` in SQL.

## Scoping decision

Train is ~10M rows × ~10 passages ≈ **100M passages**. Not indexable in a hackathon, and not
necessary.

**We use the validation split, Hindi first.** ≈ 98k rows → ≈ 980k passages before dedup.
That is a serious index that still fits in memory, and it leaves ~98k real queries to
evaluate against.

Ramp:

| Phase | Rows | Passages (pre-dedup) | Purpose |
| --- | --- | --- | --- |
| A | 2,000 | ~20k | get the loop working end to end |
| B | 25,000 | ~250k | first real latency and recall numbers |
| C | ~98,000 (full `hinval`) | ~980k | headline numbers |
| D (stretch) | + `tamval` or `benval` | +~980k | cross-lingual claim |

State the scoping choice explicitly in the submission. Deliberate scoping reads as
engineering judgment; silent truncation reads as not having noticed.

Memory, for planning: 980k vectors × 384 dims × 4 bytes ≈ **1.5 GB** at full precision, or
≈ 380 MB with int8 scalar quantisation. See [03-chunking.md](03-chunking.md), which also
multiplies this by the number of chunking strategies indexed.
