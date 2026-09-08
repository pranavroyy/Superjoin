# Fact Knowledge Layer

A system that reads PDFs, extracts grounded, atomic facts, and figures out
how facts across documents relate: **corroborate**, **contradict**, or
**reconcile through context** (different period, unit, or scope).

Built for the Superjoin VIT 2026 Engineering Intern assignment.

## Setup and Run Instructions

Requires Python 3.10+ and an Anthropic API key.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000` — upload a PDF, then check the **Facts**,
**Relationships**, and **Issues** tabs. `sample-data/` contains the
Delhivery and India-macroeconomy starter datasets for a quick test; upload
2-3 documents from the same folder to see cross-document relationships
appear.

There's also a plain HTTP API if you'd rather script it:

- `POST /api/documents` — multipart `file` upload (PDF). Returns counts of
  facts extracted and relationships found for that document.
- `GET /api/facts` (optional `?document_id=`) — all extracted facts.
- `GET /api/relationships` (optional `?relationship=corroborates|contradicts|reconciled`)
- `GET /api/issues` — logged extraction problems.
- `GET /api/summary` — counts for a dashboard.

## Video Demo

_[Add link here after recording — a PDF being processed end to end plus the
four required cases below.]_

## Approach

**Pipeline:** PDF → per-page text (PyMuPDF) → grouped into 3-page chunks →
Claude extracts atomic facts per chunk as JSON → facts stored in SQLite →
each new fact is matched against existing facts from *other* documents via
TF-IDF cosine similarity (candidate retrieval) → Claude classifies each
candidate pair as corroborates / contradicts / reconciled / unrelated, with
a written explanation → non-"unrelated" relationships are stored and
surfaced in the UI, linked back to the original quotes and page numbers.

**Fact schema is deliberately generic** — `subject`, `attribute`,
`value_raw`, `value_number`, `unit`, `period`, `scope`, `quote`, `page`,
`fact_type` (a free-text label the model invents per fact, not a fixed
enum) — instead of hard-coded fields like "revenue" or "director_name".
The documents decide what facts exist; the assignment explicitly asks for
this, and it's also what lets the same pipeline run unmodified on the
Delhivery filings and the India macroeconomy reports, which have almost no
schema overlap.

**Why TF-IDF instead of a real embedding model for candidate retrieval:**
comparing every new fact against every existing fact with an LLM call is
O(n²) and slow/expensive once you have a few hundred facts. A proper
embedding model (e.g. via a hosted embeddings API) would give better
retrieval quality, especially for facts phrased very differently ("resigned
as director" vs. "stepped down from the board"). TF-IDF + cosine similarity
was chosen because it needs no model download and runs fully offline/fast,
at the cost of missing purely-semantic (non-lexical) matches. This is the
single biggest quality lever left on the table — see Limitations.

**Why the LLM does the actual relationship judgment, not a rules engine:**
distinguishing "these numbers differ because of rounding" from "these
numbers differ because one is FY23 and one is FY24" from "these numbers
genuinely disagree" requires reading the surrounding context (period, unit,
scope) the way a person would. A rules engine over structured fields would
miss most of this — it's exactly the reasoning the assignment asks for, so
it's delegated to the model with a fact-by-fact quote as grounding, rather
than trusting the model's judgment unconditionally.

**Extraction failures are treated as data, not silently swallowed.** If a
chunk fails to parse as JSON, or the model returns a page number outside
the chunk it was given, that's logged to an `extraction_issues` table and
surfaced in the **Issues** tab instead of being dropped or (worse) trusted
blindly.

**AI tools used:** Claude (via the Anthropic API) does the actual fact
extraction and relationship classification at runtime — it's the core of
the system, not just a coding aid. Claude (as a coding assistant) was also
used to scaffold this codebase.

## The Four Required Cases

Using the `india-macroeconomy` starter dataset (Economic Survey 2024-25,
RBI Annual Report 2024-25, IMF Article IV 2025) is the fastest way to see
all four, since the three reports deliberately cover overlapping ground
from different vintages and institutions:

1. **Corroborated fact:** e.g. India's real GDP growth for a given fiscal
   year, stated independently by two of the three reports with matching or
   near-matching figures.
2. **Genuine/likely contradiction:** e.g. two reports giving materially
   different figures for the same metric, same period, same scope, with no
   stated reason for the gap.
3. **Reconciled by context:** e.g. two inflation or growth figures that
   differ because one is a provisional/advance estimate and the other a
   revised figure, or because one covers a fiscal year and the other a
   calendar year.
4. **Extraction/reasoning failure:** captured live in the **Issues** tab —
   e.g. a chunk where the source table layout confused page-level text
   extraction, or a fact whose page number the model got wrong. See
   Limitations for how this would be improved.

_(Exact examples, with quotes and page numbers, will be pulled from a real
run and pasted here / shown in the demo video before submission.)_

## Limitations and Next Steps

- **Retrieval quality is the weak point.** TF-IDF misses facts that are
  about the same thing but phrased with no shared vocabulary. Swapping in a
  hosted embedding model (e.g. Voyage AI, OpenAI embeddings) for candidate
  retrieval is the highest-leverage next step.
- **No incremental re-comparison.** New documents are compared against all
  existing facts, but existing facts are never re-compared against each
  other after the fact, and a newly uploaded document isn't compared
  against facts extracted *after* it. Re-running comparison as a background
  job (rather than only at upload time) would close this gap.
- **Table-heavy pages are the main extraction failure mode.** Financial
  statements with dense tables sometimes extract as jumbled text via plain
  PDF text extraction, which can cause the LLM to misread which number goes
  with which line item. A layout-aware extractor (e.g. table detection) or
  page-image-based extraction for pages that look table-heavy would help.
- **Value normalization is shallow.** Units like "₹ crore" vs "$ million"
  aren't converted to a common base, so the LLM has to reason about unit
  differences in natural language rather than the system doing the
  conversion and flagging only genuine numeric gaps.
- **No dedup/merge of near-identical facts** extracted from overlapping
  chunk boundaries within the same document (pages_per_chunk overlap = 0
  today, so this is rare but not impossible for facts near a chunk edge).
- **Schema evolution is implicit, not tracked.** `fact_type` is free text,
  so the schema does evolve as new document types are added, but there's no
  view of what fact types exist across the whole knowledge layer or how
  they cluster — worth adding as a simple aggregation endpoint.

## Additional Notes

`sample-data/` mirrors the provided starter datasets so the repo is
self-contained for testing. The system itself does not reference these
filenames or their contents anywhere in the code — it was built to run
against them without modification, and should behave the same way against
new PDFs.
