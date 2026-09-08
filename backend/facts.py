"""
Turns raw page text into structured, evidence-grounded facts.

Design choice: the schema is deliberately generic (subject / attribute /
value / unit / period / scope / quote / page / fact_type) rather than a
fixed set of fields like "revenue" or "director_name". The *documents*
decide what facts exist; fact_type is a free-text label the model invents
per fact (e.g. "financial_metric", "corporate_governance",
"macroeconomic_indicator"), not a fixed enum. This is what lets the system
generalize to PDFs it has never seen.
"""
from . import llm

SYSTEM_PROMPT = """You are a meticulous fact-extraction engine. You read a chunk of a \
PDF (with page markers like [PAGE 12]) and pull out atomic, checkable facts: \
things stated with a concrete number, date, status, or named relationship \
that another document could corroborate or contradict.

Rules:
- Only extract facts that are EXPLICITLY stated in the text. Never infer, \
compute, or guess a value that is not written down.
- Skip vague, purely narrative, or promotional sentences with no checkable claim.
- Every fact must include a short verbatim "quote" (<=20 words) copied \
exactly from the text that proves the fact, and the exact page number that \
quote appears on (from the nearest preceding [PAGE N] marker).
- Prefer facts likely to recur or be comparable across documents: financial \
metrics, operational metrics, dates, headcounts, ownership/governance facts, \
macroeconomic indicators, named entities and their status, etc. But do not \
force a category that doesn't fit - invent an appropriate short fact_type \
label yourself for each fact.
- If a value has a unit (%, INR crore, USD million, days, etc.) capture it \
separately in "unit". If it refers to a specific time period (FY24, Q4 FY24, \
CY2024, "as of 31 March 2025", etc.) capture it in "period". If it has a \
scope that matters for comparison (standalone vs consolidated, India vs \
global, urban vs rural, etc.) capture it in "scope", else null.
- "value_number" should be the plain numeric value with no currency symbols \
or commas (e.g. 7976.3), or null if the fact isn't fundamentally numeric \
(e.g. a status or relationship fact).
- Extract at most 40 of the most important, comparable facts from this chunk. \
Quality and comparability over quantity.

Respond with ONLY a JSON array, no prose, no markdown fences. Each item:
{
  "subject": string,
  "attribute": string,
  "value_raw": string,
  "value_number": number or null,
  "unit": string or null,
  "period": string or null,
  "scope": string or null,
  "quote": string,
  "page": integer,
  "fact_type": string
}
If there are no extractable facts in this chunk, respond with []."""


def extract_facts_from_chunk(chunk_text, start_page, end_page):
    user = (
        f"Extract facts from the following PDF text (pages {start_page}-{end_page}). "
        f"Remember: page numbers in your output must come from the [PAGE N] "
        f"markers actually preceding each quote.\n\n{chunk_text}"
    )
    try:
        result = llm.call_json(SYSTEM_PROMPT, user, max_tokens=4000)
    except Exception as e:
        return [], str(e)
    if not isinstance(result, list):
        return [], "Model did not return a JSON array"
    cleaned = []
    for item in result:
        if not isinstance(item, dict):
            continue
        if "subject" not in item or "attribute" not in item or "value_raw" not in item:
            continue
        cleaned.append(item)
    return cleaned, None
