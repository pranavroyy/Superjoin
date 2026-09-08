"""
Given candidate pairs of facts (found via TF-IDF retrieval in candidates.py),
ask the LLM to actually decide the relationship. This is where the
interesting reasoning happens - not in the retrieval step.

We batch several pairs into one call to keep latency/cost down, and always
ask for an explanation so results are auditable, not just a label.
"""
from . import llm

SYSTEM_PROMPT = """You are comparing pairs of facts that were independently \
extracted from different documents (or the same document), each with its own \
supporting quote. For EACH pair, decide the relationship between fact A and \
fact B:

- "corroborates": both facts assert essentially the same thing (same real-world \
quantity/status), even if phrased differently, rounded differently, or from a \
different document. Minor rounding/formatting differences still count as \
corroboration.
- "contradicts": the facts describe the same real-world subject/attribute at \
the same time/scope, but give materially different or incompatible values or \
statuses, and you can NOT explain the difference from what's given (e.g. \
different periods, different units, different scope). This is a genuine or \
likely conflict worth flagging to a human.
- "reconciled": the facts look contradictory at first glance (different \
numbers/statuses for what seems like "the same thing"), but the difference is \
explained by context that IS present in the two facts - e.g. different time \
periods, different units, different scope (standalone vs consolidated, India \
vs global, urban vs rural), or one is a later update superseding the other \
(e.g. a director resigning after an earlier filing listed them as active). \
Explain exactly which contextual factor reconciles them.
- "unrelated": on closer look these facts are not actually about a comparable \
subject/attribute at all, despite superficial textual similarity. Retrieval \
is imperfect, so this is an expected and fine outcome - don't force a \
relationship that isn't there.

Be conservative about "contradicts" - only use it when you genuinely cannot \
reconcile the two facts using the information given. Prefer "reconciled" \
whenever period, unit, or scope differences plausibly explain the gap.

Respond with ONLY a JSON array, no prose. One object per input pair, in the \
same order, with this shape:
{
  "pair_index": integer (matches the input index),
  "relationship": "corroborates" | "contradicts" | "reconciled" | "unrelated",
  "explanation": string (2-3 sentences max, reference the specific values/periods),
  "confidence": number between 0 and 1
}"""


def _fact_summary(fact, label):
    return (
        f'{label} [doc {fact["document_id"]}, page {fact["page"]}]: '
        f'subject="{fact["subject"]}", attribute="{fact["attribute"]}", '
        f'value="{fact["value_raw"]}"'
        + (f', unit="{fact["unit"]}"' if fact.get("unit") else "")
        + (f', period="{fact["period"]}"' if fact.get("period") else "")
        + (f', scope="{fact["scope"]}"' if fact.get("scope") else "")
        + f', quote="{fact["quote"]}"'
    )


def classify_pairs(pairs, batch_size=15):
    """pairs: list of (fact_a_dict, fact_b_dict, similarity_score)
    Returns list of dicts: {relationship, explanation, confidence} aligned to pairs."""
    all_results = [None] * len(pairs)

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start: batch_start + batch_size]
        lines = []
        for i, (fa, fb, _sim) in enumerate(batch):
            lines.append(
                f"Pair {i}:\n  {_fact_summary(fa, 'Fact A')}\n  {_fact_summary(fb, 'Fact B')}"
            )
        user = "\n\n".join(lines)
        try:
            parsed = llm.call_json(SYSTEM_PROMPT, user, max_tokens=4000)
        except Exception as e:
            for i in range(len(batch)):
                all_results[batch_start + i] = {
                    "relationship": "unrelated",
                    "explanation": f"Comparison failed: {e}",
                    "confidence": 0.0,
                }
            continue

        by_index = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "pair_index" in item:
                    by_index[item["pair_index"]] = item

        for i in range(len(batch)):
            item = by_index.get(i)
            if item:
                all_results[batch_start + i] = {
                    "relationship": item.get("relationship", "unrelated"),
                    "explanation": item.get("explanation", ""),
                    "confidence": float(item.get("confidence", 0.5)),
                }
            else:
                all_results[batch_start + i] = {
                    "relationship": "unrelated",
                    "explanation": "Model did not return a classification for this pair.",
                    "confidence": 0.0,
                }
    return all_results
