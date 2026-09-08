import os
import json
import re
from anthropic import Anthropic

MODEL = os.environ.get("SUPERJOIN_MODEL", "claude-sonnet-4-6")

_client = None


def client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def call_json(system, user, max_tokens=4000):
    """Calls Claude and parses a JSON object/array out of the response.
    Strips markdown code fences if the model wraps its output in them."""
    resp = client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )
    text = text.strip()
    # Strip ```json ... ``` fences if present
    fence = re.match(r"^```(?:json)?\s*(.*)```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to locate the first [...] or {...} block
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise
