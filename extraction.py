"""LLM knowledge-graph extraction (entities + typed relations).

Uses the raw provider SDKs (so the web app's "Integrate with AI" path works
without the heavier LangChain/deepagents stack) and returns a normalized
``{"entities": [...], "relations": [...]}`` payload. Any failure returns an
empty result so callers can fall back to the heuristic extractor.
"""

from __future__ import annotations

import json
import os
import re

import config
from providers import PROVIDERS

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from langsmith import traceable
except ImportError:  # tracing is optional
    def traceable(*dargs, **dkw):  # type: ignore
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]
        def _wrap(fn):
            return fn
        return _wrap


ENTITY_PROMPT = """You are extracting a knowledge graph from a passage of text.

Identify the 8-15 MOST IMPORTANT entities — the named people, organizations,
places, products, technologies, methods, frameworks, or concepts that this
passage is actually ABOUT. Skip generic terms and sentence-starting words
like "In", "When", "Round". Prefer multi-word canonical names.

Then identify the SEMANTIC RELATIONSHIPS between those entities — who did
what to whom, what depends on what, what is a part of what. These triples
are what makes a knowledge graph useful.

Return ONLY a JSON object with this exact shape:

{{
  "entities": [
    {{
      "name": "short canonical form (1-5 words, Title Case)",
      "kind": "person | organization | place | concept | technology | method | event | product",
      "aliases": ["optional alternate names/spellings"],
      "importance": 1-5,
      "confidence": "EXTRACTED | INFERRED | AMBIGUOUS"
    }}
  ],
  "relations": [
    {{
      "source": "<entity name as above>",
      "target": "<entity name as above>",
      "predicate": "short active-voice verb phrase (1-4 words)",
      "confidence": "EXTRACTED | INFERRED | AMBIGUOUS"
    }}
  ]
}}

Rules:
- Entity names in `relations` MUST exactly match names in `entities`.
- At least half the entities should appear in at least one relation.
- Predicates should be specific and meaningful (e.g. "expands prompt into",
  "evaluates output of", "replaces", "depends on"), not generic ("mentions",
  "related to", "is").

Text:
\"\"\"
{text}
\"\"\"

Return only the JSON object, no prose:"""


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        out = json.loads(m.group())
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


@traceable(name="kg.extract", tags=["kg", "extraction"])
def extract_kg_llm(text: str, provider: str, model: str) -> dict:
    """Return ``{"entities": [...], "relations": [...]}`` for a passage."""
    prompt = ENTITY_PROMPT.format(text=text[:8000])
    out = ""
    try:
        if provider == "anthropic":
            if Anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
                return {}
            client = config.traced_anthropic(Anthropic())
            msg = client.messages.create(
                model=model, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            out = "".join(b.text for b in msg.content if hasattr(b, "text"))
        elif provider in PROVIDERS and OpenAI is not None:
            cfg = PROVIDERS[provider]
            api_key = os.environ.get(cfg["env_key"])
            if not api_key:
                return {}
            client = config.traced_openai(OpenAI(
                api_key=api_key,
                base_url=os.environ.get(cfg.get("base_url_env", ""), cfg.get("base_url")),
            ))
            resp = client.chat.completions.create(
                model=model, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            out = resp.choices[0].message.content or ""
        else:
            return {}
    except Exception:
        return {}

    parsed = _parse_json_object(out)
    if not isinstance(parsed.get("entities"), list):
        parsed["entities"] = []
    if not isinstance(parsed.get("relations"), list):
        parsed["relations"] = []
    return parsed
