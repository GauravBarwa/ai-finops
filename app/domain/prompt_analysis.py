"""Deterministic text analysis and lexical benchmark retrieval, with explicit uncertainty.

These English instruction rules and TF-IDF matches are not a semantic LLM judge.
No provider calls, dependencies, difficulty scores or invented quality measurements.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.schemas.prompt import PromptAnalysis, PromptMatch

# Stop at supplied data so its words cannot override the actual instruction.
_DATA = re.compile(r"\b(?:text|review|document|passage|input|data)\s*:", re.I)
_CUES = {
    "summarization": r"\b(?:summari[sz]e|condense|tl;?dr|give\s+(?:me\s+)?a\s+summary)\b",
    "classification": r"\b(?:classify|categori[sz]e|label\s+(?:the\s+)?sentiment|"
    r"determine\s+(?:the\s+)?sentiment)\b",
    "extraction": r"\b(?:extract|return\s+(?:only\s+)?json|json\s+with\s+(?:keys|fields))\b",
    "reasoning": r"\b(?:calculate|solve|compute|how\s+many|how\s+much|why|prove|"
    r"compare|explain|what\s+is|which)\b",
}
_STOP = set(
    (
        "a an the is are was were be to of and or in on for with as by it its "
        "from that this these those their has have had can could would should "
        "following text please respond reply only return give me use using "
        "summarize summarise summary sentences sentence lines line words word "
        "classify classification extract extraction reasoning json label keys "
        "key fields field information below above review sentiment positive negative neutral"
    ).split()
)


def rough_tokens(text: str) -> int:
    """Conservative byte/lexeme heuristic; exact tokenization varies by provider."""
    lexemes = len(re.findall(r"\w+|[^\w\s]", text))
    return max(1, math.ceil(max(len(text.encode("utf-8")) / 3, lexemes * 1.3)))


def analyze_prompt(prompt: str, task_type: str, max_tokens: int) -> PromptAnalysis:
    instruction = _DATA.split(prompt, maxsplit=1)[0].split(":", 1)[0][:512]
    # Look only at the instruction prefix, not arbitrary quoted/source content.
    cues = []
    for task, pattern in _CUES.items():
        match = re.search(pattern, instruction, re.I)
        if match:
            cues.append((match.start(), task, match.group()))
    cues.sort()
    detected = cues[0][1] if cues else "reasoning"
    effective = detected if task_type == "auto" else task_type
    warnings = ["Token counts are rough provider-dependent estimates, not exact usage."]
    if not cues:
        warnings.append(
            "No clear English task instruction detected; reasoning is a broad fallback. "
            "Choose a task override if needed."
        )
    elif len(cues) > 1:
        warnings.append(
            "Multiple task signals found; auto uses the first instruction. "
            "Check the detected task for multi-step requests."
        )
    if task_type != "auto" and cues and detected != task_type:
        warnings.append(
            f"Prompt suggests {detected}, but your explicit {task_type} override is used."
        )
    hint = None
    length = re.search(
        r"\b(\d+)(?:\s*[-–]\s*(\d+))?\s+(words?|sentences?|lines?)\b", instruction, re.I
    )
    if length:
        count = min(max_tokens, int(length[2] or length[1]))
        per_unit = 1.5 if length[3].lower().startswith("word") else 32
        hint = max(1, min(max_tokens, math.ceil(count * per_unit)))
    if effective == "classification" and re.search(r"\bonly\b", instruction, re.I):
        hint = min(max_tokens, 16)
    return PromptAnalysis(
        detected_task_type=detected,
        effective_task_type=effective,
        task_type_source="auto" if task_type == "auto" else "explicit",
        input_tokens_rough=rough_tokens(prompt),
        requested_max_output_tokens=max_tokens,
        output_tokens_hint=hint,
        signals=[f"{task}: {cue}" for _, task, cue in cues],
        warnings=warnings,
    )


def _terms(text: str) -> Counter:
    # Match source/question content, not shared labels or JSON instruction scaffolding.
    data = _DATA.search(text)
    if data:
        text = text[data.end() :]
    elif ":" in text:
        prefix, body = text.split(":", 1)
        if any(re.search(pattern, prefix, re.I) for pattern in _CUES.values()):
            text = body
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return Counter(
        t[:-1] if len(t) > 4 and t.endswith("s") else t
        for t in tokens
        if t not in _STOP and not t.isdigit()
    )


def match_examples(prompt: str, examples: dict[str, str]) -> list[PromptMatch]:
    """Shared top topic matches for every model; instruction boilerplate is ignored.

    Similarity is lexical, not a calibrated confidence or a quality score.
    A low score yields no matches rather than pretending unrelated tasks are evidence.
    """
    query = _terms(prompt)
    documents = {key: _terms(value) for key, value in examples.items()}
    if not query or not documents:
        return []
    df = Counter(token for doc in documents.values() for token in doc)
    size = len(documents)

    def vector(terms):
        return {
            token: (1 + math.log(count)) * (1 + math.log((size + 1) / (df[token] + 1)))
            for token, count in terms.items()
        }

    q = vector(query)
    q_norm = math.sqrt(sum(value * value for value in q.values()))
    matches = []
    for key, doc in documents.items():
        v = vector(doc)
        norm = math.sqrt(sum(value * value for value in v.values()))
        score = (
            sum(value * v.get(token, 0) for token, value in q.items()) / (q_norm * norm)
            if norm
            else 0
        )
        if score >= 0.2:
            matches.append(PromptMatch(task_id=key, similarity=min(1, score)))
    matches.sort(key=lambda m: (-m.similarity, m.task_id))
    if not matches:
        return []
    return [m for m in matches[:3] if m.similarity >= matches[0].similarity * 0.8]
