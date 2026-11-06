"""Transparent heuristic scores; these are proxies, not semantic judge verdicts."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

SCORER_VERSION = "deterministic-v1"


def normalize(value: Any) -> str:
    return " ".join(str(value).casefold().split()).strip(" .!\n\t\"'`")


def parse_json_object(text: str) -> tuple[dict | None, bool]:
    """Decode strict JSON first, then objects inside fences or conversational text.

    JSONDecoder handles nested objects, braces in strings and escaped quotes.
    """
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value, True
    except (ValueError, TypeError):
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(value, dict):
                return value, False
        except ValueError:
            continue
    return None, False


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_equal(a, b) for a, b in zip(actual, expected))
        )
    return normalize(actual) == normalize(expected)


def extraction_score(output: str, reference: str, strict_json: bool = True) -> float:
    expected, _ = parse_json_object(reference)
    actual, raw_json = parse_json_object(output)
    if not expected or actual is None:
        return 0.0
    correct = sum(k in actual and _equal(actual[k], v) for k, v in expected.items())
    # Extra fields reduce precision. Wrappers cost at most five percent.
    accuracy = correct / max(len(expected), len(actual))
    return accuracy * (0.95 if strict_json and not raw_json else 1.0)


_STOPWORDS = set("a an the is are was were be to of and or in on for with as by it its".split())


def _words(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.casefold())) - _STOPWORDS


def score_output(
    task_type: str,
    output: str,
    reference: str,
    *,
    prompt: str = "",
    judge: Callable[[str, str, str], float] | None = None,
) -> float:
    """Return a bounded score without external calls unless a judge is supplied."""
    if not output.strip() or not reference.strip():
        return 0.0
    if task_type == "classification":
        return float(normalize(output) == normalize(reference))
    if task_type == "extraction":
        return extraction_score(output, reference)
    if task_type not in {"summarization", "reasoning"}:
        raise ValueError(f"Unsupported task type: {task_type}")
    expected, actual = _words(reference), _words(output)
    if not expected or not actual:
        return 0.0
    coverage = len(expected & actual) / len(expected)
    precision = len(expected & actual) / len(actual)
    # Numeric correctness matters for reasoning: a wrong answer must not pass on prose overlap.
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", reference))
    if task_type == "reasoning" and numbers:
        if not numbers.issubset(set(re.findall(r"\b\d+(?:\.\d+)?\b", output))):
            return 0.0
    score = 0.8 * coverage + 0.2 * precision
    sentence_constraint = re.search(r"\b(\d+)(?:\s*[-–]\s*(\d+))? sentences?", prompt)
    if sentence_constraint:
        lo = int(sentence_constraint[1])
        hi = int(sentence_constraint[2] or lo)
        count = len([s for s in re.split(r"[.!?]+(?:\s+|$)", output) if s.strip()])
        if not lo <= count <= hi:
            score *= 0.9
    word_constraint = re.search(r"(?:at most|max(?:imum)?)\s+(\d+) words", prompt, re.I)
    if word_constraint and len(output.split()) > int(word_constraint[1]):
        score *= 0.9
    if judge is not None:
        score = (score + max(0.0, min(1.0, judge(prompt, output, reference)))) / 2
    return max(0.0, min(1.0, score))
