"""Reading the model's reply.

The prompt asks for a bare JSON object. Models mostly comply and
reliably do not always: the common deviations are a markdown fence, a
sentence of preamble ("Here is the JSON:"), a trailing explanation, and
smart quotes from a model that has been writing prose. None of those are
errors on the model's part worth failing a request over, and all of them
are cheap to recover from.

What is *not* recovered from is a reply with no JSON object in it at all,
or one whose ``answer`` is missing. Those are genuine failures and are
raised as :class:`MalformedResponseError` so the caller can retry with a
format nudge and, failing that, report the failure rather than invent a
shape.

The parser is deliberately strict about **types** and lenient about
**packaging**. A citation given as ``"source": "2"`` is coerced to an
integer, because that is a formatting quirk; a citation given as
``"source": "the termination clause"`` is discarded, because guessing
which block was meant is exactly the kind of invention this stage exists
to prevent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.generation.base import MalformedResponseError
from app.logging_config import get_logger

logger = get_logger(__name__)

#: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

#: Characters a prose-writing model substitutes for JSON's own.
_SMART = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    " ": " ",
}


@dataclass
class ParsedAnswer:
    """The model's reply, typed. Not yet validated against the sources."""

    answer: str = ""
    #: Source numbers the model claims to have used, in the order given.
    cited_sources: List[int] = field(default_factory=list)
    #: Anything else the model put in a citation, kept only for
    #: diagnosis — the authoritative metadata comes from the record.
    raw_citations: List[Dict[str, Any]] = field(default_factory=list)
    insufficient_evidence: bool = False
    interpretation: str = ""
    #: True when the object had to be recovered from a fence or prose.
    recovered: bool = False

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_sources)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text


def _extract_object(text: str) -> Optional[str]:
    """The first balanced ``{...}`` in the text.

    Brace counting rather than a regex, because a JSON object containing
    a brace inside a string literal — which a quoted clause easily does —
    defeats every non-counting approach. String and escape state are
    tracked so a brace inside a quoted span is not counted.
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None


def _coerce_source(value: Any) -> Optional[int]:
    """A source number, or ``None`` if it is not one.

    ``"2"`` and ``2.0`` are formatting; ``"Source 2"`` is too, and is
    accepted because the number is unambiguous. Anything without a
    number in it is discarded rather than guessed at.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 and value.is_integer() else None
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            number = int(match.group(0))
            return number if number > 0 else None
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def parse_answer(text: str) -> ParsedAnswer:
    """Turn a raw reply into a :class:`ParsedAnswer`.

    Raises :class:`MalformedResponseError` when no usable object can be
    recovered.
    """
    raw = (text or "").strip()
    if not raw:
        raise MalformedResponseError("The model returned an empty reply", raw="")

    for wrong, right in _SMART.items():
        raw = raw.replace(wrong, right)

    recovered = False
    candidate = _strip_fence(raw)
    if candidate != raw:
        recovered = True

    payload: Optional[Dict[str, Any]] = None
    try:
        payload = json.loads(candidate)
    except ValueError:
        extracted = _extract_object(candidate)
        if extracted is not None:
            try:
                payload = json.loads(extracted)
                recovered = True
            except ValueError:
                payload = None

    if not isinstance(payload, dict):
        raise MalformedResponseError(
            "The model's reply did not contain a JSON object", raw=text or ""
        )

    if "answer" not in payload:
        raise MalformedResponseError(
            "The model's reply contained JSON but no 'answer' field",
            raw=text or "",
        )

    answer = payload.get("answer")
    if not isinstance(answer, str):
        # A model that returns a list of bullet strings is being
        # helpful in the wrong shape; joining is recovery, not invention.
        if isinstance(answer, (list, tuple)):
            answer = " ".join(str(part) for part in answer)
            recovered = True
        else:
            answer = "" if answer is None else str(answer)
            recovered = True

    parsed = ParsedAnswer(
        answer=answer.strip(),
        insufficient_evidence=_as_bool(payload.get("insufficient_evidence")),
        interpretation=str(payload.get("interpretation") or "").strip(),
        recovered=recovered,
    )

    citations = payload.get("citations")
    if isinstance(citations, (list, tuple)):
        for entry in citations:
            if isinstance(entry, dict):
                parsed.raw_citations.append(entry)
                number = _coerce_source(
                    entry.get("source", entry.get("source_number", entry.get("id")))
                )
            else:
                # A bare list of numbers is a common and harmless shape.
                parsed.raw_citations.append({"source": entry})
                number = _coerce_source(entry)
            if number is not None and number not in parsed.cited_sources:
                parsed.cited_sources.append(number)

    if not parsed.answer and not parsed.insufficient_evidence:
        raise MalformedResponseError(
            "The model returned an empty answer", raw=text or ""
        )

    return parsed


__all__ = ["ParsedAnswer", "parse_answer"]
