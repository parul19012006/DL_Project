"""Query preprocessing — the first stage of the pipeline.

The two retrievers want *different* things from the same question, and
that is the reason this module exists rather than the pipeline simply
passing a string around:

**The embedder wants the question as a human wrote it.** Sentence
embeddings are trained on natural language; stripping stop words and
punctuation before encoding makes the vector worse, not better. So the
semantic side receives ``ProcessedQuery.normalized`` — Unicode-normalised
and whitespace-tidied, and otherwise untouched.

**BM25 wants content words.** Lexical matching on "what", "does", "the",
"about" retrieves noise, because those words appear in every chunk of
every contract. So the keyword side receives ``ProcessedQuery.terms`` —
filler removed, stop words dropped, and each term mapped to a canonical
form so that a question about *terminating* an agreement matches a
clause headed *TERMINATION*.

Two legal-specific extractions also happen here:

**Clause references.** "What does clause 4.2 say?" is not a semantic
question at all — the user has told us exactly where to look. The
reference is pulled out so the ranking stage can boost the chunk whose
section actually is 4.2, which no amount of vector similarity reliably
does.

**Quoted phrases.** A user who types ``"no oral modification"`` in quotes
means those words in that order. The phrase is kept intact so ranking
can reward an exact occurrence.

Everything here is heuristic and deliberately conservative: a missed
reference costs a small boost, an over-eager rule costs a wrong answer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import Settings

# ---------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------

#: Deliberately small. A general-purpose stop list would remove "shall",
#: "may", "must" and "not" — the words that carry the entire legal
#: meaning of a clause ("the Vendor shall not" vs "the Vendor may").
#: Only genuinely content-free words are listed.
STOPWORDS = frozenset(
    """
    a an the this that these those there here
    is are was were be been being am
    do does did doing done
    of in on at to for from by with within into onto about as
    and or but if then than so such
    i me my we our you your it its
    what which who whom whose when where why how
    can could would should will
    please tell show give find explain summarise summarize
    """.split()
)

# ---------------------------------------------------------------------
# Conversational filler
# ---------------------------------------------------------------------

#: Leading phrases a user types before the actual question. Removed from
#: the keyword form only; the semantic form keeps them, because they are
#: harmless to an encoder and rewriting a question is a good way to
#: change what it asks.
_FILLER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*(?:hi|hey|hello)[,!.\s]+",
        r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?"
        r"(?:tell\s+me|show\s+me|explain|find|check|confirm)\s*",
        r"^\s*(?:please|kindly)\s+",
        r"^\s*i\s+(?:would\s+like|want|need)\s+to\s+know\s+",
        r"^\s*(?:what|which)\s+does\s+(?:the|this|my)\s+"
        r"(?:contract|agreement|document|nda|policy)\s+say\s+about\s+",
        r"^\s*according\s+to\s+(?:the|this|my)\s+"
        r"(?:contract|agreement|document)[,\s]+",
        r"^\s*(?:tell|explain)\s+me\s+(?:about\s+)?",
        r"^\s*(?:what|who|when|where|how|why)\s+(?:is|are|was|were)\s+"
        r"(?:the\s+)?",
        r"^\s*in\s+(?:the|this)\s+(?:contract|agreement|document)[,\s]+",
    )
]

# ---------------------------------------------------------------------
# Legal references
# ---------------------------------------------------------------------

#: "clause 4.2", "section 7", "article IV", "schedule B", "exhibit 2",
#: "paragraph 3(a)", "annex 1".
_REFERENCE_RE = re.compile(
    r"\b(clause|section|sec|article|art|schedule|sched|exhibit|annex|"
    r"appendix|paragraph|para|item)\.?\s*"
    r"([0-9]+(?:\.[0-9]+)*(?:\s*\([a-z0-9]{1,4}\))?|[ivxlcdm]{1,7}\b|[a-z]\b)",
    re.IGNORECASE,
)

#: A bare multi-level clause number typed on its own: "4.2", "12.3.1".
_BARE_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d{1,3}){1,3})(?![\w.])")

_QUOTE_RE = re.compile(r"[\"“‘']([^\"”’']{3,120})[\"”’']")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")

# ---------------------------------------------------------------------
# Term normalisation
# ---------------------------------------------------------------------

#: Legal word families mapped to one canonical token, applied to the
#: query *and* to the corpus so the mapping is symmetric.
#:
#: This is a stand-in for a stemmer, and it is a deliberate choice. A
#: real stemmer (Porter, Snowball) is another dependency and it is
#: indiscriminate: it maps "arbitrary" and "arbitration" together, and it
#: mangles the defined terms that legal drafting depends on. An explicit
#: table covers the handful of families that actually matter in
#: contracts, is readable, and is wrong only where it is written to be
#: wrong. A generic plural rule (below) handles the rest.
_TERM_FAMILIES: Dict[str, str] = {}


def _family(canonical: str, *variants: str) -> None:
    for word in (canonical, *variants):
        _TERM_FAMILIES[word] = canonical


_family("terminate", "terminates", "terminated", "terminating", "termination",
        "terminations")
_family("indemnify", "indemnifies", "indemnified", "indemnifying",
        "indemnification", "indemnity", "indemnities")
_family("confidential", "confidentiality", "confidentially")
_family("disclose", "discloses", "disclosed", "disclosing", "disclosure",
        "disclosures", "nondisclosure", "non-disclosure")
_family("liable", "liability", "liabilities")
_family("notice", "notices", "notification", "notifications", "notify",
        "notified", "notifying")
_family("pay", "pays", "paid", "paying", "payment", "payments", "payable")
_family("oblige", "obligation", "obligations", "obligated", "obliged")
_family("warrant", "warrants", "warranted", "warranty", "warranties")
_family("assign", "assigns", "assigned", "assignment", "assignments")
_family("govern", "governs", "governed", "governing", "governance")
_family("arbitrate", "arbitrates", "arbitrated", "arbitration",
        "arbitrations", "arbitrator", "arbitrators")
_family("breach", "breaches", "breached", "breaching")
_family("renew", "renews", "renewed", "renewing", "renewal", "renewals")
_family("expire", "expires", "expired", "expiring", "expiration", "expiry")
_family("amend", "amends", "amended", "amending", "amendment", "amendments")
_family("limit", "limits", "limited", "limiting", "limitation", "limitations")
_family("damage", "damages")
_family("fee", "fees")
_family("invoice", "invoices", "invoiced", "invoicing")
_family("party", "parties")
_family("agree", "agrees", "agreed", "agreement", "agreements")
_family("dispute", "disputes", "disputed")
_family("jurisdiction", "jurisdictions")
_family("force-majeure", "majeure")


def normalize_term(token: str) -> str:
    """Canonical form of one token.

    Family table first, then a conservative plural rule. Anything not
    covered is left exactly as written — including clause numbers, defined
    terms and party names, which must never be "corrected".
    """
    token = token.lower()
    mapped = _TERM_FAMILIES.get(token)
    if mapped:
        return mapped
    # Plurals only, and only where the singular is unambiguous.
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """Lexical tokens for BM25.

    Clause numbers survive intact ("4.2" is one token, not "4" and "2"),
    because a chunk that mentions 4.2 is a far stronger match for a query
    about 4.2 than one that mentions the digit 4.
    """
    if not text:
        return []
    lowered = unicodedata.normalize("NFKC", text).lower()
    out: List[str] = []
    for match in _TOKEN_RE.finditer(lowered):
        token = match.group(0).strip(".-/")
        if not token:
            continue
        if not keep_stopwords and token in STOPWORDS:
            continue
        out.append(normalize_term(token))
    return out


# ---------------------------------------------------------------------
# The processed query
# ---------------------------------------------------------------------


@dataclass
class ProcessedQuery:
    """One question, prepared for each retriever that needs it."""

    raw: str
    #: Unicode-normalised, whitespace-collapsed. What gets embedded.
    normalized: str = ""
    #: Filler-stripped. Human-readable form of what the keyword side sees.
    keyword_text: str = ""
    #: Normalised content terms for BM25.
    terms: List[str] = field(default_factory=list)
    #: Quoted spans, lowercased, in the order typed.
    phrases: List[str] = field(default_factory=list)
    #: ``[("clause", "4.2"), ("schedule", "b")]``
    references: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to retrieve on.

        A query of pure stop words ("what is the") has a normalized form
        but no terms; it is still a legitimate semantic query, so
        emptiness is decided on the normalized text alone.
        """
        return not self.normalized.strip()

    @property
    def has_terms(self) -> bool:
        return bool(self.terms)

    def reference_strings(self) -> List[str]:
        """``["clause 4.2", "4.2"]`` — both forms, for matching."""
        out: List[str] = []
        for kind, number in self.references:
            out.append(f"{kind} {number}")
            out.append(number)
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "keyword_text": self.keyword_text,
            "terms": list(self.terms),
            "phrases": list(self.phrases),
            "references": [f"{k} {n}" for k, n in self.references],
        }


def _normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    # Control characters an upstream copy-paste can carry in.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
    return re.sub(r"\s+", " ", text).strip()


def _strip_filler(text: str) -> str:
    """Remove leading conversational wrappers, repeatedly.

    "Can you please tell me what the contract says about termination"
    peels down to "termination". If stripping would leave nothing, the
    original is kept — the filler *was* the question.
    """
    current = text
    for _ in range(4):  # bounded: each pass removes at most one wrapper
        before = current
        for pattern in _FILLER_PATTERNS:
            current = pattern.sub("", current, count=1)
        current = current.strip()
        if current == before:
            break
    if not current.strip():
        return text
    return current


def _extract_references(text: str) -> List[Tuple[str, str]]:
    canonical = {
        "sec": "section",
        "art": "article",
        "sched": "schedule",
        "para": "paragraph",
    }
    out: List[Tuple[str, str]] = []
    seen = set()
    for match in _REFERENCE_RE.finditer(text):
        kind = match.group(1).lower()
        kind = canonical.get(kind, kind)
        number = re.sub(r"\s+", "", match.group(2).lower())
        key = (kind, number)
        if key not in seen:
            seen.add(key)
            out.append(key)
    # A bare "4.2" is a reference too, but only when no labelled
    # reference already claimed it.
    claimed = {number for _, number in out}
    for match in _BARE_NUMBER_RE.finditer(text):
        number = match.group(1)
        if number not in claimed:
            claimed.add(number)
            out.append(("clause", number))
    return out


def preprocess_query(
    query: str, settings: Optional[Settings] = None
) -> ProcessedQuery:
    """Prepare a raw user question for every downstream retriever."""
    raw = query or ""
    normalized = _normalize_whitespace(raw)

    processed = ProcessedQuery(raw=raw, normalized=normalized)
    if not normalized:
        return processed

    processed.phrases = [
        _normalize_whitespace(m.group(1)).lower()
        for m in _QUOTE_RE.finditer(normalized)
    ]
    processed.references = _extract_references(normalized)

    keyword_text = _strip_filler(normalized)
    processed.keyword_text = keyword_text
    processed.terms = tokenize(keyword_text)

    # A question made entirely of stop words ("what are the terms of it")
    # still deserves a lexical attempt; fall back to the full token list
    # rather than handing BM25 nothing.
    if not processed.terms:
        processed.terms = tokenize(keyword_text, keep_stopwords=True)

    return processed


__all__ = [
    "ProcessedQuery",
    "preprocess_query",
    "tokenize",
    "normalize_term",
    "STOPWORDS",
]
