"""The grounding prompt.

This module holds the instructions that make an answer *grounded*: the
model is told, in order of importance, to use only the supplied evidence,
to invent nothing, to say so when the evidence does not answer the
question, to keep its own reasoning separate from what the documents
state, and to cite by source number.

Three choices are worth explaining, because each is doing more work than
it looks:

**The question and the evidence are separate, labelled sections.** Stage
6 deliberately renders no question and no instructions into the context
block; assembling them is this stage's job. Keeping them apart means the
model can never mistake a phrase from the question for a phrase from a
document — which is one of the easier ways to get a confident answer to a
question the documents do not address.

**Interpretation gets its own field, not an inline label.** Asking a
model to mark its inferences inside prose produces inconsistent hedging
that is hard to parse and easy to lose in rendering. A separate
``interpretation`` field makes the distinction structural: the backend
can show it differently, or not at all, and ``answer`` stays as close to
the documents as the model can keep it.

**Citations are asked for by source *number* only.** The model does not
need to restate a filename, a page or a chunk id — it only has to say
which block it used, and every other field is filled in afterwards from
the authoritative record. A model asked to reproduce metadata will
sometimes reproduce it wrongly, and a wrong page number in a legal
citation is worse than no page number.

What none of this does is make wrong answers impossible. Instructions
reduce ungrounded output; they do not eliminate it, which is why every
citation is validated against the real retrieved sources afterwards and
why quoted spans are checked against the evidence text.
"""

from __future__ import annotations

from typing import Optional

#: Labels the user message is built from. The extractive provider reads
#: them back, so they are part of the contract rather than decoration.
QUESTION_LABEL = "QUESTION:"
EVIDENCE_LABEL = "EVIDENCE:"

NO_EVIDENCE = "(no evidence passages were retrieved for this question)"

#: The JSON shape the model is asked for. Shown in the system prompt.
RESPONSE_SCHEMA = """{
  "answer": "your answer, using only the evidence above",
  "citations": [{"source": 1}, {"source": 3}],
  "insufficient_evidence": false,
  "interpretation": ""
}"""

SYSTEM_PROMPT = f"""\
You are a legal document assistant. You answer questions about documents \
the user has uploaded, using ONLY the evidence passages supplied with each \
question.

Follow every rule below without exception.

1. ANSWER FROM THE EVIDENCE ONLY. Everything in your answer must be \
supported by the evidence passages provided in the {EVIDENCE_LABEL} section. \
If a passage does not say something, you do not know it.

2. DO NOT INVENT FACTS. Do not state any fact that is not in the evidence.

3. DO NOT INVENT LEGAL CLAUSES. Do not refer to a clause, section or \
schedule number that does not appear in the evidence, and do not describe \
the contents of a clause you have not been shown.

4. DO NOT INVENT DATES, PARTIES, AMOUNTS OR CASE DETAILS. Names, dates, \
figures, jurisdictions and party identities must be copied from the \
evidence, never supplied from memory or inferred from what is typical.

5. DO NOT PRETEND TO KNOWLEDGE YOU DO NOT HAVE. You have no knowledge of \
these documents beyond the passages supplied. Do not draw on general legal \
knowledge, other matters, or what contracts usually say.

6. SAY SO WHEN THE EVIDENCE IS INSUFFICIENT. If the passages do not answer \
the question, or answer only part of it, say plainly what is missing and \
set "insufficient_evidence" to true. A clear "the retrieved passages do \
not address this" is a correct and useful answer. Never fill a gap with a \
plausible guess.

7. SEPARATE EVIDENCE FROM INTERPRETATION. The "answer" field is for what \
the documents state. Put any analysis, inference, implication or practical \
advice of your own in the "interpretation" field, and leave that field \
empty if you have none. Do not present interpretation as though the \
document said it.

8. CITE YOUR SOURCES. Every passage you relied on must appear in \
"citations", identified by its source number. Cite only sources you \
actually used. Do not cite a source number that is not in the evidence.

9. QUOTE EXACTLY. If you quote a document, copy the words exactly from the \
evidence and put them in quotation marks. Do not paraphrase inside quotes.

10. RESPOND WITH JSON ONLY. Reply with a single JSON object and nothing \
else - no preamble, no explanation, no markdown code fence:

{RESPONSE_SCHEMA}

"citations" is a list of objects each containing a "source" number from \
the evidence. You do not need to repeat the document name, page or chunk \
id; those are filled in from the record and will be corrected if you \
supply them differently.\
"""

#: Appended on a retry after an unparseable reply. Deliberately short and
#: about the *format* only: restating the grounding rules here would risk
#: the model rewriting the answer rather than reformatting it.
REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON. Reply again with the same "
    "answer, this time as a single JSON object and nothing else: no "
    "markdown fence, no text before or after the object."
)


def build_user_message(question: str, context: str) -> str:
    """The question and the evidence, in labelled sections."""
    evidence = context.strip() if context and context.strip() else NO_EVIDENCE
    return (
        f"{QUESTION_LABEL}\n{question.strip()}\n\n"
        f"{EVIDENCE_LABEL}\n{evidence}\n"
    )


def build_system_prompt(extra_instructions: Optional[str] = None) -> str:
    """The grounding rules, optionally with a deployment's own addition.

    The hook exists so a firm can add house rules — a jurisdiction note,
    a required disclaimer — without editing this file or forking the
    service. Additions are appended, never substituted: the ten rules
    above are the reason the answer is grounded and are not optional.
    """
    if not extra_instructions or not extra_instructions.strip():
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n\nAdditional instructions for this deployment:\n"
        f"{extra_instructions.strip()}"
    )


__all__ = [
    "SYSTEM_PROMPT",
    "RESPONSE_SCHEMA",
    "REPAIR_INSTRUCTION",
    "QUESTION_LABEL",
    "EVIDENCE_LABEL",
    "NO_EVIDENCE",
    "build_user_message",
    "build_system_prompt",
]
