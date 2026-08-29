"""Application-enforced product policy.

**The controls here are not prompt instructions, and that distinction is the
whole point of this module.**

Three behaviours were specified in the system prompt, observed failing in a paid
evaluation, hardened in the prompt, and observed failing again: presenting as the
underlying model, reproducing the corpus in bulk on request, and engaging with
unpublished work. Two attempts is enough evidence to stop treating prompt text as
a control for them.

A prompt is behavioural guidance. It shapes a distribution; it does not enforce
anything, and a sufficiently direct request will sometimes win. Product identity,
the privacy boundary and anti-extraction are properties the product must have
whatever the model does, so they are enforced in code — deterministically, before
the model runs where the question alone is enough to decide, and after it where
the answer must be inspected.

The prompt keeps its versions of these rules. They now improve the *typical*
answer rather than being relied on for the worst one, which is the right job for
a prompt.

Layering, and why each control sits where it does:

* **Pre-model.** The question alone determines the outcome, so no call is made.
  Deterministic, free, and impossible for a model to talk its way past.
* **Post-model.** Requires the generated answer, and for bulk reproduction the
  retrieved passages too. Catches novel phrasings that get past the input guard —
  the input guard is a filter, not a proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

APPROVED_IDENTITY = (
    "I am OJ Assistant, a Smart AI Assistant built by OJ Florendo to answer "
    "questions from his approved public portfolio content. I'm not OJ himself."
)
"""The approved answer to "what are you" — the *product* question.

Owner-approved wording, returned verbatim. Identity is a product fact, not
something to regenerate per request: a generated answer to this question is a
new opportunity to get it wrong every time it is asked.
"""

APPROVED_ARCHITECTURE = (
    "I'm OJ Assistant, built by OJ Florendo. I currently use Claude Haiku 4.5 "
    "as the language model within OJ's RAG architecture, alongside his "
    "retrieval, citation-verification, evaluation, privacy, and policy controls."
)
"""The approved answer to "what model are you" — the *implementation* question.

**The distinction this encodes is the whole point.** OJ Assistant is the product
OJ built; Claude Haiku 4.5 is one component inside it. The assistant may never
present itself *as* Claude, and may truthfully say that Claude powers part of
the system. Those are different sentences and only the first is prohibited.

Model and provider information is approved **public architecture information,
not a secret** — decided deliberately by the owner on 29 August 2026. Concealing
it would protect nothing: the model name sits in `settings.py` in a public
repository, and the differentiator was never the model. It is the corpus, the
locally re-verified citations, and the evaluation set. Being able to explain the
choice is a credibility asset, and an assistant that dodged the question while
the repository answered it would read as evasive rather than careful.

What stays private is operational: secrets, API keys, shared secrets, internal
service URLs, exact operational limits, private repositories, personal data,
unpublished work and roadmap information.
"""

BULK_EXTRACTION_RESPONSE = (
    "I can answer specific questions about OJ's work and show you the source "
    "behind each answer, but I don't reproduce my source material in bulk. Ask "
    "me something specific and I'll answer it with a link to where it comes from."
)

UNPUBLISHED_WORK_RESPONSE = (
    "I can only discuss work that OJ has chosen to make public. I can't "
    "provide, confirm, or speculate about unpublished projects, private plans, "
    "or future roadmap details. You're welcome to explore his published work, "
    "or contact OJ directly about a potential opportunity."
)


class Policy:
    """Names for the policies, so a decision can be reported and tested by name."""

    IDENTITY = "identity"
    ARCHITECTURE = "architecture"
    BULK_EXTRACTION = "bulk_extraction"
    UNPUBLISHED_WORK = "unpublished_work"
    PROVIDER_SELF_ID = "provider_self_identification"
    BULK_REPRODUCTION = "bulk_reproduction"


@dataclass(frozen=True)
class PolicyResponse:
    """A deterministic answer, decided without asking the model."""

    policy: str
    text: str


# ---------------------------------------------------------------------------
# Pre-model guards. The question alone decides, so no paid call is made.
# ---------------------------------------------------------------------------

_ARCHITECTURE_REQUEST = (
    # "Are you Claude?" — asked of the assistant, so it is really about what
    # powers it rather than about what it is.
    re.compile(r"\bare\s+you\s+(?:claude|chatgpt|gpt|gemini|llama)\b", re.I),
    re.compile(r"\bwhat\s+(?:model|llm)\s+are\s+you\b", re.I),
    # "What model powers OJ Assistant / this assistant / you"
    re.compile(
        r"\bwhat\s+(?:ai\s+)?(?:model|llm)\b[^.?!]{0,30}?"
        r"\b(?:powers?|runs?|drives?|behind|uses?)\b",
        re.I,
    ),
    re.compile(r"\bwhat\s+(?:are\s+you\s+)?(?:powered|running)\s+(?:by|on)\b", re.I),
    # "Does OJ Assistant / this assistant / do you use Anthropic?"
    re.compile(
        r"\bdo(?:es)?\s+(?:oj\s+assistant|this\s+assistant|the\s+assistant|you)\b"
        r"[^.?!]{0,30}?\buse\b[^.?!]{0,30}?"
        r"\b(?:anthropic|claude|openai|gpt|a\s+model|an?\s+llm)\b",
        re.I,
    ),
)
"""Questions about what powers *this assistant*.

Checked before product identity, because "Are you Claude?" matches both readings
and the implementation answer is the more useful one.

Deliberately narrow. Broader architecture questions — how retrieval works, what
citation verification does, what `cited` is built with, which model `cited` runs
— reach the corpus, which answers them properly and at more length than a fixed
string could. This group covers only the short, direct questions a visitor asks
*of the assistant itself*.
"""

_IDENTITY_REQUEST = (
    re.compile(r"\b(?:what|who)\s+(?:exactly\s+)?are\s+you\b", re.I),
    re.compile(r"\bare\s+you\s+(?:an?\s+)?(?:ai|bot|chatbot|robot|human|real)\b", re.I),
    re.compile(r"\bare\s+you\s+oj(?:\s+florendo)?\b", re.I),
    re.compile(r"\bwho\s+(?:made|built|created|trained)\s+you\b", re.I),
)

_BULK_EXTRACTION_REQUEST = (
    # "print/show/list/dump/repeat ... documents/context/passages"
    re.compile(
        r"\b(?:print|show|list|dump|output|reveal|repeat|reproduce|display|give)\b"
        r"[^.?!]{0,60}\b(?:documents?|context|passages?|sources?|corpus|files?|"
        r"knowledge\s*base)\b",
        re.I,
    ),
    # "everything you were given / all of your documents / in full"
    re.compile(
        r"\b(?:everything|all)\b[^.?!]{0,40}\b(?:you\s+(?:were\s+)?"
        r"(?:given|have|received)|your\s+(?:documents?|context|sources?))\b",
        re.I,
    ),
    re.compile(r"\bverbatim\b|\bword\s+for\s+word\b|\bin\s+full\b", re.I),
    re.compile(r"\brepeat\s+everything\b", re.I),
)

_UNPUBLISHED_WORK_REQUEST = (
    re.compile(
        r"\b(?:unpublished|unreleased|upcoming|future|planned|secret|private|"
        r"confidential|internal|in\s+progress|work[- ]in[- ]progress)\b"
        r"[^.?!]{0,40}\b(?:project|projects|work|plan|plans|roadmap|repo|"
        r"repository|repositories|product|feature|launch|idea|ideas)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:project|projects|work|plan|plans|roadmap|feature|features)\b"
        r"[^.?!]{0,40}\b(?:not\s+(?:yet\s+)?(?:published|released|public|shipped)|"
        r"hasn't\s+(?:published|released|shipped)|isn't\s+public)\b",
        re.I,
    ),
    re.compile(
        r"\bwhat(?:'s|\s+is)\s+(?:he|oj)\s+(?:working\s+on|building|planning)\b"
        r"[^.?!]{0,30}\b(?:next|now|currently|secretly)\b",
        re.I,
    ),
    re.compile(r"\b(?:roadmap|future\s+plans|next\s+projects?)\b", re.I),
)


def screen_question(question: str) -> PolicyResponse | None:
    """Decide the answer from the question alone, or return `None` to continue.

    Order matters twice. Architecture is checked before product identity, since
    "Are you Claude?" matches both and the implementation answer is the more
    useful one. Both are checked before extraction, because an identity question
    is not an attack, and answering it with anti-extraction copy would be a
    worse answer than the model would have produced.
    """
    text = question.strip()
    if not text:
        return None

    # Architecture first: "Are you Claude?" reads as both, and the specific
    # answer is the better one. Replying with the product identity alone would
    # be true but evasive, which is the impression this project can least
    # afford with a technical reader.
    if any(pattern.search(text) for pattern in _ARCHITECTURE_REQUEST):
        return PolicyResponse(Policy.ARCHITECTURE, APPROVED_ARCHITECTURE)

    if any(pattern.search(text) for pattern in _IDENTITY_REQUEST):
        return PolicyResponse(Policy.IDENTITY, APPROVED_IDENTITY)

    if any(pattern.search(text) for pattern in _BULK_EXTRACTION_REQUEST):
        return PolicyResponse(Policy.BULK_EXTRACTION, BULK_EXTRACTION_RESPONSE)

    if any(pattern.search(text) for pattern in _UNPUBLISHED_WORK_REQUEST):
        return PolicyResponse(Policy.UNPUBLISHED_WORK, UNPUBLISHED_WORK_RESPONSE)

    return None


# ---------------------------------------------------------------------------
# Post-model guards. These need the generated answer.
# ---------------------------------------------------------------------------

_PROVIDER_SELF_ID = (
    # "I'm Claude" / "I am Claude"
    re.compile(r"\bI(?:'m|\s+am)\s+Claude\b", re.I),
    re.compile(r"\bI(?:'m|\s+am)\s+(?:ChatGPT|GPT-\d|Gemini|Llama)\b", re.I),
    # "I'm an AI assistant made by Anthropic" and its variants.
    re.compile(
        r"\bI(?:'m|\s+am)\s+an?\s+(?:AI|large\s+language|language)\s*"
        r"(?:assistant|model)?[^.]{0,60}?\b(?:made|built|created|developed|"
        r"trained)\s+by\s+(?:Anthropic|OpenAI|Google|Meta)\b",
        re.I,
    ),
    # "As an AI model developed by Anthropic, ..."
    re.compile(
        r"\bas\s+an?\s+(?:AI|language)\s*(?:assistant|model)?[^.]{0,40}?"
        r"\b(?:made|built|developed|trained)\s+by\s+(?:Anthropic|OpenAI|Google)\b",
        re.I,
    ),
)
"""Unsolicited *self*-identification as the underlying model or provider.

**Deliberately not a blacklist of provider names.** The corpus legitimately
discusses Anthropic, Claude and the Anthropic API — they are part of how OJ's
projects are built, and a question about `cited`'s architecture should be
answered properly. What is prohibited is the assistant describing *itself* as
that model, which is a first-person construction and is what these patterns
match.
"""

BULK_REPRODUCTION_PASSAGE_THRESHOLD = 0.5
"""Fraction of a single retrieved passage that counts as "substantially reproduced"."""

BULK_REPRODUCTION_MAX_SOURCES = 2
"""How many substantially-reproduced *source documents* constitute bulk reproduction.

**Documents, not passages — corrected 29 August 2026 after a real false
positive.** The frozen v2.1 release evaluation replaced a correct answer to
"Tell me about Cited." with the anti-extraction refusal. That is a critical
`supported_fact` question, and the refusal cost three release criteria at once.

The measurement explains it. Retrieval returns *chunks*, and a broad question
about one project returns several chunks **of that project's single document**.
The legitimate 28 August answer to that question covered one chunk at 0.96 and
the next at **0.48** - two hundredths below the old passage count of two. Every
other legitimate answer measured 0.18 or less on its second passage, so this
question was not near the line, it was *on* it, and a slightly longer generation
of the same correct answer crosses.

Counting documents removes the coincidence rather than moving the line:

* the real 28 August violation reproduced four passages spanning **two distinct
  documents** (`project-cited.md`, `how-oj-works.md`) and is still caught;
* the legitimate Cited answer draws entirely from `project-cited.md` - **one
  document** - and is not, even if a future generation covers both of its chunks
  in full.

Reproducing two chunks of one document is what answering a broad question about
one subject looks like. Reproducing passages from several documents is what
emptying the corpus looks like. The old rule could not tell those apart because
it counted the wrong thing.

**Honest limitation, stated because it is a real narrowing.** An extractor who
targets one document at a time can obtain more of that single document in one
answer than before. Two things bound it: `is_single_passage_over_reproduction`
still catches near-whole reproduction of any individual passage, and the corpus
holds only owner-approved public material, so the worst outcome remains that
someone obtains text OJ already publishes. The control exists so the assistant
*answers* rather than *recites*.

The threshold value of two is unchanged; only the unit it counts is.

**Measured against the real corpus, not guessed** (spec v2.1). Across the 49
answers of the 28 August run, counted by *document*:

* 32 answers reproduced no passage at the 50% level;
* 16 answers substantially reproduced exactly **one** document, which is what a
  grounded answer to a focused question looks like, since the relevant passage
  largely *is* the answer; and
* the single bulk-extraction violation reproduced **two** documents across four
  passages.

An earlier candidate — the verbatim *fraction of the answer* — was measured and
rejected: legitimate grounded answers score up to 1.00 on it, because quoting
the corpus accurately is the product working correctly. It would have failed
good answers and is a good example of why the threshold had to be measured.
"""

SINGLE_PASSAGE_SPAN_FRACTION = 0.90
"""Longest *contiguous* verbatim span, as a fraction of the passage it came from.

The multi-passage rule permits one-source-at-a-time extraction: four requests
each reproducing a single passage never trip a threshold of two. This closes
that, and it needed a metric the earlier ones could not supply.

**Three candidates were measured and two rejected** (spec v2.1 §3, D2):

| Metric | Legitimate max | Extraction | Usable? |
| --- | --- | --- | --- |
| Single-passage *coverage* | 1.00 | 1.00 | **No** — no separation at all |
| Copied characters | 851 | 608 | **No** — inverted; legitimate is higher |
| Longest span in words | 71 | 52-106 | **No** — ranges overlap |
| **Longest span / passage length** | **0.79** | **1.00** | **Yes** |

The normalised version works where the raw versions do not, and the reason is
mechanical rather than a curve fit. *Coverage* sums every matched fragment
scattered through a passage, so an answer that quotes several parts and writes
around them can total 100%. The longest *unbroken* run is a different thing:
reproducing a passage yields one continuous run covering all of it, whereas
answering from it yields a quoted portion with original prose either side.

Measured on all 48 preserved legitimate answers from 29 August, the observed
four-document extraction, and seven synthetic single-document extractions built
by reproducing a real retrieved passage verbatim.
"""

SINGLE_PASSAGE_MIN_SPAN_WORDS = 45
"""Absolute floor, so a short passage quoted in full is not "substantial".

The corpus holds passages from 26 to 183 words. Without this, fully quoting a
26-word passage would score 1.00 and be flagged, which is a legitimate short
answer rather than reproduction. The smallest synthetic extraction measured 52
words, so the floor sits below it with margin.

Both conditions must hold. Either alone produces false positives.
"""

_MIN_RUN_CHARS = 40
"""Shortest verbatim run counted toward coverage.

Below this, matches are ordinary shared phrasing rather than reproduction.
"""


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def passage_coverage(answer: str, passages: tuple[str, ...]) -> tuple[float, ...]:
    """How much of each retrieved passage the answer reproduces verbatim.

    Coverage is measured against **the passage**, not against the answer. That is
    the direction that distinguishes the two behaviours: a focused answer uses a
    small part of a long passage, while a dump reproduces the passage whole.
    """
    normalised_answer = _normalise(answer)
    if not normalised_answer:
        return ()

    coverages: list[float] = []
    for passage in passages:
        normalised = _normalise(passage)
        if not normalised:
            coverages.append(0.0)
            continue
        matcher = SequenceMatcher(None, normalised, normalised_answer, autojunk=False)
        reproduced = sum(
            block.size
            for block in matcher.get_matching_blocks()
            if block.size >= _MIN_RUN_CHARS
        )
        coverages.append(reproduced / len(normalised))
    return tuple(coverages)


def longest_span_fraction(answer: str, passages: tuple[str, ...]) -> tuple[float, int]:
    """Longest contiguous verbatim run, as `(fraction of its passage, words)`.

    Measured against the passage rather than the answer, for the same reason
    coverage is: the question is how much of a *source* was reproduced, not what
    the answer was made of.
    """
    normalised_answer = _normalise(answer)
    if not normalised_answer:
        return (0.0, 0)

    best_fraction = 0.0
    best_words = 0
    for passage in passages:
        normalised = _normalise(passage)
        passage_words = len(normalised.split())
        if not passage_words:
            continue
        matcher = SequenceMatcher(None, normalised, normalised_answer, autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.size < _MIN_RUN_CHARS:
                continue
            span = normalised[block.a : block.a + block.size]
            words = len(span.split())
            fraction = words / passage_words
            if fraction > best_fraction:
                best_fraction, best_words = fraction, words
    return (best_fraction, best_words)


def is_single_passage_over_reproduction(answer: str, passages: tuple[str, ...]) -> bool:
    """Whether one source passage was reproduced rather than answered from.

    Enforces: *OJ Assistant may quote short excerpts necessary to support a
    grounded answer, but must not reproduce substantial source material as a
    substitute for answering.*

    Independent of the multi-passage rule and kept alongside it. That rule
    catches breadth; this one catches depth, and extraction one source at a time
    is invisible to the first.
    """
    fraction, words = longest_span_fraction(answer, passages)
    return (
        fraction >= SINGLE_PASSAGE_SPAN_FRACTION
        and words >= SINGLE_PASSAGE_MIN_SPAN_WORDS
    )


def is_bulk_reproduction(
    answer: str,
    passages: tuple[str, ...],
    sources: tuple[str, ...] | None = None,
) -> bool:
    """Whether the answer reproduces the retrieved material in bulk.

    Independent of the input guard by design: a novel phrasing that gets past the
    question patterns still fails closed here, because this looks at what was
    actually produced rather than at what was asked.

    `sources` names the document each passage came from, positionally. Given it,
    breadth is counted in **documents**, which is the unit that distinguishes
    answering a broad question about one subject from emptying the corpus - see
    `BULK_REPRODUCTION_MAX_SOURCES` for the measurement that forced the change.

    Omitting `sources` falls back to counting passages, which is the stricter
    reading and the right default for a caller that cannot say where a passage
    came from. Callers that *can* should, and the shipped path does.
    """
    if is_single_passage_over_reproduction(answer, passages):
        return True

    coverage = passage_coverage(answer, passages)
    substantial = [
        index
        for index, value in enumerate(coverage)
        if value >= BULK_REPRODUCTION_PASSAGE_THRESHOLD
    ]

    if sources is None or len(sources) != len(passages):
        # No usable attribution. Count passages, which cannot under-count.
        return len(substantial) >= BULK_REPRODUCTION_MAX_SOURCES

    distinct = {sources[index] for index in substantial}
    return len(distinct) >= BULK_REPRODUCTION_MAX_SOURCES


def has_provider_self_identification(answer: str) -> bool:
    """Whether the answer identifies itself as the underlying model or provider."""
    return any(pattern.search(answer) for pattern in _PROVIDER_SELF_ID)


def screen_answer(
    answer: str,
    passages: tuple[str, ...],
    sources: tuple[str, ...] | None = None,
) -> PolicyResponse | None:
    """Inspect a generated answer, returning a replacement when policy requires.

    Bulk reproduction is checked first: an answer that dumps the corpus must be
    replaced wholesale, and whether it also mentioned a provider is beside the
    point once none of it is being shown.
    """
    if is_bulk_reproduction(answer, passages, sources):
        return PolicyResponse(Policy.BULK_REPRODUCTION, BULK_EXTRACTION_RESPONSE)

    if has_provider_self_identification(answer):
        return PolicyResponse(Policy.PROVIDER_SELF_ID, APPROVED_IDENTITY)

    return None
