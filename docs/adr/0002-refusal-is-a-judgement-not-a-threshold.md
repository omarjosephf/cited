# ADR-0002: Refusal is a judgement, not a similarity threshold

- Status: Accepted
- Date: 2026-08-04
- Owner: OJ Florendo

## Context

The product promise is that the assistant answers **only** from the supplied
documents, and says so when they do not contain the answer. Refusing correctly is
therefore not a nice-to-have — it is the feature that distinguishes this from a
system that answers confidently whether or not it should.

The original plan, recorded in the README and in ADR-0001's closing notes, was to
refuse on a **similarity threshold**: retrieve the best chunks, and if the top
score fell below some cutoff, decline to answer. It is the obvious design. It is
cheap, it needs no extra model call, and the number can be tuned.

M2 measured it before building on it.

## The measurement

Ten questions the corpus can answer, and three it cannot, scored against the real
index (10 chunks, `bge-small-en-v1.5`, cosine similarity):

| | Top score |
| --- | --- |
| In-scope questions — lowest | 0.666 |
| In-scope questions — mean | 0.740 |
| Out-of-scope questions — highest | **0.755** |
| Out-of-scope questions — mean | 0.579 |

**The ranges overlap by 0.089.** No cutoff exists that admits every answerable
question while rejecting every unanswerable one.

The instructive case is *"How do I train my own language model?"* at **0.755** —
higher than seven of the ten questions the corpus genuinely answers. The corpus
contains nothing about training models. But it is a document *about language
models*, so the question is topically adjacent to almost every chunk in it.

That is the whole problem in one example. **Embedding similarity measures topical
relatedness, not answerability.** A question can be maximally on-topic and still
have no answer present. The two are different properties, and only one of them is
what a threshold measures.

This is a property of the technique, not a threshold left untuned. No amount of
tuning separates two overlapping distributions.

## Decision

**Refusal is delegated to the model, reading the retrieved passages.**

The retrieved chunks are supplied as documents with citations enabled, and the
system prompt instructs the model to answer only from them and to say plainly
when they do not contain the answer. Deciding whether a passage answers a
question is a reading-comprehension task, which is what the model is for.

Two supporting mechanisms:

1. **An answer with no citation is not treated as an answer.** Whether the model
   declined, or produced something ungrounded, the structural signal is the same:
   nothing in the response points at a source. Both cases are surfaced to the user
   as "not answered from these documents" rather than presented as an answer. This
   is deliberately a structural check rather than string-matching a refusal
   phrase, which would break the moment the model reworded itself.

2. **A low threshold survives only as a pre-filter**, for questions so unrelated
   that a paid call is not worth making — *"what is the capital of France?"* scores
   0.428, well below anything in-scope. This saves cost on obvious noise. It is
   explicitly **not** the refusal mechanism, and must be set low enough that it
   never rejects a question the model should have been allowed to consider.

## Consequences

- Refusal costs an API call. A threshold would have been free. This is the price
  of correctness, and the pre-filter recovers the cheapest cases.
- Refusal quality now depends on prompt wording, which makes it a thing to
  measure rather than assume — M4's evaluation set scores refusal correctness
  alongside retrieval hit rate for exactly this reason.
- The system can still be wrong. A model can misjudge whether a passage answers a
  question, just as a threshold could. The difference is that it is wrong for
  reasons a human would recognise on reading the same passages, rather than
  because two probability distributions overlapped.

## What this decision does not claim

It does not make hallucination impossible. It makes an ungrounded answer
*detectable*, because an answer with no citation is visibly different from one
with a citation pointing at a real passage. Grounding does not eliminate error;
it makes error checkable — which is the property that matters in professional
use, and the same principle the demo corpus itself describes.

## Alternatives considered

**Tune the threshold harder.** Rejected: the distributions overlap, so no cutoff
exists. More tuning cannot fix a property of the measure.

**A cross-encoder re-ranker.** A second, slower model scoring (question, passage)
pairs directly would separate the two distributions far better than cosine
similarity does. Genuinely the right answer at scale, and rejected here only
because it adds a second model, a second download and real latency to solve a
problem the answering call already solves. Revisit if evaluation shows the model
accepting passages it should not.

**Ask the model a separate yes/no question first.** Two calls instead of one, to
learn something the answering call already determines. Rejected as pure cost.

## Related

- `docs/adr/0001-embeddings-without-pytorch.md` — the embedding stack this
  measurement was taken against
- `src/assistant/retrieval.py` — where the pre-filter threshold lives
- `src/assistant/answering.py` — where refusal is actually decided
