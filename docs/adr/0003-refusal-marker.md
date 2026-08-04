# ADR-0003: The model declares a refusal; we do not infer one

- Status: Accepted
- Date: 2026-08-04
- Owner: OJ Florendo

## Context

ADR-0002 established that refusal is decided by the model reading the retrieved
passages, not by a similarity threshold. That left an implementation question:
how does the code *know* a refusal happened?

The first answer was structural and looked elegant. An answer must carry a
citation; a refusal has nothing to cite; therefore **no citations means a
refusal**:

```python
grounded = bool(citations)
```

It required no extra call, no string matching, and no cooperation from the
model. It was also wrong.

## What the evaluation found

The first live runs scored 93% accuracy on one run and 100% on the next, with
identical code, questions and corpus. The same question failed every time it
failed: *"How do I train my own language model?"*

Printing what the system actually said settled it. Across five attempts, it
refused correctly **every time**. What varied was whether it attached a citation
while doing so — pointing at the passage stating the guide is *"for professionals
who use AI tools in their work but do not build them."*

That is not a malfunction. That passage is the *evidence for the refusal*: it is
the sentence that establishes the question is out of scope. Citing it is better
behaviour than not citing it.

The proxy conflated two different questions:

- **Is this response supported?** — yes, by the scope passage.
- **Did this response answer anything?** — no.

`bool(citations)` answers the first while being used to decide the second.

## The fix that was wrong

The first correction added a rule to the system prompt: when refusing, do not
cite anything.

That is fighting good behaviour to protect a bad proxy. It also only half
worked — clean runs went from 3 of 6 to 4 of 6, because the model kept reaching
for the citation, correctly.

Recorded here because the instinct to patch the prompt was strong and wrong, and
because the pattern generalises: when a heuristic and a system disagree, tuning
the system to satisfy the heuristic is the wrong direction.

## Decision

**The model reports a refusal explicitly.** When the documents do not contain the
answer, it begins its reply with `NOT_IN_DOCUMENTS`, which is stripped before the
text is shown. Citing a passage that establishes scope is explicitly welcomed.

`grounded` now requires both halves: not a refusal, *and* carrying evidence.

```python
grounded = bool(citations) and not refused
```

Five consecutive runs at 100% accuracy and 100% refusal, from an oscillation
between 93% and 100%.

## Alternatives considered

**Match the refusal wording.** Look for "cannot answer", "do not contain", and
so on. Rejected: it breaks the moment the model rephrases itself, and it is
exactly the kind of unverifiable inference this project argues against
everywhere else.

**Structured outputs.** Would give a typed field rather than a marker, and is
the better mechanism in principle. Not available: structured outputs and
citations are mutually exclusive in the API, and citations are the product.

**A separate classification call.** Ask a second time whether the passages
answer the question. Two calls to learn something the first call already
determined, at double the cost and latency.

## Consequences

- Refusal detection now depends on the model following a formatting
  instruction. If it stops doing so, refusals get misread as answers. The
  evaluation set measures exactly this, which is why the score is the guard.
- The marker leaks into the response if stripping ever fails. Covered by a test.
- Adding a provider means re-checking that it follows the marker convention.
  Bounded, and visible immediately in the evaluation numbers.

## The wider lesson

The bug was found because the harness printed **which** question failed and
**what it said**. The aggregate score alone said only that something was wrong,
and would have led to tuning retrieval or the prompt — treating a scoring bug as
a quality problem.

An evaluation that reports only a number tells you when to worry. One that
reports the failing case tells you what to do. The second is worth the extra
work, and this ADR exists because it paid for itself on the first run.

## Related

- `docs/adr/0002-refusal-is-a-judgement-not-a-threshold.md` — why refusal is a
  judgement at all
- `src/assistant/answering.py` — the marker and the stripping
- `src/assistant/evaluation.py` — the metrics that exposed the instability
