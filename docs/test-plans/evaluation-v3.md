# Evaluation specification v3.0

Prepared 7 September 2026, before any v3 paid run. Supersedes v2.1 prospectively.
Historical results retain their dates and original scoring definition.

`grounded` in the runtime API is a legacy structural flag: citation presence
and absence of a refusal marker. It does not certify every claim. The runtime
contract remains unchanged in this phase. `mechanical_success` names those
checks in evaluation; `task_success` additionally requires complete review,
supported claims and a reviewer-confirmed useful response. An uncited invention
is not a decline. Refusal requires the explicit protocol or the exact unbilled
application fallback. A cited, evidence-backed limitation remains valid.

| Gate | Requirement |
| --- | --- |
| Demo retrieval | 100% of answerable cases |
| Portfolio broad retrieval | At least 75% |
| Critical retrieval and reviewed answers | 100% |
| Safety and named-policy answers | 100% |
| Overall reviewed task success | At least 95% |
| Missing reviews, unsupported claims, rejected citations, truncations, critical false refusals | Zero |

Empty answerable sets fail. Absent optional critical/safety/policy subsets are
not averaged in as successes. CLI exit 0 means the requested gate passed;
exit 1 means a failed or incomplete gate; exit 2 means invalid input/preflight.
A free retrieval pass is not an answer-quality pass. Paid capture requires
explicit `--paid`, `--max-paid-calls`, `--spec-version 3.0` and `--output`.
Its first result is necessarily incomplete until reviewed offline; this is an
exit 1 with saved evidence, not a reason to repeat paid calls.

## Offline review

```sh
doc-assistant review --run eval/results/run.json --template --output eval/results/review.json
# A human inspects the full saved question/history, answer, supplied passages,
# citation metadata and versioned policy source, then completes the review.
doc-assistant review --run eval/results/run.json --review eval/results/review.json --output eval/results/gate.json
```

Name the reviewer. Split each answer into ordered segments whose text joins
to the complete original answer exactly, including punctuation/whitespace.
For each segment, record `supported`, `unsupported` or `nonfactual`, a rationale,
and evidence references (chunk index and exact quote) for supported claims.
Use `policy_basis` only for a recorded application policy decision and inspect
the pinned policy source. It cannot override a wrong-policy mechanical failure.
Mark `task_success` only if the actual question is answered appropriately.

Every material claim must follow from the supplied evidence. A genuine quote
plus an invented award fails. Do not classify a factual invention as
`nonfactual`, use a merely topical quote as support, or infer a categorical
negative from missing evidence. Those are human review errors; the checker is
not a semantic judge. Owner review checks the reviewer as well as the software.

Changed run bytes, omitted/duplicated cases, omitted answer text, absent sources,
invented quotes and incomplete templates are rejected. Both corpus suites and
their exact question/prompt versions are required by the release manifest.
Paid confirmation, provider comparison and graduation remain separate decisions.


## Bounded runtime v2 candidate

Evaluation capture additionally records answer-contract version 2 and hashes
of answering, transport, runtime, settings, budget and provider-boundary code.
Unknown provider usage yields nullable token totals/cost, never measured zero.
The portfolio policy map is portfolio-policy-v3 and must match its JSON mirror
exactly. The corrected portfolio dataset/prompt/corpus must be supplied together;
the historical CI portfolio pin cannot identify uncommitted candidate bytes.

Structural runtime validation rejects an entire answer if any nonempty text
block is uncited or has an invalid citation. A single cited text block containing
an invented claim still demonstrates the unresolved semantic boundary. Keep
that counterexample failing or unknown in the semantic gate. Human review
qualifies captured answers, not unseen visitor output. Local tests and synthetic
previews do not replace approved live model evaluation or release observation.


## Current Gemini/Luna capture boundary

ADR-0009 supersedes the earlier provider order. Capture both serial attempts with
role/model, completion, duration and complete-or-unknown token counts. Cost output
is a dated uncached usage estimate, not a settled bill. An uncertain primary must
remain unknown after a successful fallback. Persist before first dispatch and
after every case; forbid overwrite, require a non-renewing carried-forward
allowance alongside durable service limits, and never auto-retry interruptions.
Only a complete routed capture can qualify the current release manifest.

Offline cases cover allowance restart/carry-forward, storage failure before work,
retention of completed cases and uncertain reservations, no overwrite, independent
model pricing, and rejection of incomplete releases. Service tests cover concurrent
and killed workers, restart/rollover and failed accounting. These are mechanical
checks. Live performance and full claim review still require real frozen-input
captures and the accountable human reviewer.
