# ADR-0006: Reviewed evaluation and reproducible release inputs

- Status: Accepted by OJ Florendo on 2026-09-07; release evidence pending
- Date: 2026-09-07
- Owner: OJ Florendo
- Risk: R2; publication and production remain separate R3 actions

The owner approved the remaining decisions and release on 2026-09-07.
This accepts the decision and authorizes release through its gates; it does not
record a passing evaluation, completed protection change or deployment.

## Decision

Keep FastAPI, the answering model, local BGE retrieval and the existing API.
Change the evaluation and release process. A verified quotation establishes
quotation containment; it cannot establish support for the surrounding prose.
Evaluation v3 therefore separates mechanical checks from reviewed task success.
Unknown claim support is reported as unknown and blocks the release gate.

An offline review covers every answer character in order, records supporting
passages and an explanation for each claim, and binds to the full saved run.
It requires a human reviewer; machine validation checks completeness and quote
containment, not entailment or the honesty of the review. Owner inspection and
release approval remain necessary. A review file is not authentication.

The overall task threshold remains 95%; critical, safety and policy cases must
all pass. Unsupported claims, rejected citations, critical false refusals and
truncation must be zero. Retrieval requires 100% on demo and critical cases,
and the existing 75% broad portfolio floor. Historical v1/v2 artifacts remain
dated evidence under their original rules, not v3 quality passes.

CI and Docker install the same hashed dependency graph. Build requirements are
locked separately and the project installs without dependency resolution or
build isolation. Pin the Python base digest and model/tokenizer snapshot.
Keep the model offline after build. Add a dependency vulnerability gate and
run both corpus suites against a pinned portfolio revision. Container build
verification belongs in CI before release, not first on the production host.

The release manifest binds commits, image, corpus, prompt, dependency lock,
model files, vectors, configuration and both reviewed evaluation suites.
The verifier checks local artifacts and can compare independently collected
deployment identity; it neither deploys nor proves that an operator-supplied
observation came from production.

## Alternatives and consequences

More quotation heuristics cannot prove factual support. An uncalibrated LLM
judge would add expense and another uncertain judgement. Human claim review is
slower, but appropriate to these small release sets. A future calibrated judge
requires its own evidence and decision.

Updating dependency ranges on every CI run was rejected as release evidence:
the tested environment must match the built environment. Dependency updates
remain separate reviewed lock changes. `pypdf` moves from 6.14.2 to 6.16.1 to
close six findings reported by the new audit; see the
[upstream advisory](https://github.com/py-pdf/pypdf/security/advisories/GHSA-763m-79hh-57f2).

No visitor storage, owner account, model purchase, runtime result-contract
change or UI change is introduced. Evaluation artifacts contain synthetic
test questions and full answers; keep runs/reviews private by default.

## Rollback

Revert the build/CI implementation through a reviewed change and restore a
known-good compatible image, corpus, prompt and vector tuple when separately
authorized. Do not restore a false factual-quality label or promote old results
as a v3 pass. Live branch-protection settings are only prepared in this phase.

Related: ADR-0001/0002/0003/0004; [evaluation v3](../test-plans/evaluation-v3.md),
[builds](../runbooks/builds.md), [releases](../runbooks/release-evidence.md).
