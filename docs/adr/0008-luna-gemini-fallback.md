# ADR-0008: Luna primary and Gemini availability fallback

Provider order and process-local accounting are superseded for the current
candidate by [ADR-0009](0009-durable-budget-and-provider-order.md).
Historical verification and approval statements below retain their dated scope.

- Status: Owner-approved model arrangement; local implementation unverified
- Date: 2026-09-07
- Owner: OJ Florendo
- Risk: R2
- Extends: ADR-0006 and ADR-0007

Use GPT-5.6 Luna (`gpt-5.6-luna`) as E.V's primary answering model and Gemini
3.5 Flash-Lite (`gemini-3.5-flash-lite`) as its backup for service
unavailability. This is the product runtime configuration, separate from Codex
agent routing. The owner selected the pair and resumed local implementation;
model comparisons are closed for this task.

Retrieve evidence once and share the identical evidence, question and bounded
question/source history with either provider. No generated answer history,
primary output or primary error payload is sent to the backup. Both adapters
must satisfy the same local quote, citation and policy controls. Citations in
this arrangement are generated references verified by application code, not
Anthropic-native citations. Structural provenance does not establish semantic
support for every claim.

Use at most two serial attempts in one admitted worker job. Primary receives up
to three seconds within a shared six-second provider phase. Backup uses only
the remainder after a positively classified availability failure. Neither
refusal nor malformed, uncited, truncated or policy-failing output permits
fallback. Missing credentials, billing/quota failures, exhausted local allowance,
cancellation and overall deadline expiry also prohibit fallback. Every actual
dispatch consumes a shared daily allowance and keeps its own usage outcome,
including uncertainty.

Wire v3 includes application-owned `model_route` on answered and not-covered
results. E.V shows "Backup model used" only for a returned fallback result,
including fixed policy replacements after fallback. Pre-provider policy results
have a null route. Unavailable results disclose no provider/error details.
Bounded existing tab records remain readable without inventing historical route
metadata. The accepted E.V presentation and minimal introduction remain intact.

The owner prohibited further tests and model calls while resuming implementation.
No test, build, lint, typecheck, benchmark, provider probe or new evaluation is
authorized by this ADR. This records unverified local code, not a release pass or
a measured quality, cost or latency advantage. Historical evidence stays intact.
Existing release approval does not establish the outstanding release evidence,
provider account suitability, spend controls or production configuration.

See [runtime v3](../runbooks/assistant-runtime-v3.md) for the paired contract and
remaining qualification work. Rollback must restore a compatible backend,
portfolio, policy, prompt, corpus, vectors and configuration together.

## Verification authorization update — 7 September 2026

The owner subsequently authorized local tests, builds and type checks, and the
specific npm dependency security audit. That supersedes the earlier local-test
pause recorded above. All live AI calls and model comparisons remain paused.
Passing offline checks can qualify their covered contracts; they do not supply
new live-answer evidence or establish provider account and spending controls.
