# ADR-0009: Durable budgets and Gemini-primary routing

- Status: Authorized local implementation; production qualification pending
- Date: 2026-09-08
- Owner: OJ Florendo
- Risk: R2; provisioning and activation remain separate production actions
- Supersedes: ADR-0008's provider order and process-local budget description

The owner authorized overspending/abuse remediation, delegated the model choice
following a bounded comparison, and retained approval to publish verified-ready
work. Use Gemini 3.5 Flash-Lite as primary and GPT-5.6 Luna as the availability
fallback candidate. Adapter identity is independent of routing role; the router
assigns the role retained by wire v3 and the browser's backup disclosure.
Keep two serial attempts, zero retries, the 3/6/8/9/10-second nested time bounds,
and the same evidence, quote verification and whole-answer policy checks.
Billing, quota, authentication, malformed output and local budget failures never
trigger fallback. The separate existing Cited Anthropic demo is outside activation
scope. These changes do not prove live quality, latency or graduation.

Replace restart-resetting service counters with an explicitly initialized,
content-free SQLite reservation ledger on a persistent local volume. Commit a
fixed US$0.04 reservation before each attempt, including fallback, and never
refund it automatically. Combine daily/monthly attempts and reservation money;
current maxima are 40/200 attempts and US$0.40/US$2.00. The money limits therefore
admit at most 10 attempts per day and 50 per month. This is a reservation policy,
not an invoice guarantee: activation requires current rate and billed-token bound
verification, including thinking tokens and other account consumption.

Use one pinned Fly Machine, a real /data mount and one process-shared SQLite
admission lock. No ephemeral bootstrap, independently funded replica, expiring
lease, queue or reset endpoint is provided. Keep admission until the actual job
exits, even after caller cancellation. A missing/corrupt/replaced ledger, identity
or limits mismatch, clock regression or failed commit prevents paid dispatch.
Host/volume failure sacrifices assistant availability; the portfolio remains
usable. Do not provision a volume until its cost and activation authority are
resolved. Retain the ledger across deployment and rollback.

No prompts, answers, IPs or visitor identifiers enter the reservation ledger.
This is operational accounting, separate from the existing bounded browser tab
history and the conditional future owner-viewer roadmap. No transcript storage,
new visitor account, LLM judge or embedding-model migration is introduced.

Qualification captures require an additional persistent lifetime allowance with
explicit carry-forward and a finite call ceiling. They exercise actual bounded
jobs, retain per-attempt model/usage/uncertainty, and save after each case. An
interrupted capture is incomplete and cannot qualify release. Token estimates do
not settle reservations. Both frozen corpus suites still require full human
claim review; no agent can supply a human signature.

Rollback restores a compatible frontend/backend, corpus/prompt/policy, vectors,
configuration and image tuple. An old in-memory-counter image does not inherit
these controls. Keep answering disabled until the retained tuple and spending
boundary are verified. Existing release approval is conditional on readiness;
this ADR is not a deployment or purchase receipt.

See [runtime v3](../runbooks/assistant-runtime-v3.md) and
[durable budget operations](../runbooks/durable-budget.md).

## Shared qualification extension

[ADR-0010](../adr/0010-shared-qualification-admission.md) records the opt-in
PostgreSQL adapter and executed synthetic qualification. The current SQLite
factory/capture CLI and all actual old ledgers remain unchanged. Real migration,
serverless lifetime integration, recovery and production activation are pending.