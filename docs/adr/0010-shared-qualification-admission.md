# ADR-0010: Shared qualification admission candidate

- Status: Authorized local/staging qualification; real migration and activation pending
- Date: 2026-09-09
- Owner: OJ Florendo
- Risk: R2
- Extends: [ADR-0009](0009-durable-budget-and-provider-order.md)

## Decision

Add opt-in `assistant.shared_budget` for E.V and Cited. The companion portfolio
migration `202609090006_ev_shared_budget.sql` owns one PostgreSQL aggregate row,
two explicitly bound service policies, durable non-expiring jobs and permanent
US$0.04 attempt reservations. Atomic row-locked RPCs enforce service daily/monthly
and service/aggregate nonrenewing caps. A complete policy, ledger IDs and a reviewed
carry-forward receipt digest are checked on every operation.

No startup provisioning, retry grant, refund, expiry or reset exists. The existing
SQLite service factory and capture CLI remain authoritative and disconnected from
this adapter. Full application/host integration and activation remain pending.

## Alternatives, security and operational impact

A pinned persistent host can retain SQLite. Independent instance ledgers cannot
share the aggregate; expiring leases can admit replacement work too early.
Durable job rows deliberately sacrifice availability after process death/freeze.
Only an actual executor completion callback releases a job; caller cancellation
does not. Provider-side work may continue remotely after an HTTP timeout, so all
debits remain and the two-attempt bound applies.

Use a server-only HTTPS RPC credential, explicit managed origin, bounded response,
no retries/redirects and sanitized failures. Tables are private with RLS and
revoked direct privileges. No prompt/answer/IP/visitor identity is stored.
Missing/corrupt/mismatched/uncertain admission stops new dispatch.

## Migration, rollback and qualification

Real carry-forward remains a separate review. Preserve every existing paid
ledger and uncertain attempt; unknown service history is not zero. A rollback
or database restore must include all subsequent debits. An internally consistent
stale snapshot needs external latest-accounting evidence; counters cannot detect it.

The phase passes 788 pytest checks (seven historical skips), complete lint/format/
type checks, 90 companion local SQL checks and 11 actual staging checks across
53 requests. Both function bodies match the candidate. All synthetic pools are
disabled; zero real allowances/provider calls were created. One earlier staging
acknowledgment failed closed after job commit and is retained in failure history.
This is a bounded schema/adapter checkpoint, not hosting or release completion.

See the companion portfolio's ADR-0018, shared-budget runbook and
`docs/reviews/ev-shared-budget-qualification.md` for detailed evidence and limits.
