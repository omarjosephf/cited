# Durable answering trust boundaries


## Durable-budget and provider-order candidate — 8 September 2026

The current candidate pins Gemini primary and Luna availability fallback.
Both receive the same public evidence and bounded questions/source history;
previous generated answer prose and primary error payloads are excluded. Both
outputs remain untrusted and undergo the same quote/policy checks. No tools,
redirects, hidden retry or billing/quota/auth fallback is permitted.

The paid boundary commits a persistent reservation before dispatch. Concurrent
processes share one content-free ledger and actual-worker lock on one local
persistent volume; production pins one Machine and verifies the real mount.
Missing/replaced/corrupt storage, changed identity/limits, clock reversal or a
failed accounting commit prevents more paid work. Cancellation retains the lock
until the real job finishes. Killing that job releases admission but keeps spend.
Independent replicas/volumes and automatic ledger recreation are prohibited.
A conservative operator-reviewed carry-forward is required for bootstrap or loss.

This adds no visitor transcript store: accounting holds dates, reservation rows,
ledger identity and limits only. CLI evaluation captures synthetic fixed questions
in private artifacts under a separate non-renewing allowance. They are not wired
into public requests. A new capture cannot overwrite prior evidence or reset the
aggregate authorization; interrupted runs remain incomplete.

Residual risks: single-host/volume failure makes the assistant unavailable;
SQLite cannot survive losing all copies, so recovery fails closed until conservative
accounting is established. A host compromise or operator bypass can alter files
and configuration. Timeouts cannot cancel all remote billing. Input/output caps
and current prices must establish all billed thinking/framing bounds before the
reservation amount can support a monetary ceiling claim. Account-wide spending,
taxes, hosting and egress also need separate review. Semantic claim support and
live latency still require real captures and human review. The separate Cited
Anthropic deployment must not be accidentally migrated or lose a shared key.
