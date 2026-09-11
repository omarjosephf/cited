# Durable E.V spending and admission

Status: local release candidate. Provisioning, provider activation and release
qualification are separate from implementation. See [ADR-0009](../adr/0009-durable-budget-and-provider-order.md).

E.V uses Gemini 3.5 Flash-Lite as primary and GPT-5.6 Luna as an
availability-only backup. Both share one content-free SQLite reservation ledger
and one actual-worker admission lock. The separate existing Cited demo remains
an Anthropic deployment; this release must not activate these settings there or
revoke a credential shared with it.

## Enforced application limits

Each dispatch commits 40,000 micro-USD (US$0.04) before any network request.
No completion, error, timeout, cancellation or incomplete usage report refunds
that reservation. There is no automatic settlement or reset endpoint. The
32,000-byte request and 1,024 requested output-token limits, text-only payloads,
no tools, zero retries and at most two serial attempts constrain the paid path.
A fallback has its own reservation and only uses remaining provider time.

The maximum configured envelope is 40 attempts per UTC day, 200 per UTC month,
US$0.40 per UTC day and US$2 per UTC month across both providers. Whichever
limit is reached first wins. Full reservations therefore admit at most 10
attempts/day and 50/month; one answer using backup consumes two. These defaults
prioritize loss containment and may stop service well before actual invoices
reach US$2. Raising any maximum needs an explicit reviewed budget decision.

These are **application reservation caps**, not guarantees about hosting,
taxes, provider pricing changes, other uses of the accounts, or provider defects.
Before activation, verify the conservative per-attempt price bound against the
actual provider configuration, including all thinking tokens. Missing thinking
breakdowns are not measured zero. Keep the verified provider account limits and
headroom. A timeout does not prove the remote request stopped or was free.

## Storage and topology

Runtime opens two existing files with SQLite `mode=rw`; it never bootstraps a
new allowance. Both files carry an exact ledger ID, schema version and limits.
Startup checks integrity. Every reservation verifies the identity and limits,
uses a `BEGIN IMMEDIATE` transaction and commits with `synchronous=FULL`.
Replacing or removing either file stops new reservations. Invalid accounting,
clock regression, lock contention and storage failures fail closed. UTC month
rollover cannot discard the earlier rows. No question, answer, IP, credential,
source text or visitor identifier is stored.

Use a single Fly Machine and one 1 GB persistent local volume mounted at
`/data`. Pin the exact Machine ID with `BUDGET_MACHINE_ID`; a replica with a
different ID cannot start answering. The service verifies the actual mount and
will not accept a SQLite file on ephemeral root storage. SQLite coordinates
multiple processes sharing these same local files. Independent volumes are not
a shared budget and are prohibited by this topology.

A second SQLite file is used only for the global worker lock. Its transaction
lasts until the actual job finishes, even when an HTTP caller times out or
cancels. Process death releases the lock but preserves committed reservations.
There is no expiring lease that could overlap a still-running provider thread.

This trades high availability for bounded cost. Volume/host failure makes E.V
unavailable; the main website and contact route remain usable. Do not clone the
volume or use rolling replicas as a workaround. Keep one Uvicorn worker and
`--ha=false` on approved deployments.

## Proposed operating envelope

A 1 GB Fly volume is US$0.15/month at the published price checked 8 September
2026; automatic snapshots charge by stored bytes, with the first 10 GB free
organization-wide. Verify the current VM, region, snapshots, egress, taxes,
other deployments and provider-account use against the owner's private budget
before activation. No volume or other paid service is created by this change.

The application enforces reserved money. Before activation, verify that the
32,000-byte request (including framing) and 1,024 output-token settings also
bound every billed component, including Gemini thinking, at current rates.
Until that is established, do not describe US$0.04 as a proven worst-case invoice
cost. Usage estimates and request timeouts cannot establish final charges.

## Bootstrap, replacement and rollback

1. Qualify exact source, complete checks, compatible wire-v3 frontend/backend,
   exported corpus/vectors, reviewed captures and image identity. Obtain the
   separately required volume/activation authority after the concrete plan is
   ready. Preserve the known-good image and frontend SHA.
2. Verify only one E.V Machine, the intended mounted volume, and owner-reviewed
   project/account configuration. Do not change `cited-demo`.
3. With answering disabled, initialize a new ledger **only after reconciling
   prior spending for the active month**. Generate a fresh 64-character hex
   identity, record it privately and run
   `python -m assistant.persistent_budget --path /data/budget.sqlite3 --ledger-id <id> --carry-forward-attempts <reviewed-count>`.
   Carry forward all prior reservations/spend rounded up conservatively; the
   bootstrap charges them to today and the current month. Only use zero when
   the reviewed prior allowance is genuinely unused.
   Both files and `/data` must be writable by container UID 1000. This command
   makes no provider call and refuses any existing/partial ledger.
4. Configure the existing path, ledger identity and exact Machine ID. Verify
   account caps, both provider keys/project IDs and explicit account assertions;
   only then enable the pair through `ENABLE_FALLBACK`. Never set assertions
   merely to make startup pass. Recheck counters across a restart before release.
5. Retain the volume and ledger across every deployment. Do not delete, truncate,
   recreate, clone or restore an older ledger to regain an allowance. A lost
   ledger blocks answering. Restoration needs an operator-reviewed conservative
   carry-forward of all reservations since the snapshot, including uncertainty;
   if that cannot be established, keep answering disabled through the current
   month. No automated reset is provided.
6. If rollback is needed, disable answering and restore a compatible complete
   source/image/configuration tuple. Retain the ledger as evidence. An older
   image with in-memory counters does not inherit these safeguards; do not
   silently reactivate it as though the budget remained durable.

## Verification

Offline tests cover restart persistence, concurrent instances, shared worker
admission, process death, money/attempt exhaustion, UTC rollover, clock reversal,
identity/config mismatch, corrupt accounting and missing storage. Runtime tests
cover retained slots after cancellation, fallback charging and no fallback for
billing/quota/authentication, malformed output or local accounting failure.
These checks do not substitute for a mounted-volume restart check or live answer
and human claim review.

Sources: [Fly pricing](https://fly.io/docs/about/pricing/),
[Fly volumes](https://fly.io/docs/volumes/overview/),
[Fly runtime identity](https://fly.io/docs/machines/runtime-environment/),
[SQLite atomic commit](https://www.sqlite.org/atomiccommit.html),
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing),
[Gemini thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking).

## Shared qualification extension

[ADR-0010](../adr/0010-shared-qualification-admission.md) records the opt-in
PostgreSQL adapter and executed synthetic qualification. The current SQLite
factory/capture CLI and all actual old ledgers remain unchanged. Real migration,
serverless lifetime integration, recovery and production activation are pending.

## Existing-host release preparation, 9 September 2026

The owner selected the existing hosting and deferred optional consolidation.
Keep the integrated SQLite service factory and single-Machine worker lock;
shared PostgreSQL application activation is not a prerequisite for this topology.
The separate qualification budget and its aggregate authority still apply to
offline paid captures. No service maximum or prior debit is changed.

The candidate `fly.oj-assistant.toml` now names the ledger path, one `ev_budget`
mount, 1 GB initial size, daily/monthly reservation caps, one worker and retained
five-day snapshots. It preserves the existing 1536 MB VM and warm-machine choice.
Private ledger/Machine identities and provider activation remain prerequisites.
The public config alone fails startup validation; it cannot create an allowance.

Read-only Fly inspection found one started London E.V Machine and zero volumes.
No app, Machine, secret or volume was changed. A production cutover must therefore
include separately approved storage, a reviewed ledger carry-forward and an
actual mounted-volume restart check before enabling either provider.

Use a deliberate maintenance cutover on the sole serving Machine: preserve its
current image/configuration, stop paid intake, establish the approved volume and
its actual placement/attachment, make `/data` writable by UID 1000, initialize
only the approved new ledger with conservative carry, and pin the actual resulting
Machine identity. If volume placement requires a replacement Machine, keep it
unable to answer until the old intake is stopped and the binding is reviewed;
never silently create a funded replica. Do not bypass startup to obtain traffic.
The corpus/frontend/backend image tuple still requires final release review.

Do not use a Fly `release_command` for ledger initialization: it runs on a
temporary Machine without volumes. A regular rolling deployment may be used
only after the one-volume topology is established and checked. Budget identity
changes fail closed. Preserve all latest debits through rollback and recovery;
an older image with process-local accounting must not resume paid traffic.

Current published volume pricing is US$0.15/GB-month, billed while unattached or
stopped too. Snapshots are US$0.08/GB-month after the organization's first 10 GB;
the free amount is shared, not assumed available. Those are incremental storage
prices, not a complete hosting/account estimate or a new spending authorization.

Sources checked 9 September 2026:
[Fly pricing](https://fly.io/docs/about/pricing/),
[volume attachment](https://fly.io/docs/launch/volume-storage/) and
[deployment configuration](https://fly.io/docs/reference/configuration/).
