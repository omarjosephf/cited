# E.V runtime contract v3

Status: local candidate; current source and release checks are recorded separately. Extends the resource, privacy and
lifecycle boundaries in [runtime v2](assistant-runtime-v2.md); supersedes its
single-provider and wire-v2 requirements for this candidate.

## Provider routing

Primary is `gemini-3.5-flash-lite` through Google GenerateContent with its
minimal default thinking configuration. Backup is `gpt-5.6-luna` through OpenAI
Responses, with `store: false`, no tools and reasoning effort fixed to `none`.
Adapter model identity is separate from routing role, which the router assigns.
The selected order follows the owner's delegated model decision. Neither a
small comparison nor mocked tests establish production quality or latency.

One retrieval snapshot and the same current question and last four bounded
questions/source-label sets are used for both. Previous generated prose and
primary responses/errors never enter the backup request. Each adapter requests
structured answers and references; application code verifies every supplied
quote/source and applies the same whole-answer policy and suppression rules.
A citation match alone does not establish semantic entailment.

There are at most two serial attempts, with zero retries and no hedging.
The provider phase ends at the earlier of six seconds after provider work starts
and the backend deadline minus 0.5 seconds. Primary receives at most three
seconds; backup receives only the remaining provider-phase time. The existing
eight-second backend, nine-second proxy and ten-second browser budgets remain.
Cancellation suppresses late display; it does not prove remote work stopped.

Fallback requires an explicitly classified primary availability failure:
transient connection interruption/provider timeout, HTTP 408/500/502/503/504, or
a bounded 429 error code identifying temporary rate/capacity exhaustion.
Unclassified 429, quota/billing problems, missing/invalid credentials,
configuration/TLS errors, other HTTP errors, local admission/budget exhaustion,
cancellation and overall deadline expiry do not qualify. Neither do safety
refusal, malformed or oversized responses, truncation, failed citations or
failed policy. A missing primary key must not silently enable backup-only use.
Runtime remains inactive until explicit configuration enables the complete pair.

Each dispatch commits a fixed reservation to the same persistent daily/monthly
ledger. See [durable spending and admission](durable-budget.md) for exact caps,
single-Machine topology, failure behavior and activation requirements. Neither
provider completion nor unknown usage refunds a reservation. A separate shared
lock keeps one actual worker admitted across processes until all work returns.

Metrics retain each configured role/model and distinguish known subtotals from
unknown counts. Gemini total-minus-prompt usage can establish complete output
usage including thinking; missing or inconsistent totals remain unknown. An
unknown primary stays unknown even if backup succeeds. Neither per-provider
estimated usage nor cancellation establishes an invoice or remote termination.

## Portfolio transport

Only wire version 3 is accepted. Answered results contain exactly `version`,
`state`, `answer`, `citations` and `model_route`. Citation fields remain
`source_id`, `evidence_id` and `quote`. `model_route` is `primary` or
`fallback`, assigned by application code after handling that provider.

Not-covered results contain exactly `version`, `state`, `policy` and
`model_route`. Their route is null when no provider was used; otherwise it
retains the provider route through fixed-policy replacement or suppression.
The portfolio maps policy IDs to application-owned text. It retains valid
fallback metadata when rejecting an unsafe answered payload into fixed copy.
Unknown routes and incompatible envelopes fail closed.

Unavailable contains only `version: 3` and `state: "unavailable"`. No exception,
key/configuration state, provider payload or retry reason crosses this boundary.
Source labels and links still come from the exact corpus allowlist.

The browser receives optional camelCase `modelRoute` on answered/not-covered
results and shows a small "Backup model used" label for fallback. Screen readers
receive the same disclosure. The label survives bounded session storage;
older records keep absent metadata and are never retrospectively attributed
to the new primary. Generated answer text is still excluded from provider
history. The intro and accepted palette/fonts/layout are unchanged.

## Delivery status and qualification

The 8 September continuation supersedes the earlier local-test pause and provider
order. Offline qualification of the current files is in progress; earlier dated
results qualify only their exact source. Local tests and retrieval are authorized.
Live inference must fit the existing private, carried-forward spending allowance;
no task, process or ledger may silently grant a fresh allowance.

Generate matching corpus/vectors and bind prompt, policy, questions, runtime,
evaluation code, dependency and model hashes. Paid captures require both service
accounting and a non-renewing qualification ledger, preserve each attempt and
interrupted run, and need complete human claim review. Incomplete/stale evidence
fails the manifest. Verify Linux image provenance, mounted-volume restart behavior,
provider accounts, billed-token bounds and actual deployment identity before
activation. No credential, purchase or deployment is performed by these sources.

Rollback is paired across Cited and the portfolio, including prompt, policy,
corpus, vectors and provider configuration. A version mismatch stays unavailable.
Retain release approval and outstanding evidence separately. Keep Beta until the
actual graduation criteria and required owner review pass.
