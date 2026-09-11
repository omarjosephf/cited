# Assistant runtime and provider contract v2

For the owner-selected Luna/Gemini local candidate, see [runtime v3](assistant-runtime-v3.md).
The v2 single-provider and wire requirements below remain historical.

Status: local candidate, 7 September 2026. Publication remains subject to the
governing release process and owner approval.

## Request execution

The synchronous retrieval, provider call and validation run in a bounded thread
executor. One worker is the conservative default; configuration accepts 1–5.
There is no admission queue. Excess requests receive unavailable/503. A slot is
released only when its actual job exits, including after HTTP timeout or
disconnect. Cancellation prevents a later provider attempt and suppresses a late
answer. It does not prove a dispatched remote request stopped or was unbilled.

The service keeps health checks responsive while answer workers are occupied.
Startup warms a precomputed embedder before admitting questions. Shutdown closes
admission, waits for real jobs/warmup and closes the HTTP client. Python cannot
forcibly kill a stuck thread: a permanently stuck native operation requires the
process supervisor to stop the process. It never justifies admitting replacement
work into an already occupied slot. Synthetic 1/3/5-worker tests do not establish
shared-embedder performance at concurrency above the default.

## Ordered deadlines and resource bounds

| Boundary | Default / bound |
| --- | --- |
| Provider HTTP attempt | 6 seconds; zero SDK retries |
| Backend | 8 seconds from authenticated request-body entry |
| Portfolio proxy | 9 seconds including request and response bodies |
| Browser | 10 seconds including response-body consumption |
| Validation margin | 0.5 seconds reserved before provider dispatch |
| Raw request body | 8 KiB, before JSON parsing |
| Provider and backend response body | 48 KiB each, before JSON parsing |
| Generated answer / quote | 4,000 / 1,000 characters |
| Citations / source ID | 1–8 / 200 characters |
| History | Last four questions, 500 characters each; eight 80-character labels per turn |

The proxy overwrites X-Assistant-Deadline-Ms with its remaining integer budget.
The backend accepts only a finite positive value at most 9000 and subtracts
0.5 seconds; it can shorten, never lengthen, the configured backend budget.
Before dispatch, the provider timeout is reduced to the remaining backend time
minus the validation margin. Slow upload time therefore cannot grant a fresh
answer budget. Body limits can reject a history whose individually legal fields
still exceed the aggregate byte limit.

The provider adapter requests identity encoding, rejects compressed responses,
and bounds both successful and error response bodies before SDK parsing. It
checks elapsed time at headers and each body chunk, and closes rejected bodies.
Successful raw JSON is validated strictly against the installed native message
schema before the SDK can coerce citation indices or usage counts. Application
validation remains necessary because that schema allows additional fields.
The SDK's socket timeouts are not a guarantee that an OS/native operation can be
interrupted at an exact instant. The outer backend deadline suppresses late
results and the executor retains occupied capacity. This is not token streaming
or a second inference protocol.

## Typed results and evidence

A successful HTTP answer uses this discriminated result:
```json
{"version":2,"state":"answered","answer":"Bounded generated text","citations":[{"source_id":"project-cited.md","evidence_id":"64 lowercase hexadecimal characters","quote":"Exact supplied passage text"}]}
```

An unsupported or application policy result contains no model prose:
```json
{"version":2,"state":"not-covered","policy":"unsupported"}
```

Unavailable is `{"version":2,"state":"unavailable"}`. Non-success HTTP statuses
also map to the browser's fixed unavailable state. Legacy/unversioned service
responses fail closed. The portfolio maps recognized policy IDs to its
checked-in `content/assistant-policy.json`; backend answer fields cannot supply
policy text. The generic Cited demo uses its fixed document fallback.

Every non-whitespace generated text block must have valid citations before
flattening. Unknown content types, truncation, refusal, missing citations, any
invalid citation, invalid shape and failed policy suppress the whole answer.
There is no fragment salvage. Only application code may create fixed policy
responses; a model-supplied policy/state label is rejected.
Reproduction checks cover the generated prose and every displayed citation quote
together. Provider self-identification checks apply only to generated prose, so a
legitimate quotation does not falsely identify E.V as its answer-model provider.

Source IDs are exact corpus-relative paths, separate from labels and URLs.
Evidence IDs hash a versioned tuple of source path, page, section, supplied
chunk text and normalized quote. The backend recomputes containment and IDs.
The portfolio resolves paths through its corpus allowlist. Restored v2 citations
re-resolve labels/links from IDs; legacy stored citations retain the prior exact
label/link allowlist. Neither IDs nor quoted substrings prove semantic support.

**Known unresolved semantic boundary:** one text block can mix a valid quote with
an invented claim and still pass structural validation. The regression records
that runtime limitation and a failing/unknown semantic release result. Human
review of captured evaluation answers does not cover unseen visitor answers.
No lexical heuristic, model judge or live-quality guarantee has been added.

## Budget, metrics and privacy

Reserve one daily allowance immediately before a potentially billable attempt.
Policy/prefilter/retrieval failures before that boundary cost no allowance.
There is no refund API. Cancellation, timeout or exception after dispatch remains
counted; automatic retries and alternate providers are disabled. In-memory UTC
daily limits are per process and reset on restart, not account-wide spend caps.

Aggregate metrics separate request outcome, admission, queue, retrieval (including
query embedding), provider and validation duration. They report completed and
uncertain attempts, zero retries, known token subtotals and unknown-usage counts.
An unknown total or cost is never reported as measured zero. Attempt metrics
arrive when the actual job exits, which may be after the HTTP response.
Duration windows hold at most 2048 samples per stage and saturate at 60 seconds.
Model identity is the configured model, not an observed provider attestation;
corpus and effective prompt hashes are captured at startup.

Application logs contain fixed categories and bounded numeric metadata. SDK and
HTTP client log propagation is suppressed because verbose traces can include
request bodies and exception payloads. No questions, answers, passages, prompt,
credentials, headers, URLs, transcripts or per-visitor records are logged by the
application. Operational metrics are protected by the service secret. Browser
tab session storage remains the separate, disclosed P1 behavior.

## Candidate verification and rollback

Run the complete Cited tests, lint and type checks, plus portfolio component,
route and service tests. Controlled tests cover stalled health/admission,
timeout/disconnect, recovery, conservative attempt accounting, pre-body
authentication, body/history limits, oversized provider error bodies, fixed
fallbacks, source allowlists and sanitized logs. The final integrated portfolio
build/browser gate remains part of release preparation.

Bind the corpus checksum, actual question file, prompt, fixed policy map,
runtime/configuration source hashes, dependency/model locks and both repository
revisions in one candidate tuple. Uncommitted local files require individual
hashes alongside base HEAD; a base commit alone does not identify that candidate.
The evaluation capture also records runtime/transport/config hashes and nullable
usage. New corpus bytes invalidate old vectors and old answer-quality evidence.
Rebuild compatible vectors in the approved build workflow; never reuse stale
ones. The older pinned portfolio revision in Cited's CI remains historical until
an authorized committed candidate exists.

Roll back portfolio mapping, Cited API/runtime, prompt, policy, corpus and vectors
as one compatible tuple. A new frontend paired with an old service remains
unavailable. Do not make it accept legacy ungrounded prose to hide a mismatch.
An authorized operational disable/rollback is separate from preparing local
reversible files. Paid evaluation, live latency/quality evidence, graduation and
publication remain outstanding release decisions.
