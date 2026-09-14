# Release evidence and branch protection

The [manifest schema](../schemas/release-manifest-v3.schema.json) defines a
versioned release tuple: frontend/backend commits, backend image digest, Python
base digest, dependency graph, corpus, prompt, model/tokenizer lock, vector
artifact, evaluation/review/policy code, model settings and both corpus suites.
All artifact references are SHA-256 plus a relative path under a private release
directory. Never put secrets or visitor transcripts in the release bundle.

Build the backend from the approved clean commit. Export the portfolio corpus
using its existing exporter. Record actual file hashes, exact commits and image
digest; do not fill missing fields with guessed values or hashes of placeholders.
Keep the two corpora and their evidence separate. Run the approved evaluations,
complete human claim review and retain the full run/review artifacts.

```sh
python -m assistant.release_manifest release/manifest.json --artifacts release
```

The checker verifies files, corpus/vector compatibility, distinct suites, input
identities and the reviewed gates. Missing/stale evidence fails. This is a
preparation check; it neither approves publication nor asserts that an image is
running. Verify clean Git state, exact CI results, image provenance and rollback
under the owner's standing approval to publish verified-ready changes.

After authorized deployment, independently collect frontend revision, backend
image and complete artifact identity into an observed manifest, then compare:

```sh
python -m assistant.release_manifest release/manifest.json --artifacts release --observed release/observed.json
```

Do not copy the intended manifest into `observed.json` and call it observation.
Current short `/health` corpus prefixes alone cannot establish the full tuple.
Production identity collection and corpus reconciliation require their own
authorized release; this phase does not change the live health response.

Rollback restores a known-good compatible image/corpus/prompt/vector tuple.
Frontend rollback alone does not roll back the separate backend. Retain the
previous private manifest and artifacts until the replacement is verified.

## Prepared main protection

`docs/operations/main-protection.json` is a reviewable REST update body, not an
applied setting. It requires the existing `check` and `secrets` checks from the
GitHub Actions app, up-to-date branches, PRs and resolved discussions; it blocks
force pushes/deletion and applies to administrators. Check names/app IDs were
read from the audited revision on 7 September 2026.

The approving-review count is zero for this solo-owner repository: GitHub does
not let the sole author approve their own PR. The owner still inspects the
change and makes the final merge decision. If another human reviewer is added,
raise the count to one in a separately reviewed settings change. Do not pretend
an impossible one-person approval rule provides working review enforcement.

Before applying, inspect current protection/rulesets and repository permissions,
save a private backup, confirm required checks exist on the candidate, and get
the owner's settings approval. After applying, inspect effective rules and
verify that a controlled failing check cannot merge, without bypassing the rule.
Restore the captured settings if the change blocks the intended workflow.
No settings command is executed automatically by this repository.

Reference: [GitHub branch protection API](https://docs.github.com/en/rest/branches/branch-protection).


## Runtime v2 local candidates

Follow [runtime/provider contract](assistant-runtime-v2.md). Preserve both base
revisions plus hashes for uncommitted prompt, policy, corpus, question set and
runtime/configuration changes. Verify the Python policy map against the
portfolio JSON map. A freshly exported corpus requires matching vectors; never
reuse the older artifact by filename alone. Old evaluation runs retain their
original meaning and cannot qualify changed bytes. No local hash record is a
substitute for the authorized committed/image/reviewed release manifest.
## Routed-runtime evidence

Manifest v3 binds answer contract v3, the candidate answer-runtime source hashes,
and the complete approved behavior configuration to both reviewed suites. Missing
or extra runtime source roles, changed source bytes, different provider settings
or stale answer contracts fail verification. Manifest v1, v2 and earlier captures
remain historical evidence; they cannot qualify the Luna/Gemini candidate.

v3 drops the four spend ceilings from `answer_configuration`, on 14 September
2026. They never changed an answer, and keeping them there made one manifest
field carry two values that cannot be equal: the capture runs against the
capture-scoped ledger ADR-0015 authorises, so it can only record 150 attempts
and US$6.00, while the deployment that evidence qualifies runs at 40 and
US$0.40. The deployed envelope is asserted directly against
`fly.oj-assistant.toml` instead, by a test that reads the file that is actually
deployed. Superseded schemas stay on disk and are referenced by nothing, so an
old manifest stays readable without ever looking current.

Local retrieval, mocks and regenerated vectors are permitted under the owner's
verification approval. Paid capture is opt-in and additionally requires an
existing `--allowance-ledger`, matching `--allowance-id`, positive
`--max-paid-calls`, `--output` and `--spec-version 3.0`, plus durable service
accounting and the complete verified provider pair. The conservative preflight
requires room for two attempts per case across all ceilings before starting.
A configured credential alone never triggers inference.

An operator must explicitly initialize the qualification allowance with the
approved aggregate ceiling and all carried-forward reservations. It has no time
rollover or refund path. A new task or run must reuse it. Do not initialize a
replacement from zero to regain authority. Full usage estimates use each model's
uncached rates and remain unknown when an earlier attempt's usage is unknown;
they never authorize settlement. Reverify dated rates and billed-token bounds
before live work. Keep allowance IDs, runs and account receipts private.

The capture writes exclusive output before any dispatch, then durably replaces
it after each case. It records actual bounded runtime attempts and durations;
no synthetic response is live qualification. Interrupted files retain completed
cases, the active index and known attempt state; `capture_state: incomplete`
blocks the release manifest. Never overwrite/retry them automatically. Inspect
uncertainty and reconcile authority before planning any rerun. A `.pending`
file indicates interrupted persistence and must be preserved for inspection.

Generate a blank claim-review template from a completed run and have the human
reviewer assess every answer character against supplied evidence. Missing human
review remains a release blocker, even if structural checks pass. The CLI
returns a failing quality gate until that separate review is complete.

File hashes do not prove what is running inside a deployed image. Independently
verify image provenance and observed identity in addition to the artifact checks.
