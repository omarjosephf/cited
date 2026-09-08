# Main-branch protection

`main-protection.json` is the branch-protection configuration this repository
requires, in the exact shape GitHub's REST API accepts. It exists so the control
is reviewable, diffable, and restorable, rather than living only in a settings
page nobody can audit.

Both CI jobs are required: `check` (lint, types, tests, Docker build) and
`secrets` (credential scan). Requiring only one would let the other regress
silently.

## Apply

```bash
gh api -X PUT repos/omarjosephf/cited/branches/main/protection \
  --input docs/operations/main-protection.json
```

## Verify

Applying is not evidence. Read the live configuration back:

```bash
gh api repos/omarjosephf/cited/branches/main/protection \
  --jq '{checks: [.required_status_checks.checks[].context],
         strict: .required_status_checks.strict,
         enforce_admins: .enforce_admins.enabled,
         force_push: .allow_force_pushes.enabled,
         deletions: .allow_deletions.enabled}'
```

A protected branch returns that object. An **unprotected** branch returns
`404 Branch not protected` — which is not an error to skim past, it is the
control being absent.

GitHub also supports rulesets, configured separately, which do not appear in the
endpoint above. A complete check reads both:

```bash
gh api repos/omarjosephf/cited/rules/branches/main
```

## Why `strict` and `enforce_admins` are both on

`strict: true` requires a branch to be up to date with `main` before merging. It
costs a re-run per merge, and it is worth that here specifically because of the
evaluation pin: `eval/portfolio-source.json` ties the corpus, prompt and question
set to one portfolio commit, and `verify_manifest` checks those digests agree.
Two changes that are each green in isolation can still produce an inconsistent
tuple. `strict` is what forces them to be tested together.

`enforce_admins: true` matters more in a single-maintainer repository, not less.
The only person able to bypass protection is the only person who commits, so
without it the control is advisory against the one account it needs to bind.

## Provenance

This configuration was not written from scratch. It already existed in the
working tree as uncommitted work, with the same required checks and the same
`strict` and `enforce_admins` choices — the decisions recorded above were already
made, and are preserved here rather than invented.

One key changed. The earlier version also carried `"contexts": []` alongside
`checks`, and the API accepts one or the other:

```text
422 Invalid request.
For 'anyOf/1', {"contexts" => [], "checks" => [...], "strict" => true} is not a null.
```

Removing that empty key is the whole of the difference, and it is what let the
configuration actually be applied. The same defect existed in the sibling
portfolio repository's copy, which suggests a shared template rather than a
one-off slip — worth knowing if this file is ever used as a starting point again.

## Why this file is worth keeping

Recording a configuration is not the same as applying one. The sibling portfolio
repository committed exactly such a file as a record of intent and never applied
it, while its `SECURITY.md` claimed the branch was protected; the branch in fact
had no protection and no rulesets at all, and every merge in that window was
unenforced. That file also could not have been applied as written — it set both
`contexts` and `checks` under `required_status_checks`, which the API rejects
with `422`.

This repository was found in the same unprotected state on 8 September 2026,
though it claimed nothing, so it was a missing control rather than a false
statement.

The lesson is the one the verify step encodes: a configuration file is a
statement of intent, and only reading back the live setting is evidence.
