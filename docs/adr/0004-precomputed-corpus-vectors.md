# ADR-0004: Embed the corpus at build time, not at every start

- Status: Proposed
- Date: 2026-08-30
- Owner: OJ Florendo

## Context

The portfolio deployment keeps one machine permanently awake. It was not meant
to. `fly.oj-assistant.toml` set `min_machines_running = 0` until 30 August 2026,
when a stopped machine took a **measured 154 seconds** to answer `/health` — and
the portfolio's route gives up after 20 seconds (`REQUEST_TIMEOUT_MS` in
`src/lib/assistant/service.ts`). The first visitor after an idle period did not
get a slow answer; they got "unavailable".

Keeping a machine warm made that invisible to visitors and was recorded, in the
config and in the launch runbook, as **a stopgap rather than a fix**: the real
fix is not needing three minutes of work before the service can answer anything.

The runbook's backlog described that fix as "precomputed vectors, or lexical
retrieval", either of which would "cut startup to about a second". Measuring
first showed that those two are not interchangeable, and that only one of them
is a fix.

## Evidence

**Where the cold start actually goes.** Development machine, portfolio corpus
(10 documents, 64 chunks). The right-hand column scales by the ratio between
this machine and the deployment implied by the 154s figure (~22x):

| Startup step | Measured | Implied on `shared-cpu-1x` |
| --- | --- | --- |
| read + chunk the corpus | 0.21s | ~5s |
| import `onnxruntime` | 0.37s | ~8s |
| load `bge-small-en-v1.5` | 1.03s | ~23s |
| **embed the 64 chunks** | **7.39s** | **~160s** |
| embed one query | 0.05s | ~1s |

**80% of a cold start is embedding the corpus, not loading the model.** That is
the opposite of the assumption the backlog entry was written under, and it
decides the design: the model load is small enough to hide, the indexing is not.

**Lexical retrieval was measured and rejected.** A BM25 retriever over the same
chunks, scored by this repository's own harness against the portfolio's
54-question evaluation set:

| Retriever | Hit rate | Top-1 | Critical core |
| --- | --- | --- | --- |
| BM25, heading + body | 90% | 72% | **92% — one miss** |
| BM25, body only | 79% | 59% | **92% — one miss** |
| `bge-small`, body only (shipped) | 97% | 74% | 100% |

Critical core must be 100%. Lexical retrieval fails the release gate on the
corpus it would have to serve, so "no model at all" is not available at this
quality bar. Its appeal was real — no ONNX in the image, a start measured in
milliseconds — and it is recorded here so the option is not re-proposed on
plausibility.

**Precomputed vectors are exactly equivalent to computing them.** Same model,
same strings, so this is expected rather than surprising — but it was checked
rather than assumed. Across all 54 questions the ranks are identical, and the
largest difference in any score is **0.0**.

**Binding the matrix to an order exposed a defect in how the corpus is read.**
The image's build printed a chunk digest matching neither corpus on the
development machine. `read_corpus` sorted `Path` objects, and Windows orders
paths case-insensitively where Linux does not — so
`OJ_Florendo_Rayatchi_Public_CV.pdf` is read sixth here and *first* in the
container, from identical bytes. Sorting instead on the corpus-relative POSIX
path, the key `corpus_checksum.file_digests` already used, makes the two
platforms agree; the development machine now reproduces the image's digest
exactly (`450c9338`). Retrieval is unaffected — the chunks are the same, only
their order differed — and the evaluation scores above are unchanged by the fix.

This was latent before this change and harmless: document order set
`Chunk.index` and nothing else, within a single process. It became a real defect
the moment a stored matrix was bound to that order, which is the general shape
worth remembering — persisting something derived turns an ordering nobody
guaranteed into a contract.

**Startup, measured before and after.** The indexing work itself falls from
8.37s to **0.20s**. End to end through the running application — the lifespan
entered, `/health` answering — a warm development machine goes from **4.74s to
1.00s**; the remainder is framework startup, which this change does not touch.
The model load moves to the first question (1.22s here, ~27s implied on the
deployment), which is why it is warmed in the background rather than left to a
visitor.

## Decision

**1. The corpus is embedded when the image is built, and the matrix ships in the
image.** `doc-assistant embed` writes an `.npz`; the Dockerfile runs it once per
corpus in the image; `CORPUS_VECTORS_FILE` tells a deployment which one to read.

Built in the image rather than committed beside each corpus, because a derived
file checked in next to its source is a file that can be forgotten. Building it
from the corpus that is *in the image* means the two cannot disagree by
omission.

**2. A stored matrix is bound to the text it was built from, and a mismatch is
fatal.** `vectors.py` records a SHA-256 digest of exactly the strings that were
embedded, in order, plus the model name and dimensions. Any disagreement stops
the process.

This is the whole risk of the change. A stale matrix is not a degraded service;
it is a service where row *i* is no longer chunk *i*, so every answer arrives
fluent, cited, and attributed to the wrong passage — the one failure this
project exists to prevent, and the one a reader cannot detect from outside.
`InMemoryRetriever` already refused a matrix of the wrong *height* for this
reason. The digest extends the same guard to its *contents*, which is the half a
row count cannot see. Falling back to embedding at startup was considered and
rejected: it turns a loud failure into a slow start nobody investigates.

**3. The model is loaded in the background once the service is already
serving.** Precomputing alone would move the cold start from startup onto the
first visitor rather than removing it. `warm_embedder` pays that cost on a
background thread; the platform sees a healthy machine in seconds, and the model
is ready well before anyone finishes typing. A warm-up failure is logged and
swallowed — the first question loads the model itself, and taking a process down
over a failed optimisation trades a slow answer for none.

**4. Retrieval searches the heading as well as the body.** `Chunk.indexed_text`
prepends `section` to `text`. This is not required by the change above; it is
adopted alongside it because the vectors are being rebuilt anyway, and because
in this corpus a heading is a sentence ("Where OJ works now: E-commerce and
Social Media Operations Lead at Golden Galore Luxury") that frequently carries
the question's own vocabulary.

| Corpus | Indexed text | Hit rate | Top-1 | Separation |
| --- | --- | --- | --- | --- |
| Portfolio (54 questions) | body only | 97% | 74% | -0.173 |
| Portfolio | **heading + body** | **100%** | **79%** | **-0.131** |
| Demo (15 questions) | body only | 100% | 80% | -0.119 |
| Demo | **heading + body** | **100%** | **70%** | **-0.001** |

Stated plainly: **this costs the demo corpus one question of ten at rank 1**,
while its hit rate — what actually decides whether the answering model sees the
right passage — is unchanged at 100%. The portfolio corpus, which is the one
being launched and gated, gains the question the old index missed ("Where does
he work now?", answered by a section whose heading says exactly that). A single
rule for both corpora is preferred to a per-deployment switch, because a switch
that changes what gets embedded is one more way for a matrix and a corpus to
disagree.

## Consequences

- Cold start falls from three to four minutes to seconds on the deployment.
  **To be confirmed against the real machine** — every deployment figure above
  is scaled from this one, not observed there.
- The image gains roughly 100 KB per corpus and a build step of a few seconds.
- **The model is still required.** A question has to be embedded, so
  `onnxruntime`, the model cache and the memory they need all remain. Anyone
  reading this as "the service no longer needs the model" will size the
  container wrong.
- `min_machines_running` and `memory` are **left exactly as they are** by this
  change. Both were set against measurements, and both should be revisited
  against new measurements from a machine running this code — not against the
  numbers above.
- The paid evaluation gate has to be re-run. Retrieval output changes, so the
  frozen criteria have to be measured rather than assumed to hold. The prefilter
  is unaffected: the lowest-scoring answerable question sits at 0.610 against a
  0.45 threshold. The run costs slightly more than the last one — 53 questions
  clear the prefilter where 49 did before.
- `read_corpus` now orders documents identically on every platform. The
  docstring already claimed it was "sorted for reproducibility"; it was sorted
  for local reproducibility only.
- `FastEmbedEmbedder` now loads under a lock. With precomputed vectors the first
  load is genuinely concurrent — a background warm-up against a visitor's first
  question — and unguarded it would build two models and discard one, after the
  memory had already been taken.

## Alternatives considered

**Lexical retrieval only (BM25).** Measured above. Fails the critical core.

**Precomputed vectors with no warm-up.** Simpler, and moves the model load onto
whoever asks the first question — roughly 27 seconds on the deployment, past the
caller's 20-second timeout. It would have left the visible symptom exactly as it
is today.

**Falling back to embedding at startup when the vectors do not match.** Rejected
under decision 2: it converts the failure this design exists to make loud into a
slow start with no message.

**Committing the vectors beside each corpus artifact.** Fewer moving parts in
the build, one more thing to remember on every corpus change. The digest would
catch a forgotten rebuild, but catching it at deploy time is worse than not
being able to forget.

## Rollback

Unset `CORPUS_VECTORS_FILE` and redeploy: the service embeds the corpus at
startup exactly as before, and the previous cold-start behaviour returns. That
path is the default and is covered by its own test, so it is a configuration
change rather than a code revert.

Reverting decision 4 — the heading in the indexed text — is a code change and
invalidates every stored matrix, which the digest reports as a mismatch on the
next start rather than serving stale vectors.
