# ADR-0001: Generate embeddings with ONNX, not PyTorch

- Status: Accepted
- Date: 2026-08-03
- Owner: OJ Florendo

## Context

Retrieval needs an embedding model: something that turns a passage of text into a
vector so that "which passages are about this question?" becomes a distance
calculation.

The conventional choice is `sentence-transformers`, which depends on **PyTorch**.
That is roughly a 2 GB install, it is historically awkward on Windows, and this
project targets **Python 3.13**, which was new enough at the time of writing that
machine-learning wheels could not be assumed to exist for it.

Two risks followed. First, the development machine is Windows on Python 3.13 — if
the stack would not install there, the project could not be built at all. Second,
the deployment target is a container on a free or cheap tier, where a 2 GB
dependency is a real constraint rather than an inconvenience.

The decision therefore had to be made **before** any code depended on it. A
dependency chosen and then discovered unusable at deployment would invalidate
every module built on top of it.

## Decision

Use **`fastembed`**, which runs quantised models through **ONNX Runtime** and does
not depend on PyTorch. Default model `BAAI/bge-small-en-v1.5` (384 dimensions).

Embeddings are generated **locally**, not through a hosted API. The corpus is
embedded once at ingestion and a question is embedded per request; doing that
locally means retrieval costs nothing per query and adds no second vendor.

## Evidence

Verified by spike before writing any project code, on the actual target machine
(Windows, Python 3.13.3):

| Measure | Result |
| --- | --- |
| Install | Clean. `fastembed 0.8.0`, `onnxruntime 1.28.0`, `numpy 2.5.1` |
| PyTorch present | **No** |
| Install footprint | 159 MB venv + 64 MB model cache ≈ **223 MB** |
| Model load | 6.0s first run including download; cached thereafter |
| Embedding throughput | 4 texts in 0.04s |

Installing is not the same as working, so the spike also asserted that the
resulting geometry is *meaningful*. Against the question "What programming topics
does the training cover?":

| Passage | Cosine |
| --- | --- |
| Python fundamentals: variables, loops, functions | **0.6946** |
| Exploratory data analysis with pandas and Excel | 0.5864 |
| Refunds are issued within 14 days | 0.3162 |

The ranking is correct and the separation is wide. A vector of the right shape
would have proved nothing; a correct ordering is the actual requirement.

That 0.69 / 0.32 gap is also the first evidence for the refusal threshold — an
unrelated passage sits far enough below a relevant one that a cut-off near 0.45
is plausible. It is a starting point to be tuned against the evaluation set
(ADR to follow), not a final value.

## Alternatives considered

**`sentence-transformers` (PyTorch).** The default choice and the
best-documented. Rejected on footprint: ~2 GB against 223 MB, with no measured
retrieval-quality benefit at this corpus size, and a materially higher chance of
failing on Python 3.13 / Windows. Remains the fallback if `fastembed` is
outgrown.

**A hosted embeddings API.** Removes the local dependency entirely and typically
improves quality. Rejected because it adds a second vendor, a second key to
manage, and a per-query cost on the retrieval path — which would also be charged
on every evaluation run. Worth revisiting if retrieval quality becomes the
limiting factor.

**A larger local model.** `bge-base` and above score better on benchmarks. Not
justified before the evaluation harness exists to measure whether retrieval is
actually the weak link. Deferring this is the point of building the harness.

## Consequences

- No GPU required, and no CUDA in the deployment image.
- The container carries the model, so the image is ~223 MB rather than ~2.5 GB.
- Model load costs ~6s on cold start; the API must load once at startup, not per
  request.
- 384 dimensions is small, which keeps a numpy-based similarity search viable far
  longer than a dedicated vector database would be needed.
- If evaluation later shows retrieval is the bottleneck, the model is swappable
  behind the same interface — the alternatives above stay open.

## Rollback

Change one dependency and one module. Nothing outside `embeddings.py` knows which
library produced a vector, and the `Retriever` interface consumes vectors without
knowing their origin.
