# Document Assistant

Ask a question about a set of documents. Get an answer **with the exact passage
it came from** — or an honest "that isn't covered in these documents."

> **Status: in development.** Milestone 0 (dependency spike) is complete and
> documented in [ADR-0001](docs/adr/0001-embeddings-without-pytorch.md). The
> ingestion, retrieval and answering layers are being built next. This README
> describes what exists; sections marked *planned* do not exist yet.

---

## The problem

Employees spend around **nine hours a week** looking for information that already
exists inside their own organisation. IDC put the cost to a 1,000-person business
at over **$5M a year**. The information is not missing — it is unfindable.

Ordinary search returns a list of documents and leaves you to read them. This
returns the answer, and shows you where it came from so you can check it.

## What makes this different from "chat with your PDF"

Most demos of this kind answer confidently whether or not they should. Three
things here are deliberate:

**It cites its sources.** Every answer points to the passage that supports it,
with the page number. You can verify the answer without trusting it.

**It refuses.** When retrieval finds nothing relevant enough, it says so instead
of inventing something plausible. A confident wrong answer is worse than no
answer, because you cannot tell it is wrong.

**It is measured, not asserted.** *(planned — M4)* A committed question set scores
retrieval hit rate, citation accuracy, and whether out-of-scope questions are
correctly refused. Claims about quality in this README will be backed by numbers
that anyone can reproduce with one command.

---

## Technical decisions

The reasoning behind each significant choice lives in [`docs/adr/`](docs/adr/).
Summary:

| Decision | Choice | Why |
| --- | --- | --- |
| Embeddings | ONNX via `fastembed`, not PyTorch | 223 MB vs ~2 GB, verified working on Python 3.13/Windows before anything depended on it — [ADR-0001](docs/adr/0001-embeddings-without-pytorch.md) |
| Embedding location | Local, not a hosted API | No per-query cost on the retrieval path, no second vendor |
| Retrieval | *(planned)* numpy cosine similarity behind a `Retriever` interface | At this corpus size a vector database is complexity without benefit; the interface keeps the upgrade cheap |
| Citations | *(planned)* Claude's native citations | Structured source locations, rather than prompting for citations and hoping they are real |
| Refusal | *(planned)* similarity threshold | Starting near 0.45, from the 0.69/0.32 separation measured in ADR-0001 — then tuned against the evaluation set |

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[api,dev]"

copy .env.example .env          # then add your Anthropic API key
```

*(Ingestion and query commands land with M1–M3.)*

---

## Repository layout

```text
src/assistant/     ingestion, chunking, embeddings, retrieval, answering
eval/              question set and scoring harness
content/           the documents the demo answers from
docs/adr/          why each decision was made
```

`src/assistant/` does not import from the API or UI layer. The core is usable as
a library on its own — which is what keeps a different deployment a wrapper
rather than a rewrite.

---

## For businesses

The demo answers from documents I wrote myself. If you are considering something
similar for your own organisation, two things are worth knowing up front:

- **Your documents stay yours.** Nothing here trains a model on your content.
- **Processing your documents makes me a data processor under UK GDPR**, which
  requires a Data Processing Agreement covering scope, retention, security,
  deletion and sub-processors — the language model provider is a sub-processor
  and has to be covered too. That is a solved problem, not an obstacle, but it is
  a real obligation and worth raising before any work starts rather than after.

---

## Licence

**AGPL-3.0** — see [LICENSE](LICENSE). You may use, study, modify and share it,
and derivative works must remain under the same licence, including over a
network.

If the AGPL does not suit your situation — for example you want to build on this
inside a closed-source product — a **commercial licence is available**. Get in
touch.

---

Built by OJ Florendo · [ojfr.me](https://ojfr.me)
