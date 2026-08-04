# Document Assistant

Ask a question about a set of documents. Get an answer **with the exact passage
it came from** — or an honest "that isn't covered in these documents."

> **Status: in development.** Ingestion, chunking, embedding and retrieval are
> built and tested (60 tests, 100% coverage). Answering, the evaluation harness
> and the deployment are not. This README describes what exists; anything marked
> *planned* does not exist yet, and there is no command-line interface until M3.

---

## The problem

Microsoft's 2025 Work Trend Index found that employees are
["interrupted every two minutes during core work hours—275 times a day—by
meetings, emails, or chats"](https://www.microsoft.com/en-us/worklab/work-trend-index/breaking-down-infinite-workday)
(17 June 2025). Attention is the scarce resource, and every trip to go and find
something costs another slice of it.

The information usually is not missing. It is unfindable, and looking for it is
expensive precisely because the looking interrupts something else.

Ordinary search returns a list of documents and leaves you to read them. This
returns the answer, and shows you where it came from so you can check it.

> **A note on the statistic I did not use.** The figure normally quoted here is
> that workers spend around nine hours a week searching for information, costing
> a 1,000-person company $5M a year. Both come from an IDC paper published in
> **2001**, are almost always repeated without that date, and conflate two
> separate findings — the hours measure time spent searching, the $5M measures
> the cost of *duplicating* work. There is enough written about the provenance of
> that number to call it a myth. I could not find a recent primary source
> measuring the same thing, so I have not claimed one. A project whose entire
> premise is that you should be able to check a claim should not open with one
> you cannot.

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
| Retrieval | numpy cosine similarity behind a `Retriever` interface | At this corpus size a vector database is complexity without benefit; the interface keeps the upgrade cheap. **9/10 top-1 on the project corpus, 5.4 ms mean query** |
| Query encoding | `bge` instruction prefix applied by hand | `fastembed`'s `query_embed()` is byte-identical to `embed()`, so it applies no prefix at all. Adding it moved top-1 from 5/9 to 7/9 — measured, not assumed |
| Citations | *(planned — M3)* Claude's native citations | Structured source locations, rather than prompting for citations and hoping they are real |
| Refusal | *(planned — M3)* the model judges the retrieved passages | **Not** a similarity threshold. That was the plan until it was measured: out-of-scope questions do not reliably score lower, because embedding similarity measures topical relatedness rather than answerability. See the row below |

### The measurement that changed the design

A similarity threshold cannot separate answerable questions from unanswerable
ones on this corpus:

| | Top score |
| --- | --- |
| In-scope questions, lowest | 0.666 |
| Out-of-scope questions, highest | **0.755** |

*"How do I train my own language model?"* scores 0.755 — higher than seven of the
ten questions the corpus **can** answer — because it is topically adjacent to a
document about language models. The ranges overlap by 0.089, so no cutoff exists
that separates them.

This is a property of the technique rather than a threshold left untuned, so
refusal has to come from the model reading the retrieved passages and judging
whether they actually answer the question. A score threshold survives only as a
cheap pre-filter for the obviously unrelated (*"what is the capital of France?"*
scores 0.428).

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
