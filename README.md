# Document Assistant

Ask a question about a set of documents. Get an answer **with the exact passage
it came from** — or an honest "that isn't covered in these documents."

Every claim about quality below is reproducible with one command:
`doc-assistant eval`.

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

Most demos of this kind answer confidently whether or not they should.

**It cites its sources, and the citations are verified.** Citations come from
the Anthropic API's native citations feature, computed against the documents
actually supplied — not from asking a model to "include the source" and hoping.
Then every quote is checked again here, against the passage we sent. A quote
that does not appear in it is discarded and counted. That check has never
fired, which is the point: it is how we would know if it stopped being true.

**It refuses, and refusal is measured.** When the documents do not contain the
answer, it says so. That is scored on every evaluation run, in both directions —
questions wrongly refused as well as questions wrongly answered.

**It is measured, not asserted.** A committed question set, scored by one
command, with the failing cases printed.

---

## Results

From `doc-assistant eval` on the committed question set (15 questions,
10 answerable, 5 not) against the demo corpus:

| Retrieval | |
| --- | --- |
| Hit rate (expected section in top-4) | **100%** |
| Top-1 (expected section ranked first) | **80%** |
| Score separation | **−0.119** |

| Answering | |
| --- | --- |
| Accuracy | **100%** |
| Unanswerable questions correctly refused | **100%** |
| Answerable questions wrongly refused | **0%** |
| Citations rejected as unverifiable | **0** |

Answering figures are five consecutive runs. Before the refusal marker was
introduced they oscillated between 93% and 100% — see
[ADR-0003](docs/adr/0003-refusal-marker.md) for why, since the instability was a
bug in the scoring rather than in the model.

**Scope, stated plainly:** this is a 15-question set against a 10-chunk corpus.
It is enough to catch regressions and to have found three real bugs. It is not
enough to claim the system generalises, and a bigger corpus is the obvious next
test.

---

## Technical decisions

Reasoning for each significant choice lives in [`docs/adr/`](docs/adr/).

| Decision | Choice | Why |
| --- | --- | --- |
| Embeddings | ONNX via `fastembed`, not PyTorch | 223 MB vs ~2 GB, verified working on Python 3.13/Windows *before* anything depended on it — [ADR-0001](docs/adr/0001-embeddings-without-pytorch.md) |
| Embedding location | Local, not a hosted API | No per-query cost on the retrieval path, no second vendor |
| Retrieval | numpy cosine similarity behind a `Retriever` interface | At this corpus size a vector database is complexity without benefit; the interface keeps the upgrade cheap |
| Query encoding | `bge` instruction prefix applied by hand | `fastembed`'s `query_embed()` is byte-identical to `embed()`, so it applies no prefix at all. Adding it measurably improved top-1 |
| Citations | Claude's native citations, then verified locally | The API cannot cite text it was not sent; the local check means the guarantee lives in this repository rather than in a vendor's feature list |
| Refusal | The model judges the retrieved passages | **Not** a similarity threshold — that was the plan until it was measured. [ADR-0002](docs/adr/0002-refusal-is-a-judgement-not-a-threshold.md) |
| Answering model | Claude Haiku 4.5 | Reading four short passages is comprehension, not reasoning. $2.20 per 1,000 questions against $11.00 for the largest model |

### The measurement that changed the design

A similarity threshold cannot separate answerable questions from unanswerable
ones on this corpus:

| | Top score |
| --- | --- |
| In-scope questions, lowest | 0.666 |
| Out-of-scope questions, highest | **0.755** |

*"How do I train my own language model?"* scores 0.755 — higher than seven of the
ten questions the corpus **can** answer — because it is topically adjacent to a
document about language models. The ranges overlap, so no cutoff exists that
separates them.

**Embedding similarity measures topical relatedness, not answerability.** That is
a property of the technique, not a threshold left untuned, so refusal comes from
the model reading the passages. A score threshold survives only as a cheap
pre-filter for the obviously unrelated (*"what is the capital of France?"* scores
0.428).

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[api,dev]"
```

On Windows, that one-time setup enables a one-click start: double-click
`Start-RAG-Management-Panel.cmd`. The launcher uses this repository's `.venv`,
opens the fixed loopback address in the default browser and keeps the server in
the visible command window until you press `Ctrl+C`. It does not install
packages, call an AI provider or expose the panel to the network.

Four of the five commands cost nothing and need no API key:

```bash
doc-assistant index               # what retrieval will see
doc-assistant index --verbose     # every chunk and its citation
doc-assistant eval                # retrieval only unless --paid is explicit
doc-assistant inspect             # local read-only management panel
```

The panel automatically offers the main Cited corpus and exported
`deploy/*/content` corpora such as OJ Assistant. For an isolated worktree or a
different local layout, select both explicitly:

```powershell
doc-assistant inspect `
  --corpus-profile "Cited=content" `
  --corpus-profile "OJ Assistant=C:\path\to\oj-assistant\content"
```

It shows documents, exact retrieval chunks, chunk/token ranges and retrieval
configuration. The selected corpus can also be downloaded as a standalone HTML
report for printing or saving as PDF. It makes no provider call and has no
upload, edit or delete route. See the
[local inspector runbook](docs/runbooks/corpus-inspector.md) and the
[instructor demonstration guide](docs/runbooks/instructor-demonstration.md).

For answering, copy `.env.example` to `.env` and add an Anthropic key:

```bash
doc-assistant ask "What are the four components of a well-formed prompt?"
doc-assistant eval --paid --max-paid-calls 15 --reason "owner-approved review"
```

The second command makes provider calls; `--paid` and a hard call ceiling are
deliberately required. Plain `doc-assistant eval` remains retrieval-only.

Then the web service:

```bash
uvicorn assistant.api:app --reload
```

Adding your own documents means dropping `.md`, `.txt`, `.docx` or `.pdf` files
into `content/`. Subdirectories are read, and citations use the path relative to
`content/` so two files with the same name stay distinguishable.

---

## Repository layout

```text
src/assistant/     ingestion, chunking, retrieval, answering, API and inspector
eval/              the committed question set
content/           the documents the demo answers from
docs/adr/          why each decision was made
Start-RAG-Management-Panel.cmd  one-click Windows launcher for the local panel
```

`src/assistant/api.py` decides nothing about answers — it owns transport,
protection and presentation only. The core is usable as a library on its own,
which is what keeps a different deployment a wrapper rather than a rewrite.

---

## If you deploy this

Read [SECURITY.md](SECURITY.md) first. Two controls are **not optional** and are
not enforced by this code:

1. **A hard spend cap on the API account.** The service has its own daily
   ceiling, but that counter resets when the process restarts. Only the
   provider's cap bounds the loss when something loops.
2. **Rate limiting at the edge**, in addition to the application's own.

Do not put confidential documents in the corpus of a public deployment. Every
passage is retrievable by asking the right question — that is what the software
does.

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
