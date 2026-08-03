# Corpus

The documents the assistant answers from. Drop `.pdf`, `.docx`, `.txt` or `.md`
files here and run the ingest command.

This is **test data, not the project**. The system runs against any folder of
documents; what happens to be in here shapes the demo, not the architecture.

## What is in here

| Document | Subject |
| --- | --- |
| `prompt-engineering-fundamentals.md` | Practical prompt engineering for non-specialists |

### Provenance

`prompt-engineering-fundamentals.md` was drafted with AI assistance and then
reviewed line by line for factual accuracy by the author, who holds a
prompt-engineering certification from the Dubai Future Foundation and has
delivered AI and Python training professionally.

That review is not a formality. This project's entire claim is that an answer can
be traced to its source — so a corpus of unverified text would make every
citation technically correct and practically worthless, with the system
faithfully quoting a claim nobody ever checked. Every factual assertion here has
been confirmed by a human who is accountable for it.

## Why the corpus is small on purpose

It needs to be large enough that retrieval has to genuinely discriminate between
passages, and small enough that a reader can verify any answer against the source
in under a minute. A few thousand words meets both. Scaling to a larger corpus is
a capacity question rather than a correctness one, and it is answered by the
evaluation results, not by piling more documents in here.

## Structure matters more than length

Markdown headings and Word heading styles become the section names in citations,
so a well-headed document produces citations a reader can actually navigate to.
An unstructured wall of text retrieves worse and cites worse.

## What belongs here

**Only documents OJ wrote or owns outright.** For the public demo that means his
own training material — course outlines, programme summaries, session plans.

## What must never go here

**Client documents.** Not AAET's, not any other client's, regardless of how much
better a demo they would make.

This is not caution for its own sake. Anything in this directory is committed to
a public repository and served by a publicly reachable application. Putting a
client's material into either is a confidentiality breach and, where it contains
personal data, a UK GDPR one — processing a client's documents makes the operator
a **data processor**, which requires a Data Processing Agreement covering scope,
retention, security, deletion and sub-processors.

The tool is *built for* that use case. It is *demonstrated on* material we own.
Those are different things, and the separation is deliberate.

Third-party documents are also out unless their licence clearly permits
redistribution — course material from a university or a training provider is
usually copyrighted even when it was freely given to a student.
