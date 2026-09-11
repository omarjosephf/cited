# ADR-0005: One local read-only inspector for multiple corpora

- Status: Accepted
- Date: 2026-09-02
- Owner: OJ Florendo

## Context

The Week 4 instructor feedback asks for a RAG management panel that makes the
system understandable: the operator should be able to see the documents,
chunks and retrieval configuration rather than treating ingestion as a black
box. This repository powers both its own Cited demonstration corpus and the
exported OJ Assistant corpus. Maintaining two panels would duplicate the same
logic and make comparisons harder.

The word "management" can suggest uploads, document mutation, re-indexing,
provider calls or production administration. None of those capabilities is
needed to demonstrate corpus visibility, and each creates a materially larger
security and governance surface. In particular, this project's corpus remains
authored and reviewed rather than accepted from an unauthenticated browser.

## Decision

Build one **local, read-only** browser panel with a corpus selector. Its first
two profiles are Cited and OJ Assistant, and each profile keeps independent
counts, checksums, documents, chunks and optional vector-validation status.

The panel uses the production document enumeration, readers, chunker,
`Chunk.indexed_text()` and vector loader. It does not reimplement those rules,
load an embedding model, construct a provider client or answer questions.
Token counts are labelled estimates because the production chunker works in
words and FastEmbed does not expose a stable public tokenizer-counting API.

Corpus paths are fixed when the process starts. Browser requests select only a
generated corpus identifier; they cannot submit a filesystem path. Responses
contain corpus-relative source names and never expose the configured local
directory.

`doc-assistant inspect` binds Uvicorn to `127.0.0.1`. The application also
checks the HTTP Host value, exposes only GET routes, disables generated API
documentation, sets no-store/no-index and browser hardening headers, and uses a
per-response CSP nonce. Chunk text is inserted into the page as text, never as
HTML.

The selected corpus can be downloaded as a standalone HTML report. The report
is generated from the same immutable startup snapshot, uses corpus-relative
source names, escapes every corpus-derived string, contains no script or remote
asset, and is served with a download-safe filename and the same private browser
protections. It contains a compact preview of every chunk rather than creating
a second representation of retrieval logic.

The repository also provides a Windows command launcher. It resolves the
repository from its own location, uses the existing project virtual
environment, fixes `PYTHONPATH` to this checkout, opens only the loopback URL
after the panel responds and leaves the server visible so the operator can stop
it. It does not install dependencies or add a configurable host.

The default profile is `--corpus`, labelled Cited. Sibling
`deploy/*/content` directories are discovered as additional profiles so the
normal repository export layout presents both corpora without configuration.
Repeatable `--corpus-profile LABEL=PATH` options support isolated worktrees and
other explicit local layouts.

## Consequences

- The instructor can inspect and compare both real RAG corpora in one place.
- The panel is free to run: it makes no model or provider call.
- Document and chunk visibility is truthful because it shares production
  ingestion logic.
- Corpus selection does not combine the corpora. Retrieval boundaries remain
  independent and visible.
- Snapshots are computed at startup. A corpus change requires restarting the
  panel, which avoids mutable shared state and makes the displayed checksum
  stable for a session.
- A downloaded report is portable and printable, but it is also a persistent
  copy of the selected corpus inventory and previews that the operator must
  store and share appropriately.
- This is not a production admin console and must not be exposed publicly.
- Upload, edit, delete, embedding and deployment controls remain deliberately
  outside this decision and require their own risk review and approval.

## Alternatives considered

**A separate panel for every corpus.** Rejected because the controls and data
shape are identical; duplication would drift and makes comparison harder.

**Combine both corpora into one retrieval index.** Rejected because the user
asked to manage both, not erase their boundary. It would change answering and
evaluation behaviour rather than merely expose it.

**Allow the browser to enter arbitrary paths.** Rejected because it turns a
visibility tool into a local file-reading interface and risks path disclosure.

**Add uploads and re-index buttons now.** Deferred. That needs file-type and
size enforcement, malware and archive handling, authentication, audit records,
atomic index replacement, rollback and a separate approval boundary.

**Build a separate JavaScript application.** Rejected for this phase. A static
page served by the existing FastAPI dependency meets the need without another
package manager, dependency tree or build pipeline.

## Rollback

Stop the local process to remove all runtime effect. Reverting the `inspect`
command, inspector/report modules, static page, Windows launcher and related
documentation restores the previous repository behaviour. Any report already
downloaded must be removed separately by its owner if it is no longer wanted.
The answering API, corpus files, vectors and deployment configuration are not
changed by this decision.
