# ADR-0007: Bounded answering and a safe v2 transport

- Status: Accepted by OJ Florendo on 2026-09-07; release qualification pending
- Date: 2026-09-07
- Owner: OJ Florendo

Slow synchronous retrieval/provider work must not occupy the async event loop.
Uncertain provider failures must not replenish a paid-call allowance. Unsupported
model prose must not be presented as an application policy response.

Use a bounded executor whose slots belong to actual jobs, monotonic nested
deadlines, zero SDK retries and a reservation immediately before provider
dispatch. Admit one answer by default and reject excess work. Validate native
citations per generated text block before flattening; any unsafe block rejects
the entire response. Export typed v2 results with stable source/evidence IDs.

This preserves the provider, embedding model, retrieval architecture and fixed
corpus. It adds no persistence or inference judge. A same-block mixed invention
remains a semantic limitation and cannot qualify a release merely by containing
a valid quote. Evaluation v3 still requires captured-answer human review.

See [runtime/provider contract](../runbooks/assistant-runtime-v2.md) for exact
bounds, lifecycle, metrics, verification and compatible rollback. This extends
ADR-0002/0003's refusal behavior and ADR-0006's evidence requirements.
