# Demonstration: Cited RAG Management Panel

## Objective

Demonstrate, in about four minutes, that Cited exposes the real documents,
chunks and retrieval configuration for both Cited and OJ Assistant without
combining the corpora or calling an AI provider. Finish by downloading a
standalone report that can be opened, printed or saved as PDF.

This is a local demonstration guide. It does not publish the panel or send an
instructor message or link.

## Rehearse before the session

1. Confirm the one-time environment setup in `README.md` is complete.
2. Confirm `deploy/oj-assistant/content` contains the current approved export.
3. Double-click `Start-RAG-Management-Panel.cmd` and wait for the browser.
4. Confirm the corpus selector contains **Cited** and **OJ Assistant**.
5. Confirm the current demonstration data shows:
   - Cited: 1 document, 28 passages and 10 chunks.
   - OJ Assistant: 10 documents, 123 passages and 64 chunks.
6. Download one report, open it after stopping the panel and check Print Preview.
7. Restart once and complete the timed walkthrough below without notes.

If a count differs, do not present the old number. Restart the panel to refresh
its immutable snapshot, then confirm that the intended corpus/export is in use.

## Four-minute walkthrough

### 0:00 — Establish the boundary

Point to the green notice. Explain that this is a local, read-only visibility
tool. It cannot upload, edit or delete documents, create embeddings, answer a
question, deploy anything or spend provider credit.

### 0:35 — Show the Cited corpus

Select **Cited**. Show its totals, then the Documents table. Filter the Chunks
table and open one row. Explain that the displayed citation and text are the
same units made available to retrieval, not a manually prepared imitation.

### 1:30 — Show the independent OJ Assistant corpus

Select **OJ Assistant** and point out its independent totals and checksum. Say
explicitly that the selector changes the inspected corpus; it does not merge the
two retrieval indexes.

### 2:15 — Explain the configuration

Show the 180-word target, 40-word overlap and 25-word minimum. Point out the
estimated token equivalents of about 234, 52 and 33 tokens. Then show
`bge-small-en-v1.5`, 384 dimensions and the 512-token window. State that the
token figures are labelled estimates because production chunking is word-based.

### 3:05 — Export the evidence

Choose **Download report** for the selected corpus. Open the downloaded HTML and
show its document inventory, retrieval settings and chunk previews. Explain that
it is self-contained and printable, contains no script or remote assets, and
does not disclose local filesystem paths.

### 3:40 — Close honestly

State the limitation: this panel provides inspection, not production
administration. Uploads, re-indexing and mutation are intentionally outside its
approved scope. Press `Ctrl+C` in the command window when the demonstration is
finished.

## Recovery prompts

- **Browser did not open:** use the loopback address printed in the command
  window.
- **OJ Assistant is missing:** stop the panel, confirm the export path, then use
  the explicit `--corpus-profile LABEL=PATH` command from the inspector runbook.
- **The default port is busy:** start manually with `--port 8766` and open the
  printed address.
- **Report did not appear:** allow downloads for the local page, select the
  corpus again and retry once.

Never expose, tunnel or deploy the local panel to solve a demonstration problem.
