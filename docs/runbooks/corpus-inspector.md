# Runbook: Cited RAG Management Panel

## Purpose

Use this panel to inspect Cited and OJ Assistant locally. It shows which
documents and chunks retrieval can use, plus the active chunking, embedding and
top-k configuration. It can export a standalone, printable report for the
selected corpus. It does not upload, edit, delete, embed, answer, deploy or call
a provider.

## Start on Windows with one click

Complete the README's virtual-environment setup once. After that, double-click
`Start-RAG-Management-Panel.cmd` in the repository root. The launcher opens the
panel in the default browser when it is ready. Keep the command window open and
press `Ctrl+C` there when finished.

The launcher intentionally does not install or update packages. If it reports a
missing environment or dependency, follow the one-time setup in `README.md`
instead of changing the launcher.

## Start with the normal repository layout

Install the API and development extras once, then run from the repository root:

```powershell
doc-assistant inspect
```

This selects `content` as **Cited** and discovers each `deploy/*/content`
directory, including a normal OJ Assistant export. Open the printed loopback
address (by default `http://127.0.0.1:8765`).

## Start with explicit corpora

Use explicit profiles when the corpora are outside the normal export layout,
including when working in an isolated Git worktree:

```powershell
doc-assistant inspect `
  --corpus-profile "Cited=content" `
  --corpus-profile "OJ Assistant=C:\path\to\oj-assistant\content"
```

The label is what appears in the selector. A profile path is accepted only on
the command line at startup and is not returned to the browser.

Optionally validate an existing vector artifact by using exactly the same
label:

```powershell
doc-assistant inspect `
  --corpus-profile "Cited=content" `
  --vectors "Cited=artifacts\corpus-vectors.npz"
```

This only reads and validates the artifact. It never creates or replaces one.

## What to verify for a demonstration

1. Select **Cited** and confirm its own document, passage and chunk totals.
2. Select **OJ Assistant** and confirm its independent totals.
3. Open the Documents table and show the corpus-relative source names.
4. Filter Chunks by source or text and open one chunk to show its citation and
   exact retrieval text.
5. Point out the chunk policy, embedding model/window, top-k and clearly
   labelled estimated-token range.
6. Point out the green local/read-only notice.
7. Select **Download report**, open the saved HTML file and use the browser's
   Print command or **Save as PDF** to demonstrate that it stands alone.

Do not describe the two corpora as one combined index. The selector deliberately
keeps their retrieval boundaries separate.

## Download or print a report

Select the corpus first, then choose **Download report**. The filename includes
the selected corpus identifier. The saved HTML contains no scripts, remote
assets or local filesystem paths and can be opened without the panel running.

The report contains document names, checksums, retrieval settings and a preview
of every chunk. Review it before sharing, particularly if a custom corpus is not
public. Printing and PDF creation use the browser's standard Print command.

## Stop and refresh

Press `Ctrl+C` in the terminal to stop. The panel holds an immutable startup
snapshot, so stop and start it again after changing or exporting corpus files.

## Troubleshooting

- **A corpus is missing:** confirm its `content` directory exists, or provide it
  with `--corpus-profile LABEL=PATH`.
- **No extractable content:** confirm the directory contains a non-empty `.md`,
  `.txt`, `.docx` or `.pdf` document. README, dot-prefixed and underscore-
  prefixed files are intentionally excluded.
- **Port already in use:** add `--port` with another local port, for example
  `--port 8766`.
- **Launcher cannot find Python:** create the repository `.venv` using the
  README setup. Advanced worktree users may set `DOC_ASSISTANT_PYTHON` to an
  existing compatible virtual-environment Python executable before launching.
- **Vectors show invalid:** rebuild only through the separately approved vector
  workflow. The inspector must not repair or overwrite them.
