"""Standalone, printable HTML reports for local corpus-inspection snapshots."""

from __future__ import annotations

from html import escape
from typing import Any

from assistant.inspection import CorpusInspection


def render_report(corpus: CorpusInspection, nonce: str) -> str:
    """Render one immutable corpus snapshot without paths or active content."""
    summary = corpus.summary()
    chunking = summary["chunking"]
    embedding = summary["embedding"]
    vectors = summary["vectors"]
    document_count = len(corpus.documents)
    chunk_count = len(corpus.chunks)
    word_range = _range(summary["chunk_words"], "words")
    token_range = _range(summary["estimated_embedding_tokens"], "tokens")
    default_top_k = summary["retrieval"]["default_top_k"]
    document_rows = "".join(
        _document_row(document.as_dict()) for document in corpus.documents
    )
    chunk_rows = "".join(_chunk_row(chunk.as_dict()) for chunk in corpus.chunks)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{_text(corpus.label)} RAG management report</title>
    <style nonce="{_text(nonce)}">
      :root {{ color-scheme: light; --ink: #172033; --muted: #566176;
        --line: #d8dee8; --soft: #f4f7fb; --accent: #2858d8;
        --success: #176b45; --success-soft: #e8f7ef; }}
      * {{ box-sizing: border-box; }}
      body {{ margin: 0 auto; padding: 32px 24px 64px; max-width: 1120px;
        color: var(--ink); background: white;
        font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
      h1 {{ margin: 0 0 4px; }} h2 {{ margin: 30px 0 10px; }}
      p {{ margin: 4px 0; }} .muted {{ color: var(--muted); }}
      .notice {{ margin: 20px 0; padding: 12px 14px; color: var(--success);
        background: var(--success-soft); border: 1px solid #bce5cd;
        border-radius: 8px; }}
      .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
      .card {{ padding: 14px; background: var(--soft); border-radius: 8px; }}
      .card small {{ display: block; color: var(--muted); }}
      .card strong {{ display: block; margin-top: 2px; font-size: 20px; }}
      dl {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px 24px; }}
      dt {{ color: var(--muted); font-size: 13px; }} dd {{ margin: 2px 0 0; }}
      .table-wrap {{ overflow-x: auto; }}
      table {{ width: 100%; border-collapse: collapse; }}
      caption {{ padding-bottom: 8px; text-align: left; color: var(--muted); }}
      th, td {{ padding: 8px 9px; text-align: left; vertical-align: top;
        border: 1px solid var(--line); }} th {{ background: var(--soft); }}
      td.numeric {{ text-align: right; font-variant-numeric: tabular-nums; }}
      code {{ overflow-wrap: anywhere; }}
      @media (max-width: 700px) {{ .grid, dl {{ grid-template-columns: 1fr 1fr; }} }}
      @media (max-width: 440px) {{ .grid, dl {{ grid-template-columns: 1fr; }} }}
      @page {{ size: landscape; margin: 12mm; }}
      @media print {{ body {{ padding: 0; font-size: 9pt; max-width: none; }}
        h2 {{ break-after: avoid; }} tr, .card {{ break-inside: avoid; }}
        thead {{ display: table-header-group; }} .table-wrap {{ overflow: visible; }}
        .screen-note {{ display: none; }} }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>{_text(corpus.label)} RAG management report</h1>
        <p class="muted">Cited RAG Management Panel · standalone local snapshot</p>
        <p class="screen-note muted">Use your browser's Print command to print
          this report or save it as PDF.</p>
      </header>
      <p class="notice">Read-only report — no documents, vectors or settings
        were changed.</p>
      <div class="grid" aria-label="Corpus totals">
        <div class="card"><small>Documents</small>
          <strong>{document_count}</strong></div>
        <div class="card"><small>Passages</small>
          <strong>{corpus.passage_count}</strong></div>
        <div class="card"><small>Chunks</small>
          <strong>{chunk_count}</strong></div>
        <div class="card"><small>Embedding model</small>
          <strong>{_model_name(embedding)}</strong></div>
      </div>
      <h2>Retrieval configuration</h2>
      <dl>
        <div><dt>Chunk policy</dt><dd>{chunking["target_words"]} target words
          (~{chunking["estimated_target_tokens"]} tokens),
          {chunking["overlap_words"]}-word
          overlap (~{chunking["estimated_overlap_tokens"]} tokens), and
          {chunking["minimum_words"]}-word minimum
          (~{chunking["estimated_minimum_tokens"]} tokens)</dd></div>
        <div><dt>Actual chunk words</dt><dd>{word_range}</dd></div>
        <div><dt>Estimated embedding tokens</dt>
          <dd>{token_range}</dd></div>
        <div><dt>Embedding shape</dt><dd>{embedding["dimensions"]} dimensions /
          {embedding["window_tokens"]} token window</dd></div>
        <div><dt>Default retrieval count</dt>
          <dd>Top {default_top_k} chunks</dd></div>
        <div><dt>Vector validation</dt><dd>{_vector_status(vectors)}</dd></div>
        <div><dt>Corpus checksum</dt>
          <dd><code>{_text(corpus.checksum)}</code></dd></div>
      </dl>
      <p class="muted">{_text(embedding["token_measurement"])}</p>
      <h2>Documents</h2>
      <div class="table-wrap"><table>
        <caption>Corpus-relative sources and the units contributed to
          retrieval.</caption>
        <thead><tr><th scope="col">Source</th><th scope="col">Format</th>
          <th scope="col">Passages</th><th scope="col">Chunks</th>
          <th scope="col">Est. indexed tokens</th><th scope="col">Chunk token range</th>
          <th scope="col">Locations</th><th scope="col">SHA-256</th></tr></thead>
        <tbody>{document_rows}</tbody>
      </table></div>
      <h2>Chunk inventory</h2>
      <div class="table-wrap"><table>
        <caption>Every retrieval chunk, with a compact preview for a printable
          report.</caption>
        <thead><tr><th scope="col">#</th><th scope="col">Source and location</th>
          <th scope="col">Words</th><th scope="col">Est. tokens</th>
          <th scope="col">Preview</th></tr></thead>
        <tbody>{chunk_rows}</tbody>
      </table></div>
      <h2>Scope and interpretation</h2>
      <p>This report describes the immutable corpus snapshot loaded when the
        local panel started. It does not combine corpora, call an AI provider,
        create embeddings, or modify source files.</p>
    </main>
  </body>
</html>"""


def _document_row(document: dict[str, Any]) -> str:
    token_range = document["estimated_chunk_tokens"]
    range_text = _range(token_range, "tokens") if isinstance(token_range, dict) else "—"
    pages = document["pages"]
    sections = document["sections"]
    location_text = f"{len(pages)} page(s)" if pages else f"{len(sections)} section(s)"
    return (
        "<tr>"
        f"<td>{_text(document['source'])}</td>"
        f"<td>{_text(str(document['format']).upper())}</td>"
        f'<td class="numeric">{document["passage_count"]}</td>'
        f'<td class="numeric">{document["chunk_count"]}</td>'
        f'<td class="numeric">{document["estimated_embedding_tokens"]}</td>'
        f"<td>{range_text}</td>"
        f"<td>{location_text}</td>"
        f"<td><code>{_text(document['sha256'])}</code></td>"
        "</tr>"
    )


def _chunk_row(chunk: dict[str, Any]) -> str:
    return (
        "<tr>"
        f'<td class="numeric">{chunk["index"]}</td>'
        f"<td>{_text(chunk['citation'])}</td>"
        f'<td class="numeric">{chunk["body_words"]}</td>'
        f'<td class="numeric">{chunk["estimated_embedding_tokens"]}</td>'
        f"<td>{_text(chunk['preview'])}</td>"
        "</tr>"
    )


def _range(values: dict[str, Any], unit: str) -> str:
    return (
        f"{values['minimum']}&ndash;{values['maximum']} {unit} "
        f"(average {values['average']}, median {values['median']})"
    )


def _model_name(embedding: dict[str, Any]) -> str:
    return _text(str(embedding["model"]).split("/")[-1])


def _vector_status(vectors: dict[str, Any]) -> str:
    state = str(vectors["state"]).replace("_", " ")
    return f"{_text(state)} — {_text(vectors['message'])}"


def _text(value: object) -> str:
    return escape(str(value), quote=True)
