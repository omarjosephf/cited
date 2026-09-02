# Threat model: local corpus inspector

## Scope and assets

`doc-assistant inspect` displays fixed corpus snapshots in a local browser. The
assets are the corpus text, local filesystem paths, corpus/vector integrity,
developer credentials, and the guarantee that inspection cannot spend money or
change a corpus.

It is intentionally outside the scope of public hosting, authentication,
uploads, document editing, index generation, answering and deployment.

## Trust boundaries and threats

1. **Browser to loopback application.** Another webpage or process could try to
   reach the local server, including through DNS rebinding or a forged Host.
2. **Configured corpus to HTML page.** Corpus text is untrusted display data. A
   document could contain script or markup intended to execute in the panel.
3. **Command line to filesystem.** The operator supplies corpus and optional
   vector paths. Returning those paths to the browser would reveal local
   structure.
4. **Inspection to production logic.** A second implementation of ingestion
   could report different chunks from the ones retrieval uses.
5. **Convenience to capability creep.** An apparently harmless button could
   introduce mutation, provider cost or production impact.

## Controls

- Uvicorn is always bound to `127.0.0.1`; there is no command-line host option.
- A Host allowlist accepts only loopback names and the test host.
- Corpus paths are resolved at startup. HTTP requests can select only registered
  identifiers and cannot provide paths.
- The HTTP interface defines GET routes only and disables OpenAPI/Swagger pages.
- APIs return corpus-relative source names, never configured directory paths.
- The page uses DOM `textContent` and element construction for corpus content;
  it does not interpret corpus text as HTML.
- A deny-by-default Content Security Policy uses a fresh nonce for the one
  project-owned style and script block. Framing, MIME sniffing, browser
  capabilities, caching and indexing are restricted with response headers.
- Page size and search/source query lengths are bounded.
- Inspection calls the production readers, chunker and vector validator.
- No embedding model, API credential or answer provider is loaded or called.
- Snapshots are immutable for the lifetime of the process.

## Residual risk and operating rule

Any local process running as the same user can generally reach a loopback port.
The panel is therefore local-only, not an access-control boundary between users
of a shared computer. Anyone who can view the panel can read all configured
corpus chunks. Do not configure confidential material on an untrusted or shared
machine.

The panel must not be reverse-proxied, tunnelled, bound to a non-loopback
interface or deployed. A public or multi-user version requires authentication,
authorisation, audit logging, CSRF review and a new threat model.
