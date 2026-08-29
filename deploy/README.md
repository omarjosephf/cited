# Deployment artifacts

**Nothing in this directory is edited here.** It is a staging area for corpus
artifacts produced by another repository and copied into the container image at
build time.

The only tracked files are this README and `.gitkeep`. Everything else is
generated, ignored by Git, and re-created by the exporting repository's build
script. Editing a document here would produce a corpus that no longer matches its
checksum, and the service would refuse to start — which is the intended
behaviour, not a bug to work around.

`.gitkeep` exists so the directory is always present in the build context. The
`Dockerfile` copies this directory unconditionally, and `COPY` fails on a missing
source — so without it, every build of the default image would break the moment
the staging content was cleaned up.

## Layout of an exported artifact

```text
deploy/<deployment-name>/
├── CHECKSUM             # the corpus digest, computed by the exporter
├── content/             # the corpus documents themselves
└── system-prompt.md     # that deployment's system prompt
```

## For the OJ Assistant deployment

Produced by `scripts/export-assistant-corpus.mjs` in the portfolio repository,
which is the single editable source of that corpus. See
`docs/runbooks/assistant-corpus.md` there for the release procedure and for what
the checksum does and does not prove.
