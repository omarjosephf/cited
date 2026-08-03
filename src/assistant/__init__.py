"""A document assistant that answers only from a supplied corpus.

The package is deliberately importable without the web layer: nothing under
`assistant` may import from `assistant.api`. That boundary is what allows the
core to be embedded in a different deployment without a rewrite.
"""

__version__ = "0.1.0"
