"""Shared response-header policy for the two small browser interfaces."""

from __future__ import annotations

from starlette.responses import Response

SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
}

NONCE_PLACEHOLDER = "__CSP_NONCE__"


def content_security_policy(nonce: str | None) -> str:
    """Return the deny-by-default policy used by the project-owned pages."""
    script = f"'nonce-{nonce}'" if nonce else "'none'"
    style = f"'nonce-{nonce}'" if nonce else "'none'"
    return "; ".join(
        [
            "default-src 'none'",
            f"script-src {script}",
            f"style-src {style}",
            "connect-src 'self'",
            "img-src 'self' data:",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
            "object-src 'none'",
            "upgrade-insecure-requests",
        ]
    )


def apply_security_headers(
    response: Response, *, no_store: bool = False, no_index: bool = False
) -> None:
    """Apply project headers without overriding a response-specific CSP nonce."""
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    response.headers.setdefault(
        "Content-Security-Policy", content_security_policy(None)
    )
    if no_store:
        response.headers.setdefault("Cache-Control", "no-store")
    if no_index:
        response.headers.setdefault("X-Robots-Tag", "noindex, nofollow")
