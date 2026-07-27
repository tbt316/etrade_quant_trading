"""Credential-safe labels for outbound HTTP diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED = "<redacted>"
_SENSITIVE_FIELDS = {
    "accesstoken",
    "apikey",
    "authorization",
    "clientsecret",
    "consumerkey",
    "oauthsignature",
    "token",
}


def _is_sensitive(field: object) -> bool:
    normalized = "".join(character for character in str(field).lower() if character.isalnum())
    return normalized in _SENSITIVE_FIELDS


def redacted_request_url(
    url: str,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Return a request label with credential-like query values removed."""
    parts = urlsplit(url)
    query: list[tuple[str, Any]] = list(parse_qsl(parts.query, keep_blank_values=True))
    if params:
        query.extend((str(key), value) for key, value in params.items())
    safe_query = [
        (key, REDACTED if _is_sensitive(key) else value)
        for key, value in query
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(safe_query, doseq=True), parts.fragment)
    )


def safe_exception_summary(error: BaseException) -> str:
    """Describe a request failure without copying its potentially secret URL."""
    details = [type(error).__name__]
    status = getattr(error, "status", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        details.append(f"status={status}")
    errno = getattr(error, "errno", None)
    if errno is not None:
        details.append(f"errno={errno}")
    return " ".join(details)
