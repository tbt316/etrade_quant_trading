"""Standalone read-only operator dashboard.

The server consumes only validated runtime files.  It cannot construct an
E*TRADE client, call a market-data provider, enqueue work, or authorize broker
mutations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import re
import stat
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from live_trading.regime_shadow_store import unavailable_dashboard_payload
from live_trading.runtime_composition import (
    MAX_SESSION_SECONDS,
    ReadOnlyRuntimeContext,
    RuntimeCompositionError,
    load_read_only_runtime,
)
from live_trading.runtime_config import RuntimeConfigError


MAX_LOGIN_BODY_BYTES = 8 * 1024
MAX_LOGIN_FAILURES = 5
LOGIN_FAILURE_WINDOW_SECONDS = 60
MAX_POSITIONS_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_FUTURE_SKEW_SECONDS = 5
REQUEST_SOCKET_TIMEOUT_SECONDS = 10
POSITIONS_READ_ONLY_MARKER = (
    "Read only — all E*TRADE order actions are disabled"
)
DISABLED_EXECUTION_PATHS = frozenset(
    {
        "/api/execute_manual_order",
        "/api/execute_neutralize_order",
        "/api/review_close_position",
        "/api/close_position",
        "/api/execute_close_order",
        "/api/settings",
        "/api/verify_pin",
        "/refresh",
    }
)
FORBIDDEN_POSITIONS_MARKERS = frozenset(
    {
        *DISABLED_EXECUTION_PATHS,
        "data-close-position",
        "action-cell",
    }
)
POSITIONS_FRAME_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "script-src 'none'; connect-src 'none'; frame-ancestors 'self'; "
    "base-uri 'none'; form-action 'none'"
)
DASHBOARD_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; frame-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'self'"
)
LOGIN_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


class ReadOnlyDashboardError(RuntimeError):
    """Raised when the dashboard cannot preserve its read-only contract."""


class LoginAttemptLimiter:
    """Small in-memory global limiter for the loopback-only login surface."""

    def __init__(self) -> None:
        self._failures: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - LOGIN_FAILURE_WINDOW_SECONDS
        while self._failures and self._failures[0] <= cutoff:
            self._failures.popleft()

    def allows_attempt(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return len(self._failures) < MAX_LOGIN_FAILURES

    def record_failure(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            self._failures.append(now)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    available: bool
    stale: bool
    content: bytes | None = field(repr=False)
    sha256: str | None
    modified_at: str | None
    size_bytes: int | None
    reason: str | None


@dataclass(frozen=True, slots=True, repr=False)
class ReadOnlyDashboardApplication:
    """Immutable dependencies shared by all dashboard request handlers."""

    runtime: ReadOnlyRuntimeContext
    template: bytes
    login_limiter: LoginAttemptLimiter = field(
        default_factory=LoginAttemptLimiter,
        compare=False,
        repr=False,
    )
    secure_cookie: bool = False

    def __post_init__(self) -> None:
        if type(self.runtime) is not ReadOnlyRuntimeContext:
            raise TypeError(
                "runtime must be an exact ReadOnlyRuntimeContext instance"
            )
        if type(self.template) is not bytes or not self.template:
            raise TypeError("template must be non-empty exact bytes")
        if type(self.login_limiter) is not LoginAttemptLimiter:
            raise TypeError(
                "login_limiter must be an exact LoginAttemptLimiter instance"
            )
        if type(self.secure_cookie) is not bool:
            raise TypeError("secure_cookie must be an exact boolean")
        if self.runtime.broker_mutations_enabled:
            raise ReadOnlyDashboardError(
                "dashboard runtime unexpectedly permits broker mutations"
            )

    def __repr__(self) -> str:
        return (
            "ReadOnlyDashboardApplication("
            f"mode={self.runtime.mode!r}, "
            f"config_sha256={self.runtime.source_sha256!r}, "
            "auth=[REDACTED])"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        environ: Mapping[str, str] | None = None,
        secure_cookie: bool = False,
    ) -> "ReadOnlyDashboardApplication":
        runtime = load_read_only_runtime(config_path, environ=environ)
        template = (
            importlib.resources.files("live_trading")
            .joinpath("read_only_dashboard.html")
            .read_bytes()
        )
        return cls(
            runtime=runtime,
            template=template,
            login_limiter=LoginAttemptLimiter(),
            secure_cookie=secure_cookie,
        )

    def positions_snapshot(
        self,
        *,
        now: datetime | None = None,
    ) -> ArtifactSnapshot:
        return _read_positions_artifact(
            self.runtime.paths.positions_artifact_file,
            max_age_seconds=self.runtime.max_snapshot_age_seconds,
            now=now,
        )

    def regime_payload(self) -> dict[str, Any]:
        try:
            payload = self.runtime.regime_shadow_reader.dashboard_payload()
        except Exception:
            return unavailable_dashboard_payload("invalid")
        if (
            not isinstance(payload, dict)
            or payload.get("may_authorize_execution") is not False
        ):
            return unavailable_dashboard_payload("invalid")
        return payload

    def status_payload(self) -> dict[str, Any]:
        positions = self.positions_snapshot()
        regime = self.regime_payload()
        return {
            "schema_version": 1,
            "mode": self.runtime.mode,
            "read_only": True,
            "execution_enabled": False,
            "broker_reads_configured": self.runtime.broker_reads_enabled,
            "config_sha256": self.runtime.source_sha256,
            "positions": {
                "available": positions.available,
                "stale": positions.stale,
                "sha256": positions.sha256,
                "modified_at": positions.modified_at,
                "size_bytes": positions.size_bytes,
                "reason": positions.reason,
            },
            "regime": {
                "available": bool(regime.get("available")),
                "status": regime.get("status", "unavailable"),
                "as_of_session": regime.get("as_of_session"),
                "effective_session": regime.get("effective_session"),
                "composite_label": regime.get(
                    "composite_label",
                    "unavailable+unavailable",
                ),
                "stale": bool(regime.get("stale")),
                "may_authorize_execution": False,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _private_regular_file(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and not stat.S_IMODE(metadata.st_mode) & 0o077
        and metadata.st_nlink == 1
    )


def _safe_open_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _unavailable_snapshot(
    reason: str,
    *,
    stale: bool = False,
    sha256: str | None = None,
    modified_at: str | None = None,
    size_bytes: int | None = None,
) -> ArtifactSnapshot:
    return ArtifactSnapshot(
        available=False,
        stale=stale,
        content=None,
        sha256=sha256,
        modified_at=modified_at,
        size_bytes=size_bytes,
        reason=reason,
    )


def _read_positions_artifact(
    path: Path,
    *,
    max_age_seconds: int,
    now: datetime | None = None,
) -> ArtifactSnapshot:
    """Read one owner-only artifact through a descriptor-verified parent."""

    if (
        not isinstance(path, Path)
        or not path.name
        or type(max_age_seconds) is not int
        or max_age_seconds <= 0
    ):
        return _unavailable_snapshot("invalid_path")
    current_time = datetime.now(timezone.utc) if now is None else now
    if type(current_time) is not datetime or current_time.tzinfo is None:
        raise TypeError("now must be an exact timezone-aware datetime")
    current_time = current_time.astimezone(timezone.utc)
    try:
        parent_before = os.lstat(path.parent)
        if (
            stat.S_ISLNK(parent_before.st_mode)
            or not stat.S_ISDIR(parent_before.st_mode)
            or parent_before.st_uid != os.geteuid()
            or stat.S_IMODE(parent_before.st_mode) != 0o700
        ):
            return _unavailable_snapshot("unsafe_parent")
        parent_descriptor = os.open(
            path.parent,
            _safe_open_flags(
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            ),
        )
    except FileNotFoundError:
        return _unavailable_snapshot("missing")
    except OSError:
        return _unavailable_snapshot("unsafe_parent")
    try:
        parent_after = os.fstat(parent_descriptor)
        if not _same_file(parent_before, parent_after):
            return _unavailable_snapshot("unsafe_parent")
        try:
            before = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return _unavailable_snapshot("missing")
        except OSError:
            return _unavailable_snapshot("unsafe_file")
        if not _private_regular_file(before):
            return _unavailable_snapshot("unsafe_file")
        if before.st_size > MAX_POSITIONS_ARTIFACT_BYTES:
            return _unavailable_snapshot("oversized")
        try:
            descriptor = os.open(
                path.name,
                _safe_open_flags(
                    os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                ),
                dir_fd=parent_descriptor,
            )
        except OSError:
            return _unavailable_snapshot("unsafe_file")
        try:
            after = os.fstat(descriptor)
            if not _same_file(before, after):
                return _unavailable_snapshot("changed")
            chunks: list[bytes] = []
            remaining = MAX_POSITIONS_ARTIFACT_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            final = os.fstat(descriptor)
            try:
                path_after = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                return _unavailable_snapshot("changed")
            if (
                not _private_regular_file(final)
                or not _private_regular_file(path_after)
                or not _same_snapshot(before, after)
                or not _same_snapshot(after, final)
                or not _same_snapshot(final, path_after)
                or len(content) != final.st_size
            ):
                return _unavailable_snapshot("changed")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    if len(content) > MAX_POSITIONS_ARTIFACT_BYTES:
        return _unavailable_snapshot("oversized")
    try:
        text = content.decode("utf-8")
    except UnicodeError:
        return _unavailable_snapshot("invalid_encoding")
    lowered = text.lower()
    if (
        POSITIONS_READ_ONLY_MARKER not in text
        or any(marker.lower() in lowered for marker in FORBIDDEN_POSITIONS_MARKERS)
    ):
        return _unavailable_snapshot("legacy_or_executable")
    digest = hashlib.sha256(content).hexdigest()
    modified_at = datetime.fromtimestamp(
        final.st_mtime,
        tz=timezone.utc,
    )
    age_seconds = (current_time - modified_at).total_seconds()
    if age_seconds < -MAX_ARTIFACT_FUTURE_SKEW_SECONDS:
        return _unavailable_snapshot(
            "future_timestamp",
            sha256=digest,
            modified_at=modified_at.isoformat(),
            size_bytes=len(content),
        )
    if age_seconds > max_age_seconds:
        return _unavailable_snapshot(
            "stale",
            stale=True,
            sha256=digest,
            modified_at=modified_at.isoformat(),
            size_bytes=len(content),
        )
    return ArtifactSnapshot(
        available=True,
        stale=False,
        content=content,
        sha256=digest,
        modified_at=modified_at.isoformat(),
        size_bytes=len(content),
        reason=None,
    )


def _decode_json_object(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            ValueError("non-finite JSON value")
        ),
    )
    if type(value) is not dict:
        raise ValueError("JSON body must be an object")
    return value


def _positions_fallback() -> bytes:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Positions unavailable</title><style>"
        "body{margin:0;padding:18px;color:#e2e8f0;background:#0f172a;"
        "font:15px/1.5 system-ui,sans-serif}.notice{border:1px solid #ef4444;"
        "border-radius:10px;padding:14px;background:#450a0a}strong{display:block;"
        "margin-bottom:6px;color:#fecaca}</style></head>"
        "<body data-positions-artifact-state=\"unavailable\"><div class=\"notice\" "
        "role=\"alert\"><strong>"
        f"{POSITIONS_READ_ONLY_MARKER}"
        "</strong>Current position data is unavailable or failed its "
        "read-only integrity checks.</div></body></html>"
    ).encode("utf-8")


def _login_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>E*TRADE Monitor Login</title>
  <style>
    :root{color-scheme:dark}body{margin:0;min-height:100vh;display:grid;
    place-items:center;background:#07111f;color:#e5edf7;font:15px/1.5
    system-ui,sans-serif}form{width:min(360px,calc(100vw - 36px));padding:28px;
    display:grid;gap:14px;background:#101d2d;border:1px solid #26384e;
    border-radius:16px;box-shadow:0 24px 70px #0008}h1{margin:0;font-size:22px}
    p{margin:0 0 4px;color:#9eb0c7}input,button{box-sizing:border-box;width:100%;
    border-radius:9px;padding:12px;font:inherit}input{border:1px solid #38506d;
    background:#091522;color:#f8fafc}button{border:0;background:#2d81f7;
    color:white;font-weight:700;cursor:pointer}.error{min-height:21px;color:#fca5a5}
  </style>
</head>
<body>
  <form id="login-form">
    <h1>Read-only monitor</h1>
    <p>Execution is disabled at the server boundary.</p>
    <input id="username" autocomplete="username" placeholder="Username" required>
    <input id="password" type="password" autocomplete="current-password"
           placeholder="Password" required>
    <button type="submit">Sign in</button>
    <div id="error" class="error" role="alert"></div>
  </form>
  <script>
    document.getElementById('login-form').addEventListener('submit', async e => {
      e.preventDefault();
      const response = await fetch('/api/login', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          username:document.getElementById('username').value,
          password:document.getElementById('password').value
        })
      });
      if(response.ok){location.href='/dashboard';return}
      document.getElementById('error').textContent='Invalid username or password.';
    });
  </script>
</body>
</html>"""


def _cookie(
    name: str,
    value: str,
    *,
    max_age: int,
    secure: bool,
) -> str:
    secure_attribute = "; Secure" if secure else ""
    return (
        f"{name}={value}; Max-Age={max_age}; "
        f"Path=/; HttpOnly; SameSite=Strict{secure_attribute}"
    )


def make_handler(
    application: ReadOnlyDashboardApplication,
) -> type[BaseHTTPRequestHandler]:
    if type(application) is not ReadOnlyDashboardApplication:
        raise TypeError(
            "application must be an exact ReadOnlyDashboardApplication instance"
        )

    class ReadOnlyDashboardHandler(BaseHTTPRequestHandler):
        server_version = "EtradeReadOnlyDashboard/1"
        sys_version = ""

        def log_message(self, format: str, *args: object) -> None:
            return

        def log_request(
            self,
            code: int | str = "-",
            size: int | str = "-",
        ) -> None:
            print(
                json.dumps(
                    {
                        "event": "read_only_dashboard_request",
                        "method": self.command,
                        "path": self._path() or "<invalid>",
                        "status": code,
                        "time": datetime.now(timezone.utc).isoformat(),
                    },
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )

        @property
        def app(self) -> ReadOnlyDashboardApplication:
            return application

        def _path(self) -> str | None:
            try:
                parsed = urlsplit(self.path)
            except (UnicodeError, ValueError):
                return None
            if (
                parsed.scheme
                or parsed.netloc
                or not parsed.path.startswith("/")
            ):
                return None
            return parsed.path

        def _session_value(self) -> str | None:
            try:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
                morsel = cookie.get(self.app.runtime.session_cookie_name)
                return morsel.value if morsel is not None else None
            except Exception:
                return None

        def _authenticated(self) -> bool:
            return self.app.runtime.dashboard_auth.verify_session(
                self._session_value()
            )

        def _headers(
            self,
            *,
            content_type: str,
            content_length: int,
            csp: str | None = None,
            x_frame_options: str = "DENY",
            extra: Mapping[str, str] | None = None,
        ) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Permissions-Policy",
                "camera=(), geolocation=(), microphone=(), payment=()",
            )
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("X-Frame-Options", x_frame_options)
            self.send_header("Vary", "Authorization, Cookie")
            if csp is not None:
                self.send_header("Content-Security-Policy", csp)
            for name, value in (extra or {}).items():
                self.send_header(name, value)

        def _send_bytes(
            self,
            code: int,
            payload: bytes,
            *,
            content_type: str,
            csp: str | None = None,
            x_frame_options: str = "DENY",
            extra: Mapping[str, str] | None = None,
        ) -> None:
            self.send_response(code)
            self._headers(
                content_type=content_type,
                content_length=len(payload),
                csp=csp,
                x_frame_options=x_frame_options,
                extra=extra,
            )
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _send_json(
            self,
            code: int,
            payload: Mapping[str, Any],
            *,
            extra: Mapping[str, str] | None = None,
        ) -> None:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_bytes(
                code,
                encoded,
                content_type="application/json; charset=utf-8",
                extra=extra,
            )

        def _require_auth(self) -> bool:
            if self._authenticated():
                return True
            self._send_json(401, {"error": "Authentication required"})
            return False

        def _redirect(self, location: str) -> None:
            self.send_response(302)
            self._headers(
                content_type="text/plain; charset=utf-8",
                content_length=0,
                extra={"Location": location},
            )
            self.end_headers()

        def _read_login_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            get_all = getattr(self.headers, "get_all", None)
            all_lengths = (
                get_all("Content-Length")
                if callable(get_all)
                else [raw_length] if raw_length is not None else []
            )
            if (
                len(all_lengths) != 1
                or raw_length is None
                or re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_length) is None
            ):
                raise ValueError("missing content length")
            if self.headers.get("Content-Type", "").lower() != (
                "application/json"
            ):
                raise ValueError("invalid content type")
            length = int(raw_length)
            if length < 2 or length > MAX_LOGIN_BODY_BYTES:
                raise ValueError("invalid body length")
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("incomplete body")
            data = _decode_json_object(payload)
            if set(data) != {"username", "password"}:
                raise ValueError("invalid login fields")
            return data

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            path = self._path()
            if path is None:
                self._send_json(400, {"error": "Invalid request target"})
                return
            if path == "/healthz":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "read_only": True,
                        "execution_enabled": False,
                    },
                )
                return
            if path == "/login":
                self._send_bytes(
                    200,
                    _login_html(),
                    content_type="text/html; charset=utf-8",
                    csp=LOGIN_CSP,
                )
                return
            if path in {"/", "/dashboard"} and not self._authenticated():
                self._redirect("/login")
                return
            if not self._require_auth():
                return
            if path == "/":
                self._redirect("/dashboard")
                return
            if path == "/dashboard":
                self._send_bytes(
                    200,
                    self.app.template,
                    content_type="text/html; charset=utf-8",
                    csp=DASHBOARD_CSP,
                )
                return
            if path == "/readyz":
                self._send_json(
                    200,
                    {
                        "status": "ready",
                        "read_only": True,
                        "execution_enabled": False,
                        "config_sha256": (
                            self.app.runtime.source_sha256
                        ),
                    },
                )
                return
            if path == "/api/status":
                self._send_json(200, self.app.status_payload())
                return
            if path == "/api/regime_v2_shadow":
                payload = self.app.regime_payload()
                self._send_json(
                    200 if payload.get("available") else 503,
                    payload,
                )
                return
            if path == "/api/positions_version":
                snapshot = self.app.positions_snapshot()
                self._send_json(
                    200 if snapshot.available else 503,
                    {
                        "available": snapshot.available,
                        "stale": snapshot.stale,
                        "sha256": snapshot.sha256,
                        "modified_at": snapshot.modified_at,
                        "size_bytes": snapshot.size_bytes,
                        "reason": snapshot.reason,
                    },
                )
                return
            if path == "/api/positions":
                snapshot = self.app.positions_snapshot()
                self._send_bytes(
                    200 if snapshot.available else 503,
                    snapshot.content
                    if snapshot.available and snapshot.content is not None
                    else _positions_fallback(),
                    content_type="text/html; charset=utf-8",
                    csp=POSITIONS_FRAME_CSP,
                    x_frame_options="SAMEORIGIN",
                )
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            path = self._path()
            if path is None:
                self._send_json(400, {"error": "Invalid request target"})
                return
            if path == "/api/login":
                if not self.app.login_limiter.allows_attempt():
                    self._send_json(
                        429,
                        {"error": "Too many login attempts"},
                        extra={
                            "Retry-After": str(
                                LOGIN_FAILURE_WINDOW_SECONDS
                            )
                        },
                    )
                    return
                try:
                    data = self._read_login_body()
                except (
                    UnicodeError,
                    ValueError,
                    json.JSONDecodeError,
                    RecursionError,
                ):
                    self.app.login_limiter.record_failure()
                    self._send_json(400, {"error": "Invalid login request"})
                    return
                if not self.app.runtime.dashboard_auth.verify_login(
                    data["username"],
                    data["password"],
                ):
                    self.app.login_limiter.record_failure()
                    self._send_json(403, {"error": "Invalid username or password"})
                    return
                self.app.login_limiter.reset()
                session = self.app.runtime.dashboard_auth.issue_session()
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "read_only": True,
                        "execution_enabled": False,
                    },
                    extra={
                        "Set-Cookie": _cookie(
                            self.app.runtime.session_cookie_name,
                            session,
                            max_age=MAX_SESSION_SECONDS,
                            secure=self.app.secure_cookie,
                        )
                    },
                )
                return
            if path == "/api/logout":
                self._send_json(
                    200,
                    {"status": "ok"},
                    extra={
                        "Set-Cookie": _cookie(
                            self.app.runtime.session_cookie_name,
                            "",
                            max_age=0,
                            secure=self.app.secure_cookie,
                        )
                    },
                )
                return
            if not self._require_auth():
                return
            if path in DISABLED_EXECUTION_PATHS:
                self._send_json(
                    503,
                    {
                        "code": "READ_ONLY_RUNTIME",
                        "error": "Trading actions are disabled.",
                        "read_only": True,
                        "execution_enabled": False,
                    },
                )
                return
            self._send_json(405, {"error": "Method not allowed"})

        def do_OPTIONS(self) -> None:
            self._send_json(405, {"error": "Method not allowed"})

        def do_PUT(self) -> None:
            self._send_json(405, {"error": "Method not allowed"})

        def do_PATCH(self) -> None:
            self._send_json(405, {"error": "Method not allowed"})

        def do_DELETE(self) -> None:
            self._send_json(405, {"error": "Method not allowed"})

    return ReadOnlyDashboardHandler


class ReadOnlyHTTPServer(HTTPServer):
    """Single-request loopback server with a bounded socket deadline."""

    request_queue_size = 16

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(REQUEST_SOCKET_TIMEOUT_SECONDS)
        return request, client_address


def create_server(
    application: ReadOnlyDashboardApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ReadOnlyHTTPServer:
    if host != "127.0.0.1":
        raise ReadOnlyDashboardError("dashboard host must be loopback")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ReadOnlyDashboardError("dashboard port is invalid")
    return ReadOnlyHTTPServer((host, port), make_handler(application))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the broker-isolated read-only E*TRADE dashboard"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", required=True, type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)
    serve.add_argument("--secure-cookie", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        application = ReadOnlyDashboardApplication.from_config(
            args.config,
            secure_cookie=args.secure_cookie,
        )
        server = create_server(
            application,
            host=args.host,
            port=args.port,
        )
    except (
        OSError,
        ReadOnlyDashboardError,
        RuntimeCompositionError,
        RuntimeConfigError,
    ) as exc:
        print(f"read-only dashboard startup rejected: {exc}")
        return 2
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactSnapshot",
    "DISABLED_EXECUTION_PATHS",
    "ReadOnlyDashboardApplication",
    "ReadOnlyDashboardError",
    "ReadOnlyHTTPServer",
    "create_server",
    "main",
    "make_handler",
]
