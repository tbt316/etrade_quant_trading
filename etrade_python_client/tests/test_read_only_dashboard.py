from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from email.message import Message
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from live_trading.read_only_dashboard import (
    LOGIN_FAILURE_WINDOW_SECONDS,
    MAX_LOGIN_FAILURES,
    ReadOnlyDashboardApplication,
    ReadOnlyDashboardError,
    create_server,
    make_handler,
)
from live_trading.positions_artifact import (
    MAX_ARTIFACT_FUTURE_SKEW_SECONDS,
    POSITIONS_READ_ONLY_MARKER,
    PositionsArtifactSigningKey,
    build_positions_snapshot,
    render_positions_html,
)

ARTIFACT_KEY_TEXT = (
    "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8"
)
ARTIFACT_KEY = PositionsArtifactSigningKey.from_text(
    ARTIFACT_KEY_TEXT
)


def _write_application(tmp_path: Path) -> ReadOnlyDashboardApplication:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    root = private / "runtime"
    for relative in (
        "",
        "state",
        "cache",
        "logs",
        "artifacts",
        "execution",
        "data",
        "model",
    ):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
    config = {
        "schema_version": 1,
        "mode": "shadow",
        "runtime_root": "runtime",
        "strategy": {
            "enabled": False,
            "strategy_id": "disabled",
            "symbols": [],
        },
        "data": {
            "require_complete_snapshots": True,
            "max_snapshot_age_seconds": 300,
        },
        "model": {
            "enabled": False,
            "required_for_entry": False,
            "max_signal_age_seconds": 86_400,
        },
        "execution": {
            "selected_account_id_key": "account-key-private",
            "account_allowlist": [{
                "account_id": "12345678",
                "account_id_key": "account-key-private",
                "institution_type": "BROKERAGE",
            }],
            "broker_mutations_enabled": False,
        },
        "risk": {
            "max_order_contracts": 0,
            "max_order_loss_cents": 0,
            "max_account_open_risk_cents": 0,
            "max_daily_loss_cents": 0,
            "max_quote_age_seconds": 30,
        },
    }
    config_path = private / "runtime-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    os.chmod(config_path, 0o600)
    return ReadOnlyDashboardApplication.from_config(
        config_path,
        environ={
            "ETRADE_DASHBOARD_USER": "readonly-operator",
            "ETRADE_DASHBOARD_PASSWORD": "correct-horse-battery-staple",
            "ETRADE_DASHBOARD_PIN": "A9~strong",
            "ETRADE_DASHBOARD_SESSION_SECRET": (
                "test-session-secret-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
            ),
            "ETRADE_POSITIONS_ARTIFACT_HMAC_KEY": (
                ARTIFACT_KEY_TEXT
            ),
        },
    )


def _artifact_path(
    application: ReadOnlyDashboardApplication,
) -> Path:
    return application.runtime.positions_artifact_reader.path


def _write_positions_artifact(
    application: ReadOnlyDashboardApplication,
    *,
    source_as_of: datetime | None = None,
    symbol: str = "SPY",
) -> Path:
    observed = source_as_of or datetime.now(timezone.utc)
    snapshot = build_positions_snapshot(
        [
            SimpleNamespace(
                symbol=symbol,
                security_type="Option",
                quantity=-2,
                last_price=1.25,
                price_paid=2.50,
                market_value=-250.00,
                total_gain=250.00,
                call_put="CALL",
                expiration_date="2026-08-21",
                strike_price=650,
                underlying_last_price=640.25,
                osi_key=(
                    f"{symbol.replace('.', '').ljust(6, '-')}"
                    "260821C00650000"
                ),
                option_multiplier=100,
                options_adjusted_flag=False,
                option_deliverables="100 shares",
            )
        ],
        broker_environment="production",
        source_as_of=observed,
    )
    artifact = _artifact_path(application)
    artifact.write_bytes(
        render_positions_html(
            snapshot,
            signing_key=ARTIFACT_KEY,
            runtime_binding=(
                application.runtime
                .positions_artifact_reader
                .expected_runtime_binding
            ),
        )
    )
    os.chmod(artifact, 0o600)
    return artifact


def _request(
    application: ReadOnlyDashboardApplication,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: Mapping[str, str] | Message | None = None,
) -> tuple[int, dict[str, str], bytes]:
    handler_type = make_handler(application)
    handler = object.__new__(handler_type)
    handler.command = method
    handler.path = path
    handler.headers = headers if headers is not None else {}
    handler.rfile = io.BytesIO(body or b"")
    handler.wfile = io.BytesIO()
    response_status: list[int] = []
    response_headers: dict[str, str] = {}
    handler.send_response = response_status.append
    handler.send_header = (
        lambda name, value: response_headers.__setitem__(
            name.lower(),
            value,
        )
    )
    handler.end_headers = lambda: None
    getattr(handler, f"do_{method}")()
    return (
        response_status[-1],
        response_headers,
        handler.wfile.getvalue(),
    )


def _raw_request(
    application: ReadOnlyDashboardApplication,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str] | Message | None = None,
) -> bytes:
    handler_type = make_handler(application)
    handler = object.__new__(handler_type)
    handler.command = method
    handler.path = path
    handler.request_version = "HTTP/1.1"
    handler.headers = headers if headers is not None else {}
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    getattr(handler, f"do_{method}")()
    return handler.wfile.getvalue()


def _login(application: ReadOnlyDashboardApplication) -> str:
    payload = json.dumps(
        {
            "username": "readonly-operator",
            "password": "correct-horse-battery-staple",
        }
    ).encode("utf-8")
    status, headers, body = _request(
        application,
        "POST",
        "/api/login",
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    assert status == 200, body
    return headers["set-cookie"].split(";", 1)[0]


def test_login_status_positions_and_regime_are_broker_isolated(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    _write_positions_artifact(application)
    assert "SPY" not in repr(
        application.positions_snapshot()
    )

    health_status, health_headers, health_body = _request(
        application,
        "GET",
        "/healthz",
    )
    assert health_status == 200
    assert json.loads(health_body)["execution_enabled"] is False
    assert "access-control-allow-origin" not in health_headers

    unauthenticated, redirect_headers, _ = _request(
        application,
        "GET",
        "/dashboard",
    )
    assert unauthenticated == 302
    assert redirect_headers["location"] == "/login"
    root_status, root_headers, _ = _request(application, "GET", "/")
    assert root_status == 302
    assert root_headers["location"] == "/login"

    cookie = _login(application)
    root_status, root_headers, _ = _request(
        application,
        "GET",
        "/",
        headers={"Cookie": cookie},
    )
    assert root_status == 302
    assert root_headers["location"] == "/dashboard"
    status, headers, template = _request(
        application,
        "GET",
        "/dashboard",
        headers={"Cookie": cookie},
    )
    assert status == 200
    assert b"Broker-isolated operator view" in template
    assert b"Execution" in template
    assert b"V2 shadow advisory" in template
    assert b"cannot authorize execution" in template
    assert b'id="refresh-now"' in template
    assert b'id="positions-frame" src="about:blank" sandbox' in template
    assert b"nextPositionsVersion !== positionsVersion" in template
    assert b"/api/positions?sha256=" in template
    assert b"frame.srcdoc = body" in template
    assert (
        template.index(b"frame.srcdoc = body")
        < template.index(b"positionsVersion = nextVersion")
    )
    assert b"positionsLoadGeneration += 1" in template
    assert b"generation !== positionsLoadGeneration" in template
    assert b"Verified freshness window expired" in template
    assert b"Status poll failed; previous artifact hidden" in template
    assert b"visibilitychange" in template
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["permissions-policy"] == (
        "camera=(), geolocation=(), microphone=(), payment=()"
    )

    status, _, body = _request(
        application,
        "GET",
        "/api/status",
        headers={"Cookie": cookie},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["read_only"] is True
    assert payload["execution_enabled"] is False
    assert payload["positions"]["available"] is True
    assert payload["positions"]["expires_at"] is not None
    assert payload["regime"]["may_authorize_execution"] is False
    assert set(payload["positions"]) == {
        "available",
        "expires_at",
        "modified_at",
        "reason",
        "sha256",
        "size_bytes",
        "source_as_of",
        "source_generation",
        "stale",
    }
    assert "account" not in json.dumps(payload).lower()

    status, headers, body = _request(
        application,
        "GET",
        f"/api/positions?sha256={payload['positions']['sha256']}",
        headers={"Cookie": cookie},
    )
    assert status == 200
    assert b"SPY" in body
    assert b"CALL Short" in body
    assert b"$650.00" in body
    assert headers["x-frame-options"] == "SAMEORIGIN"
    assert "connect-src 'none'" in headers["content-security-policy"]
    raw = _raw_request(
        application,
        "GET",
        f"/api/positions?sha256={payload['positions']['sha256']}",
        headers={"Cookie": cookie},
    )
    raw_headers = raw.split(b"\r\n\r\n", 1)[0].lower()
    assert raw_headers.count(b"\r\nx-frame-options:") == 1
    assert b"\r\nx-frame-options: sameorigin" in raw_headers

    status, _, body = _request(
        application,
        "GET",
        "/api/regime_v2_shadow",
        headers={"Cookie": cookie},
    )
    assert status == 503
    regime = json.loads(body)
    assert regime["available"] is False
    assert regime["may_authorize_execution"] is False


def test_positions_route_never_serves_a_generation_newer_than_requested(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    now = datetime.now(timezone.utc)
    _write_positions_artifact(
        application,
        source_as_of=now,
        symbol="SPY",
    )
    cookie = _login(application)
    status, _, body = _request(
        application,
        "GET",
        "/api/status",
        headers={"Cookie": cookie},
    )
    assert status == 200
    first_sha256 = json.loads(body)["positions"]["sha256"]

    _write_positions_artifact(
        application,
        source_as_of=now + timedelta(seconds=1),
        symbol="QQQ",
    )
    status, _, body = _request(
        application,
        "GET",
        f"/api/positions?sha256={first_sha256}",
        headers={"Cookie": cookie},
    )
    assert status == 409
    assert b"QQQ" not in body
    assert b"Current position data is unavailable" in body

    status, _, _ = _request(
        application,
        "GET",
        "/api/positions",
        headers={"Cookie": cookie},
    )
    assert status == 400

    status, _, body = _request(
        application,
        "GET",
        "/api/status",
        headers={"Cookie": cookie},
    )
    second_sha256 = json.loads(body)["positions"]["sha256"]
    assert second_sha256 != first_sha256
    status, _, body = _request(
        application,
        "GET",
        f"/api/positions?sha256={second_sha256}",
        headers={"Cookie": cookie},
    )
    assert status == 200
    assert b"QQQ" in body


def test_disabled_route_rejects_before_reading_declared_body(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    cookie = _login(application)
    status, _, body = _request(
        application,
        "POST",
        "/api/execute_close_order",
        headers={
            "Cookie": cookie,
            "Content-Length": "1048576",
        },
    )

    assert status == 503
    assert json.loads(body) == {
        "code": "READ_ONLY_RUNTIME",
        "error": "Trading actions are disabled.",
        "execution_enabled": False,
        "read_only": True,
    }


def test_login_rejects_wrong_duplicate_and_oversized_inputs(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    wrong = json.dumps(
        {
            "username": "readonly-operator",
            "password": "wrong",
        }
    ).encode("utf-8")
    status, headers, _ = _request(
        application,
        "POST",
        "/api/login",
        body=wrong,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(wrong)),
        },
    )
    assert status == 403
    assert "set-cookie" not in headers

    duplicate = (
        b'{"username":"readonly-operator","username":"other",'
        b'"password":"correct-horse-battery-staple"}'
    )
    status, _, _ = _request(
        application,
        "POST",
        "/api/login",
        body=duplicate,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(duplicate)),
        },
    )
    assert status == 400

    status, _, _ = _request(
        application,
        "POST",
        "/api/login",
        headers={
            "Content-Type": "application/json",
            "Content-Length": "8193",
        },
    )
    assert status == 400

    nested = b"[" * 1_100 + b"0" + b"]" * 1_100
    status, _, _ = _request(
        application,
        "POST",
        "/api/login",
        body=nested,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(nested)),
        },
    )
    assert status == 400


def test_login_is_rate_limited_and_basic_auth_is_not_accepted(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    wrong = json.dumps(
        {
            "username": "readonly-operator",
            "password": "wrong",
        }
    ).encode("utf-8")
    for _ in range(MAX_LOGIN_FAILURES):
        status, _, _ = _request(
            application,
            "POST",
            "/api/login",
            body=wrong,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(wrong)),
            },
        )
        assert status == 403

    status, headers, body = _request(
        application,
        "POST",
        "/api/login",
        headers={"Content-Length": "1048576"},
    )
    assert status == 429
    assert headers["retry-after"] == str(LOGIN_FAILURE_WINDOW_SECONDS)
    assert json.loads(body) == {"error": "Too many login attempts"}

    basic_parent = tmp_path / "basic"
    basic_parent.mkdir()
    status, _, _ = _request(
        _write_application(basic_parent),
        "GET",
        "/api/status",
        headers={"Authorization": "Basic cmVhZG9ubHk6c2VjcmV0"},
    )
    assert status == 401


def test_secure_cookie_requires_explicit_operator_configuration(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    payload = json.dumps(
        {
            "username": "readonly-operator",
            "password": "correct-horse-battery-staple",
        }
    ).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
        "X-Forwarded-Proto": "https",
    }

    status, headers, _ = _request(
        application,
        "POST",
        "/api/login",
        body=payload,
        headers=request_headers,
    )
    assert status == 200
    assert "; Secure" not in headers["set-cookie"]
    assert headers["set-cookie"].startswith(
        f"{application.runtime.session_cookie_name}="
    )

    secure_application = replace(application, secure_cookie=True)
    status, headers, _ = _request(
        secure_application,
        "POST",
        "/api/login",
        body=payload,
        headers=request_headers,
    )
    assert status == 200
    assert "; Secure" in headers["set-cookie"]


def test_request_target_and_login_framing_are_strict(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    payload = json.dumps(
        {
            "username": "readonly-operator",
            "password": "correct-horse-battery-staple",
        }
    ).encode("utf-8")

    for target in ("http://[", "http://example.test/api/status"):
        status, _, _ = _request(application, "GET", target)
        assert status == 400

    invalid_headers = (
        {
            "Content-Length": str(len(payload)),
        },
        {
            "Content-Type": "text/plain",
            "Content-Length": str(len(payload)),
        },
        {
            "Content-Type": "application/json",
            "Content-Length": f"0{len(payload)}",
        },
    )
    for headers in invalid_headers:
        status, _, _ = _request(
            application,
            "POST",
            "/api/login",
            body=payload,
            headers=headers,
        )
        assert status == 400

    duplicate_lengths = Message()
    duplicate_lengths.add_header("Content-Type", "application/json")
    duplicate_lengths.add_header("Content-Length", str(len(payload)))
    duplicate_lengths.add_header("Content-Length", str(len(payload)))
    status, _, _ = _request(
        application,
        "POST",
        "/api/login",
        body=payload,
        headers=duplicate_lengths,
    )
    assert status == 400


def test_positions_artifact_rejects_legacy_actions_and_symlinks(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    artifact = _artifact_path(application)
    artifact.write_text(
        f"{POSITIONS_READ_ONLY_MARKER}"
        '<button data-close-position="legacy">Close</button>',
        encoding="utf-8",
    )
    os.chmod(artifact, 0o600)
    snapshot = application.positions_snapshot()
    assert snapshot.available is False
    assert snapshot.reason == "untrusted_artifact"

    artifact.unlink()
    target = tmp_path / "outside.html"
    target.write_text(POSITIONS_READ_ONLY_MARKER, encoding="utf-8")
    os.chmod(target, 0o600)
    artifact.symlink_to(target)
    snapshot = application.positions_snapshot()
    assert snapshot.available is False
    assert snapshot.reason == "unsafe_file"

    artifact.unlink()
    os.mkfifo(artifact, mode=0o600)
    snapshot = application.positions_snapshot()
    assert snapshot.available is False
    assert snapshot.reason == "unsafe_file"


def test_positions_artifact_open_is_nonblocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _write_application(tmp_path)
    artifact = _write_positions_artifact(application)
    original_open = os.open
    observed_flags: list[int] = []

    def inspecting_open(path, flags, *args, **kwargs):
        if path == artifact.name and kwargs.get("dir_fd") is not None:
            observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        "live_trading.positions_artifact.os.open",
        inspecting_open,
    )
    assert application.positions_snapshot().available is True
    assert observed_flags
    assert all(
        flags & getattr(os, "O_NONBLOCK", 0)
        for flags in observed_flags
    )


def test_positions_artifact_rejects_stale_and_future_timestamps(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    now = datetime.now(timezone.utc)
    artifact = _write_positions_artifact(application, source_as_of=now)

    old = now - timedelta(
        seconds=(
            application.runtime.max_snapshot_age_seconds + 1
        )
    )
    os.utime(artifact, (old.timestamp(), old.timestamp()))
    stale = application.positions_snapshot(now=now)
    assert stale.available is False
    assert stale.stale is True
    assert stale.reason == "stale"
    assert stale.sha256 is not None
    assert stale.modified_at == old.isoformat()

    source_old = now - timedelta(
        seconds=application.runtime.max_snapshot_age_seconds + 1
    )
    artifact = _write_positions_artifact(
        application,
        source_as_of=source_old,
    )
    os.utime(artifact, (now.timestamp(), now.timestamp()))
    source_stale = application.positions_snapshot(now=now)
    assert source_stale.available is False
    assert source_stale.stale is True
    assert source_stale.source_as_of == source_old.isoformat()

    future = now + timedelta(
        seconds=MAX_ARTIFACT_FUTURE_SKEW_SECONDS + 1
    )
    os.utime(artifact, (future.timestamp(), future.timestamp()))
    future_snapshot = application.positions_snapshot(now=now)
    assert future_snapshot.available is False
    assert future_snapshot.stale is False
    assert future_snapshot.reason == "future_timestamp"


def test_application_and_server_reject_unsafe_capability_or_bind(
    tmp_path: Path,
) -> None:
    application = _write_application(tmp_path)
    rendered = repr(application)
    assert "correct-horse-battery-staple" not in rendered
    assert "A9~strong" not in rendered
    with pytest.raises(ReadOnlyDashboardError, match="loopback"):
        create_server(application, host="0.0.0.0")
    with pytest.raises(ReadOnlyDashboardError, match="loopback"):
        create_server(application, host="localhost")
