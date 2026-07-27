from __future__ import annotations

import fcntl
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from live_trading.positions_artifact import (
    POSITIONS_READ_ONLY_MARKER,
    PositionRow,
    PositionsArtifactError,
    PositionsArtifactReader,
    PositionsArtifactSigningKey,
    PositionsSnapshot,
    build_positions_snapshot,
    inspect_positions_html,
    positions_identity_fingerprint,
    render_positions_html,
)
from live_trading.positions_artifact_publisher import (
    PositionsArtifactCommitUnknown,
    PositionsArtifactPublishError,
    PositionsArtifactPublisher,
)


NOW = datetime(2026, 7, 27, 18, 30, tzinfo=timezone.utc)
SIGNING_KEY = PositionsArtifactSigningKey.from_text(
    "test-positions-artifact-key-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
)
RUNTIME_BINDING = "a" * 64


def _osi_key(
    symbol: str,
    call_put: str,
    expiration_date: str,
    strike_price: object,
) -> str:
    try:
        expiration = datetime.strptime(
            expiration_date,
            "%Y-%m-%d",
        ).strftime("%y%m%d")
        strike = int(float(strike_price) * 1_000)
        root = symbol.replace(".", "").replace("-", "")[:6]
        return (
            f"{root.ljust(6, '-')}{expiration}"
            f"{call_put[0].upper()}{strike:08d}"
        )
    except (IndexError, TypeError, ValueError):
        return "SPY---260821C00650000"


def _option(
    *,
    symbol: str = "SPY",
    quantity: int = -2,
    call_put: str = "CALL",
    expiration_date: str = "2026-08-21",
    strike_price: float = 650.0,
    last_price: object = 1.25,
    price_paid: object = 2.50,
    market_value: object = -250.0,
    total_gain: object = 250.0,
    underlying_last_price: object = 640.25,
    osi_key: str | None = None,
    option_multiplier: object = 100,
    options_adjusted_flag: object = False,
    option_deliverables: object = "100 shares",
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        security_type="Option",
        quantity=quantity,
        last_price=last_price,
        price_paid=price_paid,
        market_value=market_value,
        total_gain=total_gain,
        call_put=call_put,
        expiration_date=expiration_date,
        strike_price=strike_price,
        underlying_last_price=underlying_last_price,
        osi_key=(
            osi_key
            if osi_key is not None
            else _osi_key(
                symbol,
                call_put,
                expiration_date,
                strike_price,
            )
        ),
        option_multiplier=option_multiplier,
        options_adjusted_flag=options_adjusted_flag,
        option_deliverables=option_deliverables,
        position_id="sensitive-position-id",
        lots_url="https://broker.invalid/private",
    )


def _equity(
    *,
    symbol: str = "BRK.B",
    quantity: object = 12.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        security_type="Stock",
        quantity=quantity,
        last_price=475.125,
        price_paid=425.25,
        market_value=5_939.06,
        total_gain=623.44,
        position_id="sensitive-equity-id",
    )


def _snapshot(
    positions: list[object] | None = None,
    *,
    source_as_of: datetime = NOW,
) -> PositionsSnapshot:
    return build_positions_snapshot(
        [_option(), _equity()] if positions is None else positions,
        broker_environment="production",
        source_as_of=source_as_of,
    )


def _publisher(tmp_path: Path) -> tuple[PositionsArtifactPublisher, Path]:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    artifacts = private / "artifacts"
    artifacts.mkdir(mode=0o700)
    path = artifacts / "positions.html"
    return (
        PositionsArtifactPublisher(
            path,
            max_source_age_seconds=300,
            signing_key=SIGNING_KEY,
            broker_environment="production",
            runtime_binding=RUNTIME_BINDING,
        ),
        path,
    )


def _render(
    snapshot: PositionsSnapshot,
    *,
    signing_key: PositionsArtifactSigningKey = SIGNING_KEY,
    runtime_binding: str = RUNTIME_BINDING,
) -> bytes:
    return render_positions_html(
        snapshot,
        signing_key=signing_key,
        runtime_binding=runtime_binding,
    )


def _inspect(
    payload: bytes,
    *,
    signing_key: PositionsArtifactSigningKey = SIGNING_KEY,
    broker_environment: str = "production",
    runtime_binding: str = RUNTIME_BINDING,
):
    return inspect_positions_html(
        payload,
        signing_key=signing_key,
        expected_broker_environment=broker_environment,
        expected_runtime_binding=runtime_binding,
    )


def _reader(path: Path) -> PositionsArtifactReader:
    return PositionsArtifactReader(
        path,
        max_age_seconds=300,
        signing_key=SIGNING_KEY,
        expected_broker_environment="production",
        expected_runtime_binding=RUNTIME_BINDING,
    )


def test_snapshot_is_primitive_canonical_and_redacted() -> None:
    snapshot = _snapshot([_option(symbol="QQQ"), _equity(symbol="AAPL")])

    assert [row.symbol for row in snapshot.rows] == ["AAPL", "QQQ"]
    assert snapshot.rows[0].quantity_microunits == 12_500_000
    assert snapshot.rows[1].quantity_microunits == -2_000_000
    assert snapshot.rows[1].strike_price_micros == 650_000_000
    assert snapshot.rows[1].side == "SHORT"
    assert snapshot.rows[1].valuation_status == "BEST_EFFORT_MARK"
    assert len(snapshot.source_generation) == 64
    rendered = repr(snapshot)
    assert "QQQ" not in rendered
    assert "sensitive-position-id" not in rendered
    assert "broker.invalid" not in rendered
    assert set(snapshot.rows[1].canonical_value()) == {
        "call_put",
        "expiration_date",
        "mark_price_micros",
        "market_value_cents",
        "option_adjusted",
        "option_deliverables",
        "option_multiplier",
        "option_osi_key",
        "price_paid_micros",
        "quantity_microunits",
        "security_type",
        "strike_price_micros",
        "symbol",
        "total_gain_cents",
        "underlying_price_micros",
    }


def test_identity_fingerprint_ignores_marks_but_detects_quantity() -> None:
    first = _option()
    changed_mark = _option(last_price=99.25)
    changed_quantity = _option(quantity=-3)
    changed_contract = _option(
        osi_key="SPYW--260821C00650000",
    )

    assert positions_identity_fingerprint([first]) == (
        positions_identity_fingerprint([changed_mark])
    )
    assert positions_identity_fingerprint([first]) != (
        positions_identity_fingerprint([changed_quantity])
    )
    assert positions_identity_fingerprint([first]) != (
        positions_identity_fingerprint([changed_contract])
    )


def test_renderer_is_deterministic_static_and_self_describing() -> None:
    snapshot = _snapshot()

    first = _render(snapshot)
    second = _render(snapshot)
    metadata = _inspect(first)
    text = first.decode("utf-8")

    assert first == second
    assert metadata.source_as_of == NOW
    assert metadata.source_generation == snapshot.source_generation
    assert POSITIONS_READ_ONLY_MARKER in text
    assert "SPY" in text
    assert "BRK.B" in text
    assert "CALL Short" in text
    assert "Standard ×100 · SPY---260821C00650000" in text
    assert "$650.00" in text
    assert "-$250.00" in text
    assert "$-250.00" not in text
    assert "12.5" in text
    assert "best-effort display data" in text
    for marker in (
        "<script",
        "<form",
        "<button",
        "<iframe",
        "<object",
        "<embed",
        "<svg",
        "href=",
        "src=",
        "onclick=",
        "javascript:",
        "position_id",
        "lots_url",
        "sensitive-position-id",
        "broker.invalid",
    ):
        assert marker not in text.casefold()


def test_confirmed_empty_snapshot_is_distinct_and_valid() -> None:
    payload = _render(_snapshot([]))
    metadata = _inspect(payload)

    assert b"Confirmed empty portfolio" in payload
    assert b"no open positions" in payload
    assert metadata.source_as_of == NOW


@pytest.mark.parametrize(
    ("position", "code"),
    (
        (_option(symbol='SPY"><script>alert(1)</script>'), "position_symbol_invalid"),
        (_option(quantity=0), "position_quantity_invalid"),
        (_option(quantity=0.5), "option_contract_invalid"),
        (_option(call_put="UNKNOWN"), "option_contract_invalid"),
        (_option(strike_price=float("nan")), "position_strike_invalid"),
        (_option(last_price=float("inf")), "position_mark_invalid"),
        (
            _option(options_adjusted_flag=True),
            "adjusted_option_unsupported",
        ),
        (
            _option(options_adjusted_flag=None),
            "option_adjustment_status_invalid",
        ),
        (
            _option(option_multiplier=10),
            "adjusted_option_unsupported",
        ),
        (
            _option(osi_key="SPY---260821P00650000"),
            "option_osi_key_mismatch",
        ),
        (
            _option(osi_key="not-an-osi-key"),
            "option_osi_key_invalid",
        ),
        (
            _option(option_deliverables="unsafe\nvalue"),
            "option_deliverables_invalid",
        ),
        (
            _option(option_deliverables="50 shares plus cash"),
            "adjusted_option_unsupported",
        ),
        (
            SimpleNamespace(
                symbol="SPY",
                security_type="Unknown",
                quantity=1,
            ),
            "position_security_type_invalid",
        ),
    ),
)
def test_snapshot_rejects_invalid_required_data(
    position: object,
    code: str,
) -> None:
    with pytest.raises(PositionsArtifactError) as caught:
        _snapshot([position])

    assert caught.value.code == code


def test_optional_valuation_absence_is_visible_not_coerced_to_zero() -> None:
    snapshot = _snapshot(
        [
            _option(
                last_price=None,
                price_paid=None,
                market_value=None,
                total_gain=None,
                underlying_last_price=None,
            )
        ]
    )
    row = snapshot.rows[0]
    payload = _render(snapshot)

    assert row.mark_price_micros is None
    assert row.market_value_cents is None
    assert row.valuation_status == "POSITION_ONLY"
    assert b"Position only" in payload
    assert payload.count("—".encode("utf-8")) >= 5


def test_exact_types_and_generation_are_enforced() -> None:
    valid = _snapshot()

    with pytest.raises(
        PositionsArtifactError,
        match="snapshot_generation_mismatch",
    ):
        replace(valid, source_generation="a" * 64)
    with pytest.raises(
        PositionsArtifactError,
        match="snapshot_rows_invalid",
    ):
        PositionsSnapshot(
            schema_version=1,
            source="etrade_portfolio",
            broker_environment="production",
            source_as_of=NOW,
            source_generation=valid.source_generation,
            rows=list(valid.rows),  # type: ignore[arg-type]
        )
    with pytest.raises(
        PositionsArtifactError,
        match="position_quantity_invalid",
    ):
        replace(valid.rows[0], quantity_microunits=True)
    with pytest.raises(
        PositionsArtifactError,
        match="option_osi_key_invalid",
    ):
        replace(valid.rows[1], option_osi_key="not-an-osi-key")
    with pytest.raises(
        PositionsArtifactError,
        match="adjusted_option_unsupported",
    ):
        replace(
            valid.rows[1],
            option_deliverables="50 shares plus cash",
        )
    with pytest.raises(
        PositionsArtifactError,
        match="positions_input_invalid",
    ):
        build_positions_snapshot(
            iter([_option()]),  # type: ignore[arg-type]
            broker_environment="production",
            source_as_of=NOW,
        )


def test_space_padded_osi_identity_is_visibly_preserved() -> None:
    payload = _render(
        _snapshot(
            [
                _option(
                    osi_key="SPY   260821C00650000",
                )
            ]
        )
    )
    text = payload.decode("utf-8")

    assert "Standard ×100 · SPY   260821C00650000" in text
    assert "white-space:pre" in text


@pytest.mark.parametrize(
    "injection",
    (
        b"<script>alert(1)</script>",
        b"<form></form>",
        b'<div onclick="alert(1)">x</div>',
        b"<svg></svg>",
        b"<style>@import 'https://invalid';</style>",
        b'<meta http-equiv="refresh" content="0">',
    ),
)
def test_static_inspector_rejects_active_html(injection: bytes) -> None:
    payload = _render(_snapshot())
    tampered = payload.replace(b"</main>", injection + b"</main>")

    with pytest.raises(PositionsArtifactError):
        _inspect(tampered)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.replace(b"SPY", b"QQQ", 1),
        lambda payload: payload.replace(
            b"</style>",
            b".banner{display:none}.group:before{content:'Execution enabled'}"
            b"</style>",
            1,
        ),
        lambda payload: payload.replace(
            b"2026-07-27T18:30:00+00:00",
            b"2026-07-27T18:31:00+00:00",
            1,
        ),
    ),
)
def test_signature_rejects_static_semantic_or_freshness_tampering(
    mutate,
) -> None:
    payload = _render(_snapshot())
    tampered = mutate(payload)
    assert tampered != payload

    with pytest.raises(
        PositionsArtifactError,
        match="artifact_signature_invalid",
    ):
        _inspect(tampered)


def test_runtime_binding_and_environment_are_closed() -> None:
    snapshot = _snapshot()
    other_binding = "b" * 64
    payload = _render(snapshot, runtime_binding=other_binding)

    with pytest.raises(
        PositionsArtifactError,
        match="artifact_metadata_invalid",
    ):
        _inspect(payload)
    with pytest.raises(
        PositionsArtifactError,
        match="artifact_metadata_invalid",
    ):
        _inspect(
            payload,
            broker_environment="sandbox",
            runtime_binding=other_binding,
        )


def test_publisher_rejects_snapshot_from_another_environment(
    tmp_path: Path,
) -> None:
    publisher, path = _publisher(tmp_path)
    now = datetime.now(timezone.utc)
    sandbox = build_positions_snapshot(
        [_option()],
        broker_environment="sandbox",
        source_as_of=now,
    )

    with pytest.raises(
        PositionsArtifactPublishError,
        match="positions_snapshot_environment_mismatch",
    ):
        publisher.publish(sandbox, now=now)

    assert not path.exists()


def test_publisher_writes_exact_owner_only_artifact_accepted_by_reader(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    snapshot = _snapshot(source_as_of=now)

    receipt = publisher.publish(snapshot, now=now)
    reader = _reader(path)
    observed = reader.read(now=now)

    assert receipt.state == "committed_durable"
    assert receipt.sha256 == observed.sha256
    assert receipt.source_generation == snapshot.source_generation
    assert stat_mode(path) == 0o600
    assert observed.available is True
    assert observed.source_as_of == now.isoformat()
    assert observed.source_generation == snapshot.source_generation
    assert observed.expires_at is not None
    assert path.read_bytes() == observed.content
    lock = path.parent / f".{path.name}.publish.lock"
    assert stat_mode(lock) == 0o600


def test_identical_publish_is_noop_and_does_not_refresh_mtime(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    snapshot = _snapshot(source_as_of=now)
    first = publisher.publish(snapshot, now=now)
    first_mtime = path.stat().st_mtime_ns

    second = publisher.publish(
        snapshot,
        now=now + timedelta(seconds=10),
    )

    assert first.state == "committed_durable"
    assert second.state == "already_current"
    assert path.stat().st_mtime_ns == first_mtime


def test_already_current_revalidates_configured_parent_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, path = _publisher(tmp_path)
    now = datetime.now(timezone.utc)
    snapshot = _snapshot(source_as_of=now)
    publisher.publish(snapshot, now=now)
    module = __import__(
        "live_trading.positions_artifact_publisher",
        fromlist=["_read_existing"],
    )
    original_read = module._read_existing
    detached = path.parent.with_name("detached-artifacts")

    def swap_parent(*args, **kwargs):
        result = original_read(*args, **kwargs)
        path.parent.rename(detached)
        path.parent.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(module, "_read_existing", swap_parent)
    with pytest.raises(
        PositionsArtifactPublishError,
        match="positions_artifact_parent_changed",
    ):
        publisher.publish(snapshot, now=now)

    assert not path.exists()
    assert (detached / path.name).exists()


def test_publisher_rejects_source_regression_and_conflict(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    newer = _snapshot(
        [_option(symbol="QQQ")],
        source_as_of=now,
    )
    publisher.publish(newer, now=now)
    original = path.read_bytes()

    with pytest.raises(
        PositionsArtifactPublishError,
        match="positions_snapshot_regression",
    ):
        publisher.publish(
            _snapshot(
                [_option(symbol="SPY")],
                source_as_of=now - timedelta(seconds=1),
            ),
            now=now,
        )
    with pytest.raises(
        PositionsArtifactPublishError,
        match="positions_snapshot_generation_conflict",
    ):
        publisher.publish(
            _snapshot(
                [_option(symbol="SPY")],
                source_as_of=now,
            ),
            now=now,
        )

    assert path.read_bytes() == original


def test_source_freshness_cannot_be_reset_by_file_mtime(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    source_time = now - timedelta(seconds=299)
    publisher.publish(
        _snapshot(source_as_of=source_time),
        now=now,
    )
    later = now + timedelta(seconds=2)
    os.utime(path, (later.timestamp(), later.timestamp()))

    observed = _reader(path).read(now=later)

    assert observed.available is False
    assert observed.stale is True
    assert observed.reason == "stale"
    assert observed.source_as_of == source_time.isoformat()


@pytest.mark.parametrize("kind", ("symlink", "hardlink", "fifo"))
def test_publisher_rejects_unsafe_existing_targets(
    tmp_path: Path,
    kind: str,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    os.chmod(outside, 0o600)
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, path)
    else:
        os.mkfifo(path, mode=0o600)

    with pytest.raises(
        PositionsArtifactPublishError,
        match="existing_positions_artifact_unsafe",
    ):
        publisher.publish(_snapshot(source_as_of=now), now=now)

    assert outside.read_bytes() == b"preserve"


def test_publisher_does_not_repair_an_unsafe_existing_lock(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    lock = path.parent / f".{path.name}.publish.lock"
    lock.write_bytes(b"untrusted")
    os.chmod(lock, 0o644)

    with pytest.raises(
        PositionsArtifactPublishError,
        match="positions_publish_lock_unsafe",
    ):
        publisher.publish(_snapshot(source_as_of=now), now=now)

    assert stat_mode(lock) == 0o644
    assert lock.read_bytes() == b"untrusted"
    assert not path.exists()


def test_publisher_lock_contention_fails_without_waiting(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    lock = path.parent / f".{path.name}.publish.lock"
    lock.touch(mode=0o600)
    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            PositionsArtifactPublishError,
            match="positions_publisher_busy",
        ):
            publisher.publish(_snapshot(source_as_of=now), now=now)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert not path.exists()


def test_publisher_rejects_replaced_lock_path_after_flock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    lock = path.parent / f".{path.name}.publish.lock"
    detached = path.parent / "detached.lock"
    module = __import__(
        "live_trading.positions_artifact_publisher",
        fromlist=["fcntl"],
    )
    original_flock = module.fcntl.flock
    swapped = False

    def replace_lock(descriptor: int, operation: int):
        nonlocal swapped
        result = original_flock(descriptor, operation)
        if not swapped and operation & fcntl.LOCK_EX:
            swapped = True
            lock.rename(detached)
            lock.touch(mode=0o600)
            os.chmod(lock, 0o600)
        return result

    monkeypatch.setattr(module.fcntl, "flock", replace_lock)

    with pytest.raises(
        PositionsArtifactPublishError,
        match="positions_publish_lock_changed",
    ):
        publisher.publish(_snapshot(source_as_of=now), now=now)

    assert not path.exists()


def test_failure_before_replace_preserves_previous_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    original_snapshot = _snapshot(
        [_option(symbol="SPY")],
        source_as_of=now,
    )
    publisher.publish(original_snapshot, now=now)
    original = path.read_bytes()

    def fail_replace(*_args, **_kwargs):
        raise PermissionError("simulated")

    monkeypatch.setattr(
        "live_trading.positions_artifact_publisher.os.replace",
        fail_replace,
    )
    with pytest.raises(PositionsArtifactCommitUnknown):
        publisher.publish(
            _snapshot(
                [_option(symbol="QQQ")],
                source_as_of=now + timedelta(seconds=1),
            ),
            now=now + timedelta(seconds=1),
        )

    assert path.read_bytes() == original
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_same_size_temp_mutation_cannot_receive_durable_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    module = __import__(
        "live_trading.positions_artifact_publisher",
        fromlist=["os"],
    )
    original_replace = module.os.replace

    def corrupt_then_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ):
        descriptor = os.open(
            source,
            os.O_RDWR,
            dir_fd=src_dir_fd,
        )
        try:
            os.pwrite(descriptor, b"X", 0)
        finally:
            os.close(descriptor)
        return original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(module.os, "replace", corrupt_then_replace)

    with pytest.raises(PositionsArtifactCommitUnknown):
        publisher.publish(_snapshot(source_as_of=now), now=now)

    assert path.exists()
    observed = _reader(path).read(now=now)
    assert observed.available is False
    assert observed.reason == "untrusted_artifact"


def test_directory_fsync_failure_is_commit_unknown_not_false_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    original_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated directory fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(
        "live_trading.positions_artifact_publisher.os.fsync",
        fail_directory_fsync,
    )
    snapshot = _snapshot(source_as_of=now)

    with pytest.raises(PositionsArtifactCommitUnknown) as caught:
        publisher.publish(snapshot, now=now)

    assert caught.value.sha256
    assert path.exists()
    assert snapshot.source_generation.encode("ascii") in path.read_bytes()


def test_reader_rejects_wrong_mode_and_tampered_static_contract(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    publisher.publish(_snapshot(source_as_of=now), now=now)
    os.chmod(path, 0o700)

    wrong_mode = _reader(path).read(now=now)
    assert wrong_mode.available is False
    assert wrong_mode.reason == "unsafe_file"

    os.chmod(path, 0o600)
    payload = path.read_bytes().replace(
        b"</main>",
        b"<script>alert(1)</script></main>",
    )
    path.write_bytes(payload)
    os.chmod(path, 0o600)
    tampered = _reader(path).read(now=now)
    assert tampered.available is False
    assert tampered.reason == "untrusted_artifact"


def test_reader_rejects_parent_mode_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    publisher.publish(_snapshot(source_as_of=now), now=now)
    module = __import__(
        "live_trading.positions_artifact",
        fromlist=["os"],
    )
    original_open = module.os.open
    changed = False

    def chmod_parent_then_open(target, *args, **kwargs):
        nonlocal changed
        if not changed and Path(target) == path.parent:
            changed = True
            os.chmod(path.parent, 0o777)
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", chmod_parent_then_open)
    try:
        observed = _reader(path).read(now=now)
    finally:
        os.chmod(path.parent, 0o700)

    assert observed.available is False
    assert observed.reason == "unsafe_parent"


def test_reader_converts_filesystem_io_error_to_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    publisher.publish(_snapshot(source_as_of=now), now=now)

    def fail_read(*_args, **_kwargs):
        raise OSError("simulated EIO")

    monkeypatch.setattr(
        "live_trading.positions_artifact.os.read",
        fail_read,
    )

    observed = _reader(path).read(now=now)

    assert observed.available is False
    assert observed.reason == "artifact_io_error"


def test_reader_rejects_artifact_from_different_renderer_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    publisher, path = _publisher(tmp_path)
    module = __import__(
        "live_trading.positions_artifact",
        fromlist=["POSITIONS_RENDERER_BUILD_SHA256"],
    )
    current_build = module.POSITIONS_RENDERER_BUILD_SHA256
    monkeypatch.setattr(
        module,
        "POSITIONS_RENDERER_BUILD_SHA256",
        "b" * 64,
    )
    publisher.publish(_snapshot(source_as_of=now), now=now)
    monkeypatch.setattr(
        module,
        "POSITIONS_RENDERER_BUILD_SHA256",
        current_build,
    )

    observed = _reader(path).read(now=now)

    assert observed.available is False
    assert observed.reason == "untrusted_artifact"


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
