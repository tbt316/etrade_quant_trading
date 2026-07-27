"""Pure, read-only contract for the static positions artifact.

The artifact contains a sanitized projection of one complete E*TRADE
portfolio observation.  It deliberately excludes account identifiers, order
identifiers, broker payloads, URLs, credentials, and inferred spread
pairings.  Rendering is deterministic and performs no filesystem, network, or
clock access.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo


POSITIONS_ARTIFACT_SCHEMA_VERSION = 1
POSITIONS_READ_ONLY_MARKER = (
    "Read only — all E*TRADE order actions are disabled"
)
MAX_POSITIONS_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_PUBLISHED_POSITIONS_BYTES = 2 * 1024 * 1024
MAX_POSITION_ROWS = 2_000
MAX_ARTIFACT_FUTURE_SKEW_SECONDS = 5
POSITIONS_ARTIFACT_PREFIX = (
    "<!doctype html>\n"
    "<!-- etrade-read-only-positions:v1 -->\n"
)
_SIGNATURE_PLACEHOLDER = "0" * 64
_SIGNATURE_DOMAIN = b"etrade-positions-artifact-hmac.v1\0"
try:
    POSITIONS_RENDERER_BUILD_SHA256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
except OSError as exc:  # pragma: no cover - fail-closed installation error
    raise RuntimeError(
        "positions renderer source identity is unavailable"
    ) from exc

_SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,14}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CLOSED_SECURITY_TYPES = frozenset({"EQUITY", "OPTION"})
_CLOSED_BROKER_ENVIRONMENTS = frozenset({"sandbox", "production"})
_CLOSED_CALL_PUT = frozenset({"CALL", "PUT"})
_MAX_QUANTITY_MICROUNITS = 10**15
_MAX_PRICE_MICROS = 10**18
_MAX_CURRENCY_CENTS = 10**18
_MICRO_UNITS = Decimal("1000000")
_CENT_UNITS = Decimal("100")
_NEW_YORK = ZoneInfo("America/New_York")
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_FORBIDDEN_ACTIVE_MARKERS = (
    "<script",
    "<form",
    "<button",
    "<a ",
    "<iframe",
    "<object",
    "<embed",
    "<svg",
    "<link",
    "<img",
    "<video",
    "<audio",
    "<meta http-equiv",
    "javascript:",
    "@import",
    "url(",
    "data-close-position",
    "action-cell",
    "/api/execute_",
    "/api/review_close_position",
    "/api/close_position",
    "/api/settings",
    "/api/verify_pin",
    "/refresh",
)
_ALLOWED_TAGS = frozenset(
    {
        "html",
        "head",
        "meta",
        "title",
        "style",
        "body",
        "header",
        "main",
        "section",
        "div",
        "p",
        "span",
        "strong",
        "h1",
        "h2",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "small",
    }
)
_GLOBAL_ATTRIBUTES = frozenset({"class", "role", "aria-label"})
_TAG_ATTRIBUTES = {
    "html": frozenset({"lang"}),
    "meta": frozenset({"charset", "name", "content"}),
    "body": frozenset(
        {
            "data-broker-environment",
            "data-positions-artifact-schema",
            "data-renderer-build-sha256",
            "data-runtime-binding",
            "data-source-generation",
        }
    ),
    "th": frozenset({"scope"}),
}


class PositionsArtifactError(RuntimeError):
    """A positions snapshot or static artifact violated its closed contract."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True, repr=False)
class PositionsArtifactSigningKey:
    """Opaque HMAC key shared only by the isolated publisher and reader."""

    value: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.value) is not bytes
            or not 32 <= len(self.value) <= 4096
        ):
            raise ValueError(
                "positions artifact signing key must be 32-4096 bytes"
            )

    @classmethod
    def from_text(cls, value: str) -> "PositionsArtifactSigningKey":
        if type(value) is not str:
            raise TypeError("positions artifact signing key must be text")
        return cls(value.encode("utf-8"))

    def __repr__(self) -> str:
        return "PositionsArtifactSigningKey([REDACTED])"


@dataclass(frozen=True, slots=True, repr=False)
class PositionRow:
    """One display-only broker position using fixed integer units."""

    symbol: str
    security_type: str
    quantity_microunits: int
    mark_price_micros: int | None
    price_paid_micros: int | None
    market_value_cents: int | None
    total_gain_cents: int | None
    call_put: str | None
    expiration_date: date | None
    strike_price_micros: int | None
    underlying_price_micros: int | None
    option_osi_key: str | None
    option_multiplier: int | None
    option_adjusted: bool | None
    option_deliverables: str | None

    def __post_init__(self) -> None:
        if (
            type(self.symbol) is not str
            or _SYMBOL.fullmatch(self.symbol) is None
        ):
            raise PositionsArtifactError("position_symbol_invalid")
        if (
            type(self.security_type) is not str
            or self.security_type not in _CLOSED_SECURITY_TYPES
        ):
            raise PositionsArtifactError("position_security_type_invalid")
        _bounded_exact_int(
            self.quantity_microunits,
            "position_quantity_invalid",
            -_MAX_QUANTITY_MICROUNITS,
            _MAX_QUANTITY_MICROUNITS,
            nonzero=True,
        )
        _optional_bounded_exact_int(
            self.mark_price_micros,
            "position_mark_invalid",
            0,
            _MAX_PRICE_MICROS,
        )
        _optional_bounded_exact_int(
            self.price_paid_micros,
            "position_price_paid_invalid",
            0,
            _MAX_PRICE_MICROS,
        )
        _optional_bounded_exact_int(
            self.market_value_cents,
            "position_market_value_invalid",
            -_MAX_CURRENCY_CENTS,
            _MAX_CURRENCY_CENTS,
        )
        _optional_bounded_exact_int(
            self.total_gain_cents,
            "position_total_gain_invalid",
            -_MAX_CURRENCY_CENTS,
            _MAX_CURRENCY_CENTS,
        )
        _optional_bounded_exact_int(
            self.strike_price_micros,
            "position_strike_invalid",
            1,
            _MAX_PRICE_MICROS,
        )
        _optional_bounded_exact_int(
            self.underlying_price_micros,
            "position_underlying_invalid",
            1,
            _MAX_PRICE_MICROS,
        )
        if self.security_type == "OPTION":
            if (
                type(self.call_put) is not str
                or self.call_put not in _CLOSED_CALL_PUT
                or type(self.expiration_date) is not date
                or self.strike_price_micros is None
                or self.quantity_microunits % 1_000_000
                or type(self.option_osi_key) is not str
                or not _valid_option_text(
                    self.option_osi_key,
                    maximum=64,
                )
                or type(self.option_multiplier) is not int
                or isinstance(self.option_multiplier, bool)
                or self.option_multiplier != 100
                or self.option_adjusted is not False
                or (
                    self.option_deliverables is not None
                    and type(self.option_deliverables) is not str
                )
            ):
                raise PositionsArtifactError("option_contract_invalid")
            if not 1970 <= self.expiration_date.year <= 2200:
                raise PositionsArtifactError("option_expiration_invalid")
            _option_osi_key(
                self.option_osi_key,
                symbol=self.symbol,
                call_put=self.call_put,
                expiration=self.expiration_date,
                strike_price_micros=self.strike_price_micros,
            )
            if (
                _option_deliverables(
                    self.option_deliverables,
                    symbol=self.symbol,
                )
                != self.option_deliverables
            ):
                raise PositionsArtifactError(
                    "option_deliverables_invalid"
                )
        elif any(
            value is not None
            for value in (
                self.call_put,
                self.expiration_date,
                self.strike_price_micros,
                self.underlying_price_micros,
                self.option_osi_key,
                self.option_multiplier,
                self.option_adjusted,
                self.option_deliverables,
            )
        ):
            raise PositionsArtifactError("equity_contract_invalid")

    def __repr__(self) -> str:
        return "PositionRow([REDACTED])"

    @property
    def side(self) -> str:
        return "LONG" if self.quantity_microunits > 0 else "SHORT"

    @property
    def valuation_status(self) -> str:
        return (
            "BEST_EFFORT_MARK"
            if self.mark_price_micros is not None
            else "POSITION_ONLY"
        )

    def canonical_value(self) -> dict[str, Any]:
        return {
            "call_put": self.call_put,
            "expiration_date": (
                self.expiration_date.isoformat()
                if self.expiration_date is not None
                else None
            ),
            "mark_price_micros": self.mark_price_micros,
            "market_value_cents": self.market_value_cents,
            "option_adjusted": self.option_adjusted,
            "option_deliverables": self.option_deliverables,
            "option_multiplier": self.option_multiplier,
            "option_osi_key": self.option_osi_key,
            "price_paid_micros": self.price_paid_micros,
            "quantity_microunits": self.quantity_microunits,
            "security_type": self.security_type,
            "strike_price_micros": self.strike_price_micros,
            "symbol": self.symbol,
            "total_gain_cents": self.total_gain_cents,
            "underlying_price_micros": self.underlying_price_micros,
        }


@dataclass(frozen=True, slots=True, repr=False)
class PositionsSnapshot:
    """One complete, sanitized E*TRADE portfolio observation."""

    schema_version: int
    source: str
    broker_environment: str
    source_as_of: datetime
    source_generation: str
    rows: tuple[PositionRow, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != POSITIONS_ARTIFACT_SCHEMA_VERSION
        ):
            raise PositionsArtifactError("snapshot_schema_invalid")
        if (
            type(self.source) is not str
            or self.source != "etrade_portfolio"
        ):
            raise PositionsArtifactError("snapshot_source_invalid")
        if (
            type(self.broker_environment) is not str
            or self.broker_environment not in _CLOSED_BROKER_ENVIRONMENTS
        ):
            raise PositionsArtifactError("snapshot_environment_invalid")
        if (
            type(self.source_as_of) is not datetime
            or self.source_as_of.tzinfo is None
            or self.source_as_of.utcoffset() is None
            or self.source_as_of.utcoffset().total_seconds() != 0
        ):
            raise PositionsArtifactError("snapshot_source_time_invalid")
        if (
            type(self.source_generation) is not str
            or _SHA256.fullmatch(self.source_generation) is None
        ):
            raise PositionsArtifactError("snapshot_generation_invalid")
        if (
            type(self.rows) is not tuple
            or len(self.rows) > MAX_POSITION_ROWS
            or any(type(row) is not PositionRow for row in self.rows)
        ):
            raise PositionsArtifactError("snapshot_rows_invalid")
        if tuple(sorted(self.rows, key=_row_sort_key)) != self.rows:
            raise PositionsArtifactError("snapshot_rows_not_canonical")
        expected = _snapshot_generation(
            self.broker_environment,
            self.source_as_of,
            self.rows,
        )
        if self.source_generation != expected:
            raise PositionsArtifactError("snapshot_generation_mismatch")

    def __repr__(self) -> str:
        return (
            "PositionsSnapshot("
            f"schema_version={self.schema_version}, "
            f"source={self.source!r}, "
            f"broker_environment={self.broker_environment!r}, "
            f"source_as_of={self.source_as_of.isoformat()!r}, "
            f"source_generation={self.source_generation!r}, "
            "rows=[REDACTED])"
        )


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    schema_version: int
    source_as_of: datetime
    source_generation: str
    broker_environment: str
    runtime_binding: str
    renderer_build_sha256: str
    signature: str


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    available: bool
    stale: bool
    content: bytes | None = field(repr=False)
    sha256: str | None
    modified_at: str | None
    size_bytes: int | None
    reason: str | None
    source_as_of: str | None
    source_generation: str | None
    expires_at: str | None


def _require_signing_key(value: PositionsArtifactSigningKey) -> None:
    if type(value) is not PositionsArtifactSigningKey:
        raise TypeError(
            "signing_key must be an exact PositionsArtifactSigningKey"
        )


def _require_sha256(value: object, code: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PositionsArtifactError(code)


def _signature_meta_line(signature: str) -> str:
    return (
        '  <meta name="etrade-positions-signature" '
        f'content="{signature}">'
    )


def build_positions_snapshot(
    positions: Sequence[object],
    *,
    broker_environment: str,
    source_as_of: datetime | None = None,
) -> PositionsSnapshot:
    """Project one complete legacy portfolio result into exact display DTOs."""

    if type(positions) not in {list, tuple}:
        raise PositionsArtifactError("positions_input_invalid")
    if len(positions) > MAX_POSITION_ROWS:
        raise PositionsArtifactError("positions_input_oversized")
    observed = _utc_datetime(source_as_of)
    rows = tuple(sorted((_position_row(value) for value in positions), key=_row_sort_key))
    generation = _snapshot_generation(
        broker_environment,
        observed,
        rows,
    )
    return PositionsSnapshot(
        schema_version=POSITIONS_ARTIFACT_SCHEMA_VERSION,
        source="etrade_portfolio",
        broker_environment=broker_environment,
        source_as_of=observed,
        source_generation=generation,
        rows=rows,
    )


def positions_identity_fingerprint(positions: Sequence[object]) -> str:
    """Hash only stable position identity and quantity for scan comparison."""

    if type(positions) not in {list, tuple}:
        raise PositionsArtifactError("positions_input_invalid")
    if len(positions) > MAX_POSITION_ROWS:
        raise PositionsArtifactError("positions_input_oversized")
    identities = []
    for row in (_position_row(value) for value in positions):
        identities.append(
            {
                "call_put": row.call_put,
                "expiration_date": (
                    row.expiration_date.isoformat()
                    if row.expiration_date is not None
                    else None
                ),
                "quantity_microunits": row.quantity_microunits,
                "security_type": row.security_type,
                "strike_price_micros": row.strike_price_micros,
                "symbol": row.symbol,
                "option_adjusted": row.option_adjusted,
                "option_deliverables": row.option_deliverables,
                "option_multiplier": row.option_multiplier,
                "option_osi_key": row.option_osi_key,
            }
        )
    identities.sort(
        key=lambda value: (
            value["symbol"],
            value["security_type"],
            value["expiration_date"] or "",
            value["call_put"] or "",
            value["strike_price_micros"] or 0,
            value["quantity_microunits"],
        )
    )
    return hashlib.sha256(
        b"etrade-positions-identity.v1\0"
        + json.dumps(
            identities,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def render_positions_html(
    snapshot: PositionsSnapshot,
    *,
    signing_key: PositionsArtifactSigningKey,
    runtime_binding: str,
) -> bytes:
    """Render deterministic, static, self-contained positions HTML."""

    if type(snapshot) is not PositionsSnapshot:
        raise TypeError("snapshot must be an exact PositionsSnapshot")
    _require_signing_key(signing_key)
    _require_sha256(runtime_binding, "artifact_runtime_binding_invalid")
    observed_iso = snapshot.source_as_of.isoformat()
    observed_et = snapshot.source_as_of.astimezone(_NEW_YORK).strftime(
        "%b %d, %Y %I:%M:%S %p %Z"
    )
    groups: dict[str, list[PositionRow]] = {}
    for row in snapshot.rows:
        groups.setdefault(row.symbol, []).append(row)

    if groups:
        sections = "".join(
            _render_symbol_group(symbol, rows, snapshot.source_as_of)
            for symbol, rows in groups.items()
        )
    else:
        sections = (
            '<section class="empty" role="status">'
            "<h2>Confirmed empty portfolio</h2>"
            "<p>E*TRADE returned a complete portfolio snapshot with no open "
            "positions.</p></section>"
        )

    option_count = sum(
        row.security_type == "OPTION" for row in snapshot.rows
    )
    equity_count = len(snapshot.rows) - option_count
    symbol_count = len(groups)
    html_document = f"""{POSITIONS_ARTIFACT_PREFIX}<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="etrade-positions-schema" content="1">
  <meta name="etrade-positions-source-as-of" content="{html.escape(observed_iso, quote=True)}">
  <meta name="etrade-positions-source-generation" content="{snapshot.source_generation}">
  <meta name="etrade-positions-broker-environment" content="{snapshot.broker_environment}">
  <meta name="etrade-positions-runtime-binding" content="{runtime_binding}">
  <meta name="etrade-positions-renderer-build-sha256" content="{POSITIONS_RENDERER_BUILD_SHA256}">
{_signature_meta_line(_SIGNATURE_PLACEHOLDER)}
  <title>Read-only E*TRADE positions</title>
  <style>
    :root{{color-scheme:dark;--bg:#07111f;--panel:#101d2d;--line:#26384e;
    --muted:#9eb0c7;--text:#edf4fc;--blue:#60a5fa;--green:#5ee0a0;
    --red:#fda4af;--amber:#fbbf24}}*{{box-sizing:border-box}}body{{margin:0;
    background:var(--bg);color:var(--text);font:14px/1.45 system-ui,
    -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{padding:20px;
    border-bottom:1px solid var(--line);background:#0b1726}}h1,h2,p{{margin:0}}
    h1{{font-size:22px}}.banner{{margin-bottom:12px;padding:10px 12px;
    border:1px solid #2563eb;border-radius:9px;background:#172554;
    color:#bfdbfe;font-weight:750}}.provenance{{display:flex;gap:8px 18px;
    flex-wrap:wrap;margin-top:9px;color:var(--muted)}}.summary{{display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;padding:16px 20px}}
    .metric,.group,.empty{{border:1px solid var(--line);border-radius:12px;
    background:var(--panel)}}.metric{{padding:13px}}.metric strong{{display:block;
    font-size:20px;color:white}}.metric span{{color:var(--muted)}}main{{display:grid;
    gap:14px;padding:0 20px 22px}}.group{{overflow:hidden}}.group-title{{display:flex;
    align-items:center;justify-content:space-between;gap:10px;padding:13px 15px;
    border-bottom:1px solid var(--line)}}h2{{font-size:18px}}.count{{color:var(--muted)}}
    .table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;
    min-width:940px}}th,td{{padding:10px 12px;text-align:right;
    border-bottom:1px solid #1d2c3e;white-space:nowrap}}th{{color:var(--muted);
    font-size:11px;text-transform:uppercase;letter-spacing:.055em}}th:first-child,
    td:first-child{{text-align:left}}tbody tr:last-child td{{border-bottom:0}}
    .kind{{display:inline-block;padding:3px 7px;border-radius:999px;
    background:#1e293b;color:#dbeafe;font-size:11px;font-weight:750}}
    .long{{color:var(--green)}}.short{{color:var(--red)}}.gain{{color:var(--green)}}
    .loss{{color:var(--red)}}.muted{{color:var(--muted)}}.quality{{color:var(--amber);
    font-size:12px}}.contract{{display:block;margin-top:3px;color:#cbd5e1;
    font:10px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    white-space:pre}}.empty{{padding:22px}}.empty p{{margin-top:7px;color:var(--muted)}}
    @media(max-width:720px){{header{{padding:16px}}.summary{{grid-template-columns:
    repeat(2,minmax(0,1fr));padding:12px 16px}}main{{padding:0 16px 18px}}
    .desktop-secondary{{display:none}}table{{min-width:620px}}th,td{{padding:9px}}}}
  </style>
</head>
<body data-positions-artifact-schema="1" data-source-generation="{snapshot.source_generation}" data-broker-environment="{snapshot.broker_environment}" data-runtime-binding="{runtime_binding}" data-renderer-build-sha256="{POSITIONS_RENDERER_BUILD_SHA256}">
  <header>
    <div class="banner" role="status">{POSITIONS_READ_ONLY_MARKER}</div>
    <h1>Confirmed portfolio positions</h1>
    <div class="provenance">
      <span>Source: E*TRADE portfolio</span>
      <span>Observed: {html.escape(observed_et)}</span>
      <span>Environment: {html.escape(snapshot.broker_environment)}</span>
      <span>Generation: {snapshot.source_generation[:12]}</span>
    </div>
    <p class="quality">Position identity and quantity are confirmed; option contracts are verified as standard 100-share contracts. Marks and valuations are best-effort display data and are not execution inputs.</p>
  </header>
  <div class="summary" aria-label="Portfolio position summary">
    <div class="metric"><strong>{len(snapshot.rows)}</strong><span>Position rows</span></div>
    <div class="metric"><strong>{symbol_count}</strong><span>Symbols</span></div>
    <div class="metric"><strong>{option_count}</strong><span>Option legs</span></div>
    <div class="metric"><strong>{equity_count}</strong><span>Equity rows</span></div>
  </div>
  <main>{sections}</main>
</body>
</html>
"""
    payload = html_document.encode("utf-8")
    if len(payload) > MAX_PUBLISHED_POSITIONS_BYTES:
        raise PositionsArtifactError("rendered_artifact_oversized")
    signature = hmac.new(
        signing_key.value,
        _SIGNATURE_DOMAIN + payload,
        hashlib.sha256,
    ).hexdigest()
    placeholder_line = _signature_meta_line(
        _SIGNATURE_PLACEHOLDER
    ).encode("ascii")
    signed_line = _signature_meta_line(signature).encode("ascii")
    if payload.count(placeholder_line) != 1:
        raise PositionsArtifactError("rendered_artifact_signature_invalid")
    payload = payload.replace(placeholder_line, signed_line, 1)
    inspect_positions_html(
        payload,
        signing_key=signing_key,
        expected_broker_environment=snapshot.broker_environment,
        expected_runtime_binding=runtime_binding,
    )
    return payload


def inspect_positions_html(
    payload: bytes,
    *,
    signing_key: PositionsArtifactSigningKey,
    expected_broker_environment: str,
    expected_runtime_binding: str,
) -> ArtifactMetadata:
    """Validate the closed static-HTML grammar and return source metadata."""

    _require_signing_key(signing_key)
    if expected_broker_environment not in _CLOSED_BROKER_ENVIRONMENTS:
        raise ValueError("expected broker environment is invalid")
    _require_sha256(
        expected_runtime_binding,
        "artifact_runtime_binding_invalid",
    )
    if type(payload) is not bytes or not payload:
        raise PositionsArtifactError("artifact_payload_invalid")
    if len(payload) > MAX_POSITIONS_ARTIFACT_BYTES:
        raise PositionsArtifactError("artifact_payload_oversized")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PositionsArtifactError("artifact_encoding_invalid") from exc
    if not text.startswith(POSITIONS_ARTIFACT_PREFIX):
        raise PositionsArtifactError("artifact_prologue_invalid")
    lowered = text.casefold()
    if (
        POSITIONS_READ_ONLY_MARKER not in text
        or any(marker in lowered for marker in _FORBIDDEN_ACTIVE_MARKERS)
    ):
        raise PositionsArtifactError("artifact_static_contract_invalid")
    parser = _StaticPositionsHTMLParser()
    try:
        parser.feed(text)
        parser.close()
    except (UnicodeError, ValueError) as exc:
        raise PositionsArtifactError("artifact_static_contract_invalid") from exc
    if (
        parser.open_tags
        or parser.declarations != ["doctype html"]
        or parser.comments != ["etrade-read-only-positions:v1"]
        or parser.charset != "utf-8"
        or parser.meta.get("viewport")
        != "width=device-width,initial-scale=1"
    ):
        raise PositionsArtifactError("artifact_static_contract_invalid")
    schema = parser.meta.get("etrade-positions-schema")
    source_raw = parser.meta.get("etrade-positions-source-as-of")
    generation = parser.meta.get("etrade-positions-source-generation")
    broker_environment = parser.meta.get(
        "etrade-positions-broker-environment"
    )
    runtime_binding = parser.meta.get(
        "etrade-positions-runtime-binding"
    )
    renderer_build_sha256 = parser.meta.get(
        "etrade-positions-renderer-build-sha256"
    )
    signature = parser.meta.get("etrade-positions-signature")
    if (
        schema != str(POSITIONS_ARTIFACT_SCHEMA_VERSION)
        or source_raw is None
        or generation is None
        or _SHA256.fullmatch(generation) is None
        or broker_environment != expected_broker_environment
        or runtime_binding != expected_runtime_binding
        or renderer_build_sha256 != POSITIONS_RENDERER_BUILD_SHA256
        or signature is None
        or _SHA256.fullmatch(signature) is None
        or parser.body_schema != schema
        or parser.body_generation != generation
        or parser.body_broker_environment != broker_environment
        or parser.body_runtime_binding != runtime_binding
        or parser.body_renderer_build_sha256
        != renderer_build_sha256
    ):
        raise PositionsArtifactError("artifact_metadata_invalid")
    signature_line = _signature_meta_line(signature).encode("ascii")
    placeholder_line = _signature_meta_line(
        _SIGNATURE_PLACEHOLDER
    ).encode("ascii")
    if payload.count(signature_line) != 1:
        raise PositionsArtifactError("artifact_signature_invalid")
    unsigned = payload.replace(signature_line, placeholder_line, 1)
    expected_signature = hmac.new(
        signing_key.value,
        _SIGNATURE_DOMAIN + unsigned,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise PositionsArtifactError("artifact_signature_invalid")
    try:
        source_as_of = datetime.fromisoformat(source_raw)
    except ValueError as exc:
        raise PositionsArtifactError("artifact_metadata_invalid") from exc
    if (
        source_as_of.tzinfo is None
        or source_as_of.utcoffset() is None
        or source_as_of.utcoffset().total_seconds() != 0
    ):
        raise PositionsArtifactError("artifact_metadata_invalid")
    return ArtifactMetadata(
        schema_version=POSITIONS_ARTIFACT_SCHEMA_VERSION,
        source_as_of=source_as_of.astimezone(timezone.utc),
        source_generation=generation,
        broker_environment=broker_environment,
        runtime_binding=runtime_binding,
        renderer_build_sha256=renderer_build_sha256,
        signature=signature,
    )


class PositionsArtifactReader:
    """Read-only descriptor-verified capability for one artifact path."""

    __slots__ = (
        "path",
        "max_age_seconds",
        "expected_broker_environment",
        "expected_runtime_binding",
        "_signing_key",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        max_age_seconds: int,
        signing_key: PositionsArtifactSigningKey,
        expected_broker_environment: str | None,
        expected_runtime_binding: str,
    ) -> None:
        candidate = Path(path)
        if (
            not candidate.is_absolute()
            or not candidate.name
            or candidate.name in {".", ".."}
        ):
            raise ValueError("path must identify an absolute artifact file")
        if (
            type(max_age_seconds) is not int
            or not 1 <= max_age_seconds <= 86_400
        ):
            raise ValueError("max_age_seconds is invalid")
        _require_signing_key(signing_key)
        if (
            expected_broker_environment is not None
            and expected_broker_environment
            not in _CLOSED_BROKER_ENVIRONMENTS
        ):
            raise ValueError("expected broker environment is invalid")
        _require_sha256(
            expected_runtime_binding,
            "artifact_runtime_binding_invalid",
        )
        self.path = candidate
        self.max_age_seconds = max_age_seconds
        self.expected_broker_environment = expected_broker_environment
        self.expected_runtime_binding = expected_runtime_binding
        self._signing_key = signing_key

    def __repr__(self) -> str:
        return (
            "PositionsArtifactReader("
            f"path={self.path!r}, "
            f"max_age_seconds={self.max_age_seconds}, "
            f"enabled={self.enabled})"
        )

    @property
    def enabled(self) -> bool:
        return self.expected_broker_environment is not None

    def read(self, *, now: datetime | None = None) -> ArtifactSnapshot:
        if not self.enabled:
            return _unavailable_snapshot("broker_positions_disabled")
        return _read_positions_artifact(
            self.path,
            max_age_seconds=self.max_age_seconds,
            signing_key=self._signing_key,
            expected_broker_environment=self.expected_broker_environment,
            expected_runtime_binding=self.expected_runtime_binding,
            now=now,
        )


class _StaticPositionsHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.body_broker_environment: str | None = None
        self.body_runtime_binding: str | None = None
        self.body_renderer_build_sha256: str | None = None
        self.body_schema: str | None = None
        self.body_generation: str | None = None
        self.open_tags: list[str] = []
        self.declarations: list[str] = []
        self.comments: list[str] = []
        self.charset: str | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self._start(tag, attrs, closed=False)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self._start(tag, attrs, closed=True)

    def _start(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        *,
        closed: bool,
    ) -> None:
        if tag not in _ALLOWED_TAGS:
            raise ValueError("unreviewed HTML tag")
        names = [name for name, _value in attrs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate HTML attribute")
        allowed = _GLOBAL_ATTRIBUTES | _TAG_ATTRIBUTES.get(tag, frozenset())
        if any(name not in allowed or name.startswith("on") for name in names):
            raise ValueError("unreviewed HTML attribute")
        for name, value in attrs:
            if value is None or _unsafe_text(value):
                raise ValueError("invalid HTML attribute")
            if name == "class" and re.fullmatch(r"[a-z0-9 -]+", value) is None:
                raise ValueError("invalid CSS class")
        mapping = dict(attrs)
        if tag == "meta" and "charset" in mapping:
            if set(mapping) != {"charset"} or self.charset is not None:
                raise ValueError("invalid charset metadata")
            self.charset = mapping["charset"]
        if tag == "meta" and "name" in mapping:
            name = mapping["name"]
            content = mapping.get("content")
            if (
                name is None
                or content is None
                or name in self.meta
                or name
                not in {
                    "viewport",
                    "etrade-positions-schema",
                    "etrade-positions-source-as-of",
                    "etrade-positions-source-generation",
                    "etrade-positions-broker-environment",
                    "etrade-positions-runtime-binding",
                    "etrade-positions-renderer-build-sha256",
                    "etrade-positions-signature",
                }
            ):
                raise ValueError("invalid metadata")
            self.meta[name] = content
        if tag == "body":
            self.body_broker_environment = mapping.get(
                "data-broker-environment"
            )
            self.body_runtime_binding = mapping.get(
                "data-runtime-binding"
            )
            self.body_renderer_build_sha256 = mapping.get(
                "data-renderer-build-sha256"
            )
            self.body_schema = mapping.get(
                "data-positions-artifact-schema"
            )
            self.body_generation = mapping.get("data-source-generation")
        if tag not in {"meta"} and not closed:
            self.open_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.open_tags or self.open_tags[-1] != tag:
            raise ValueError("unbalanced HTML")
        self.open_tags.pop()

    def handle_entityref(self, name: str) -> None:
        raise ValueError("unresolved entity")

    def handle_charref(self, name: str) -> None:
        raise ValueError("unresolved character reference")

    def handle_decl(self, decl: str) -> None:
        self.declarations.append(decl.casefold())

    def handle_comment(self, data: str) -> None:
        self.comments.append(data.strip())

    def handle_data(self, data: str) -> None:
        if any(
            character == "\x00" or character in _BIDI_CONTROLS
            for character in data
        ):
            raise ValueError("unsafe HTML text")


def _position_row(value: object) -> PositionRow:
    symbol = _required_symbol(_attribute(value, "symbol"))
    raw_security_type = _attribute(value, "security_type")
    security_type = {
        "Stock": "EQUITY",
        "EQ": "EQUITY",
        "Option": "OPTION",
        "OPTN": "OPTION",
    }.get(raw_security_type)
    if security_type is None:
        raise PositionsArtifactError("position_security_type_invalid")
    quantity_microunits = _scaled_integer(
        _attribute(value, "quantity"),
        _MICRO_UNITS,
        "position_quantity_invalid",
        -_MAX_QUANTITY_MICROUNITS,
        _MAX_QUANTITY_MICROUNITS,
        nonzero=True,
    )
    mark = _optional_positive_scaled_integer(
        _attribute(value, "last_price", required=False),
        _MICRO_UNITS,
        "position_mark_invalid",
        _MAX_PRICE_MICROS,
    )
    price_paid = _optional_positive_scaled_integer(
        _attribute(value, "price_paid", required=False),
        _MICRO_UNITS,
        "position_price_paid_invalid",
        _MAX_PRICE_MICROS,
    )
    market_value = _optional_scaled_integer(
        _attribute(value, "market_value", required=False),
        _CENT_UNITS,
        "position_market_value_invalid",
        -_MAX_CURRENCY_CENTS,
        _MAX_CURRENCY_CENTS,
    )
    total_gain = _optional_scaled_integer(
        _attribute(value, "total_gain", required=False),
        _CENT_UNITS,
        "position_total_gain_invalid",
        -_MAX_CURRENCY_CENTS,
        _MAX_CURRENCY_CENTS,
    )
    if security_type == "OPTION":
        call_put_raw = _attribute(value, "call_put")
        if type(call_put_raw) is not str:
            raise PositionsArtifactError("option_contract_invalid")
        call_put = call_put_raw.upper()
        if call_put not in _CLOSED_CALL_PUT:
            raise PositionsArtifactError("option_contract_invalid")
        expiration = _expiration_date(
            _attribute(value, "expiration_date")
        )
        strike = _positive_scaled_integer(
            _attribute(value, "strike_price"),
            _MICRO_UNITS,
            "position_strike_invalid",
            _MAX_PRICE_MICROS,
        )
        underlying = _optional_positive_scaled_integer(
            _attribute(value, "underlying_last_price", required=False),
            _MICRO_UNITS,
            "position_underlying_invalid",
            _MAX_PRICE_MICROS,
        )
        option_osi_key = _option_osi_key(
            _attribute(value, "osi_key"),
            symbol=symbol,
            call_put=call_put,
            expiration=expiration,
            strike_price_micros=strike,
        )
        adjusted = _attribute(value, "options_adjusted_flag")
        if type(adjusted) is not bool:
            raise PositionsArtifactError(
                "option_adjustment_status_invalid"
            )
        if adjusted:
            raise PositionsArtifactError(
                "adjusted_option_unsupported"
            )
        option_multiplier = _option_multiplier(
            _attribute(value, "option_multiplier")
        )
        option_deliverables = _option_deliverables(
            _attribute(value, "option_deliverables"),
            symbol=symbol,
        )
    else:
        call_put = None
        expiration = None
        strike = None
        underlying = None
        option_osi_key = None
        option_multiplier = None
        adjusted = None
        option_deliverables = None
    return PositionRow(
        symbol=symbol,
        security_type=security_type,
        quantity_microunits=quantity_microunits,
        mark_price_micros=mark,
        price_paid_micros=price_paid,
        market_value_cents=market_value,
        total_gain_cents=total_gain,
        call_put=call_put,
        expiration_date=expiration,
        strike_price_micros=strike,
        underlying_price_micros=underlying,
        option_osi_key=option_osi_key,
        option_multiplier=option_multiplier,
        option_adjusted=adjusted,
        option_deliverables=option_deliverables,
    )


def _render_symbol_group(
    symbol: str,
    rows: list[PositionRow],
    source_as_of: datetime,
) -> str:
    row_html = "".join(_render_row(row, source_as_of) for row in rows)
    count_label = f"{len(rows)} row" + ("" if len(rows) == 1 else "s")
    return (
        '<section class="group">'
        '<div class="group-title">'
        f"<h2>{html.escape(symbol)}</h2>"
        f'<span class="count">{count_label}</span>'
        "</div>"
        '<div class="table-wrap"><table>'
        "<thead><tr>"
        '<th scope="col">Position</th>'
        '<th scope="col">Expiry</th>'
        '<th scope="col">Quantity</th>'
        '<th scope="col">Strike</th>'
        '<th scope="col">Mark</th>'
        '<th scope="col" class="desktop-secondary">Cost basis</th>'
        '<th scope="col">Market value</th>'
        '<th scope="col">Total gain</th>'
        '<th scope="col" class="desktop-secondary">Underlying</th>'
        '<th scope="col">Valuation</th>'
        "</tr></thead>"
        f"<tbody>{row_html}</tbody></table></div></section>"
    )


def _render_row(row: PositionRow, source_as_of: datetime) -> str:
    side_class = "long" if row.side == "LONG" else "short"
    position_label = (
        f"{row.call_put} {row.side.title()}"
        if row.security_type == "OPTION"
        else f"Equity {row.side.title()}"
    )
    if row.expiration_date is None:
        expiration = "—"
    else:
        dte = (
            row.expiration_date
            - source_as_of.astimezone(_NEW_YORK).date()
        ).days
        dte_label = f"{dte} DTE" if dte >= 0 else "Expired"
        expiration = f"{row.expiration_date.isoformat()} · {dte_label}"
    gain_attribute = ""
    if row.total_gain_cents is not None:
        gain_class = "gain" if row.total_gain_cents >= 0 else "loss"
        gain_attribute = f' class="{gain_class}"'
    quality = (
        "Best-effort mark"
        if row.valuation_status == "BEST_EFFORT_MARK"
        else "Position only"
    )
    contract = (
        ""
        if row.security_type == "EQUITY"
        else (
            f"Standard ×{row.option_multiplier} · "
            f"{row.option_osi_key.strip()}"
        )
    )
    contract_html = (
        ""
        if not contract
        else (
            f'<small class="contract">'
            f"{html.escape(contract)}</small>"
        )
    )
    return (
        "<tr>"
        f'<td><span class="kind">{html.escape(row.security_type.title())}</span> '
        f'<strong class="{side_class}">{html.escape(position_label)}</strong>'
        f"{contract_html}</td>"
        f"<td>{html.escape(expiration)}</td>"
        f"<td>{_format_quantity(row.quantity_microunits)}</td>"
        f"<td>{_format_price(row.strike_price_micros)}</td>"
        f"<td>{_format_price(row.mark_price_micros)}</td>"
        '<td class="desktop-secondary">'
        f"{_format_price(row.price_paid_micros)}</td>"
        f"<td>{_format_cents(row.market_value_cents)}</td>"
        f"<td{gain_attribute}>{_format_cents(row.total_gain_cents)}</td>"
        '<td class="desktop-secondary">'
        f"{_format_price(row.underlying_price_micros)}</td>"
        f'<td class="muted">{html.escape(quality)}</td>'
        "</tr>"
    )


def _format_quantity(value: int) -> str:
    decimal = Decimal(value) / _MICRO_UNITS
    if decimal == decimal.to_integral():
        return f"{int(decimal):,}"
    return f"{decimal.normalize():,f}"


def _format_price(value: int | None) -> str:
    if value is None:
        return "—"
    decimal = Decimal(value) / _MICRO_UNITS
    return f"${decimal:,.2f}"


def _format_cents(value: int | None) -> str:
    if value is None:
        return "—"
    decimal = Decimal(value) / _CENT_UNITS
    sign = "-" if decimal < 0 else ""
    return f"{sign}${abs(decimal):,.2f}"


def _snapshot_generation(
    broker_environment: str,
    source_as_of: datetime,
    rows: tuple[PositionRow, ...],
) -> str:
    if broker_environment not in _CLOSED_BROKER_ENVIRONMENTS:
        raise PositionsArtifactError("snapshot_environment_invalid")
    payload = {
        "broker_environment": broker_environment,
        "rows": [row.canonical_value() for row in rows],
        "schema_version": POSITIONS_ARTIFACT_SCHEMA_VERSION,
        "source": "etrade_portfolio",
        "source_as_of": source_as_of.isoformat(),
    }
    return hashlib.sha256(
        b"etrade-positions-snapshot.v1\0"
        + json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _row_sort_key(row: PositionRow) -> tuple[Any, ...]:
    return (
        row.symbol,
        row.security_type,
        row.expiration_date or date.max,
        row.call_put or "",
        row.strike_price_micros or 0,
        row.option_osi_key or "",
        row.quantity_microunits,
        row.market_value_cents if row.market_value_cents is not None else 0,
    )


def _attribute(
    value: object,
    name: str,
    *,
    required: bool = True,
) -> object:
    try:
        result = getattr(value, name)
    except Exception as exc:
        if required:
            raise PositionsArtifactError(
                "position_required_field_missing"
            ) from exc
        return None
    return result


def _required_symbol(value: object) -> str:
    if type(value) is not str or _unsafe_text(value):
        raise PositionsArtifactError("position_symbol_invalid")
    if _SYMBOL.fullmatch(value) is None:
        raise PositionsArtifactError("position_symbol_invalid")
    return value


def _valid_option_text(
    value: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> bool:
    return (
        type(value) is str
        and (allow_empty or bool(value.strip()))
        and len(value) <= maximum
        and value.isascii()
        and not _unsafe_text(value)
    )


def _option_osi_key(
    value: object,
    *,
    symbol: str,
    call_put: str,
    expiration: date,
    strike_price_micros: int,
) -> str:
    if type(value) is not str or not _valid_option_text(
        value,
        maximum=64,
    ):
        raise PositionsArtifactError("option_osi_key_invalid")
    if (
        len(value) != 21
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.- "
            for character in value[:6]
        )
        or not value[6:12].isdigit()
        or value[12] not in {"C", "P"}
        or not value[13:].isdigit()
    ):
        raise PositionsArtifactError("option_osi_key_invalid")
    root = value[:6].rstrip("- ")
    if not root:
        raise PositionsArtifactError("option_osi_key_invalid")
    try:
        osi_expiration = date(
            2000 + int(value[6:8]),
            int(value[8:10]),
            int(value[10:12]),
        )
    except ValueError as exc:
        raise PositionsArtifactError("option_osi_key_invalid") from exc
    normalized_symbol = symbol.replace(".", "").replace("-", "")
    normalized_root = root.replace(".", "").replace("-", "")
    if (
        normalized_root not in {
            normalized_symbol,
            f"{normalized_symbol}W",
        }
        or value[12] != call_put[0]
        or osi_expiration != expiration
        or int(value[13:]) * 1_000 != strike_price_micros
    ):
        raise PositionsArtifactError("option_osi_key_mismatch")
    return value


def _option_multiplier(value: object) -> int:
    multiplier = _decimal(value, "option_multiplier_invalid")
    if multiplier != Decimal("100"):
        raise PositionsArtifactError("adjusted_option_unsupported")
    return 100


def _option_deliverables(
    value: object,
    *,
    symbol: str,
) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not _valid_option_text(
        value,
        maximum=512,
        allow_empty=True,
    ):
        raise PositionsArtifactError("option_deliverables_invalid")
    if not value.strip():
        return None
    if value not in {
        "100 shares",
        f"100 shares of {symbol}",
    }:
        raise PositionsArtifactError("adjusted_option_unsupported")
    return value


def _unsafe_text(value: str) -> bool:
    return any(
        ord(character) < 32
        or 0x7F <= ord(character) <= 0x9F
        or character in _BIDI_CONTROLS
        for character in value
    )


def _expiration_date(value: object) -> date:
    if type(value) is datetime:
        result = value.date()
    elif type(value) is date:
        result = value
    elif type(value) is str:
        try:
            result = date.fromisoformat(value)
        except ValueError as exc:
            raise PositionsArtifactError("option_expiration_invalid") from exc
    else:
        raise PositionsArtifactError("option_expiration_invalid")
    if not 1970 <= result.year <= 2200:
        raise PositionsArtifactError("option_expiration_invalid")
    return result


def _utc_datetime(value: datetime | None) -> datetime:
    result = datetime.now(timezone.utc) if value is None else value
    if (
        type(result) is not datetime
        or result.tzinfo is None
        or result.utcoffset() is None
    ):
        raise PositionsArtifactError("snapshot_source_time_invalid")
    return result.astimezone(timezone.utc)


def _decimal(value: object, code: str) -> Decimal:
    if type(value) is bool or value is None:
        raise PositionsArtifactError(code)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PositionsArtifactError(code) from exc
    if not result.is_finite():
        raise PositionsArtifactError(code)
    return result


def _scaled_integer(
    value: object,
    scale: Decimal,
    code: str,
    minimum: int,
    maximum: int,
    *,
    nonzero: bool = False,
) -> int:
    scaled = _decimal(value, code) * scale
    integral = scaled.to_integral_value(rounding=ROUND_HALF_EVEN)
    if scaled != integral:
        raise PositionsArtifactError(code)
    result = int(integral)
    _bounded_exact_int(
        result,
        code,
        minimum,
        maximum,
        nonzero=nonzero,
    )
    return result


def _optional_scaled_integer(
    value: object,
    scale: Decimal,
    code: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    return _scaled_integer(value, scale, code, minimum, maximum)


def _positive_scaled_integer(
    value: object,
    scale: Decimal,
    code: str,
    maximum: int,
) -> int:
    return _scaled_integer(value, scale, code, 1, maximum)


def _optional_positive_scaled_integer(
    value: object,
    scale: Decimal,
    code: str,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    number = _decimal(value, code)
    if number == 0:
        return None
    return _positive_scaled_integer(number, scale, code, maximum)


def _bounded_exact_int(
    value: object,
    code: str,
    minimum: int,
    maximum: int,
    *,
    nonzero: bool = False,
) -> None:
    if (
        type(value) is not int
        or not minimum <= value <= maximum
        or (nonzero and value == 0)
    ):
        raise PositionsArtifactError(code)


def _optional_bounded_exact_int(
    value: object,
    code: str,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return
    _bounded_exact_int(value, code, minimum, maximum)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _safe_open_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _private_artifact(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
    )


def _private_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o700
    )


def _unavailable_snapshot(
    reason: str,
    *,
    stale: bool = False,
    sha256: str | None = None,
    modified_at: str | None = None,
    size_bytes: int | None = None,
    source_as_of: str | None = None,
    source_generation: str | None = None,
    expires_at: str | None = None,
) -> ArtifactSnapshot:
    return ArtifactSnapshot(
        available=False,
        stale=stale,
        content=None,
        sha256=sha256,
        modified_at=modified_at,
        size_bytes=size_bytes,
        reason=reason,
        source_as_of=source_as_of,
        source_generation=source_generation,
        expires_at=expires_at,
    )


def _read_positions_artifact(
    path: Path,
    *,
    max_age_seconds: int,
    signing_key: PositionsArtifactSigningKey,
    expected_broker_environment: str,
    expected_runtime_binding: str,
    now: datetime | None = None,
) -> ArtifactSnapshot:
    current_time = _utc_datetime(now)
    try:
        parent_before = os.lstat(path.parent)
        if (
            stat.S_ISLNK(parent_before.st_mode)
            or not _private_directory(parent_before)
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
    final: os.stat_result | None = None
    content = b""
    try:
        parent_after = os.fstat(parent_descriptor)
        if (
            not _same_file(parent_before, parent_after)
            or not _private_directory(parent_after)
        ):
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
        if not _private_artifact(before):
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
                try:
                    chunk = os.read(descriptor, min(64 * 1024, remaining))
                except InterruptedError:
                    continue
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
                not _private_artifact(final)
                or not _private_artifact(path_after)
                or not _same_snapshot(before, after)
                or not _same_snapshot(after, final)
                or not _same_snapshot(final, path_after)
                or len(content) != final.st_size
            ):
                return _unavailable_snapshot("changed")
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            parent_path_after = os.lstat(path.parent)
        except OSError:
            return _unavailable_snapshot("unsafe_parent")
        if (
            stat.S_ISLNK(parent_path_after.st_mode)
            or not _same_file(parent_after, parent_path_after)
            or not _private_directory(parent_path_after)
            or not _private_directory(parent_after)
        ):
            return _unavailable_snapshot("unsafe_parent")
    except OSError:
        return _unavailable_snapshot("artifact_io_error")
    finally:
        try:
            os.close(parent_descriptor)
        except OSError:
            pass
    if final is None or len(content) > MAX_POSITIONS_ARTIFACT_BYTES:
        return _unavailable_snapshot("oversized")
    digest = hashlib.sha256(content).hexdigest()
    modified_at = datetime.fromtimestamp(
        final.st_mtime,
        tz=timezone.utc,
    )
    try:
        metadata = inspect_positions_html(
            content,
            signing_key=signing_key,
            expected_broker_environment=expected_broker_environment,
            expected_runtime_binding=expected_runtime_binding,
        )
    except PositionsArtifactError:
        return _unavailable_snapshot(
            "untrusted_artifact",
            sha256=digest,
            modified_at=modified_at.isoformat(),
            size_bytes=len(content),
        )
    source_iso = metadata.source_as_of.isoformat()
    expires_at = min(
        modified_at,
        metadata.source_as_of,
    ) + timedelta(seconds=max_age_seconds)
    common = {
        "sha256": digest,
        "modified_at": modified_at.isoformat(),
        "size_bytes": len(content),
        "source_as_of": source_iso,
        "source_generation": metadata.source_generation,
        "expires_at": expires_at.isoformat(),
    }
    publication_age = (current_time - modified_at).total_seconds()
    source_age = (current_time - metadata.source_as_of).total_seconds()
    if (
        publication_age < -MAX_ARTIFACT_FUTURE_SKEW_SECONDS
        or source_age < -MAX_ARTIFACT_FUTURE_SKEW_SECONDS
    ):
        return _unavailable_snapshot("future_timestamp", **common)
    if publication_age > max_age_seconds or source_age > max_age_seconds:
        return _unavailable_snapshot("stale", stale=True, **common)
    return ArtifactSnapshot(
        available=True,
        stale=False,
        content=content,
        sha256=digest,
        modified_at=modified_at.isoformat(),
        size_bytes=len(content),
        reason=None,
        source_as_of=source_iso,
        source_generation=metadata.source_generation,
        expires_at=expires_at.isoformat(),
    )


__all__ = [
    "ArtifactMetadata",
    "ArtifactSnapshot",
    "MAX_ARTIFACT_FUTURE_SKEW_SECONDS",
    "MAX_POSITION_ROWS",
    "MAX_POSITIONS_ARTIFACT_BYTES",
    "MAX_PUBLISHED_POSITIONS_BYTES",
    "POSITIONS_ARTIFACT_PREFIX",
    "POSITIONS_ARTIFACT_SCHEMA_VERSION",
    "POSITIONS_READ_ONLY_MARKER",
    "POSITIONS_RENDERER_BUILD_SHA256",
    "PositionRow",
    "PositionsArtifactError",
    "PositionsArtifactReader",
    "PositionsArtifactSigningKey",
    "PositionsSnapshot",
    "build_positions_snapshot",
    "inspect_positions_html",
    "positions_identity_fingerprint",
    "render_positions_html",
]
