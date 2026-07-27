from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from live_trading.order_intent_ledger import (
    SCHEMA_VERSION,
    OrderIntent,
    OrderIntentLedger,
    OrderIntentLedgerError,
)


APPEND_ONLY_TABLES = (
    "order_events",
    "broker_order_history",
    "amendment_history",
    "outbound_authorizations",
    "transport_send_attempts",
    "broker_preview_receipts",
    "transport_response_receipts",
    "broker_read_receipts",
    "broker_read_manifests",
    "broker_read_manifest_members",
    "capacity_decisions",
    "opening_risk_prerequisites",
    "opening_quote_receipts",
    "opening_risk_lineages",
    "reservation_absorptions",
    "closing_reservations",
    "closing_reservation_voids",
    "closing_reservation_absorptions",
    "cancel_authorizations",
    "cancel_send_attempts",
    "cancel_response_receipts",
    "cancel_resolutions",
)

DURABLE_STATE_TABLES = (
    "order_intents",
    "margin_reservations",
    "order_cancellations",
)

REPLACE_TRIGGER_BY_TABLE = {
    "order_events": "prevent_order_event_replace",
    "broker_order_history": "prevent_broker_order_history_replace",
    "amendment_history": "prevent_amendment_history_replace",
    "outbound_authorizations": "prevent_outbound_authorization_replace",
    "transport_send_attempts": "prevent_transport_send_attempt_replace",
    "broker_preview_receipts": "prevent_broker_preview_receipt_replace",
    "transport_response_receipts": (
        "prevent_transport_response_receipt_replace"
    ),
    "broker_read_receipts": "prevent_broker_read_receipt_replace",
    "broker_read_manifests": "prevent_broker_read_manifest_replace",
    "broker_read_manifest_members": "prevent_broker_read_member_replace",
    "capacity_decisions": "prevent_capacity_decision_replace",
    "opening_risk_prerequisites": (
        "prevent_opening_risk_prerequisite_replace"
    ),
    "opening_quote_receipts": "prevent_opening_quote_receipt_replace",
    "opening_risk_lineages": "prevent_opening_risk_lineage_replace",
    "reservation_absorptions": "prevent_reservation_absorption_replace",
    "closing_reservations": "prevent_closing_reservation_replace",
    "closing_reservation_voids": "prevent_closing_void_replace",
    "closing_reservation_absorptions": (
        "prevent_closing_absorption_replace"
    ),
    "cancel_authorizations": "prevent_cancel_authorization_replace",
    "cancel_send_attempts": "prevent_cancel_send_attempt_replace",
    "cancel_response_receipts": "prevent_cancel_response_receipt_replace",
    "cancel_resolutions": "prevent_cancel_resolution_replace",
    "order_intents": "prevent_order_intent_replace",
    "margin_reservations": "prevent_margin_reservation_replace",
    "order_cancellations": "prevent_order_cancellation_replace",
}

PREEXISTING_SCHEMA_16_REPLACE_TRIGGERS = {
    "prevent_opening_quote_receipt_replace",
    "prevent_opening_risk_lineage_replace",
}

ALTERNATE_UNIQUE_IDENTITIES = {
    "order_intents": {
        frozenset({"client_order_id"}),
        frozenset({"broker_order_id"}),
        frozenset(
            {
                "account_id",
                "environment",
                "idempotency_scope",
                "idempotency_key",
            }
        ),
    },
    "amendment_history": {frozenset({"client_order_id"})},
    "broker_preview_receipts": {
        frozenset({"account_id", "environment", "preview_id"})
    },
    "broker_read_manifest_members": {
        frozenset({"evidence_sha256", "member_role"}),
        frozenset({"evidence_sha256", "receipt_sha256"}),
    },
    "opening_risk_prerequisites": {
        frozenset({"intent_id"}),
        frozenset({"decision_sha256"}),
    },
    "opening_risk_lineages": {
        frozenset({"intent_id"}),
        frozenset({"prerequisite_sha256"}),
    },
    "reservation_absorptions": {frozenset({"intent_id"})},
    "closing_reservations": {frozenset({"intent_id"})},
    "closing_reservation_voids": {
        frozenset({"intent_id"}),
        frozenset({"reservation_sha256"}),
    },
    "closing_reservation_absorptions": {frozenset({"intent_id"})},
    "cancel_authorizations": {
        frozenset({"intent_id", "fencing_token"})
    },
    "cancel_send_attempts": {frozenset({"authorization_sha256"})},
    "cancel_resolutions": {frozenset({"intent_id"})},
}


def _ledger_path(tmp_path: Path, name: str) -> Path:
    return tmp_path / name / "orders.sqlite3"


def _normalized_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").lower().split())


def _seed_value(column: sqlite3.Row) -> Any:
    declared_type = str(column["type"]).upper()
    if "INT" in declared_type:
        return 1
    if "BLOB" in declared_type:
        return b"seed"
    return "seed"


def _different_value(value: Any, ordinal: int) -> Any:
    if isinstance(value, int):
        return value + ordinal + 1
    if isinstance(value, bytes):
        return value + f"-alt-{ordinal}".encode("ascii")
    return f"{value}-alt-{ordinal}"


def _seed_every_guarded_table(path: Path) -> None:
    collision_triggers = set(REPLACE_TRIGGER_BY_TABLE.values())
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        insert_triggers = connection.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type = 'trigger'
            """
        ).fetchall()
        for trigger in insert_triggers:
            sql = _normalized_sql(str(trigger["sql"]))
            if (
                "before insert on" in sql
                and trigger["name"] not in collision_triggers
            ):
                connection.execute(
                    f'DROP TRIGGER "{trigger["name"]}"'
                )
        for table in REPLACE_TRIGGER_BY_TABLE:
            columns = connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
            names = [str(column["name"]) for column in columns]
            values = [_seed_value(column) for column in columns]
            placeholders = ",".join("?" for _ in values)
            quoted_names = ",".join(f'"{name}"' for name in names)
            connection.execute(
                f'INSERT INTO "{table}" ({quoted_names}) '
                f"VALUES ({placeholders})",
                values,
            )
        connection.commit()


def test_exact_collision_inventory_and_mutable_projection_boundary(
    tmp_path: Path,
) -> None:
    path = _ledger_path(tmp_path, "inventory")
    OrderIntentLedger(path, run_id="collision-inventory")
    with sqlite3.connect(path) as connection:
        trigger_rows = connection.execute(
            """
            SELECT name, tbl_name, sql FROM sqlite_master
            WHERE type = 'trigger'
            """
        ).fetchall()
    triggers = {
        name: (table, _normalized_sql(sql))
        for name, table, sql in trigger_rows
    }
    assert set(REPLACE_TRIGGER_BY_TABLE.values()) <= set(triggers)
    for table, name in REPLACE_TRIGGER_BY_TABLE.items():
        trigger_table, sql = triggers[name]
        assert trigger_table == table
        assert f"before insert on {table}" in sql
        assert "raise(" in sql
        assert "abort" in sql
    for projection in (
        "reservation_caps",
        "amendment_leases",
        "ledger_metadata",
    ):
        assert not any(
            table == projection and "before insert" in sql
            for table, sql in triggers.values()
        )


def test_replace_is_rejected_by_primary_and_every_alternate_unique_identity(
    tmp_path: Path,
) -> None:
    path = _ledger_path(tmp_path, "replace")
    OrderIntentLedger(path, run_id="replace-guards")
    _seed_every_guarded_table(path)

    observed_alternate_identities: dict[
        str, set[frozenset[str]]
    ] = {}
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        for table in REPLACE_TRIGGER_BY_TABLE:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f'INSERT OR REPLACE INTO "{table}" '
                    f'SELECT * FROM "{table}" LIMIT 1'
                )
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f'REPLACE INTO "{table}" '
                    f'SELECT * FROM "{table}" LIMIT 1'
                )

            seed = connection.execute(
                f'SELECT * FROM "{table}" LIMIT 1'
            ).fetchone()
            assert seed is not None
            columns = [str(name) for name in seed.keys()]
            for index in connection.execute(
                f'PRAGMA index_list("{table}")'
            ).fetchall():
                if int(index["unique"]) != 1 or index["origin"] != "u":
                    continue
                identity_columns = {
                    str(row["name"])
                    for row in connection.execute(
                        f'PRAGMA index_info("{index["name"]}")'
                    ).fetchall()
                }
                observed_alternate_identities.setdefault(
                    table, set()
                ).add(frozenset(identity_columns))
                values = [
                    seed[column]
                    if column in identity_columns
                    else _different_value(seed[column], ordinal)
                    for ordinal, column in enumerate(columns)
                ]
                placeholders = ",".join("?" for _ in values)
                quoted_names = ",".join(
                    f'"{column}"' for column in columns
                )
                with pytest.raises(sqlite3.IntegrityError):
                    connection.execute(
                        f'INSERT OR REPLACE INTO "{table}" '
                        f"({quoted_names}) VALUES ({placeholders})",
                        values,
                    )
        assert observed_alternate_identities == (
            ALTERNATE_UNIQUE_IDENTITIES
        )


def test_append_only_update_delete_and_state_row_delete_are_rejected(
    tmp_path: Path,
) -> None:
    path = _ledger_path(tmp_path, "mutations")
    OrderIntentLedger(path, run_id="mutation-guards")
    _seed_every_guarded_table(path)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        for table in APPEND_ONLY_TABLES:
            column = connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchone()["name"]
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f'UPDATE "{table}" SET "{column}" = "{column}"'
                )
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(f'DELETE FROM "{table}"')
        for table in DURABLE_STATE_TABLES:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(f'DELETE FROM "{table}"')


def test_current_schema_refuses_missing_and_malformed_collision_guard(
    tmp_path: Path,
) -> None:
    path = _ledger_path(tmp_path, "current-tamper")
    OrderIntentLedger(path, run_id="before-current-tamper")
    trigger = "prevent_broker_read_receipt_replace"
    with sqlite3.connect(path) as connection:
        connection.execute(f'DROP TRIGGER "{trigger}"')

    with pytest.raises(
        OrderIntentLedgerError,
        match="triggers are incomplete",
    ):
        OrderIntentLedger(path, run_id="missing-current-trigger")

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'trigger' AND name = ?
            """,
            (trigger,),
        ).fetchone() is None
        connection.execute(
            f"""
            CREATE TRIGGER "{trigger}"
            BEFORE INSERT ON broker_read_receipts
            WHEN 0
            BEGIN
                SELECT RAISE(ABORT, 'never runs');
            END
            """
        )

    with pytest.raises(
        OrderIntentLedgerError,
        match="trigger definition is invalid",
    ):
        OrderIntentLedger(path, run_id="malformed-current-trigger")


def test_genuine_schema_16_migration_preserves_rows_and_installs_guards(
    tmp_path: Path,
) -> None:
    path = _ledger_path(tmp_path, "schema-16")
    before = OrderIntentLedger(path, run_id="before-schema-16")
    intent = OrderIntent.build(
        account_id="1000000001",
        environment="sandbox",
        strategy_id="migration-fixture",
        decision_id="decision-1",
        idempotency_scope="decision",
        idempotency_key="migration-key",
        intent_kind="OPENING",
        order_payload={
            "securityType": "OPTN",
            "orderAction": "SPREAD",
            "priceType": "NET_CREDIT",
            "limitPrice": 1.25,
            "orderTerm": "GOOD_FOR_DAY",
            "spreadType": "VERTICAL",
            "legs": [
                {
                    "symbol": "SPY",
                    "callPut": "PUT",
                    "expiryYear": 2026,
                    "expiryMonth": 8,
                    "expiryDay": 21,
                    "strikePrice": 620,
                    "orderAction": "SELL_OPEN",
                    "quantity": 1,
                },
                {
                    "symbol": "SPY",
                    "callPut": "PUT",
                    "expiryYear": 2026,
                    "expiryMonth": 8,
                    "expiryDay": 21,
                    "strikePrice": 615,
                    "orderAction": "BUY_OPEN",
                    "quantity": 1,
                },
            ],
        },
    )
    created = before.create_intent(intent, intent_id="migration-intent")
    with sqlite3.connect(path) as connection:
        preserved_before = connection.execute(
            """
            SELECT intent_id, client_order_id, payload_hash, state
            FROM order_intents WHERE intent_id = ?
            """,
            (created.intent.intent_id,),
        ).fetchone()
        event_count_before = connection.execute(
            """
            SELECT COUNT(*) FROM order_events WHERE intent_id = ?
            """,
            (created.intent.intent_id,),
        ).fetchone()[0]
        for trigger in (
            set(REPLACE_TRIGGER_BY_TABLE.values())
            - PREEXISTING_SCHEMA_16_REPLACE_TRIGGERS
        ):
            connection.execute(f'DROP TRIGGER "{trigger}"')
        connection.execute("DROP TRIGGER prevent_order_intent_delete")
        connection.execute(
            """
            UPDATE ledger_metadata SET schema_version = 16
            WHERE singleton = 1
            """
        )

    OrderIntentLedger(path, run_id="schema-16-migration")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            """
            SELECT schema_version FROM ledger_metadata
            WHERE singleton = 1
            """
        ).fetchone()[0] == SCHEMA_VERSION
        assert connection.execute(
            """
            SELECT intent_id, client_order_id, payload_hash, state
            FROM order_intents WHERE intent_id = ?
            """,
            (created.intent.intent_id,),
        ).fetchone() == preserved_before
        assert connection.execute(
            """
            SELECT COUNT(*) FROM order_events WHERE intent_id = ?
            """,
            (created.intent.intent_id,),
        ).fetchone()[0] == event_count_before
        trigger_names = {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'trigger'
                """
            )
        }
    assert set(REPLACE_TRIGGER_BY_TABLE.values()) <= trigger_names
    assert "prevent_order_intent_delete" in trigger_names
