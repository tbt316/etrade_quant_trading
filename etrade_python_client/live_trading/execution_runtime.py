"""Single reviewed composition root for supervised E*TRADE manual opening.

Only this module may construct the hardened mutation transport.  It returns the
narrow :class:`ManualOpenService`; callers never receive the transport, reader,
ledger, or gateway individually.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from rauth import OAuth1Session

from live_trading.etrade_broker_reader import ETradeBrokerReader
from live_trading.etrade_broker_transport import (
    ETradeBrokerTransport,
    SelectedBrokerAccount,
)
from live_trading.etrade_order_gateway import (
    EtradeOrderGateway,
    GatewayReconciliationRequired,
)
from live_trading.manual_open import ManualOpenService, ManualOpenUnavailable
from live_trading.order_intent_ledger import OrderIntentLedger
from live_trading.runtime_config import (
    RuntimeConfig,
    validate_runtime_directories,
)
from live_trading.runtime_safety import RuntimeSafetyBoundary


def build_manual_open_service(
    *,
    session: OAuth1Session,
    runtime_config: RuntimeConfig,
    runtime_safety: RuntimeSafetyBoundary,
    proposal_secret: str,
    clock=None,
) -> ManualOpenService:
    """Construct, reconcile, and return the sole manual-opening capability."""

    if type(session) is not OAuth1Session:
        raise ManualOpenUnavailable(
            "manual opening requires an exact authenticated OAuth session"
        )
    if type(runtime_config) is not RuntimeConfig:
        raise ManualOpenUnavailable(
            "manual opening requires an exact runtime configuration"
        )
    if type(runtime_safety) is not RuntimeSafetyBoundary:
        raise ManualOpenUnavailable(
            "manual opening requires an exact runtime safety boundary"
        )
    if (
        runtime_config.schema_version < 2
        or not runtime_config.execution.broker_mutations_enabled
    ):
        raise ManualOpenUnavailable(
            "supervised manual opening is disabled by runtime configuration"
        )
    selected = runtime_config.selected_account
    if selected is None:
        raise ManualOpenUnavailable(
            "manual opening requires one exact selected account"
        )
    validate_runtime_directories(runtime_config.paths)
    effective_clock = clock or (lambda: datetime.now(timezone.utc))
    account = SelectedBrokerAccount(
        selected.account_id,
        selected.account_id_key,
        selected.institution_type,
    )
    ledger = OrderIntentLedger(
        runtime_config.paths.order_intent_ledger_file,
        clock=effective_clock,
    )
    transport = ETradeBrokerTransport(
        session=session,
        ledger=ledger,
        runtime_safety=runtime_safety,
        selected_account=account,
        clock=effective_clock,
    )
    reader = ETradeBrokerReader(
        session=session,
        ledger=ledger,
        runtime_safety=runtime_safety,
        selected_account=account,
        clock=effective_clock,
    )
    gateway = EtradeOrderGateway(
        runtime_safety=runtime_safety,
        ledger=ledger,
        transport=transport,
        reader=reader,
        opening_risk_budget=(
            Decimal(runtime_config.risk.max_account_open_risk_cents)
            / Decimal("100")
        ),
        daily_opening_risk_budget=(
            Decimal(runtime_config.risk.max_daily_loss_cents)
            / Decimal("100")
        ),
        clock=effective_clock,
    )
    try:
        gateway.start()
    except GatewayReconciliationRequired:
        # Durable history remains available while unresolved broker work keeps
        # the narrow mutation capability fail-closed.
        pass
    return ManualOpenService(
        gateway=gateway,
        quote_reader=reader,
        runtime_config=runtime_config,
        runtime_safety=runtime_safety,
        proposal_secret=proposal_secret,
        clock=effective_clock,
    )


__all__ = ["build_manual_open_service"]
