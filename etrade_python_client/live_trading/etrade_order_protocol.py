"""Pure, deterministic E*TRADE order-protocol serialization.

This module has no broker session and performs no I/O.  It is the shared
source of truth used by both the durable ledger verifier and the isolated
mutation transport for request identity.
"""

from __future__ import annotations

from urllib.parse import quote
from xml.etree import ElementTree as ET


def cancel_order_route(account_id_key: str) -> str:
    """Return the exact account-bound cancellation route."""

    if (
        type(account_id_key) is not str
        or not account_id_key
        or len(account_id_key) > 256
        or not account_id_key.isascii()
    ):
        raise ValueError("account id key is not canonical")
    encoded_account = quote(account_id_key, safe="")
    return f"/v1/accounts/{encoded_account}/orders/cancel"


def cancel_order_xml(broker_order_id: str) -> bytes:
    """Serialize the sole supported cancellation request."""

    if (
        type(broker_order_id) is not str
        or not broker_order_id.isascii()
        or not broker_order_id.isdigit()
        or broker_order_id.startswith("0")
        or len(broker_order_id) > 32
    ):
        raise ValueError("broker order id is not canonical")
    root = ET.Element("CancelOrderRequest")
    child = ET.SubElement(root, "orderId")
    child.text = broker_order_id
    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
    )
