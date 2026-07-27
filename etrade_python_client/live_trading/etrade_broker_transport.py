"""Private, no-retry E*TRADE order-mutation transport.

The durable order ledger authorizes exact JSON bytes.  This module is the
single adapter that may transform those bytes into E*TRADE XML and issue an
HTTP mutation.  It deliberately contains no retry, reconciliation, strategy,
or risk-policy logic.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import multiprocessing
import re
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import quote, urlsplit
from xml.etree import ElementTree as ET

from rauth import OAuth1Session
from rauth.utils import CaseInsensitiveDict, OAuth1Auth
from requests import PreparedRequest, Request
from requests.adapters import HTTPAdapter
from requests.exceptions import Timeout
from urllib3.util.retry import Retry

from live_trading.order_intent_ledger import (
    OrderIntentLedger,
    OutboundAuthorization,
    TransportRequestEvidence,
    TransportResponseEvidence,
    wire_order_payload,
)
from live_trading.runtime_safety import RuntimeSafetyBoundary


_MAX_AUTHORIZATION_BYTES = 32 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_RESPONSE_NODES = 512
_REQUEST_TIMEOUT = (3.05, 10.0)
_RESPONSE_WALL_TIMEOUT_SECONDS = 12.0
_TOTAL_EXCHANGE_TIMEOUT_SECONDS = 15.0
_EXCHANGE_RESULT_HEADER_BYTES = 39
_EXCHANGE_RESULT_BUFFER_BYTES = (
    _EXCHANGE_RESULT_HEADER_BYTES + _MAX_RESPONSE_BYTES + 1
)
_EXCHANGE_KIND_CODES = {
    "RESPONSE": 1,
    "TIMEOUT": 2,
    "TRANSPORT_ERROR": 3,
    "MALFORMED_RESPONSE": 4,
}
_EXCHANGE_CODE_KINDS = {
    code: kind for kind, code in _EXCHANGE_KIND_CODES.items()
}
_MAX_BROKER_INT64 = 9_223_372_036_854_775_807
_ETRADE_ORIGINS = {
    "sandbox": "https://apisb.etrade.com",
    "production": "https://api.etrade.com",
}
_TRANSPORT_OPERATIONS = frozenset(
    {"SUBMIT_PREVIEW", "SUBMIT_PLACE", "AMEND_PREVIEW", "AMEND_PLACE"}
)
_OPTION_SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,14}\Z")


class ETradeBrokerTransportError(RuntimeError):
    """A local transport contract is invalid before broker I/O."""


@dataclass(frozen=True, repr=False)
class _PreparedExchange:
    """Pickle-safe exact request handed to the isolated HTTP worker."""

    method: Literal["POST", "PUT"]
    url: str
    headers: tuple[tuple[str, str], ...] = field(repr=False)
    body: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _validate_prepared_exchange(self)


@dataclass(frozen=True, repr=False)
class _ExchangeResult:
    """Bounded result returned by the isolated HTTP worker."""

    kind: Literal[
        "RESPONSE", "TIMEOUT", "TRANSPORT_ERROR", "MALFORMED_RESPONSE"
    ]
    http_status: int | None = None
    raw_response: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        _validate_exchange_result(self)


@dataclass(frozen=True)
class SelectedBrokerAccount:
    """Account identity obtained from the authenticated List Accounts result."""

    account_id: str = field(repr=False)
    account_id_key: str = field(repr=False)
    institution_type: str

    def __post_init__(self) -> None:
        if type(self) is not SelectedBrokerAccount:
            raise ETradeBrokerTransportError(
                "selected account must use its exact immutable type"
            )
        _broker_numeric_text(self.account_id, "account id")
        _identifier(self.account_id_key, "account key")
        _identifier(self.institution_type, "institution type")

    def runtime_mapping(self) -> dict[str, str]:
        return {
            "accountId": self.account_id,
            "accountIdKey": self.account_id_key,
            "institutionType": self.institution_type,
        }


@dataclass(frozen=True)
class BoundBrokerRequest:
    """Exact request authorized for one account, route, and ledger fence."""

    account_id: str = field(repr=False)
    account_id_key: str = field(repr=False)
    institution_type: str
    environment: Literal["sandbox", "production"]
    intent_id: str
    owner: str
    authorization_operation: Literal["SUBMIT", "AMEND"]
    fencing_token: int
    transport_operation: Literal[
        "SUBMIT_PREVIEW", "SUBMIT_PLACE", "AMEND_PREVIEW", "AMEND_PLACE"
    ]
    http_method: Literal["POST", "PUT"]
    route: str = field(repr=False)
    client_order_id: str = field(repr=False)
    target_broker_order_id: str | None = field(repr=False)
    preview_id: str | None = field(repr=False)
    authorization_payload_digest: str
    final_xml_bytes: bytes = field(repr=False)
    final_xml_sha256: str
    transport_seal: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self) is not BoundBrokerRequest:
            raise ETradeBrokerTransportError("bound request must use its exact type")
        _broker_numeric_text(self.account_id, "account id")
        _identifier(self.account_id_key, "account key")
        _identifier(self.institution_type, "institution type")
        _identifier(self.intent_id, "intent id")
        _identifier(self.owner, "owner")
        _identifier(self.client_order_id, "client order id")
        if type(self.environment) is not str or self.environment not in {
            "sandbox",
            "production",
        }:
            raise ETradeBrokerTransportError("bound request environment is invalid")
        if (
            type(self.authorization_operation) is not str
            or self.authorization_operation not in {"SUBMIT", "AMEND"}
        ):
            raise ETradeBrokerTransportError("bound authorization operation is invalid")
        if type(self.fencing_token) is not int or self.fencing_token <= 0:
            raise ETradeBrokerTransportError("bound fencing token is invalid")
        if (
            type(self.transport_operation) is not str
            or self.transport_operation not in _TRANSPORT_OPERATIONS
        ):
            raise ETradeBrokerTransportError("bound transport operation is invalid")
        if type(self.http_method) is not str or self.http_method not in {"POST", "PUT"}:
            raise ETradeBrokerTransportError("bound HTTP method is invalid")
        if type(self.route) is not str:
            raise ETradeBrokerTransportError("bound route is invalid")
        if self.target_broker_order_id is not None:
            _broker_numeric_text(
                self.target_broker_order_id, "target broker order id"
            )
        if self.preview_id is not None:
            _broker_numeric_text(self.preview_id, "preview id")
        _sha256(self.authorization_payload_digest, "authorization payload digest")
        if type(self.final_xml_bytes) is not bytes or not self.final_xml_bytes:
            raise ETradeBrokerTransportError("bound XML must be non-empty exact bytes")
        _sha256(self.final_xml_sha256, "bound XML digest")
        if not hmac.compare_digest(
            hashlib.sha256(self.final_xml_bytes).hexdigest(), self.final_xml_sha256
        ):
            raise ETradeBrokerTransportError("bound XML digest does not verify")
        if type(self.transport_seal) is not bytes or len(self.transport_seal) != 32:
            raise ETradeBrokerTransportError("bound request seal is invalid")
        _validate_bound_request_shape(self)


@dataclass(frozen=True)
class BrokerMessage:
    code: int
    message_type: Literal["WARNING", "INFO", "INFO_HOLD", "ERROR"]
    description: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self) is not BrokerMessage:
            raise ETradeBrokerTransportError("broker message must use its exact type")
        _validate_broker_message(
            self.description, self.code, self.message_type
        )


@dataclass(frozen=True)
class BrokerReply:
    """Bound, redacted result of exactly one broker mutation request."""

    request: BoundBrokerRequest
    disposition: Literal["ACKNOWLEDGED", "UNKNOWN"]
    http_status: int | None
    broker_status: str | None
    client_order_id: str = field(repr=False)
    echoed_client_order_id: str | None = field(repr=False)
    broker_order_id: str | None = field(repr=False)
    preview_id: str | None = field(repr=False)
    broker_messages: tuple[BrokerMessage, ...]
    raw_response_digest: str | None
    observed_at: datetime
    unknown_reason: str | None

    def __post_init__(self) -> None:
        if type(self) is not BrokerReply or type(self.request) is not BoundBrokerRequest:
            raise ETradeBrokerTransportError("broker reply must use exact bound types")
        if type(self.disposition) is not str or self.disposition not in {
            "ACKNOWLEDGED",
            "UNKNOWN",
        }:
            raise ETradeBrokerTransportError("broker reply disposition is invalid")
        if self.http_status is not None and (
            type(self.http_status) is not int or not 100 <= self.http_status <= 599
        ):
            raise ETradeBrokerTransportError("broker reply HTTP status is invalid")
        _optional_identifier(self.broker_status, "broker status")
        _identifier(self.client_order_id, "client order id")
        if self.client_order_id != self.request.client_order_id:
            raise ETradeBrokerTransportError("broker reply client id is not request-bound")
        _optional_identifier(self.echoed_client_order_id, "echoed client order id")
        if self.broker_order_id is not None:
            _broker_numeric_text(self.broker_order_id, "broker order id")
        if self.preview_id is not None:
            _broker_numeric_text(self.preview_id, "preview id")
        if (
            type(self.broker_messages) is not tuple
            or any(type(message) is not BrokerMessage for message in self.broker_messages)
        ):
            raise ETradeBrokerTransportError("broker reply messages are invalid")
        if self.raw_response_digest is not None:
            _sha256(self.raw_response_digest, "raw response digest")
        if (
            type(self.observed_at) is not datetime
            or self.observed_at.tzinfo is not timezone.utc
        ):
            raise ETradeBrokerTransportError("broker reply time must be exact UTC")
        if (self.disposition == "UNKNOWN") != (self.unknown_reason is not None):
            raise ETradeBrokerTransportError("unknown reply reason is inconsistent")
        if self.unknown_reason is not None:
            _identifier(self.unknown_reason, "unknown reason")
        if self.disposition == "ACKNOWLEDGED":
            if self.broker_messages and not _messages_allow_ack(
                self.request.transport_operation, self.broker_messages
            ):
                raise ETradeBrokerTransportError(
                    "broker messages require review before acknowledgement"
                )
            if self.request.transport_operation.endswith("PREVIEW") and self.preview_id is None:
                raise ETradeBrokerTransportError("preview acknowledgement lacks preview id")
            if self.request.transport_operation.endswith("PLACE") and self.broker_order_id is None:
                raise ETradeBrokerTransportError("place acknowledgement lacks broker order id")


class ETradeBrokerTransport:
    """Issue one ledger-claimed E*TRADE mutation through a pinned adapter."""

    def __init__(
        self,
        *,
        session: OAuth1Session,
        ledger: OrderIntentLedger,
        runtime_safety: RuntimeSafetyBoundary,
        selected_account: SelectedBrokerAccount,
    ) -> None:
        if type(session) is not OAuth1Session:
            raise ETradeBrokerTransportError(
                "order transport requires an exact rauth OAuth1Session"
            )
        if type(ledger) is not OrderIntentLedger:
            raise ETradeBrokerTransportError(
                "order transport requires an exact durable ledger"
            )
        if type(runtime_safety) is not RuntimeSafetyBoundary:
            raise ETradeBrokerTransportError(
                "order transport requires an exact runtime safety boundary"
            )
        if type(selected_account) is not SelectedBrokerAccount:
            raise ETradeBrokerTransportError(
                "order transport requires an exact selected account"
            )
        if (
            runtime_safety.environment not in {"sandbox", "production"}
            or runtime_safety.expected_account_id is None
            or runtime_safety.expected_account_id_key is None
            or runtime_safety.expected_institution_type is None
        ):
            raise ETradeBrokerTransportError(
                "order-capable runtime requires an explicitly armed account"
            )
        runtime_safety.assert_current()
        runtime_safety.verify_account(selected_account.runtime_mapping())
        credentials = (
            session.consumer_key,
            session.consumer_secret,
            session.access_token,
            session.access_token_secret,
        )
        if any(type(value) is not str or not value for value in credentials):
            raise ETradeBrokerTransportError(
                "OAuth session lacks exact consumer/access credentials"
            )
        self._oauth = OAuth1Session(
            session.consumer_key,
            session.consumer_secret,
            access_token=session.access_token,
            access_token_secret=session.access_token_secret,
        )
        self._oauth.trust_env = False
        self._oauth.params = {}
        self._oauth.proxies = {}
        self._oauth.hooks = {"response": []}
        self._oauth.cookies.clear()
        self._oauth.verify = True
        self._oauth.cert = None
        self._ledger = ledger
        self._runtime_safety = runtime_safety
        self._selected_account = selected_account
        self._account_id = selected_account.account_id
        self._account_id_key = selected_account.account_id_key
        self._institution_type = selected_account.institution_type
        self._environment = runtime_safety.environment
        self._binding_secret = secrets.token_bytes(32)
        self._send_lock = threading.Lock()

    def preview(self, authorization: OutboundAuthorization) -> BrokerReply:
        request = self._build_request(
            authorization=authorization,
            expected_authorization_operation="SUBMIT",
            transport_operation="SUBMIT_PREVIEW",
            http_method="POST",
            target_broker_order_id=None,
            preview_id=None,
        )
        return self._execute(request, authorization)

    def place(
        self, authorization: OutboundAuthorization, preview: BrokerReply
    ) -> BrokerReply:
        preview_id = self._preview_id(
            preview, authorization, expected_operation="SUBMIT_PREVIEW"
        )
        request = self._build_request(
            authorization=authorization,
            expected_authorization_operation="SUBMIT",
            transport_operation="SUBMIT_PLACE",
            http_method="POST",
            target_broker_order_id=None,
            preview_id=preview_id,
        )
        return self._execute(request, authorization)

    def preview_change(
        self, target_broker_order_id: str, authorization: OutboundAuthorization
    ) -> BrokerReply:
        request = self._build_request(
            authorization=authorization,
            expected_authorization_operation="AMEND",
            transport_operation="AMEND_PREVIEW",
            http_method="PUT",
            target_broker_order_id=_broker_numeric_text(
                target_broker_order_id, "target broker order id"
            ),
            preview_id=None,
        )
        return self._execute(request, authorization)

    def place_change(
        self,
        target_broker_order_id: str,
        authorization: OutboundAuthorization,
        preview: BrokerReply,
    ) -> BrokerReply:
        preview_id = self._preview_id(
            preview, authorization, expected_operation="AMEND_PREVIEW"
        )
        request = self._build_request(
            authorization=authorization,
            expected_authorization_operation="AMEND",
            transport_operation="AMEND_PLACE",
            http_method="PUT",
            target_broker_order_id=_broker_numeric_text(
                target_broker_order_id, "target broker order id"
            ),
            preview_id=preview_id,
        )
        return self._execute(request, authorization)

    def _preview_id(
        self,
        preview: BrokerReply,
        authorization: OutboundAuthorization,
        *,
        expected_operation: Literal["SUBMIT_PREVIEW", "AMEND_PREVIEW"],
    ) -> str:
        if (
            type(preview) is not BrokerReply
            or preview.disposition != "ACKNOWLEDGED"
            or type(preview.request) is not BoundBrokerRequest
            or preview.request.transport_operation != expected_operation
            or preview.preview_id is None
        ):
            raise ETradeBrokerTransportError(
                "place requires an acknowledged typed preview receipt"
            )
        self._verify_bound_request(preview.request)
        if (
            preview.request.account_id != self._account_id
            or preview.request.account_id_key != self._account_id_key
            or preview.request.environment != self._environment
            or preview.request.intent_id != authorization.intent_id
            or preview.request.owner != authorization.owner
            or preview.request.fencing_token != authorization.fencing_token
            or preview.request.authorization_payload_digest
            != authorization.payload_digest
            or preview.client_order_id != authorization.client_order_id
        ):
            raise ETradeBrokerTransportError(
                "preview receipt does not match the place authorization"
            )
        return _broker_numeric_text(preview.preview_id, "preview id")

    def _build_request(
        self,
        *,
        authorization: OutboundAuthorization,
        expected_authorization_operation: Literal["SUBMIT", "AMEND"],
        transport_operation: Literal[
            "SUBMIT_PREVIEW", "SUBMIT_PLACE", "AMEND_PREVIEW", "AMEND_PLACE"
        ],
        http_method: Literal["POST", "PUT"],
        target_broker_order_id: str | None,
        preview_id: str | None,
    ) -> BoundBrokerRequest:
        order = _authorized_vertical(authorization, expected_authorization_operation)
        xml_bytes = _order_xml(
            order,
            place=transport_operation.endswith("PLACE"),
            preview_id=preview_id,
        )
        encoded_account = quote(self._account_id_key, safe="")
        if transport_operation in {"AMEND_PREVIEW", "AMEND_PLACE"}:
            encoded_target = quote(target_broker_order_id or "", safe="")
            action = "place" if transport_operation == "AMEND_PLACE" else "preview"
            route = (
                f"/v1/accounts/{encoded_account}/orders/"
                f"{encoded_target}/change/{action}"
            )
        else:
            suffix = "place" if transport_operation == "SUBMIT_PLACE" else "preview"
            route = f"/v1/accounts/{encoded_account}/orders/{suffix}"
        fields = {
            "account_id": self._account_id,
            "account_id_key": self._account_id_key,
            "institution_type": self._institution_type,
            "environment": self._environment,
            "intent_id": authorization.intent_id,
            "owner": authorization.owner,
            "authorization_operation": authorization.operation,
            "fencing_token": authorization.fencing_token,
            "transport_operation": transport_operation,
            "http_method": http_method,
            "route": route,
            "client_order_id": authorization.client_order_id,
            "target_broker_order_id": target_broker_order_id,
            "preview_id": preview_id,
            "authorization_payload_digest": authorization.payload_digest,
            "final_xml_bytes": xml_bytes,
            "final_xml_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        }
        transport_seal = hmac.new(
            self._binding_secret,
            _bound_request_material(fields),
            hashlib.sha256,
        ).digest()
        return BoundBrokerRequest(
            **fields,
            transport_seal=transport_seal,
        )

    def _execute(
        self,
        request: BoundBrokerRequest,
        authorization: OutboundAuthorization,
    ) -> BrokerReply:
        with self._send_lock:
            self._verify_bound_request(request)
            trusted_request = _copy_bound_request(request)
            self._verify_bound_request(trusted_request)
            origin = _ETRADE_ORIGINS[trusted_request.environment]
            url = origin + trusted_request.route
            oauth = self._oauth
            prepared = _prepare_oauth_request(oauth, trusted_request, url)
            _validate_prepared_request(prepared, trusted_request, url)

            runtime_safety = self._runtime_safety
            selected_account = self._selected_account
            runtime_safety.assert_current()
            runtime_safety.verify_account(selected_account.runtime_mapping())
            evidence = _transport_evidence(trusted_request)
            self._ledger.claim_transport_send(evidence, authorization)
            runtime_safety.assert_current()
            runtime_safety.verify_account(selected_account.runtime_mapping())
            _validate_prepared_request(prepared, trusted_request, url)

            try:
                exchange = _isolated_exchange(
                    prepared,
                    timeout_seconds=_TOTAL_EXCHANGE_TIMEOUT_SECONDS,
                )
            except Exception:
                reply = _unknown_reply(
                    trusted_request, "TRANSPORT_ERROR"
                )
            else:
                if exchange.kind == "TIMEOUT":
                    reply = _unknown_reply(trusted_request, "TIMEOUT")
                elif exchange.kind == "TRANSPORT_ERROR":
                    reply = _unknown_reply(
                        trusted_request, "TRANSPORT_ERROR"
                    )
                elif exchange.kind == "MALFORMED_RESPONSE":
                    reply = _unknown_reply(
                        trusted_request, "MALFORMED_RESPONSE"
                    )
                else:
                    try:
                        reply = _parse_reply_bytes(
                            trusted_request,
                            exchange.http_status,
                            exchange.raw_response,
                        )
                    except Exception:
                        reply = _unknown_reply(
                            trusted_request, "MALFORMED_RESPONSE"
                        )
            try:
                self._ledger.record_transport_response(
                    evidence, _transport_response_evidence(reply)
                )
            except Exception as exc:
                raise ETradeBrokerTransportError(
                    "broker response could not be persisted durably"
                ) from exc
            return reply

    def _verify_bound_request(self, request: BoundBrokerRequest) -> None:
        if type(request) is not BoundBrokerRequest:
            raise ETradeBrokerTransportError(
                "transport can send only an exact bound request"
            )
        expected_seal = hmac.new(
            self._binding_secret,
            _bound_request_material(request),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(request.transport_seal, expected_seal):
            raise ETradeBrokerTransportError(
                "bound request was not issued by this transport"
            )
        if not hmac.compare_digest(
            hashlib.sha256(request.final_xml_bytes).hexdigest(),
            request.final_xml_sha256,
        ):
            raise ETradeBrokerTransportError("bound XML digest does not verify")
        _validate_bound_request_shape(request)


def _transport_evidence(request: BoundBrokerRequest) -> TransportRequestEvidence:
    return TransportRequestEvidence(
        account_id=request.account_id,
        account_id_key=request.account_id_key,
        institution_type=request.institution_type,
        environment=request.environment,
        intent_id=request.intent_id,
        owner=request.owner,
        authorization_operation=request.authorization_operation,
        fencing_token=request.fencing_token,
        transport_operation=request.transport_operation,
        http_method=request.http_method,
        route=request.route,
        client_order_id=request.client_order_id,
        target_broker_order_id=request.target_broker_order_id,
        preview_id=request.preview_id,
        authorization_payload_digest=request.authorization_payload_digest,
        final_xml_bytes=request.final_xml_bytes,
        final_xml_sha256=request.final_xml_sha256,
    )


def _transport_response_evidence(
    reply: BrokerReply,
) -> TransportResponseEvidence:
    return TransportResponseEvidence(
        disposition=reply.disposition,
        http_status=reply.http_status,
        broker_status=reply.broker_status,
        broker_order_id=reply.broker_order_id,
        preview_id=reply.preview_id,
        message_codes=tuple(
            message.code for message in reply.broker_messages
        ),
        message_types=tuple(
            message.message_type for message in reply.broker_messages
        ),
        message_description_digests=tuple(
            hashlib.sha256(message.description.encode("utf-8")).hexdigest()
            for message in reply.broker_messages
        ),
        raw_response_digest=reply.raw_response_digest,
        observed_at=reply.observed_at,
        unknown_reason=reply.unknown_reason,
    )


def _copy_bound_request(request: BoundBrokerRequest) -> BoundBrokerRequest:
    return BoundBrokerRequest(
        account_id=request.account_id,
        account_id_key=request.account_id_key,
        institution_type=request.institution_type,
        environment=request.environment,
        intent_id=request.intent_id,
        owner=request.owner,
        authorization_operation=request.authorization_operation,
        fencing_token=request.fencing_token,
        transport_operation=request.transport_operation,
        http_method=request.http_method,
        route=request.route,
        client_order_id=request.client_order_id,
        target_broker_order_id=request.target_broker_order_id,
        preview_id=request.preview_id,
        authorization_payload_digest=request.authorization_payload_digest,
        final_xml_bytes=request.final_xml_bytes,
        final_xml_sha256=request.final_xml_sha256,
        transport_seal=request.transport_seal,
    )


def _require_pinned_adapter(adapter: HTTPAdapter) -> None:
    """Reject any adapter configuration that could replay a mutation."""

    if type(adapter) is not HTTPAdapter:
        raise ETradeBrokerTransportError(
            "order transport requires the standard no-retry HTTP adapter"
        )
    retries = adapter.max_retries
    retry_fields = (
        getattr(retries, "total", None),
        getattr(retries, "connect", None),
        getattr(retries, "read", None),
        getattr(retries, "redirect", None),
        getattr(retries, "status", None),
        getattr(retries, "other", None),
    )
    if retry_fields[0] not in {0, False} or any(
        value not in {None, 0, False} for value in retry_fields[1:]
    ):
        raise ETradeBrokerTransportError(
            "order transport forbids configured HTTP retries"
        )


def _new_pinned_adapter() -> HTTPAdapter:
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=0,
            connect=0,
            read=0,
            redirect=0,
            status=0,
            other=0,
        )
    )
    _require_pinned_adapter(adapter)
    return adapter


def _validate_prepared_exchange(exchange: _PreparedExchange) -> None:
    if type(exchange) is not _PreparedExchange:
        raise ETradeBrokerTransportError("isolated exchange request type is invalid")
    parsed = urlsplit(exchange.url)
    if (
        type(exchange.method) is not str
        or exchange.method not in {"POST", "PUT"}
        or type(exchange.url) is not str
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or type(exchange.headers) is not tuple
        or type(exchange.body) is not bytes
        or not exchange.body
    ):
        raise ETradeBrokerTransportError("isolated exchange request is invalid")
    normalized: dict[str, str] = {}
    for pair in exchange.headers:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or not pair[0]
        ):
            raise ETradeBrokerTransportError(
                "isolated exchange headers are invalid"
            )
        name = pair[0].lower()
        if name in normalized:
            raise ETradeBrokerTransportError(
                "isolated exchange headers contain duplicates"
            )
        normalized[name] = pair[1]
    expected_headers = {
        "content-type",
        "accept",
        "accept-encoding",
        "consumerkey",
        "content-length",
        "authorization",
    }
    if (
        set(normalized) != expected_headers
        or normalized["content-type"] != "application/xml"
        or normalized["accept"] != "application/json"
        or normalized["accept-encoding"] != "identity"
        or normalized["content-length"] != str(len(exchange.body))
        or not normalized["consumerkey"]
        or not normalized["authorization"].startswith("OAuth ")
    ):
        raise ETradeBrokerTransportError(
            "isolated exchange headers changed the sealed mutation"
        )


def _validate_exchange_result(result: _ExchangeResult) -> None:
    if (
        type(result) is not _ExchangeResult
        or type(result.kind) is not str
        or result.kind
        not in {
            "RESPONSE",
            "TIMEOUT",
            "TRANSPORT_ERROR",
            "MALFORMED_RESPONSE",
        }
        or type(result.raw_response) is not bytes
    ):
        raise ETradeBrokerTransportError("isolated exchange result is invalid")
    if result.kind == "RESPONSE":
        if (
            type(result.http_status) is not int
            or not 100 <= result.http_status <= 599
            or len(result.raw_response) > _MAX_RESPONSE_BYTES + 1
        ):
            raise ETradeBrokerTransportError(
                "isolated exchange response is invalid"
            )
    elif result.http_status is not None or result.raw_response:
        raise ETradeBrokerTransportError(
            "isolated exchange failure contains unexpected response data"
        )


def _serialized_prepared_request(prepared: Any) -> _PreparedExchange:
    if type(prepared) is not PreparedRequest:
        raise ETradeBrokerTransportError(
            "isolated exchange requires an exact prepared request"
        )
    headers = getattr(prepared, "headers", None)
    if not hasattr(headers, "items"):
        raise ETradeBrokerTransportError("prepared request headers are invalid")
    return _PreparedExchange(
        method=prepared.method,
        url=prepared.url,
        headers=tuple((name, value) for name, value in headers.items()),
        body=prepared.body,
    )


def _reconstruct_prepared_request(exchange: _PreparedExchange) -> PreparedRequest:
    _validate_prepared_exchange(exchange)
    prepared = PreparedRequest()
    prepared.prepare(
        method=exchange.method,
        url=exchange.url,
        headers=dict(exchange.headers),
        data=exchange.body,
    )
    reconstructed = _serialized_prepared_request(prepared)
    if reconstructed != exchange:
        raise ETradeBrokerTransportError(
            "isolated worker changed the prepared request"
        )
    return prepared


def _write_exchange_result(output: Any, result: _ExchangeResult) -> None:
    _validate_exchange_result(result)
    if len(output) != _EXCHANGE_RESULT_BUFFER_BYTES:
        raise ETradeBrokerTransportError(
            "isolated exchange result buffer is invalid"
        )
    status = result.http_status or 0
    header = struct.pack(
        "!BHI32s",
        _EXCHANGE_KIND_CODES[result.kind],
        status,
        len(result.raw_response),
        hashlib.sha256(result.raw_response).digest(),
    )
    end = _EXCHANGE_RESULT_HEADER_BYTES + len(result.raw_response)
    if end > len(output):
        raise ETradeBrokerTransportError(
            "isolated exchange result exceeds its fixed buffer"
        )
    output[_EXCHANGE_RESULT_HEADER_BYTES:end] = result.raw_response
    # The authenticated header is the commit record and is written last.
    output[:_EXCHANGE_RESULT_HEADER_BYTES] = header


def _read_exchange_result(output: Any) -> _ExchangeResult:
    if len(output) != _EXCHANGE_RESULT_BUFFER_BYTES:
        raise ETradeBrokerTransportError(
            "isolated exchange result buffer is invalid"
        )
    header = bytes(output[:_EXCHANGE_RESULT_HEADER_BYTES])
    code, status, raw_length, expected_digest = struct.unpack(
        "!BHI32s", header
    )
    kind = _EXCHANGE_CODE_KINDS.get(code)
    if kind is None or raw_length > _MAX_RESPONSE_BYTES + 1:
        raise ETradeBrokerTransportError(
            "isolated exchange result frame is invalid"
        )
    end = _EXCHANGE_RESULT_HEADER_BYTES + raw_length
    raw = bytes(output[_EXCHANGE_RESULT_HEADER_BYTES:end])
    if not hmac.compare_digest(
        hashlib.sha256(raw).digest(), expected_digest
    ):
        raise ETradeBrokerTransportError(
            "isolated exchange result frame is incomplete"
        )
    result = _ExchangeResult(
        kind,
        http_status=status or None,
        raw_response=raw,
    )
    _validate_exchange_result(result)
    return result


def _exchange_worker(output: Any, exchange: _PreparedExchange) -> None:
    """Perform exactly one mutation in a disposable process."""

    adapter: HTTPAdapter | None = None
    try:
        prepared = _reconstruct_prepared_request(exchange)
        adapter = _new_pinned_adapter()
        response = adapter.send(
            prepared,
            stream=True,
            timeout=_REQUEST_TIMEOUT,
            verify=True,
            cert=None,
            proxies={},
        )
        try:
            status, raw = _read_bounded_response(response)
        except (Timeout, TimeoutError):
            result = _ExchangeResult("TIMEOUT")
        except Exception:
            result = _ExchangeResult("MALFORMED_RESPONSE")
        else:
            result = _ExchangeResult(
                "RESPONSE", http_status=status, raw_response=raw
            )
    except (Timeout, TimeoutError):
        result = _ExchangeResult("TIMEOUT")
    except Exception:
        result = _ExchangeResult("TRANSPORT_ERROR")
    finally:
        if adapter is not None:
            adapter.close()
        try:
            _write_exchange_result(output, result)
        except Exception:
            pass


def _stop_exchange_process(process: Any) -> None:
    try:
        process.join(timeout=0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.25)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.25)
    except Exception:
        pass


def _isolated_exchange(
    prepared: Any, *, timeout_seconds: float
) -> _ExchangeResult:
    """Enforce one hard deadline over connect, headers, and response body."""

    if (
        type(timeout_seconds) is not float
        or not 0 < timeout_seconds <= _TOTAL_EXCHANGE_TIMEOUT_SECONDS
    ):
        raise ETradeBrokerTransportError("total exchange deadline is invalid")
    exchange = _serialized_prepared_request(prepared)
    context = multiprocessing.get_context("spawn")
    output = context.RawArray("B", _EXCHANGE_RESULT_BUFFER_BYTES)
    process = context.Process(
        target=_exchange_worker,
        args=(output, exchange),
        daemon=True,
    )
    deadline = time.monotonic() + timeout_seconds
    started = False
    try:
        process.start()
        started = True
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
        if process.is_alive():
            return _ExchangeResult("TIMEOUT")
        try:
            result = _read_exchange_result(output)
        except Exception:
            return _ExchangeResult("TRANSPORT_ERROR")
        return result
    except Exception:
        return _ExchangeResult("TRANSPORT_ERROR")
    finally:
        if started:
            _stop_exchange_process(process)


def _prepare_oauth_request(
    oauth: OAuth1Session,
    request: BoundBrokerRequest,
    url: str,
):
    if (
        type(oauth) is not OAuth1Session
        or oauth.trust_env is not False
        or oauth.params
        or oauth.proxies
        or oauth.hooks != {"response": []}
        or oauth.cookies
        or oauth.verify is not True
        or oauth.cert is not None
    ):
        raise ETradeBrokerTransportError(
            "private OAuth signer configuration is unsafe"
        )
    headers = CaseInsensitiveDict(
        {
            "Content-Type": "application/xml",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "consumerKey": oauth.consumer_key,
        }
    )
    signing_kwargs = {
        "headers": headers,
        "data": memoryview(request.final_xml_bytes),
    }
    oauth_params = oauth._get_oauth_params(signing_kwargs)
    oauth_params["oauth_signature"] = oauth.signature.sign(
        oauth.consumer_secret,
        oauth.access_token_secret,
        request.http_method,
        url,
        oauth_params,
        signing_kwargs,
    )
    return Request(
        method=request.http_method,
        url=url,
        headers=dict(headers),
        data=request.final_xml_bytes,
        auth=OAuth1Auth(oauth_params, ""),
    ).prepare()


def _validate_prepared_request(
    prepared: Any,
    request: BoundBrokerRequest,
    expected_url: str,
) -> None:
    parsed = urlsplit(prepared.url)
    body = prepared.body
    headers = prepared.headers
    if (
        prepared.method != request.http_method
        or prepared.url != expected_url
        or parsed.scheme != "https"
        or parsed.query
        or parsed.fragment
        or type(body) is not bytes
        or not hmac.compare_digest(body, request.final_xml_bytes)
        or headers.get("Content-Type") != "application/xml"
        or headers.get("Accept") != "application/json"
        or headers.get("Accept-Encoding") != "identity"
        or headers.get("Content-Length") != str(len(request.final_xml_bytes))
        or type(headers.get("Authorization")) is not str
        or not headers["Authorization"].startswith("OAuth ")
        or "Cookie" in headers
        or "Transfer-Encoding" in headers
    ):
        raise ETradeBrokerTransportError(
            "prepared OAuth request changed its sealed mutation"
        )


def _bound_request_material(value: Any) -> bytes:
    names = (
        "account_id",
        "account_id_key",
        "institution_type",
        "environment",
        "intent_id",
        "owner",
        "authorization_operation",
        "fencing_token",
        "transport_operation",
        "http_method",
        "route",
        "client_order_id",
        "target_broker_order_id",
        "preview_id",
        "authorization_payload_digest",
        "final_xml_sha256",
    )
    if type(value) is dict:
        document = {name: value[name] for name in names}
    elif type(value) is BoundBrokerRequest:
        document = {name: getattr(value, name) for name in names}
    else:
        raise ETradeBrokerTransportError("bound request material is invalid")
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validate_bound_request_shape(request: BoundBrokerRequest) -> None:
    operation = request.transport_operation
    expected_authorization = "SUBMIT" if operation.startswith("SUBMIT") else "AMEND"
    expected_method = "POST" if operation.startswith("SUBMIT") else "PUT"
    if (
        request.authorization_operation != expected_authorization
        or request.http_method != expected_method
    ):
        raise ETradeBrokerTransportError(
            "bound request operation and method are inconsistent"
        )
    is_amend = operation.startswith("AMEND")
    is_place = operation.endswith("PLACE")
    if is_amend != (request.target_broker_order_id is not None):
        raise ETradeBrokerTransportError(
            "bound request target is inconsistent"
        )
    if is_place != (request.preview_id is not None):
        raise ETradeBrokerTransportError(
            "bound request preview is inconsistent"
        )
    encoded_account = quote(request.account_id_key, safe="")
    if is_amend:
        encoded_target = quote(request.target_broker_order_id or "", safe="")
        action = "place" if is_place else "preview"
        expected_route = (
            f"/v1/accounts/{encoded_account}/orders/"
            f"{encoded_target}/change/{action}"
        )
    else:
        action = "place" if is_place else "preview"
        expected_route = f"/v1/accounts/{encoded_account}/orders/{action}"
    if request.route != expected_route:
        raise ETradeBrokerTransportError("bound request route is inconsistent")
    try:
        root = ET.fromstring(request.final_xml_bytes)
    except ET.ParseError as exc:
        raise ETradeBrokerTransportError("bound XML is invalid") from exc
    expected_root = "PlaceOrderRequest" if is_place else "PreviewOrderRequest"
    if (
        root.tag != expected_root
        or root.findtext("./orderType") != "SPREADS"
        or root.findtext("./clientOrderId") != request.client_order_id
    ):
        raise ETradeBrokerTransportError("bound XML identity is inconsistent")
    bound_preview = root.findtext("./PreviewIds/previewId")
    if bound_preview != request.preview_id:
        raise ETradeBrokerTransportError("bound XML preview is inconsistent")


def _authorized_vertical(
    authorization: OutboundAuthorization,
    expected_operation: Literal["SUBMIT", "AMEND"],
) -> dict[str, Any]:
    if type(authorization) is not OutboundAuthorization:
        raise ETradeBrokerTransportError("authorization must use its exact immutable type")
    primitive_fields = (
        (authorization.intent_id, str),
        (authorization.operation, str),
        (authorization.owner, str),
        (authorization.fencing_token, int),
        (authorization.client_order_id, str),
        (authorization.payload_bytes, bytes),
        (authorization.payload_digest, str),
    )
    if any(type(value) is not expected for value, expected in primitive_fields):
        raise ETradeBrokerTransportError("authorization fields must be exact primitives")
    _identifier(authorization.intent_id, "intent id")
    _identifier(authorization.owner, "owner")
    _identifier(authorization.client_order_id, "client order id")
    if (
        len(authorization.client_order_id) != 10
        or not authorization.client_order_id.isascii()
        or not authorization.client_order_id.isdigit()
    ):
        raise ETradeBrokerTransportError("client order id must contain ten digits")
    if authorization.operation != expected_operation:
        raise ETradeBrokerTransportError("authorization operation does not match request")
    if authorization.fencing_token <= 0:
        raise ETradeBrokerTransportError("authorization fencing token is invalid")
    _sha256(authorization.payload_digest, "authorization payload digest")
    if (
        not authorization.payload_bytes
        or len(authorization.payload_bytes) > _MAX_AUTHORIZATION_BYTES
        or not hmac.compare_digest(
            hashlib.sha256(authorization.payload_bytes).hexdigest(),
            authorization.payload_digest,
        )
    ):
        raise ETradeBrokerTransportError("authorization payload does not verify")
    try:
        payload = json.loads(
            authorization.payload_bytes.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETradeBrokerTransportError("authorization payload is invalid") from exc
    if type(payload) is not dict:
        raise ETradeBrokerTransportError("authorization payload must be an object")
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if not hmac.compare_digest(canonical_bytes, authorization.payload_bytes):
        raise ETradeBrokerTransportError("authorization payload is not canonical")
    payload = dict(payload)
    client_order_id = payload.pop("client_order_id", None)
    if type(client_order_id) is not str or client_order_id != authorization.client_order_id:
        raise ETradeBrokerTransportError("authorization client order id is inconsistent")
    _validate_opening_vertical_payload(payload)
    try:
        wire_order_payload(payload)
    except Exception as exc:
        raise ETradeBrokerTransportError("authorization order schema is invalid") from exc
    if (
        payload.get("securityType") != "OPTN"
        or payload.get("orderAction") != "SPREAD"
        or payload.get("spreadType") != "VERTICAL"
        or type(payload.get("legs")) is not list
        or len(payload["legs"]) != 2
        or any(
            type(leg) is not dict
            or type(leg.get("orderAction")) is not str
            or not leg["orderAction"].endswith("_OPEN")
            for leg in payload["legs"]
        )
    ):
        raise ETradeBrokerTransportError(
            "transport currently supports opening vertical options only"
        )
    payload["client_order_id"] = client_order_id
    return payload


def _validate_opening_vertical_payload(payload: dict[str, Any]) -> None:
    required_top_level = {
        "securityType",
        "orderAction",
        "priceType",
        "limitPrice",
        "orderTerm",
        "spreadType",
        "legs",
    }
    if set(payload) != required_top_level:
        raise ETradeBrokerTransportError(
            "transport requires an exact opening-vertical payload"
        )
    exact_enums = (
        (payload["securityType"], {"OPTN"}),
        (payload["orderAction"], {"SPREAD"}),
        (payload["priceType"], {"NET_CREDIT", "NET_DEBIT"}),
        (payload["orderTerm"], {"GOOD_FOR_DAY"}),
        (payload["spreadType"], {"VERTICAL"}),
    )
    if any(type(value) is not str or value not in allowed for value, allowed in exact_enums):
        raise ETradeBrokerTransportError("opening-vertical enum is invalid")
    legs = payload["legs"]
    if type(legs) is not list or len(legs) != 2:
        raise ETradeBrokerTransportError("opening vertical requires exactly two legs")
    required_leg = {
        "symbol",
        "callPut",
        "expiryYear",
        "expiryMonth",
        "expiryDay",
        "strikePrice",
        "orderAction",
        "quantity",
    }
    identities: list[tuple[str, str, int, int, int, int]] = []
    strikes: list[Decimal] = []
    actions: set[str] = set()
    for leg in legs:
        if type(leg) is not dict or set(leg) != required_leg:
            raise ETradeBrokerTransportError("opening-vertical leg shape is invalid")
        symbol = leg["symbol"]
        call_put = leg["callPut"]
        action = leg["orderAction"]
        expiry = (leg["expiryYear"], leg["expiryMonth"], leg["expiryDay"])
        quantity = leg["quantity"]
        if (
            type(symbol) is not str
            or _OPTION_SYMBOL.fullmatch(symbol) is None
            or type(call_put) is not str
            or call_put not in {"PUT", "CALL"}
            or type(action) is not str
            or action not in {"BUY_OPEN", "SELL_OPEN"}
            or any(type(value) is not int for value in expiry)
            or type(quantity) is not int
            or quantity <= 0
        ):
            raise ETradeBrokerTransportError(
                "opening-vertical leg value is invalid"
            )
        try:
            date(*expiry)
        except ValueError as exc:
            raise ETradeBrokerTransportError(
                "opening-vertical expiry is invalid"
            ) from exc
        strike = _finite_decimal(leg["strikePrice"], "strike price", positive=True)
        identities.append((symbol, call_put, *expiry, quantity))
        strikes.append(strike)
        actions.add(action)
    if identities[0] != identities[1]:
        raise ETradeBrokerTransportError(
            "opening-vertical leg identities are inconsistent"
        )
    if actions != {"BUY_OPEN", "SELL_OPEN"} or strikes[0] == strikes[1]:
        raise ETradeBrokerTransportError(
            "opening vertical requires one buy, one sell, and distinct strikes"
        )
    limit_price = _finite_decimal(
        payload["limitPrice"],
        "limit price",
        positive=payload["priceType"] == "NET_DEBIT",
    )
    width = abs(strikes[0] - strikes[1])
    sell_index = next(
        index for index, leg in enumerate(legs) if leg["orderAction"] == "SELL_OPEN"
    )
    buy_index = 1 - sell_index
    call_put = legs[0]["callPut"]
    short_risk = (
        call_put == "PUT" and strikes[sell_index] > strikes[buy_index]
    ) or (
        call_put == "CALL" and strikes[sell_index] < strikes[buy_index]
    )
    expected_price_type = "NET_CREDIT" if short_risk else "NET_DEBIT"
    if payload["priceType"] != expected_price_type:
        raise ETradeBrokerTransportError(
            "opening-vertical price type does not match its risk orientation"
        )
    if (
        payload["priceType"] == "NET_CREDIT"
        and not (Decimal("0") <= limit_price < width)
    ) or (
        payload["priceType"] == "NET_DEBIT"
        and not (Decimal("0") < limit_price <= width)
    ):
        raise ETradeBrokerTransportError(
            "opening-vertical limit price is outside its bounded width"
        )


def _finite_decimal(value: Any, label: str, *, positive: bool) -> Decimal:
    if type(value) not in {int, float} or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ETradeBrokerTransportError(f"{label} is invalid")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or (parsed <= 0 if positive else parsed < 0):
        raise ETradeBrokerTransportError(f"{label} is invalid")
    return parsed


def _order_xml(
    order: dict[str, Any], *, place: bool, preview_id: str | None
) -> bytes:
    root = ET.Element("PlaceOrderRequest" if place else "PreviewOrderRequest")
    _element(root, "orderType", "SPREADS")
    _element(root, "clientOrderId", order["client_order_id"])
    if place:
        if preview_id is None:
            raise ETradeBrokerTransportError("place request requires preview id")
        preview_ids = ET.SubElement(root, "PreviewIds")
        _element(preview_ids, "previewId", preview_id)
    broker_order = ET.SubElement(root, "Order")
    _element(broker_order, "allOrNone", "false")
    _element(broker_order, "priceType", order["priceType"])
    _element(broker_order, "limitPrice", order["limitPrice"])
    _element(broker_order, "stopPrice", 0)
    _element(broker_order, "orderTerm", order["orderTerm"])
    _element(broker_order, "marketSession", "REGULAR")
    for leg in order["legs"]:
        instrument = ET.SubElement(broker_order, "Instrument")
        product = ET.SubElement(instrument, "Product")
        _element(product, "securityType", "OPTN")
        for field in (
            "symbol",
            "callPut",
            "expiryYear",
            "expiryMonth",
            "expiryDay",
            "strikePrice",
        ):
            _element(product, field, leg[field])
        _element(instrument, "orderAction", leg["orderAction"])
        _element(instrument, "quantityType", "QUANTITY")
        _element(instrument, "quantity", leg["quantity"])
        _element(instrument, "orderedQuantity", leg["quantity"])
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _element(parent: ET.Element, tag: str, value: Any) -> None:
    child = ET.SubElement(parent, tag)
    child.text = str(value)


def _parse_reply(request: BoundBrokerRequest, response: Any) -> BrokerReply:
    try:
        status, raw = _read_bounded_response(response)
    except Exception:
        return _unknown_reply(
            request,
            "MALFORMED_RESPONSE",
            observed_at=datetime.now(timezone.utc),
        )
    return _parse_reply_bytes(request, status, raw)


def _parse_reply_bytes(
    request: BoundBrokerRequest,
    status: int | None,
    raw: bytes,
) -> BrokerReply:
    observed_at = datetime.now(timezone.utc)
    if (
        type(status) is not int
        or not 100 <= status <= 599
        or type(raw) is not bytes
    ):
        return _unknown_reply(
            request,
            "MALFORMED_RESPONSE",
            observed_at=observed_at,
        )
    raw_digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > _MAX_RESPONSE_BYTES:
        return _unknown_reply(
            request,
            "RESPONSE_TOO_LARGE",
            http_status=status,
            raw_response_digest=raw_digest,
            observed_at=observed_at,
        )
    if status != 200:
        return _unknown_reply(
            request,
            "HTTP_STATUS",
            http_status=status,
            raw_response_digest=raw_digest,
            observed_at=observed_at,
        )
    try:
        parsed = _parse_response(request.transport_operation, raw)
    except Exception:
        return _unknown_reply(
            request,
            "MALFORMED_RESPONSE",
            http_status=status,
            raw_response_digest=raw_digest,
            observed_at=observed_at,
        )
    if (
        parsed["account_id"] != request.account_id
        or parsed["order_type"] != "SPREADS"
    ):
        return _unknown_reply(
            request,
            "MALFORMED_RESPONSE",
            http_status=status,
            raw_response_digest=raw_digest,
            observed_at=observed_at,
        )
    broker_status = parsed["broker_status"]
    broker_messages = parsed["broker_messages"]
    parsed_preview = parsed["preview_id"]
    if request.transport_operation.endswith("PREVIEW"):
        if parsed_preview is None:
            return _unknown_reply(
                request,
                "MALFORMED_RESPONSE",
                http_status=status,
                raw_response_digest=raw_digest,
                observed_at=observed_at,
            )
        reply_preview = parsed_preview
        broker_order_id = None
    else:
        broker_order_id = parsed["broker_order_id"]
        if broker_order_id is None:
            return _unknown_reply(
                request,
                "MALFORMED_RESPONSE",
                http_status=status,
                raw_response_digest=raw_digest,
                observed_at=observed_at,
            )
        reply_preview = request.preview_id
    if broker_messages and not _messages_allow_ack(
        request.transport_operation, broker_messages
    ):
        return _unknown_reply(
            request,
            "REVIEW_REQUIRED",
            http_status=status,
            broker_status=broker_status,
            broker_order_id=broker_order_id,
            preview_id=reply_preview,
            broker_messages=broker_messages,
            raw_response_digest=raw_digest,
            observed_at=observed_at,
        )
    allowed_statuses = {None, "OPEN"}
    if broker_status not in allowed_statuses:
        return _unknown_reply(
            request,
            "BROKER_STATUS_REQUIRES_RECONCILIATION",
            http_status=status,
            broker_status=broker_status,
            broker_order_id=broker_order_id,
            preview_id=reply_preview,
            raw_response_digest=raw_digest,
            observed_at=observed_at,
        )
    return BrokerReply(
        request=request,
        disposition="ACKNOWLEDGED",
        http_status=status,
        broker_status=broker_status,
        client_order_id=request.client_order_id,
        # E*TRADE documents that clientOrderId is not returned. Binding comes
        # from this single no-redirect/no-retry HTTP exchange, not an echo.
        echoed_client_order_id=None,
        broker_order_id=broker_order_id,
        preview_id=reply_preview,
        broker_messages=broker_messages,
        raw_response_digest=raw_digest,
        observed_at=observed_at,
        unknown_reason=None,
    )


def _read_bounded_response(response: Any) -> tuple[int, bytes]:
    status = getattr(response, "status_code", None)
    raw_stream = getattr(response, "raw", None)
    reader = getattr(raw_stream, "read1", None)
    read_chunk_size = 8192
    if not callable(reader):
        reader = getattr(raw_stream, "read", None)
        read_chunk_size = 1
    closer = getattr(response, "close", None)
    if (
        type(status) is not int
        or not 100 <= status <= 599
        or not callable(reader)
        or not callable(closer)
    ):
        raise ETradeBrokerTransportError("broker response stream is invalid")
    try:
        headers = getattr(response, "headers", None)
        if headers is not None:
            get_header = getattr(headers, "get", None)
            if not callable(get_header):
                raise ETradeBrokerTransportError(
                    "broker response headers are invalid"
                )
            content_encoding = get_header("Content-Encoding")
            if content_encoding is not None and (
                type(content_encoding) is not str
                or content_encoding.strip().lower() != "identity"
            ):
                raise ETradeBrokerTransportError(
                    "encoded broker responses are forbidden"
                )
            content_length = get_header("Content-Length")
            if content_length is not None and (
                type(content_length) is not str
                or not content_length.isascii()
                or not content_length.isdigit()
                or int(content_length) > _MAX_RESPONSE_BYTES
            ):
                raise ETradeBrokerTransportError(
                    "broker response length is invalid"
                )
        deadline = time.monotonic() + _RESPONSE_WALL_TIMEOUT_SECONDS
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_RESPONSE_BYTES:
            if time.monotonic() >= deadline:
                raise TimeoutError("broker response wall deadline expired")
            amount = min(
                read_chunk_size,
                _MAX_RESPONSE_BYTES + 1 - total,
            )
            try:
                chunk = reader(amount, decode_content=False)
            except TypeError:
                chunk = reader(amount)
            if type(chunk) is not bytes:
                raise ETradeBrokerTransportError(
                    "broker response bytes are invalid"
                )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        payload = b"".join(chunks)
    finally:
        closer()
    if type(payload) is not bytes:
        raise ETradeBrokerTransportError("broker response bytes are invalid")
    return status, payload


def _parse_response(operation: str, raw: bytes) -> dict[str, Any]:
    stripped = raw.lstrip()
    if not stripped:
        raise ETradeBrokerTransportError("broker response is empty")
    if stripped.startswith(b"<"):
        if b"<!doctype" in stripped.lower() or b"<!entity" in stripped.lower():
            raise ETradeBrokerTransportError("broker XML declarations are forbidden")
        root = ET.fromstring(raw)
        expected_root = _response_root(operation)
        if root.tag != expected_root:
            raise ETradeBrokerTransportError("broker XML response root is invalid")
        nodes = list(root.iter())
        if (
            len(nodes) > _MAX_RESPONSE_NODES
            or any(
                type(element.tag) is not str
                or "{" in element.tag
                or "}" in element.tag
                or element.attrib
                or element.tag.lower() == "error"
                for element in nodes
            )
        ):
            raise ETradeBrokerTransportError("broker XML response is unsafe")
        broker_messages = _validate_xml_messages(root)
        account_id = _broker_numeric_id(
            _one_xml_text(root, "accountId"), "response account id"
        )
        order_type = _one_xml_text(root, "orderType")
        if operation.endswith("PREVIEW"):
            identifier_container = _one_xml_child(root, "PreviewIds")
            preview_id = _broker_numeric_id(
                _one_xml_identifier(identifier_container, "previewId"),
                "response preview id",
            )
            broker_order_id = None
        else:
            identifier_container = _one_xml_child(root, "OrderIds")
            broker_order_id = _broker_numeric_id(
                _one_xml_identifier(identifier_container, "orderId"),
                "response order id",
            )
            preview_id = None
        broker_status = _xml_order_status(root)
        return {
            "account_id": account_id,
            "order_type": order_type,
            "preview_id": preview_id,
            "broker_order_id": broker_order_id,
            "broker_status": broker_status,
            "broker_messages": broker_messages,
        }
    document = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    expected_root = _response_root(operation)
    if type(document) is not dict or set(document) != {expected_root}:
        raise ETradeBrokerTransportError("broker JSON response must be an object")
    root = document[expected_root]
    if type(root) is not dict:
        raise ETradeBrokerTransportError("broker JSON response root is invalid")
    _validate_json_tree(root)
    broker_messages = _validate_json_messages(root)
    account_id = _broker_numeric_id(
        root.get("accountId"), "response account id"
    )
    order_type = root.get("orderType")
    if type(order_type) is not str:
        raise ETradeBrokerTransportError("response order type is invalid")
    if operation.endswith("PREVIEW"):
        preview_id = _broker_numeric_id(
            _one_json_identifier(root.get("PreviewIds"), "previewId"),
            "response preview id",
        )
        broker_order_id = None
    else:
        broker_order_id = _broker_numeric_id(
            _one_json_identifier(root.get("OrderIds"), "orderId"),
            "response order id",
        )
        preview_id = None
    return {
        "account_id": account_id,
        "order_type": order_type,
        "preview_id": preview_id,
        "broker_order_id": broker_order_id,
        "broker_status": _json_order_status(root),
        "broker_messages": broker_messages,
    }


def _response_root(operation: str) -> str:
    if operation.endswith("PREVIEW"):
        return "PreviewOrderResponse"
    return "PlaceOrderResponse"


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ETradeBrokerTransportError("broker JSON response has duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> Any:
    raise ETradeBrokerTransportError(
        "non-finite JSON numbers are forbidden"
    )


def _validate_json_tree(value: Any) -> None:
    pending = [value]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        if visited > _MAX_RESPONSE_NODES:
            raise ETradeBrokerTransportError("broker response is too complex")
        if type(current) is dict:
            for key, child in current.items():
                if type(key) is not str:
                    raise ETradeBrokerTransportError("broker response key is invalid")
                if key.lower() == "error":
                    raise ETradeBrokerTransportError("broker response contains an error")
                if type(child) in {dict, list}:
                    pending.append(child)
        elif type(current) is list:
            pending.extend(current)
        elif type(current) is float and not math.isfinite(current):
            raise ETradeBrokerTransportError(
                "broker response contains a non-finite number"
            )


def _one_json_identifier(value: Any, key: str) -> Any:
    if type(value) is dict:
        items = [value]
    elif type(value) is list:
        items = value
    else:
        raise ETradeBrokerTransportError("broker response identifier group is invalid")
    if (
        len(items) != 1
        or type(items[0]) is not dict
        or key not in items[0]
        or set(items[0]) - {key, "cashMargin"}
    ):
        raise ETradeBrokerTransportError("broker response identifier is ambiguous")
    cash_margin = items[0].get("cashMargin")
    if cash_margin is not None and (
        type(cash_margin) is not str
        or cash_margin not in {"CASH", "MARGIN", "INVALID"}
    ):
        raise ETradeBrokerTransportError(
            "broker response cash margin is invalid"
        )
    return items[0][key]


def _json_order_status(root: dict[str, Any]) -> str | None:
    items = _json_orders(root)
    if not items:
        return None
    status = items[0].get("status")
    if status is None:
        return None
    return _broker_status(status)


def _json_orders(root: dict[str, Any]) -> list[dict[str, Any]]:
    orders = root.get("Order")
    if orders is None:
        return []
    if type(orders) is dict:
        items = [orders]
    elif type(orders) is list:
        items = orders
    else:
        raise ETradeBrokerTransportError("broker response order is invalid")
    if len(items) != 1 or type(items[0]) is not dict:
        raise ETradeBrokerTransportError("broker response order is ambiguous")
    return items


def _validate_json_messages(
    root: dict[str, Any],
) -> tuple[BrokerMessage, ...]:
    allowed_containers: list[dict[str, Any]] = []
    if "messageList" in root:
        container = root["messageList"]
        if type(container) is not dict:
            raise ETradeBrokerTransportError("broker message list is invalid")
        allowed_containers.append(container)
    for order in _json_orders(root):
        if "messages" in order:
            container = order["messages"]
            if type(container) is not dict:
                raise ETradeBrokerTransportError("broker messages are invalid")
            allowed_containers.append(container)

    allowed_container_ids = {id(container) for container in allowed_containers}
    pending: list[Any] = [root]
    while pending:
        current = pending.pop()
        if type(current) is dict:
            for key, value in current.items():
                if key in {"messageList", "messages"} and id(value) not in allowed_container_ids:
                    raise ETradeBrokerTransportError(
                        "broker message container is misplaced"
                    )
                if key == "Message" and id(current) not in allowed_container_ids:
                    raise ETradeBrokerTransportError(
                        "broker message entry is misplaced"
                    )
                if type(value) in {dict, list}:
                    pending.append(value)
        elif type(current) is list:
            pending.extend(current)

    result: list[BrokerMessage] = []
    for container in allowed_containers:
        if set(container) != {"Message"}:
            raise ETradeBrokerTransportError("broker message container is malformed")
        messages = container["Message"]
        if type(messages) is dict:
            items = [messages]
        elif type(messages) is list:
            items = messages
        else:
            raise ETradeBrokerTransportError("broker message list is malformed")
        if not items:
            raise ETradeBrokerTransportError("broker message list is empty")
        for message in items:
            if type(message) is not dict or set(message) != {
                "description",
                "code",
                "type",
            }:
                raise ETradeBrokerTransportError("broker message is malformed")
            result.append(
                BrokerMessage(
                    code=_message_code(message["code"]),
                    message_type=message["type"],
                    description=message["description"],
                )
            )
    return tuple(result)


def _validate_xml_messages(root: ET.Element) -> tuple[BrokerMessage, ...]:
    orders = [child for child in list(root) if child.tag == "Order"]
    if len(orders) > 1:
        raise ETradeBrokerTransportError("broker XML response order is ambiguous")
    allowed_containers = [
        child for child in list(root) if child.tag == "messageList"
    ]
    if len(allowed_containers) > 1:
        raise ETradeBrokerTransportError("broker XML message list is ambiguous")
    if orders:
        order_containers = [
            child for child in list(orders[0]) if child.tag == "messages"
        ]
        if len(order_containers) > 1:
            raise ETradeBrokerTransportError("broker XML messages are ambiguous")
        allowed_containers.extend(order_containers)
    allowed_ids = {id(container) for container in allowed_containers}
    for node in root.iter():
        if node.tag in {"messageList", "messages"} and id(node) not in allowed_ids:
            raise ETradeBrokerTransportError(
                "broker XML message container is misplaced"
            )
        if node.tag == "Message" and not any(
            id(node) == id(child)
            for container in allowed_containers
            for child in list(container)
        ):
            raise ETradeBrokerTransportError("broker XML message is misplaced")
    result: list[BrokerMessage] = []
    for container in allowed_containers:
        messages = list(container)
        if not messages or any(message.tag != "Message" for message in messages):
            raise ETradeBrokerTransportError(
                "broker XML message container is malformed"
            )
        for message in messages:
            message_tags = [child.tag for child in list(message)]
            if len(message_tags) != 3 or set(message_tags) != {
                "description",
                "code",
                "type",
            }:
                raise ETradeBrokerTransportError("broker XML message is malformed")
            description = _one_xml_text(message, "description")
            code_text = _one_xml_text(message, "code")
            if (
                not code_text.isascii()
                or not code_text.isdigit()
                or len(code_text) > 10
            ):
                raise ETradeBrokerTransportError(
                    "broker message code is invalid"
                )
            result.append(
                BrokerMessage(
                    code=int(code_text),
                    message_type=_one_xml_text(message, "type"),
                    description=description,
                )
            )
    return tuple(result)


def _validate_broker_message(description: Any, code: Any, message_type: Any) -> None:
    if (
        type(description) is not str
        or not description
        or len(description) > 4096
        or type(code) is not int
        or code < 0
        or code > 2_147_483_647
        or type(message_type) is not str
        or message_type not in {"WARNING", "INFO", "INFO_HOLD", "ERROR"}
    ):
        raise ETradeBrokerTransportError("broker message is invalid")


def _message_code(value: Any) -> int:
    if type(value) is int:
        normalized = value
    elif (
        type(value) is str
        and value
        and value.isascii()
        and value.isdigit()
        and len(value) <= 10
    ):
        normalized = int(value)
    else:
        raise ETradeBrokerTransportError("broker message code is invalid")
    if not 0 <= normalized <= 2_147_483_647:
        raise ETradeBrokerTransportError("broker message code is invalid")
    return normalized


def _messages_allow_ack(
    operation: str, messages: tuple[BrokerMessage, ...]
) -> bool:
    return (
        operation.endswith("PLACE")
        and bool(messages)
        and all(
            message.message_type == "WARNING" and message.code == 1026
            for message in messages
        )
    )


def _one_xml_child(parent: ET.Element, tag: str) -> ET.Element:
    matches = [child for child in list(parent) if child.tag == tag]
    if len(matches) != 1:
        raise ETradeBrokerTransportError("broker XML response field is ambiguous")
    return matches[0]


def _one_xml_text(parent: ET.Element, tag: str) -> str:
    child = _one_xml_child(parent, tag)
    if list(child) or type(child.text) is not str:
        raise ETradeBrokerTransportError("broker XML response scalar is invalid")
    return child.text.strip()


def _one_xml_identifier(parent: ET.Element, tag: str) -> str:
    children = list(parent)
    tags = [child.tag for child in children]
    if (
        tags.count(tag) != 1
        or tags.count("cashMargin") > 1
        or set(tags) - {tag, "cashMargin"}
    ):
        raise ETradeBrokerTransportError(
            "broker XML response identifier is ambiguous"
        )
    if "cashMargin" in tags:
        cash_margin = _one_xml_text(parent, "cashMargin")
        if cash_margin not in {"CASH", "MARGIN", "INVALID"}:
            raise ETradeBrokerTransportError(
                "broker XML cash margin is invalid"
            )
    return _one_xml_text(parent, tag)


def _xml_order_status(root: ET.Element) -> str | None:
    orders = [child for child in list(root) if child.tag == "Order"]
    if not orders:
        return None
    if len(orders) != 1:
        raise ETradeBrokerTransportError("broker XML response order is ambiguous")
    statuses = [child for child in list(orders[0]) if child.tag == "status"]
    if not statuses:
        return None
    if len(statuses) != 1 or list(statuses[0]) or type(statuses[0].text) is not str:
        raise ETradeBrokerTransportError("broker XML response status is invalid")
    return _broker_status(statuses[0].text.strip())


def _broker_status(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.upper()
        or not value.replace("_", "").isascii()
        or not value.replace("_", "").isalpha()
    ):
        raise ETradeBrokerTransportError("broker status is invalid")
    return value


def _unknown_reply(
    request: BoundBrokerRequest,
    reason: str,
    *,
    http_status: int | None = None,
    broker_status: str | None = None,
    broker_order_id: str | None = None,
    preview_id: str | None = None,
    broker_messages: tuple[BrokerMessage, ...] = (),
    raw_response_digest: str | None = None,
    observed_at: datetime | None = None,
) -> BrokerReply:
    return BrokerReply(
        request=request,
        disposition="UNKNOWN",
        http_status=http_status,
        broker_status=broker_status,
        client_order_id=request.client_order_id,
        echoed_client_order_id=None,
        broker_order_id=broker_order_id,
        preview_id=request.preview_id if preview_id is None else preview_id,
        broker_messages=broker_messages,
        raw_response_digest=raw_response_digest,
        observed_at=observed_at or datetime.now(timezone.utc),
        unknown_reason=reason,
    )


def _identifier(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise ETradeBrokerTransportError(f"{label} is invalid")
    return value


def _optional_identifier(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, label)


def _broker_numeric_id(value: Any, label: str) -> str:
    if type(value) is int:
        if value <= 0:
            raise ETradeBrokerTransportError(f"{label} is invalid")
        normalized = str(value)
    elif type(value) is str:
        if (
            not value
            or len(value) > 19
            or not value.isascii()
            or not value.isdigit()
            or value[0] == "0"
        ):
            raise ETradeBrokerTransportError(f"{label} is invalid")
        normalized = value
    else:
        raise ETradeBrokerTransportError(f"{label} is invalid")
    if len(normalized) > 19:
        raise ETradeBrokerTransportError(f"{label} is invalid")
    if int(normalized) > _MAX_BROKER_INT64:
        raise ETradeBrokerTransportError(f"{label} is invalid")
    return normalized


def _broker_numeric_text(value: Any, label: str) -> str:
    if type(value) is not str:
        raise ETradeBrokerTransportError(f"{label} is invalid")
    return _broker_numeric_id(value, label)


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ETradeBrokerTransportError(f"{label} is invalid")
    return value
