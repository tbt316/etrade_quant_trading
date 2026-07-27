"""Test-session safety boundaries."""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any


class OutboundNetworkBlocked(RuntimeError):
    """A deterministic test attempted to reach a non-loopback host."""


def _host_is_loopback(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    text = str(host).strip().lower()
    if text == "localhost" or text.endswith(".localhost"):
        return True
    if "%" in text:
        text = text.split("%", 1)[0]
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def _require_loopback(host: Any) -> None:
    if not _host_is_loopback(host):
        raise OutboundNetworkBlocked(
            f"outbound network is disabled during tests: host={host!r}"
        )


def _install_network_guard() -> None:
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyaddr = socket.gethostbyaddr
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex
    original_getnameinfo = socket.getnameinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendmsg = getattr(socket.socket, "sendmsg", None)
    original_sendto = socket.socket.sendto

    def guarded_getaddrinfo(
        host: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        _require_loopback(host)
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(
        connection: socket.socket,
        address: Any,
    ) -> Any:
        if connection.family in {socket.AF_INET, socket.AF_INET6}:
            _require_loopback(address[0])
        return original_connect(connection, address)

    def guarded_connect_ex(
        connection: socket.socket,
        address: Any,
    ) -> int:
        if connection.family in {socket.AF_INET, socket.AF_INET6}:
            _require_loopback(address[0])
        return original_connect_ex(connection, address)

    def guarded_gethostbyaddr(host: Any) -> tuple[str, list[str], list[str]]:
        _require_loopback(host)
        return original_gethostbyaddr(host)

    def guarded_gethostbyname(host: Any) -> str:
        _require_loopback(host)
        return original_gethostbyname(host)

    def guarded_gethostbyname_ex(
        host: Any,
    ) -> tuple[str, list[str], list[str]]:
        _require_loopback(host)
        return original_gethostbyname_ex(host)

    def guarded_getnameinfo(
        address: Any,
        flags: int,
    ) -> tuple[str, str]:
        _require_loopback(address[0])
        return original_getnameinfo(address, flags)

    def guarded_sendto(
        connection: socket.socket,
        data: bytes,
        *args: Any,
    ) -> int:
        address = args[-1]
        if connection.family in {socket.AF_INET, socket.AF_INET6}:
            _require_loopback(address[0])
        return original_sendto(connection, data, *args)

    def guarded_sendmsg(
        connection: socket.socket,
        buffers: Any,
        *args: Any,
    ) -> int:
        if (
            connection.family in {socket.AF_INET, socket.AF_INET6}
            and len(args) >= 3
        ):
            _require_loopback(args[2][0])
        assert original_sendmsg is not None
        return original_sendmsg(connection, buffers, *args)

    socket.getaddrinfo = guarded_getaddrinfo
    socket.gethostbyaddr = guarded_gethostbyaddr
    socket.gethostbyname = guarded_gethostbyname
    socket.gethostbyname_ex = guarded_gethostbyname_ex
    socket.getnameinfo = guarded_getnameinfo
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    if original_sendmsg is not None:
        socket.socket.sendmsg = guarded_sendmsg
    socket.socket.sendto = guarded_sendto


if os.environ.get("ETRADE_TEST_NETWORK") == "deny":
    _install_network_guard()
