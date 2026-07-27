from __future__ import annotations

from pathlib import Path


UNIT = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "etrade-read-only-dashboard.service.example"
)


def test_read_only_dashboard_unit_is_inert_and_hardened() -> None:
    content = UNIT.read_text(encoding="utf-8")

    assert "\n[Install]\n" not in content
    assert (
        "ConditionPathExists="
        "/etc/etrade/read-only-dashboard.approved"
    ) in content
    assert (
        "etrade-read-only-dashboard serve "
        "--config /var/lib/etrade/runtime-config.json "
        "--host 127.0.0.1 --port 8765 --secure-cookie"
    ) in content
    for directive in (
        "UMask=0077",
        "NoNewPrivileges=true",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "ProtectSystem=strict",
        "ProtectHome=true",
        "ReadOnlyPaths=/var/lib/etrade/runtime-config.json",
        "ReadOnlyPaths=/var/lib/etrade/runtime/artifacts",
        "ReadOnlyPaths=/var/lib/etrade/runtime/model",
        "InaccessiblePaths=/var/lib/etrade/runtime/state",
        "InaccessiblePaths=/var/lib/etrade/runtime/cache",
        "InaccessiblePaths=/var/lib/etrade/runtime/logs",
        "InaccessiblePaths=/var/lib/etrade/runtime/execution",
        "InaccessiblePaths=/var/lib/etrade/runtime/data",
        "PrivateTmp=true",
        "PrivateDevices=true",
        "ProtectProc=invisible",
        "RestrictNamespaces=true",
        "MemoryDenyWriteExecute=true",
        "SystemCallFilter=@system-service",
        "IPAddressDeny=any",
        "IPAddressAllow=localhost",
        "SocketBindAllow=ipv4:tcp:8765",
    ):
        assert directive in content
    assert "ReadWritePaths=" not in content
    assert "ETRADE_" not in content
    assert "--trade" not in content
    assert "--environment" not in content
