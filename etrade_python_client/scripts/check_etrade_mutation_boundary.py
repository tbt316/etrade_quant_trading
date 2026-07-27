#!/usr/bin/env python3
"""Fail closed when production code bypasses the E*TRADE mutation gateway.

The scan is intentionally lexical and conservative. It examines every tracked
Python source beneath the application root, including tracked scratch and
backup files. Tests and this checker are the only exclusions.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


APPLICATION_ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_PATH = "live_trading/etrade_broker_transport.py"
GATEWAY_PATH = "live_trading/etrade_order_gateway.py"
READER_PATH = "live_trading/etrade_broker_reader.py"
RUNTIME_SAFETY_PATH = "live_trading/runtime_safety.py"
CHECKER_PATH = "scripts/check_etrade_mutation_boundary.py"
RUNTIME_SAFETY_MODULE = "live_trading.runtime_safety"
TRANSPORT_MODULE = "live_trading.etrade_broker_transport"
CENTRAL_REJECTOR = "live_trading.runtime_safety.reject_legacy_execution"

TRANSPORT_MUTATION_METHODS = frozenset(
    {"preview", "place", "preview_change", "place_change"}
)
TRANSPORT_INTERNAL_CAPABILITIES = frozenset(
    {
        "_PreparedExchange",
        "_build_request",
        "_exchange_worker",
        "_execute",
        "_isolated_mutation_exchange",
        "_new_pinned_adapter",
        "_prepare_oauth_request",
        "_reconstruct_prepared_request",
        "_run_isolated_exchange",
        "_serialized_prepared_request",
    }
)
GATEWAY_MUTATION_CALLERS = {
    "preview": "live_trading.etrade_order_gateway.EtradeOrderGateway.submit_opening",
    "place": "live_trading.etrade_order_gateway.EtradeOrderGateway.submit_opening",
    "preview_change": (
        "live_trading.etrade_order_gateway.EtradeOrderGateway.reprice_opening"
    ),
    "place_change": (
        "live_trading.etrade_order_gateway.EtradeOrderGateway.reprice_opening"
    ),
}
HTTP_MUTATION_METHODS = frozenset(
    {"post", "put", "patch", "delete", "request", "send"}
)
HTTP_CLIENT_MODULE_PREFIXES = (
    "aiohttp",
    "httpx",
    "rauth",
    "requests",
    "urllib3",
)
NON_HTTP_MUTATION_CALLS = frozenset(
    {
        ("ai_agents/gemini_analyst.py", "self.client.files.delete"),
    }
)
# Exact path/name exceptions for attributes that share HTTP mutation names but
# are demonstrably data/read-only APIs. Keep this list narrow: a new bound
# mutation alias must fail closed instead of being inferred safe.
NON_HTTP_MUTATION_REFERENCES = frozenset(
    {
        ("accounts/accounts.py", "response.request"),
        ("accounts/accounts_bo.py", "response.request"),
        ("ai_agents/report_scraper.py", "route.request"),
        ("ai_agents/report_scraper.py", "self.page.context.request"),
        ("ai_agents/report_scraper.py", "self.page.request"),
        ("backtesting/experiment_manager.py", "args.delete"),
        ("market/market.py", "response.request"),
        ("market/market_bo.py", "response.request"),
        ("order/order_bo.py", "response.request"),
        ("scratch/fslr_30dte_iv.py", "fig.patch"),
        ("scratch/mu_30dte_iv.py", "fig.patch"),
        ("scratch/rklb_20delta_call_iv.py", "fig.patch"),
        ("scratch/run_comparison_2019_2025.py", "fig.patch"),
    }
)
VENDOR_MUTATION_CAPABILITIES = frozenset({"ETradeOrder"})
VENDOR_MUTATION_METHODS = frozenset(
    {
        "change_preview_equity_order",
        "change_preview_option_order",
        "perform_request",
        "place_changed_equity_order",
        "place_changed_option_order",
        "place_equity_order",
        "place_option_order",
        "preview_equity_order",
        "preview_option_order",
    }
)
LEGACY_MUTATION_METHODS = frozenset(
    {
        "cancel_all_order",
        "cancel_order",
        "change_order_limit",
        "check_short_availability",
        "filter_order",
        "options_selection",
        "place_option_order",
        "place_order",
        "place_order_old",
        "preview_order",
        "preview_order_menu",
        "preview_order_old",
        "previous_order",
        "refresh_order_limit",
        "refresh_order_limit_old",
        "release_margin",
        "user_select_order",
        "view_orders",
        "wait_and_adjust_until_filled",
    }
)
PROTECTED_TRANSPORT_NAMES = frozenset(
    {
        "ETradeBrokerTransport",
        "etrade_broker_transport",
        "_transport",
        *TRANSPORT_MUTATION_METHODS,
        *TRANSPORT_INTERNAL_CAPABILITIES,
    }
)
PROTECTED_MUTATION_NAMES = frozenset(
    {
        *PROTECTED_TRANSPORT_NAMES,
        *VENDOR_MUTATION_CAPABILITIES,
        *VENDOR_MUTATION_METHODS,
        *LEGACY_MUTATION_METHODS,
    }
)
ALLOWED_TRANSPORT_IMPORTS = {
    GATEWAY_PATH: frozenset(
        {
            "BrokerReply",
            "ETradeBrokerTransport",
            "ETradeBrokerTransportError",
            "SelectedBrokerAccount",
        }
    ),
    READER_PATH: frozenset(
        {
            "SelectedBrokerAccount",
            "_ExchangeResult",
            "_isolated_get_exchange",
        }
    ),
}
MUTATION_LITERAL = re.compile(
    r"(?ix)"
    r"(?:/orders/(?:preview|place|cancel)(?:\.json)?(?:\b|/))"
    r"|(?:/change/(?:preview|place)(?:\.json)?(?:\b|/))"
    r"|(?:<\s*/?\s*(?:preview|place|cancel|change)orderrequest\b)"
)


@dataclass(frozen=True)
class TombstoneSpec:
    path: str
    reason: str


REQUIRED_TOMBSTONES: Mapping[str, TombstoneSpec] = {
    "order.order.Order.__init__": TombstoneSpec(
        "order/order.py", "order.order.Order.__init__"
    ),
    **{
        f"order.order_bo.Order.{name}": TombstoneSpec(
            "order/order_bo.py", f"order.order_bo.Order.{name}"
        )
        for name in (
            "preview_order",
            "preview_order_old",
            "place_order",
            "place_order_old",
            "change_order_limit",
            "wait_and_adjust_until_filled",
            "previous_order",
            "user_select_order",
            "preview_order_menu",
            "cancel_order",
            "place_option_order",
            "cancel_all_order",
            "refresh_order_limit_old",
            "refresh_order_limit",
            "options_selection",
            "view_orders",
            "filter_order",
        )
    },
    "core_api.stock_trade_class.LiveTradeAgent.check_short_availability": (
        TombstoneSpec(
            "core_api/stock_trade_class.py",
            "core_api.stock_trade_class.LiveTradeAgent.check_short_availability",
        )
    ),
    "core_api.stock_trade_class.LiveTradeAgent.place_order": TombstoneSpec(
        "core_api/stock_trade_class.py",
        "core_api.stock_trade_class.LiveTradeAgent.place_order",
    ),
    "live_trading.etrade_put_credit_spread.release_margin": TombstoneSpec(
        "live_trading/etrade_put_credit_spread.py",
        "live_trading.etrade_put_credit_spread.release_margin",
    ),
    "live_trading.etrade_cover_call_new.enqueue_manual_trade_request": TombstoneSpec(
        "live_trading/etrade_cover_call_new.py",
        "legacy dashboard manual-trade queue",
    ),
    "live_trading.etrade_cover_call_new._execute_neutralize_leg": TombstoneSpec(
        "live_trading/etrade_cover_call_new.py",
        "legacy dashboard neutralize worker",
    ),
    "live_trading.etrade_cover_call_new.release_margin": TombstoneSpec(
        "live_trading/etrade_cover_call_new.py",
        "legacy automatic margin release",
    ),
    "live_trading.etrade_cover_call_new.submit_manual_open_fast_async": (
        TombstoneSpec(
            "live_trading/etrade_cover_call_new.py",
            "legacy dashboard manual-open scheduler",
        )
    ),
    "live_trading.etrade_cover_call_new._manual_open_fast_worker": TombstoneSpec(
        "live_trading/etrade_cover_call_new.py",
        "legacy dashboard manual-open worker",
    ),
    "live_trading.etrade_cover_call_new.submit_order_async": TombstoneSpec(
        "live_trading/etrade_cover_call_new.py",
        "legacy dashboard order scheduler",
    ),
    "live_trading.etrade_cover_call_new._order_worker": TombstoneSpec(
        "live_trading/etrade_cover_call_new.py",
        "legacy dashboard order worker",
    ),
    "live_trading.etrade_cover_call_new.monitor_and_nudge_stale_orders": (
        TombstoneSpec(
            "live_trading/etrade_cover_call_new.py",
            "legacy stale-order nudge worker",
        )
    ),
    "scratch.check_orders.get_today_orders": TombstoneSpec(
        "scratch/check_orders.py", "scratch.check_orders.get_today_orders"
    ),
}


@dataclass(frozen=True, order=True)
class Diagnostic:
    path: str
    line: int
    column: int
    code: str
    message: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.column}: "
            f"{self.code}: {self.message}"
        )


class BoundaryConfigurationError(RuntimeError):
    """The checker could not deterministically identify tracked sources."""


def _module_name(relative_path: str) -> str:
    path = PurePosixPath(relative_path).with_suffix("")
    parts = list(path.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _contains_dotted_name(node: ast.AST, target: str) -> bool:
    return any(
        isinstance(candidate, ast.Attribute)
        and _dotted_name(candidate) == target
        for candidate in ast.walk(node)
    )


def _resolved_import_module(
    relative_path: str, module_name: str, node: ast.ImportFrom
) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = module_name.split(".") if module_name else []
    if PurePosixPath(relative_path).name != "__init__.py" and package_parts:
        package_parts.pop()
    levels_to_ascend = node.level - 1
    if levels_to_ascend > len(package_parts):
        return ""
    if levels_to_ascend:
        package_parts = package_parts[:-levels_to_ascend]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


class _ModuleBindingVisitor(ast.NodeVisitor):
    """Collect module bindings without descending into local/class scopes."""

    def __init__(self) -> None:
        self.bindings: list[tuple[str, ast.AST]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bindings.append((node.name, node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bindings.append((node.name, node))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bindings.append((node.name, node))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bindings.append(
                (alias.asname or alias.name.split(".", 1)[0], node)
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bindings.append((alias.asname or alias.name, node))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bindings.append((node.id, node))


class _SourceVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        relative_path: str,
        tree: ast.Module,
        required_tombstones: Mapping[str, TombstoneSpec],
    ) -> None:
        self.relative_path = relative_path
        self.module_name = _module_name(relative_path)
        self.tree = tree
        self.required_tombstones = required_tombstones
        self.diagnostics: list[Diagnostic] = []
        self.definitions: dict[str, list[ast.AST]] = {}
        self.scope: list[str] = []
        self.parents: dict[ast.AST, ast.AST] = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        self.has_exact_reject_binding = self._has_exact_reject_binding()
        self._reported_gateway_attributes: set[tuple[int, int]] = set()

    @property
    def qualified_scope(self) -> str:
        pieces = [self.module_name, *self.scope]
        return ".".join(piece for piece in pieces if piece)

    def add(
        self,
        node: ast.AST,
        code: str,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                self.relative_path,
                line if line is not None else getattr(node, "lineno", 1),
                (
                    column
                    if column is not None
                    else getattr(node, "col_offset", 0) + 1
                ),
                code,
                message,
            )
        )

    def _has_exact_reject_binding(self) -> bool:
        exact_import_lines: list[int] = []
        for statement in self.tree.body:
            if isinstance(statement, ast.ImportFrom):
                resolved_module = _resolved_import_module(
                    self.relative_path, self.module_name, statement
                )
                for alias in statement.names:
                    if (
                        resolved_module == RUNTIME_SAFETY_MODULE
                        and alias.name == "reject_legacy_execution"
                        and alias.asname is None
                    ):
                        exact_import_lines.append(statement.lineno)
        binding_visitor = _ModuleBindingVisitor()
        binding_visitor.visit(self.tree)
        if len(exact_import_lines) != 1:
            return False
        exact_line = exact_import_lines[0]
        later_or_same_conflicts = [
            (name, node)
            for name, node in binding_visitor.bindings
            if name in {"reject_legacy_execution", "*"}
            and getattr(node, "lineno", 0) >= exact_line
        ]
        return (
            len(later_or_same_conflicts) == 1
            and later_or_same_conflicts[0][0] == "reject_legacy_execution"
            and getattr(later_or_same_conflicts[0][1], "lineno", 0)
            == exact_line
        )

    def _visit_definition_expressions(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.args.vararg and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def _validate_required_tombstone(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qualified_name: str,
        spec: TombstoneSpec,
    ) -> None:
        owner = self.parents.get(node)
        owner_is_direct = isinstance(owner, ast.Module) or (
            isinstance(owner, ast.ClassDef)
            and isinstance(self.parents.get(owner), ast.Module)
        )
        if not owner_is_direct:
            self.add(
                node,
                "TOMBSTONE_LOCATION",
                (
                    f"{qualified_name} must be defined directly at module "
                    "or top-level class scope"
                ),
            )
        if not self.has_exact_reject_binding:
            self.add(
                node,
                "TOMBSTONE_IMPORT",
                (
                    "required tombstones need one unshadowed module-level "
                    f"`from {RUNTIME_SAFETY_MODULE} import "
                    "reject_legacy_execution`"
                ),
            )
        if node.decorator_list:
            self.add(
                node,
                "TOMBSTONE_DECORATOR",
                f"{qualified_name} must not have decorators",
            )
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        shadows_rejector = any(
            argument.arg == "reject_legacy_execution"
            for argument in arguments
        ) or (
            node.args.vararg is not None
            and node.args.vararg.arg == "reject_legacy_execution"
        ) or (
            node.args.kwarg is not None
            and node.args.kwarg.arg == "reject_legacy_execution"
        )
        if shadows_rejector:
            self.add(
                node,
                "TOMBSTONE_SIGNATURE",
                (
                    f"{qualified_name} must not shadow "
                    "reject_legacy_execution with a parameter"
                ),
            )
        body = list(node.body)
        if body and _is_docstring(body[0]):
            body = body[1:]
        if len(body) != 1:
            self.add(
                node,
                "TOMBSTONE_BODY",
                (
                    f"{qualified_name} must contain only an optional docstring "
                    "and the unconditional reject call"
                ),
            )
            return
        statement = body[0]
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
        ):
            self.add(
                statement,
                "TOMBSTONE_BODY",
                f"{qualified_name} does not unconditionally reject",
            )
            return
        call = statement.value
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id == "reject_legacy_execution"
            and len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            self.add(
                call,
                "TOMBSTONE_BODY",
                (
                    f"{qualified_name} must make the exact unqualified "
                    "reject_legacy_execution(reason) call"
                ),
            )
            return
        if call.args[0].value != spec.reason:
            self.add(
                call.args[0],
                "TOMBSTONE_REASON",
                (
                    f"{qualified_name} rejection reason must be "
                    f"{spec.reason!r}"
                ),
            )

    def _validate_central_rejector(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        owner = self.parents.get(node)
        binding_visitor = _ModuleBindingVisitor()
        binding_visitor.visit(self.tree)
        rejector_bindings = [
            binding
            for binding in binding_visitor.bindings
            if binding[0] == "reject_legacy_execution"
        ]
        if (
            not isinstance(node, ast.FunctionDef)
            or not isinstance(owner, ast.Module)
            or node.decorator_list
            or len(rejector_bindings) != 1
            or rejector_bindings[0][1] is not node
        ):
            self.add(
                node,
                "CENTRAL_REJECTOR_DEFINITION",
                (
                    "reject_legacy_execution must be one unshadowed, "
                    "undecorated module-level synchronous function"
                ),
            )
        expected = ast.parse(
            "def reject_legacy_execution(surface: str) -> NoReturn:\n"
            "    raise LegacyExecutionDisabled(\n"
            "        f\"legacy execution surface is quarantined and disabled: "
            "{surface}; \"\n"
            "        \"route broker mutations through ETradeOrderGateway\"\n"
            "    )\n"
        ).body[0]
        assert isinstance(expected, ast.FunctionDef)
        body = list(node.body)
        if body and _is_docstring(body[0]):
            body = body[1:]
        if (
            ast.dump(node.args, include_attributes=False)
            != ast.dump(expected.args, include_attributes=False)
            or (
                ast.dump(node.returns, include_attributes=False)
                if node.returns is not None
                else None
            )
            != ast.dump(expected.returns, include_attributes=False)
            or len(body) != 1
            or ast.dump(body[0], include_attributes=False)
            != ast.dump(expected.body[0], include_attributes=False)
        ):
            self.add(
                node,
                "CENTRAL_REJECTOR_BODY",
                (
                    "reject_legacy_execution must unconditionally raise the "
                    "fixed LegacyExecutionDisabled error"
                ),
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        qualified_name = ".".join(
            piece for piece in (self.qualified_scope, node.name) if piece
        )
        self.definitions.setdefault(qualified_name, []).append(node)
        spec = self.required_tombstones.get(qualified_name)
        if spec is not None:
            self._validate_required_tombstone(node, qualified_name, spec)
        if qualified_name == CENTRAL_REJECTOR:
            self._validate_central_rejector(node)
        if (
            self.relative_path == GATEWAY_PATH
            and self.qualified_scope
            == "live_trading.etrade_order_gateway.EtradeOrderGateway"
        ):
            is_property = any(
                _dotted_name(decorator) == "property"
                for decorator in node.decorator_list
            )
            if node.name == "transport" and is_property:
                self.add(
                    node,
                    "GATEWAY_TRANSPORT_EXPOSURE",
                    "the gateway must not expose a public transport property",
                )
            if not node.name.startswith("_") and any(
                isinstance(candidate, ast.Return)
                and candidate.value is not None
                and _contains_dotted_name(candidate.value, "self._transport")
                for candidate in ast.walk(node)
            ):
                self.add(
                    node,
                    "GATEWAY_TRANSPORT_EXPOSURE",
                    "a public gateway property must not return its transport",
                )
        self._visit_definition_expressions(node)
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        if self.relative_path == TRANSPORT_PATH:
            return
        for alias in node.names:
            if alias.name == TRANSPORT_MODULE:
                self.add(
                    node,
                    "TRANSPORT_IMPORT",
                    (
                        "the mutation transport module may only be imported "
                        "through the reviewed exact symbol allowlist"
                    ),
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _resolved_import_module(
            self.relative_path, self.module_name, node
        )
        if module == "pyetrade" or module.startswith("pyetrade."):
            for alias in node.names:
                if (
                    alias.name in VENDOR_MUTATION_CAPABILITIES
                    or alias.name == "*"
                ):
                    self.add(
                        node,
                        "VENDOR_MUTATION_CAPABILITY",
                        (
                            "direct pyetrade order capability imports are "
                            "forbidden"
                        ),
                    )
        if self.relative_path != TRANSPORT_PATH:
            for alias in node.names:
                if alias.name in TRANSPORT_INTERNAL_CAPABILITIES:
                    self.add(
                        node,
                        "TRANSPORT_CAPABILITY_IMPORT",
                        (
                            f"importing raw transport capability "
                            f"{alias.name!r} is forbidden"
                        ),
                    )
        if self.relative_path == TRANSPORT_PATH:
            return
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in HTTP_CLIENT_MODULE_PREFIXES
        ):
            for alias in node.names:
                if alias.name in HTTP_MUTATION_METHODS:
                    self.add(
                        node,
                        "RAW_HTTP_MUTATION_IMPORT",
                        (
                            f"importing HTTP mutation callable "
                            f"{alias.name!r} is forbidden"
                        ),
                    )
        imports_transport_module = module == TRANSPORT_MODULE
        imports_transport_from_package = (
            module == "live_trading"
            and any(
                alias.name == "etrade_broker_transport"
                for alias in node.names
            )
        )
        if imports_transport_from_package:
            self.add(
                node,
                "TRANSPORT_IMPORT",
                "importing the transport module object is forbidden",
            )
            return
        if not imports_transport_module:
            return
        allowed = ALLOWED_TRANSPORT_IMPORTS.get(self.relative_path, frozenset())
        for alias in node.names:
            if alias.name not in allowed or alias.asname is not None:
                self.add(
                    node,
                    "TRANSPORT_IMPORT",
                    (
                        f"{alias.name!r} is not an exact reviewed transport "
                        f"import for {self.relative_path}"
                    ),
                )

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _dotted_name(node.func)
        final_name = call_name.rsplit(".", 1)[-1] if call_name else ""

        if self.relative_path != TRANSPORT_PATH:
            if final_name in HTTP_MUTATION_METHODS:
                allowed_non_http_call = (
                    self.relative_path,
                    call_name or "",
                ) in NON_HTTP_MUTATION_CALLS
                if not allowed_non_http_call:
                    self.add(
                        node,
                        "RAW_HTTP_MUTATION",
                        (
                            f"direct {final_name}() is forbidden outside "
                            f"{TRANSPORT_PATH}"
                        ),
                    )

            if final_name in TRANSPORT_MUTATION_METHODS:
                expected_scope = GATEWAY_MUTATION_CALLERS[final_name]
                expected_call = f"self._transport.{final_name}"
                if not (
                    self.relative_path == GATEWAY_PATH
                    and self.qualified_scope == expected_scope
                    and call_name == expected_call
                ):
                    self.add(
                        node,
                        "TRANSPORT_MUTATION_CALL",
                        (
                            f"{final_name}() is allowed only as "
                            f"{expected_call} in {expected_scope}"
                        ),
                    )

            if final_name in LEGACY_MUTATION_METHODS:
                self.add(
                    node,
                    "LEGACY_MUTATION_CALL",
                    f"legacy mutation call {final_name}() is forbidden",
                )

            if final_name in VENDOR_MUTATION_METHODS:
                self.add(
                    node,
                    "VENDOR_MUTATION_CALL",
                    f"direct pyetrade mutation call {final_name}() is forbidden",
                )

            if final_name in TRANSPORT_INTERNAL_CAPABILITIES:
                self.add(
                    node,
                    "TRANSPORT_CAPABILITY_CALL",
                    (
                        f"raw transport capability {final_name}() may only "
                        f"run inside {TRANSPORT_PATH}"
                    ),
                )

            if call_name == "ETradeBrokerTransport":
                self.add(
                    node,
                    "TRANSPORT_CONSTRUCTION",
                    "production code must not construct the mutation transport here",
                )

            self._check_reflection_call(node, call_name)

        self.generic_visit(node)

    def _check_reflection_call(
        self, node: ast.Call, call_name: str | None
    ) -> None:
        reflection_name = (
            call_name.rsplit(".", 1)[-1] if call_name is not None else ""
        )
        string_arguments = [
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        ]
        reflected_attribute_arguments: list[ast.AST] = []
        if reflection_name in {"getattr", "setattr", "delattr", "hasattr"}:
            reflected_attribute_arguments = list(node.args[1:2])
        elif reflection_name == "__getattribute__":
            reflected_attribute_arguments = list(
                node.args[1:2] if len(node.args) >= 2 else node.args[:1]
            )
        elif reflection_name in {"attrgetter", "methodcaller"}:
            reflected_attribute_arguments = list(node.args[:1])
        reflected_attribute_names = [
            argument.value
            for argument in reflected_attribute_arguments
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        ]
        if reflection_name in {"__import__", "import_module"} and any(
            value == TRANSPORT_MODULE for value in string_arguments
        ):
            self.add(
                node,
                "TRANSPORT_REFLECTION",
                "dynamic import of the mutation transport is forbidden",
            )
        if reflection_name in {
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
            "__getattribute__",
            "attrgetter",
            "methodcaller",
        }:
            target = _dotted_name(node.args[0]) if node.args else None
            if any(
                value.lower() in HTTP_MUTATION_METHODS
                for value in reflected_attribute_names
            ):
                self.add(
                    node,
                    "RAW_HTTP_MUTATION_REFLECTION",
                    "reflective HTTP mutation access is forbidden",
                )
            if (
                target is not None and "transport" in target.lower()
            ) or any(
                value in PROTECTED_TRANSPORT_NAMES
                for value in reflected_attribute_names
            ):
                self.add(
                    node,
                    "TRANSPORT_REFLECTION",
                    "reflective mutation-transport access is forbidden",
                )
            if any(
                value in LEGACY_MUTATION_METHODS
                for value in reflected_attribute_names
            ):
                self.add(
                    node,
                    "LEGACY_MUTATION_REFLECTION",
                    "reflective legacy mutation access is forbidden",
                )
            if any(
                value in VENDOR_MUTATION_CAPABILITIES
                or value in VENDOR_MUTATION_METHODS
                for value in reflected_attribute_names
            ):
                self.add(
                    node,
                    "VENDOR_MUTATION_REFLECTION",
                    "reflective pyetrade mutation access is forbidden",
                )
        if reflection_name in {"eval", "exec"} and any(
            any(
                protected in value
                for protected in (
                    TRANSPORT_MODULE,
                    *PROTECTED_MUTATION_NAMES,
                )
            )
            for value in string_arguments
        ):
            self.add(
                node,
                "MUTATION_REFLECTION",
                "dynamic execution referencing a mutation surface is forbidden",
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.relative_path != TRANSPORT_PATH:
            dotted = _dotted_name(node)
            if (
                self.relative_path == GATEWAY_PATH
                and dotted is not None
                and (
                    dotted == "self.transport"
                    or dotted.startswith("self.transport.")
                )
            ):
                key = (
                    getattr(node, "lineno", 1),
                    getattr(node, "col_offset", 0),
                )
                if key not in self._reported_gateway_attributes:
                    self._reported_gateway_attributes.add(key)
                    self.add(
                        node,
                        "GATEWAY_TRANSPORT_EXPOSURE",
                        "gateway transport state must remain private as self._transport",
                    )
            parent = self.parents.get(node)
            is_direct_call = (
                isinstance(parent, ast.Call) and parent.func is node
            )
            if node.attr in TRANSPORT_MUTATION_METHODS and not is_direct_call:
                self.add(
                    node,
                    "TRANSPORT_MUTATION_REFERENCE",
                    f"reference to transport mutation {node.attr!r} is forbidden",
                )
            if (
                node.attr in HTTP_MUTATION_METHODS
                and not is_direct_call
                and (
                    self.relative_path,
                    dotted or "",
                )
                not in NON_HTTP_MUTATION_REFERENCES
            ):
                self.add(
                    node,
                    "RAW_HTTP_MUTATION_REFERENCE",
                    f"reference to HTTP mutation {node.attr!r} is forbidden",
                )
            if node.attr in LEGACY_MUTATION_METHODS and not is_direct_call:
                self.add(
                    node,
                    "LEGACY_MUTATION_REFERENCE",
                    f"reference to legacy mutation {node.attr!r} is forbidden",
                )
            if (
                node.attr in VENDOR_MUTATION_METHODS
                and not is_direct_call
            ):
                self.add(
                    node,
                    "VENDOR_MUTATION_REFERENCE",
                    f"reference to pyetrade mutation {node.attr!r} is forbidden",
                )
            if (
                node.attr in TRANSPORT_INTERNAL_CAPABILITIES
                and not is_direct_call
            ):
                self.add(
                    node,
                    "TRANSPORT_CAPABILITY_REFERENCE",
                    (
                        f"reference to raw transport capability "
                        f"{node.attr!r} is forbidden"
                    ),
                )
            if (
                node.attr == "ETradeBrokerTransport"
                and self.relative_path not in {GATEWAY_PATH, READER_PATH}
            ):
                self.add(
                    node,
                    "TRANSPORT_REFERENCE",
                    "mutation transport class reference is forbidden here",
                )
            if node.attr in VENDOR_MUTATION_CAPABILITIES:
                self.add(
                    node,
                    "VENDOR_MUTATION_CAPABILITY",
                    "direct pyetrade order capability access is forbidden",
                )
            if (
                node.attr == "_transport"
                and self.relative_path
                not in {
                    GATEWAY_PATH,
                    "live_trading/regime_market_data_gateway.py",
                }
            ):
                self.add(
                    node,
                    "GATEWAY_TRANSPORT_ACCESS",
                    "private gateway transport access is forbidden",
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            self.relative_path not in {
                TRANSPORT_PATH,
                GATEWAY_PATH,
                READER_PATH,
            }
            and node.id == "ETradeBrokerTransport"
        ):
            self.add(
                node,
                "TRANSPORT_REFERENCE",
                "mutation transport class reference is forbidden here",
            )
        if node.id in VENDOR_MUTATION_CAPABILITIES:
            self.add(
                node,
                "VENDOR_MUTATION_CAPABILITY",
                "direct pyetrade order capability reference is forbidden",
            )
        if (
            self.relative_path != TRANSPORT_PATH
            and node.id in TRANSPORT_INTERNAL_CAPABILITIES
        ):
            self.add(
                node,
                "TRANSPORT_CAPABILITY_REFERENCE",
                (
                    f"reference to raw transport capability "
                    f"{node.id!r} is forbidden"
                ),
            )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self.relative_path != TRANSPORT_PATH:
            key = node.slice
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value
                in {
                    "_transport",
                    *TRANSPORT_INTERNAL_CAPABILITIES,
                }
            ):
                self.add(
                    node,
                    "TRANSPORT_CAPABILITY_LOOKUP",
                    "dictionary-style transport capability access is forbidden",
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if self.relative_path == TRANSPORT_PATH:
            return
        value = node.value
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="ignore")
        elif isinstance(value, str):
            text = value
        else:
            return
        if MUTATION_LITERAL.search(text):
            self.add(
                node,
                "ETRADE_MUTATION_LITERAL",
                (
                    "E*TRADE mutation URL/XML literal is owned exclusively by "
                    f"{TRANSPORT_PATH}"
                ),
            )


def tracked_production_python_files(root: Path) -> tuple[str, ...]:
    root = root.resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BoundaryConfigurationError(
            f"git ls-files failed for {root}: {error or 'unknown error'}"
        )
    relative_paths: list[str] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative_path = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BoundaryConfigurationError(
                "tracked Python path is not valid UTF-8"
            ) from exc
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise BoundaryConfigurationError(
                f"tracked Python path escapes the application root: {relative_path}"
            )
        if pure_path.parts and pure_path.parts[0] == "tests":
            continue
        if relative_path == CHECKER_PATH:
            continue
        relative_paths.append(relative_path)
    return tuple(sorted(set(relative_paths)))


def scan_paths(
    root: Path,
    relative_paths: Iterable[str],
    *,
    required_tombstones: Mapping[str, TombstoneSpec] | None = None,
) -> list[Diagnostic]:
    root = root.resolve()
    selected_paths = tuple(sorted(set(relative_paths)))
    required = (
        REQUIRED_TOMBSTONES
        if required_tombstones is None
        else required_tombstones
    )
    diagnostics: list[Diagnostic] = []
    definitions: dict[str, list[ast.AST]] = {}

    for relative_path in selected_paths:
        path = root / PurePosixPath(relative_path)
        if not path.exists():
            diagnostics.append(
                Diagnostic(
                    relative_path,
                    1,
                    1,
                    "TRACKED_SOURCE_MISSING",
                    "tracked production Python source is missing",
                )
            )
            continue
        if path.is_symlink():
            diagnostics.append(
                Diagnostic(
                    relative_path,
                    1,
                    1,
                    "TRACKED_SOURCE_SYMLINK",
                    "production Python sources must not be symbolic links",
                )
            )
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            diagnostics.append(
                Diagnostic(
                    relative_path,
                    1,
                    1,
                    "SOURCE_READ_ERROR",
                    f"could not read tracked source: {type(exc).__name__}",
                )
            )
            continue
        try:
            tree = ast.parse(source, filename=relative_path)
        except SyntaxError as exc:
            diagnostics.append(
                Diagnostic(
                    relative_path,
                    exc.lineno or 1,
                    exc.offset or 1,
                    "SYNTAX_ERROR",
                    exc.msg,
                )
            )
            continue
        visitor = _SourceVisitor(
            relative_path=relative_path,
            tree=tree,
            required_tombstones=required,
        )
        visitor.visit(tree)
        diagnostics.extend(visitor.diagnostics)
        for qualified_name, nodes in visitor.definitions.items():
            definitions.setdefault(qualified_name, []).extend(nodes)

    for qualified_name, spec in required.items():
        nodes = definitions.get(qualified_name, [])
        if not nodes:
            diagnostics.append(
                Diagnostic(
                    spec.path,
                    1,
                    1,
                    "TOMBSTONE_MISSING",
                    f"required tombstone {qualified_name} is missing",
                )
            )
        elif len(nodes) > 1:
            for duplicate in nodes[1:]:
                diagnostics.append(
                    Diagnostic(
                        spec.path,
                        getattr(duplicate, "lineno", 1),
                        getattr(duplicate, "col_offset", 0) + 1,
                        "TOMBSTONE_DUPLICATE",
                        f"required tombstone {qualified_name} is redefined",
                )
            )

    rejector_nodes = definitions.get(CENTRAL_REJECTOR, [])
    if (
        RUNTIME_SAFETY_PATH in selected_paths
        or required_tombstones is None
    ):
        if not rejector_nodes:
            diagnostics.append(
                Diagnostic(
                    RUNTIME_SAFETY_PATH,
                    1,
                    1,
                    "CENTRAL_REJECTOR_MISSING",
                    (
                        f"required central rejector {CENTRAL_REJECTOR} "
                        "is missing"
                    ),
                )
            )
        elif len(rejector_nodes) > 1:
            diagnostics.append(
                Diagnostic(
                    RUNTIME_SAFETY_PATH,
                    getattr(rejector_nodes[1], "lineno", 1),
                    getattr(rejector_nodes[1], "col_offset", 0) + 1,
                    "CENTRAL_REJECTOR_DUPLICATE",
                    (
                        f"required central rejector {CENTRAL_REJECTOR} "
                        "has multiple definitions"
                    ),
                )
            )

    return sorted(set(diagnostics))


def scan_repository(root: Path = APPLICATION_ROOT) -> tuple[list[Diagnostic], int]:
    relative_paths = tracked_production_python_files(root)
    return scan_paths(root, relative_paths), len(relative_paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check every tracked production Python source for E*TRADE "
            "mutation-boundary bypasses."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=APPLICATION_ROOT,
        help="application root containing the tracked Python sources",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        diagnostics, source_count = scan_repository(args.root)
    except BoundaryConfigurationError as exc:
        print(f"mutation-boundary configuration error: {exc}", file=sys.stderr)
        return 2
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic.render(), file=sys.stderr)
        print(
            (
                f"E*TRADE mutation boundary failed with "
                f"{len(diagnostics)} violation(s)."
            ),
            file=sys.stderr,
        )
        return 1
    print(
        (
            "E*TRADE mutation boundary passed for "
            f"{source_count} tracked production Python source(s)."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
