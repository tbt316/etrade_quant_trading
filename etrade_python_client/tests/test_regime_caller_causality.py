import ast
import subprocess
import unittest
from pathlib import Path

import pandas as pd

from backtesting.regime_bridge import build_lagged_final_risk_map
from live_trading.market_sessions import (
    MarketSessionUnavailable,
    filter_to_nyse_sessions,
    latest_available_session_before,
    prior_nyse_session_map,
    require_nyse_session_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUARANTINED_REGIME_BACKTESTS = (
    "scratch/run_best_backtest.py",
    "scratch/run_backtest_2021.py",
    "scratch/optimize_backtest.py",
    "scratch/compare_backtests.py",
    "scratch/run_k_comparison.py",
    "scratch.py",
    "backtesting/run_spx_width_comparison.py",
)


class ExpandingRegimeCallerGuardTests(unittest.TestCase):
    def test_every_tracked_expanding_caller_declares_fit_end(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "*.py"],
            cwd=PROJECT_ROOT,
            text=True,
        ).splitlines()
        violations = []
        for relative_path in tracked:
            if relative_path.startswith("tests/"):
                continue
            path = PROJECT_ROOT / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else None
                    )
                )
                if function_name != "train_regime_hmm":
                    continue
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                }
                expanding = keywords.get("expanding_window")
                if (
                    isinstance(expanding, ast.Constant)
                    and expanding.value is True
                    and "fit_end" not in keywords
                ):
                    violations.append(
                        f"{relative_path}:{node.lineno}"
                    )
        self.assertEqual(violations, [])

    def test_unsafe_legacy_backtest_scripts_are_stable_tombstones(self):
        for relative_path in QUARANTINED_REGIME_BACKTESTS:
            source = (PROJECT_ROOT / relative_path).read_text(
                encoding="utf-8"
            )
            tree = ast.parse(source)
            calls = [
                node
                for node in ast.walk(tree)
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "train_regime_hmm"
                )
            ]
            self.assertEqual(calls, [], relative_path)
            self.assertIn("UNSAFE_REGIME_", source)


class ExactPriorSessionProbabilityTests(unittest.TestCase):
    def test_prior_session_mapping_handles_weekends_and_holidays(self):
        mapping = prior_nyse_session_map(
            ["2025-01-06", "2025-01-21"]
        )
        self.assertEqual(mapping["2025-01-06"], "2025-01-03")
        self.assertEqual(mapping["2025-01-21"], "2025-01-17")

    def test_prior_session_mapping_rejects_noncanonical_inputs(self):
        invalid_sequences = (
            [],
            {"2025-01-06"},
            ("2025-01-06" for _ in range(1)),
            ["2025-1-06"],
            ["2025-01-06T00:00:00"],
            [pd.Timestamp("2025-01-06")],
            ["2025-01-06", "2025-01-06"],
            ["2025-01-21", "2025-01-06"],
            ["2025-02-30"],
        )
        for invalid in invalid_sequences:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    MarketSessionUnavailable,
                    "INVALID_NYSE_SESSION_SEQUENCE",
                ):
                    prior_nyse_session_map(invalid)

    def test_overlay_trace_uses_exact_prior_session_without_gap_fill(self):
        feature_df = pd.DataFrame(
            {
                "Detected_Regime_State": [0, 1],
                "Detected_Regime_Label": [
                    "Expansion (0)",
                    "Cautious Decline (1)",
                ],
            },
            index=pd.to_datetime(["2025-01-03", "2025-01-17"]),
        )
        result = build_lagged_final_risk_map(
            feature_df,
            ["2025-01-06", "2025-01-07", "2025-01-21"],
        )
        self.assertEqual(
            result.by_entry_session["2025-01-06"].state,
            0,
        )
        self.assertNotIn("2025-01-07", result.by_entry_session)
        self.assertEqual(
            result.unavailable_code_by_entry["2025-01-07"],
            "FINAL_OVERLAY_PRIOR_SESSION_MISSING",
        )
        self.assertEqual(
            result.by_entry_session["2025-01-21"].state,
            1,
        )

    def test_latest_available_fit_end_is_strictly_pretest(self):
        index = pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06"]
        )
        self.assertEqual(
            latest_available_session_before(index, "2025-01-06"),
            "2025-01-03",
        )

    def test_latest_available_fit_end_ignores_weekend_rows(self):
        index = pd.to_datetime(
            ["2025-01-03", "2025-01-04", "2025-01-05"]
        )
        self.assertEqual(
            latest_available_session_before(index, "2025-01-06"),
            "2025-01-03",
        )

    def test_modeling_frame_excludes_weekends_and_exchange_holidays(self):
        frame = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5]},
            index=pd.to_datetime(
                [
                    "2025-01-17",
                    "2025-01-18",
                    "2025-01-19",
                    "2025-01-20",
                    "2025-01-21",
                ]
            ),
        )
        filtered = filter_to_nyse_sessions(frame)
        self.assertEqual(
            filtered.index.strftime("%Y-%m-%d").tolist(),
            ["2025-01-17", "2025-01-21"],
        )
        require_nyse_session_index(filtered.index)
        with self.assertRaisesRegex(
            MarketSessionUnavailable,
            "NON_NYSE_MODELING_SESSION_PRESENT",
        ):
            require_nyse_session_index(frame.index)

if __name__ == "__main__":
    unittest.main()
