from __future__ import annotations

import unittest

from run_benchmarks import validate_execution_selection


class ExecutionSelectionTests(unittest.TestCase):
    def test_baseline_only_uncertainty_is_supported(self) -> None:
        validate_execution_selection(
            {
                "run_baseline": True,
                "traj_test": False,
                "range_test": True,
                "chunk_test": False,
            },
            group_runs=[],
            classic_baselines=["valhalla_meili"],
        )

    def test_baseline_only_trajectory_is_supported(self) -> None:
        validate_execution_selection(
            {
                "run_baseline": True,
                "traj_test": True,
                "range_test": False,
                "chunk_test": False,
            },
            group_runs=[],
            classic_baselines=["raw"],
        )

    def test_empty_method_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No learned model_groups or classic baselines"):
            validate_execution_selection(
                {
                    "run_baseline": True,
                    "traj_test": False,
                    "range_test": True,
                    "chunk_test": False,
                },
                group_runs=[],
                classic_baselines=[],
            )

    def test_baseline_only_chunk_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Baseline-only chunk evaluation"):
            validate_execution_selection(
                {
                    "run_baseline": True,
                    "traj_test": False,
                    "range_test": False,
                    "chunk_test": True,
                },
                group_runs=[],
                classic_baselines=["raw"],
            )


if __name__ == "__main__":
    unittest.main()
