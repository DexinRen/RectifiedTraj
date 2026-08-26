from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baseline.models.valhalla_meili.client import parse_trace_attributes_response
from baseline.models.valhalla_meili.map_tools import (
    apply_soft_buffer,
    ensure_dataset_map,
    load_processor_bounds,
)
from baseline.models.valhalla_meili.model import (
    ValhallaMeiliBaselineModel,
    build_request_windows,
)
from utils.evaluations.classic_baseline_runner import run_classic_baselines_filtered


def _model_config() -> dict:
    return {
        "base_url": "http://127.0.0.1:8002",
        "costing": "auto",
        "shape_match": "map_snap",
        "window_points": 500,
        "overlap_points": 50,
        "timeout_sec": 60.0,
        "auto_start": False,
        "map_id": "TestDataset",
        "compose_file": "docker-compose.yml",
        "processed_map_root": "dataset/processed/map",
        "port": 8002,
        "startup_timeout_sec": 120.0,
        "build_timeout_sec": 7200.0,
    }


class WindowTests(unittest.TestCase):
    def test_long_schedule_covers_every_point(self) -> None:
        packet = build_request_windows(5000, 500, 50)
        coverage = np.zeros(5000, dtype=bool)
        for window in packet["windows"]:
            coverage[window["start"] : window["end"]] = True
            self.assertLessEqual(window["end"] - window["start"], 500)
        for previous, current in zip(packet["windows"], packet["windows"][1:]):
            self.assertEqual(previous["end"] - current["start"], 50)
        self.assertTrue(bool(np.all(coverage)))
        self.assertEqual(packet["windows"][0], {"start": 0, "end": 500})
        self.assertEqual(packet["windows"][-1], {"start": 4500, "end": 5000})

    def test_501_points_uses_two_overlapping_windows(self) -> None:
        windows = build_request_windows(501, 500, 50)["windows"]
        self.assertEqual(windows, [{"start": 0, "end": 500}, {"start": 450, "end": 501}])


class ResponseTests(unittest.TestCase):
    def test_unmatched_and_discontinuous_points_are_rejected(self) -> None:
        payload = {
            "edges": [{"id": 10}],
            "matched_points": [
                {"type": "matched", "edge_index": 0, "lat": 35.0, "lon": 139.0},
                {"type": "unmatched", "edge_index": 0, "lat": 35.1, "lon": 139.1},
                {
                    "type": "interpolated",
                    "edge_index": 0,
                    "lat": 35.2,
                    "lon": 139.2,
                    "begin_route_discontinuity": True,
                },
            ],
        }
        packet = parse_trace_attributes_response(payload, 3)
        self.assertEqual(packet["accepted_mask"].tolist(), [True, False, False])
        self.assertEqual(packet["unmatched_points"], 2)
        self.assertEqual(packet["discontinuity_points"], 1)

    def test_edge_less_http_success_is_rejected(self) -> None:
        payload = {
            "edges": [],
            "matched_points": [
                {"type": "matched", "edge_index": 0, "lat": 35.0, "lon": 139.0},
                {"type": "matched", "edge_index": 0, "lat": 35.1, "lon": 139.1},
            ],
        }
        packet = parse_trace_attributes_response(payload, 2)
        self.assertEqual(packet["error_code"], 2)
        self.assertFalse(bool(np.any(packet["accepted_mask"])))


class MapMetadataTests(unittest.TestCase):
    def test_processor_bounds_are_the_only_bbox_input(self) -> None:
        payload = {
            "parquet_processor": {
                "dataset_noisy_boundary_corners": {
                    "max_lat_min_lon": [36.0, 139.0],
                    "max_lat_max_lon": [36.0, 140.0],
                    "min_lat_min_lon": [35.0, 139.0],
                    "min_lat_max_lon": [35.0, 140.0],
                }
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state_TestDataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            packet = load_processor_bounds(path)
        self.assertEqual(
            packet["bbox"],
            {"min_lon": 139.0, "min_lat": 35.0, "max_lon": 140.0, "max_lat": 36.0},
        )

    def test_soft_buffer_clips_at_source_border(self) -> None:
        packet = apply_soft_buffer(
            {"min_lon": 139.0, "min_lat": 35.0, "max_lon": 140.0, "max_lat": 36.0},
            {"min_lon": 138.0, "min_lat": 34.0, "max_lon": 140.005, "max_lat": 37.0},
            1.0,
        )
        self.assertAlmostEqual(packet["bbox"]["max_lon"], 140.005)
        self.assertLess(packet["applied_buffer_km"]["east"], 1.0)
        self.assertAlmostEqual(packet["applied_buffer_km"]["west"], 1.0, places=6)

    def test_soft_buffer_refuses_more_than_one_kilometer(self) -> None:
        with self.assertRaisesRegex(ValueError, "may not exceed"):
            apply_soft_buffer(
                {"min_lon": 139.0, "min_lat": 35.0, "max_lon": 140.0, "max_lat": 36.0},
                {"min_lon": 138.0, "min_lat": 34.0, "max_lon": 141.0, "max_lat": 37.0},
                1.01,
            )

    def test_existing_tailored_map_does_not_read_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            map_dir = root / "processed" / "TestDataset"
            map_dir.mkdir(parents=True)
            map_path = map_dir / "TestDataset.osm.pbf"
            map_path.write_bytes(b"test-map")
            packet = ensure_dataset_map(
                {
                    "map_id": "TestDataset",
                    "state_file": str(root / "missing-state.json"),
                    "source": "japan",
                    "raw_map_root": str(root / "raw"),
                    "processed_map_root": str(root / "processed"),
                    "buffer_km": 1.0,
                    "map_tools_dockerfile": str(root / "missing-Dockerfile"),
                }
            )
        self.assertEqual(packet["status"], "existing")

    @mock.patch("baseline.models.valhalla_meili.map_tools.cut_dataset_map")
    def test_missing_tailored_map_calls_automatic_cutter(
        self,
        cut_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_root = root / "processed"
            expected_map = processed_root / "TestDataset" / "TestDataset.osm.pbf"
            expected_manifest = processed_root / "TestDataset" / "map_manifest.json"

            def publish_map(**_kwargs) -> dict:
                expected_map.parent.mkdir(parents=True)
                expected_map.write_bytes(b"test-map")
                expected_manifest.write_text("{}\n", encoding="utf-8")
                return {
                    "error_code": 0,
                    "map_path": str(expected_map),
                    "manifest_path": str(expected_manifest),
                    "manifest": {},
                }

            cut_mock.side_effect = publish_map
            packet = ensure_dataset_map(
                {
                    "map_id": "TestDataset",
                    "state_file": str(root / "state_TestDataset.json"),
                    "source": "japan",
                    "raw_map_root": str(root / "raw"),
                    "processed_map_root": str(processed_root),
                    "buffer_km": 1.0,
                    "map_tools_dockerfile": str(root / "Dockerfile.map_tools"),
                }
            )
        self.assertEqual(packet["status"], "created")
        cut_mock.assert_called_once()


class ModelTests(unittest.TestCase):
    @mock.patch("baseline.models.valhalla_meili.model.request_trace_attributes")
    def test_rejected_points_preserve_raw_input_coordinates(
        self,
        request_mock: mock.Mock,
    ) -> None:
        request_mock.return_value = {
            "error_code": 1,
            "valhalla_error_code": None,
            "adapter_error_code": 0,
            "http_status": 200,
            "positions_latlon": np.asarray([[35.01, 139.01], [np.nan, np.nan]]),
            "accepted_mask": np.asarray([True, False]),
            "diagnostics": {
                "transport_error": None,
                "point_type_counts": {"matched": 1, "unmatched": 1},
                "unmatched_points": 1,
                "discontinuity_points": 0,
                "invalid_response": False,
            },
        }
        seq = np.asarray([[35.0, 139.0, 0.0], [35.1, 139.1, 1.0]])

        packet = ValhallaMeiliBaselineModel(
            dataset_name="TestDataset",
            config=_model_config(),
        ).predict_packet(seq)

        np.testing.assert_allclose(
            packet["positions_latlon"],
            np.asarray([[35.01, 139.01], [35.1, 139.1]]),
        )
        self.assertEqual(packet["accepted_mask"].tolist(), [True, False])
        self.assertFalse(packet["complete"])
        self.assertEqual(packet["diagnostics"]["fallback_policy"], "raw_input")
        self.assertEqual(packet["diagnostics"]["fallback_points"], 1)

    @mock.patch("baseline.create_baseline_model")
    def test_partial_trajectory_is_scored_with_raw_fallback(
        self,
        create_model_mock: mock.Mock,
    ) -> None:
        class PartialModel:
            def resource_usage_roots(self) -> dict:
                return {"error_code": 0, "pids": []}

            def predict_packet(self, seq: np.ndarray) -> dict:
                return {
                    "error_code": 1,
                    "positions_latlon": np.asarray(seq[:, :2], dtype=float),
                    "accepted_mask": np.asarray([True, False]),
                    "complete": False,
                    "diagnostics": {
                        "attempted_requests": 1,
                        "accepted_requests": 0,
                        "rejected_requests": 1,
                        "http_status_counts": {"200": 1},
                        "valhalla_error_code_counts": {},
                        "adapter_error_code_counts": {},
                        "transport_error_counts": {},
                        "point_type_counts": {"matched": 1, "unmatched": 1},
                        "request_records": [],
                    },
                }

            def deconst(self) -> dict:
                return {"error_code": 0}

        class Trajectory:
            noisy_gps = np.asarray([[139.0, 35.0], [139.1, 35.1]])
            clean_gps = np.asarray([[139.0, 35.0], [139.1, 35.1]])
            timestamps = np.asarray([0.0, 1.0])

        class Evaluator:
            def _gps_to_enu_batch(
                self,
                gps: np.ndarray,
                _ref_lat: float,
                _ref_lon: float,
            ) -> np.ndarray:
                return np.asarray(gps, dtype=float)

            def _compute_pointwise_metrics(self, errors: np.ndarray) -> dict:
                return {
                    "avg": float(np.mean(errors)),
                    "med": float(np.median(errors)),
                    "p95": float(np.percentile(errors, 95)),
                    "std": float(np.std(errors)),
                }

            def _compute_trajectory_pointwise_profile(
                self,
                _trajectories: list,
                _errors: np.ndarray,
            ) -> dict:
                return {"avg_list": [], "avg_list_norm": []}

            def _compute_bytewise_metrics(
                self,
                _trajectories: list,
                _errors: np.ndarray,
            ) -> dict:
                return {"avg_list": [], "avg_list_norm": []}

            def _compute_chunkwise_metrics(
                self,
                _trajectories: list,
                _errors: np.ndarray,
                _k: int,
                _q1: int,
                _q2: int,
            ) -> dict:
                return {"avg_list": [], "avg_list_norm": []}

            def _build_traj_p_val_rows_from_lists(self, *_args) -> list:
                return []

            def _save_results(self, _result: dict) -> dict:
                return {"error_code": 0}

        class Manager:
            trajectory_evaluator = Evaluator()

        create_model_mock.return_value = PartialModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_classic_baselines_filtered(
                Manager(),
                [Trajectory()],
                "TestDataset_traj_native_1_2",
                ["valhalla_meili"],
                dataset_name_hint="TestDataset",
                baseline_config={},
                diagnostics_output_dir=temp_dir,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model_name"], "valhalla_meili_raw_fallback")
        self.assertEqual(results[0]["accepted_trajectories"], 0)
        self.assertEqual(results[0]["partial_trajectories"], 1)
        self.assertEqual(results[0]["scored_trajectories"], 1)
        self.assertEqual(results[0]["avg_l2_err_pw"], 0.0)

    @mock.patch("baseline.models.valhalla_meili.model.request_trace_attributes")
    def test_valhalla_error_codes_are_counted_separately(self, request_mock: mock.Mock) -> None:
        request_mock.return_value = {
            "error_code": 171,
            "valhalla_error_code": 171,
            "adapter_error_code": 0,
            "http_status": 400,
            "positions_latlon": np.full((2, 2), np.nan),
            "accepted_mask": np.zeros(2, dtype=bool),
            "diagnostics": {
                "transport_error": None,
                "point_type_counts": {},
                "unmatched_points": 2,
                "discontinuity_points": 0,
                "invalid_response": False,
            },
        }
        seq = np.asarray([[35.0, 139.0, 0.0], [35.1, 139.1, 1.0]])
        packet = ValhallaMeiliBaselineModel(
            dataset_name="TestDataset",
            config=_model_config(),
        ).predict_packet(seq)
        self.assertEqual(packet["diagnostics"]["valhalla_error_code_counts"], {"171": 1})
        self.assertEqual(packet["diagnostics"]["adapter_error_code_counts"], {})
        self.assertEqual(packet["diagnostics"]["http_status_counts"], {"400": 1})

    @mock.patch("baseline.models.valhalla_meili.model.request_trace_attributes")
    def test_overlap_stitching_is_point_aligned(self, request_mock: mock.Mock) -> None:
        def response(seq: np.ndarray, **_kwargs) -> dict:
            n_points = len(seq)
            return {
                "error_code": 0,
                "valhalla_error_code": None,
                "adapter_error_code": 0,
                "http_status": 200,
                "positions_latlon": np.asarray(seq[:, :2], dtype=float),
                "accepted_mask": np.ones(n_points, dtype=bool),
                "diagnostics": {
                    "transport_error": None,
                    "point_type_counts": {"matched": n_points},
                    "unmatched_points": 0,
                    "discontinuity_points": 0,
                    "invalid_response": False,
                },
            }

        request_mock.side_effect = response
        seq = np.column_stack(
            [
                np.linspace(35.0, 35.1, 501),
                np.linspace(139.0, 139.1, 501),
                np.arange(501),
            ]
        )
        packet = ValhallaMeiliBaselineModel(
            dataset_name="TestDataset",
            config=_model_config(),
        ).predict_packet(seq)
        np.testing.assert_allclose(packet["positions_latlon"], seq[:, :2])
        self.assertTrue(packet["complete"])
        self.assertEqual(packet["diagnostics"]["attempted_requests"], 2)


if __name__ == "__main__":
    unittest.main()
