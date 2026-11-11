"""Independent invariants at the sampling, masking and scoring boundaries."""

import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import error_metrics, observation_mask
from interval_metrics import normal_bounds, score_intervals
from nearest_model import nearest_grid
from prepare_synthetic_cpt import load_sections, read_section, sample_cpts
from prepare_layouts import nested, signature
from tabpfn_cpt_protocol import read_csv, training_arrays, query_grid


class ProtocolTests(unittest.TestCase):
    def test_baseline_matches_adopted_placements(self):
        expected = read_csv(ROOT / "tests/fixtures/baseline_placements.csv")
        rng = np.random.RandomState(20232023)
        dummy = np.ones((32, 512), np.float32)
        for i in range(8):
            sampled = sample_cpts(dummy, rng, 0.99, 51)
            actual = signature(sampled)
            wanted = tuple(
                zip(
                    map(int, expected[i]["columns"].split()),
                    map(int, expected[i]["depths"].split()),
                )
            )
            self.assertEqual(actual, wanted)

    def test_lexicographic_order_and_example_hashes(self):
        rows = load_sections(
            ROOT / "configs/synthetic_cpt/raw_sections.csv",
            "test",
            "relative_path_lexicographic",
        )
        self.assertEqual([r["section_id"] for r in rows[:4]], ["0", "1", "10", "100"])
        for r in rows[:2]:
            from common import sha256

            path = ROOT / "examples" / r["relative_path"]
            self.assertEqual(sha256(path), r["sha256"])
            self.assertEqual(read_section(path).shape, (32, 512))

    def test_nested_layout_and_common_mask_match_adopted_values(self):
        rng = np.random.RandomState(20232023)
        source = np.stack(
            [
                sample_cpts(np.ones((32, 512), np.float32), rng, 0.99, 51)
                for _ in range(4)
            ]
        )
        columns, depths = nested(source)
        expected = read_csv(ROOT / "tests/fixtures/counts_placements.csv")
        for i in range(4):
            np.testing.assert_array_equal(
                columns[i], list(map(int, expected[i]["columns"].split()))
            )
            np.testing.assert_array_equal(
                depths[i], list(map(int, expected[i]["depths"].split()))
            )
            target = ~observation_mask(columns[i], depths[i], 48)
            old = np.zeros((32, 512), bool)
            for count in (3, 6, 12, 24, 48):
                observed = observation_mask(columns[i], depths[i], count)
                self.assertFalse(np.any(old & ~observed))
                self.assertFalse(np.any(target & observed))
                self.assertEqual(observed.any(axis=0).sum(), count)
                old = observed

    def test_unobserved_values_do_not_enter_context_or_nearest(self):
        source = np.full((32, 512), np.nan, np.float32)
        mask = np.zeros_like(source, dtype=bool)
        source[3, 7] = 2.0
        mask[3, 7] = True
        X, y = training_arrays(source, mask)
        np.testing.assert_array_equal(X, [[7, 3]])
        np.testing.assert_array_equal(y, [2.0])
        prediction, donors = nearest_grid(source, mask)
        np.testing.assert_array_equal(prediction, np.full((32, 512), 2.0))
        self.assertTrue(np.all(donors == 3 * 512 + 7))
        np.testing.assert_array_equal(query_grid()[3 * 512 + 7], [7, 3])

    def test_rmse_is_computed_within_each_section(self):
        target = np.zeros((1, 2))
        mask = np.ones((1, 2), bool)
        a = error_metrics(np.array([[0.0, 2.0]]), target, mask)
        b = error_metrics(np.zeros((1, 2)), target, mask)
        self.assertAlmostEqual((a["rmse"] + b["rmse"]) / 2, np.sqrt(2) / 2)
        self.assertNotAlmostEqual(
            (a["rmse"] + b["rmse"]) / 2, np.sqrt((a["mse"] + b["mse"]) / 2)
        )

    def test_interval_score_and_mask(self):
        truth = np.full((32, 512), 2.0)
        truth[0, 0] = 1e9
        observed = np.zeros_like(truth, dtype=bool)
        observed[0, 0] = True
        bounds = np.stack([np.zeros_like(truth), np.ones_like(truth)])[None]
        scores = score_intervals(truth, observed, bounds, [0.5])
        r = next(r for r in scores if r["scope"] == "unobserved_pixels")
        self.assertEqual(r["coverage"], 0)
        self.assertEqual(r["width_mean"], 1)
        self.assertEqual(r["interval_score"], 5)
        zero = normal_bounds(np.zeros_like(truth), np.zeros_like(truth), [0.5, 0.9])
        np.testing.assert_array_equal(zero, 0)


if __name__ == "__main__":
    unittest.main()
