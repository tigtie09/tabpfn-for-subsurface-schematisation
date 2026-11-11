"""Analytical fields, independent scalar checks and cached-run validation."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_spatial_variation import (
    LAGS, METHODS, compare_output_summaries, descriptors, main,
    ratios_to_target, score_sections, summaries,
)
from common import sha256, write_json


class SpatialMetricTests(unittest.TestCase):
    def test_affine_field_uses_absolute_differences_without_distance_division(self):
        z, x = np.indices((32, 512))
        field = 2 + 2 * z + 3 * x
        wanted = np.asarray([2 * lag if d == "V" else 3 * lag for d, lag in LAGS])
        actual = descriptors(field[None])
        np.testing.assert_array_equal(actual[0, :, 0], wanted)
        np.testing.assert_array_equal(actual[0, :, 1], wanted)

    def test_layer_transition_has_positive_mean_but_zero_median(self):
        z, x = np.indices((32, 512))
        actual = descriptors((2 + 10 * (z >= 16) + 4 * (x >= 256))[None])[0]
        self.assertAlmostEqual(actual[0, 0], 10 / 31)
        self.assertAlmostEqual(actual[4, 0], 4 / 511)
        self.assertEqual(actual[0, 1], 0)
        self.assertEqual(actual[4, 1], 0)

    def test_all_lags_match_independent_scalar_pairs_including_edges(self):
        fields = np.random.default_rng(41).normal(size=(2, 32, 512)).astype(np.float32)
        actual = score_sections(fields, 2, 1)
        for section, field in enumerate(fields):
            for index, (direction, lag) in enumerate(LAGS):
                dz, dx = (lag, 0) if direction == "V" else (0, lag)
                pairs = [abs(float(field[z + dz, x + dx]) - float(field[z, x]))
                         for z in range(32 - dz) for x in range(512 - dx)]
                expected = (math.fsum(pairs) / len(pairs), statistics.median(pairs))
                np.testing.assert_allclose(actual[section, index], expected, rtol=1e-12, atol=1e-12)

    def test_ratios_are_formed_before_across_section_medians(self):
        shape = (3, len(LAGS), 2)
        target = np.broadcast_to(np.array([1, 2, 100])[:, None, None], shape)
        values = np.broadcast_to(np.array([3, 1, 100])[:, None, None], shape)
        ratios = ratios_to_target(values, target)
        rows = summaries({method: ratios for method in METHODS})
        self.assertEqual(len(rows), 96)
        self.assertTrue(all(row["median"] == 1 for row in rows))
        self.assertEqual(np.median(values[:, 0, 0]) / np.median(target[:, 0, 0]), 1.5)

    def test_median_field_is_not_the_mean_of_draw_descriptors(self):
        z, x = np.indices((32, 512))
        fields = np.asarray([2 + z + x / 8, -2 - z - x / 8])
        target = descriptors(fields[:1])
        baseline = np.mean([ratios_to_target(descriptors(f[None]), target) for f in fields], axis=0)
        median_field = ratios_to_target(descriptors(np.median(fields, axis=0)[None]), target)
        np.testing.assert_array_equal(baseline, 1)
        np.testing.assert_array_equal(median_field, 0)
        rows, findings = compare_output_summaries(baseline, median_field, baseline / 2)
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["change"] == -1 for row in rows))
        self.assertTrue(all(item["max_absolute_change_in_reported_medians"] == 1
                            for item in findings.values()))

    def test_zero_denominators_and_nonfinite_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive target"):
            ratios_to_target(np.ones((1, 12, 2)), np.zeros((1, 12, 2)))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            descriptors(np.full((1, 32, 512), np.nan))


class SpatialRunTests(unittest.TestCase):
    def setUp(self):
        # Preserve generated fixtures under ignored outputs, per repository policy.
        (ROOT / "outputs").mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="spatial-test-", dir=ROOT / "outputs"))
        self.data = self.root / "data"
        self.data.mkdir()
        z, x = np.indices((32, 512))
        self.truth = np.repeat((2 + 2 * z + 3 * x)[None], 3, axis=0).astype(np.float64)
        source = np.zeros_like(self.truth)
        columns = [8, 100, 200, 300, 400, 500]
        source[:, :, columns] = self.truth[:, :, columns]
        np.save(self.data / "src_images.npy", source[..., None])
        np.save(self.data / "tar_images.npy", self.truth[..., None])
        self.hashes = {name: sha256(self.data / name) for name in ("src_images.npy", "tar_images.npy")}
        write_json(self.data / "manifest.json", dict(
            normalized=False, output_sha256=self.hashes,
            config=dict(split="test", section_manifest="configs/synthetic_cpt/raw_sections.csv"),
        ))
        self.baseline = self.root / "baseline"
        for method, multiplier in (("tabpfn", .5), ("kriging", .8), ("nearest_neighbour", 0)):
            self.write_run(self.baseline / method, method, 0, self.truth * multiplier)
        draws = [self.truth * np.array([1, 1, 10])[:, None, None],
                 self.truth * np.array([10, 1, 1])[:, None, None]]
        hashes = {}
        for seed, fields in enumerate(draws):
            hashes[str(seed)] = self.write_run(self.baseline / "schemagan" / f"seed_{seed}",
                                                "schemagan", seed, fields)
        directory = self.baseline / "schemagan_summary"
        directory.mkdir()
        np.save(directory / "medians.npy", (draws[0] + draws[1]) / 2)
        write_json(directory / "run.json", dict(
            status="completed", n_draws=2, seeds=[0, 1], data_sha256=self.hashes,
            input_predictions_sha256=hashes,
            contract=dict(n_sections=3, seeds=[0, 1], count=None, layouts_sha256=None,
                          data_sha256=self.hashes, input_predictions_sha256=hashes),
            files={"medians.npy": sha256(directory / "medians.npy")},
        ))

    def write_run(self, directory, method, seed, fields):
        directory.mkdir(parents=True)
        np.save(directory / "predictions.npy", fields)
        digest = sha256(directory / "predictions.npy")
        write_json(directory / "run.json", dict(
            status="completed", contract=dict(method=method, inference_seed=seed,
                count=None, layouts_sha256=None, data_sha256=self.hashes, n_sections=3),
            files={"predictions.npy": digest},
        ))
        return digest

    def run_analysis(self, *extra):
        command = ["analyse_spatial_variation", "--baseline", str(self.baseline),
                   "--data", str(self.data), "--output", str(self.root / "analysis"), *extra]
        with patch.object(sys, "argv", command), redirect_stdout(io.StringIO()):
            main()

    def test_completed_runs_produce_draw_average_then_section_median(self):
        self.run_analysis("--block-size", "2")
        output = self.root / "analysis"
        with np.load(output / "per_section.npz") as values:
            np.testing.assert_array_equal(values["schemagan"][:, 0, 0], [5.5, 1, 5.5])
            np.testing.assert_allclose(values["tabpfn"], .5)
        report = json.loads((output / "run.json").read_text())
        self.assertFalse(report["full_analysis"])
        self.assertEqual(report["n_sections"], 3)
        self.assertEqual(report["n_draws"], 2)
        self.assertTrue(all(sha256(output / name) == digest for name, digest in report["files"].items()))
        text = (output / "table_b1.md").read_text()
        self.assertIn("| SchemaGAN | 5.50", text)

    def test_prefix_limit_keeps_matching_sections_and_marks_partial_run(self):
        self.run_analysis("--limit", "2")
        report = json.loads((self.root / "analysis/run.json").read_text())
        self.assertEqual(report["n_sections"], 2)
        self.assertFalse(report["full_analysis"])

    def test_mixed_count_inputs_are_rejected(self):
        path = self.baseline / "tabpfn/run.json"
        record = json.loads(path.read_text())
        record["contract"]["count"] = 12
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "matching baseline"):
            self.run_analysis()

    def test_changed_prediction_is_rejected(self):
        np.save(self.baseline / "tabpfn/predictions.npy", self.truth)
        with self.assertRaisesRegex(ValueError, "Changed prediction"):
            self.run_analysis()

    def test_median_cache_is_checked_against_the_individual_draws(self):
        directory = self.baseline / "schemagan_summary"
        np.save(directory / "medians.npy", self.truth)
        record = json.loads((directory / "run.json").read_text())
        record["files"]["medians.npy"] = sha256(directory / "medians.npy")
        (directory / "run.json").write_text(json.dumps(record))
        with self.assertRaises(AssertionError):
            self.run_analysis()


if __name__ == "__main__":
    unittest.main()
