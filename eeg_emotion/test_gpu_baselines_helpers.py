#!/usr/bin/env python3
"""Unit tests for GPU baseline runner helpers."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from run_gpu_baselines import (
    average_probabilities,
    balanced_rank_predictions,
    resolve_eval_seed,
    make_bd_conformer_kwargs,
    make_criterion,
    parse_ensemble_seeds,
    resolve_signal_samples,
)
from raw_dataset import standardize_by_window


class GPUBaselineHelperTests(unittest.TestCase):
    def test_parse_ensemble_seeds_defaults_to_base_seed(self):
        self.assertEqual(parse_ensemble_seeds(None, 42), (42,))
        self.assertEqual(parse_ensemble_seeds("", 42), (42,))

    def test_parse_ensemble_seeds_accepts_comma_separated_ints(self):
        self.assertEqual(parse_ensemble_seeds("42, 2026,7", 1), (42, 2026, 7))

    def test_average_probabilities_averages_member_outputs(self):
        first = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
        second = np.asarray([[0.6, 0.4], [0.9, 0.1]], dtype=np.float32)

        averaged = average_probabilities([first, second])

        np.testing.assert_allclose(averaged, np.asarray([[0.4, 0.6], [0.8, 0.2]], dtype=np.float32))

    def test_average_probabilities_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            average_probabilities([])

    def test_balanced_rank_predictions_forces_half_positive(self):
        probas = np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.3, 0.7],
                [0.7, 0.3],
                [0.4, 0.6],
                [0.6, 0.4],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )

        preds = balanced_rank_predictions(probas)

        self.assertEqual(int(preds.sum()), 4)
        np.testing.assert_array_equal(preds, np.asarray([0, 1, 0, 1, 0, 1, 0, 1]))

    def test_balanced_rank_predictions_handles_window_count(self):
        probas = np.column_stack([
            np.linspace(1.0, 0.0, 72),
            np.linspace(0.0, 1.0, 72),
        ]).astype(np.float32)

        preds = balanced_rank_predictions(probas)

        self.assertEqual(preds.shape, (72,))
        self.assertEqual(int(preds.sum()), 36)

    def test_make_criterion_uses_requested_label_smoothing(self):
        criterion = make_criterion(0.1)

        self.assertAlmostEqual(criterion.label_smoothing, 0.1)

    def test_resolve_eval_seed_defaults_to_training_seed(self):
        args = type("Args", (), {"eval_seed": None})()

        self.assertEqual(resolve_eval_seed(args, training_seed=42), 42)

    def test_resolve_eval_seed_uses_override(self):
        args = type("Args", (), {"eval_seed": 42})()

        self.assertEqual(resolve_eval_seed(args, training_seed=13), 42)

    def test_resolve_signal_samples_uses_overrides_when_present(self):
        cfg = {"signal": {"sample_rate": 250, "window_size_sec": 10, "train_stride_sec": 5}}

        self.assertEqual(resolve_signal_samples(cfg, train_stride_sec=None), (2500, 1250))
        self.assertEqual(resolve_signal_samples(cfg, train_stride_sec=2.5), (2500, 625))

    def test_make_bd_conformer_kwargs_supports_new_braindecode_names(self):
        args = type("Args", (), {"num_layers": 2, "num_heads": 4, "dropout": 0.2})()
        cfg = {"signal": {"n_channels": 30, "sample_rate": 250}}

        kwargs = make_bd_conformer_kwargs(
            args,
            cfg,
            window_size=2500,
            signature_parameters={"num_layers", "num_heads"},
        )

        self.assertEqual(kwargs["num_layers"], 2)
        self.assertEqual(kwargs["num_heads"], 4)
        self.assertNotIn("att_depth", kwargs)
        self.assertNotIn("att_heads", kwargs)

    def test_standardize_by_window_normalizes_each_window_channel(self):
        X = np.asarray(
            [
                [[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]],
                [[-2.0, 0.0, 2.0, 4.0], [5.0, 7.0, 9.0, 11.0]],
            ],
            dtype=np.float32,
        )

        standardized = standardize_by_window(X)

        np.testing.assert_allclose(standardized.mean(axis=-1), np.zeros((2, 2)), atol=1e-6)
        np.testing.assert_allclose(standardized.std(axis=-1), np.ones((2, 2)), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
