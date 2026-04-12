#!/usr/bin/env python3
"""Unit tests for GPU baseline runner helpers."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from run_gpu_baselines import average_probabilities, make_criterion, parse_ensemble_seeds, resolve_signal_samples


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

    def test_make_criterion_uses_requested_label_smoothing(self):
        criterion = make_criterion(0.1)

        self.assertAlmostEqual(criterion.label_smoothing, 0.1)

    def test_resolve_signal_samples_uses_overrides_when_present(self):
        cfg = {"signal": {"sample_rate": 250, "window_size_sec": 10, "train_stride_sec": 5}}

        self.assertEqual(resolve_signal_samples(cfg, train_stride_sec=None), (2500, 1250))
        self.assertEqual(resolve_signal_samples(cfg, train_stride_sec=2.5), (2500, 625))


if __name__ == "__main__":
    unittest.main()
