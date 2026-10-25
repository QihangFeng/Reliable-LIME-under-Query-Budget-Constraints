"""Analytical checks for the evaluation metrics, without a pretrained model."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import numpy as np

from reliable_lime.metrics import (
    deletion_test,
    fidelity_wmse_from_run,
    get_ref_label,
    jaccard,
    mask_superpixels,
    stability_jaccard,
    topk_superpixels_from_exp,
)
from reliable_lime.experiments import budget_sweep_metrics


class MetricTests(unittest.TestCase):
    def test_fidelity_is_weighted_mean_squared_error(self):
        model = SimpleNamespace(
            _debug_y_true=np.array([1., 2., 4.]),
            _debug_y_pred=np.array([0., 2., 2.]),
            _debug_weights=np.array([1., 2., 3.]),
        )
        explanation = SimpleNamespace(_debug_easy_models={2: model})
        run = {"exp": explanation, "label": 2}
        self.assertAlmostEqual(fidelity_wmse_from_run(run), 13 / 6)
        self.assertAlmostEqual(fidelity_wmse_from_run(run, label=2), 13 / 6)

    def test_topk_uses_absolute_weight_including_negative_features(self):
        explanation = SimpleNamespace(local_exp={1: [(0, 0.6), (1, 0.1), (2, -0.9)]})
        self.assertEqual(topk_superpixels_from_exp(explanation, 1, k=2), [2, 0])

    def test_jaccard_treats_inputs_as_sets(self):
        self.assertAlmostEqual(jaccard([0, 0, 1], [1, 2]), 1 / 3)
        self.assertEqual(jaccard([1], [1]), 1)
        self.assertEqual(jaccard([], []), 0)

    def test_stability_averages_all_run_pairs(self):
        runs = [
            {"exp": SimpleNamespace(local_exp={0: [(i, 1.), (1, .5)]}), "label": 0}
            for i in [0, 2, 3]
        ]
        self.assertAlmostEqual(stability_jaccard(runs, k=2), 1 / 3)
        self.assertTrue(np.isnan(stability_jaccard(runs[:1], k=2)))

    def test_mask_changes_only_selected_regions_and_preserves_input(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        segments = np.array([[0, 1], [0, 1]])
        masked = mask_superpixels(image, segments, [1], hide_color=[.1, .2, .3])
        np.testing.assert_allclose(masked[:, 1], [[.1, .2, .3], [.1, .2, .3]])
        np.testing.assert_array_equal(masked[:, 0], image[:, 0])
        np.testing.assert_array_equal(image, np.ones_like(image))
        self.assertEqual(masked.dtype, image.dtype)

    def test_deletion_is_probability_drop_and_includes_evaluation_queries(self):
        image = np.ones((2, 2, 3))
        segments = np.array([[0, 1], [0, 1]])
        image_counts = []

        def classifier(images):
            image_counts.append(len(images))
            p = images[:, :, :, 0].mean(axis=(1, 2))
            return np.stack([p, 1 - p], axis=1)

        self.assertAlmostEqual(
            deletion_test(image, segments, classifier, label=0, topk_ids=[0], k=1),
            .5,
        )
        self.assertEqual(sum(image_counts), 2)
        self.assertEqual(
            deletion_test(image, segments, classifier, label=0, topk_ids=[0, 1], k=2),
            1.,
        )

    def test_reference_label_uses_original_image_probability(self):
        image = np.zeros((2, 2, 3))

        def classifier(images):
            self.assertEqual(images.shape, (1, 2, 2, 3))
            return np.array([[.1, .7, .2]])

        label, probabilities = get_ref_label(image, classifier)
        self.assertEqual(label, 1)
        np.testing.assert_array_equal(probabilities, [.1, .7, .2])

    def test_budget_sweep_aggregates_run_metrics_and_all_seed_pairs(self):
        image = np.ones((1, 3, 3))
        segments = np.array([[0, 1, 2]])

        def classifier(images):
            p = images[:, :, :, 0].mean(axis=(1, 2))
            return np.stack([p, 1 - p], axis=1)

        def run_once(img, classifier_fn, segmentation_fn, budget,
                     label_to_explain, seed, **kwargs):
            model = SimpleNamespace(
                _debug_y_true=np.array([1., 2.]),
                _debug_y_pred=np.array([1., 1.]) if seed == 0 else np.array([0., 2.]),
                _debug_weights=np.array([1., 3.]),
            )
            explanation = SimpleNamespace(
                local_exp={0: [(seed * 2, 1.), (1, .5)]},
                _debug_easy_models={0: model},
            )
            return {"exp": explanation, "label": 0, "runtime_sec": budget / 10 + seed,
                    "queries": budget}

        with contextlib.redirect_stdout(io.StringIO()):
            runs, summary = budget_sweep_metrics(
                run_once, image, segments, lambda img: segments, classifier,
                budgets=[20, 40], seeds=[0, 1], ref_label=0, k=2,
            )
        self.assertEqual(len(runs), 4)
        np.testing.assert_array_equal(summary["budget"], [20, 40])
        np.testing.assert_allclose(summary["stability_jaccard"], [1 / 3, 1 / 3])
        np.testing.assert_allclose(summary["fidelity_wmse_mean"], [.5, .5])
        np.testing.assert_allclose(summary["faith_drop_k_mean"], [2 / 3, 2 / 3])
        np.testing.assert_allclose(summary["runtime_sec_mean"], [2.5, 4.5])
        np.testing.assert_array_equal(summary["queries"], [20, 40])


if __name__ == "__main__":
    unittest.main()
