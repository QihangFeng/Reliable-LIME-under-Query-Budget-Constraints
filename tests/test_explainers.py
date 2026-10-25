"""Regression checks against numerical outputs captured before extraction.

The fixture was generated from notebook commit 3ea1706 using only its class and
function definitions, with the deterministic image/model in tests._fixtures.
Tests do not execute a historical notebook, run Git, download model weights,
or require a GPU. Timings are intentionally excluded from comparison.
"""

import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from reliable_lime.experiments import (
    run_lime_image_once_coarse_to_fine_fixed_label,
    run_lime_image_once_fixed_label,
)
from reliable_lime.lime_image import ImageExplanation
from reliable_lime.metrics import (
    deletion_test,
    fidelity_wmse_from_run,
    topk_superpixels_from_exp,
)
from tests._fixtures import CountingClassifier, snapshot_run, synthetic_image


class NotebookRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = Path(__file__).parent / "fixtures" / "notebook_reference.json"
        cls.cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
        cls.actual = {}
        for index, case in enumerate(cls.cases):
            image, segments = synthetic_image()
            classifier = CountingClassifier()
            expected = case["expected"]
            runner = (
                run_lime_image_once_fixed_label if case["method"] == "vanilla"
                else run_lime_image_once_coarse_to_fine_fixed_label
            )
            with contextlib.redirect_stderr(io.StringIO()):
                run = runner(
                    image, classifier, lambda image: segments,
                    budget=expected["budget"], seed=expected["seed"],
                    label_to_explain=1, hide_color=0, **case["kwargs"],
                )
                snapshot = snapshot_run(
                    run, classifier, fidelity_wmse_from_run,
                    topk_superpixels_from_exp, deletion_test,
                )
            cls.actual[index] = snapshot

    def assertSnapshotEqual(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        exact_fields = {
            "budget", "seed", "label", "reported_queries", "actual_images_seen",
            "batch_sizes", "classifier_inputs_sha256", "mask", "used_features",
            "fit_masks_shape", "fit_masks_sha256", "split", "candidates",
            "candidate_k_eff", "topk_ids",
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                if field in exact_fields:
                    self.assertEqual(actual[field], value)
                else:
                    np.testing.assert_allclose(
                        actual[field], value, rtol=1e-10, atol=1e-12,
                        err_msg="Notebook extraction changed {}".format(field),
                    )

    def test_vanilla_matches_original_inputs_explanations_and_metrics(self):
        for index, case in enumerate(self.cases):
            if case["method"] == "vanilla":
                with self.subTest(budget=case["expected"]["budget"], seed=case["expected"]["seed"]):
                    self.assertSnapshotEqual(self.actual[index], case["expected"])

    def test_coarse_to_fine_matches_original_inputs_explanations_and_metrics(self):
        for index, case in enumerate(self.cases):
            if case["method"] == "coarse_to_fine":
                with self.subTest(budget=case["expected"]["budget"], seed=case["expected"]["seed"]):
                    self.assertSnapshotEqual(self.actual[index], case["expected"])

    def test_existing_coarse_to_fine_query_accounting_is_characterized(self):
        # Characterize the notebook behavior; changing this requires a separate
        # algorithm change and new experimental results, rather than a refactor.
        for index, case in enumerate(self.cases):
            if case["method"] != "coarse_to_fine":
                continue
            actual = self.actual[index]
            budget = actual["budget"]
            with self.subTest(budget=budget):
                self.assertEqual(actual["reported_queries"], budget)
                self.assertEqual(actual["actual_images_seen"], budget - 1)
                self.assertEqual(actual["fit_masks_shape"][0], budget - 2)
                self.assertEqual(len(actual["fit_labels"]), budget - 2)
                self.assertEqual(sum(actual["split"]), budget)

    def test_baseline_queries_and_fit_rows_match_requested_budget(self):
        for index, case in enumerate(self.cases):
            if case["method"] == "vanilla":
                actual = self.actual[index]
                with self.subTest(budget=actual["budget"]):
                    self.assertEqual(actual["actual_images_seen"], actual["budget"])
                    self.assertEqual(len(actual["fit_labels"]), actual["budget"])


class ImageExplanationTests(unittest.TestCase):
    def setUp(self):
        self.image, self.segments = synthetic_image()
        self.explanation = ImageExplanation(self.image, self.segments)
        self.explanation.local_exp[1] = [(0, .8), (1, -.6), (2, .2)]

    def test_positive_mask_selects_positive_regions_only(self):
        image, mask = self.explanation.get_image_and_mask(1, num_features=2)
        np.testing.assert_array_equal(mask, np.isin(self.segments, [0, 2]).astype(int))
        np.testing.assert_array_equal(image, self.image)

    def test_negative_mask_can_hide_the_other_regions(self):
        image, mask = self.explanation.get_image_and_mask(
            1, positive_only=False, negative_only=True, hide_rest=True,
        )
        selected = self.segments == 1
        np.testing.assert_array_equal(mask, selected.astype(int))
        np.testing.assert_array_equal(image[selected], self.image[selected])
        np.testing.assert_array_equal(image[~selected], np.zeros_like(image[~selected]))

    def test_signed_mask_distinguishes_positive_and_negative_regions(self):
        _, mask = self.explanation.get_image_and_mask(
            1, positive_only=False, num_features=2,
        )
        expected = np.zeros_like(self.segments)
        expected[self.segments == 0] = 1
        expected[self.segments == 1] = -1
        np.testing.assert_array_equal(mask, expected)

    def test_unknown_label_and_incompatible_filters_raise(self):
        with self.assertRaises(KeyError):
            self.explanation.get_image_and_mask(99)
        with self.assertRaises(ValueError):
            self.explanation.get_image_and_mask(1, positive_only=True, negative_only=True)


if __name__ == "__main__":
    unittest.main()
