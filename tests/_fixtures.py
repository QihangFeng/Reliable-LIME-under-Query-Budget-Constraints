"""Deterministic inputs shared by the recorded notebook regression cases."""

import hashlib

import numpy as np


def synthetic_image():
    """A small RGB gradient with nine distinct, fixed superpixels."""
    y, x = np.indices((12, 12))
    image = np.stack(
        [(x + 1) / 13, (y + 2) / 14, (x + y + 1) / 25], axis=-1
    ).astype(np.float64)
    segments = ((y // 4) * 3 + x // 4).astype(np.int64)
    return image, segments


class CountingClassifier:
    """A deterministic, nonlinear three-class model that records actual work."""

    def __init__(self):
        self.images_seen = 0
        self.batch_sizes = []
        self.inputs = []

    def __call__(self, images):
        images = np.asarray(images)
        self.images_seen += len(images)
        self.batch_sizes.append(len(images))
        self.inputs.append(images.copy())
        spatial = np.linspace(0.25, 1.5, 12 * 12).reshape(1, 12, 12, 1)
        signal = np.mean(images * spatial, axis=(1, 2, 3))
        contrast = (
            images[:, :6, :, 0].mean(axis=(1, 2))
            - images[:, 6:, :, 1].mean(axis=(1, 2))
        )
        logits = np.stack(
            [0.2 + signal, -0.15 + 1.7 * signal + 0.2 * contrast,
             0.4 - 0.8 * signal + 0.3 * signal ** 2],
            axis=1,
        )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def snapshot_run(run, classifier, fidelity_fn, topk_fn, deletion_fn):
    """Record fitted results and classifier inputs before metric-only queries."""
    explanation = run["exp"]
    label = run["label"]
    model = explanation._debug_easy_models[label]
    inputs = np.concatenate(classifier.inputs, axis=0)
    snapshot = {
        "budget": run["budget"],
        "seed": run["seed"],
        "label": int(label),
        "reported_queries": run["queries"],
        "actual_images_seen": classifier.images_seen,
        "batch_sizes": classifier.batch_sizes.copy(),
        "classifier_inputs_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
        "intercept": float(explanation.intercept[label]),
        "local_exp": [[int(i), float(w)] for i, w in explanation.local_exp[label]],
        "score": float(explanation.score[label]),
        "local_pred": np.asarray(explanation.local_pred[label]).tolist(),
        "mask": run["mask"].tolist(),
        "fit_labels": model._debug_y_true.tolist(),
        "fit_predictions": model._debug_y_pred.tolist(),
        "fit_weights": model._debug_weights.tolist(),
        "used_features": model._debug_used_features.tolist(),
    }
    if hasattr(explanation, "_debug_data"):
        masks = np.asarray(explanation._debug_data, dtype="<i8")
        snapshot.update({
            "fit_masks_shape": list(masks.shape),
            "fit_masks_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
            "split": list(explanation._debug_split),
            "candidates": sorted(explanation._debug_candidates),
            "candidate_k_eff": explanation._debug_candidate_k_eff,
            "epsilon": explanation._debug_eps,
        })
    snapshot["fidelity_wmse"] = fidelity_fn(run, label)
    snapshot["topk_ids"] = topk_fn(explanation, label, k=3)
    snapshot["faith_drop_k"] = deletion_fn(
        explanation.image, explanation.segments, classifier, label,
        snapshot["topk_ids"], hide_color=0, k=3,
    )
    return snapshot
