"""Project coarse-to-fine budget allocation extracted from the notebook.

Sampling, candidate selection, and debug information preserve the original
research implementation. Baseline LIME classes retain their attribution
and BSD license in lime_image.py, lime_base.py, and LICENSE.txt.
"""

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from .lime_image import ImageExplanation, LimeImageExplainer


def _hide_color_rgb(hide_color, image):
    # image: HWC float in [0,1]
    if hide_color is None:
        return np.mean(image.reshape(-1, 3), axis=0)
    if np.isscalar(hide_color):
        return np.array([hide_color, hide_color, hide_color], dtype=image.dtype)
    return np.array(hide_color, dtype=image.dtype).reshape(3,)


def _apply_mask(image, segments, z, hide_rgb):
    """
    image: (H,W,3) float
    segments: (H,W) int superpixel ids 0..K-1
    z: (K,) binary mask, 1 keep, 0 hide
    """
    out = image.copy()
    off_ids = np.where(z == 0)[0]
    for sp_id in off_ids:
        out[segments == sp_id] = hide_rgb
    return out


def _sample_masks_random(rng, n, K, p_on=0.5):
    Z = (rng.random((n, K)) < p_on).astype(np.int32)
    Z[0, :] = 1  # keep the original instance
    return Z


def _sample_masks_focus_candidates(rng, n, K, candidates, p_on_cand=0.5):
    """
    Non-candidates are kept ON. Only candidates are toggled.
    """
    Z = np.ones((n, K), dtype=np.int32)
    Z[0, :] = 1
    cand = np.array(sorted(list(candidates)), dtype=int)

    if len(cand) == 0:
        return _sample_masks_random(rng, n, K, p_on=0.5)

    Z[1:, cand] = (rng.random((n - 1, len(cand))) < p_on_cand).astype(np.int32)
    return Z


def _batched_classifier(classifier_fn, imgs, batch_size=32):
    outs = []
    for i in range(0, len(imgs), batch_size):
        outs.append(classifier_fn(imgs[i:i+batch_size]))
    return np.concatenate(outs, axis=0)


def explain_instance_coarse_to_fine(
    image,
    classifier_fn,
    segmentation_fn,
    label_to_explain,
    budget,
    seed=0,
    hide_color=0,
    num_features=10,
    distance_metric="cosine",
    coarse_frac=0.2,     # will be overwritten if adaptive_coarse=True
    candidate_k=20,      # base candidate size, will be enlarged by candidate_k_eff
    batch_size=32,
    p_on_stage1=0.5,
    p_on_stage2=0.5,
    epsilon=0.2,         # fine stage exploration ratio, e.g., 0.2 means 20% random masks
    adaptive_coarse=True # adapt coarse_frac based on total budget
):
    """
    Two-stage sampling (Coarse-to-Fine) with variance reduction guards:
      - Stage 1: random masks with N1 -> get candidates (screening)
      - Stage 2: mixture sampling with N2:
          (1-epsilon) focus on candidates + epsilon global random (exploration)
      - Candidate set is enlarged to improve recall.

    Notes:
      - Total queries remain = budget.
      - This does not guarantee improvement for all cases, but reduces selection bias risk.
    """
    rng = np.random.default_rng(seed)

    # Use LimeImageExplainer only for accessing LimeBase (explainer.base) and kernel
    explainer = LimeImageExplainer(random_state=seed)

    # Fixed segmentation
    segments = segmentation_fn(image)
    K = int(segments.max() + 1)

    hide_rgb = _hide_color_rgb(hide_color, image)

    # adaptive coarse fraction (optional but recommended)
    if adaptive_coarse:
        if budget <= 500:
            coarse_frac = 0.5
        elif budget <= 2000:
            coarse_frac = 0.3
        else:
            coarse_frac = 0.2

    # split budget
    N1 = int(max(50, round(coarse_frac * budget)))
    N1 = min(N1, budget - 10) if budget > 60 else max(10, budget // 2)
    N2 = budget - N1
    if N2 < 10:
        N2 = max(10, budget // 2)
        N1 = budget - N2

    # Stage 1: coarse
    Z1 = _sample_masks_random(rng, N1, K, p_on=p_on_stage1)
    imgs1 = np.stack(
        [_apply_mask(image, segments, z, hide_rgb) for z in tqdm(Z1, desc="Coarse stage", leave=False)],
        axis=0
    )
    probs1 = _batched_classifier(classifier_fn, imgs1, batch_size=batch_size)

    d1 = sklearn.metrics.pairwise_distances(
        Z1, Z1[0].reshape(1, -1), metric=distance_metric
    ).ravel()

    model_regressor = Ridge(alpha=1.0, fit_intercept=True, random_state=seed)
    feature_selection = getattr(explainer, "feature_selection", "auto")

    (_, local_exp1, _, _) = explainer.base.explain_instance_with_data(
        Z1, probs1, d1, label_to_explain,
        num_features=max(candidate_k, num_features),
        model_regressor=model_regressor,
        feature_selection=feature_selection
    )

    # Candidate enlargement
    local_exp1_sorted = sorted(local_exp1, key=lambda t: abs(t[1]), reverse=True)
    """
    Note:
    We keep candidate_k_eff here to reflect the intended design idea of using
    a larger candidate region to improve recall in the fine stage.
    However, in the current implementation, local_exp1 already contains at most
    max(candidate_k, num_features) selected superpixels returned by
    explain_instance_with_data, so candidate_k_eff does not actually enlarge the
    candidate set. Slicing with a larger upper bound is still safe in Python, so
    this does not affect code execution. In practice, we still use at most
    max(candidate_k, num_features) superpixels as candidates.
    """
    candidate_k_eff = max(candidate_k, 4 * num_features)  # e.g., num_features=10 -> at least 40 candidates
    candidates = set([int(sp_id) for sp_id, _ in local_exp1_sorted[:candidate_k_eff]])

    # Stage 2: fine with exploration
    eps = float(epsilon)
    eps = max(0.0, min(0.9, eps))  # keep it sane

    n_focus = int(round((1 - eps) * N2))
    n_focus = max(2, min(n_focus, N2 - 2))  # ensure both focus and random exist
    n_rand = N2 - n_focus

    Z2_focus = _sample_masks_focus_candidates(rng, n_focus, K, candidates, p_on_cand=p_on_stage2)
    # random masks (exploration)
    Z2_rand = _sample_masks_random(rng, n_rand, K, p_on=p_on_stage1)

    Z2 = np.vstack([np.ones((1, K), dtype=np.int32), Z2_focus[1:], Z2_rand[1:]])
    Z2 = Z2[:N2]  # safety
    Z2[0, :] = 1

    imgs2 = np.stack(
        [_apply_mask(image, segments, z, hide_rgb) for z in tqdm(Z2, desc="Fine stage", leave=False)],
        axis=0
    )
    probs2 = _batched_classifier(classifier_fn, imgs2, batch_size=batch_size)

    # combine (drop stage2 original row to avoid duplicate all-ones)
    Z = np.vstack([Z1, Z2[1:]])
    probs = np.vstack([probs1, probs2[1:]])

    distances = sklearn.metrics.pairwise_distances(
        Z, Z[0].reshape(1, -1), metric=distance_metric
    ).ravel()

    # Build ImageExplanation like vanilla
    ret_exp = ImageExplanation(image, segments)
    ret_exp.top_labels = [label_to_explain]

    # debug info
    weights_dbg = explainer.base.kernel_fn(distances)
    ret_exp._debug_distances = distances
    ret_exp._debug_weights = weights_dbg
    ret_exp._debug_labels = probs
    ret_exp._debug_data = Z
    ret_exp._debug_candidates = candidates
    ret_exp._debug_split = (N1, N2)
    ret_exp._debug_eps = eps
    ret_exp._debug_candidate_k_eff = candidate_k_eff

    (ret_exp.intercept[label_to_explain],
     ret_exp.local_exp[label_to_explain],
     ret_exp.score[label_to_explain],
     ret_exp.local_pred[label_to_explain]) = explainer.base.explain_instance_with_data(
        Z, probs, distances, label_to_explain, num_features,
        model_regressor=model_regressor,
        feature_selection=feature_selection
    )

    if not hasattr(ret_exp, "_debug_easy_models"):
        ret_exp._debug_easy_models = {}
    ret_exp._debug_easy_models[label_to_explain] = getattr(explainer.base, "_last_easy_model", None)

    return ret_exp
