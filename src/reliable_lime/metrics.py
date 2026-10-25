"""Fidelity, stability, and deletion metrics extracted from the notebook."""

import itertools

import numpy as np

def get_ref_label(img, classifier_fn):
    p = classifier_fn(img[None, ...])[0]
    return int(np.argmax(p)), p


def fidelity_wmse_from_run(r, label=None):
    exp = r["exp"]
    if label is None:
        label = r["label"]
    m = exp._debug_easy_models[label]
    y_true = m._debug_y_true
    y_pred = m._debug_y_pred
    w = m._debug_weights
    return float(np.sum(w * (y_true - y_pred)**2) / np.sum(w))


def topk_superpixels_from_exp(exp, label, k=10):
    local_exp = exp.local_exp[label]  # list of (sp_id, weight)
    local_exp = sorted(local_exp, key=lambda t: abs(t[1]), reverse=True)
    return [int(sp_id) for sp_id, _ in local_exp[:k]]

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

def stability_jaccard(run_list, k=10):
    topks = [topk_superpixels_from_exp(r["exp"], r["label"], k=k) for r in run_list]
    if len(topks) < 2:
        return np.nan
    vals = [jaccard(a, b) for a, b in itertools.combinations(topks, 2)]
    return float(np.mean(vals))


def mask_superpixels(img, segments, sp_ids, hide_color=0):
    out = img.copy()
    if np.isscalar(hide_color):
        fill = np.array([hide_color, hide_color, hide_color], dtype=out.dtype)
    else:
        fill = np.array(hide_color, dtype=out.dtype).reshape(3,)
    m = np.isin(segments, np.array(sp_ids, dtype=int))
    out[m] = fill
    return out

def deletion_test(img, segments, classifier_fn, label, topk_ids, hide_color=0, k=10):
    p0 = float(classifier_fn(img[None, ...])[0, label])
    probs = [p0]
    for m in range(1, k + 1):
        img_del = mask_superpixels(img, segments, topk_ids[:m], hide_color=hide_color)
        pm = float(classifier_fn(img_del[None, ...])[0, label])
        probs.append(pm)

    probs = np.array(probs, dtype=float)
    drops = probs[0] - probs

    x = np.linspace(0, 1, k + 1)
    drop_k = float(drops[-1])

    return drop_k
