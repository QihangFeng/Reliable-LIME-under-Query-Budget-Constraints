"""Explanation overlays and budget comparison figures from the notebook."""

import math

import matplotlib.pyplot as plt
from torchvision import models

from .experiments import run_lime_image_once

def plot_budget_sweep(
    img,
    classifier_fn,
    budgets,
    segmentation_fn,
    seed=0,
    num_features=10,
    positive_only=True,
    title_prefix="LIME",
    show=True,
):
    categories = None
    try:
        categories = models.ResNet18_Weights.DEFAULT.meta["categories"]
    except Exception:
        pass
    results = []
    n = len(budgets)
    ncols = 3
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(4*ncols, 4*nrows))
    for i, B in enumerate(budgets, 1):
        print(f"[{i}/{len(budgets)}] budget={B}, seed={seed} ...", end="")
        r = run_lime_image_once(
            img, classifier_fn, segmentation_fn,
            budget=B, seed=seed, num_features=num_features,
            positive_only=positive_only
        )
        results.append(r)
        label = r["label"]
        name = categories[label] if categories is not None else str(label)

        plt.subplot(nrows, ncols, i)
        plt.imshow(r["temp"])
        plt.imshow(r["mask"], alpha=0.4)
        plt.axis("off")
        plt.title(f"{title_prefix}, N={B}\n{label} {name}")
    plt.tight_layout()
    if show:
        plt.show()
    return results


def plot_budget_sweep_generic(
    run_once_fn,
    img, classifier_fn, segmentation_fn,
    budgets, ref_label,
    seed=0,
    k=10, hide_color=0, num_features=10, positive_only=True,
    ncols=3,
    title_prefix="LIME",
    run_once_kwargs=None,
    show=True,
):
    if run_once_kwargs is None:
        run_once_kwargs = {}

    results = []
    for B in budgets:
        r = run_once_fn(
            img, classifier_fn, segmentation_fn,
            budget=B, label_to_explain=ref_label,
            seed=seed, hide_color=hide_color,
            num_features=num_features, positive_only=positive_only,
            **run_once_kwargs
        )
        results.append(r)

    nrows = math.ceil(len(budgets) / ncols)
    plt.figure(figsize=(4*ncols, 4*nrows))

    for i, r in enumerate(results):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(r["temp"])
        ax.imshow(r["mask"], alpha=0.4)
        ax.axis("off")
        ax.set_title(f"{title_prefix}, N={r['budget']}\nlabel={r['label']}")

    plt.tight_layout()
    if show:
        plt.show()

    return results
