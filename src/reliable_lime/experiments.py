"""Single explanation runs and repeated budget sweeps.

The returned ``queries`` field preserves the notebook's configured budget.
It does not include evaluation calls or measure actual classifier predictions.
"""

import time

import numpy as np
import pandas as pd

from .coarse_to_fine import explain_instance_coarse_to_fine
from .lime_image import LimeImageExplainer
from .metrics import (
    deletion_test,
    fidelity_wmse_from_run,
    stability_jaccard,
    topk_superpixels_from_exp,
)


def clear_output(wait=False):
    """Clear progress only inside an IPython session, when available."""
    try:
        from IPython import get_ipython
        from IPython.display import clear_output as clear
    except ImportError:
        return
    if get_ipython() is not None:
        clear(wait=wait)

def run_lime_image_once(
    img,
    classifier_fn,
    segmentation_fn,
    budget,
    seed=0,
    top_labels=5,
    hide_color=0,
    num_features=10,
    positive_only=True,
):
    explainer = LimeImageExplainer(random_state=seed)
    exp = explainer.explain_instance(
        img,
        classifier_fn,
        top_labels=top_labels,
        hide_color=hide_color,
        num_samples=budget,
        segmentation_fn=segmentation_fn
    )
    label = exp.top_labels[0]
    temp, mask = exp.get_image_and_mask(
        label,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False
    )
    return {
        "budget": budget,
        "seed": seed,
        "label": label,
        "temp": temp,
        "mask": mask,
        "exp": exp,
        "score": exp.score if hasattr(exp, "score") else None,
    }


def run_lime_image_once_fixed_label(
    img,
    classifier_fn,
    segmentation_fn,
    budget,
    label_to_explain=None,
    seed=0,
    top_labels=5,
    hide_color=0,
    num_features=10,
    positive_only=True,
):
    explainer = LimeImageExplainer(random_state=seed)

    t0 = time.perf_counter()
    if label_to_explain is not None:
        try:
            exp = explainer.explain_instance(
                img,
                classifier_fn,
                labels=[label_to_explain],
                top_labels=None,
                hide_color=hide_color,
                num_samples=budget,
                segmentation_fn=segmentation_fn
            )
            label = label_to_explain
        except TypeError:
            exp = explainer.explain_instance(
                img,
                classifier_fn,
                top_labels=top_labels,
                hide_color=hide_color,
                num_samples=budget,
                segmentation_fn=segmentation_fn
            )
            label = label_to_explain if label_to_explain in exp.local_exp else exp.top_labels[0]
    else:
        exp = explainer.explain_instance(
            img,
            classifier_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=budget,
            segmentation_fn=segmentation_fn
        )
        label = exp.top_labels[0]
    t1 = time.perf_counter()

    temp, mask = exp.get_image_and_mask(
        label,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False
    )

    return {
        "budget": budget,
        "seed": seed,
        "label": label,
        "temp": temp,
        "mask": mask,
        "exp": exp,
        "score": exp.score if hasattr(exp, "score") else None,
        # Cost
        "runtime_sec": t1 - t0,
        "queries": budget,
    }


def run_lime_image_once_coarse_to_fine_fixed_label(
    img,
    classifier_fn,
    segmentation_fn,
    budget,
    label_to_explain,
    seed=0,
    hide_color=0,
    num_features=10,
    positive_only=True,
    # coarse-to-fine params
    coarse_frac=0.2,
    candidate_k=20,
    batch_size=32,
    p_on_stage1=0.5,
    p_on_stage2=0.5,
    epsilon=0.2,
    adaptive_coarse=True,
):
    t0 = time.perf_counter()

    exp = explain_instance_coarse_to_fine(
        image=img,
        classifier_fn=classifier_fn,
        segmentation_fn=segmentation_fn,
        label_to_explain=label_to_explain,
        budget=budget,
        seed=seed,
        hide_color=hide_color,
        num_features=num_features,
        coarse_frac=coarse_frac,
        candidate_k=candidate_k,
        batch_size=batch_size,
        p_on_stage1=p_on_stage1,
        p_on_stage2=p_on_stage2,
        epsilon=epsilon,
        adaptive_coarse=adaptive_coarse,
    )

    t1 = time.perf_counter()

    label = label_to_explain
    temp, mask = exp.get_image_and_mask(
        label,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False
    )

    return {
        "budget": budget,
        "seed": seed,
        "label": label,
        "temp": temp,
        "mask": mask,
        "exp": exp,
        "score": exp.score if hasattr(exp, "score") else None,
        "runtime_sec": t1 - t0,
        "queries": budget,
    }


def budget_sweep_metrics(
    run_once_fn,
    img, segments, segmentation_fn, classifier_fn,
    budgets, seeds, ref_label,
    k=10, hide_color=0, num_features=10, positive_only=True,
    run_once_kwargs=None
):
    if run_once_kwargs is None:
        run_once_kwargs = {}
    rows = []
    summary_rows = []

    total = len(budgets) * len(seeds)
    run_idx = 0

    for budget_idx,B in enumerate(budgets):
        runs_B = []
        for seed_idx, s in enumerate(seeds):
            run_idx += 1
            clear_output(wait=True)
            print(f"\033[1m[{budget_idx+1}/{len(budgets)}] Budget={B}\033[0m")
            print(f"[{seed_idx+1}/{len(seeds)}] budget={B}, seed={s}")
            r = run_once_fn(
                img, classifier_fn, segmentation_fn,
                budget=B, label_to_explain=ref_label,
                seed=s, hide_color=hide_color,
                num_features=num_features, positive_only=positive_only,
                **run_once_kwargs
            )
            runs_B.append(r)

            fid_wmse = fidelity_wmse_from_run(r, ref_label)
            topk = topk_superpixels_from_exp(r["exp"], r["label"], k=k)
            drop_k = deletion_test(
                img, segments, classifier_fn,
                label=r["label"], topk_ids=topk,
                hide_color=hide_color, k=k
            )

            rows.append({
                "budget": B,
                "seed": s,
                "label": r["label"],
                "stability_jaccard": np.nan,
                "fidelity_wmse": fid_wmse,
                "faith_drop_k": drop_k,
                "runtime_sec": r["runtime_sec"],
                "queries": r["queries"],
            })
        stab = stability_jaccard(runs_B, k=k)
        for rr in rows[-len(seeds):]:
            rr["stability_jaccard"] = stab

        dfB = pd.DataFrame(rows[-len(seeds):])
        summary_rows.append({
            "budget": B,
            "stability_jaccard": stab,
            "fidelity_wmse_mean": float(dfB["fidelity_wmse"].mean()),
            "faith_drop_k_mean": float(dfB["faith_drop_k"].mean()),
            "runtime_sec_mean": float(dfB["runtime_sec"].mean()),
            "queries": B
        })

    df_runs = pd.DataFrame(rows)
    df_summary = pd.DataFrame(summary_rows).sort_values("budget")
    return df_runs, df_summary
