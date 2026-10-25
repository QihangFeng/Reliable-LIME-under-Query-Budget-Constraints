"""Run the notebook's budget experiments and save tables and figures."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("vanilla", "coarse-to-fine", "both"), default="both"
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[200, 500, 1000, 2000, 3000, 5000]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n-segments", type=int, default=100)
    parser.add_argument("--compactness", type=float, default=10)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--coarse-frac", type=float, default=0.2)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Coarse-to-fine classifier batch size; vanilla keeps notebook batches of 10.",
    )
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument(
        "--fixed-coarse-frac", action="store_true",
        help="Use --coarse-frac instead of the notebook's adaptive coarse fraction.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Save metric tables only; skip additional runs for explanation plots.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: a new UTC timestamped directory under results/runs/.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if any(budget < 20 for budget in args.budgets):
        parser.error("budgets must be at least 20 for the existing two-stage sampler")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("seeds must be between 0 and 2**32 - 1")
    if min(args.n_segments, args.num_features, args.candidate_k, args.batch_size) < 1:
        parser.error("segment, feature, candidate, and batch counts must be positive")
    if not 0 < args.coarse_frac < 1:
        parser.error("--coarse-frac must be between 0 and 1")
    if not 0 <= args.epsilon <= 0.9:
        parser.error("--epsilon must be between 0 and 0.9")
    if args.compactness <= 0:
        parser.error("--compactness must be positive")
    if len(set(args.budgets)) != len(args.budgets):
        parser.error("budgets must be unique")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reliable_lime.classifier import ResNet18Classifier, load_example_image
    from reliable_lime.experiments import (
        budget_sweep_metrics,
        run_lime_image_once_coarse_to_fine_fixed_label,
        run_lime_image_once_fixed_label,
    )
    from reliable_lime.metrics import get_ref_label
    from reliable_lime.segmentation import make_fixed_segmentation_fn
    from reliable_lime.visualization import plot_budget_sweep_generic

    classifier = ResNet18Classifier(None if args.device == "auto" else args.device)
    image = load_example_image()
    segments, segmentation_fn = make_fixed_segmentation_fn(
        image, n_segments=args.n_segments, compactness=args.compactness
    )
    ref_label, probabilities = get_ref_label(image, classifier)
    output_dir = args.output_dir or (
        Path("results/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_kwargs = {
        "coarse_frac": args.coarse_frac,
        "candidate_k": args.candidate_k,
        "batch_size": args.batch_size,
        "epsilon": args.epsilon,
        "adaptive_coarse": not args.fixed_coarse_frac,
    }
    settings = vars(args).copy()
    settings.update({
        "output_dir": str(output_dir.resolve()),
        "resolved_device": str(classifier.device),
        "image": "skimage.data.cat",
        "reference_label": ref_label,
        "reference_category": classifier.categories[ref_label],
        "reference_probability": float(probabilities[ref_label]),
        "coarse_to_fine": coarse_kwargs,
        "versions": {
            name: version(name) for name in (
                "reliable-lime", "numpy", "scipy", "scikit-learn", "scikit-image",
                "torch", "torchvision",
            )
        },
        "query_accounting": (
            "queries is the configured sampling budget, excluding reference-label, "
            "deletion-evaluation, and plotting calls. The preserved coarse-to-fine "
            "sampler predicts budget-1 images and fits budget-2 rows."
        ),
    })
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )

    methods = [
        ("vanilla", run_lime_image_once_fixed_label, {}),
        ("coarse-to-fine", run_lime_image_once_coarse_to_fine_fixed_label, coarse_kwargs),
    ]
    for name, run_once, kwargs in methods:
        if args.method not in ("both", name):
            continue
        runs, summary = budget_sweep_metrics(
            run_once, image, segments, segmentation_fn, classifier,
            args.budgets, args.seeds, ref_label,
            k=args.num_features, num_features=args.num_features,
            run_once_kwargs=kwargs,
        )
        runs.to_csv(output_dir / (name + "_runs.csv"), index=False)
        summary.to_csv(output_dir / (name + "_summary.csv"), index=False)
        summary.to_html(output_dir / (name + "_summary.html"), index=False)
        if not args.no_plots:
            plot_budget_sweep_generic(
                run_once, image, classifier, segmentation_fn,
                args.budgets, ref_label, seed=args.seeds[0],
                k=args.num_features, num_features=args.num_features,
                title_prefix=name, run_once_kwargs=kwargs, show=False,
            )
            plt.gcf().savefig(output_dir / (name + "_explanations.png"), dpi=150)
            plt.close()
        print("\n{}\n{}".format(name, summary.to_string(index=False)))
    print("\nSaved results to {}".format(output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
