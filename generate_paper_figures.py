"""
Generate IEEE-paper-ready figures (ROC/PR and tuned-threshold confusion matrix)
from saved prediction artifacts in versions/<version>/predictions/*.npz.

Usage:
  python generate_paper_figures.py --version v1.2.2
  python generate_paper_figures.py --version v1.2.2 --model Ensemble_Stacking
"""

import argparse
import glob
import os

import numpy as np

import config
from utils import (
    plot_roc_pr_curves,
    plot_confusion_matrix_at_threshold,
    tune_threshold_by_f1,
)


def _load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        default=config.CURRENT_VERSION,
        help="Version folder under ./versions (default: config.CURRENT_VERSION)",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=None,
        help="Generate from multiple versions (e.g. --versions v1.1 v1.2 v1.2.2).",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Scan ./versions/* for prediction artifacts and plot them together.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for tuned confusion matrix (default: best by F1 on validation).",
    )
    args = parser.parse_args()

    if args.all_versions:
        version_names = sorted(
            d
            for d in os.listdir(config.VERSIONS_DIR)
            if os.path.isdir(os.path.join(config.VERSIONS_DIR, d))
        )
    elif args.versions:
        version_names = args.versions
    else:
        version_names = [args.version]

    curve_items = []
    model_infos = []

    for version_name in version_names:
        version_dir = os.path.join(config.VERSIONS_DIR, version_name)
        predictions_dir = os.path.join(version_dir, "predictions")
        npz_files = sorted(glob.glob(os.path.join(predictions_dir, "*.npz")))
        for path in npz_files:
            blob = _load_npz(path)
            model_name = str(blob.get("model_name", os.path.splitext(os.path.basename(path))[0]))
            y_test = blob["y_test"]
            y_test_score = blob["y_test_score"]
            curve_items.append(
                {"label": f"{version_name}/{model_name}", "y_true": y_test, "y_score": y_test_score}
            )

            y_valid = blob.get("y_valid", None)
            y_valid_score = blob.get("y_valid_score", None)
            tuned_threshold = float(blob.get("tuned_threshold", np.nan))
            tuned_f1_valid = blob.get("tuned_threshold_f1_valid", None)
            if tuned_f1_valid is None and y_valid is not None and y_valid_score is not None:
                tuned_f1_valid = tune_threshold_by_f1(y_valid, y_valid_score)["f1"]

            model_infos.append(
                {
                    "version": version_name,
                    "model_name": model_name,
                    "label": f"{version_name}/{model_name}",
                    "path": path,
                    "tuned_threshold": tuned_threshold,
                    "tuned_f1_valid": float(tuned_f1_valid) if tuned_f1_valid is not None else float("nan"),
                }
            )

    if not model_infos:
        raise SystemExit(
            "No prediction artifacts found.\n"
            "Run training first (train.py) to generate versions/<version>/predictions/*.npz."
        )

    if len(version_names) == 1:
        plots_dir = os.path.join(config.VERSIONS_DIR, version_names[0], "plots")
    else:
        plots_dir = os.path.join(config.RESULTS_DIR, "paper_figures")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) ROC + PR curves (side-by-side)
    roc_pr_path = os.path.join(plots_dir, "roc_pr_curves_all_models.png")
    title_suffix = f" ({version_names[0]})" if len(version_names) == 1 else " (all versions)"
    plot_roc_pr_curves(curve_items, roc_pr_path, title_suffix=title_suffix)

    # 2) Tuned-threshold confusion matrix (best model by validation F1, or user-specified)
    if args.model:
        candidates = [m for m in model_infos if m["model_name"] == args.model]
        if not candidates:
            available = ", ".join(sorted({m["model_name"] for m in model_infos}))
            raise SystemExit(f"Model not found: {args.model}\nAvailable: {available}")
        selected = max(candidates, key=lambda m: m["tuned_f1_valid"])
    else:
        selected = max(model_infos, key=lambda m: m["tuned_f1_valid"])

    selected_blob = _load_npz(selected["path"])
    threshold = float(selected_blob.get("tuned_threshold", np.nan))
    if not np.isfinite(threshold):
        y_valid = selected_blob.get("y_valid")
        y_valid_score = selected_blob.get("y_valid_score")
        if y_valid is None or y_valid_score is None:
            threshold = 0.5
        else:
            threshold = tune_threshold_by_f1(y_valid, y_valid_score)["threshold"]

    cm_path = os.path.join(plots_dir, "confusion_matrix_tuned_best.png")
    plot_confusion_matrix_at_threshold(
        selected_blob["y_test"],
        selected_blob["y_test_score"],
        threshold,
        selected["label"],
        cm_path,
    )

    print("\n✅ Paper figures generated")
    print(f" - {roc_pr_path}")
    print(f" - {cm_path}")


if __name__ == "__main__":
    main()
