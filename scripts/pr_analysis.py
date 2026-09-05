#!/usr/bin/env python3
# =============================================================================
# Purpose: Evaluate classification performance (precision/recall) of two
#          alternative brain coverage metrics against visually assessed
#          full-brain DWI cropping, using predefined coverage thresholds.
#
#          Metrics compared:
#            1) full_brain_mask coverage
#            2) regional_masks (min of sub-region masks) coverage
#
#          Classification rule: predict "full" if coverage >= threshold,
#          evaluated at each of several predefined thresholds.
#
#          Tasks (positive class = "full" in both):
#            1) full_vs_cropped_excl_minimal
#               - minimally_cropped rows dropped entirely
#               - full = 1, cropped = 0
#            2) full_vs_cropped_minimal_as_full
#               - minimally_cropped grouped WITH full
#               - full & minimally_cropped = 1, cropped = 0
#
# Created on 03/04/2026 by Samantha Keppler
# =============================================================================

import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "base_path": "/your/base/path",

    # Input file (combined CSV with full-brain visual QC and comparison metrics)
    "input_csv": "{base}/input_files/pr_analysis_data.csv",

    # Predefined coverage thresholds to evaluate (same scale as the coverage
    # columns, which are 0-100 percentages, not 0-1 fractions).
    "thresholds": [98, 95, 92],

    # Full-brain visual QC label column
    "label_col": "visual_qc_full_brain",

    # Coverage metrics to compare in order
    "metric_order": [
        "full_brain_mask",
        "regional_masks",
    ],

    # Coverage metric column mappings
    "coverage_cols": {
        "full_brain_mask": ("coverage_full_brain_mask", None),
        "regional_masks": ("min_coverage_regional_masks", None),
    },

    # Pretty labels for plots / output readability
    "metric_display_names": {
        "full_brain_mask": "Full Brain Mask",
        "regional_masks": "Regional Masks",
    },

    # Classification tasks (positive class = "full" in every case)
    "tasks": [
        ("full_vs_cropped_excl_minimal", "Full vs. Cropped"),
        ("full_vs_cropped_minimal_as_full", "(Full + Minimally Cropped) vs. Cropped"),
    ],
}


def build_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    base = out["base_path"]

    out["output_dir"] = f"{base}/results/predefined_thresholds"
    out["output_csv"] = f"{out['output_dir']}/predefined_threshold_results.csv"
    out["output_pr_png_template"] = f"{out['output_dir']}/pr_curve_{{task}}.png"
    out["output_cm_png_template"] = f"{out['output_dir']}/confusion_matrices_{{task}}.png"

    return out


# =============================================================================
# HELPERS
# =============================================================================

def parse_visual_label(x) -> str:
    """
    Map free-text visual QC values to one of:
      cropped, minimally_cropped, full, unknown
    """
    if pd.isna(x):
        return "unknown"

    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)

    if "minimally cropped" in s:
        return "minimally_cropped"
    if s.startswith("cropped") or (" cropped" in s and "minimally" not in s):
        return "cropped"
    if "full" in s:
        return "full"

    return "unknown"


def make_binary(labels: np.ndarray, task: str) -> np.ndarray:
    """
    Convert 3-way labels into binary outcome for a given task.
    Positive class (1) is always "full" (possibly grouped with
    minimally_cropped, depending on task). Returns NaN where the
    label is unknown/unusable for that task.
    """
    y = np.full(len(labels), np.nan)

    if task == "full_vs_cropped_excl_minimal":
        # minimally_cropped dropped entirely
        y[labels == "full"] = 1
        y[labels == "cropped"] = 0

    elif task == "full_vs_cropped_minimal_as_full":
        # minimally_cropped grouped with full
        y[(labels == "full") | (labels == "minimally_cropped")] = 1
        y[labels == "cropped"] = 0

    else:
        raise ValueError(f"Unknown task: {task}")

    return y


def resolve_col(df: pd.DataFrame, preferred: Optional[str], fallback: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    if fallback and fallback in df.columns:
        return fallback
    return None


def classification_at_threshold(
    y_true: np.ndarray,
    coverage: np.ndarray,
    threshold: float
) -> Optional[Dict[str, Any]]:
    """
    Predict positive ("full") if coverage >= threshold.
    Returns precision/recall/counts, or None if there aren't at least
    one positive and one negative case after filtering.
    """
    m = np.isfinite(y_true) & np.isfinite(coverage)

    y = y_true[m].astype(int)
    cov = coverage[m].astype(float)

    if len(y) == 0 or len(np.unique(y)) < 2:
        return None

    y_pred = (cov >= threshold).astype(int)

    tp = int(np.sum((y == 1) & (y_pred == 1)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_total": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "n_predicted_positive": int(y_pred.sum()),
    }


def plot_pr_curve_for_task(
    task_desc: str,
    metric_curves: List[Tuple[str, np.ndarray, np.ndarray]],
    average_precisions: Dict[str, float],
    out_png: str
) -> None:
    """
    Plot the full PR curve for a single task (full_brain_mask only),
    with average precision (AP) annotated on the plot.

    metric_curves: list of (display_name, recall_array, precision_array)
    average_precisions: dict of display_name -> average precision score
    """
    if not metric_curves:
        print(f"[WARN] No PR curve to plot for {task_desc}")
        return

    fig, ax = plt.subplots(figsize=(8, 6.5))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ap_lines = []
    for i, (display_name, recall, precision) in enumerate(metric_curves):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(recall, precision, label=display_name, color=color, alpha=0.8)

        ap = average_precisions.get(display_name)
        if ap is not None:
            ap_lines.append(f"Average Precision = {ap:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve: {task_desc}", fontsize=11, wrap=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize="small")

    if ap_lines:
        ax.text(
            0.98, 0.02, "\n".join(ap_lines),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved PR curve figure: {out_png}")


def plot_confusion_matrix_grid(
    task_desc: str,
    grid_results: List[List[Optional[Dict[str, Any]]]],
    metric_display_names: List[str],
    thresholds: List[float],
    out_png: str
) -> None:
    """
    Plot a grid of 2x2 confusion matrices: rows = metrics, cols = thresholds.

    grid_results[i][j] corresponds to metric i, threshold j, and is the
    dict returned by classification_at_threshold (or None if unavailable).
    """
    n_rows = len(metric_display_names)
    n_cols = len(thresholds)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for i, display_name in enumerate(metric_display_names):
        for j, threshold in enumerate(thresholds):
            ax = axes[i][j]
            res = grid_results[i][j]

            if res is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                cm = np.array([[res["tp"], res["fn"]],
                                [res["fp"], res["tn"]]])
                ax.imshow(cm, cmap="Blues")

                labels = [["TP", "FN"], ["FP", "TN"]]
                for r in range(2):
                    for c in range(2):
                        ax.text(
                            c, r, f"{labels[r][c]}\n{cm[r, c]}",
                            ha="center", va="center", fontsize=9
                        )

                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Predicted Full", "Predicted Cropped"], fontsize=7)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Actual Full", "Actual Cropped"], fontsize=7, rotation=90, va="center")

                prec = res["precision"]
                rec = res["recall"]
                prec_str = f"{prec:.2f}" if np.isfinite(prec) else "NA"
                rec_str = f"{rec:.2f}" if np.isfinite(rec) else "NA"
                ax.set_title(f"P={prec_str}  R={rec_str}", fontsize=8)

            if i == 0:
                ax.annotate(
                    f"Threshold = {threshold / 100:.2f}",
                    xy=(0.5, 1.25), xycoords="axes fraction",
                    ha="center", fontsize=9, fontweight="bold"
                )
            if j == 0:
                ax.annotate(
                    display_name,
                    xy=(-0.45, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=9, fontweight="bold", rotation=90
                )

    fig.suptitle(f"Confusion Matrices: {task_desc}", fontsize=11, wrap=True)
    fig.tight_layout(rect=[0.03, 0, 1, 0.93])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved confusion matrix figure: {out_png}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = build_runtime_config(CONFIG)

    base = cfg["base_path"]
    input_csv = cfg["input_csv"].format(base=base)
    output_dir = cfg["output_dir"]
    output_csv = cfg["output_csv"]

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]

    for col in ["dataset", "participant_id", cfg["label_col"]]:
        if col not in df.columns:
            raise KeyError(f"Input CSV is missing required column: {col}")

    start = datetime.now()
    print(f"Started at {start}")
    print(f"Input CSV:   {input_csv}")
    print(f"Output CSV:  {output_csv}")
    print(f"Thresholds:  {cfg['thresholds']}")
    print(f"Rows:        {df.shape[0]}  columns: {df.shape[1]}")
    print(f"Datasets:    {df['dataset'].nunique()}")

    labels = df[cfg["label_col"]].apply(parse_visual_label).values

    rows: List[Dict[str, Any]] = []

    for task_code, task_desc in cfg["tasks"]:
        y = make_binary(labels, task_code)

        for metric in cfg["metric_order"]:
            cov_pref, cov_fallback = cfg["coverage_cols"][metric]
            cov_col = resolve_col(df, cov_pref, cov_fallback)

            if cov_col is None:
                print(f"[WARN] Skipping {metric} (missing columns). coverage={cov_pref}/{cov_fallback}")
                continue

            coverage_raw = pd.to_numeric(df[cov_col], errors="coerce").values

            for threshold in cfg["thresholds"]:
                res = classification_at_threshold(y, coverage_raw, threshold)

                if res is None:
                    print(f"[WARN] No valid classification for {task_code} / {metric} @ {threshold} "
                          f"(need >=1 positive and >=1 negative with finite coverage).")
                    row = {
                        "analysis": task_code,
                        "analysis_description": task_desc,
                        "metric": metric,
                        "metric_display_name": cfg["metric_display_names"].get(metric, metric),
                        "coverage_column": cov_col,
                        "threshold": threshold,
                        "precision": np.nan,
                        "recall": np.nan,
                        "tp": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "tn": np.nan,
                        "n_total": np.nan,
                        "n_positive": np.nan,
                        "n_negative": np.nan,
                        "n_predicted_positive": np.nan,
                    }
                    rows.append(row)
                    continue

                row = {
                    "analysis": task_code,
                    "analysis_description": task_desc,
                    "metric": metric,
                    "metric_display_name": cfg["metric_display_names"].get(metric, metric),
                    "coverage_column": cov_col,
                    "threshold": threshold,
                    "precision": round(float(res["precision"]), 4) if np.isfinite(res["precision"]) else np.nan,
                    "recall": round(float(res["recall"]), 4) if np.isfinite(res["recall"]) else np.nan,
                    "tp": res["tp"],
                    "fp": res["fp"],
                    "fn": res["fn"],
                    "tn": res["tn"],
                    "n_total": res["n_total"],
                    "n_positive": res["n_positive"],
                    "n_negative": res["n_negative"],
                    "n_predicted_positive": res["n_predicted_positive"],
                }
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved predefined-threshold results to: {output_csv}")

    # -------------------------------------------------------------------
    # Visualizations: one PR curve (full_brain_mask only) + one confusion
    # matrix grid (all metrics x all thresholds), per task
    # -------------------------------------------------------------------
    for task_code, task_desc in cfg["tasks"]:
        y = make_binary(labels, task_code)

        metric_curves = []
        average_precisions: Dict[str, float] = {}
        grid_results: List[List[Optional[Dict[str, Any]]]] = []
        metric_display_names = []

        for metric in cfg["metric_order"]:
            cov_pref, cov_fallback = cfg["coverage_cols"][metric]
            cov_col = resolve_col(df, cov_pref, cov_fallback)
            if cov_col is None:
                continue

            coverage_raw = pd.to_numeric(df[cov_col], errors="coerce").values
            display_name = cfg["metric_display_names"].get(metric, metric)

            # full PR curve (sweep all thresholds) : full_brain_mask only 
            if metric == "full_brain_mask":
                m = np.isfinite(y) & np.isfinite(coverage_raw)
                y_valid = y[m].astype(int)
                score = coverage_raw[m].astype(float)  # higher coverage -> more likely "full"

                if len(y_valid) > 0 and len(np.unique(y_valid)) == 2:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_valid, score)
                    metric_curves.append((display_name, recall_curve, precision_curve))
                    average_precisions[display_name] = average_precision_score(y_valid, score)

            # confusion matrices at the predefined thresholds (all metrics) 
            row_results = []
            for threshold in cfg["thresholds"]:
                res = classification_at_threshold(y, coverage_raw, threshold)
                row_results.append(res)

            grid_results.append(row_results)
            metric_display_names.append(display_name)

        pr_png = cfg["output_pr_png_template"].format(task=task_code)
        plot_pr_curve_for_task(task_desc, metric_curves, average_precisions, pr_png)

        cm_png = cfg["output_cm_png_template"].format(task=task_code)
        plot_confusion_matrix_grid(
            task_desc, grid_results, metric_display_names, cfg["thresholds"], cm_png
        )

    print(f"Total runtime: {datetime.now() - start}")


if __name__ == "__main__":
    main()
