#!/usr/bin/env python3
"""
Precision/recall of brain coverage metrics at predefined thresholds.

Evaluates how well two coverage metrics (full brain mask coverage and
minimum regional mask coverage) classify visually assessed full-brain
DWI cropping, predicting "full" when coverage >= threshold at each of
several predefined thresholds. Run for two classification tasks
(positive class = "full" in both):
    - Full vs. Cropped (minimally cropped rows excluded)
    - (Full + Minimally Cropped) vs. Cropped

Outputs:
    - CSV of precision, recall, and confusion counts per task/metric/threshold
    - PR curve figure per task (full brain mask, with average precision)
    - Confusion matrix grid figure per task (metrics x thresholds)
"""

import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_DATA_PATH = "/path/to/your/input/data/input_data.csv"
OUTPUT_DIR = "/path/to/your/output/folder"

OUTPUT_CSV_NAME = "predefined_threshold_results.csv"
PR_CURVE_FIG_TEMPLATE = "pr_curve_{task}.png"
CONFUSION_MATRIX_FIG_TEMPLATE = "confusion_matrices_{task}.png"

# Coverage thresholds (%) to evaluate
THRESHOLDS = [98, 95, 92]

# Full-brain visual QC label column
LABEL_COL = "visual_qc_full_brain"

# Coverage metrics, in plotting/output order
METRIC_ORDER = [
    "full_brain_mask",
    "regional_masks",
]

# Metric -> (preferred column, fallback column)
COVERAGE_COLS = {
    "full_brain_mask": ("coverage_full_brain_mask", None),
    "regional_masks": ("min_coverage_regional_masks", None),
}

METRIC_DISPLAY_NAMES = {
    "full_brain_mask": "Full Brain Mask",
    "regional_masks": "Regional Masks",
}

# Classification tasks as (task code, display description)
TASKS = [
    ("full_vs_cropped_excl_minimal", "Full vs. Cropped"),
    ("full_vs_cropped_minimal_as_full", "(Full + Minimally Cropped) vs. Cropped"),
]


# ----------------------------------------------------------------------
# CLASSIFICATION
# ----------------------------------------------------------------------
def parse_visual_label(x) -> str:
    """Map free-text visual QC values to one of:
    cropped, minimally_cropped, full, unknown."""
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
    """Convert 3-way labels to a binary outcome for a given task.
    Positive class (1) is always "full" (grouped with minimally_cropped
    for the minimal_as_full task). Returns NaN where the label is
    unknown or excluded for that task."""
    y = np.full(len(labels), np.nan)

    if task == "full_vs_cropped_excl_minimal":
        # minimally_cropped left as NaN, i.e. dropped
        y[labels == "full"] = 1
        y[labels == "cropped"] = 0

    elif task == "full_vs_cropped_minimal_as_full":
        y[(labels == "full") | (labels == "minimally_cropped")] = 1
        y[labels == "cropped"] = 0

    else:
        raise ValueError(f"Unknown task: {task}")

    return y


def resolve_col(df: pd.DataFrame, preferred: Optional[str], fallback: Optional[str]) -> Optional[str]:
    """Return the preferred column if present, else the fallback, else None."""
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
    """Predict positive ("full") if coverage >= threshold. Returns
    precision/recall/counts, or None if there isn't at least one
    positive and one negative case after filtering."""
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


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
def plot_pr_curve_for_task(
    task_desc: str,
    metric_curves: List[Tuple[str, np.ndarray, np.ndarray]],
    average_precisions: Dict[str, float],
    out_png: str
) -> None:
    """Plot the full PR curve for one task, with average precision (AP)
    annotated.

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
    """Plot a grid of 2x2 confusion matrices (rows = metrics,
    cols = thresholds). grid_results[i][j] is the output of
    classification_at_threshold for metric i, threshold j (or None)."""
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


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    output_csv = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)

    if not os.path.exists(INPUT_DATA_PATH):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_DATA_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    for col in ["dataset", "participant_id", LABEL_COL]:
        if col not in df.columns:
            raise KeyError(f"Input CSV is missing required column: {col}")

    start = datetime.now()
    print(f"Started at {start}")
    print(f"Input CSV:   {INPUT_DATA_PATH}")
    print(f"Output CSV:  {output_csv}")
    print(f"Thresholds:  {THRESHOLDS}")
    print(f"Rows:        {df.shape[0]}  columns: {df.shape[1]}")
    print(f"Datasets:    {df['dataset'].nunique()}")

    labels = df[LABEL_COL].apply(parse_visual_label).values

    # Threshold results table
    rows: List[Dict[str, Any]] = []

    for task_code, task_desc in TASKS:
        y = make_binary(labels, task_code)

        for metric in METRIC_ORDER:
            cov_pref, cov_fallback = COVERAGE_COLS[metric]
            cov_col = resolve_col(df, cov_pref, cov_fallback)

            if cov_col is None:
                print(f"[WARN] Skipping {metric} (missing columns). coverage={cov_pref}/{cov_fallback}")
                continue

            coverage_raw = pd.to_numeric(df[cov_col], errors="coerce").values

            for threshold in THRESHOLDS:
                res = classification_at_threshold(y, coverage_raw, threshold)

                if res is None:
                    print(f"[WARN] No valid classification for {task_code} / {metric} @ {threshold} "
                          f"(need >=1 positive and >=1 negative with finite coverage).")
                    row = {
                        "analysis": task_code,
                        "analysis_description": task_desc,
                        "metric": metric,
                        "metric_display_name": METRIC_DISPLAY_NAMES.get(metric, metric),
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
                    "metric_display_name": METRIC_DISPLAY_NAMES.get(metric, metric),
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

    # Figures per task: PR curve (full_brain_mask only) + confusion
    # matrix grid (all metrics x all thresholds)
    for task_code, task_desc in TASKS:
        y = make_binary(labels, task_code)

        metric_curves = []
        average_precisions: Dict[str, float] = {}
        grid_results: List[List[Optional[Dict[str, Any]]]] = []
        metric_display_names = []

        for metric in METRIC_ORDER:
            cov_pref, cov_fallback = COVERAGE_COLS[metric]
            cov_col = resolve_col(df, cov_pref, cov_fallback)
            if cov_col is None:
                continue

            coverage_raw = pd.to_numeric(df[cov_col], errors="coerce").values
            display_name = METRIC_DISPLAY_NAMES.get(metric, metric)

            # Full PR curve (sweeps all thresholds), full_brain_mask only
            if metric == "full_brain_mask":
                m = np.isfinite(y) & np.isfinite(coverage_raw)
                y_valid = y[m].astype(int)
                score = coverage_raw[m].astype(float)  # higher coverage -> more likely "full"

                if len(y_valid) > 0 and len(np.unique(y_valid)) == 2:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_valid, score)
                    metric_curves.append((display_name, recall_curve, precision_curve))
                    average_precisions[display_name] = average_precision_score(y_valid, score)

            # Confusion matrices at the predefined thresholds, all metrics
            row_results = []
            for threshold in THRESHOLDS:
                res = classification_at_threshold(y, coverage_raw, threshold)
                row_results.append(res)

            grid_results.append(row_results)
            metric_display_names.append(display_name)

        pr_png = os.path.join(OUTPUT_DIR, PR_CURVE_FIG_TEMPLATE.format(task=task_code))
        plot_pr_curve_for_task(task_desc, metric_curves, average_precisions, pr_png)

        cm_png = os.path.join(OUTPUT_DIR, CONFUSION_MATRIX_FIG_TEMPLATE.format(task=task_code))
        plot_confusion_matrix_grid(
            task_desc, grid_results, metric_display_names, THRESHOLDS, cm_png
        )

    print(f"Total runtime: {datetime.now() - start}")


if __name__ == "__main__":
    main()