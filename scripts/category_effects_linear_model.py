#!/usr/bin/env python3
"""
Category effects on brain coverage QC metrics (linear model).

For each of the four coverage QC tasks (full brain, superior cerebrum,
inferior cerebrum, cerebellum/midbrain), fits coverage ~ QC category +
dataset fixed effect with heteroscedasticity-robust (HC3) standard
errors, and computes three pairwise contrasts between visual QC
categories:
    - full vs cropped
    - full vs minimally cropped
    - minimally cropped vs cropped

Outputs:
    - CSV of dataset-adjusted contrasts for all tasks
    - Whole-brain figure (violin plot + forest plot)
    - Regional figure (violin plots + forest plots for the three sub-regions)
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_DATA_PATH = "/path/to/your/input/data/input_data.csv"
OUTPUT_DIR = "/path/to/your/output/folder"

OUTPUT_CSV_NAME = "category_effects_linear_model.csv"
WHOLE_BRAIN_FIG_NAME = "whole_brain_coverage_distribution.png"
REGIONAL_FIG_NAME = "regional_coverage_distributions.png"

TASKS = ["full_brain", "superior_cerebrum", "inferior_cerebrum", "cerebellum_and_midbrain"]

# Reference level for the QC category dummies
QC_REFERENCE = "cropped"

# Contrasts computed for every task, as (label, group_a, group_b),
# reported as group_a minus group_b.
CONTRASTS = [
    ("full vs cropped", "full", "cropped"),
    ("full vs minimally cropped", "full", "minimally cropped"),
    ("minimally cropped vs cropped", "minimally cropped", "cropped"),
]

# Contrasts shown in the forest plots. All forest panels share one
# x-axis so effect sizes are comparable across tasks.
FOREST_PLOT_CONTRASTS = ["full vs cropped", "full vs minimally cropped", "minimally cropped vs cropped"]

# Violin plot display order, left to right
VIOLIN_ORDER = ["full", "minimally cropped", "cropped"]

WHOLE_BRAIN_SPEC = {
    "title": "Full Brain Mask Coverage Values by Cropping Category",
    "group_col": "visual_qc_full_brain",
    "value_col": "coverage_full_brain",
    "task": "full_brain",
}

REGIONAL_SPECS = [
    {
        "title": "Superior Cerebrum Coverage Values by Cropping Category",
        "group_col": "visual_qc_superior_cerebrum",
        "value_col": "coverage_superior_cerebrum",
        "task": "superior_cerebrum",
    },
    {
        "title": "Inferior Cerebrum Coverage Values by Cropping Category",
        "group_col": "visual_qc_inferior_cerebrum",
        "value_col": "coverage_inferior_cerebrum",
        "task": "inferior_cerebrum",
    },
    {
        "title": "Cerebellum and Midbrain Coverage Values by Cropping Category",
        "group_col": "visual_qc_cerebellum_and_midbrain",
        "value_col": "coverage_cerebellum_and_midbrain",
        "task": "cerebellum_and_midbrain",
    },
]


# ----------------------------------------------------------------------
# MODEL FITTING
# ----------------------------------------------------------------------
def fit_ols(X, y):
    """Fit y = X @ beta via least squares (no statsmodels dependency).

    Returns beta, an HC3 (heteroscedasticity-robust "sandwich")
    coefficient covariance matrix, residual df, rank, number of
    columns, and the classical residual SD. HC3 is used for all
    SEs/CIs/p-values because residual variance differs substantially
    across QC categories; the classical residual SD is used only to
    scale standardized effect sizes.
    """
    n, p = X.shape
    XtX_pinv = np.linalg.pinv(X.T @ X)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df_resid = n - rank

    resid_sd = np.sqrt(np.sum(resid ** 2) / df_resid)

    # HC3 weights each residual by (1 - leverage)^2 instead of assuming
    # one shared variance across all observations.
    hat_diag = np.einsum("ij,jk,ik->i", X, XtX_pinv, X)
    hat_diag = np.clip(hat_diag, 0, 1 - 1e-8)  # guard against leverage == 1
    weights = (resid / (1 - hat_diag)) ** 2
    meat = X.T @ (X * weights[:, None])
    cov_beta = XtX_pinv @ meat @ XtX_pinv

    return beta, cov_beta, df_resid, rank, p, resid_sd


def contrast_test(beta, cov_beta, df_resid, c):
    """Test a linear contrast c'beta = 0 (t-test with covariance-adjusted SE)."""
    est = c @ beta
    se = np.sqrt(c @ cov_beta @ c)
    t_stat = est / se
    p_val = 2 * stats.t.sf(np.abs(t_stat), df_resid)
    t_crit = stats.t.ppf(0.975, df_resid)
    return est, se, t_stat, p_val, est - t_crit * se, est + t_crit * se


def build_design_matrix(sub):
    """QC dummies (ref = QC_REFERENCE) + dataset dummies (ref = alphabetically first) + intercept."""
    qc_dummies = pd.get_dummies(sub["qc"], prefix="qc").astype(float)
    qc_dummies = qc_dummies.drop(columns=[f"qc_{QC_REFERENCE}"])

    dataset_dummies = pd.get_dummies(sub["dataset"], prefix="ds").astype(float)
    ref_dataset_col = sorted(dataset_dummies.columns)[0]
    dataset_dummies = dataset_dummies.drop(columns=[ref_dataset_col])

    X = pd.concat(
        [pd.Series(1.0, index=sub.index, name="intercept"), qc_dummies, dataset_dummies],
        axis=1,
    )
    return X


def fit_task_model(df, task):
    """Fit the coverage ~ QC category + dataset model for one task."""
    cov_col = f"coverage_{task}"
    qc_col = f"visual_qc_{task}"

    sub = df[["dataset", cov_col, qc_col]].dropna().copy()
    sub = sub.rename(columns={cov_col: "coverage", qc_col: "qc"})

    X = build_design_matrix(sub)
    col_names = X.columns.tolist()
    y = sub["coverage"].values

    beta, cov_beta, df_resid, rank, p, resid_sd = fit_ols(X.values, y)
    if rank < p:
        print(f"WARNING [{task}]: design matrix is rank-deficient ({rank} of {p}) "
              f"-- some QC category may be fully confounded with dataset")

    return {
        "sub": sub,
        "col_names": col_names,
        "beta": beta,
        "cov_beta": cov_beta,
        "df_resid": df_resid,
        "resid_sd": resid_sd,
        "n": len(y),
        "n_datasets": sub["dataset"].nunique(),
    }


def get_contrast_vector(col_names, group_a, group_b):
    """Build a contrast vector for group_a minus group_b. Either group may
    be the reference level, which has no column (coefficient implicitly 0)."""
    p = len(col_names)
    c = np.zeros(p)
    if group_a != QC_REFERENCE:
        c[col_names.index(f"qc_{group_a}")] += 1
    if group_b != QC_REFERENCE:
        c[col_names.index(f"qc_{group_b}")] -= 1
    return c


def run_linear_models(df):
    """Fit one model per task and compute all contrasts."""
    rows = []
    fitted = {}
    for task in TASKS:
        model = fit_task_model(df, task)
        fitted[task] = model

        for label, group_a, group_b in CONTRASTS:
            c = get_contrast_vector(model["col_names"], group_a, group_b)
            est, se, t_stat, p_val, ci_low, ci_high = contrast_test(
                model["beta"], model["cov_beta"], model["df_resid"], c
            )
            rows.append({
                "task": task,
                "contrast": label,
                "n": model["n"],
                "n_datasets": model["n_datasets"],
                "adjusted_mean_diff": est,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t_stat": t_stat,
                "df_resid": model["df_resid"],
                "p_value": p_val,
                "standardized_effect_size": est / model["resid_sd"],
            })

    return pd.DataFrame(rows), fitted


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
def format_pvalue(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def get_pvalue(results_df, task, group_a, group_b):
    """Look up the p-value for a group_a vs group_b comparison from the
    linear model results (contrast label order may be either way)."""
    label_ab = f"{group_a} vs {group_b}"
    label_ba = f"{group_b} vs {group_a}"
    match = results_df[
        (results_df["task"] == task) & (results_df["contrast"].isin([label_ab, label_ba]))
    ]
    if match.empty:
        return None
    return match.iloc[0]["p_value"]


def add_significance_brackets(ax, task, present_labels, label_to_values, results_df):
    """Draw p-value brackets below the violins; returns the lowest y used."""
    positions = {g: i + 1 for i, g in enumerate(present_labels)}
    comparisons = []
    for i in range(len(present_labels)):
        for j in range(i + 1, len(present_labels)):
            g1, g2 = present_labels[i], present_labels[j]
            p = get_pvalue(results_df, task, g1, g2)
            if p is None:
                continue
            comparisons.append((positions[g1], positions[g2], p))

    if not comparisons:
        print(f"Warning: no matching p-values found for task '{task}'")
        return 25

    margin = 5
    step = 6
    tick = 1.3
    pad = 0.12

    comparisons.sort(key=lambda c: (c[1] - c[0]))  # narrow spans first
    placed = {}
    min_y = None

    for x1, x2, p in comparisons:
        spanned_labels = present_labels[x1 - 1:x2]
        local_min = min(label_to_values[lbl].min() for lbl in spanned_labels)
        y = local_min - margin

        overlapping_ys = [
            yy for (xx1, xx2), yy in placed.items() if not (xx2 <= x1 or xx1 >= x2)
        ]
        if overlapping_ys:
            y = min(y, min(overlapping_ys) - step)

        placed[(x1, x2)] = y
        span = x2 - x1
        if span == 1:
            x1d, x2d = x1 + pad, x2 - pad
        else:
            x1d, x2d = x1, x2
        ax.plot([x1d, x1d, x2d, x2d], [y + tick, y, y, y + tick], color="black", linewidth=1)
        ax.text((x1d + x2d) / 2, y - 0.4, format_pvalue(p), ha="center", va="top", fontsize=9)
        min_y = y if min_y is None else min(min_y, y)

    return min(25, min_y - 3)


def draw_violin_panel(ax, df, spec, results_df, show_ylabel=True):
    """Violin plot of raw coverage by QC category, with significance brackets."""
    group_col, value_col, task = spec["group_col"], spec["value_col"], spec["task"]

    plot_df = df[[group_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[group_col, value_col])

    data, labels = [], []
    for g in VIOLIN_ORDER:
        values = plot_df.loc[plot_df[group_col] == g, value_col].values
        if len(values) > 0:
            data.append(values)
            labels.append(g)

    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for i, values in enumerate(data, start=1):
        x = np.random.normal(i, 0.05, size=len(values))
        ax.scatter(x, values, alpha=0.4, s=10)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([label.title() for label in labels], rotation=20)
    if show_ylabel:
        ax.set_ylabel("Coverage (%)")
    ax.set_title(spec["title"])

    label_to_values = dict(zip(labels, data))
    bottom = add_significance_brackets(ax, task, labels, label_to_values, results_df)
    return bottom


def compute_shared_forest_xlim(results_df, pad_frac=0.08):
    """Shared x-axis range across all forest panels, so zero and scale
    are aligned across tasks."""
    forest_rows = results_df[results_df["contrast"].isin(FOREST_PLOT_CONTRASTS)]
    x_min = min(0, forest_rows["ci_low"].min())
    x_max = forest_rows["ci_high"].max()
    pad = (x_max - x_min) * pad_frac
    return (x_min - pad, x_max + pad)


def draw_forest_panel(ax, task, results_df, xlim):
    """Forest plot of dataset-adjusted contrasts with 95% CIs."""
    task_results = results_df[results_df["task"] == task]
    plot_rows = task_results[task_results["contrast"].isin(FOREST_PLOT_CONTRASTS)]
    plot_rows = plot_rows.set_index("contrast").loc[FOREST_PLOT_CONTRASTS].reset_index()

    y_pos = np.arange(len(plot_rows))[::-1]
    est = plot_rows["adjusted_mean_diff"].values
    ci_low = plot_rows["ci_low"].values
    ci_high = plot_rows["ci_high"].values

    ax.errorbar(
        est, y_pos, xerr=[est - ci_low, ci_high - est], fmt="o", color="black",
        ecolor="black", elinewidth=1.5, capsize=4, markersize=6,
    )
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_rows["contrast"])
    ax.set_xlabel("Adjusted mean difference in\ncoverage (%), 95% CI")
    ax.set_title("Effect size (dataset-adjusted)")
    ax.set_ylim(-0.5, len(plot_rows) - 0.5)
    ax.set_xlim(xlim)


def make_whole_brain_figure(df, results_df, forest_xlim):
    fig, (ax_violin, ax_forest) = plt.subplots(1, 2, figsize=(14, 6))
    bottom = draw_violin_panel(ax_violin, df, WHOLE_BRAIN_SPEC, results_df)
    ax_violin.set_ylim(min(25, bottom), 100)
    ax_violin.set_yticks(range(25, 101, 5))
    draw_forest_panel(ax_forest, WHOLE_BRAIN_SPEC["task"], results_df, forest_xlim)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, WHOLE_BRAIN_FIG_NAME)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_regional_figure(df, results_df, forest_xlim):
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    bottoms = []
    for row, spec in enumerate(REGIONAL_SPECS):
        bottoms.append(draw_violin_panel(axes[row, 0], df, spec, results_df))
        draw_forest_panel(axes[row, 1], spec["task"], results_df, forest_xlim)

    shared_bottom = min(25, min(bottoms))
    for row in range(len(REGIONAL_SPECS)):
        axes[row, 0].set_ylim(shared_bottom, 100)
        axes[row, 0].set_yticks(range(25, 101, 5))

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, REGIONAL_FIG_NAME)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_DATA_PATH)
    results, _ = run_linear_models(df)

    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    results.to_csv(csv_path, index=False)
    print(f"Saved {len(results)} contrast results to {csv_path}")
    print(results.to_string(index=False))

    for col in [WHOLE_BRAIN_SPEC["group_col"]] + [s["group_col"] for s in REGIONAL_SPECS]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    forest_xlim = compute_shared_forest_xlim(results)

    wb_path = make_whole_brain_figure(df, results, forest_xlim)
    print(f"Saved figure: {wb_path}")

    reg_path = make_regional_figure(df, results, forest_xlim)
    print(f"Saved figure: {reg_path}")


if __name__ == "__main__":
    main()