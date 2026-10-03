#!/usr/bin/env python3
"""
Category effects on brain coverage QC metrics (Welch's t-tests).

For each of the four coverage QC tasks (full brain, superior cerebrum,
inferior cerebrum, cerebellum/midbrain), runs three pairwise Welch's
t-tests comparing coverage (%) across visual QC categories:
    - full vs cropped
    - full vs minimally cropped
    - minimally cropped vs cropped

Outputs:
    - CSV of test results (n, means, SDs, t, df, p, Cohen's d) for all tasks
"""

import os

import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_DATA_PATH = "/path/to/your/input/data/input_data.csv"
OUTPUT_DIR = "/path/to/your/output/folder"

OUTPUT_CSV_NAME = "category_effects_ttests.csv"

TASKS = ["full_brain", "superior_cerebrum", "inferior_cerebrum", "cerebellum_and_midbrain"]

# Pairwise comparisons per task, as (group_a, group_b),
# reported as group_a minus group_b.
COMPARISONS = [
    ("full", "cropped"),
    ("full", "minimally cropped"),
    ("minimally cropped", "cropped"),
]


# ----------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------
def welch_df(a, b):
    """Welch-Satterthwaite degrees of freedom (computed manually for
    compatibility with older scipy versions that don't expose
    result.df on ttest_ind output)."""
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    return num / denom


def cohens_d(a, b):
    """Pooled-SD Cohen's d for two independent samples."""
    n1, n2 = len(a), len(b)
    pooled_sd = (((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)) ** 0.5
    if pooled_sd == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_sd


def run_ttests(df):
    """Run all pairwise Welch's t-tests for every task."""
    rows = []
    for task in TASKS:
        cov_col = f"coverage_{task}"
        qc_col = f"visual_qc_{task}"

        sub = df[[cov_col, qc_col]].dropna()

        for group_a, group_b in COMPARISONS:
            a = sub.loc[sub[qc_col] == group_a, cov_col]
            b = sub.loc[sub[qc_col] == group_b, cov_col]

            if len(a) < 2 or len(b) < 2:
                rows.append({
                    "task": task,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": a.mean() if len(a) else float("nan"),
                    "mean_b": b.mean() if len(b) else float("nan"),
                    "sd_a": a.std(ddof=1) if len(a) > 1 else float("nan"),
                    "sd_b": b.std(ddof=1) if len(b) > 1 else float("nan"),
                    "t_stat": float("nan"),
                    "df": float("nan"),
                    "p_value": float("nan"),
                    "cohens_d": float("nan"),
                    "note": "insufficient n (<2) in one or both groups",
                })
                continue

            result = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test

            rows.append({
                "task": task,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "sd_a": a.std(ddof=1),
                "sd_b": b.std(ddof=1),
                "t_stat": result.statistic,
                "df": welch_df(a, b),
                "p_value": result.pvalue,
                "cohens_d": cohens_d(a, b),
                "note": "",
            })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_DATA_PATH)
    results = run_ttests(df)

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} test results to {out_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()