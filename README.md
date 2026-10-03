# Brain Coverage QC Metric — Validation Analysis

Supplementary code accompanying [paper citation — TBD]. This repository
documents the analysis used to validate the automated brain coverage QC
metric implemented in
[Brain-Coverage-dseg](https://github.com/samkeppler/Brain-Coverage-dseg)
against expert manual QC ratings.

## Scripts

| Script | Analysis |
|---|---|
| `category_effects_ttests.py` | Pairwise Welch's t-tests comparing coverage (%) across manual QC categories, per region |
| `category_effects_linear_model.py` | Dataset-adjusted linear model of coverage ~ QC category, with robust (HC3) standard errors; also generates the paper's violin + forest plot figures |
| `pr_analysis.py` | Precision/recall evaluation of the automated coverage metric against manual full-brain QC ratings at predefined thresholds; generates the PR curve and confusion matrix figures |

## Sample data

The `sample_data/` folder contains **synthetic** example inputs so the scripts
can be run end to end. All values (visual QC ratings and coverage percentages)
were randomly generated to resemble realistic patterns; they are not real
participant data, and results produced from them do not reflect the findings
reported in the paper.
