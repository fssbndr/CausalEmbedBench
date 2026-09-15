"""Shared utilities for E_leaderboard.py and F_plots.py: the
metric-group registry (single source of truth for which columns belong to which
evaluation-CSV group), a couple of small dataframe helpers, and Pareto-frontier
(k, normalized error) trade-off analysis."""

import numpy as np
import pandas as pd

from utils.D_evaluation_utils import (
    BALANCE_COLS,
    CATE_COLS,
    OVERLAP_COLS,
    VALIDITY_COLS,
)

PRED_METRICS_BINARY = ["AUROC", "AUPRC", "AUROC_flex", "AUPRC_flex"]
PRED_METRICS_CONTINUOUS = ["R2", "RMSE", "R2_flex", "RMSE_flex"]


def pred_metrics_for(outcome_type: str) -> list[str]:
    return PRED_METRICS_BINARY if outcome_type == "binary" else PRED_METRICS_CONTINUOUS


# group_name -> (full_cols, summary_cols, pm_format_cols)
#   full_cols:    every column belonging to this group (CSV merge + cross-trial ranking)
#   summary_cols: subset shown in the leaderboard.md "Core metrics" table
#   pm_cols:      subset of summary_cols formatted as "mean +/- std" in markdown
# Board-computed columns (PEHE_best / PEHE_plugin_mean / PEHE_pseudo_mean / ATE_coverage_95
# / ATE_CI_width) appear in summary_cols but NOT full_cols -- they are derived in
# build_leaderboard, not read from a CSV, so listing them in full_cols would trip the
# partial-group WARNING.
METRIC_GROUPS = {
    "ate":     (["true_ATE", "ATE_hat", "ATE_error", "ATE_SE",
                 "true_ATT", "ATT_hat", "ATT_error"],
                ["ATE_error", "ATT_error", "ATE_coverage_95", "ATE_CI_width"], ["ATE_error", "ATT_error"]),
    "cate":    (CATE_COLS
                + ["PEHE", "tau_hat_mean", "tau_recovery_R2",
                   "PEHE_rselected", "PEHE_drselected",
                   "cal_alpha", "cal_beta", "ECE_CATE"],
                # tau_recovery_R2 / ECE_CATE dropped from the summary as monotone
                # re-expressions of PEHE / cal_beta; still in leaderboard.csv full_cols.
                ["PEHE_mean", "PEHE_best", "PEHE_plugin_mean", "PEHE_pseudo_mean",
                 "PEHE_mean_rank", "PEHE_mean_rank_within_dim",
                 "PEHE_rselected", "PEHE_drselected", "cal_beta"],
                ["PEHE_rselected", "PEHE_drselected"]),
    # CATE_COLS now spans every base tier (lr/poly2/gbm), so no separate cate_gbm /
    # cate_poly2 groups -- they were subsets of "cate".
    # Melnychuk Def. 1 oracle references -- emitted in pehe_cate.csv (own group so
    # a stale CSV without them is skipped, not warned).
    "validity":   (VALIDITY_COLS, ["RICB_true_mean_norm", "HET_loss_norm"], []),
    "cf":      (["CF_RMSE"], ["CF_RMSE"], ["CF_RMSE"]),
    "policy":  (["policy_value", "oracle_policy_value", "regret",
                 "regret_weighted", "policy_accuracy"],
                # policy_accuracy dropped from the summary (unweighted sign-agreement,
                # subsumed by regret_weighted); still in leaderboard.csv full_cols.
                ["regret", "regret_weighted"], ["regret", "regret_weighted"]),
    "overlap": (OVERLAP_COLS + BALANCE_COLS,
                ["ps_AUROC_delta", "cf_variance", "overlap_dinf", "decon_bias",
                 "SMD_mean_abs", "Mahalanobis_D", "Energy_dist"], []),
}

HIGHER_IS_BETTER = {"AUROC", "AUPRC", "R2", "policy_value"}


def present_cols(cols: list[str], df) -> list[str]:
    """Columns from `cols` that actually exist in `df` (or in a column/name collection)."""
    have = df.columns if hasattr(df, "columns") else df
    return [c for c in cols if c in have]


def raw_first(df: pd.DataFrame, method_col: str = "method") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (raw row(s), everything else) -- the raw baseline is always
    displayed/sorted first, with the rest ordered however the caller likes."""
    return df[df[method_col] == "raw"], df[df[method_col] != "raw"]


def pareto_frontier_mask(k: np.ndarray, error: np.ndarray) -> np.ndarray:
	"""
	Compute non-dominated mask for Pareto frontier minimizing both k and error.

	A point (k[i], error[i]) is non-dominated (on the frontier) iff there is no other
	point j with k[j] <= k[i] and error[j] <= error[i] (both strict on at least one).

	Algorithm: sort by k ascending, sweep forward keeping a running-min error;
	a point is on the frontier iff its error is strictly less than all previous points.
	O(n log n) due to sorting.

	Args:
		k: 1-D array of embedding dimensions
		error: 1-D array of (normalized) causal errors, same shape as k

	Returns:
		Boolean array, True where (k[i], error[i]) is non-dominated.
	"""
	if len(k) == 0:
		return np.array([], dtype=bool)

	# Sort by k ascending (error ascending as tiebreak for determinism)
	order = np.lexsort((error, k))
	sorted_k = k[order]
	sorted_error = error[order]

	# Sweep forward: a point is on frontier iff its error is strictly below the running min
	frontier_mask_sorted = np.zeros(len(sorted_k), dtype=bool)
	running_min_error = np.inf
	for i in range(len(sorted_k)):
		if sorted_error[i] < running_min_error:
			frontier_mask_sorted[i] = True
			running_min_error = sorted_error[i]

	# Unsort back to original order
	frontier_mask = np.zeros(len(k), dtype=bool)
	frontier_mask[order] = frontier_mask_sorted
	return frontier_mask


