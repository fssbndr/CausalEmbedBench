# Aggregate all evaluation CSVs into a leaderboard with mean ± std across seeds.
# Produces ID (full-cohort) and OOD (eICU+Amsterdam) leaderboards, then compares.
#
# Reads:  data/evaluations/{trial}[/knob_variants/<preset>]/{prediction,ate_preservation,
#                                   pehe_cate,counterfactual,policy_value,propensity_overlap}.csv
#         data/evaluations_ood/{trial}[/knob_variants/<preset>]/  (same structure)
# Writes: data/evaluations/{trial}[/knob_variants/<preset>]/leaderboard.csv / leaderboard.md
#         data/evaluations_ood/{trial}[/knob_variants/<preset>]/leaderboard.csv / leaderboard.md
#         data/evaluations/{trial}[/knob_variants/<preset>]/leaderboard_ood_comparison.md
#
# leaderboard.csv also carries a per-(meta-learner x base-learner) oracle rank
# (PEHE_{learner}_{base}_rank, non-raw rows only) alongside the pooled PEHE_mean_rank.
#
# --knobs-preset selects which oracle-knob variant to read/write (default: baseline).

import argparse
import functools
import json
import operator
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "B_oracles", "shared"))
from knobs import resolve_out_dir
from utils.C_embedding_utils import parse_embedding_name
from utils.D_evaluation_utils import (
    BASES,
    CATE_COLS,
    LEARNERS,
    PLUGIN_LEARNERS,
    PSEUDO_LEARNERS,
)
from utils.leaderboard_utils import (
    METRIC_GROUPS,
    pareto_frontier_mask,
    pred_metrics_for,
    present_cols,
    raw_first,
)
from utils.plot_utils import _method_sort_key, display_trial_name

# --- metric-column config for build_leaderboard --------------------------------
# family -> (learner subset | None = all learners, row-wise agg); applied per base learner
PEHE_FAMILIES = {
    "PEHE_mean": (None, "mean"), "PEHE_best": (None, "min"),
    "PEHE_plugin_mean": (PLUGIN_LEARNERS, "mean"),
    "PEHE_pseudo_mean": (PSEUDO_LEARNERS, "mean"),
}
# Comparison to the raw baseline: {m}_dlog = log m - log m_ref, ref = raw_tuned
# (GBM-nuisance raw) where present else plain raw (variance-stabilized; 0 = tuned-raw
# parity). Strictly-positive error metrics only.
DLOG_METRICS = ["PEHE_mean", "PEHE_best", "PEHE_plugin_mean", "PEHE_pseudo_mean",
                "PEHE_rselected", "PEHE_drselected", "ATE_error", "CF_RMSE", "ECE_CATE"]
OOD_KEY_METRICS = ["ATE_error", "PEHE_mean", "PEHE_mean_rank_within_dim", "CF_RMSE", "regret"]

parser = argparse.ArgumentParser()
parser.add_argument("trial")
parser.add_argument("--knobs-preset", default="baseline",
                     help="Named oracle-knob preset the evaluations were generated with "
                          "(default: baseline).")
args = parser.parse_args()
TRIAL = args.trial
KNOBS_PRESET = args.knobs_preset
ORACLE_DIR = resolve_out_dir(TRIAL, KNOBS_PRESET, base="data/oracles")
EVAL_DIR = resolve_out_dir(TRIAL, KNOBS_PRESET, base="data/evaluations")
EVAL_OOD_DIR = resolve_out_dir(TRIAL, KNOBS_PRESET, base="data/evaluations_ood")

# Read outcome_type and x_varies_by_seed flag from causal params
with open(f"{ORACLE_DIR}/causal_params.json") as f:
    cp = json.load(f)
OUTCOME_TYPE = cp.get("outcome_type", "binary")
X_VARIES_BY_SEED = cp.get("x_varies_by_seed", False)


# ------------------------------------------------------------------------------
# Helper: build a leaderboard from one evaluation directory
# ------------------------------------------------------------------------------
def build_leaderboard(eval_dir: str, trial: str, outcome_type: str = "binary", knobs_preset: str = "baseline",
                      has_individual_cf: bool = True, x_varies_by_seed: bool = False, distribution: str = "ID") -> tuple[pd.DataFrame, int]:
    """Return (board, n_seeds, metrics) for one evaluation directory. `board` is stamped
    with its own identity (trial/preset/distribution/...) so leaderboard.csv is
    self-describing -- downstream consumers don't path-parse."""
    # Trial covariate dim d (raw's dim=None -> d fill-in) and sample size n (n-regime facet).
    oracle_dir = resolve_out_dir(trial, knobs_preset, base="data/oracles")
    with open(f"{oracle_dir}/feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(f"{oracle_dir}/causal_params.json") as f:
        cp_trial = json.load(f)

    d_trial = len(feature_cols)
    n_trial = cp_trial.get("n")

    d = Path(eval_dir)
    pred   = pd.read_csv(d / "prediction.csv")
    ate    = pd.read_csv(d / "ate_preservation.csv")
    cate   = pd.read_csv(d / "pehe_cate.csv")
    cf     = pd.read_csv(d / "counterfactual.csv")
    policy = pd.read_csv(d / "policy_value.csv")
    ovl    = pd.read_csv(d / "propensity_overlap.csv")

    KEY = ["embedding", "seed"]

    # Outcome metrics (linear-probe pair always present; _flex pair only on fresh CSVs)
    pred_metrics = present_cols(pred_metrics_for(outcome_type), pred)
    per_seed = pred[["embedding", "seed"] + pred_metrics]

    # Optional groups: include whatever subset of a group's columns is present (JOBS-style
    # trials lack the individual-CF groups; a stale CSV may lack some columns).
    sources = {"ate": ate, "cate": cate,
               "validity": cate, "cf": cf, "policy": policy, "overlap": ovl}
    metrics = {}
    for name, (cols, _, _) in METRIC_GROUPS.items():
        df = sources[name]
        present = [c for c in cols if c in df.columns]
        if present and len(present) < len(cols):
            print(f"   WARNING: {name}.csv has {len(present)}/{len(cols)} expected columns "
                  f"(missing {sorted(set(cols) - set(present))}) -- including only the "
                  "present subset. Stale evaluation CSV? Rerun D_evaluation.py.")
        metrics[name] = present
        if metrics[name]:
            per_seed = per_seed.merge(df[["embedding", "seed"] + present], on=KEY)

    FLOAT_COLS = pred_metrics + functools.reduce(operator.iadd, metrics.values(), [])

    # tau_sd normalises the board-derived PEHE families (below); merged explicitly so it
    # stays out of METRIC_GROUPS / CANDIDATE_METRICS / summaries.
    if "tau_sd" in cate.columns:
        per_seed = per_seed.merge(cate[["embedding", "seed", "tau_sd"]], on=KEY)
        FLOAT_COLS = FLOAT_COLS + ["tau_sd"]

    # ATE interval coverage across seeds from the AIPW ATE_SE: coverage_95 = fraction of
    # seeds whose 95% Wald band covers the true ATE; CI_width = mean band width.
    ate_cov = None
    if {"ATE_hat", "true_ATE", "ATE_SE"}.issubset(per_seed.columns):
        _hit = (np.abs(per_seed["ATE_hat"] - per_seed["true_ATE"]) <= 1.96 * per_seed["ATE_SE"]).astype(float)
        ate_cov = pd.DataFrame({
            "embedding": per_seed["embedding"].values,
            "ATE_coverage_95": _hit.values,
            "ATE_CI_width": (2 * 1.96 * per_seed["ATE_SE"]).values,
        }).groupby("embedding").mean()

    agg_mean = per_seed.groupby("embedding")[FLOAT_COLS].mean()
    agg_std  = per_seed.groupby("embedding")[FLOAT_COLS].std(ddof=1)

    board = agg_mean.reset_index()
    for col in FLOAT_COLS:
        board[f"{col}_std"] = agg_std[col].values

    if ate_cov is not None:
        board["ATE_coverage_95"] = board["embedding"].map(ate_cov["ATE_coverage_95"]).round(4)
        board["ATE_CI_width"]    = board["embedding"].map(ate_cov["ATE_CI_width"]).round(4)

    def _parse_name(name: str):
        return parse_embedding_name(name) or (name, None, None)

    board[["method", "dim", "rho"]] = pd.DataFrame(
        board["embedding"].map(_parse_name).tolist(), index=board.index
    )

    raw_row, rest = raw_first(board)
    rest = rest.copy()
    rest["_method_order"] = rest["method"].map(lambda m: _method_sort_key(m)[0])
    rest = rest.sort_values(["_method_order", "dim"]).drop(columns="_method_order")
    board = pd.concat([raw_row, rest], ignore_index=True)

    num_cols = FLOAT_COLS + [f"{c}_std" for c in FLOAT_COLS]
    board[num_cols] = board[num_cols].round(4)

    # PEHE families. Bare names (PEHE_mean, ...) aggregate over ALL base tiers -- the
    # global mean that the plots and E_diagnostic_validity read; each tier also gets a
    # _{base} companion. pehe_cols_all is the full grid used for the global ranking.
    pehe_cols_all = present_cols(CATE_COLS, board)
    pehe_family_cols = []
    for fam, (subset, agg) in PEHE_FAMILIES.items():
        cols = (pehe_cols_all if subset is None
                else [f"PEHE_{l}_{b}" for l in subset for b in BASES if f"PEHE_{l}_{b}" in board.columns])
        board[fam] = getattr(board[cols], agg)(axis=1).round(4) if cols else np.nan
    for bl in BASES:
        tier_cols = present_cols([f"PEHE_{l}_{bl}" for l in LEARNERS], board)
        if not tier_cols:
            continue
        for fam, (subset, agg) in PEHE_FAMILIES.items():
            cols = tier_cols if subset is None else [f"PEHE_{l}_{bl}" for l in subset if f"PEHE_{l}_{bl}" in board.columns]
            board[f"{fam}_{bl}"] = getattr(board[cols], agg)(axis=1).round(4) if cols else np.nan
            pehe_family_cols.append(f"{fam}_{bl}")
    board["PEHE_mean_rank"] = (
        board[pehe_cols_all].rank(axis=0, ascending=True).mean(axis=1).round(2)
        if pehe_cols_all else np.nan
    )
    board["PEHE_mean_rank_within_dim"] = np.nan

    # raw and raw_tuned are references, not ranked candidates.
    candidate = ~board["method"].isin(["raw", "raw_tuned"])
    if pehe_cols_all:
        rank_cols = [f"{c}_rank" for c in pehe_cols_all]
        board.loc[candidate, rank_cols] = board.loc[candidate, pehe_cols_all].rank(axis=0, ascending=True)

        # Same as PEHE_mean_rank, but grouped by dim -- controls for capacity.
        board.loc[candidate, "PEHE_mean_rank_within_dim"] = (
            board.loc[candidate].groupby("dim")[pehe_cols_all].rank(ascending=True).mean(axis=1).round(2)
        )
        for col in ("PEHE_best", "PEHE_plugin_mean", "PEHE_pseudo_mean"):
            board.loc[candidate, f"{col}_rank"] = board.loc[candidate, col].rank(ascending=True)

    # raw / raw_tuned: dim -> d_trial (Pareto x-position), rho -> 1.0.
    board.loc[board["method"].isin(["raw", "raw_tuned"]), ["dim", "rho"]] = d_trial, 1.0
    board["rho"] = board["rho"].round(4)

    # Comparison to the raw baseline. dlog reference is raw_tuned (GBM-nuisance raw) so
    # "0 = parity" means parity with a fairly-fit raw baseline, not an untuned one; falls
    # back to plain raw where no raw_tuned row exists (OOD leaderboards).
    raw_mask = board["method"] == "raw"
    raw_pehe = float(board.loc[raw_mask, "PEHE_mean"].iloc[0])
    ref_mask = board["method"] == "raw_tuned"
    if not ref_mask.any():
        ref_mask = raw_mask

    dl_cols = [m for m in DLOG_METRICS + pehe_family_cols if m in board.columns]
    lg = np.log(board[dl_cols].where(board[dl_cols] > 0))
    board[[f"{m}_dlog" for m in dl_cols]] = lg.sub(lg.loc[ref_mask].iloc[0]).round(4).to_numpy()

    # Scale-free PEHE families for the cross-trial capacity plots: PEHE / SD(tau^x).
    if "tau_sd" in board.columns:
        _sd = board["tau_sd"].where(board["tau_sd"] > 0)
        for fam in ("PEHE_mean", "PEHE_plugin_mean", "PEHE_pseudo_mean"):
            if fam in board.columns:
                board[f"{fam}_norm"] = (board[fam] / _sd).round(4)

    # Pareto frontier on dlog PEHE (raw_tuned at dlog=0, and excluded from the frontier;
    # plain raw sits at a positive dlog). pareto_frontier_mask returns all-False on an
    # all-NaN column, so no missing-metric special case is needed.
    frontier_metric = ("PEHE_mean_dlog"
                       if board.get("PEHE_mean_dlog", pd.Series(dtype=float)).notna().any()
                       else "PEHE_mean")
    board["frontier_metric"] = frontier_metric
    board["on_frontier"] = (
        pareto_frontier_mask(board["dim"].values, board[frontier_metric].values)
        & (board["method"] != "raw_tuned").values
    )

    # baseline_slack: PEHE(raw) - best PEHE with tuned (poly2/GBM) nuisances on the same
    # raw features. Large => raw baseline is nuisance-limited, not representation-inferior.
    rt = cate[cate["embedding"] == "raw_tuned"] if "embedding" in cate.columns else cate.iloc[0:0]
    if len(rt):
        _tuned_suffixes = tuple(f"_{b}" for b in BASES)
        tuned_cols = [c for c in rt.columns if c.startswith("PEHE_") and c.endswith(_tuned_suffixes)]
        tuned_best = float(np.nanmin(rt[tuned_cols].to_numpy(dtype=float)))
        board["baseline_slack"] = round(raw_pehe - tuned_best, 4)
        slack_frac = (raw_pehe - tuned_best) / raw_pehe
        if slack_frac > 0.15:
            print(f"   WARNING: raw_tuned (GBM nuisances) beats raw PEHE by {slack_frac:.1%} "
                  "-- raw baseline may be nuisance-limited.")

    # Self-describing identity (see docstring).
    board["trial"] = trial
    board["preset"] = knobs_preset
    board["outcome_type"] = outcome_type
    board["n_features"] = d_trial
    board["n"] = n_trial
    board["has_individual_cf"] = has_individual_cf
    board["distribution"] = distribution

    return board, len(per_seed["seed"].unique()), metrics


# Each optional group's summary display: (group name, [display columns], [pm()-formatted columns])
DISPLAY_GROUPS = [(name, summary_cols, pm_cols)
                   for name, (_, summary_cols, pm_cols) in METRIC_GROUPS.items()]


def write_leaderboard(board: pd.DataFrame, n_seeds: int, out_dir: str, outcome_type: str = "binary", metrics: dict | None = None) -> None:
    """Write leaderboard.csv and leaderboard.md into out_dir."""
    metrics = metrics or {}
    d = Path(out_dir)
    pred_metrics = pred_metrics_for(outcome_type)

    summary_cols = ["embedding", "dim"] + pred_metrics
    for name, cols, _ in DISPLAY_GROUPS:
        if metrics.get(name):
            summary_cols += cols

    available_cols = [c for c in summary_cols if c in board.columns]
    print(board[available_cols].to_string(index=False))
    board.to_csv(d / "leaderboard.csv", index=False)

    def pm(row, col):
        val = row.get(col)
        return "nan" if pd.isna(val) else f"{val:.4f} +/- {row.get(col+'_std', np.nan):.4f}"

    core_md_rows, pehe_md_rows = [], []
    for _, row in board.iterrows():
        dim_str = str(int(row["dim"])) if pd.notna(row.get("dim")) else "raw"
        entry = {"embedding": row["embedding"], "dim": dim_str}
        entry.update({m: pm(row, m) for m in pred_metrics if m in row.index})

        for name, cols, pm_cols in DISPLAY_GROUPS:
            if not metrics.get(name):
                continue
            for col in cols:
                entry[col] = pm(row, col) if col in pm_cols else row.get(col, np.nan)

        core_md_rows.append(entry)

        if metrics.get("cate"):
            pehe_md_rows.append({"embedding": row["embedding"], "dim": dim_str,
                                 **{col: pm(row, col) for col in CATE_COLS if col in board.columns}})

    with open(d / "leaderboard.md", "w") as f:
        f.write("# Causal Embedding Benchmark Leaderboard\n\n")
        f.write(f"*{n_seeds} oracle seeds; values = mean +/- std*\n\n")
        f.write("## Core metrics\n\n")
        f.write(pd.DataFrame(core_md_rows).to_markdown(index=False))
        f.write("\n\n*Headline causal metrics: PEHE_rselected / PEHE_drselected (R-/DR-loss-selected "
                "learner x base, two-level; lower is better). PEHE_best / PEHE_mean / PEHE_plugin_mean / "
                "PEHE_pseudo_mean are companions. Comparison to raw is Delta-log PEHE (`*_dlog`, "
                "0 = tuned-raw parity). regret / regret_weighted: lower better; "
                "cal_beta -> 1; ATE_coverage_95 -> 0.95 (under per-Phi nuisance refit it also "
                "reflects nuisance fit on Z, not only coverage of the truth).*\n\n")
        f.write("## PEHE by meta-learner × base learner\n\n")
        f.write("*Columns: PEHE\\_{learner}\\_{base}  "
                "(S/Lo/T/X/RA/Z/F/U/R/DR × lr/poly2/gbm added via `--bases`; "
                "lower is better)*\n\n")
        f.write(pd.DataFrame(pehe_md_rows).to_markdown(index=False))
        f.write("\n")

    print(f"\nWritten {d}/leaderboard.csv and leaderboard.md")


# ------------------------------------------------------------------------------
# ID leaderboard
# ------------------------------------------------------------------------------
print(f"=== In-distribution leaderboard ({display_trial_name(TRIAL)}, full cohort) ===")
id_board, id_n_seeds, id_metrics = build_leaderboard(
    EVAL_DIR, trial=TRIAL, outcome_type=OUTCOME_TYPE, knobs_preset=KNOBS_PRESET,
    has_individual_cf=cp.get("has_individual_cf", True), x_varies_by_seed=X_VARIES_BY_SEED, distribution="ID")
write_leaderboard(id_board, id_n_seeds, EVAL_DIR, outcome_type=OUTCOME_TYPE, metrics=id_metrics)

# ------------------------------------------------------------------------------
# OOD leaderboard (if available for this trial)
# ------------------------------------------------------------------------------
ood_csv = Path(f"{EVAL_OOD_DIR}/prediction.csv")
if ood_csv.exists():
    print(f"\n=== OOD leaderboard ({display_trial_name(TRIAL)}) ===")
    ood_board, ood_n_seeds, ood_metrics = build_leaderboard(
        EVAL_OOD_DIR, trial=TRIAL, outcome_type=OUTCOME_TYPE, knobs_preset=KNOBS_PRESET,
        has_individual_cf=cp.get("has_individual_cf", True), x_varies_by_seed=X_VARIES_BY_SEED, distribution="OOD")
    write_leaderboard(ood_board, ood_n_seeds, EVAL_OOD_DIR, outcome_type=OUTCOME_TYPE, metrics=ood_metrics)

    # ── ID vs OOD comparison ──────────────────────────────────────────────────────
    KEY_METRICS = pred_metrics_for(OUTCOME_TYPE) + OOD_KEY_METRICS

    id_sub  = id_board [["embedding"] + KEY_METRICS].set_index("embedding")
    ood_sub = ood_board[["embedding"] + KEY_METRICS].set_index("embedding")

    comp = id_sub.copy()
    for m in KEY_METRICS:
        comp[f"{m}_ood"]   = ood_sub[m]
        comp[f"{m}_delta"] = (ood_sub[m] - id_sub[m]).round(4)
    comp = comp.reset_index()

    raw_row, rest_ = raw_first(comp, method_col="embedding")
    comp = pd.concat([raw_row, rest_.sort_values("PEHE_mean")], ignore_index=True)

    err_note = "For AUROC: negative delta = OOD worse" if OUTCOME_TYPE == "binary" else "For R2: negative delta = OOD worse"
    print(f"\n=== ID vs OOD comparison (delta = OOD − ID; positive = OOD worse for error metrics; {err_note}) ===")
    print(comp[["embedding"] + [f"{m}_delta" for m in KEY_METRICS]].to_string(index=False))

    rows = []
    for _, row in comp.iterrows():
        entry = {"embedding": row["embedding"]}
        for m in KEY_METRICS:
            entry[f"{m}_id"]    = f"{row[m]:.4f}"
            entry[f"{m}_ood"]   = f"{row[f'{m}_ood']:.4f}"
            entry[f"{m}_delta"] = f"{row[f'{m}_delta']:+.4f}"
        rows.append(entry)

    with open(f"{EVAL_DIR}/leaderboard_ood_comparison.md", "w") as f:
        f.write("# ID vs OOD Comparison\n\n")
        if OUTCOME_TYPE == "binary":
            f.write("*delta = OOD − ID. For error metrics (ATE_error, PEHE_mean, CF_RMSE, regret): "
                    "positive delta = worse in OOD. For AUROC/AUPRC: negative delta = worse in OOD.*\n\n")
        else:
            f.write("*delta = OOD − ID. For error metrics (ATE_error, PEHE_mean, CF_RMSE, regret): "
                    "positive delta = worse in OOD. For R2/RMSE: positive delta (R2) or negative delta (RMSE) = worse in OOD.*\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False))
        f.write("\n")

    print(f"\nWritten {EVAL_DIR}/leaderboard_ood_comparison.md")
