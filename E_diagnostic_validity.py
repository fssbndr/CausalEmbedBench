# Diagnostic validity: (1) does a diagnostic correlate with PEHE degradation (row-level,
# pooled), and (2) selection regret -- does *acting* on a diagnostic's argmin beat guessing,
# block by block? Stats only, no plotting -- see F_plots.py.
#
# Reads (1): data/evaluations[_ood]/{trial}[/knob_variants/{preset}]/{pehe_cate,propensity_overlap}.csv (per-seed)
# Reads (2): data/evaluations[_ood]/{trial}[/knob_variants/{preset}]/leaderboard.csv (seed-aggregated; run E_leaderboard.py first)
# Writes: output/diagnostic_validity.{csv,md}, output/selection_regret.{csv,md,_nemenyi.csv}

import os
import sys
from pathlib import Path

import autorank
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import bootstrap as scipy_bootstrap
from scipy.stats import kendalltau, rankdata, spearmanr, wilcoxon
from sklearn.utils.parallel import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.C_embedding_utils import parse_embedding_name
from utils.D_evaluation_utils import BASES, LEARNERS
from utils.leaderboard_utils import present_cols
from utils.plot_utils import METHOD_FAMILY

os.makedirs("output", exist_ok=True)

# ps_AUROC, SMD_mean_abs, decon_bias dropped as weak/uninformative selectors (see
# selection_regret.md); still computed in propensity_overlap.csv. No CATE-derived
# oracle-free signal is analysed: raw R-/DR-loss(Z) carry Var(Y|Z).
# Four families: balance-in-Z, overlap/tails, outcome content, representation integrity
# (ps_AUROC_delta = e(Z) vs e(X) treatment predictability).
DIAGNOSTIC_COLS = [
    "SMD_max_abs", "Mahalanobis_D", "pred_score",
    "ESS_trt_frac", "ESS_ctrl_frac", "frac_extreme",
    "cf_variance", "overlap_dinf",
    "ps_AUROC_delta",
]
TARGET = "PEHE_dlog"
# Oracle-only references (Melnychuk et al. 2024 Def. 1: RICB / loss of heterogeneity).
# Need true_tau, so not deployable selectors -- correlated against PEHE_dlog as a stratum.
ORACLE_REF_COLS = ["RICB_true_rmse", "HET_loss_rmse"]

# Plot/table labels for the raw column names above -- these are cheap by-products of fitting
# a propensity model e(Z) on the embedding (utils/D_evaluation_utils.py:13-23), checking
# whether Z=Phi(X) still supports valid causal adjustment. Three families:
#   Balance (SMD_max_abs, Mahalanobis_D)  -- do treated/control still differ systematically on Z?
#   Overlap (ESS_*_frac, frac_extreme)    -- is IPW on e(Z) numerically stable (no near-deterministic T)?
#   Outcome (pred_score)                  -- does Z retain Y-relevant signal, independent of confounding structure?
DIAGNOSTIC_LABELS = {
    "SMD_max_abs":     "SMD, max (balance)",
    "Mahalanobis_D":   "Mahalanobis dist. (balance)",
    "RICB_true_rmse":  "RICB, RMSE (oracle ref -- Melnychuk Def. 1 (ii))",
    "HET_loss_rmse":   "Heterogeneity loss, RMSE (oracle ref -- Melnychuk Def. 1 (i))",
    "pred_score":      "Outcome prediction (Task 1: AUROC/R2)",
    "ESS_trt_frac":    "Effective N, treated (overlap)",
    "ESS_ctrl_frac":   "Effective N, control (overlap)",
    "frac_extreme":    "Extreme-weight fraction (overlap)",
    "decon_bias":      "Deconfounding bias, |.| (D'Amour & Franks 2021)",
    "cf_variance":     "Counterfactual variance (Zhang, Bellot & van der Schaar 2020)",
    "overlap_dinf":    "Overlap divergence D_inf (Zhang, Bellot & van der Schaar 2020)",
    "ps_AUROC_delta":  "Treatment predictability, e(Z) vs e(X) (Neubrander et al. 2026)",
}

DIAGNOSTIC_CAPTION = (
    "Balance (SMD max, Mahalanobis): do treated/control still differ systematically on the embedding?\n"
    "Overlap (Effective N, extreme-weight frac., D_inf, counterfactual variance): is IPW on e(Z) numerically stable?\n"
    "Outcome (pred_score): does Z predict Y at all (Task 1 AUROC for binary trials, R2 for continuous)?\n"
    "Representation integrity (ps_AUROC_delta): does e(Z) encode more treatment information than e(X)?\n"
    "See this module's DIAGNOSTIC_LABELS for column mapping."
)


def _parse_eval_path(path: Path) -> tuple[str, str, str]:
    """(trial, preset, distribution) from a pehe_cate.csv path, following knobs.py's
    resolve_out_dir convention: data/evaluations[_ood]/{trial}[/knob_variants/{preset}]/."""
    parts = path.parts
    data_idx = parts.index("data")
    root = parts[data_idx + 1]  # "evaluations" or "evaluations_ood"
    distribution = "OOD" if root == "evaluations_ood" else "ID"
    trial = parts[data_idx + 2]
    is_variant = len(parts) > data_idx + 3 and parts[data_idx + 3] == "knob_variants"
    preset = parts[data_idx + 4] if is_variant else "baseline"
    return trial, preset, distribution


def _pred_score(pred: pd.DataFrame) -> pd.Series | None:
    """Unify AUROC (binary trials) and R^2 (continuous trials) into one column -- a given
    trial only ever populates one of the two, so this is a straight pick, not a merge."""
    auroc = pred["AUROC"] if "AUROC" in pred.columns else None
    r2 = pred["R2"] if "R2" in pred.columns else None
    if auroc is not None and r2 is not None:
        return auroc.combine_first(r2)
    return auroc if auroc is not None else r2


def load_diagnostic_pehe_table() -> pd.DataFrame:
    """Long-format table: one row per (trial, preset, distribution, embedding, seed), with
    PEHE/PEHE_dlog and every diagnostic column. Excludes raw rows from the returned table
    (they're the normalization reference, not embeddings under evaluation)."""
    rows = []
    for cate_path in sorted(Path("data").glob("evaluations*/**/pehe_cate.csv")):
        ovl_path = cate_path.parent / "propensity_overlap.csv"
        pred_path = cate_path.parent / "prediction.csv"
        if not ovl_path.exists():
            print(f"   WARNING: no propensity_overlap.csv next to {cate_path}, skipping")
            continue
        cate = pd.read_csv(cate_path)
        ovl = pd.read_csv(ovl_path)
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            pred["pred_score"] = _pred_score(pred)
            ovl = ovl.merge(pred[["embedding", "seed", "pred_score"]], on=["embedding", "seed"], how="left")
        diag_cols = [c for c in DIAGNOSTIC_COLS + ORACLE_REF_COLS if c in ovl.columns]
        merged = cate.merge(ovl[["embedding", "seed"] + diag_cols], on=["embedding", "seed"])

        trial, preset, distribution = _parse_eval_path(cate_path)
        merged["trial"], merged["preset"], merged["distribution"] = trial, preset, distribution
        rows.append(merged)

    if not rows:
        raise FileNotFoundError(
            "No pehe_cate.csv found under data/evaluations*/. Run D_evaluation.py first."
        )

    df = pd.concat(rows, ignore_index=True)
    df[["method", "dim"]] = pd.DataFrame(
        df["embedding"].map(lambda name: (parse_embedding_name(name) or (name, None))[:2]).tolist(),
        index=df.index,
    )
    df["method_family"] = df["method"].map(lambda m: METHOD_FAMILY.get(m, "causal"))

    # PEHE basis for dlog: the mean over the meta-learner grid at the logistic/ridge base
    # (poly2/gbm coded but not run) -- the same aggregate E_leaderboard writes as the bare
    # PEHE_mean -- rather than the bare PEHE (= PEHE_T_lr) column. Falls back to PEHE where
    # the grid columns are absent.
    grid_cols = present_cols([f"PEHE_{l}_lr" for l in LEARNERS], df)
    df["PEHE_global"] = df[grid_cols].mean(axis=1) if grid_cols else df["PEHE"]

    # Per-trial/preset/distribution/seed PEHE normalization: relative inflation from
    # compression, not raw scale, so pooling across oracle families is meaningful. Reference
    # is raw_tuned (tuned-nuisance raw) where present else plain raw -- one shared choice
    # with E_leaderboard's *_dlog, so "0 = parity" means a fairly-fit raw baseline.
    keys = ["trial", "preset", "distribution", "seed"]
    raw = df[df["method"] == "raw"][keys + ["PEHE_global"]].rename(columns={"PEHE_global": "PEHE_raw"})
    tuned = (df[df["method"] == "raw_tuned"][keys + ["PEHE_global"]]
             .rename(columns={"PEHE_global": "PEHE_tuned"}))
    df = df.merge(raw, on=keys, how="left").merge(tuned, on=keys, how="left")
    df["PEHE_ref"] = df["PEHE_tuned"].fillna(df["PEHE_raw"])
    # Variance-stabilized target: Delta-log PEHE (global-mean basis) vs the tuned-raw
    # baseline (0 = tuned-raw parity).
    with np.errstate(divide="ignore", invalid="ignore"):
        df["PEHE_dlog"] = np.where(
            (df["PEHE_global"] > 0) & (df["PEHE_ref"] > 0),
            np.log(df["PEHE_global"]) - np.log(df["PEHE_ref"]),
            np.nan,
        )

    return df[~df["method"].isin(["raw", "raw_tuned"])].reset_index(drop=True)


def _trial_index_groups(df: pd.DataFrame, trial_col: str = "trial") -> dict:
    """{trial_name: row-index array} -- resampling unit for the bootstrap helpers below."""
    return {t: df.index[df[trial_col] == t].to_numpy() for t in df[trial_col].unique()}


def _bootstrap_ci_multi(sub: pd.DataFrame, diagnostic: str, statfns: dict, target: str = TARGET,
                         n_boot: int = 1000, seed: int = 42) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI for multiple statistics on shared resamples, at the trial
    level (not row level, to avoid pseudo-replication). Single-trial input short-circuits
    to a point estimate instead of 1000 identical iterations."""
    x, y = sub[diagnostic].to_numpy(), sub[target].to_numpy()
    trial_labels = sub["trial"].to_numpy()
    trials = np.unique(trial_labels)
    if len(trials) < 2:
        if x.min() == x.max() or y.min() == y.max():
            return {name: (np.nan, np.nan) for name in statfns}
        return {name: (float(fn(x, y)),) * 2 for name, fn in statfns.items()}

    trial_positions = {t: np.flatnonzero(trial_labels == t) for t in trials}
    rng = np.random.default_rng(seed)
    stats = {name: [] for name in statfns}
    for _ in range(n_boot):
        sampled_trials = rng.choice(trials, size=len(trials), replace=True)
        pos = np.concatenate([trial_positions[t] for t in sampled_trials])
        rx, ry = x[pos], y[pos]
        if rx.min() == rx.max() or ry.min() == ry.max():
            continue
        for name, fn in statfns.items():
            stats[name].append(fn(rx, ry))
    return {
        name: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)
        for name, vals in stats.items()
    }


def _spearman_kendall_one(df: pd.DataFrame, diagnostic: str, target: str = TARGET) -> dict | None:
    sub = df.dropna(subset=[diagnostic, target])
    if len(sub) < 2:
        return None
    rho, _ = spearmanr(sub[diagnostic], sub[target])
    tau, _ = kendalltau(sub[diagnostic], sub[target])
    cis = _bootstrap_ci_multi(sub, diagnostic, {
        "rho": lambda a, b: spearmanr(a, b)[0], "tau": lambda a, b: kendalltau(a, b)[0],
    }, target=target)
    rho_lo, rho_hi = cis["rho"]
    tau_lo, tau_hi = cis["tau"]
    return {
        "diagnostic": diagnostic, "target": target, "n": len(sub),
        "spearman_rho": rho, "spearman_ci_lo": rho_lo, "spearman_ci_hi": rho_hi,
        "kendall_tau": tau, "kendall_ci_lo": tau_lo, "kendall_ci_hi": tau_hi,
    }


def spearman_kendall_table(df: pd.DataFrame, diagnostics: list[str] = DIAGNOSTIC_COLS) -> pd.DataFrame:
    """Per-diagnostic rank correlation against PEHE_dlog, with trial-level bootstrap CIs.
    Parallelized over diagnostics -- safe since callers never wrap this in their own Parallel."""
    rows = Parallel(n_jobs=-1)(delayed(_spearman_kendall_one)(df, d) for d in diagnostics)
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False).reset_index(drop=True)


def diagnostic_correlation_matrix(df: pd.DataFrame, diagnostics: list[str] = DIAGNOSTIC_COLS) -> pd.DataFrame:
    """Pairwise Spearman among the diagnostic *values* (not vs PEHE): the "selectors carry
    ~2-3 effective signals, not 10" redundancy check (appendix app:results-detail)."""
    present = present_cols(diagnostics, df)
    return df[present].corr(method="spearman").round(3)


def write_outputs(strata_results: dict[str, pd.DataFrame], corr_matrix: pd.DataFrame | None = None) -> None:
    csv_rows = []
    for stratum, corr_df in strata_results.items():
        c = corr_df.copy(); c["stratum"], c["table"] = stratum, "correlation"
        csv_rows.append(c)
    pd.concat(csv_rows, ignore_index=True).to_csv("output/diagnostic_validity.csv", index=False)
    print("   Saved output/diagnostic_validity.csv")

    with open("output/diagnostic_validity.md", "w") as f:
        f.write("# Diagnostic-to-PEHE Validity\n\n")
        f.write("Which propensity diagnostics predict CATE-estimation degradation "
                "(PEHE_dlog = log PEHE - log raw-features PEHE; 0 = raw parity)?\n\n")

        f.write("## Diagnostic legend\n\n")
        legend_df = pd.DataFrame([
            {"column": col, "label": DIAGNOSTIC_LABELS.get(col, col)} for col in DIAGNOSTIC_COLS
        ])
        f.write(legend_df.to_markdown(index=False))
        f.write(f"\n\n*{DIAGNOSTIC_CAPTION}*\n\n")

        if corr_matrix is not None:
            f.write("## Pairwise diagnostic correlation (Spearman, stratum=all)\n\n")
            f.write(corr_matrix.to_markdown())
            f.write("\n\n")

        for stratum, corr_df in strata_results.items():
            corr_display = corr_df.copy()
            corr_display.insert(1, "label", corr_display["diagnostic"].map(lambda c: DIAGNOSTIC_LABELS.get(c, c)))
            f.write(f"## {stratum}\n\n")
            f.write("### Rank correlation vs. PEHE_dlog (trial-level bootstrap 95% CI)\n\n")
            f.write(corr_display.to_markdown(index=False))
            f.write("\n\n")
    print("   Saved output/diagnostic_validity.md")


# Block-level selection-regret analysis. `rho` (relative capacity k/d) is in the block key
# so a diagnostic is scored on picking a method at a fixed capacity, not on jointly
# choosing method + capacity. rho comes straight from the embedding grid (RELATIVE_RATIOS,
# a 7-point set), carried through leaderboard.csv -- no binning.
EPS_REGRET_DENOM = 1e-6  # near-tied candidates -> norm_regret undefined, not 0
MIN_BLOCK_SIZE = 3       # fewer candidates than this -> tau-b/argmin too noisy
BLOCK_KEY = ["trial", "preset", "distribution", "learner", "base", "rho"]
_RAW_KEY = ["trial", "preset", "distribution", "learner", "base"]  # raw has no capacity axis

# Datasets whose treatment assignment is NOT a fitted logistic of X (TWINS: prior-drawn
# logistic; IHDP/NEWS: published closed-form). Used for an independence-check stratum that
# is not congenial to the logistic-propensity diagnostics.
NONFITTED_ASSIGNMENT_TRIALS = {"twins", "ihdp", "news"}

# Sign to make "lower is better". 0 = needs a transform (ps_AUROC: |x - 0.5|), not a flip.
# Keeps the analysis-dropped by-products too -- still read from propensity_overlap.csv.
DIAGNOSTIC_DIRECTION = {
    "SMD_mean_abs": +1, "SMD_max_abs": +1, "Mahalanobis_D": +1,
    "ps_AUROC": 0,
    "pred_score": -1,
    "ESS_trt_frac": -1, "ESS_ctrl_frac": -1, "frac_extreme": +1,
    "decon_bias": +1, "cf_variance": +1, "overlap_dinf": +1,
    "ps_AUROC_delta": +1,
}

# CATE base learner is logistic/ridge only (poly2/gbm coded but not run). present_cols()
# drops whatever a given leaderboard.csv lacks, so this stays correct if more tiers appear.
_COL_TO_LEARNER_BASE = {f"PEHE_{l}_{b}": (l, b) for l in LEARNERS for b in BASES}


def _oriented(df: pd.DataFrame, diagnostic: str) -> pd.Series:
    """Diagnostic values transformed so lower is always better."""
    if diagnostic == "ps_AUROC":
        return (df[diagnostic] - 0.5).abs()
    return df[diagnostic] if DIAGNOSTIC_DIRECTION[diagnostic] == +1 else -df[diagnostic]


def load_block_table() -> pd.DataFrame:
    """One row per (trial, preset, distribution, embedding[incl. raw], learner, base),
    melted from leaderboard.csv's PEHE_{learner}_{base} columns."""
    board_paths = sorted(Path("data").glob("evaluations*/**/leaderboard.csv"))
    if not board_paths:
        raise FileNotFoundError(
            "No leaderboard.csv found under data/evaluations*/. Run E_leaderboard.py first."
        )
    board = pd.concat((pd.read_csv(p) for p in board_paths), ignore_index=True)
    board["pred_score"] = _pred_score(board)

    id_vars = present_cols(["trial", "preset", "distribution", "embedding", "method", "dim",
                            "rho"] + DIAGNOSTIC_COLS, board)
    value_vars = present_cols(list(_COL_TO_LEARNER_BASE), board)
    if not value_vars:
        raise ValueError(
            "No PEHE_{learner}_{base} columns in leaderboard.csv -- stale evaluations "
            "(schema predates current LEARNERS/BASES)? Rerun D_evaluation.py + E_leaderboard.py."
        )
    # Subset first: board also carries a literal "PEHE" column (the PEHE_T_lr default),
    # which collides with melt's value_name="PEHE" even though it's outside id_vars/value_vars.
    melted = board[id_vars + value_vars].melt(id_vars=id_vars, value_vars=value_vars,
                                               var_name="learner_base", value_name="PEHE")

    lb = melted["learner_base"].map(_COL_TO_LEARNER_BASE)
    melted["learner"] = lb.map(lambda t: t[0])
    melted["base"] = lb.map(lambda t: t[1])
    melted["is_raw"] = melted["method"] == "raw"

    return melted.dropna(subset=["PEHE"]).reset_index(drop=True)


def block_kendall_tau(df: pd.DataFrame, diagnostics: list[str] = DIAGNOSTIC_COLS) -> pd.DataFrame:
    """Kendall's tau-b between each diagnostic and PEHE, per block."""
    rows = []
    n_skipped_small = 0
    for key, block in df[~df["is_raw"]].groupby(BLOCK_KEY):
        if len(block) < MIN_BLOCK_SIZE:
            n_skipped_small += 1
            continue
        entry = dict(zip(BLOCK_KEY, key))
        for diagnostic in diagnostics:
            tau, _ = kendalltau(_oriented(block, diagnostic), block["PEHE"], variant="b")
            rows.append({**entry, "diagnostic": diagnostic, "tau": tau, "m_candidates": len(block)})
    if n_skipped_small:
        print(f"   NOTE: {n_skipped_small} blocks skipped for tau (m < {MIN_BLOCK_SIZE} candidates)")
    return pd.DataFrame(rows)


def block_selection_regret(df: pd.DataFrame, diagnostics: list[str] = DIAGNOSTIC_COLS) -> pd.DataFrame:
    """Per block, per diagnostic (+ 'random'/'raw' baselines): regret of trusting that
    diagnostic's pick vs. the oracle-best. Degenerate blocks (near-tied candidates) get
    norm_regret=NaN, degenerate=True -- kept, not dropped.

    Also emits normalization-free companions: selected_rank (1 = oracle-best), is_top,
    and spread (the worst-best PEHE gap = the norm_regret denominator), so downstream
    aggregates can guard against near-tie artifacts. `random`'s rank/is_top are analytic
    ((m+1)/2, 1/m) -- its PEHE is the mean, a scalar that is never an actual candidate.
    """
    rows = []
    skipped_nan = []
    raw_pehe = df[df["is_raw"]].groupby(_RAW_KEY)["PEHE"].first()  # no capacity axis
    for key, block in df.groupby(BLOCK_KEY):
        candidates = block[~block["is_raw"]]
        if len(candidates) < MIN_BLOCK_SIZE:
            continue
        entry = dict(zip(BLOCK_KEY, key))
        pehe_sorted = np.sort(candidates["PEHE"].to_numpy())
        pehe_oracle = float(pehe_sorted[0])
        pehe_worst = float(pehe_sorted[-1])
        denom = pehe_worst - pehe_oracle
        degenerate = bool(denom < EPS_REGRET_DENOM)

        def _regret_row(name: str, pehe_selected: float) -> dict:
            raw_regret = pehe_selected - pehe_oracle
            norm_regret = np.nan if degenerate else raw_regret / denom
            # rank among the (sorted) candidate PEHEs; ties share the better rank
            rank = int(np.searchsorted(pehe_sorted, pehe_selected, side="left") + 1)
            return {**entry, "diagnostic": name, "PEHE_selected": pehe_selected,
                    "PEHE_oracle": pehe_oracle, "PEHE_worst": pehe_worst,
                    "raw_regret": raw_regret, "norm_regret": norm_regret,
                    "selected_rank": rank, "is_top": rank <= 1,
                    "spread": denom, "degenerate": degenerate, "m_candidates": len(candidates)}

        for diagnostic in diagnostics:
            oriented = _oriented(candidates, diagnostic)
            if oriented.isna().all():
                skipped_nan.append({**entry, "diagnostic": diagnostic})
                continue
            picked = candidates.loc[oriented.idxmin(), "PEHE"]
            rows.append(_regret_row(diagnostic, picked))

        # Random baseline, closed form: E[PEHE_selected] under uniform selection = mean(PEHE).
        # Its rank/is_top are analytic (the mean is never an actual candidate).
        m = len(candidates)
        rows.append({**_regret_row("random", candidates["PEHE"].mean()),
                     "selected_rank": (m + 1) / 2, "is_top": 1 / m})
        # .get: raw is only evaluated for some base tiers (e.g. lr), so a base=gbm block
        # has no raw reference -- NaN row, not a KeyError crash.
        rows.append(_regret_row("raw", raw_pehe.get(tuple(entry[k] for k in _RAW_KEY), np.nan)))

    if skipped_nan:
        examples = ", ".join(f"{s['diagnostic']}@{s['trial']}/{s['preset']}" for s in skipped_nan[:5])
        more = f" (+{len(skipped_nan) - 5} more)" if len(skipped_nan) > 5 else ""
        print(f"   NOTE: {len(skipped_nan)} (block, diagnostic) pairs skipped for selection regret "
              f"(all-NaN diagnostic): {examples}{more}")
    return pd.DataFrame(rows)


_WILCOXON_COLS = ["diagnostic", "W", "p", "n_pairs", "median_diff",
                  "rank_biserial_r", "median_diff_vs_raw"]


def _rank_biserial(diff: pd.Series) -> float:
    """Matched-pairs rank-biserial correlation (Kerby) for a paired difference.
    Negative = diagnostic regret below the comparison baseline's (i.e. better)."""
    d = diff[diff != 0].to_numpy()
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d))
    w_plus, w_minus = r[d > 0].sum(), r[d < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def wilcoxon_vs_baseline(regret_df: pd.DataFrame, baseline: str = "random",
                          diagnostics: list[str] = DIAGNOSTIC_COLS) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank per diagnostic vs. a baseline ("random" or "raw"), block
    by block, with effect sizes: median paired difference (vs. this baseline and vs. raw)
    and the matched-pairs rank-biserial correlation. Negative median_diff = lower regret
    than the baseline."""
    index_cols = [c for c in BLOCK_KEY if c in regret_df.columns]
    wide = regret_df.pivot_table(index=index_cols, columns="diagnostic", values="norm_regret")
    rows = []
    for diagnostic in diagnostics:
        if diagnostic not in wide.columns or baseline not in wide.columns:
            continue
        diff = (wide[diagnostic] - wide[baseline]).dropna()
        diff_raw = ((wide[diagnostic] - wide["raw"]).dropna()
                    if "raw" in wide.columns else pd.Series(dtype=float))
        med_raw = float(diff_raw.median()) if len(diff_raw) else np.nan
        if len(diff) == 0 or not (diff != 0).any():
            rows.append({"diagnostic": diagnostic, "baseline": baseline, "W": np.nan, "p": np.nan,
                         "n_pairs": len(diff),
                         "median_diff": float(diff.median()) if len(diff) else np.nan,
                         "rank_biserial_r": np.nan, "median_diff_vs_raw": med_raw})
            continue
        stat = wilcoxon(diff)
        rows.append({"diagnostic": diagnostic, "baseline": baseline, "W": float(stat.statistic),
                     "p": float(stat.pvalue), "n_pairs": len(diff), "median_diff": float(diff.median()),
                     "rank_biserial_r": _rank_biserial(diff), "median_diff_vs_raw": med_raw})
    if not rows:
        return pd.DataFrame(columns=["diagnostic", "baseline"] + _WILCOXON_COLS[1:])
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def _bootstrap_ci_blocks(df: pd.DataFrame, value_col: str, trial_col: str = "trial",
                          n_boot: int = 1000, seed: int = 42, method: str = "BCa") -> tuple[float, float]:
    """Trial-level bootstrap CI on median(value_col). method="BCa" (right-skewed regret,
    bounded at 0) or "percentile" (sensitivity check -- BCa can misbehave on a
    bounded/discrete statistic at n~=57 trials)."""
    df = df.dropna(subset=[value_col])
    trial_rows = _trial_index_groups(df, trial_col)
    trial_names = np.array(list(trial_rows))
    if len(trial_names) < 2:
        vals = df[value_col]
        med = float(vals.median()) if len(vals) else np.nan
        return med, med

    def stat(trial_idx):
        sampled = trial_names[np.asarray(trial_idx, dtype=int)]
        idx = np.concatenate([trial_rows[t] for t in sampled])
        vals = df.loc[idx, value_col]
        return float(vals.median()) if len(vals) else np.nan

    rng = np.random.default_rng(seed)
    trial_positions = np.arange(len(trial_names))
    res = scipy_bootstrap((trial_positions,), stat, n_resamples=n_boot,
                           method=method, random_state=rng, vectorized=False)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def _bootstrap_summary_one(regret_df: pd.DataFrame, tau_df: pd.DataFrame, diagnostic: str,
                            headline: bool = False) -> dict:
    regret_sub = regret_df[(regret_df["diagnostic"] == diagnostic) & ~regret_df["degenerate"]]
    tau_sub = tau_df[tau_df["diagnostic"] == diagnostic]
    regret_lo, regret_hi = _bootstrap_ci_blocks(regret_sub, "norm_regret")
    tau_lo, tau_hi = _bootstrap_ci_blocks(tau_sub, "tau")
    # normalization-free companions (WS3): computed over ALL blocks (degenerate ok -- rank
    # is well-defined even when the worst-best gap is ~0).
    rank_all = regret_df[regret_df["diagnostic"] == diagnostic]
    out = {
        "diagnostic": diagnostic,
        "n_trials": int(regret_sub["trial"].nunique()) if len(regret_sub) else 0,
        "n_blocks_regret": len(regret_sub),
        "median_norm_regret": float(regret_sub["norm_regret"].median()) if len(regret_sub) else np.nan,
        "norm_regret_ci_lo": regret_lo, "norm_regret_ci_hi": regret_hi,
        "median_rank": float(rank_all["selected_rank"].median()) if len(rank_all) else np.nan,
        "p_top1": float(rank_all["is_top"].mean()) if len(rank_all) else np.nan,
        "n_blocks_tau": len(tau_sub),
        "median_tau": float(tau_sub["tau"].median()) if len(tau_sub) else np.nan,
        "tau_ci_lo": tau_lo, "tau_ci_hi": tau_hi,
    }
    if headline:  # percentile-vs-BCa sensitivity is only reported for the headline stratum
        lo_pct, hi_pct = _bootstrap_ci_blocks(regret_sub, "norm_regret", method="percentile")
        out["norm_regret_pct_ci_lo"], out["norm_regret_pct_ci_hi"] = lo_pct, hi_pct
    return out


def bootstrap_summary(regret_df: pd.DataFrame, tau_df: pd.DataFrame,
                       diagnostics: list[str] = DIAGNOSTIC_COLS, headline: bool = False) -> pd.DataFrame:
    """Per diagnostic: median norm_regret and median tau, with trial-level BCa bootstrap CIs.
    Parallelized over diagnostics -- safe since callers never wrap this in their own Parallel."""
    rows = Parallel(n_jobs=-1)(
        delayed(_bootstrap_summary_one)(regret_df, tau_df, diagnostic, headline) for diagnostic in diagnostics
    )
    return pd.DataFrame(rows).sort_values("median_norm_regret").reset_index(drop=True)


def run_critical_difference(regret_df: pd.DataFrame, diagnostics: list[str] = DIAGNOSTIC_COLS):
    """Friedman/Nemenyi critical-difference ranking of diagnostics by norm_regret
    (order='ascending': lower regret ranks higher).

    Collapsed to one row per (trial, diagnostic) first: the ~6,200 blocks share
    trials/embeddings/presets, so Friedman/Nemenyi run on them overstate power -- the
    honest replication unit is the ~56 trials. autorank still needs complete cases, so
    trials missing a diagnostic/baseline are dropped and counted."""
    cols = diagnostics + ["random", "raw"]
    agg = regret_df.groupby(["trial", "diagnostic"], as_index=False)["norm_regret"].mean()
    wide = agg.pivot_table(index="trial", columns="diagnostic", values="norm_regret")
    # dropna(axis=1, how="all") first: a diagnostic/baseline that is all-NaN in a stratum
    # would otherwise drop every trial via the row-wise dropna() and kill the CD diagram.
    wide = wide[[c for c in cols if c in wide.columns]].dropna(axis=1, how="all").dropna()
    n_dropped = agg["trial"].nunique() - len(wide)

    if len(wide) < 5:
        print(f"   WARNING: only {len(wide)} complete-case trials for CD diagram "
              "(autorank needs >=5) -- skipping.")
        return None, wide, n_dropped

    result = autorank.autorank(wide, alpha=0.05, order="ascending")
    return result, wide, n_dropped


def selection_regret_breakdown(block_df: pd.DataFrame, diagnostics: list[str] = DIAGNOSTIC_COLS) -> dict:
    """Headline stratum `baseline` (baseline knob preset only) plus row-filtered strata of
    it -- one per relative capacity rho, and fitted-vs-nonfitted assignment. Each NON-baseline
    knob preset is also emitted, computed on that preset's own blocks, as an appendix
    robustness check (not part of the headline). A per-base-learner split appears only once a
    multi-base CATE run exists."""
    tau_full = block_kendall_tau(block_df, diagnostics)
    regret_full = block_selection_regret(block_df, diagnostics)
    tau_b = tau_full[tau_full["preset"] == "baseline"]
    regret_b = regret_full[regret_full["preset"] == "baseline"]

    strata = {"baseline": (tau_b, regret_b)}
    if block_df["base"].nunique() > 1:  # only meaningful once a multi-base CATE run exists
        for base in sorted(block_df["base"].unique()):
            strata[f"base={base}"] = (tau_b[tau_b["base"] == base], regret_b[regret_b["base"] == base])

    # Relative-capacity strata of the baseline headline: does the diagnostic ranking hold
    # across rho = k/d? rho is a 7-point grid (RELATIVE_RATIOS) -- plain groupby, no binning.
    for r in sorted(regret_b["rho"].dropna().unique()):
        strata[f"rho={r:g}"] = (tau_b[tau_b["rho"] == r], regret_b[regret_b["rho"] == r])

    # Congeniality check (baseline only): datasets whose assignment is not a fitted logistic
    # of X are not congenial to the logistic-propensity diagnostics (W2 in review). Few
    # trials, so trial-level bootstrap CIs here are wide -- read descriptively, not as a test.
    nonfit = sorted(set(block_df["trial"]) & NONFITTED_ASSIGNMENT_TRIALS)
    if nonfit:
        m_tau, m_reg = tau_b["trial"].isin(nonfit), regret_b["trial"].isin(nonfit)
        strata["assignment=nonfitted"] = (tau_b[m_tau], regret_b[m_reg])
        strata["assignment=fitted"] = (tau_b[~m_tau], regret_b[~m_reg])

    # Appendix robustness: each non-baseline knob preset, on its own blocks.
    for preset in sorted(p for p in block_df["preset"].unique() if p != "baseline"):
        strata[f"preset={preset}"] = (tau_full[tau_full["preset"] == preset],
                                       regret_full[regret_full["preset"] == preset])

    results = {}
    for name, (tau_df, regret_df) in strata.items():
        wilcoxon_df = pd.concat(
            [wilcoxon_vs_baseline(regret_df, b, diagnostics) for b in ("random", "raw")],
            ignore_index=True,
        )
        # + random/raw so the baselines get the same bootstrap CI rows as the diagnostics
        boot_df = bootstrap_summary(regret_df, tau_df, diagnostics + ["random", "raw"],
                                    headline=(name == "baseline"))
        results[name] = (tau_df, regret_df, wilcoxon_df, boot_df)

    return results


def _capture_autorank_report(cd_result) -> str:
    """autorank.create_report() only prints; adapt it into a string."""
    import io
    from contextlib import redirect_stdout

    report = io.StringIO()
    with redirect_stdout(report):
        autorank.create_report(cd_result)
    return report.getvalue()


def write_selection_regret_outputs(strata_results: dict, cd_result, cd_wide: pd.DataFrame, cd_n_dropped: int) -> None:
    csv_rows = []
    for stratum, (tau_df, regret_df, wilcoxon_df, boot_df) in strata_results.items():
        t = tau_df.copy(); t["stratum"], t["table"] = stratum, "tau"
        r = regret_df.copy(); r["stratum"], r["table"] = stratum, "regret"
        w = wilcoxon_df.copy(); w["stratum"], w["table"] = stratum, "wilcoxon"
        b = boot_df.copy(); b["stratum"], b["table"] = stratum, "bootstrap"
        csv_rows += [t, r, w, b]
    pd.concat(csv_rows, ignore_index=True).to_csv("output/selection_regret.csv", index=False)
    print("   Saved output/selection_regret.csv")

    if cd_result is not None:
        nemenyi = sp.posthoc_nemenyi_friedman(cd_wide.values)
        nemenyi.columns = cd_wide.columns
        nemenyi.index = cd_wide.columns
        nemenyi.to_csv("output/selection_regret_nemenyi.csv")
        print("   Saved output/selection_regret_nemenyi.csv")

    with open("output/selection_regret.md", "w") as f:
        f.write("# Selection Regret: Does Trusting a Diagnostic Beat Guessing?\n\n")
        f.write(
            "For every (trial, distribution, meta-learner, base-learner, relative capacity "
            "rho) block, a diagnostic picks the embedding it prefers (no oracle PEHE access); "
            "regret is the PEHE gap between that pick and the true oracle-best embedding, "
            "normalized by the gap between the oracle-best and oracle-worst embedding in that "
            "block (0 = picked the oracle best, 1 = picked the oracle worst). `random` = "
            "expected regret of picking uniformly at random; `raw` = regret of always "
            "deploying uncompressed features.\n\n"
            "The `baseline` stratum (baseline knob preset) is the headline; the `rho=` and "
            "`assignment=` strata are filters of it; each `preset=` stratum is a separate "
            "robustness check computed on that non-baseline preset's own blocks.\n\n"
        )
        f.write(
            "*Selectors are propensity/overlap/balance by-products of e(Z) plus the two "
            "Zhang et al. (2020) overlap criteria (cf_variance, overlap_dinf). No CATE-derived "
            "signal is analysed: raw held-out R-/DR-loss carry Var(Y|Z), so they do not "
            "compare across embeddings.*\n\n"
        )

        for stratum, (tau_df, regret_df, wilcoxon_df, boot_df) in strata_results.items():
            n_degenerate = int(regret_df["degenerate"].sum()) if len(regret_df) else 0
            f.write(f"## {stratum}\n\n")
            f.write(f"*{len(regret_df)} (block, diagnostic-or-baseline) rows; "
                    f"{n_degenerate} degenerate (near-tied candidates, norm_regret undefined).*\n\n")

            f.write("### Median regret / tau, trial-level bootstrap 95% CI\n\n")
            boot_display = boot_df.copy()
            boot_display.insert(1, "label", boot_display["diagnostic"].map(lambda c: DIAGNOSTIC_LABELS.get(c, c)))
            f.write(boot_display.to_markdown(index=False))
            f.write("\n\n")

            f.write("### Wilcoxon signed-rank vs. random and raw baselines\n\n")
            f.write(wilcoxon_df.to_markdown(index=False))
            f.write("\n\n")

        if cd_result is not None:
            f.write("## Critical-difference ranking (autorank, logistic/ridge base)\n\n")
            f.write(f"*Trial-level means: {len(cd_wide)} complete-case trials used "
                    f"(the block count is not the replication unit -- blocks share "
                    f"trials/embeddings/presets); {cd_n_dropped} trials dropped for a "
                    "missing diagnostic/baseline.*\n\n")
            f.write("```\n" + _capture_autorank_report(cd_result) + "```\n\n")
        else:
            f.write("## Critical-difference ranking\n\nSkipped -- fewer than 5 complete-case trials.\n\n")

    print("   Saved output/selection_regret.md")


def _present_diagnostics(df: pd.DataFrame) -> list[str]:
    present = present_cols(DIAGNOSTIC_COLS, df)
    if len(present) < len(DIAGNOSTIC_COLS):
        print(f"   WARNING: missing diagnostic columns {sorted(set(DIAGNOSTIC_COLS) - set(present))}, skipping them")
    return present


if __name__ == "__main__":
    print("1. Loading per-seed pehe_cate.csv + propensity_overlap.csv")
    table = load_diagnostic_pehe_table()
    # Headline analysis is the baseline knob preset only (per-preset stress is an appendix
    # robustness check, done on the block table in selection_regret_breakdown).
    table = table[table["preset"] == "baseline"].reset_index(drop=True)
    row_diagnostics = _present_diagnostics(table)
    print(f"   {len(table)} baseline-preset rows across {table['trial'].nunique()} trial(s), "
          f"{table['embedding'].nunique()} embeddings")

    print("\n2. Row-level rank correlation vs. PEHE_dlog (baseline) + pairwise redundancy")
    results = {
        "baseline": spearman_kendall_table(table, row_diagnostics),
        # Oracle-only references (need true_tau): do Melnychuk Def. 1's two validity gaps
        # (RICB, loss of heterogeneity) actually drive PEHE_dlog? Rendered as a stratum.
        "oracle_refs_vs_PEHE_dlog": spearman_kendall_table(table, ORACLE_REF_COLS),
    }
    corr_matrix = diagnostic_correlation_matrix(table, row_diagnostics)
    for stratum, corr_df in results.items():
        print(f"   {stratum}: {len(corr_df)} diagnostics, top by |spearman_rho|: "
              f"{corr_df.iloc[0]['diagnostic'] if len(corr_df) else 'n/a'}")

    print("\n3. Writing output/diagnostic_validity.{csv,md}")
    write_outputs(results, corr_matrix)

    print("\n4. Loading leaderboard.csv for block-level selection-regret analysis")
    block_table = load_block_table()
    block_diagnostics = _present_diagnostics(block_table)
    print(f"   {len(block_table)} (embedding, learner, base) rows across "
          f"{block_table['trial'].nunique()} trial(s)")

    print("\n5. Computing block-level tau-b / selection regret (baseline headline + rho/assignment/preset strata)")
    regret_results = selection_regret_breakdown(block_table, block_diagnostics)
    for stratum, (tau_df, regret_df, wilcoxon_df, boot_df) in regret_results.items():
        n_beat_random = int((wilcoxon_df["median_diff"] < 0).sum())
        print(f"   {stratum}: {len(regret_df['diagnostic'].unique())} diagnostics+baselines, "
              f"{n_beat_random} beat random by median regret")

    print("\n6. Running critical-difference ranking (autorank, baseline blocks)")
    # run_critical_difference means over everything but (trial, diagnostic); it runs on the
    # baseline-preset headline stratum (logistic/ridge base learner).
    cd_result, cd_wide, cd_n_dropped = run_critical_difference(regret_results["baseline"][1], block_diagnostics)

    print("\n7. Writing output/selection_regret.{csv,md}")
    write_selection_regret_outputs(regret_results, cd_result, cd_wide, cd_n_dropped)

    print("\nDone.")
