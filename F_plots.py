# The pipeline's single plotting entry point: cross-trial robustness/failure-mode analysis,
# plus the diagnostic-validity and selection-regret figures (stats for those two live in
# E_diagnostic_validity.py, which stays plot-free).
#
# Reads:  data/evaluations[_ood]/{trial}[/knob_variants/{preset}]/leaderboard.csv
#         output/diagnostic_validity.csv, output/selection_regret.csv
# Writes: output/cross_trial_comparison.{csv,md}
#         latex/figures/regret_cd_diagram.pdf          (paper figure)
#         plots/pareto_frontier_grid_PEHE_mean.png     (ID, baseline -- headline capacity/PEHE tradeoff)
#         plots/capacity_curve_by_method_{mean,median}_PEHE_mean.png (ID, per-method capacity grid)
#         plots/ood/... (same two, OOD -- only if any trial has OOD data)
#         plots/id_vs_ood_comparison.png               (ID vs. OOD generalization)
#         plots/knob_sensitivity.png                   (cross-preset: metric vs. each knob's value)
#         plots/knob_variants/<preset>/...             (same ID/OOD suite + diagnostic/regret plots, one dir per non-baseline preset)
#         plots/regret_distribution.png, plots/regret_cd_diagram.png
#
# Everything else this module used to generate was cut: none of it mapped to a figure
# actually used downstream.

import os
import sys
from pathlib import Path

import autorank
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import E_diagnostic_validity as dv
from utils.leaderboard_utils import (
    HIGHER_IS_BETTER,
    METRIC_GROUPS,
    PRED_METRICS_BINARY,
    PRED_METRICS_CONTINUOUS,
    present_cols,
)
from utils.plot_utils import (
    METHOD_FAMILY,
    SUPERVISION_FAMILY,
    display_trial_name,
    generate_standard_plot_suite,
    plot_id_vs_ood_comparison,
    plot_knob_sensitivity,
    plot_regret_distribution,
)

os.makedirs("output", exist_ok=True)

# Paper figures land here (referenced by latex/main.tex, latex/appendix.tex).
LATEX_FIG_DIR = Path("latex/figures")

# Canonical metric list and direction (shared across ranking, summary, and plotting)
CANDIDATE_METRICS = list(dict.fromkeys(
    PRED_METRICS_BINARY + PRED_METRICS_CONTINUOUS + ["PEHE_mean"]
    + [c for cols, _, _ in METRIC_GROUPS.values() for c in cols]
))


def compute_present_metrics(df: pd.DataFrame) -> list[str]:
    """CANDIDATE_METRICS columns that exist in `df` and have at least one non-NaN value."""
    return [m for m in present_cols(CANDIDATE_METRICS, df) if df[m].notna().any()]


# ---------------------------------------------------------------------------
# 1. Load every leaderboard on disk, slice out the ID/baseline cross-trial table
# ---------------------------------------------------------------------------
print("1. Loading leaderboards")

#    Each leaderboard.csv is already self-describing (trial/preset/distribution/outcome_type/
#    n_features/has_individual_cf columns, stamped by E_leaderboard.py's build_leaderboard()) --
#    glob + concat, no path-parsing, no oracle-metadata re-reading.
board_paths = list(Path("data").glob("evaluations*/**/leaderboard.csv"))
full_all = (pd.concat((pd.read_csv(p) for p in board_paths), ignore_index=True)
            if board_paths else pd.DataFrame())
if len(full_all) == 0:
    print("   No leaderboards found. Run C_embeddings.py, D_evaluation.py, E_leaderboard.py first.")
    sys.exit(1)

full = full_all[(full_all["distribution"] == "ID") & (full_all["preset"] == "baseline")].reset_index(drop=True)
trials = sorted(full["trial"].unique())
trial_meta = full.drop_duplicates("trial").set_index("trial")[["outcome_type", "n_features", "has_individual_cf"]].to_dict("index")

display_trials = [display_trial_name(t) for t in trials]
print(f"   Found {len(trials)} trials: {', '.join(display_trials)}")

for trial in trials:
    trial_board = full[full["trial"] == trial]
    dims = trial_board["dim"].dropna().unique()
    n_features = trial_meta[trial]["n_features"]
    # rho is the embedding grid coordinate (RELATIVE_RATIOS), carried on leaderboard.csv.
    rhos = sorted(trial_board["rho"].dropna().unique())

    if len(dims) > 0:
        dims_sorted = sorted(dims)
        dim_range = f"{int(dims_sorted[0]):3d}-{int(dims_sorted[-1]):3d}"
        rho_range = f"{rhos[0]:.3f}-{rhos[-1]:.3f}" if rhos else "N/A"
        rho_str = ", ".join(f"{r:g}" for r in rhos)
    else:
        dim_range = "N/A"
        rho_range = "N/A"
        rho_str = "N/A"

    print(f"   {display_trial_name(trial):17s}: {trial_meta[trial]['outcome_type']:10s}  d={n_features:5d}  "
          f"k={dim_range:>8s}  ρ={rho_range:>8s}  {len(trial_board):3d} embeddings", end="")
    print(f"      Relative capacities: {rho_str}")

present_metrics = compute_present_metrics(full)

# ---------------------------------------------------------------------------
# 2. Build summary tables per metric
# ---------------------------------------------------------------------------
print("\n2. Building summary tables")

# Pivot: embedding (with method/dim) × trial, values = metric
summary_tables = {}

for metric in present_metrics:
    if metric not in full.columns or full[metric].isna().all():
        continue

    # Get one row per (embedding, trial)
    pivot_data = full[["embedding", "method", "dim", "trial", "n_features", metric]].drop_duplicates()

    # Pivot to get trials as columns
    pivot = pivot_data.pivot(index="embedding", columns="trial", values=metric)
    pivot = pivot.round(4)

    # Sort by mean (excluding NaN); for higher-is-better metrics, negate for sort
    if metric in HIGHER_IS_BETTER:
        pivot["mean"] = -pivot.mean(axis=1, skipna=True)
    else:
        pivot["mean"] = pivot.mean(axis=1, skipna=True)
    pivot = pivot.sort_values("mean")
    pivot = pivot.drop("mean", axis=1)

    summary_tables[metric] = pivot
    print(f"   {metric}: {len(pivot)} embeddings")

# Write combined CSV
full.to_csv("output/cross_trial_comparison.csv", index=False)
print("   Saved output/cross_trial_comparison.csv")

# ---------------------------------------------------------------------------
# 3. Write markdown summary
# ---------------------------------------------------------------------------
with open("output/cross_trial_comparison.md", "w") as f:
    f.write("# Cross-Trial Embedding Comparison\n\n")

    # Trial overview
    f.write("## Trial characteristics\n\n")
    meta_rows = []
    for trial in trials:
        meta_rows.append({
            "Trial": trial,
            "Outcome": trial_meta[trial]["outcome_type"],
            "Features": trial_meta[trial]["n_features"],
            "Has CF": "Yes" if trial_meta[trial]["has_individual_cf"] else "No",
        })
    meta_df = pd.DataFrame(meta_rows)
    f.write(meta_df.to_markdown(index=False))
    f.write("\n\n*Has CF: Has individual counterfactuals (Y0/Y1). JOBS=No (only population ATE).*\n\n")

    # Summary tables
    f.write("## Embedding performance by trial\n\n")
    for metric, pivot in summary_tables.items():
        f.write(f"### {metric}\n\n")
        if metric in ["AUROC", "R2"]:
            f.write("*Higher is better. Sorted by mean across trials (descending).*\n\n")
        else:
            f.write("*Lower is better. Sorted by mean across trials.*\n\n")
        # Compute mean for display
        mean_col = pivot.mean(axis=1, skipna=True)
        display = pivot.copy()
        display.insert(0, "Mean", mean_col)
        f.write(display.to_markdown())
        f.write("\n\n")

    # Relative capacity aggregation: show performance by ρ = k/d (cross-trial comparable)
    f.write("## Performance by relative capacity (ρ = k/d)\n\n")
    f.write(
        "Median metric across trials, grouped by (method, ρ). "
        "Relative capacity ρ = k/d standardizes compression/expansion regimes: "
        "ρ=0.5 means half-compression, ρ=1.0 matches input dimension, ρ=2.0 doubles capacity. "
        "This aggregation is comparable across trials with different feature dimensions.\n\n"
    )

    # Build ρ-aggregated table for PEHE_mean and a few other key metrics
    rho_pivots = {}
    for metric in ["PEHE_mean", "ATE_error", "regret"]:
        if metric not in full.columns or full[metric].isna().all():
            continue

        rho_summary = (
            full[["method", "rho", metric]]
            .dropna(subset=[metric])
            .groupby(["method", "rho"], as_index=False)[metric]
            .median()
            .sort_values(["method", "rho"])
        )

        if len(rho_summary) > 0:
            rho_pivot = rho_summary.pivot(index="method", columns="rho", values=metric).round(4)
            rho_pivots[metric] = rho_pivot
            f.write(f"**{metric} by (method, relative capacity ρ)**\n\n")
            f.write(rho_pivot.to_markdown())
            f.write("\n\n")

    # Embedding-family gap by relative capacity -- the generator for the paper's
    # tab:family-gap (ΔlogPEHE, agnostic vs causal/supervised, by ρ). ID / baseline preset.
    if "PEHE_mean_dlog" in full.columns:
        fam = full.assign(
            family=full["method"].map(METHOD_FAMILY).map(SUPERVISION_FAMILY)
        )
        fam = fam[(fam["method"] != "raw") & fam["family"].notna()].dropna(
            subset=["PEHE_mean_dlog", "rho"]
        )
        if len(fam) > 0:
            g = (fam.groupby(["family", "rho"])["PEHE_mean_dlog"]
                    .agg(["mean", "sem"]).round(4).reset_index())
            wide = g.pivot(index="rho", columns="family", values="mean")
            if {"agnostic", "supervised"} <= set(wide.columns):
                wide["gap (agnostic − causal/sup.)"] = (wide["agnostic"] - wide["supervised"]).round(4)
            f.write("**ΔlogPEHE by embedding family and relative capacity ρ** "
                    "(0 = tuned-raw parity; lower is better; SE over the trial pool)\n\n")
            f.write(wide.to_markdown())
            f.write("\n\n")
            f.write(g.rename(columns={"mean": "ΔlogPEHE", "sem": "SE"}).to_markdown(index=False))
            f.write("\n\n")

    f.write("## Robustness analysis\n\n")
    f.write("Methods ranking well across trials at comparable capacity (ρ) tend to generalize. "
            "Ranked within each ρ bucket rather than pooled across dims, so a method isn't "
            "rewarded just for having a larger embedding available in some trial.\n\n")

    if "PEHE_mean" in rho_pivots:
        top_n = 5
        rho_rank = rho_pivots["PEHE_mean"].rank(axis=0, method="min")
        rob_df = (rho_rank.mean(axis=1, skipna=True).rename("Mean rank (within ρ)")
                  .sort_values().head(top_n).reset_index())
        f.write(f"**Top {min(top_n, len(rob_df))} most robust methods "
                "(lowest mean PEHE_mean rank within capacity bucket):**\n\n")
        f.write(rob_df.to_markdown(index=False))
        f.write("\n\n")

print("   Saved output/cross_trial_comparison.md")

# ---------------------------------------------------------------------------
# 4. Generate plots: ID/OOD suite for baseline + every knob preset, ID-vs-OOD comparison,
#    knob-sensitivity.
# ---------------------------------------------------------------------------
fig_dir = Path("plots/")
ood_full = full_all[(full_all["distribution"] == "OOD") & (full_all["preset"] == "baseline")]


def preset_fig_dir(preset: str) -> Path:
    """"baseline" -> plots/; any other preset -> plots/knob_variants/<preset>/."""
    return fig_dir if preset == "baseline" else fig_dir / "knob_variants" / preset


knob_presets = sorted(p for p in full_all["preset"].unique() if p != "baseline")
print(f"\n4. Generating plots for preset(s): {', '.join(['baseline'] + knob_presets)}")
for preset in ["baseline"] + knob_presets:
    preset_dir = preset_fig_dir(preset)
    preset_id = full if preset == "baseline" else full_all[(full_all["distribution"] == "ID") & (full_all["preset"] == preset)]
    preset_ood = ood_full if preset == "baseline" else full_all[(full_all["distribution"] == "OOD") & (full_all["preset"] == preset)]
    if len(preset_id) > 0:
        generate_standard_plot_suite(preset_id, preset_dir, sorted(preset_id["trial"].unique()))
    if len(preset_ood) > 0:
        generate_standard_plot_suite(preset_ood, preset_dir / "ood", sorted(preset_ood["trial"].unique()))

if len(ood_full) > 0:
    print("\n5. Generating ID-vs-OOD comparison plot")
    id_vs_ood = full.merge(ood_full, on=["trial", "embedding", "method"], suffixes=("", "_ood"))
    plot_id_vs_ood_comparison(id_vs_ood, fig_dir)

print("\n6. Generating knob-sensitivity plot")
# {knob_field: [(preset_name, knob_value), ...]}: each preset's one knob that differs from
# baseline (skips multi-knob presets like stress_all).
with open("B_oracles/shared/oracle_knob_grid.yaml") as f:
    knob_grid = yaml.safe_load(f)
baseline_knobs = knob_grid["baseline"]
axis_map = {}
for preset, cfg in knob_grid.items():
    changed = [k for k, v in cfg.items() if k in baseline_knobs and v != baseline_knobs[k]]
    if len(changed) == 1:
        axis_map.setdefault(changed[0], []).append((preset, cfg[changed[0]]))
for points in axis_map.values():
    points.append(("baseline", 1.0))
    points.sort(key=lambda pv: pv[1])
plot_knob_sensitivity(full_all[full_all["distribution"] == "ID"], fig_dir, axis_map)

# ---------------------------------------------------------------------------
# 7. Diagnostic-validity + selection-regret figures (stats computed in
#    E_diagnostic_validity.py; this module only plots them, see file header).
# ---------------------------------------------------------------------------
print("\n7. Generating diagnostic-validity + selection-regret plots")


def _strata_to_plot(table: pd.DataFrame) -> list[str]:
    """"baseline" (the headline) plus every non-baseline preset stratum. The rho= /
    assignment= strata are summarised in output/selection_regret.md only, not plotted."""
    strata = set(table["stratum"].unique())
    presets = sorted(s for s in strata if s.startswith("preset="))
    return ["baseline"] + presets


def _stratum_fig_dir(stratum: str) -> Path:
    """"baseline" -> plots/; "preset=<name>" -> plots/knob_variants/<name>/."""
    return preset_fig_dir("baseline" if stratum == "baseline" else stratum.removeprefix("preset="))


regret_csv = Path("output/selection_regret.csv")
if regret_csv.exists():
    regret_table = pd.read_csv(regret_csv)

    for stratum in _strata_to_plot(regret_table):
        regret_sub = regret_table[(regret_table["table"] == "regret") & (regret_table["stratum"] == stratum)]
        if len(regret_sub) == 0:
            continue
        stratum_dir = _stratum_fig_dir(stratum)
        plot_regret_distribution(regret_sub, stratum_dir, labels=dv.DIAGNOSTIC_LABELS)

        # autorank's result object doesn't round-trip through CSV, so re-run it here --
        # but on regret_sub, already in hand, not by reloading raw data.
        cd_result, cd_wide, _ = dv.run_critical_difference(regret_sub)
        if cd_result is not None:
            ax = autorank.plot_stats(cd_result, allow_insignificant=True)
            if ax is not None:
                cd_path = stratum_dir / "regret_cd_diagram.png"
                cd_path.parent.mkdir(parents=True, exist_ok=True)
                ax.figure.savefig(cd_path, dpi=300, bbox_inches="tight")
                print(f"   Saved {cd_path}")
                if stratum == "baseline":  # headline paper figure (vector)
                    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
                    ax.figure.savefig(LATEX_FIG_DIR / "regret_cd_diagram.pdf", bbox_inches="tight")
                    print(f"   Saved {LATEX_FIG_DIR / 'regret_cd_diagram.pdf'}")
else:
    print(f"   WARNING: {regret_csv} not found -- run E_diagnostic_validity.py first. Skipping.")

print("\nDone.")
