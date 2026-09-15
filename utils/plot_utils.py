# Plotting utilities for cross-trial embedding comparison (F_plots.py).
#
# Each function accepts the preprocessed cross-trial DataFrame and metadata, then saves a PNG
# to the provided figure directory. Unlike D_evaluation_utils.py (pure numeric functions),
# this module has I/O side effects (savefig).
#
# Functions expose metric parameters to enable flexible reuse across PEHE_mean, ATE_error,
# CF_RMSE, regret, outcome metrics (AUROC/AUPRC or R2/RMSE), and other dimensions.

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .leaderboard_utils import pareto_frontier_mask


def display_trial_name(trial: str) -> str:
	"""Format trial name for display: uppercase the five named trials, keep rctbench_trial* lowercase."""
	if trial in {"ihdp", "news", "twins", "jobs", "vasst"}:
		return trial.upper()
	return trial


# --- Plot styling ---
PLOT_FONT: str = "DejaVu Sans"

def _pick_available_font(preferred: str) -> str:
	candidates = [preferred, "Arial", "Helvetica"]
	available = {f.name for f in mpl.font_manager.fontManager.ttflist}
	for name in candidates:
		if any(name.lower() in a.lower() for a in available):
			return name
	return mpl.rcParams.get("font.family", ["sans-serif"])

def apply_plot_style(font: str | None = None) -> None:
	"""Apply a consistent plotting style across all cross-trial comparison plots."""
	font_family = _pick_available_font(font or PLOT_FONT)
	sns.set_theme(context="notebook", style="whitegrid")
	plt.rcParams.update({
		"font.family": font_family,
		"font.size": 11,
		"axes.titlesize": 13,
		"axes.labelsize": 11,
		"xtick.labelsize": 9,
		"ytick.labelsize": 9,
		"legend.fontsize": 8,
		"figure.titlesize": 14,
		"lines.linewidth": 1.5,
		"figure.facecolor": "white",
		"axes.linewidth": 1.0,
		"axes.edgecolor": "black",
		"grid.alpha": 0.3,
		"grid.linestyle": "--",
		"grid.linewidth": 0.5,
	})
	mpl.rcParams["pdf.fonttype"] = 42
	mpl.rcParams["ps.fonttype"] = 42
	mpl.rcParams["svg.fonttype"] = "none"

apply_plot_style()

# Panel-specific and exception overrides (named for clarity, not magic numbers)
PANEL_TITLE_FONTSIZE = 10
PANEL_LABEL_FONTSIZE = 9
PANEL_TICK_FONTSIZE = 8
LEGEND_NCOL = 2
FRONTIER_COLOR = "red"
BASELINE_LINE_KWARGS = {"color": "gray", "linestyle": "--", "linewidth": 1.5, "alpha": 0.5}

# Marker shapes cycled across distinct embedding dimensions (capacity), independent of the
# color channel (which encodes method family) -- lets a plot carry both without a color-only
# 2D encoding collision.
DIM_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8"]


def _dim_marker_map(dims) -> dict:
	"""{dim -> marker shape}, cycling DIM_MARKERS in ascending dim order. NaN (raw) excluded."""
	uniq = sorted({int(d) for d in dims if pd.notna(d)})
	return {d: DIM_MARKERS[i % len(DIM_MARKERS)] for i, d in enumerate(uniq)}

# Method display names: maps internal method keys to publication-friendly names.
# Order: raw -> linear unsupervised -> neural unsupervised -> causal methods (by lineage/chronological).
METHOD_DISPLAY_NAMES = {
    "raw": "raw",
    # --- linear unsupervised
    "randomproj": "RandomProjection",
    "pca": "PCA",
    "fa": "FactorAnalysis",
    "fastica": "FastICA",
    # --- neural unsupervised
    "ae": "AutoEncoder",
    "vae": "VAE",
    # --- causal methods (lineage/chronological order)
    "bnn": "BNN",             # 2016, base representation-balancing method
    "tarnet": "TARNet",       # 2017, split-head evolution of BNN (CFRNet.py, alpha=0)
    "cfrnet": "CFRNet",       # 2017, TARNet + MMD balance penalty (CFRNet.py, alpha>0)
    "dcn": "DCN-PD",          # 2017, TARNet shape + propensity-dropout balancing
    "site": "SITE",           # 2018, CFRNet/TARNet + similarity-preserving terms
    "cfrisw": "CFR-ISW",      # 2019, CFRNet + importance-sampling weights
    "dragonnet": "DragonNet", # 2019, CFRNet + propensity head + targeted regularization
    "drcfr": "DR-CFR",        # 2020, disentangled representation (treatment/confounder/outcome)
    "nice": "NICE",           # 2021, DragonNet backbone + IRM
    "cevae": "CEVAE",         # 2017, VAE + T/Y supervision
    "deeptreat": "DeepTreat", # 2018, bias-removing AE + IPW ITE heads
    "tedvae": "TEDVAE",       # 2021, disentangled latent, CEVAE-style exposure
}

# Method family taxonomy: groups methods for color-safe categorical encoding (max 3 hues: validated all-pairs scatter)
METHOD_FAMILY = {
	"raw": "raw",
	# --- linear unsupervised (blue)
	"randomproj": "linear", "pca": "linear", "fa": "linear", "fastica": "linear",
	# --- neural unsupervised (orange)
	"ae": "neural", "vae": "neural",
	# --- causal methods (aqua)
	"bnn": "causal", "tarnet": "causal", "cfrnet": "causal", "dcn": "causal",
	"site": "causal", "cfrisw": "causal", "dragonnet": "causal", "drcfr": "causal",
	"nice": "causal", "cevae": "causal", "deeptreat": "causal", "tedvae": "causal",
}

FAMILY_COLORS = {
	"linear": "#2a78d6",    # blue (validated palette.md slot 1)
	"neural": "#eb6834",    # orange (validated palette.md slot 2)
	"causal": "#1baf7a",    # aqua (validated palette.md slot 3; WARN vs light surface -> always direct-label)
}

# Supervised-vs-agnostic grouping for the headline capacity/PEHE figure: agnostic embeddings
# (linear/neural) never see T,Y at fit time, causal ("supervised") methods do.
SUPERVISION_FAMILY = {"linear": "agnostic", "neural": "agnostic", "causal": "supervised"}
SUPERVISION_COLORS = {
	"agnostic": "#2a78d6",   # blue (palette.md slot 1)
	"supervised": "#1baf7a", # aqua (palette.md slot 3)
}

CONTEXT_GRAY = "#c3c2b7"  # recessive gray for context/background series (from palette.md)
RAW_BASELINE_COLOR = "gray"

# Canonical method ordering for consistent colors and legend arrangement
_METHOD_ORDER = {m: i for i, m in enumerate(METHOD_DISPLAY_NAMES)}

def _method_sort_key(method: str) -> tuple[int, str]:
	"""Sort key: canonical METHOD_DISPLAY_NAMES order, unknown methods sorted alphabetically after known ones."""
	return (_METHOD_ORDER.get(method, len(_METHOD_ORDER)), method)


def _group_of(method: str, color_by: str) -> str:
	"""Map a method to its color-group label: family (linear/neural/causal) or
	supervision (agnostic/supervised grouping)."""
	family = METHOD_FAMILY.get(method, "causal")
	if color_by == "supervision":
		return SUPERVISION_FAMILY.get(family, "supervised")
	return family


def _build_family_legend(labeled_set: set, all_methods: list, family_map: dict, color_by: str = "family") -> list:
	"""Build legend elements for family/supervision-based coloring: one entry per group
	present in labeled_set, plus a gray "Other methods" entry iff some methods are unlabeled."""
	order = ("linear", "neural", "causal") if color_by == "family" else ("agnostic", "supervised")
	unique_groups = sorted(
		set(_group_of(m, color_by) for m in labeled_set if m != "raw"),
		key=lambda x: order.index(x) if x in order else len(order)
	)
	elements = [
		Line2D([0], [0], color=family_map[next(m for m in all_methods if _group_of(m, color_by) == g)],
			linewidth=2, label=g.capitalize())
		for g in unique_groups
	]
	if len(labeled_set) < len(all_methods):
		elements.append(Line2D([0], [0], color=CONTEXT_GRAY, linewidth=1, alpha=0.25, label="Other methods"))
	return elements


def _family_color_map(methods: list, color_by: str = "family") -> dict:
	"""Map methods to group-based colors (safe for scatter/bubble: few hues). Raw always
	returns baseline gray. color_by="family" (linear/neural/causal, default) or
	"supervision" (agnostic/supervised grouping)."""
	colors = FAMILY_COLORS if color_by == "family" else SUPERVISION_COLORS
	return {m: RAW_BASELINE_COLOR if METHOD_FAMILY.get(m) == "raw" else colors.get(_group_of(m, color_by), RAW_BASELINE_COLOR) for m in methods}


def _draw_pareto_frontier(ax, k_vals, error_vals, marker_size: float, linewidth: float) -> None:
	"""Draw Pareto frontier: staircase line + star markers from frontier mask."""
	frontier_mask = pareto_frontier_mask(k_vals, error_vals)
	if not frontier_mask.any():
		return
	frontier_k, frontier_error = k_vals[frontier_mask], error_vals[frontier_mask]
	order = np.argsort(frontier_k)
	frontier_k_sorted, frontier_error_sorted = frontier_k[order], frontier_error[order]
	ax.plot(
		frontier_k_sorted, frontier_error_sorted, color=FRONTIER_COLOR,
		linewidth=linewidth, alpha=0.8, zorder=10,
	)
	ax.scatter(
		frontier_k_sorted, frontier_error_sorted, color=FRONTIER_COLOR,
		s=marker_size, marker="*", edgecolors="darkred", linewidth=1, zorder=11,
	)


def _finish_axes(ax, xlabel: str, ylabel: str, title: str, legend: bool = True) -> None:
	"""Apply the standard xlabel/ylabel/title/legend/grid tail shared by most single-panel plots."""
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	if legend:
		ax.legend(loc="best", framealpha=0.95, ncol=LEGEND_NCOL)
	ax.grid(True, alpha=0.3)


def _draw_trial_panel(
	ax, trial_data: pd.DataFrame, norm_col: str,
	methods: list, family_map: dict,
	scatter_size: float, marker_size: float, frontier_linewidth: float,
	draw_frontier: bool = True,
) -> None:
	"""Draw one trial's Pareto panel: every method in its family color (scatter + line),
	raw baseline as a star, frontier staircase on top (only if draw_frontier)."""
	for method in methods:
		method_data = trial_data[trial_data["method"] == method]
		if len(method_data) == 0:
			continue
		color = family_map[method]
		ax.scatter(
			method_data["dim"], method_data[norm_col], color=color,
			s=scatter_size, alpha=0.7, edgecolors="black", linewidth=0.5, zorder=2,
		)
		method_data_sorted = method_data.sort_values("dim")
		ax.plot(method_data_sorted["dim"], method_data_sorted[norm_col],
			color=color, linewidth=1.5, alpha=0.7, zorder=2)

	# Raw baseline
	raw_data = trial_data[trial_data["method"] == "raw"]
	if len(raw_data) > 0:
		ax.scatter(raw_data["dim"], raw_data[norm_col], color=RAW_BASELINE_COLOR,
			s=scatter_size * 0.8, marker="*", alpha=0.7, edgecolors="black", linewidth=0.8, zorder=3)

	if draw_frontier:
		_draw_pareto_frontier(
			ax, trial_data["dim"].values, trial_data[norm_col].values,
			marker_size=marker_size, linewidth=frontier_linewidth,
		)
	if norm_col.endswith("_dlog"):  # Δlog view: raw sits at 0 by construction
		ax.axhline(y=0.0, **BASELINE_LINE_KWARGS)


def _save_fig(fig, fig_dir: Path, filename: str, dpi: int = 300) -> None:
    """Save figure with tight layout, close, and print status. Supports subdirectories in filename."""
    fig.tight_layout()
    full_path = fig_dir / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {full_path}")


def _rho_tick_label(rho: float) -> str:
	"""Fraction label for a binned rho tick: 0.25 -> '1/4', 2.0 -> '2'."""
	if rho >= 1:
		return f"{rho:g}"
	return f"1/{int(round(1.0 / rho))}"


def _set_rho_xaxis(ax) -> None:
	"""Standard log₂(ρ=k/d) x-axis: ticks at 1/8…4, fraction labels, 'ρ = k/d (log₂)'."""
	log_ticks = np.array([-3, -2, -1, 0, 1, 2])
	ax.set_xticks(log_ticks)
	ax.set_xticklabels([_rho_tick_label(r) for r in 2.0 ** log_ticks], fontsize=PANEL_TICK_FONTSIZE)
	ax.set_xlabel("ρ = k/d (log₂)", fontsize=PANEL_LABEL_FONTSIZE)


N_REGIMES = ["n<200", "200≤n<1000", "n≥1000"]
N_REGIME_COLORS = {"n<200": "#f0a202", "200≤n<1000": "#1baf7a", "n≥1000": "#2a4d69"}


def _pooled(full: pd.DataFrame) -> pd.DataFrame:
	"""Trial-pooled capacity-plot input: drop trials whose raw-X CATE is ~unrecoverable
	(raw tau_recovery_R2 < 0.1) or that carry no sample size `n` (stale leaderboard), and
	add the n_regime facet column. No-op if already pooled."""
	if "n_regime" in full.columns:
		return full
	raw = full[full["method"] == "raw"]
	bad = set(raw.loc[raw["tau_recovery_R2"] < 0.1, "trial"])
	out = full[~full["trial"].isin(bad) & full["n"].notna()].copy()
	n_missing = full.loc[full["n"].isna(), "trial"].nunique()
	if n_missing:
		print(f"   WARNING: {n_missing} trial(s) have no `n` (stale leaderboard.csv -- rerun "
		      "E_leaderboard.py); excluded from the pooled capacity curves.")
	out["n_regime"] = pd.cut(out["n"], [-np.inf, 200, 1000, np.inf], labels=N_REGIMES).astype(object)
	return out


def _present_regimes(df: pd.DataFrame) -> list:
	return [r for r in N_REGIMES if r in set(df["n_regime"])]


def _metric_base(value_col: str) -> tuple[bool, str]:
	"""(is_dlog, name-without-'_dlog')."""
	dlog = value_col.endswith("_dlog")
	return dlog, value_col[:-5] if dlog else value_col


def _style_rho_panel(ax, ylabel: str, title: str) -> None:
	"""ρ=1 marker + log₂(ρ) x-axis + labels/grid/ticks -- every ρ-capacity panel."""
	ax.axvline(x=0, **BASELINE_LINE_KWARGS)
	_set_rho_xaxis(ax)
	ax.set_ylabel(ylabel, fontsize=PANEL_LABEL_FONTSIZE)
	ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
	ax.grid(True, alpha=0.2)
	ax.tick_params(labelsize=PANEL_TICK_FONTSIZE)


def _aggregate_by_rho(
	full: pd.DataFrame, value_col: str, stat: str = "mean",
	group_cols: tuple[str, ...] = ("method", "rho"),
) -> pd.DataFrame | None:
	"""Median/mean `value_col` per `group_cols`, ρ binned to powers of two. `group_cols`
	must include `rho`; its other entries must be columns of `full`."""
	if value_col not in full.columns or "rho" not in full.columns:
		return None
	binned = full.assign(rho=full["rho"].apply(
		lambda r: r if pd.isna(r) or r <= 0 else 2.0 ** np.round(np.log2(r))))
	rows = binned[list(group_cols) + [value_col]].dropna(subset=["rho", value_col])
	if rows.empty:
		return None
	return rows.groupby(list(group_cols))[value_col].agg(stat).reset_index().sort_values("rho")


def plot_capacity_curve_by_method(
	full: pd.DataFrame,
	fig_dir: Path,
	value_col: str = "PEHE_mean_dlog",
	stat: str = "mean",
) -> None:
	"""One panel per method: x = ρ, y = `value_col`, one median line per sample-size regime
	(the PEHE minimum's location is n-dependent). `*_dlog` = tuned-raw-relative (0 = parity)."""
	is_dlog, base = _metric_base(value_col)
	y_label = f"Δlog {base} vs tuned raw (lower = better)" if is_dlog else f"{base} (lower = better)"
	full = _pooled(full)
	curve = _aggregate_by_rho(full, value_col, stat, group_cols=("method", "n_regime", "rho"))
	if curve is None:
		return
	parity_y = 0.0 if is_dlog else full.loc[full["method"] == "raw", value_col].mean()
	curve = curve.assign(log_rho=np.log2(curve["rho"]))
	methods = sorted(curve["method"].unique(), key=_method_sort_key)
	regimes = _present_regimes(curve)

	n_cols = min(5, len(methods))
	n_rows = math.ceil(len(methods) / n_cols)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
	                         squeeze=False, sharex=True, sharey=True)
	axes = axes.flatten()
	for ax, method in zip(axes, methods):
		md = curve[curve["method"] == method]
		for reg in regimes:
			c = md[md["n_regime"] == reg].sort_values("log_rho")
			ax.plot(c["log_rho"], c[value_col], marker="o", linewidth=2, markersize=6, color=N_REGIME_COLORS[reg])
		if np.isfinite(parity_y):
			ax.axhline(y=parity_y, **BASELINE_LINE_KWARGS)
		_style_rho_panel(ax, y_label, METHOD_DISPLAY_NAMES.get(method, method))
	for ax in axes[len(methods):]:
		ax.set_visible(False)

	fig.suptitle(f"Capacity Curve by Method ({base}, {stat})", fontsize=13, y=0.995)
	legend_elements = [Line2D([0], [0], color=N_REGIME_COLORS[r], linewidth=2, label=r) for r in regimes]
	legend_elements.append(Line2D([0], [0], **BASELINE_LINE_KWARGS, label="ρ=1 & Δlog=0" if is_dlog else "ρ=1 & raw baseline"))
	fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
	           bbox_to_anchor=(0.5, -0.01), framealpha=0.95, fontsize=9)

	_save_fig(fig, fig_dir, f"capacity_curve_by_method_{stat}_{base}.png")


def plot_pareto_frontier_grid(
	full: pd.DataFrame,
	trials: list[str],
	fig_dir: Path,
	value_col: str = "PEHE_mean_dlog",
	# The headline figure groups by supervised-vs-agnostic (does the embedding see T,Y
	# at fit time), not by the finer linear/neural/causal method taxonomy.
	color_by: str = "supervision",
	draw_frontier: bool | None = None,
) -> None:
	"""Per-trial Pareto-frontier grid (x = k). `draw_frontier` defaults to True for PEHE_*
	(the red best-k staircase is a bias-variance sweet spot, not meaningful for the
	validity metrics). `color_by`: "supervision" or "family"."""
	if value_col not in full.columns or full[value_col].isna().all():
		return
	is_dlog, base = _metric_base(value_col)
	y_label = f"Δlog {base} vs raw" if is_dlog else f"{base} (lower = better)"
	if draw_frontier is None:
		draw_frontier = value_col.startswith("PEHE")

	n_trials = len(trials)
	n_cols = min(5, math.ceil(math.sqrt(n_trials)))
	n_rows = math.ceil(n_trials / n_cols)

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
	axes = axes.flatten()

	all_methods = sorted(set(full["method"].unique()), key=_method_sort_key)
	family_map = _family_color_map(all_methods, color_by=color_by)

	for idx, trial in enumerate(trials):
		ax = axes[idx]
		trial_data = full[full["trial"] == trial]

		if len(trial_data) == 0:
			ax.text(0.5, 0.5, f"No data: {trial}", ha="center", va="center", transform=ax.transAxes)
			ax.set_visible(False)
			continue

		_draw_trial_panel(ax, trial_data, value_col, all_methods, family_map,
			scatter_size=50, marker_size=80, frontier_linewidth=2, draw_frontier=draw_frontier)

		ax.set_xlabel("k", fontsize=PANEL_LABEL_FONTSIZE)
		ax.set_ylabel(y_label, fontsize=PANEL_LABEL_FONTSIZE)
		ax.set_title(display_trial_name(trial), fontsize=PANEL_TITLE_FONTSIZE)
		ax.grid(True, alpha=0.2)
		ax.tick_params(labelsize=PANEL_TICK_FONTSIZE)

	for ax in axes[len(trials):]:
		ax.set_visible(False)

	fig.suptitle(f"Pareto Frontiers by Trial ({base})", y=0.995)

	legend_elements = _build_family_legend(set(all_methods), all_methods, family_map, color_by=color_by)
	fig.legend(handles=legend_elements, loc="lower center", ncol=LEGEND_NCOL,
		bbox_to_anchor=(0.5, -0.02), framealpha=0.95)

	_save_fig(fig, fig_dir, f"pareto_frontier_grid_{base}.png")


def plot_id_vs_ood_comparison(
	id_vs_ood: pd.DataFrame,
	fig_dir: Path,
	metrics: tuple = ("PEHE_mean", "ATE_error", "CF_RMSE", "regret"),
) -> None:
	"""
	In-distribution vs. out-of-distribution comparison: one panel per metric, scatter of
	each embedding's ID value (x) against its OOD value (y), colored by method family.
	Points above the y=x line are worse OOD (for these error metrics); points on the line
	generalize perfectly to the held-out sites.

	Each method's dims are connected in ascending order (a capacity trajectory, not an
	unordered cloud) and marker shape encodes dim (DIM_MARKERS), so capacity is readable
	independent of the family-color channel -- otherwise every dim of a method collapses
	into visually-identical same-color dots with no way to tell which capacity is which.

	Args:
		id_vs_ood: DataFrame with columns [embedding, method, dim, trial, {metric}, {metric}_ood]
			per metric in `metrics` (built from leaderboard_ood_comparison.csv, concatenated
			across trials — currently only VASST has OOD sources, so this is typically a
			single-trial scatter). `dim` is always populated by build_leaderboard()
			(E_leaderboard.py), including a backfilled value for "raw" -- never NaN.
		fig_dir: directory to save PNG to
		metrics: metric names to panel (each needs both {metric} and {metric}_ood present)
	"""
	if id_vs_ood is None or len(id_vs_ood) == 0:
		return

	metrics_present = [m for m in metrics if m in id_vs_ood.columns and f"{m}_ood" in id_vs_ood.columns]
	if not metrics_present:
		return

	n_metrics = len(metrics_present)
	n_cols = min(2, n_metrics)
	n_rows = math.ceil(n_metrics / n_cols)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
	axes = axes.flatten()

	all_methods = sorted(id_vs_ood["method"].unique(), key=_method_sort_key)
	family_color_map = _family_color_map(all_methods)
	dim_marker_map = _dim_marker_map(id_vs_ood["dim"])

	for idx, metric in enumerate(metrics_present):
		ax = axes[idx]
		data = id_vs_ood.dropna(subset=[metric, f"{metric}_ood"])

		for method in all_methods:
			method_data = data[data["method"] == method].sort_values("dim")
			if len(method_data) == 0:
				continue

			# Capacity trajectory: thin line through this method's dims in ascending order.
			ax.plot(method_data[metric], method_data[f"{metric}_ood"], color=family_color_map[method],
			        linewidth=1.2, alpha=0.5, zorder=1)
			for dim_val, dim_data in method_data.groupby("dim"):
				ax.scatter(
					dim_data[metric], dim_data[f"{metric}_ood"], color=family_color_map[method],
					marker=dim_marker_map.get(int(dim_val), "o"),
					s=70, alpha=0.85, edgecolors="black", linewidth=0.5, zorder=2,
				)

		if len(data) > 0:
			lo = min(data[metric].min(), data[f"{metric}_ood"].min())
			hi = max(data[metric].max(), data[f"{metric}_ood"].max())
			ax.plot([lo, hi], [lo, hi], **BASELINE_LINE_KWARGS)

		_finish_axes(ax, f"ID {metric}", f"OOD {metric}", metric, legend=False)

	for ax in axes[len(metrics_present):]:
		ax.set_visible(False)

	fig.suptitle("In-Distribution vs. Out-of-Distribution", y=0.995)

	legend_elements = _build_family_legend(set(all_methods), all_methods, family_color_map) + [
		Line2D([0], [0], **BASELINE_LINE_KWARGS, label="y = x (no OOD shift)"),
	] + [
		Line2D([0], [0], marker=marker, color="w", markerfacecolor="black", markeredgecolor="black",
		       markersize=8, label=f"k={dim_val}")
		for dim_val, marker in dim_marker_map.items()
	]
	fig.legend(handles=legend_elements, loc="lower center", ncol=min(6, len(legend_elements)),
	           bbox_to_anchor=(0.5, -0.02), framealpha=0.95)

	_save_fig(fig, fig_dir, "id_vs_ood_comparison.png")


def plot_diagnostic_importance(
	importance_df: pd.DataFrame,
	fig_dir: Path,
	covariate: str = "dim",
	labels: dict[str, str] | None = None,
	caption: str | None = None,
) -> None:
	"""
	Horizontal bar chart, permutation importance ranked descending: the headline
	diagnostic-importance figure. `covariate` (the capacity control, not a diagnostic finding) is drawn
	in gray to visually separate it from the propensity diagnostics being ranked.

	Args:
		importance_df: DataFrame with columns [feature, importance_mean, importance_std],
			from E_diagnostic_validity.py's permutation_importance_ranking (the "all" stratum).
		fig_dir: directory to save PNG to
		covariate: feature name to render in gray as the capacity control (not a diagnostic)
		labels: optional {raw column name -> display label} map for y-tick text (e.g. mapping
			"ESS_ctrl_frac" -> "Effective N, control (overlap)"); raw names used if omitted.
			Coloring/covariate matching always uses the raw `feature` column, not the label.
		caption: optional footnote text rendered below the chart (e.g. explaining what the
			diagnostic families mean) so the PNG is self-explanatory standalone.
	"""
	if len(importance_df) == 0:
		return

	labels = labels or {}
	df = importance_df.sort_values("importance_mean", ascending=True)
	colors = [CONTEXT_GRAY if f == covariate else SUPERVISION_COLORS["supervised"] for f in df["feature"]]
	y_labels = [labels.get(f, f) for f in df["feature"]]

	fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1.5))
	ax.barh(y_labels, df["importance_mean"], xerr=df["importance_std"],
	        color=colors, alpha=0.85, edgecolor="black", linewidth=0.5, capsize=3)
	ax.set_xlabel("Permutation importance (Δlog PEHE, cross-validated by trial)")
	ax.set_title("Which diagnostics predict PEHE degradation?")
	ax.grid(axis="x", alpha=0.3)

	legend_elements = [
		Line2D([0], [0], marker="s", color="w", markerfacecolor=SUPERVISION_COLORS["supervised"],
		       markersize=10, label="Propensity diagnostic", markeredgecolor="black", markeredgewidth=0.5),
		Line2D([0], [0], marker="s", color="w", markerfacecolor=CONTEXT_GRAY,
		       markersize=10, label=f"{labels.get(covariate, covariate)}", markeredgecolor="black", markeredgewidth=0.5),
	]
	ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95)

	if caption:
		fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, color="#333333", wrap=True)

	_save_fig(fig, fig_dir, "diagnostic_importance.png")


def plot_regret_distribution(
	regret_df: pd.DataFrame,
	fig_dir: Path,
	labels: dict[str, str] | None = None,
	caption: str | None = None,
) -> None:
	"""
	Box plot of normalized selection regret per diagnostic (+ random/raw baselines), sorted
	by median regret ascending.

	Args:
		regret_df: columns [diagnostic, norm_regret, degenerate], from
			E_diagnostic_validity.py's block_selection_regret().
		fig_dir: directory to save PNG to
		labels: optional {raw column name -> display label} map for x-tick text.
		caption: optional footnote text rendered below the chart.
	"""
	# .astype(bool): read back from a CSV stacked with tables lacking "degenerate", this
	# column round-trips as object dtype, and `~` on an object Series bitwise-inverts
	# instead of negating (~True == -2, not False).
	degenerate = regret_df["degenerate"].astype(bool)
	df = regret_df[~degenerate].dropna(subset=["norm_regret"])
	if len(df) == 0:
		return

	labels = labels or {}
	order = (df.groupby("diagnostic")["norm_regret"].median().sort_values().index.tolist())
	data = [df.loc[df["diagnostic"] == d, "norm_regret"].values for d in order]
	x_labels = [labels.get(d, d) for d in order]
	colors = [CONTEXT_GRAY if d in ("random", "raw") else SUPERVISION_COLORS["supervised"] for d in order]

	fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(order)), 5))
	bp = ax.boxplot(data, tick_labels=x_labels, patch_artist=True, showfliers=False, widths=0.6)
	for patch, color in zip(bp["boxes"], colors):
		patch.set_facecolor(color)
		patch.set_alpha(0.85)
		patch.set_edgecolor("black")
		patch.set_linewidth(0.5)
	ax.set_ylabel("Normalized selection regret (0 = oracle-best, 1 = oracle-worst)")
	ax.set_title("Does trusting a diagnostic beat guessing? Regret by diagnostic")
	ax.grid(axis="y", alpha=0.3)
	plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

	legend_elements = [
		Line2D([0], [0], marker="s", color="w", markerfacecolor=SUPERVISION_COLORS["supervised"],
		       markersize=10, label="Diagnostic", markeredgecolor="black", markeredgewidth=0.5),
		Line2D([0], [0], marker="s", color="w", markerfacecolor=CONTEXT_GRAY,
		       markersize=10, label="Baseline (random / raw)", markeredgecolor="black", markeredgewidth=0.5),
	]
	ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95)

	if caption:
		fig.text(0.5, -0.05, caption, ha="center", va="top", fontsize=8, color="#333333", wrap=True)

	_save_fig(fig, fig_dir, "regret_distribution.png")


def plot_knob_sensitivity(
	full_all: pd.DataFrame,
	fig_dir: Path,
	axis_map: dict[str, list[tuple[str, float]]],
	metric: str = "PEHE_mean",
) -> None:
	"""
	Small multiples, one panel per knob: x-axis = knob value, y-axis = median
	Δlog {metric} vs raw per supervision family. baseline (knob value 1.0) is the
	shared reference point on every panel.

	Args:
		full_all: cross-trial DataFrame spanning every preset, one row per (trial, preset, embedding)
		fig_dir: directory to save the PNG to
		axis_map: {knob_field: [(preset_name, knob_value), ...]}, built by the caller
		metric: canonical metric name (uses the {metric}_dlog column)
	"""
	norm_col = f"{metric}_dlog"  # Delta-log vs raw (canonical); the ratio *_norm family was removed
	if norm_col not in full_all.columns or full_all[norm_col].isna().all():
		return

	knobs_present = [k for k, points in axis_map.items()
	                 if full_all["preset"].isin([p for p, _ in points]).any()]
	if not knobs_present:
		return

	full_all = full_all.assign(_family=full_all["method"].map(lambda m: _group_of(m, "supervision")))
	families = sorted(full_all.loc[full_all["method"] != "raw", "_family"].unique())
	n_cols = min(3, len(knobs_present))
	n_rows = math.ceil(len(knobs_present) / n_cols)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)
	axes = axes.flatten()

	for idx, knob in enumerate(knobs_present):
		ax = axes[idx]
		points = axis_map[knob]
		preset_order = [p for p, _ in points]
		knob_data = full_all[full_all["preset"].isin(preset_order)]
		medians = knob_data.groupby(["_family", "preset"])[norm_col].median()

		for family in families:
			curve = medians.get(family, pd.Series(dtype=float)).reindex(preset_order).dropna()
			if len(curve) == 0:
				continue
			x = [dict(points)[p] for p in curve.index]
			ax.plot(x, curve.values, marker="o", linewidth=2, markersize=7,
				color=SUPERVISION_COLORS[family], alpha=0.85, label=family.capitalize())

		ax.axvline(x=1.0, **BASELINE_LINE_KWARGS)
		ax.axhline(y=0.0, **BASELINE_LINE_KWARGS)  # Delta-log tuned-raw parity
		ax.set_xlabel(knob, fontsize=PANEL_LABEL_FONTSIZE)
		ax.set_ylabel(f"Δlog {metric} vs tuned raw", fontsize=PANEL_LABEL_FONTSIZE)
		ax.set_title(knob, fontsize=PANEL_TITLE_FONTSIZE)
		ax.grid(True, alpha=0.2)
		ax.tick_params(labelsize=PANEL_TICK_FONTSIZE)

	for ax in axes[len(knobs_present):]:
		ax.set_visible(False)

	fig.suptitle(f"Knob Sensitivity ({metric})", y=0.995)
	legend_elements = [
		Line2D([0], [0], color=SUPERVISION_COLORS[f], linewidth=2, label=f.capitalize())
		for f in families
	] + [Line2D([0], [0], **BASELINE_LINE_KWARGS, label="baseline (knob=1.0 & Δlog=0)")]
	fig.legend(handles=legend_elements, loc="lower center", ncol=min(4, len(legend_elements)),
	           bbox_to_anchor=(0.5, -0.02), framealpha=0.95)

	_save_fig(fig, fig_dir, "knob_sensitivity.png")


def plot_capacity_curve_by_trial(
	full: pd.DataFrame, fig_dir: Path, value_col: str = "PEHE_mean_dlog",
	stat: str = "median", color_by: str = "family", trials: list[str] | None = None,
	per_page: int = 30,
) -> None:
	"""One panel per trial, x = ρ, one line per method. Curated trials first, then
	rctbench; paginated at `per_page`."""
	is_dlog, base = _metric_base(value_col)
	agg = _aggregate_by_rho(full, value_col, stat, group_cols=("trial", "method", "rho"))
	if agg is None:
		return
	agg = agg.assign(log_rho=np.log2(agg["rho"]))
	is_signed = not is_dlog and bool((agg[value_col] < 0).any())
	y_label = f"Δlog {base} vs raw" if is_dlog else base
	trials = sorted(trials or agg["trial"].unique(),
	                key=lambda t: (str(t).startswith("rctbench_trial"), str(t)))
	methods = sorted(agg["method"].unique(), key=_method_sort_key)
	cmap = _family_color_map(methods, color_by=color_by)
	raw_means = full[full["method"] == "raw"].groupby("trial")[value_col].mean()

	pages = [trials[i:i + per_page] for i in range(0, len(trials), per_page)]
	for pi, page in enumerate(pages):
		n_cols = min(5, len(page))
		n_rows = math.ceil(len(page) / n_cols)
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.9 * n_rows),
		                         squeeze=False, sharex=True, sharey=True)
		axes = axes.flatten()
		for ax, trial in zip(axes, page):
			td = agg[agg["trial"] == trial]
			for m in methods:
				md = td[td["method"] == m].sort_values("log_rho")
				ax.plot(md["log_rho"], md[value_col], marker="o", ms=4, lw=1.5, alpha=0.55, color=cmap[m])
			if is_dlog or is_signed:
				ax.axhline(y=0.0, **BASELINE_LINE_KWARGS)
			elif pd.notna(raw_means.get(trial)):
				ax.axhline(y=raw_means[trial], **BASELINE_LINE_KWARGS)
			_style_rho_panel(ax, y_label, display_trial_name(trial))
		for ax in axes[len(page):]:
			ax.set_visible(False)
		fig.suptitle(f"Capacity by Trial ({base}, {stat}) — one line per method", y=0.995)
		fig.legend(handles=_build_family_legend(set(methods), methods, cmap, color_by=color_by),
		           loc="lower center", ncol=LEGEND_NCOL, bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
		tag = f"_p{pi}" if len(pages) > 1 else ""
		_save_fig(fig, fig_dir, f"capacity_curve_by_trial_{stat}_{base}{tag}.png", dpi=200)


def plot_capacity_overview(full: pd.DataFrame, fig_dir: Path, stat: str = "median") -> None:
	"""Infinite-data representation references (A–B) beside the finite-n PEHE (C), so the
	PEHE minimum is not read as the best embedding dimension. One line per sample-size
	regime; / SD(tau^x) so pooling reflects compression not effect-size scale."""
	spec = [("HET_loss_norm", "(A) infinite-data — loss of heterogeneity", False),
	        ("RICB_true_mean_norm", "(B) infinite-data — RICB (signed ATE bias)", True),
	        ("PEHE_mean_norm", "(C) finite-n — PEHE", False)]
	panels = [s for s in spec if s[0] in full.columns and full[s[0]].notna().any()]
	if not panels:
		return
	full = _pooled(full)
	non_raw = full[full["method"] != "raw"]
	regimes = _present_regimes(non_raw)

	fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2), squeeze=False)
	for ax, (col, title, signed) in zip(axes.flatten(), panels):
		agg = _aggregate_by_rho(non_raw, col, stat, group_cols=("n_regime", "rho"))
		if agg is not None:
			for reg in regimes:
				c = agg[agg["n_regime"] == reg].sort_values("rho")
				ax.plot(np.log2(c["rho"]), c[col], marker="o", lw=2, ms=6, color=N_REGIME_COLORS[reg], label=reg)
		raw_ref = full.loc[full["method"] == "raw", col].mean()
		if np.isfinite(raw_ref):
			ax.axhline(y=raw_ref, **BASELINE_LINE_KWARGS)
		if signed:
			ax.axhline(y=0.0, color="black", lw=1, alpha=0.6)
			ax.text(0.03, 0.04, "0 = ATE unbiased", transform=ax.transAxes, fontsize=7, alpha=0.7)
		_style_rho_panel(ax, f"{col.replace('_norm', '')} / SD(τ)", title)

	fig.suptitle("Capacity overview — infinite-data references (A–B) vs finite-n PEHE (C)", y=0.995)
	legend = [Line2D([0], [0], color=N_REGIME_COLORS[r], linewidth=2, label=r) for r in regimes]
	legend.append(Line2D([0], [0], **BASELINE_LINE_KWARGS, label="raw baseline"))
	fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
	_save_fig(fig, fig_dir, "capacity_overview.png")


def plot_pehe_family_split(full: pd.DataFrame, fig_dir: Path, stat: str = "median") -> None:
	"""Plug-in (S/Lo/T/X) vs pseudo-outcome (RA/Z/F/U/R/DR) PEHE vs ρ, per sample-size
	regime -- pseudo-outcome learners carry propensity weights, so compression (which tames
	overlap) should help them more."""
	cols = [("PEHE_plugin_mean_dlog", "plug-in", FAMILY_COLORS["linear"]),
	        ("PEHE_pseudo_mean_dlog", "pseudo-outcome", FAMILY_COLORS["neural"])]
	present = [c for c in cols if c[0] in full.columns and full[c[0]].notna().any()]
	if len(present) < 2:
		return
	full = _pooled(full)
	regimes = _present_regimes(full)

	fig, axes = plt.subplots(1, len(regimes), figsize=(4.5 * len(regimes), 4.0),
	                         squeeze=False, sharex=True, sharey=True)
	for ax, reg in zip(axes.flatten(), regimes):
		sub = full[full["n_regime"] == reg]
		for col, label, colour in present:
			agg = _aggregate_by_rho(sub, col, stat, group_cols=("rho",))
			if agg is not None:
				ax.plot(np.log2(agg["rho"]), agg[col], marker="o", lw=2.5, ms=7, color=colour, label=label)
		ax.axhline(y=0.0, **BASELINE_LINE_KWARGS)
		_style_rho_panel(ax, "Δlog PEHE vs raw (lower = better)", reg)
	axes.flatten()[0].legend(fontsize=8)
	fig.suptitle(f"Plug-in vs pseudo-outcome PEHE ({stat})", y=0.995)
	_save_fig(fig, fig_dir, f"pehe_family_split_{stat}.png")


def generate_standard_plot_suite(full: pd.DataFrame, fig_dir: Path, trials: list[str]) -> None:
	"""All cross-trial figures for one (distribution, knob preset). Each plot no-ops when
	its column is absent, so the calls are unconditional. Reused for plots/, plots/ood/,
	plots/knob_variants/<preset>/."""
	fig_dir.mkdir(parents=True, exist_ok=True)
	pooled = _pooled(full)  # trial-pooled curves share this; warns once about stale `n`

	plot_pareto_frontier_grid(full, trials, fig_dir, value_col="PEHE_mean_dlog")
	for stat in ("mean", "median"):
		plot_capacity_curve_by_method(pooled, fig_dir, value_col="PEHE_mean_dlog", stat=stat)
	for col in ("PEHE_mean_dlog", "HET_loss_norm", "RICB_true_mean_norm"):
		plot_capacity_curve_by_trial(full, fig_dir, value_col=col, trials=trials)
	plot_capacity_overview(pooled, fig_dir)
	plot_pehe_family_split(pooled, fig_dir)

	for base in ("poly2", "gbm"):  # only present under --bases lr,poly2,gbm
		plot_capacity_curve_by_method(pooled, fig_dir, value_col=f"PEHE_mean_{base}_dlog", stat="median")
	for col in ("RICB_true_mean_norm", "HET_loss_norm"):
		plot_pareto_frontier_grid(full, trials, fig_dir, value_col=col, draw_frontier=False)
		plot_capacity_curve_by_method(pooled, fig_dir, value_col=col, stat="median")
