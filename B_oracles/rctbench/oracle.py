# Semi-synthetic oracle generation for the causal embedding benchmark (RCTBENCH trials).
#
# Reads:  raw/RCTBENCH/{meta_data,data-dictionary,trial{id}}.csv
# Writes: data/oracles/rctbench_trial{id}/semi_synthetic.parquet
#         data/oracles/rctbench_trial{id}/feature_cols.json
#         data/oracles/rctbench_trial{id}/causal_params.json
#         data/oracles/rctbench_trials.txt              (manifest of included trial ids)
#
# Each trial only has its own observed (X, T, Y), no published subgroup table to anchor
# against (unlike VASST). See oracle_fit.py: tau_0 (trial's own adjusted effect) sets the
# level, a shrunk per-arm MLP fit sets the heterogeneity shape.

import os
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from config import SEEDS
from covariates import build_covariate_matrix
from knobs import (
    FITTED_KNOBS,
    apply_knobs,
    draw_binary,
    draw_continuous,
    knob_params,
    output_exists,
    parse_oracle_args,
    resolve_out_dir,
)
from oracle_fit import fit_binary_oracle, fit_continuous_oracle
from oracle_io import assemble_seed_df, write_causal_params, write_feature_cols

args, knob_config = parse_oracle_args(
    skip_help="Skip trials whose output parquet already exists",
    knob_note="",
)

print("RCTBENCH Oracle Generation")
print("=" * 80)
print(f"Knob preset: '{args.knobs_preset}' -> {knob_config}")

RCTBENCH_PATH = "raw/RCTBENCH"
os.makedirs("data/oracles", exist_ok=True)

meta = pd.read_csv(f"{RCTBENCH_PATH}/meta_data.csv", sep=";")
dd = pd.read_csv(f"{RCTBENCH_PATH}/data-dictionary.csv", sep=";")

print(f"\nLoaded {len(meta)} trials, {len(dd)} variable definitions")

# Filter: 2-arm RCTs, binary/continuous outcomes, n≥40
# Rationale: evaluation tasks (PEHE, counterfactual, policy) require:
#   - Binary T (well-defined potential outcomes)
#   - Binary or continuous Y (PEHE/CF assume Y ∈ ℝ)
#   - n≥40 for covariate balance, outcome model stability
# Exclude: time-to-event (censoring), ordinal (ranking), multi-arm (arm selection),
#   composite (multiple outcomes), n<40 (sample size)
included_trials, skipped = [], {}
for _, row in meta.iterrows():
    trial_id = int(row["Trial_ID"])
    outcome_type = row["Primary Outcome Type"]
    n_arms = row["# of Arm"]
    n = int(row["Sample Size"])

    # Check: 2-arm only
    if not (np.isnan(n_arms) or n_arms == 2):
        reason = f"multi-arm ({int(n_arms)})"
        skipped[reason] = skipped.get(reason, 0) + 1
        continue

    # Check: binary or continuous outcome
    if outcome_type not in ["Binary", "Continuous"]:
        reason = f"outcome '{outcome_type}'"
        skipped[reason] = skipped.get(reason, 0) + 1
        continue

    # Check: sample size ≥ 40
    if n < 40:
        reason = f"n={n}<40"
        skipped[reason] = skipped.get(reason, 0) + 1
        continue

    included_trials.append({
        "trial_id": trial_id, "trial_num": row["Trial Number/Name"],
        "outcome_type": outcome_type, "control_group": row["Control Group"],
        "sample_size": n
    })

print(f"Included: {len(included_trials)}, Skipped: {sum(skipped.values())}")

def match_control_label(control_label, t_vals, n):
    """Match control label to treatment arm. Heuristic: exact -> substring -> fuzzy."""
    control_lower = str(control_label).lower()

    # Try exact match
    for v in t_vals:
        if str(v).lower() == control_lower:
            return v

    # Try substring match
    for v in t_vals:
        v_str = str(v).lower()
        if control_lower in v_str or v_str in control_lower:
            return v

    # Try fuzzy match (>60%)
    best_v = max(t_vals, key=lambda v: SequenceMatcher(None, control_lower, str(v).lower()).ratio())
    if SequenceMatcher(None, control_lower, str(best_v).lower()).ratio() >= 0.6:
        return best_v

    # For large trials, guess first arm; for small, fail
    return t_vals[0] if n >= 100 else None

print("\nProcessing trials:")
included_ids = []

for trial_info in included_trials:
    trial_id = trial_info["trial_id"]
    trial_num = trial_info["trial_num"]
    outcome_type = trial_info["outcome_type"]
    control_label = trial_info["control_group"]

    oracle_dir = resolve_out_dir(f"rctbench_trial{trial_id}", args.knobs_preset)

    if args.skip_existing and output_exists(oracle_dir):
        print(f"  {trial_num}: already exists, skipping")
        included_ids.append(f"rctbench_trial{trial_id}")
        continue

    print(f"  {trial_num}: ", end="", flush=True)

    df = pd.read_csv(f"{RCTBENCH_PATH}/trial{trial_id}.csv")
    trial_dd = dd[dd["Trial_ID"] == trial_id]

    # Get treatment and binarize
    t_defs = trial_dd[trial_dd["variable_role"] == "Treatment assignment"]
    if len(t_defs) == 0:
        print("skip")
        continue
    t_col = t_defs.iloc[0]["variable_name"]
    t_vals = [v for v in df[t_col].dropna().unique()]
    control_val = match_control_label(control_label, t_vals, trial_info["sample_size"])
    if control_val is None:
        print("skip")
        continue
    treatment = (df[t_col] != control_val).astype(int)

    # Get outcome
    o_defs = trial_dd[
        (trial_dd["variable_role"] == "Primary outcome") &
        (trial_dd["variable_type"].str.lower().str.contains(outcome_type.lower(), na=False))
    ]
    if len(o_defs) == 0:
        print("skip")
        continue
    outcome_col = o_defs.iloc[0]["variable_name"]

    if outcome_type == "Binary":
        binary_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}
        outcome = df[outcome_col].astype(str).str.lower().map(binary_map)
        outcome = outcome.fillna(pd.to_numeric(df[outcome_col], errors="coerce"))
    else:  # Continuous
        outcome = pd.to_numeric(df[outcome_col], errors="coerce")

    # Get and clean covariates (dictionary-driven encoding: see covariates.py)
    X = build_covariate_matrix(df, trial_dd)

    # Final validation
    valid_idx = treatment.notna() & outcome.notna()
    treatment, outcome, X = treatment[valid_idx].reset_index(drop=True), outcome[valid_idx].reset_index(drop=True), X[valid_idx].reset_index(drop=True)

    # Trials where treatment/outcome filtering leaves no valid rows have nothing left to
    # compute variance over (an empty column's .std() can return pd.NA depending on dtype,
    # which breaks the `== 0` comparison below) -- bail out before the zero-variance check.
    if len(X) < 40:
        print("skip")
        continue

    # Drop zero-variance covariates (e.g. eligibility/inclusion-criterion variables
    # constant across the whole trial): StandardScaler can't rescale them, and a constant
    # column makes X rank-deficient, which crashes FastICA/FactorAnalysis whitening.
    zero_var_cols = [c for c in X.columns if X[c].std() == 0]
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    if len(X.columns) < 3:
        print("skip")
        continue

    ate = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    print(f"n={len(X)}, d={len(X.columns)}, ATE={ate:.4f}")

    n_control, n_treated = int((treatment == 0).sum()), int((treatment == 1).sum())
    if n_control < 15 or n_treated < 15:
        print(f"  skip: arm too small (ctrl={n_control}, trt={n_treated})")
        continue

    # Standardize covariates (also written to the parquet as feature columns, like VASST's X_std)
    feature_cols = list(X.columns)
    X_std = StandardScaler().fit_transform(X.to_numpy().astype(np.float64))
    T_arr = treatment.to_numpy().astype(np.float64)
    Y_arr = outcome.to_numpy().astype(np.float64)

    _fit_kw = dict(
        alpha_alignment=knob_config.alpha_alignment,
        assignment_family=knob_config.assignment_family,
    )
    if outcome_type == "Binary":
        fitted_oracle = fit_binary_oracle(X_std, T_arr, Y_arr, **_fit_kw)
    else:
        fitted_oracle = fit_continuous_oracle(X_std, T_arr, Y_arr, **_fit_kw)

    # Apply post-fit oracle-parameterization knobs (knob_config loaded once at top from
    # --knobs-preset; alpha_alignment was already applied pre-fit above, inside
    # fit_binary/continuous_oracle)
    fitted_oracle = apply_knobs(fitted_oracle, knob_config)

    print(f"tau_0={fitted_oracle.tau_0:.4f}  "
          f"true_tau: mean={fitted_oracle.tau_fitted.mean():.4f} std={fitted_oracle.tau_fitted.std():.4f}  "
          f"fitted e(X): mean={fitted_oracle.propensity.mean():.3f} (pi_T={fitted_oracle.pi_T:.3f})")

    # Monte-Carlo draws of Y0/Y1/T (no row bootstrap, same X_std every seed; true_tau fixed)
    n = len(X)
    X_std_f32 = X_std.astype(np.float32)
    oracle_dfs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        draws = draw_binary(fitted_oracle, rng) if outcome_type == "Binary" else draw_continuous(fitted_oracle, rng)
        oracle_dfs.append(assemble_seed_df(
            seed, n, draws["T_synthetic"], draws["Y_obs_synthetic"], draws["Y0"], draws["Y1"],
            fitted_oracle.tau_fitted, f"RCTBench_trial{trial_id}", X_std_f32, feature_cols,
            extra_cols={"T_observed": T_arr.astype(np.float32), "Y_observed": Y_arr.astype(np.float32),
                        "propensity_score": fitted_oracle.propensity.astype(np.float32),
                        # deterministic E[Y|X,A=0] -- noise-free RICB regression target (see D_evaluation)
                        "mu0": fitted_oracle.mu0.astype(np.float32)},
        ))

    # Write oracle (oracle_dir computed earlier in the loop, namespaced by knobs preset)
    os.makedirs(oracle_dir, exist_ok=True)
    pl.concat(oracle_dfs).write_parquet(f"{oracle_dir}/semi_synthetic.parquet")

    write_feature_cols(oracle_dir, feature_cols)

    write_causal_params(oracle_dir, {
        "n": int(X_std.shape[0]),
        "outcome_type": outcome_type.lower(),
        "true_ATE": float(fitted_oracle.tau_fitted.mean()),
        "has_individual_cf": True,
        "ood_sources": [],
        **knob_params(knob_config, FITTED_KNOBS),
    })

    included_ids.append(f"rctbench_trial{trial_id}")

# Write manifest
with open("data/oracles/rctbench_trials.txt", "w") as f:
    for trial_id in included_ids:
        f.write(f"{trial_id}\n")

print(f"\nIncluded: {len(included_ids)} trials")
for trial_id in included_ids[:10]:
    print(f"  {trial_id}")
if len(included_ids) > 10:
    print(f"  ... +{len(included_ids) - 10} more")

print(f"\nSkipped ({sum(skipped.values())} total):")
for reason, count in sorted(skipped.items()):
    print(f"  {reason}: {count}")

print("\nManifest: data/oracles/rctbench_trials.txt")
