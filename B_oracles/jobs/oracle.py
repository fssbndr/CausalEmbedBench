# Semi-synthetic oracle generation for the causal embedding benchmark (JOBS trial).
#
# Reads:  raw/JOBS/jobs_DW_bin.new.10.{train,test}.npz
# Writes: data/oracles/jobs/semi_synthetic.parquet
#         data/oracles/jobs/feature_cols.json
#         data/oracles/jobs/causal_params.json
#
# JOBS (LaLonde/NSW Jobs program), via Shalit et al. 2016 (CFRNet): real observational
# outcome model. Covariates: 17 demographic/economic indicators, len(config.SEEDS) replications.
# Treatment: job training (binary). Outcome: unemployment status after the program (binary).
#
# The train/test npz splits combine a randomized experimental subsample with a non-randomized
# PSID comparison group (flagged by `e`); PSID rows are all-control with no valid
# counterfactual, so only the randomized subsample (e==1) is kept.
#
# Each replication's randomized subsample is structurally identical to an RCTBench trial
# (real RCT, X/T/Y, no published subgroup counterfactuals), so it's fit with the same
# T-learner (oracle_fit.fit_binary_oracle), giving JOBS real synthetic Y0(X)/Y1(X) and full
# knob support, matching RCTBench rather than the previous Y0=Y1=yf placeholder.

import os
import sys

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from config import SEEDS
from knobs import (
    FITTED_KNOBS,
    apply_knobs,
    draw_binary,
    knob_params,
    parse_oracle_args,
    setup_oracle_dir,
)
from oracle_fit import fit_binary_oracle
from oracle_io import assemble_seed_df, should_log_seed, skip_if_exists, write_causal_params, write_feature_cols

args, knob_config = parse_oracle_args(
    skip_help="Skip regeneration if the output parquet already exists",
    knob_note="All 5 knobs apply, like RCTBench.",
)

# ---------------------------------------------------------------------------
JOBS_PATH = "raw/JOBS"
TRIAL = "jobs"
OUT_DIR = setup_oracle_dir(TRIAL, args.knobs_preset)

skip_if_exists(OUT_DIR, args.skip_existing)

# ---------------------------------------------------------------------------
# 1. Load and merge the train/test JOBS NPZ splits (10 replications each)
# ---------------------------------------------------------------------------
print("1. Loading JOBS replications from NPZ (train + test)")

data_train = np.load(f"{JOBS_PATH}/jobs_DW_bin.new.10.train.npz")
data_test = np.load(f"{JOBS_PATH}/jobs_DW_bin.new.10.test.npz")

# Shapes: x=(n, 17, 10), t/yf/e=(n, 10); train n=2570, test n=642
x_all = np.concatenate([data_train["x"], data_test["x"]], axis=0)
t_all = np.concatenate([data_train["t"], data_test["t"]], axis=0)
yf_all = np.concatenate([data_train["yf"], data_test["yf"]], axis=0)
e_all = np.concatenate([data_train["e"], data_test["e"]], axis=0)

n_features = x_all.shape[1]
feature_cols = [f"x{i+1}" for i in range(n_features)]
write_feature_cols(OUT_DIR, feature_cols)

# ---------------------------------------------------------------------------
# 2. Fit a T-learner once, draw len(SEEDS) Monte-Carlo samples for stability
# ---------------------------------------------------------------------------
# The randomized-experiment subsample (e==1) is identical across all 10 NPZ "seed" columns
# (only the PSID/train-test split varies) — so fit once and Monte-Carlo sample, exactly like
# VASST/RCTBench, rather than redundantly refitting the same data 10 times.
print("2. Fitting T-learner oracle (randomized subsample only, PSID dropped)")

keep = e_all[:, 0] == 1
x = x_all[keep, :, 0]
t_raw = t_all[keep, 0].astype(np.float64)
yf = yf_all[keep, 0].astype(np.float64)
n = len(t_raw)

X_std = StandardScaler().fit_transform(x.astype(np.float64))
fitted_oracle = fit_binary_oracle(
    X_std, t_raw, yf,
    alpha_alignment=knob_config.alpha_alignment,
    assignment_family=knob_config.assignment_family,
)
fitted_oracle = apply_knobs(fitted_oracle, knob_config)

print(f"   n={n}  tau_0={fitted_oracle.tau_0:.4f}  "
      f"true_tau: mean={fitted_oracle.tau_fitted.mean():.4f}  "
      f"fitted e(X): mean={fitted_oracle.propensity.mean():.3f}")

x_f32 = x.astype(np.float32)
oracle_dfs = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    draws = draw_binary(fitted_oracle, rng)

    if should_log_seed(seed):
        print(f"   seed={seed:2d}  mean(T)={draws['T_synthetic'].mean():.3f}  "
              f"mean(Y0)={draws['Y0'].mean():.3f}  mean(Y1)={draws['Y1'].mean():.3f}")

    oracle_dfs.append(assemble_seed_df(
        seed, n, draws["T_synthetic"], draws["Y_obs_synthetic"], draws["Y0"], draws["Y1"],
        fitted_oracle.tau_fitted, "JOBS", x_f32, feature_cols,
        extra_cols={"T_observed": t_raw.astype(np.float32), "Y_observed": yf.astype(np.float32),
                    "propensity_score": fitted_oracle.propensity.astype(np.float32),
                    # deterministic E[Y|X,A=0] -- noise-free RICB regression target (see D_evaluation)
                    "mu0": fitted_oracle.mu0.astype(np.float32)},
    ))

# ---------------------------------------------------------------------------
# 3. Write causal parameters
# ---------------------------------------------------------------------------
print("3. Writing metadata")

global_ate = fitted_oracle.ate_fitted

causal_params = {
    "n": int(n),
    "outcome_type": "binary",
    "true_ATE": global_ate,
    "has_individual_cf": True,  # now fitted, unlike the previous Y0=Y1=yf placeholder
    "ood_sources": [],
    **knob_params(knob_config, FITTED_KNOBS),
}
write_causal_params(OUT_DIR, causal_params)

print(f"   Saved {OUT_DIR}/causal_params.json (has_individual_cf=true, global ATE={global_ate:.4f})")

# ---------------------------------------------------------------------------
# 4. Write Parquet
# ---------------------------------------------------------------------------
print(f"4. Writing {OUT_DIR}/semi_synthetic.parquet")

oracle_df = pl.concat(oracle_dfs)
oracle_df.write_parquet(f"{OUT_DIR}/semi_synthetic.parquet")
print(f"   Written {len(oracle_df)} rows × {oracle_df.width} columns "
      f"({len(SEEDS)} seeds × {n} patients, {n_features} features).")
print("   Done.")
