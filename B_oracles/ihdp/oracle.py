# Semi-synthetic oracle generation for the causal embedding benchmark (IHDP trial).
#
# Reads:  raw/IHDP/ihdp_npci_1-1000.train.npz
# Writes: data/oracles/ihdp/semi_synthetic.parquet
#         data/oracles/ihdp/feature_cols.json     (ordered feature column list)
#         data/oracles/ihdp/causal_params.json    (trial metadata)
#
# IHDP (Infant Health and Development Program) provides 1000 semi-synthetic replications
# with ground-truth potential outcomes (mu0, mu1). We use the first len(config.SEEDS) as seeds.
#
# Only gamma/s/nu (heterogeneity/scale/noise) apply here: mu0/mu1 are a real published
# response surface, directly analogous to VASST/RCTBench's fitted mu0/mu1. beta/alpha don't
# apply — T is the published, literature-standard assignment and must stay untouched (a
# fitted propensity or covariate split would break benchmark comparability). At the identity
# preset (knobs.py default), output is the exact published data, unchanged.

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from config import SEEDS
from knobs import (
    parse_oracle_args,
    PUBLISHED_KNOBS,
    knob_params,
    resample_published_outcome,
    setup_oracle_dir,
)
from oracle_io import assemble_seed_df, should_log_seed, skip_if_exists, write_causal_params, write_feature_cols

args, knob_config = parse_oracle_args(
    skip_help="Skip regeneration if the output parquet already exists",
    knob_note="Only gamma/s/nu have any effect for IHDP — T is the published, literature-standard assignment and stays untouched.",
)

# ---------------------------------------------------------------------------
IHDP_PATH = "raw/IHDP"
TRIAL = "ihdp"
OUT_DIR = setup_oracle_dir(TRIAL, args.knobs_preset)

skip_if_exists(OUT_DIR, args.skip_existing)

# ---------------------------------------------------------------------------
# 1. Load IHDP NPZ (1000 replications, use first len(SEEDS) as seeds)
# ---------------------------------------------------------------------------
print("1. Loading IHDP replications from NPZ")

data_train = np.load(f"{IHDP_PATH}/ihdp_npci_1-1000.train.npz")

# Shape: x=(672, 25, 1000), t=(672, 1000), mu0=(672, 1000), mu1=(672, 1000)
x_all = data_train["x"]
t_all = data_train["t"]
mu0_all = data_train["mu0"]
mu1_all = data_train["mu1"]
yf_all = data_train["yf"]
ycf_all = data_train["ycf"]

n_features = x_all.shape[1]

reps = []

for seed in SEEDS:
    x = x_all[:, :, seed]
    t_raw = t_all[:, seed]
    mu0_cont = mu0_all[:, seed]
    mu1_cont = mu1_all[:, seed]

    y0, y1, y_obs = resample_published_outcome(
        seed, t_raw, mu0_cont, mu1_cont, yf_all[:, seed], ycf_all[:, seed], knob_config,
    )

    reps.append({"x": x, "t": t_raw, "y": y_obs, "mu0": y0, "mu1": y1, "seed": seed})

print(f"   Loaded {len(reps)} replications (seeds {SEEDS[0]}-{SEEDS[-1]} from 1000 available)")

# ---------------------------------------------------------------------------
# 2. Build stacked oracle dataframe (seed × patients)
# ---------------------------------------------------------------------------
print("2. Building oracle dataframe")

feature_cols = [f"x{i}" for i in range(1, n_features + 1)]
dfs = []

for rep in reps:
    n = len(rep["t"])
    seed = rep["seed"]
    y0, y1 = rep["mu0"], rep["mu1"]
    true_tau = y1 - y0

    dfs.append(assemble_seed_df(
        seed, n, rep["t"].astype(np.float64), rep["y"], y0, y1, true_tau, "IHDP",
        rep["x"], feature_cols,
    ))

    if should_log_seed(seed):
        ate = np.mean(true_tau)
        print(f"   seed={seed:2d}  n={n}  ATE={ate:.4f}  "
              f"mean(Y0)={y0.mean():.3f}  mean(Y1)={y1.mean():.3f}  "
              f"mean(T)={rep['t'].mean():.3f}")

# ---------------------------------------------------------------------------
# 3. Assemble and write Parquet
# ---------------------------------------------------------------------------
print(f"3. Writing {OUT_DIR}/semi_synthetic.parquet")

oracle = pl.concat(dfs)
oracle.write_parquet(f"{OUT_DIR}/semi_synthetic.parquet")
print(f"   Written {len(oracle)} rows × {oracle.width} columns "
      f"({len(SEEDS)} seeds × 747 patients)")
print(f"   Global ATE: {oracle['true_tau'].mean():.4f}")

# ---------------------------------------------------------------------------
# 4. Write metadata
# ---------------------------------------------------------------------------
print("4. Writing metadata")

write_feature_cols(OUT_DIR, feature_cols)

causal_params = {
    "n": len(reps[0]["t"]),  # per-seed n
    "outcome_type": "continuous",
    "true_ATE": float(oracle["true_tau"].mean()),
    "ood_sources": [],
    "x_varies_by_seed": True,  # each seed is a distinct published replication, not a noise redraw
    **knob_params(knob_config, PUBLISHED_KNOBS, "published"),
}
write_causal_params(OUT_DIR, causal_params)

print(f"   Saved {OUT_DIR}/causal_params.json")
print("   Done.")
