# Semi-synthetic oracle generation for the causal embedding benchmark (TWINS trial).
#
# Reads:  raw/TWINS/twin_pairs_{X,T,Y}_3years_samesex.csv
# Writes: data/oracles/twins/semi_synthetic.parquet
#         data/oracles/twins/feature_cols.json
#         data/oracles/twins/causal_params.json
#
# TWINS (Almond et al. 2005), as an observational-study benchmark following
# Louizos et al. 2017 (CEVAE), NeurIPS: real dual-outcome data from US twin
# births 1989-1991. Each row is a twin pair; both twins' birth weights and
# 3-year mortality are observed.
# Treatment: T=1 for being born the heavier twin.
# Outcome: 3-year infant mortality (binary).
#
# Ground truth: Y0 = mortality of lighter twin, Y1 = mortality of heavier twin.
# Both are real, observed outcomes for every pair. To simulate an
# observational (confounded) study, treatment assignment is redrawn per seed
# as a logistic function of GESTAT10 (gestation weeks category) plus noise on
# the other covariates (CEVAE Sec 5.2), rather than a fixed/degenerate
# assignment — only the resulting factual outcome is "observed" per draw.

import dataclasses
import os
import sys

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from config import SEEDS
from knobs import (
    overlap_transform,
    parse_oracle_args,
    sample_treatment,
    setup_oracle_dir,
)
from oracle_io import (
    assemble_seed_df,
    should_log_seed,
    skip_if_exists,
    write_causal_params,
    write_feature_cols
)

args, knob_config = parse_oracle_args(
    skip_help="Skip regeneration if the output parquet already exists",
    knob_note="Only beta (overlap) has any effect for TWINS — Y0/Y1 are real historical outcomes, never perturbed.",
)

# ---------------------------------------------------------------------------
TWINS_PATH = "raw/TWINS"
TRIAL = "twins"
OUT_DIR = setup_oracle_dir(TRIAL, args.knobs_preset)

skip_if_exists(OUT_DIR, args.skip_existing)

# ---------------------------------------------------------------------------
# 1. Load the three files: X (covariates), T (birth weights), Y (mortality)
# ---------------------------------------------------------------------------
print("1. Loading TWINS data")

X_df = pl.read_csv(f"{TWINS_PATH}/twin_pairs_X_3years_samesex.csv")
T_df = pl.read_csv(f"{TWINS_PATH}/twin_pairs_T_3years_samesex.csv")
Y_df = pl.read_csv(f"{TWINS_PATH}/twin_pairs_Y_3years_samesex.csv")

# Drop the index columns (unnamed CSV index + NBER infant_id, marked "index
# do not use" in covar_type.txt)
X_df = X_df.drop("", "Unnamed: 0", "infant_id_0", "infant_id_1")
T_df = T_df.drop("")
Y_df = Y_df.drop("")

# CEVAE restricts to pairs where both twins weigh <2kg (outcome is otherwise
# too rare to be informative); this also matches their reported n=11,984.
both_under_2kg = (T_df["dbirwt_0"] < 2000) & (T_df["dbirwt_1"] < 2000)
X_df = X_df.filter(both_under_2kg)
Y_df = Y_df.filter(both_under_2kg)

n = len(X_df)
print(f"   X shape: {X_df.shape}  (both twins <2kg, matches CEVAE n=11,984)")

# ---------------------------------------------------------------------------
# 2. Extract mortality potential outcomes (both are real, observed values)
# ---------------------------------------------------------------------------
print("2. Extracting potential outcomes")

Y0 = Y_df["mort_0"].to_numpy().astype(np.int8)  # lighter twin mortality
Y1 = Y_df["mort_1"].to_numpy().astype(np.int8)  # heavier twin mortality
true_tau = (Y1 - Y0).astype(np.float32)

print(f"   mean(Y0): {Y0.mean():.3f}  (lighter twin mortality)")
print(f"   mean(Y1): {Y1.mean():.3f}  (heavier twin mortality)")
print(f"   ATE (mean tau): {true_tau.mean():.4f}")

# ---------------------------------------------------------------------------
# 3. Encode covariates (z-score for the propensity model below)
# ---------------------------------------------------------------------------
print("3. Encoding covariates")

# CEVAE doesn't specify missing-data handling for TWINS; several covariates
# have substantial null rates (e.g. orfath, feduc6, tobacco), so mean-fill.
X_encoded = X_df.fill_null(strategy="mean")
feature_cols = list(X_encoded.columns)

gestat10 = X_encoded["gestat10"].to_numpy().astype(np.float64)
other_cols = [c for c in feature_cols if c != "gestat10"]
X_other = X_encoded.select(other_cols).to_numpy().astype(np.float64)
# Raw covariates span wildly different scales (binary flags vs. state codes),
# so standardize before applying the small fixed-scale N(0, 0.1) weights
# below — otherwise large-scale columns would dominate the propensity score.
X_other = (X_other - X_other.mean(axis=0)) / (X_other.std(axis=0) + 1e-8)

print(f"   Feature columns ({len(feature_cols)}): {feature_cols[:5]}... (first 5)")
print(f"   X_encoded shape: {X_encoded.shape}")

write_feature_cols(OUT_DIR, feature_cols)

# ---------------------------------------------------------------------------
# 4. Write causal parameters (outcome_type + trial metadata)
# ---------------------------------------------------------------------------
print("4. Writing metadata")

causal_params = {
    "n":             int(X_encoded.shape[0]),
    "outcome_type":  "binary",
    "true_ATE":      float(true_tau.mean()),
    "ood_sources":   [],
    "knobs": dataclasses.asdict(knob_config),
    "active_knobs": ["kappa_overlap"],  # Y0/Y1 are real historical outcomes, never perturbed
    "active_assignment_family": knob_config.assignment_family,
}
write_causal_params(OUT_DIR, causal_params)

print(f"   Saved {OUT_DIR}/causal_params.json")

# ---------------------------------------------------------------------------
# 5. Simulate confounded treatment assignment (10 seeds)
# ---------------------------------------------------------------------------
# t_i | x_i, z_i ~ Bernoulli(sigmoid(w_o^T x_i + w_h * (z_i/10 - 0.1)))
# w_o ~ N(0, 0.1*I) over the other covariates, w_h ~ N(5, 0.1), z = GESTAT10.
# Redrawing (w_o, w_h, t) per seed gives 10 independent confounded draws over
# the same fixed pair of potential outcomes (Y0, Y1).
print("5. Generating 10 confounded treatment-assignment draws")

n_other = X_other.shape[1]
X_vals = X_encoded.to_numpy().astype(np.float32)
oracle_dfs = []

# assignment_family="tree": build a reference logistic assignment once, then fit a
# RandomForest e(X) on it -- a non-smooth boundary for the congeniality-triangulation
# independence check. p_tree_base is seed-independent; only the Bernoulli draw varies.
p_tree_base = None
if knob_config.assignment_family == "tree":
    _rng0 = np.random.default_rng(0)
    _logits0 = X_other @ _rng0.normal(0, 0.1, size=n_other) + _rng0.normal(5, 0.1) * (gestat10 / 10 - 0.1)
    _t_ref = _rng0.binomial(1, 1 / (1 + np.exp(-_logits0)))
    _feat = np.column_stack([X_other, gestat10 / 10 - 0.1])
    _rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=max(5, len(_t_ref) // 100),
                                 random_state=42, n_jobs=-1).fit(_feat, _t_ref)
    p_tree_base = np.clip(_rf.predict_proba(_feat)[:, 1], 1e-8, 1 - 1e-8)

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    if p_tree_base is not None:
        p_treat = overlap_transform(p_tree_base, knob_config.kappa_overlap)
    else:
        w_o = rng.normal(0, 0.1, size=n_other)
        w_h = rng.normal(5, 0.1)
        logits = X_other @ w_o + w_h * (gestat10 / 10 - 0.1)
        p_treat = overlap_transform(1 / (1 + np.exp(-logits)), knob_config.kappa_overlap)
    T_synthetic = sample_treatment(p_treat, rng)

    Y_obs_synthetic = np.where(T_synthetic == 1, Y1, Y0).astype(np.float32)

    oracle_dfs.append(assemble_seed_df(
        seed, n, T_synthetic, Y_obs_synthetic, Y0.astype(np.float32), Y1.astype(np.float32),
        true_tau, "TWINS", X_vals, feature_cols,
    ))

    if should_log_seed(seed):
        print(f"   seed={seed:2d}  n={n}  mean(T)={T_synthetic.mean():.3f}  "
              f"ATE={true_tau.mean():.4f}  "
              f"mean(Y0)={Y0.mean():.3f}  mean(Y1)={Y1.mean():.3f}")

# ---------------------------------------------------------------------------
# 6. Write Parquet (all seeds, assembled in the loop above)
# ---------------------------------------------------------------------------
print(f"6. Writing {OUT_DIR}/semi_synthetic.parquet")
oracle_df = pl.concat(oracle_dfs)
oracle_df.write_parquet(f"{OUT_DIR}/semi_synthetic.parquet")
print(f"   Written {len(oracle_df)} rows × {oracle_df.width} columns "
      f"({len(SEEDS)} seeds × {n} patients).")
print("   Done.")
