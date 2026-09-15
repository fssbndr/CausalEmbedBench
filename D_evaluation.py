# Run all 5 benchmark evaluation tasks for every embedding × oracle seed.
# Writes both in-distribution (full cohort) and OOD (eICU+Amsterdam) results.
#
# Parallelism: one flat Parallel pool over (seed x embedding) jobs sized to
# available_cpus(). CV folds inside evaluate_cate run serially -- no nested pool.
#
# Reads:  data/oracles/{trial}[/knob_variants/<preset>]/semi_synthetic.parquet
#         data/oracles/{trial}[/knob_variants/<preset>]/feature_cols.json
#         data/oracles/{trial}[/knob_variants/<preset>]/causal_params.json
#         data/embeddings/{trial}[/knob_variants/<preset>]/*.parquet
#         data/embeddings_ood/{trial}[/knob_variants/<preset>]/*.parquet
# Writes: data/evaluations/{trial}[/knob_variants/<preset>]/{prediction,ate_preservation,
#                           pehe_cate,counterfactual,policy_value,propensity_overlap}.csv
#         data/evaluations_ood/{trial}[/knob_variants/<preset>]/  (same structure)
#
# --knobs-preset selects which oracle-knob variant to read/write (default: baseline).
# Anti-leakage: supervised embeddings evaluated on (T,Y) one seed offset from fit;
# X identical across seeds, only (T,Y) independent redraws -- except IHDP/NEWS
# (causal_params["x_varies_by_seed"]), where X/true_tau also vary per seed.

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cap BLAS threads before numpy/sklearn import so Parallel(prefer="processes") workers
# inherit it -- otherwise each worker's BLAS spawns its own pool and oversubscribes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils.parallel import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "B_oracles", "shared"))
from config import CV_FOLDS, SAMPLE_N, SEEDS, available_cpus
from knobs import resolve_out_dir
from utils.C_embedding_utils import SUPERVISED_METHODS, parse_embedding_name
from utils.D_evaluation_utils import (
    BASES,
    CATE_COLS,
    LEARNERS,
    OVERLAP_COLS,
    VALIDITY_COLS,
    evaluate_balance,
    evaluate_cate,
    evaluate_prediction,
    evaluate_treatment_tasks,
    fit_lr_nuisance_folds,
    fit_reference_nuisance,
    has_degenerate_arm,
)

parser = argparse.ArgumentParser()
parser.add_argument("trial")
parser.add_argument("--knobs-preset", default="baseline",
                     help="Named oracle-knob preset the oracle/embeddings were generated with "
                          "(default: baseline).")
parser.add_argument("--bases", default=None,
                     help="Comma-separated CATE base learners: lr (default), poly2, gbm. "
                          "Pass 'lr,poly2,gbm' to add PEHE_{learner}_poly2 / _gbm columns.")
args = parser.parse_args()
TRIAL = args.trial
CLI_BASES = args.bases.split(",") if args.bases else None
ORACLE_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/oracles")
EMB_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/embeddings")
EMB_OOD_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/embeddings_ood")
EVAL_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/evaluations")
EVAL_OOD_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/evaluations_ood")
N_OUTER = available_cpus()

TASK_FILES = [
    ("pred",    "prediction.csv"),
    ("ate",     "ate_preservation.csv"),
    ("cate",    "pehe_cate.csv"),
    ("cf",      "counterfactual.csv"),
    ("policy",  "policy_value.csv"),
    ("overlap", "propensity_overlap.csv"),
]


def _method_of(name: str) -> str:
    parsed = parse_embedding_name(name)
    return parsed[0] if parsed is not None else name  # "raw" (or anything unparsed) -> itself


def _load_embeddings(emb_dir: Path, oracle_sorted: pl.DataFrame, seed: int | None = None, fit_seed: int | None = None) -> dict[str, np.ndarray]:
    """Load embeddings from directory, filtering seed-stacked files by seed.

    seed=None: seed-independent files only. seed=int: seed-stacked files filtered to
    `seed`, except supervised-method files go to `fit_seed` if given (anti-leakage --
    unsupervised fits carry no T/Y leakage risk, so they stay on `seed`).
    """
    embeddings = {}

    # ID-check reference: seed 0 for seed-independent files, else that seed.
    oracle_ref = oracle_sorted.filter(pl.col("seed") == (seed if seed is not None else 0))

    for emb_path in sorted(emb_dir.glob("*.parquet")):
        emb_df = pl.read_parquet(emb_path)
        has_seed_col = "seed" in emb_df.columns
        if has_seed_col != (seed is not None):
            continue  # schema doesn't match what this call is looking for
        if has_seed_col:
            target_seed = fit_seed if (fit_seed is not None and _method_of(emb_path.stem) in SUPERVISED_METHODS) else seed
            emb_df = emb_df.filter(pl.col("seed") == target_seed)

        emb_df = emb_df.sort("ID")
        assert (oracle_ref["ID"] == emb_df["ID"]).all(), \
            f"ID mismatch in {emb_path.stem}"
        dim_cols = [c for c in emb_df.columns if c not in ("ID", "seed")]
        E = emb_df.select(dim_cols).to_numpy().astype(np.float64)

        n_bad = int((~np.isfinite(E)).any(axis=1).sum())
        if n_bad:
            print(f"   WARNING: {emb_path.stem} has {n_bad}/{len(E)} rows with NaN/Inf — skipping this embedding", flush=True)
            continue

        embeddings[emb_path.stem] = E
    return embeddings


def _eval_one(name: str, E_X: np.ndarray, T, Y, oracle: dict, ref: tuple, seed: int,
              outcome_type: str = "binary", cv: int = 3, has_individual_cf: bool = True,
              fit_seed: int | None = None, bases: list | None = None) -> dict:
    """Evaluate one (embedding, seed) pair."""
    method = _method_of(name)
    meta = {"embedding": name, "method": method, "dim": E_X.shape[1], "seed": seed,
            "fit_seed": fit_seed if fit_seed is not None else seed}
    t0 = time.time()

    # Balance needs only T/Z -- compute unconditionally so degenerate-arm seeds still get it.
    r_bal = evaluate_balance(E_X, T)

    # Tasks 2-6 need both Y classes in both arms (a small OOD subset can lack them,
    # unfixable by CV reshuffling). Skip rather than fit degenerate models.
    degenerate = has_degenerate_arm(T, Y, outcome_type=outcome_type)
    run_cf = has_individual_cf and not degenerate

    # Task 1: always compute outcome prediction
    r_pred = evaluate_prediction(E_X, Y, outcome_type=outcome_type, cv=cv)

    # Tasks 2-6: conditional on run_cf
    if run_cf:
        lr_folds = fit_lr_nuisance_folds(E_X, T, Y, outcome_type=outcome_type, cv=cv)
        r_cate = evaluate_cate(E_X, T, Y, lr_folds, oracle,
                               outcome_type=outcome_type, cv=cv, bases=bases)
        r_ovl = {**evaluate_treatment_tasks(E_X, T, Y, lr_folds, ref, cv=cv), **r_bal}
        # ATE / policy / CF are computed in evaluate_cate (REF_BASE nuisances) -- split into buckets.
        r_ate = {k: r_cate.pop(k) for k in ("true_ATE", "ATE_hat", "ATE_error", "ATE_SE",
                                            "true_ATT", "ATT_hat", "ATT_error")}
        r_pol = {k: r_cate.pop(k) for k in ("policy_value", "oracle_policy_value", "regret",
                                            "regret_weighted", "policy_accuracy")}
        r_cf  = {"CF_RMSE": r_cate.pop("CF_RMSE")}
    else:
        cate_cols = [f"PEHE_{l}_{b}" for l in LEARNERS for b in (bases or BASES)]
        r_cate = {**{col: np.nan for col in cate_cols + VALIDITY_COLS + [
                      "PEHE", "tau_hat_mean", "tau_recovery_R2",
                      "PEHE_rselected", "PEHE_drselected",
                      "cal_alpha", "cal_beta", "ECE_CATE"]},
                  "rselected": "", "drselected": "",
                  **{f"{t}selected_{b}": "" for t in ("r", "dr") for b in (bases or BASES)}}
        r_ate = {"ATE_error": np.nan, "ATE_SE": np.nan, "ATT_error": np.nan}
        r_cf  = {"CF_RMSE": np.nan}
        r_pol = {"policy_value": np.nan, "oracle_policy_value": np.nan, "regret": np.nan,
                 "regret_weighted": np.nan, "policy_accuracy": np.nan}
        r_ovl = {**{k: np.nan for k in OVERLAP_COLS}, **r_bal}

    # Print: same format for both paths
    pred_key = "AUROC" if outcome_type == "binary" else "R2"
    pred_val = r_pred[pred_key]
    if run_cf:
        status = f"ATE={r_ate['ATE_error']:7.4f}  PEHE={r_cate['PEHE']:7.4f}  regret={r_pol['regret']:7.4f}"
    elif degenerate:
        status = "skipped CF: one arm has zero-variance Y in this seed's draw"
    else:
        status = "no CF available"
    print(f"  seed={seed:2.0f}  {name:<25}  {time.time()-t0:3.0f}s  {pred_key}={pred_val:6.3f}  {status}", flush=True)

    return {
        "pred":    {**meta, **r_pred},
        "cate":    {**meta, **r_cate},
        "ate":     {**meta, **r_ate},
        "cf":      {**meta, **r_cf},
        "policy":  {**meta, **r_pol},
        "overlap": {**meta, **r_ovl},
    }


def _collect(results: list[dict]) -> dict:
    acc = {task: [] for task, _ in TASK_FILES}
    for r in results:
        for task in acc:
            acc[task].append(r[task])
    return acc


def _write_csvs(acc: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for task, fname in TASK_FILES:
        pd.DataFrame(acc[task]).to_csv(f"{out_dir}/{fname}", index=False)


# ------------------------------------------------------------------------------
# 1. Load oracle
# ------------------------------------------------------------------------------
print("1. Loading oracle")
oracle = pl.read_parquet(f"{ORACLE_DIR}/semi_synthetic.parquet")
with open(f"{ORACLE_DIR}/feature_cols.json") as f:
    feature_cols = json.load(f)
with open(f"{ORACLE_DIR}/causal_params.json") as f:
    cp = json.load(f)

# Use seed 0 for reference (X, true_tau identical across seeds); when x_varies_by_seed,
# per-seed X/true_tau come from draws[seed] instead.
oracle_seed0 = oracle.filter(pl.col("seed") == 0)
X_std    = oracle_seed0.select(feature_cols).to_numpy().astype(np.float64)
true_tau = oracle_seed0["true_tau"].to_numpy().astype(np.float64)
print(f"   N={len(X_std)}  true_ATE={true_tau.mean():.4f}  X_std shape={X_std.shape}")

# Deterministic oracle quantities for the RICB (mu0 -> noise-free arm-wise plug-in;
# propensity_score -> PCR_oracle). Seed-invariant like true_tau; absent for real-assignment
# trials (IHDP/NEWS/TWINS), which fall back to the noisy plug-in.
DET = {} if cp.get("x_varies_by_seed", False) else {
    k: oracle_seed0[c].to_numpy().astype(np.float64)
    for k, c in (("mu0", "mu0"), ("ps", "propensity_score"))
    if c in oracle_seed0.columns}


def _orc(d: dict, tau: np.ndarray, i) -> dict:
    """Ground-truth sub-dict passed to evaluate_cate for rows `i` of oracle draw `d`."""
    return {"tau": tau, "ATE": float(tau.mean()), "Y0": d["Y0"][i], "Y1": d["Y1"][i],
            **{k: v[i] for k, v in DET.items()}}

# Read OOD sources and outcome_type from causal params
source            = oracle_seed0["source_dataset"].to_numpy()
ood_sources       = cp.get("ood_sources", [])
outcome_type      = cp.get("outcome_type", "binary")
has_individual_cf = cp.get("has_individual_cf", True)  # Default: assume CF available
x_varies_by_seed  = cp.get("x_varies_by_seed", False)
ood_idx           = np.where(np.isin(source, ood_sources))[0] if ood_sources else np.array([], dtype=int)
true_tau_ood      = true_tau[ood_idx] if len(ood_idx) > 0 else np.array([], dtype=np.float64)

if x_varies_by_seed and ood_sources:
    raise NotImplementedError("OOD evaluation isn't implemented for x_varies_by_seed trials")

if len(ood_idx) > 0:
    print(f"   OOD patients ({'/'.join(ood_sources)}): {len(ood_idx)} / {len(source)}")
else:
    print("   OOD: not applicable for this trial")
print(f"   Outcome type: {outcome_type}")

# ------------------------------------------------------------------------------
# 2. Setup for per-seed evaluation
# ------------------------------------------------------------------------------
oracle_sorted = oracle.sort("ID")
id_emb_dir = Path(EMB_DIR)
ood_emb_dir = Path(EMB_OOD_DIR)

# Load oracle draws (T and Y for each seed)
print(f"\n2. Loading {len(SEEDS)} oracle draws from semi_synthetic.parquet  "
      f"(SAMPLE_N={SAMPLE_N if SAMPLE_N else 'full'}, CV_FOLDS={CV_FOLDS})")
oracle_full = oracle
draws = {}

for seed in SEEDS:
    oracle_seed = oracle_full.filter(pl.col("seed") == seed)
    idx = (
        np.random.default_rng(seed).choice(len(oracle_seed), size=SAMPLE_N, replace=False)
        if SAMPLE_N
        else np.arange(len(oracle_seed))
    )
    draws[seed] = {
        "idx": idx,
        "T":   oracle_seed["T_synthetic"].to_numpy().astype(np.float64),
        "Y":   oracle_seed["Y_obs_synthetic"].to_numpy().astype(np.float64),
        "Y0":  oracle_seed["Y0"].to_numpy().astype(np.float64),
        "Y1":  oracle_seed["Y1"].to_numpy().astype(np.float64),
    }
    if x_varies_by_seed:
        draws[seed]["X"]        = oracle_seed.select(feature_cols).to_numpy().astype(np.float64)
        draws[seed]["true_tau"] = oracle_seed["true_tau"].to_numpy().astype(np.float64)
    d = draws[seed]
    if seed % 10 == 0:
        print(f"   seed={seed:2d}  n={len(idx)}  "
              f"mean(T)={d['T'][idx].mean():.3f}  mean(Y)={d['Y'][idx].mean():.3f}")

# Raw-covariate reference nuisances (e(X), m0(X), m1(X)): fit once per seed, reused.
nuisance_ref_by_seed = {
    seed: fit_reference_nuisance(
        (draws[seed]["X"] if x_varies_by_seed else X_std)[draws[seed]["idx"]],
        draws[seed]["T"][draws[seed]["idx"]],
        draws[seed]["Y"][draws[seed]["idx"]],
        outcome_type=outcome_type, cv=CV_FOLDS)
    for seed in SEEDS
}

# Load unsupervised embeddings once (seed-independent)
print(f"\n3. Loading ID unsupervised embeddings from {id_emb_dir}")
id_unsupervised = _load_embeddings(id_emb_dir, oracle_sorted, seed=None)
print(f"   {len(id_unsupervised)} unsupervised embeddings")

print(f"   Loading OOD unsupervised embeddings from {ood_emb_dir}")
ood_unsupervised = _load_embeddings(ood_emb_dir, oracle_sorted, seed=None)
print(f"   {len(ood_unsupervised)} unsupervised embeddings")

# ------------------------------------------------------------------------------
# 4. Evaluate all (seed × embedding) pairs — ID
#    Load embeddings per seed, construct all jobs, parallelize over all seeds
#    and embeddings in one go via Parallel.
# ------------------------------------------------------------------------------
print(f"\n4. Evaluating ID embeddings ({len(SEEDS)} seeds)")
id_jobs = []

for seed in SEEDS:
    # Anti-leakage: evaluate a supervised embedding on a different (T,Y) draw than it
    # was fit on. X is shared across seeds, so fit_seed != eval_seed costs nothing.
    fit_seed = (seed + 1) % len(SEEDS)
    id_seeded = _load_embeddings(id_emb_dir, oracle_sorted, seed=seed, fit_seed=fit_seed)
    id_embeddings = {**id_unsupervised, **id_seeded}

    # Create jobs for this seed × all embeddings
    d = draws[seed]
    idx = d["idx"]
    ref_nuis = nuisance_ref_by_seed[seed]
    seed_tau = d["true_tau"][idx] if x_varies_by_seed else true_tau[idx]
    id_jobs.extend([
        {"name": name, "E_X": E_X[idx], "T": d["T"][idx], "Y": d["Y"][idx],
         "oracle": _orc(d, seed_tau, idx), "seed": seed,
         "ref": ref_nuis, "bases": CLI_BASES,
         "fit_seed": fit_seed if _method_of(name) in SUPERVISED_METHODS else seed}
        for name, E_X in id_embeddings.items()
    ])

    # raw features with all tuned nuisance tiers -> E_leaderboard's baseline_slack check
    # and the dlog reference (raw_tuned's PEHE_mean must span the same tiers as candidates).
    if "raw" in id_embeddings:
        id_jobs.append(
            {"name": "raw_tuned", "E_X": id_embeddings["raw"][idx],
             "T": d["T"][idx], "Y": d["Y"][idx], "oracle": _orc(d, seed_tau, idx), "seed": seed,
             "ref": ref_nuis,
             "fit_seed": seed, "bases": BASES}
        )

# Evaluate all jobs in parallel
id_results = Parallel(n_jobs=N_OUTER, prefer="processes")(
    delayed(_eval_one)(**job, outcome_type=outcome_type, cv=CV_FOLDS, has_individual_cf=has_individual_cf) for job in id_jobs
)

id_acc = _collect(id_results)
_write_csvs(id_acc,  EVAL_DIR)

# ------------------------------------------------------------------------------
# 5. Evaluate OOD (if applicable)
# ------------------------------------------------------------------------------
print(f"\n5. Evaluating OOD embeddings ({len(SEEDS)} seeds)")
if len(ood_idx) == 0:
    print("   OOD: not applicable")
else:
    ood_ATE = float(true_tau_ood.mean())
    ood_jobs = []
    for seed in SEEDS:
        fit_seed = (seed + 1) % len(SEEDS)  # anti-leakage, same as ID loop above
        ood_supervised = _load_embeddings(ood_emb_dir, oracle_sorted, seed=fit_seed)
        ood_embeddings = {**ood_unsupervised, **ood_supervised}
        d = draws[seed]
        # same physical rows as ID, sliced to the OOD subset -- no refit
        ref_nuis = tuple(a[ood_idx] for a in nuisance_ref_by_seed[seed])
        ood_jobs.extend([
            {"name": name, "E_X": E_X_full[ood_idx], "T": d["T"][ood_idx], "Y": d["Y"][ood_idx],
             "oracle": _orc(d, true_tau_ood, ood_idx), "seed": seed,
             "ref": ref_nuis, "bases": CLI_BASES,
             "fit_seed": seed if name in ood_unsupervised else fit_seed}
            for name, E_X_full in ood_embeddings.items()
        ])

    ood_results = Parallel(n_jobs=N_OUTER, prefer="processes")(
        delayed(_eval_one)(**job, outcome_type=outcome_type, cv=CV_FOLDS, has_individual_cf=has_individual_cf)
        for job in ood_jobs
    )
    ood_acc = _collect(ood_results)
    _write_csvs(ood_acc, EVAL_OOD_DIR)

print("   Done. Run E_leaderboard.py to aggregate results.")
