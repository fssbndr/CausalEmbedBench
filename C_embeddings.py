# Fit and save all benchmark embeddings.
# ID:  fit on all patients          -> data/embeddings/{trial}[/knob_variants/<preset>]/
# OOD: fit on MIMIC-III+IV only,
#      transform all patients       -> data/embeddings_ood/{trial}[/knob_variants/<preset>]/
#
# Reads:  data/oracles/{trial}[/knob_variants/<preset>]/semi_synthetic.parquet
#         data/oracles/{trial}[/knob_variants/<preset>]/feature_cols.json
# Writes: data/embeddings/{trial}[/knob_variants/<preset>]/{raw,<method>_d*}.parquet
#         data/embeddings_ood/{trial}[/knob_variants/<preset>]/  (same)
#
# --knobs-preset selects which oracle-knob variant to read/write (default: baseline,
# the flat path with no knob_variants/ nesting — see B_oracles/shared/knobs.py's
# resolve_out_dir). Every method x dim [x seed] fit is dispatched as its own job
# through a single joblib.Parallel pool, so all cores stay busy regardless of how
# many seeds vs. methods there are. Pass --skip-existing to skip jobs whose output
# parquet already exists (useful for resuming a killed/crashed run).

import argparse
import json
import os
import sys

# Cap BLAS threads before numpy/torch/sklearn import so every worker process
# spawned by Parallel(prefer="processes") inherits it — without this, each
# concurrent job's own BLAS backend can spawn its own thread pool, oversubscribing
# far past N_JOBS workers x 1 thread each.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import polars as pl
import torch
from sklearn.utils.parallel import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "B_oracles", "shared"))
from config import CUDA_MIN_N, EPOCHS, SEEDS, available_cpus
from knobs import resolve_out_dir
from utils.C_embedding_utils import (
    SUPERVISED_METHODS,
    UNSUPERVISED_METHODS,
    embedding_filename,
    fit_ae,
    fit_bnn,
    fit_cevae,
    fit_cfrisw,
    fit_cfrnet,
    fit_dcn,
    fit_deeptreat,
    fit_dragonnet,
    fit_drcfr,
    fit_fa,
    fit_fastica,
    fit_nice,
    fit_pca,
    fit_random_projection,
    fit_site,
    fit_tedvae,
    fit_vae,
    k_grid_for_trial,
)

parser = argparse.ArgumentParser()
parser.add_argument("trial")
parser.add_argument("--skip-existing", action="store_true",
                    help="Skip embeddings whose output parquet already exists instead of recomputing them")
parser.add_argument("--knobs-preset", default="baseline",
                    help="Named oracle-knob preset the oracle was generated with (default: baseline). "
                         "Reads from and writes to the matching knob_variants/<preset>/ subdirectory.")
args = parser.parse_args()
TRIAL = args.trial
ORACLE_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/oracles")
EMB_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/embeddings")
EMB_OOD_DIR = resolve_out_dir(TRIAL, args.knobs_preset, base="data/embeddings_ood")

print("1. Loading oracle data")
with open(f"{ORACLE_DIR}/feature_cols.json") as f:
    feature_cols = json.load(f)

oracle  = pl.read_parquet(f"{ORACLE_DIR}/semi_synthetic.parquet")

# Load causal params (includes OOD sources)
with open(f"{ORACLE_DIR}/causal_params.json") as f:
    cp = json.load(f)
ood_sources = cp.get("ood_sources", [])
outcome_type = cp.get("outcome_type", "binary")
x_varies_by_seed = cp.get("x_varies_by_seed", False)

# Get unique IDs from first seed (same for all seeds; ID is just a positional index)
oracle_seed0 = oracle.filter(pl.col("seed") == 0)
ids     = oracle_seed0["ID"].to_numpy()

X_ref   = oracle_seed0.select(feature_cols).to_numpy().astype(np.float32)
source  = oracle_seed0["source_dataset"].to_numpy()

# OOD split: only if trial defines OOD sources (e.g., IHDP has empty list, VASST has eICU+Amsterdam)
if ood_sources:
    X_ood = X_ref[np.isin(source, ood_sources)]
else:
    X_ood = np.empty((0, X_ref.shape[1]), dtype=np.float32)

print(f"   X_ref: {X_ref.shape}  |  X_ood ({'/'.join(ood_sources) if ood_sources else 'none'}): {X_ood.shape[0]} patients")

# When X varies by seed (IHDP, NEWS), fit+transform every embedding on that seed's
# own X instead of the shared X_ref/X_ood above (kept only as seed-0 reference stats).
X_ref_by_seed, X_ood_by_seed = {}, {}
if x_varies_by_seed:
    for seed in SEEDS:
        oracle_seed = oracle.filter(pl.col("seed") == seed)
        X_seed = oracle_seed.select(feature_cols).to_numpy().astype(np.float32)
        source_seed = oracle_seed["source_dataset"].to_numpy()
        X_ref_by_seed[seed] = X_seed
        X_ood_by_seed[seed] = (
            X_seed[np.isin(source_seed, ood_sources)] if ood_sources
            else np.empty((0, X_seed.shape[1]), dtype=np.float32)
        )

# Load pre-computed oracle T and Y from seed 0 (for reference stats only)
T_ref  = oracle_seed0["T_synthetic"].to_numpy().astype(np.float32)
Y_ref  = oracle_seed0["Y_obs_synthetic"].to_numpy().astype(np.float32)
if X_ood.shape[0] > 0:
    T_ood = T_ref[np.isin(source, ood_sources)]
    Y_ood = Y_ref[np.isin(source, ood_sources)]
print(f"   mean(T)={T_ref.mean():.3f}  mean(Y)={Y_ref.mean():.3f}")

# Compute trial-specific k-grid (per-trial capacity hyperparameter, no k <= d restriction)
d_trial = X_ref.shape[1]
# NEWS uses fixed baseline of 400 for rho=1.0; other trials scale with d
baseline = 400 if TRIAL == "news" else None
K_GRID = k_grid_for_trial(d_trial, baseline=baseline)
print(f"   k-grid for {TRIAL} (d={d_trial}): {K_GRID}")


# ------------------------------------------------------------------------------
#   Method dispatch tables (name -> fit function)
# ------------------------------------------------------------------------------

def _fit_unsupervised(method: str, X: np.ndarray, d: int):
    if   method == "ae":         return fit_ae(               X, d, epochs=EPOCHS)
    elif method == "fa":         return fit_fa(               X, d)
    elif method == "fastica":    return fit_fastica(          X, d)
    elif method == "pca":        return fit_pca(              X, d)
    elif method == "randomproj": return fit_random_projection(X, d)
    elif method == "vae":        return fit_vae(              X, d, epochs=EPOCHS)
    raise ValueError(f"Unknown unsupervised method: {method}")


def _fit_supervised(method: str, X: np.ndarray, T: np.ndarray, Y: np.ndarray, d: int, outcome_type: str):
    if   method == "bnn":       return fit_bnn(      X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "cevae":     return fit_cevae(    X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "cfrisw":    return fit_cfrisw(   X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "cfrnet":    return fit_cfrnet(   X, T, Y, d=d, epochs=EPOCHS, alpha=1.0, outcome_type=outcome_type)
    elif method == "dcn":       return fit_dcn(      X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "deeptreat": return fit_deeptreat(X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "dragonnet": return fit_dragonnet(X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "drcfr":     return fit_drcfr(    X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "nice":      return fit_nice(     X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "site":      return fit_site(     X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    elif method == "tarnet":    return fit_cfrnet(   X, T, Y, d=d, epochs=EPOCHS, alpha=0.0, outcome_type=outcome_type)
    elif method == "tedvae":    return fit_tedvae(   X, T, Y, d=d, epochs=EPOCHS,            outcome_type=outcome_type)
    raise ValueError(f"Unknown supervised method: {method}")


# ------------------------------------------------------------------------------
#   Job dispatch (one fit per method x dim [x seed], run in a worker process)
# ------------------------------------------------------------------------------

def _run_job(job: dict, X_fit: np.ndarray, X_ref: np.ndarray, ids: np.ndarray, use_cuda: bool = False):
    """
    Fit one (method, dim[, seed]) embedding and return it (worker for parallelization).

    Unsupervised jobs are normally seed-independent: fit on X_fit, transform X_ref, and
    write their own final parquet directly (one file per method x dim, no merging needed).

    Supervised jobs train on (X_fit, T_fit, Y_fit) for a single seed, avoiding
    information leakage across seeds. They return their DataFrame instead of writing
    it, so the parent process can stack all seeds into one file per method x dim
    (matching the seed-stacked file format D_evaluation.py expects). Unsupervised jobs
    do the same (job carries a "seed" key) when X varies by seed.

    Returns:
        None for seed-independent unsupervised jobs (already written to disk), or
        (out_path, pl.DataFrame) otherwise (to be stacked by the caller)
    """
    torch.set_num_threads(1)  # many single-model jobs run concurrently; avoid oversubscribing cores
    if use_cuda:
        # sets every torch.tensor(...) call in utils/methods/*.py onto GPU (none pass device= explicitly)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    if job["kind"] == "unsupervised":
        try:
            _, model = _fit_unsupervised(job["method"], X_fit, job["d"])
        except (ValueError, np.linalg.LinAlgError):
            if job["method"] == "fastica":
                print(f"   WARNING: {job['method']} d={job['d']} skipped (numerical instability)")
                return None
            raise
        E = model.transform(X_ref).astype(np.float32)
        if "seed" in job:  # X varies by seed: stack like supervised jobs instead of writing directly
            df = pl.DataFrame({
                "seed": job["seed"], "ID": ids,
                **{f"dim_{i}": E[:, i] for i in range(E.shape[1])},
            })
            return (job["out_path"], df)
        pl.DataFrame({
            "ID": ids,
            **{f"dim_{i}": E[:, i] for i in range(E.shape[1])},
        }).write_parquet(job["out_path"])
        print(f"   Saved {job['out_path']}  shape={E.shape}")
        return None

    _, model = _fit_supervised(job["method"], X_fit, job["T_fit"], job["Y_fit"], job["d"], job["outcome_type"])
    E = model.transform(X_ref).astype(np.float32)
    df = pl.DataFrame({
        "seed": job["seed"],
        "ID": ids,
        **{f"dim_{i}": E[:, i] for i in range(E.shape[1])},
    })
    return (job["out_path"], df)


def build_jobs(out_dir: str, seed_args: list, outcome_type: str, skip_existing: bool, k_grid: list, n_features: int, n_samples: int, x_varies_by_seed: bool = False) -> list:
    """
    Flatten unsupervised (method x dim) and supervised (method x dim x seed) fits
    into a single job list to dispatch through one Parallel pool.

    When skip_existing, jobs whose output parquet already exists are dropped here
    (not inside the worker), so a resumed run doesn't even spawn a process for them.

    Args:
        out_dir: directory embedding parquets are written to
        seed_args: list of (seed, T_fit_seed, Y_fit_seed), see stage_seed_targets()
        outcome_type: "binary" or "continuous", per trial
        skip_existing: skip jobs whose out_path already exists
        k_grid: list of (k, rho) embedding-dimension/capacity-ratio pairs to test
        n_features: raw covariate dimensionality (for k > n_features guard)
        n_samples: number of samples (for k > n_samples guard on linear methods)
        x_varies_by_seed: if True, also dispatch unsupervised fits once per seed
            (each seed has its own X), instead of a single seed-independent fit

    Returns:
        list of job dicts, see _run_job()
    """
    jobs = []
    # Linear methods (PCA, FA, FastICA, RandomProjection) cannot support k > n_features
    linear_methods = {"pca", "fa", "fastica", "randomproj"}

    for method in UNSUPERVISED_METHODS:
        for d, rho in k_grid:
            # Linear methods cannot support k > min(n_samples, n_features)
            if method in linear_methods and d > min(n_samples, n_features):
                print(f"   Skipping {method} d={d} (exceeds min(n_samples, n_features)={min(n_samples, n_features)})")
                continue
            out_path = f"{out_dir}/{embedding_filename(method, d, rho)}"
            if skip_existing and os.path.exists(out_path):
                print(f"   Skipping {out_path} (already exists)")
                continue
            if x_varies_by_seed:
                for seed, _, _ in seed_args:
                    jobs.append({"kind": "unsupervised", "method": method, "d": d, "out_path": out_path, "seed": seed})
            else:
                jobs.append({"kind": "unsupervised", "method": method, "d": d, "out_path": out_path})

    for method in SUPERVISED_METHODS:
        for d, rho in k_grid:
            out_path = f"{out_dir}/{embedding_filename(method, d, rho)}"
            if skip_existing and os.path.exists(out_path):
                print(f"   Skipping {out_path} (already exists)")
                continue
            for seed, T_fit, Y_fit in seed_args:
                jobs.append({
                    "kind": "supervised", "method": method, "d": d, "seed": seed,
                    "T_fit": T_fit, "Y_fit": Y_fit, "outcome_type": outcome_type,
                    "out_path": out_path,
                })

    return jobs


def stage_seed_targets(fit_indices: np.ndarray) -> list:
    """
    Pre-stage (seed, T_fit_seed, Y_fit_seed) for every oracle seed (avoids re-filtering
    the oracle inside worker processes).

    Args:
        fit_indices: indices into the full patient array to subset T/Y to
            (e.g. MIMIC-III+IV only, for OOD)

    Returns:
        list of (seed, T_fit_seed, Y_fit_seed)
    """
    out = []
    for seed in SEEDS:
        oracle_seed = oracle.filter(pl.col("seed") == seed)
        T_seed = oracle_seed["T_synthetic"].to_numpy().astype(np.float32)
        Y_seed = oracle_seed["Y_obs_synthetic"].to_numpy().astype(np.float32)
        out.append((seed, T_seed[fit_indices], Y_seed[fit_indices]))
    return out


def run_embeddings(out_dir: str, X_fit: np.ndarray, seed_args: list, outcome_type: str, skip_existing: bool, k_grid: list, n_features: int,
                   x_varies_by_seed: bool = False, X_fit_by_seed: dict | None = None, X_transform_by_seed: dict | None = None) -> None:
    """
    Fit every embedding method x dim [x seed] for one population (ID or OOD).

    Uses joblib.Parallel, sized to available_cpus() (SLURM-allocation-aware, not the
    node total). Each (trial, preset) pair gets its own SLURM job — cluster-wide
    parallelism mostly comes from many such jobs running concurrently, but a job's
    allocation can still be >1 core, so this still fans out across whatever it gets.

    When x_varies_by_seed, X_fit/X_ref above are ignored in favor of X_fit_by_seed[seed]/
    X_transform_by_seed[seed] (every job, including unsupervised ones, is fit+transformed on
    its own seed's covariates instead of one shared X).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Raw features (no compression — reference upper bound)
    raw_path = f"{out_dir}/raw.parquet"
    if not (skip_existing and os.path.exists(raw_path)):
        if x_varies_by_seed:
            raw_dfs = [
                pl.DataFrame({
                    "seed": seed, "ID": ids,
                    **{f"dim_{i}": X_transform_by_seed[seed][:, i] for i in range(X_transform_by_seed[seed].shape[1])},
                })
                for seed, _, _ in seed_args
            ]
            raw_stacked = pl.concat(raw_dfs).sort("seed")
            raw_stacked.write_parquet(raw_path)
            print(f"   Saved {raw_path}  shape={raw_stacked.shape}")
        else:
            pl.DataFrame({
                "ID": ids,
                **{f"dim_{i}": X_ref[:, i] for i in range(X_ref.shape[1])},
            }).write_parquet(raw_path)
            print(f"   Saved {raw_path}  shape={X_ref.shape}")

    jobs = build_jobs(out_dir, seed_args, outcome_type, skip_existing, k_grid, n_features, X_fit.shape[0], x_varies_by_seed=x_varies_by_seed)
    if not jobs:
        print("   Nothing to do (all outputs already exist)")
        return

    N_JOBS = available_cpus()
    use_cuda = X_fit.shape[0] > CUDA_MIN_N
    print(f"   Dispatching {len(jobs)} embedding-fit jobs across {N_JOBS} workers{' (CUDA)' if use_cuda else ''}...")
    if x_varies_by_seed:
        results = Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(_run_job)(job, X_fit_by_seed[job["seed"]], X_transform_by_seed[job["seed"]], ids, use_cuda) for job in jobs
        )
    else:
        results = Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(_run_job)(job, X_fit, X_ref, ids, use_cuda) for job in jobs
        )

    # Stack supervised results (one file per method x dim, across all seeds)
    merged = {}
    for r in results:
        if r is None:
            continue
        out_path, df = r
        merged.setdefault(out_path, []).append(df)

    for out_path, dfs in merged.items():
        stacked = pl.concat(dfs).sort("seed")
        stacked.write_parquet(out_path)
        print(f"   Saved {out_path}  shape={stacked.shape}")


print("\n2. ID embeddings (fit on all patients)")
id_fit_indices = np.arange(len(X_ref))  # all patients
run_embeddings(EMB_DIR, X_ref, stage_seed_targets(id_fit_indices), outcome_type, args.skip_existing, K_GRID, d_trial,
               x_varies_by_seed=x_varies_by_seed, X_fit_by_seed=X_ref_by_seed, X_transform_by_seed=X_ref_by_seed)

if X_ood.shape[0] > 0:
    print(f"\n3. OOD embeddings (fit on {'/'.join(ood_sources)} only, transform all)")
    ood_fit_indices = np.where(np.isin(source, ood_sources))[0]
    run_embeddings(EMB_OOD_DIR, X_ood, stage_seed_targets(ood_fit_indices), outcome_type, args.skip_existing, K_GRID, d_trial,
                   x_varies_by_seed=x_varies_by_seed, X_fit_by_seed=X_ood_by_seed, X_transform_by_seed=X_ref_by_seed)

print("\nDone.")
