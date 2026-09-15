"""Shared per-trial oracle.py I/O and DataFrame-assembly scaffolding (the DGP-fitting
mechanisms themselves live in knobs.py/oracle_fit.py and deliberately differ per trial)."""

import json
import sys

import numpy as np
import polars as pl

from knobs import output_exists


def assemble_seed_df(seed, n: int, T_synthetic, Y_obs_synthetic, Y0, Y1, true_tau, source_dataset,
                      X: np.ndarray, feature_cols: list[str], ids=None, extra_cols: dict | None = None) -> pl.DataFrame:
    """Build one seed's oracle dataframe: the 8 columns every trial writes, plus feature_cols
    from X and any trial-specific extras (e.g. VASST's T_observed/propensity_score)."""
    ids = range(n) if ids is None else ids
    df = pl.DataFrame({
        "seed":            [seed] * n,
        "ID":              list(ids),
        "T_synthetic":     T_synthetic,
        "Y_obs_synthetic": Y_obs_synthetic,
        "Y0":              Y0,
        "Y1":              Y1,
        "true_tau":        true_tau,
        "source_dataset":  [source_dataset] * n if isinstance(source_dataset, str) else source_dataset,
        **(extra_cols or {}),
    })
    # Dtype is preserved as-is (not forced to float32): callers already cast upstream where
    # they want to (5/6 trials do; IHDP's raw covariates are float64 and were never cast).
    x_df = pl.DataFrame({col: np.asarray(X[:, i]) for i, col in enumerate(feature_cols)})
    return df.hstack(x_df)


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_feature_cols(out_dir: str, feature_cols: list[str]) -> None:
    _write_json(f"{out_dir}/feature_cols.json", feature_cols)


def write_causal_params(out_dir: str, causal_params: dict) -> None:
    """Writes causal_params.json. Callers print their own (trial-specific) confirmation
    message afterward, since some append extra detail (e.g. realized ATE)."""
    _write_json(f"{out_dir}/causal_params.json", causal_params)


def skip_if_exists(out_dir: str, skip_existing: bool) -> None:
    """Exit early if --skip-existing was passed and out_dir already has an oracle."""
    if skip_existing and output_exists(out_dir):
        print(f"{out_dir}/semi_synthetic.parquet already exists, skipping.")
        sys.exit(0)


def should_log_seed(seed: int) -> bool:
    """Progress-print gate shared by every oracle.py: log the first seed and every 5th."""
    return seed == 0 or (seed + 1) % 5 == 0
