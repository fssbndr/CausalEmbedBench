# Dictionary-driven covariate encoding for RCTBENCH trials.
#
# RCTBENCH's data-dictionary.csv carries the R source data's own typing
# (variable_type/r_class/levels_or_range) per variable. That's ground truth for how a
# column should be encoded -- pandas' inferred CSV dtype is not: many nominal factors
# (e.g. hospital-site ids) are stored as small integers and would otherwise silently be
# treated as continuous, and R's ordered-factor level order (in levels_or_range) is lost
# if every factor is blindly one-hot encoded.

import numpy as np
import pandas as pd


def _normalize_missing(s: pd.Series) -> pd.Series:
    """Empty/whitespace-only strings are missing, not their own category."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    stripped = s.astype("string").str.strip()
    return stripped.mask(stripped == "", other=pd.NA)


def _mode_fill(s: pd.Series, default=np.nan) -> pd.Series:
    """Fill missing values with the column's mode (or `default` if every value is missing,
    e.g. NaN to leave an all-missing numeric column as-is rather than type-punning in a
    string sentinel)."""
    mode = s.mode()
    return s.fillna(mode.iloc[0] if len(mode) > 0 else default)


def _encode_continuous(s: pd.Series) -> pd.Series:
    """Coerce to numeric and median-impute."""
    numeric = pd.to_numeric(s, errors="coerce")
    return numeric.fillna(numeric.median())


def build_covariate_matrix(df: pd.DataFrame, trial_dd: pd.DataFrame, max_factor_levels: int = 15) -> pd.DataFrame:
    """Build the standardization-ready covariate matrix for one trial, using the data
    dictionary's variable_type/r_class/levels_or_range to decide encoding instead of
    pandas' inferred dtype."""
    dd_cov = trial_dd[trial_dd["variable_role"] == "Baseline covariate"].drop_duplicates("variable_name").set_index("variable_name")

    cov_cols = [c for c in dd_cov.index if c in df.columns]
    X = df[cov_cols].copy()
    X = X[[c for c in X.columns if X[c].isnull().sum() / len(X) <= 0.5]]

    out_cols = []
    for col in X.columns:
        meta = dd_cov.loc[col]
        variable_type, r_class = meta["variable_type"], meta["r_class"]

        if r_class == "ordered/factor":
            # Rank-encode using the level order given in levels_or_range (the R
            # ordered-factor's own level sequence), then median-impute.
            s = _normalize_missing(X[col])
            levels = [lvl.strip() for lvl in str(meta["levels_or_range"]).split(";")]
            rank = {lvl: float(i) for i, lvl in enumerate(levels)}
            out = s.map(rank).astype(float)
            out = out.fillna(out.median()) if out.notna().any() else out
            out_cols.append(out.rename(col))
        elif variable_type == "binary":
            # Map to {0.0, 1.0} regardless of source representation (numeric 2-valued
            # -> min/max; text -> sorted uniques), then mode-impute.
            s = _normalize_missing(X[col])
            uniques = sorted(s.dropna().unique(), key=str)
            mapping = {v: float(i) for i, v in enumerate(uniques[:2])}
            out_cols.append(_mode_fill(s.map(mapping).astype(float)).rename(col))
        elif variable_type == "factor":
            # One-hot encode a nominal factor, dropping (with a printed reason) if
            # high-cardinality -- almost certainly free text, not a real categorical covariate.
            n_unique = int(meta["n_unique_nonmissing"]) if pd.notna(meta["n_unique_nonmissing"]) else X[col].nunique()
            if n_unique > max_factor_levels:
                print(f"    dropped {col}: high-cardinality factor ({n_unique} levels)")
                continue
            s = _mode_fill(_normalize_missing(X[col]))
            out_cols.append(pd.get_dummies(s.astype("string"), prefix=col, drop_first=True, dummy_na=False))
        elif variable_type in ("continuous", "time-to-event/continuous time"):
            out_cols.append(_encode_continuous(X[col]).rename(col))
        else:
            print(f"    warning: {col} has unrecognized variable_type '{variable_type}', "
                  f"falling back to dtype-based encoding")
            if pd.api.types.is_numeric_dtype(X[col]):
                out_cols.append(_encode_continuous(X[col]).rename(col))
            else:
                out_cols.append(_mode_fill(_normalize_missing(X[col])).rename(col))

    return pd.concat(out_cols, axis=1) if out_cols else pd.DataFrame(index=X.index)