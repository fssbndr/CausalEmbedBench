#!/usr/bin/env python
"""Oracle validation: `.venv/bin/python B_oracles/shared/validate.py <trial>`.

<trial> in {vasst, twins, ihdp, news, jobs, rctbench}. Validates the baseline preset at
data/oracles/<trial>/, writes VALIDATION_REPORT.txt + validation_checks.csv there, exits
non-zero iff a gating check FAILs.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)  # scipy.stats.anderson method= nag

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knobs import overlap_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLES = PROJECT_ROOT / "data/oracles"
VASST_DIR = Path(__file__).resolve().parents[1] / "vasst"
KNOWN = ("vasst", "twins", "ihdp", "news", "jobs", "rctbench")
KEY_NUMERIC_COLS = ("propensity_score", "Y0", "Y1", "true_tau", "log_or")
TAU_BOUNDS = {"binary": (-1.0, 1.0)}  # .get(type) -> None for continuous


# --- check primitives ------------------------------------------------------------------
def _ate(df):
    return float(df["true_tau"].mean())


def _perturbed(params):
    k = params.get("knobs", {})
    return k.get("gamma_hetero", 1.0) != 1.0 or k.get("xi_effect", 1.0) != 1.0


def _check(label, ok, critical=True, **fields):
    status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
    return {"check": label, "status": status, "critical": critical, **fields}


def close(label, got, want, tol, **fields):
    return _check(label, abs(got - want) <= tol, got=got, want=want, tol=tol, **fields)


def within(label, val, lo, hi, **fields):
    return _check(label, lo <= val <= hi, val=val, range=[lo, hi], **fields)


# --- generic checks ------------------------------------------------------------------
def ranges_and_safety(df, outcome_type):
    in_range = {"binary": lambda s: s.is_in([0.0, 1.0]).all()}.get(
        outcome_type, lambda s: s.is_finite().all())
    ok = {"no_nan_inf": not any(df[c].is_nan().any() or df[c].is_infinite().any()
                                for c in KEY_NUMERIC_COLS if c in df.columns)}
    ok |= {f"{y}_range": bool(in_range(df[y])) for y in ("Y0", "Y1")}
    bounds = TAU_BOUNDS.get(outcome_type)
    if bounds:
        ok["tau_range"] = bool(bounds[0] <= df["true_tau"].min() and df["true_tau"].max() <= bounds[1])
    if "propensity_score" in df.columns:
        ps = df["propensity_score"]
        ok["propensity_range"] = bool(ps.min() > 0.0 and ps.max() < 1.0)
    return _check("Ranges & safety", all(ok.values()), **{k: bool(v) for k, v in ok.items()})


def residual_normality(df):
    """Report-only: Anderson-Darling of the realized residual vs Normal. Only IHDP/NEWS keep
    noise-free Y0/Y1 plus a separately-noised factual Y, so only they can run it."""
    t = df["T_synthetic"].to_numpy()
    resid = df["Y_obs_synthetic"].to_numpy() - np.where(
        t == 1, df["Y1"].to_numpy(), df["Y0"].to_numpy())
    a2 = scipy.stats.anderson(resid / max(resid.std(), 1e-12), "norm")
    return _check("Residual normality", "INFO", critical=False,
                  anderson_A2=float(a2.statistic), A2_crit_5pct=float(a2.critical_values[2]),
                  skew=float(scipy.stats.skew(resid)), kurtosis=float(scipy.stats.kurtosis(resid)))


# --- VASST published-trial grounding ------------------------------------------------
def vasst_subgroup(df, cohort, params):
    """Non-gating: low lactate -> more benefit (VASST Table 3)."""
    gated = _perturbed(params) or params.get("knobs", {}).get("gamma_hetero", 1.0) == 0.0
    seed0 = df.filter(pl.col("seed") == 0).to_pandas()
    lac = cohort.filter(pl.col("Treatment Arm").is_in(["Early", "Never"]))["Max Lactate (0-6h)"].to_numpy()
    aligned = len(lac) == len(seed0)
    q1 = seed0.loc[lac <= np.nanquantile(lac, 0.25), "true_tau"].mean() if aligned else np.nan
    q4 = seed0.loc[lac >= np.nanquantile(lac, 0.75), "true_tau"].mean() if aligned else np.nan
    status = "SKIP" if gated or not aligned else ("PASS" if q1 < q4 else "FAIL")
    return _check("Lactate subgroup direction", status, critical=False,
                  q1_tau=float(q1), q4_tau=float(q4))


def vasst_checks(df, params):
    ref = yaml.safe_load((VASST_DIR / "reference_values.yaml").read_text())["primary_effect"]
    cohort = pl.read_parquet(PROJECT_ROOT / "data/cohorts/vasst/STUDY_COHORT.parquet")
    ate, lo, hi = _ate(df), ref["ard_ci_lower"], ref["ard_ci_upper"]

    treated_frac = (cohort.select("Treatment Arm") == "Early").sum().item() / len(cohort)
    beta = params.get("knobs", {}).get("kappa_overlap", 1.0)
    want_ps = float(overlap_transform(np.array([treated_frac]), beta)[0])
    ps = df["propensity_score"]

    return [
        _check("Primary ATE in ARD CI", "SKIP" if _perturbed(params) else lo <= ate <= hi,
               ate=ate, ci=[lo, hi]),
        close("Baseline risk P(Y0)", float(df["Y0"].mean()),
              ref["mortality_norepinephrine_pct"] / 100, 0.03),
        close("Propensity calibration", float(ps.mean()), want_ps, 0.02,
              ps_min=float(ps.min()), ps_max=float(ps.max())),
        vasst_subgroup(df, cohort, params),
    ]


# --- per-trial check lists ------------------------------------------------------------
EXTRA = {
    "vasst": vasst_checks,
    "twins": lambda df, p: [within("ATE ~ 0 (CEVAE no-effect)", _ate(df), -0.05, 0.01)],
    "ihdp": lambda df, p: [residual_normality(df),
                           within("ATE in published IHDP range", _ate(df), 2.0, 7.0)],
    "news": lambda df, p: [residual_normality(df)],
}


def build_checks(trial, df, params):
    target = float(params.get("true_ATE", params.get("realized_ATE")))  # VASST writes realized_ATE
    return [
        close("ATE self-consistency", _ate(df), target, max(0.01, 0.02 * abs(target))),
        ranges_and_safety(df, params["outcome_type"]),
        *EXTRA.get(trial, lambda *_: [])(df, params),
    ]


# --- run + report ------------------------------------------------------------------
def render(title, sections):
    out = ["=" * 80, f"ORACLE VALIDATION: {title}", "=" * 80, ""]
    any_fail = False
    for name, checks in sections:
        if len(sections) > 1:
            out.append(f"--- {name} ---")
        for c in checks:
            any_fail |= c["status"] == "FAIL" and c["critical"]
            fields = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in c.items() if k not in ("check", "status", "critical"))
            tag = "" if c["critical"] else " [non-gating]"
            out.append(f"  {c['check']}: {c['status']}{tag}" + (f" ({fields})" if fields else ""))
        out.append("")
    out += ["=" * 80, f"OVERALL: {'FAIL' if any_fail else 'PASS'}", "=" * 80]
    return "\n".join(out), any_fail


def main(trial):
    if trial not in KNOWN:
        sys.exit(f"unknown trial '{trial}'; expected one of {list(KNOWN)}")
    multi = trial == "rctbench"
    dirs = ([(t, ORACLES / t) for t in (ORACLES / "rctbench_trials.txt").read_text().split()]
            if multi else [(trial, ORACLES / trial)])
    out_dir = ORACLES if multi else ORACLES / trial
    prefix = "rctbench_" if multi else ""
    title = f"RCTBENCH ({len(dirs)} trials)" if multi else trial.upper()

    sections, rows = [], []
    for name, d in dirs:
        df = pl.read_parquet(d / "semi_synthetic.parquet")
        params = json.loads((d / "causal_params.json").read_text())
        checks = build_checks(trial, df, params)
        sections.append((name, checks))
        rows += [{"trial": name, **c} for c in checks]

    report, any_fail = render(title, sections)
    (out_dir / f"{prefix}VALIDATION_REPORT.txt").write_text(report)
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}validation_checks.csv", index=False)
    print(report)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: validate.py <trial>")
    main(sys.argv[1])
