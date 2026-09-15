# Vasopressor utilities for reprodICU studies
# ------------------------------------------------------------------------------

import polars as pl
import reprodICU
from reprodICU.utils.clinical.pharmocological.ALIGNED_UNITS import ALIGNED_UNITS
from reprodICU.utils.FIX_WINDOW_BORDERS import FIX_WINDOW_BORDERS

SECONDS_IN_1H = 60 * 60

# region lazyload
# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
meds: pl.LazyFrame = reprodICU.medications


# region constants
# ------------------------------------------------------------------------------
NOREPINEPHRINE_MIN_RATE, NOREPINEPHRINE_MAX_RATE = 0.001, 5     # µg/kg/min
VASOPRESSIN_MIN_RATE,    VASOPRESSIN_MAX_RATE    = 0.01,  1     # U/min
VASOPRESSIN_VANISH_MAX_RATE                      = 0.06         # U/min  (Gordon et al. VANISH 2016)
NE_SHOCK_THRESHOLD                               = 0.1          # µg/kg/min  (shock onset threshold)

VASOPRESSORS_LIST = ["norepinephrine", "vasopressin (USP)"]


# region vasopressors
# ------------------------------------------------------------------------------
# Converting the units to the units used for equivalence calculations
# - norepinephrine: mcg/kg/min
# - vasopressin: U/min
def get_vasopressors_data(
    # DATA: pl.LazyFrame,
    T_0: pl.LazyFrame | None = None,
    TIMEWINDOW_IN_SECONDS: int = SECONDS_IN_1H,
):
    # If T_0 not provided, anchor to ICU admission (T_0 = 0)
    if T_0 is None:
        T_0 = info.select("Global ICU Stay ID").with_columns(pl.lit(0).alias("T_0"))

    VASOPRESSORS = meds.filter(
        pl.col("Drug Ingredient").is_in(VASOPRESSORS_LIST),
    ).filter(pl.col("Drug Rate").gt(0))

    DATA = (
        ALIGNED_UNITS(medications=VASOPRESSORS, patient_information=info)
        .with_columns(
            pl.when(
                (
                    pl.col("Drug Ingredient").eq("norepinephrine")
                    & pl.col("Drug Rate (fixed units)").le(NOREPINEPHRINE_MIN_RATE)
                )
                | (
                    pl.col("Drug Ingredient").eq("vasopressin (USP)")
                    & pl.col("Drug Rate (fixed units)").le(VASOPRESSIN_MIN_RATE)
                )
            )
            .then(0)
            .otherwise(pl.col("Drug Rate (fixed units)"))
            .alias("Drug Rate (fixed units)"),
            (
                pl.col("Drug End Relative to Admission (seconds)")
                - pl.col("Drug Start Relative to Admission (seconds)")
            ).alias("Drug Duration (seconds)"),
        )
        # Drop all bolus-like administrations
        .filter(pl.col("Drug Duration (seconds)") > 60)
        .join(T_0, on="Global ICU Stay ID", how="left")
        .with_columns(
            pl.col("Drug Start Relative to Admission (seconds)")
            .sub(pl.col("T_0"))
            .alias("Drug Start Relative to T_0 (seconds)"),
            pl.col("Drug End Relative to Admission (seconds)")
            .sub(pl.col("T_0"))
            .alias("Drug End Relative to T_0 (seconds)"),
        )
    ) # fmt: skip

    return (
        FIX_WINDOW_BORDERS(DATA, TIMEWINDOW_IN_SECONDS)
        .with_columns(
            pl.col("Drug Duration (windows)").alias("Drug Duration (hours)"),
            pl.col("Window Relative to T_0").alias("Hours Relative to T_0"),
        )
        .filter(pl.col("Drug Duration (hours)") > 0)
        .group_by(
            "Global ICU Stay ID",
            "Hours Relative to T_0",
            "Drug Ingredient",
        )
        .agg(
            (
                (
                    pl.col("Drug Rate (fixed units)")
                    * pl.col("Drug Duration (hours)")
                ).sum()
                / pl.col("Drug Duration (hours)").sum()
            ).alias("weighted avg rate")
        )
        .collect()
        .pivot(
            on="Drug Ingredient",
            index=["Global ICU Stay ID", "Hours Relative to T_0"],
            values="weighted avg rate",
        )
        .sort("Global ICU Stay ID", "Hours Relative to T_0")
        .lazy()
        .select(
            "Global ICU Stay ID",
            "Hours Relative to T_0",
            pl.col("norepinephrine").alias("Norepinephrine (avg mcg/kg/min during hour)"),
            pl.col("vasopressin (USP)").alias("Vasopressin (avg units/min during hour)"),
        )
    ) # fmt: skip


def get_norepinephrine_data(
    T_0: pl.LazyFrame | None = None,
    TIMEWINDOW_IN_SECONDS: int = SECONDS_IN_1H,
) -> pl.LazyFrame:
    return get_vasopressors_data(T_0, TIMEWINDOW_IN_SECONDS).select(
        "Global ICU Stay ID",
        "Hours Relative to T_0",
        pl.col("Norepinephrine (avg mcg/kg/min during hour)"),
    )


def get_vasopressin_data(
    T_0: pl.LazyFrame | None = None,
    TIMEWINDOW_IN_SECONDS: int = SECONDS_IN_1H,
) -> pl.LazyFrame:
    return get_vasopressors_data(T_0, TIMEWINDOW_IN_SECONDS).select(
        "Global ICU Stay ID",
        "Hours Relative to T_0",
        pl.col("Vasopressin (avg units/min during hour)"),
    )


# region shock onset
# ------------------------------------------------------------------------------
def get_shock_onset_t0(sepsis_onset: pl.LazyFrame) -> pl.LazyFrame:
    """T_0 per ICU stay (seconds from ICU admission): first hour where Sepsis-3
    criteria are met while the patient is actively receiving NE > NE_SHOCK_THRESHOLD.

    Operationalized per OVISS (Kalimouttou et al., JAMA 2025): shock onset is the
    first epoch where vasopressor need and Sepsis-3 criteria are simultaneously met.

    sepsis_onset: LazyFrame with columns (Global ICU Stay ID,
                  Hours Relative to ICU Admission), one row per patient-hour where
                  Sepsis-3 shock criteria are satisfied.
    """
    # Anchor all patients at ICU admission (T_0 = 0) to express NE coverage
    # in hours-from-admission — consistent with the SEPSIS.parquet time axis.
    icu_anchor = info.select("Global ICU Stay ID").with_columns(pl.lit(0).alias("T_0"))

    return (
        get_norepinephrine_data(T_0=icu_anchor)
        .filter(
            pl.col("Norepinephrine (avg mcg/kg/min during hour)").ge(NE_SHOCK_THRESHOLD)
        )
        .rename({"Hours Relative to T_0": "Hours Relative to ICU Admission"})
        .select("Global ICU Stay ID", "Hours Relative to ICU Admission")
        # Keep only hours where Sepsis-3 shock criteria are also met
        .join(
            sepsis_onset.select("Global ICU Stay ID", "Hours Relative to ICU Admission"),
            on=["Global ICU Stay ID", "Hours Relative to ICU Admission"],
            how="inner",
        )
        .group_by("Global ICU Stay ID")
        .agg(
            (pl.col("Hours Relative to ICU Admission").min() * SECONDS_IN_1H)
            .cast(pl.Int64)
            .alias("T_0")
        )
    )
