# Russell JA, Walley KR, Singer J, Gordon AC, Hébert PC, Cooper DJ, Holmes CL, Mehta S, Granton JT, Storms MM, Cook DJ, Presneill JJ, Ayers D; VASST Investigators.
# Vasopressin versus norepinephrine infusion in patients with septic shock.
# N Engl J Med. 2008 Feb 28;358(9):877-87.
# doi: 10.1056/NEJMoa067373. PMID: 18305265.
# ------------------------------------------------------------------------------
# ruff: noqa: E402

import os
import sys
from pathlib import Path

# Add project root to path so utils can be imported
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import reprodICU
from pycomorb import comorbidity
from tableone import TableOne

from A_cohorts.utils import (
    NOREPINEPHRINE_MAX_RATE,
    VASOPRESSIN_MIN_RATE,
    VASOPRESSIN_MAX_RATE,
    VASOPRESSIN_VANISH_MAX_RATE,
    get_norepinephrine_data,
    get_shock_onset_t0,
    get_vasopressin_data,
)

from reprodICU.utils.clinical.pharmocological.ALIGNED_UNITS import ALIGNED_UNITS
from reprodICU.utils.scores import SOFA, APACHE2
from reprodICU.utils.sepsis import SEPSIS

FOLDER = "../data/cohorts/vasst/"
os.makedirs(FOLDER, exist_ok=True)

# fmt: off
# region lazyload
# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info:   pl.LazyFrame = reprodICU.patient_information
meds:   pl.LazyFrame = reprodICU.medications
vitals: pl.LazyFrame = reprodICU.timeseries_vitals
labs:   pl.LazyFrame = reprodICU.timeseries_labs
resp:   pl.LazyFrame = reprodICU.timeseries_respiratory
inout:  pl.LazyFrame = reprodICU.timeseries_intakeoutput
diags:  pl.LazyFrame = reprodICU.diagnoses

vent: pl.LazyFrame = reprodICU.VENTILATION_DURATION

SECONDS_IN_1H  = 60 * 60
SECONDS_IN_6H  = 6  * SECONDS_IN_1H
SECONDS_IN_24H = 24 * SECONDS_IN_1H

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
# fmt: on

# region 0. study design
################################################################################
# 0. Study design
print("0. Setting up the study design")
# ------------------------------------------------------------------------------
# 0.1 Defining time zero (T_0): first hour where Sepsis-3 criteria and NE
#     infusion > NE_SHOCK_THRESHOLD coincide — following OVISS operationalization
#     (Kalimouttou et al., JAMA 2025).
#     T_0 is expressed in seconds from ICU admission.
# ------------------------------------------------------------------------------

# Sepsis-3 shock onset data must be computed before T_0 (it is an input to T_0).
if not os.path.exists(FOLDER + "SEPSIS.parquet"):
    SEPSIS().sink_parquet(FOLDER + "SEPSIS.parquet")
    print("   -> calculated Sepsis-3 shock onset data")

# Hours (relative to ICU admission) where Sepsis-3 shock criteria are met.
# SEPSIS() implements: suspected infection + continuous vasopressor need +
# lactate >= 2.0 mmol/L (Singer et al. 2016).
SEPSIS_ONSET_HOURS = (
    pl.scan_parquet(FOLDER + "SEPSIS.parquet")
    .rename({"timeframe": "Hours Relative to ICU Admission"})
    .filter(pl.col("SEPSIS").is_not_null())
    .select(STAY_KEY, "Hours Relative to ICU Admission")
)

# region t_0
# ------------------------------------------------------------------------------
T_0 = get_shock_onset_t0(sepsis_onset=SEPSIS_ONSET_HOURS)

# region t_landmark
# ------------------------------------------------------------------------------
# 0.2 Defining the landmark time (T_landmark = T_0 + 6h)
#     All eligible patients must survive and remain in ICU up to T_landmark.
#     Covariates and exposure classification are anchored at T_landmark.
# ------------------------------------------------------------------------------
T_LANDMARK = T_0.with_columns((pl.col("T_0") + SECONDS_IN_6H).alias("T_LANDMARK"))

# region t_1
# ------------------------------------------------------------------------------
# 0.3 Defining end of follow-up per stay (T_1 = ICU discharge, seconds from admission)
# ------------------------------------------------------------------------------
T_1 = T_LANDMARK.join(
    info.select(
        STAY_KEY,
        pl.col("ICU Length of Stay (days)").mul(SECONDS_IN_24H).alias("T_1"),
    ),
    on=STAY_KEY,
    how="left",
)

# region t_outcome
# ------------------------------------------------------------------------------
# 0.4 Defining outcome window: in-hospital mortality and 30-day mortality from T_0
# ------------------------------------------------------------------------------

# region CIV
# ------------------------------------------------------------------------------
# 0.5 Defining clinically implausible values
# adapted from "Data Cleaning for Clinically Impossible Values – ICU"
# https://docs.google.com/spreadsheets/d/1GDEz7k5AWbXNez652PPVq-bxys0OjTDbOPVQvdSfLAw/edit by DvW

# region 1. initial cohort
################################################################################
# 1. Defining the primary study cohort
print("1. Defining the initial study cohort")
# ------------------------------------------------------------------------------
# Here, all variables used to select patients of interest should be created
# (considering the time of eligibility assessment).
# However, dropping patients not of interest should be done below (## 5. Figure 1)
# ------------------------------------------------------------------------------
# Inclusion criteria:
# - age >= 18 years
# - first ICU admission of the hospital stay
#   [Sepsis-3 + NE > threshold are encoded in T_0 itself — see section 0]
#
# Exclusion criteria:
# - vasopressin or terlipressin initiated before norepinephrine onset
# - transferred from another ICU
# - death or ICU discharge before T_landmark (< 6h after shock onset)
# - implausible vasopressor rates

# region 1.0 inclusion_exclusion
# ------------------------------------------------------------------------------
# 1.0 Defining data source(s) for inclusion/exclusion criteria
#     Base cohort: all patients for whom a T_0 (shock onset) can be identified.
#     By construction, T_0 requires both NE > NE_SHOCK_THRESHOLD and Sepsis-3
#     criteria to be simultaneously met, so the sepsis check is implicit.
# ------------------------------------------------------------------------------
CASE_IDS = T_0.join(
    info.select(STAY_KEY, "Source Dataset").unique(),
    on=STAY_KEY,
    how="left",
)
AGES = info.select(STAY_KEY, "Admission Age (years)")
ORIGINS = info.select(STAY_KEY, "Admission Origin")
ICU_STAY_NUM = info.select(
    "Global Person ID",
    STAY_KEY,
    "ICU Stay Sequential Number (per Person ID)",
)

# region 1.1 eligibility
# ------------------------------------------------------------------------------
# 1.1 Defining time of eligibility
# ------------------------------------------------------------------------------
INCLUSION_EXCLUSION = CASE_IDS

# 1.2 Cleaning raw data for clinically implausible values
# 1.3 Quality check

# region 1.4 aggregation
# ------------------------------------------------------------------------------
# 1.4 Aggregation of variables
# ------------------------------------------------------------------------------

# region 1.4.I inclusion
# ------------------------------------------------------------------------------
# 1.4.I.1 age >= 18
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(AGES, on=STAY_KEY, how="left")
    .with_columns((pl.col("Admission Age (years)") >= 18).alias("Inclusion: age > 18"))
    .drop("Admission Age (years)")
)

# 1.4.I.2 first ICU admission of the hospital stay
# Uses Global Hospital Stay ID ordering; patients with multiple ICU stays per
# hospital stay are restricted to the first stay (by Global ICU Stay ID order,
# which reflects chronological order within reprodICU).
FIRST_ICU_STAY = CASE_IDS.join(ICU_STAY_NUM, on=STAY_KEY, how="left").select(
    STAY_KEY,
    pl.col("ICU Stay Sequential Number (per Person ID)")
    .eq(1)
    .fill_null(True)
    .alias("Inclusion: first ICU admission"),
)
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    FIRST_ICU_STAY, on=STAY_KEY, how="left"
).with_columns(pl.col("Inclusion: first ICU admission").fill_null(False))

# region 1.4.E exclusion
# ------------------------------------------------------------------------------
# 1.4.E.1 vasopressin or terlipressin initiated before norepinephrine onset
# Patients for whom vasopressin was the primary vasopressor are excluded;
# the clinical question concerns *adjunctive* vasopressin to norepinephrine.
MEDS_VASOPRESSIN = meds.filter(
    pl.col("Drug Ingredient").is_in(["vasopressin (USP)", "terlipressin"])
)
VASOPRESSIN_BEFORE_NE = (
    ALIGNED_UNITS(
        medications=MEDS_VASOPRESSIN,
        patient_information=info,
    )
    .join(T_0, on=STAY_KEY, how="inner")
    .filter(
        pl.col("Drug Rate (fixed units)").ge(VASOPRESSIN_MIN_RATE),
        pl.col("Drug Start Relative to Admission (seconds)").lt(pl.col("T_0")),
        # Exclude bolus administrations defined as drug duration <= 60 seconds
        (
            pl.col("Drug End Relative to Admission (seconds)")
            - pl.col("Drug Start Relative to Admission (seconds)")
        ).gt(60),
    )
    .group_by(STAY_KEY)
    .agg(pl.len().gt(0).alias("vasopressin_before_ne"))
)
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(VASOPRESSIN_BEFORE_NE, on=STAY_KEY, how="left")
    .with_columns(
        pl.col("vasopressin_before_ne")
        .not_()
        .fill_null(True)
        .alias("Exclusion: vasopressin before norepinephrine onset")
    )
    .drop("vasopressin_before_ne")
)

# 1.4.E.2 transferred from another ICU
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(ORIGINS, on=STAY_KEY, how="left")
    .with_columns(
        pl.col("Admission Origin")
        .ne_missing("Other ICU")
        .alias("Exclusion: transferred from another ICU")
    )
    .drop("Admission Origin")
)

# 1.4.E.3 death or ICU discharge before T_landmark (< 6h after shock onset)
# These patients cannot be classified into exposure arms at the landmark and
# are excluded to prevent immortal time bias.
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(
        T_1.select(STAY_KEY, "T_LANDMARK", "T_1"),
        on=STAY_KEY,
        how="left",
    )
    .with_columns(
        pl.col("T_1")
        .ge(pl.col("T_LANDMARK"))
        .fill_null(False)
        .alias("Exclusion: survived to T_landmark")
    )
    .drop("T_LANDMARK", "T_1")
)

# 1.4.E.4 implausible vasopressor rates
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(
        get_norepinephrine_data(T_0=T_0)
        .group_by(STAY_KEY)
        .agg(
            pl.col("Norepinephrine (avg mcg/kg/min during hour)")
            .ge(NOREPINEPHRINE_MAX_RATE)
            .any()
            .alias("implausible norepinephrine")
        )
        .join(
            get_vasopressin_data(T_0=T_0)
            .group_by(STAY_KEY)
            .agg(
                pl.col("Vasopressin (avg units/min during hour)")
                .ge(VASOPRESSIN_MAX_RATE)
                .any()
                .alias("implausible vasopressin")
            ),
            on=STAY_KEY,
            how="full",
            coalesce=True,
        ),
        on=STAY_KEY,
        how="left",
    )
    .with_columns(
        (pl.col("implausible norepinephrine") | pl.col("implausible vasopressin"))
        .not_()
        .fill_null(True)
        .alias("Exclusion: implausible vasopressor rates")
    )
    .drop("implausible norepinephrine", "implausible vasopressin")
)

# region 1.5 wide format
# ------------------------------------------------------------------------------
# 1.5 Push aggregated variable to wide format DF for analyses
# ------------------------------------------------------------------------------
INCLUSION_CRITERIA = [
    "Inclusion: age > 18",
    "Inclusion: first ICU admission",
    "Exclusion: vasopressin before norepinephrine onset",
    "Exclusion: transferred from another ICU",
    "Exclusion: survived to T_landmark",
    "Exclusion: implausible vasopressor rates",
]

INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.unique().with_columns(
    pl.all_horizontal(*INCLUSION_CRITERIA).alias("is_included")
)

INCLUDED = INCLUSION_EXCLUSION.filter(pl.col("is_included")).select(STAY_KEY).unique()  # fmt: skip
SOURCE_DATABASE = info.select(STAY_KEY, "Source Dataset")

print(
    INCLUSION_EXCLUSION.join(SOURCE_DATABASE, on=STAY_KEY, how="left")
    .select("Source Dataset", "is_included")
    .group_by("Source Dataset")
    .sum()
    .sort("Source Dataset")
    .collect()
)

# | Source Dataset | is_included |
# | -------------- | ----------- |
# | AmsterdamUMCdb | 0           |
# | eICU-CRD       | 0           |
# | HiRID          | 0           |
# | MIMIC-III      | 0           |
# | MIMIC-IV       | 0           |
# | NWICU          | 0           |
# | SICdb          | 0           |

INCLUSION_EXCLUSION.join(
    info.select(STAY_KEY, "Global Hospital Stay ID", "Global Person ID"),
    on=STAY_KEY,
    how="left",
).sink_csv(FOLDER + "INCLUSION_COHORT.csv")

# region 1.X flowchart
# ------------------------------------------------------------------------------
# 1.X Reproducing the flow of the flowchart.
# ------------------------------------------------------------------------------
for source_filter in [None, "MIMIC-III", "eICU-CRD"]:
    label = f" from {source_filter}" if source_filter else ""
    df = (
        INCLUSION_EXCLUSION.join(SOURCE_DATABASE, on=STAY_KEY, how="left")
        .pipe(
            lambda lf: (
                lf.filter(pl.col("Source Dataset") == source_filter)
                if source_filter
                else lf
            )
        )
        .collect()
        .to_pandas()
    )
    for criterion in INCLUSION_CRITERIA:
        idxRem = ~df[criterion].fillna(False).astype(bool)
        print(
            "{:6d} - removing {:6d} ({:5.2f}%) patients{} - {}.".format(
                df.shape[0], np.sum(idxRem), 100.0 * np.mean(idxRem), label, criterion
            )
        )
        df = df.loc[~idxRem, :]
    print("{:6d} - final cohort{}.".format(df.shape[0], label), end="\n\n")

# region 2. primary exposure
################################################################################
# 2. Defining the primary exposure
print("2. Defining the primary exposure")
# ------------------------------------------------------------------------------
# Landmark design: exposure is classified statically at T_landmark (T_0 + 6h).
# Treatment arms (mutually exclusive):
#   Early (A = 1): vasopressin initiated 0–6h after T_0, peak dose <= 0.06 U/min
#   Delayed (A = 0): no vasopressin 0–6h, but initiated 6–24h after T_0
#   Never: no vasopressin within 24h after T_0 (primary comparator per White 2025;
#          included as delayed-equivalent arm in sensitivity analysis)

# region 2.1 PERIOD OF EXPOSURE
# ------------------------------------------------------------------------------
# 2.1 Defining the exposure window relative to T_0
# ------------------------------------------------------------------------------
# Vasopressin data is computed relative to T_0 (shock onset) via get_vasopressin_data.
# Hours 0–6 correspond to the 0->T_landmark window.
# Hours 6–24 correspond to the T_landmark->T_0+24h window.

# 2.2 Cleaning raw data for clinically implausible values
# 2.3 Quality check

# region 2.4 aggregation
# ------------------------------------------------------------------------------
# 2.4 Aggregation of final primary exposure variable
# ------------------------------------------------------------------------------
VASOPRESSIN_ALL = get_vasopressin_data(T_0=T_0)

# Early vasopressin: any infusion >= VASOPRESSIN_MIN_RATE in hours 0–6 after T_0,
# with peak dose <= VASOPRESSIN_VANISH_MAX_RATE (matching VANISH dosing protocol)
VASOPRESSIN_EARLY = (
    VASOPRESSIN_ALL.filter(
        pl.col("Hours Relative to T_0").is_between(0, 6, closed="both"),
        pl.col("Vasopressin (avg units/min during hour)").ge(VASOPRESSIN_MIN_RATE),
        pl.col("Vasopressin (avg units/min during hour)").le(
            VASOPRESSIN_VANISH_MAX_RATE
        ),
    )
    .group_by(STAY_KEY)
    .agg(
        pl.col("Vasopressin (avg units/min during hour)")
        .max()
        .alias("Peak Vasopressin 0-6h (U/min)"),
        pl.col("Hours Relative to T_0").min().alias("Vasopressin Onset Hour"),
    )
)

# Delayed vasopressin: no vasopressin 0–6h, but initiated 6–24h after T_0
VASOPRESSIN_DELAYED = (
    VASOPRESSIN_ALL.filter(
        pl.col("Hours Relative to T_0").is_between(6, 24, closed="both"),
        pl.col("Vasopressin (avg units/min during hour)").ge(VASOPRESSIN_MIN_RATE),
    )
    .group_by(STAY_KEY)
    .agg(
        pl.col("Vasopressin (avg units/min during hour)")
        .max()
        .alias("Peak Vasopressin 6-24h (U/min)"),
    )
)

# region 2.5 wide format
# ------------------------------------------------------------------------------
# 2.5 Push aggregated exposure variable to wide format DF for analyses
#     Result: one row per patient with static treatment arm assignment
# ------------------------------------------------------------------------------
EXPOSURE = (
    INCLUDED.join(VASOPRESSIN_EARLY, on=STAY_KEY, how="left")
    .join(VASOPRESSIN_DELAYED, on=STAY_KEY, how="left")
    .with_columns(
        pl.col("Peak Vasopressin 0-6h (U/min)")
        .is_not_null()
        .alias("Early Vasopressin"),
        (
            pl.col("Peak Vasopressin 6-24h (U/min)").is_not_null()
            & pl.col("Peak Vasopressin 0-6h (U/min)").is_null()
        ).alias("Delayed Vasopressin"),
    )
    .with_columns(
        pl.when(pl.col("Early Vasopressin"))
        .then(pl.lit("Early"))
        .when(pl.col("Delayed Vasopressin"))
        .then(pl.lit("Delayed"))
        .otherwise(pl.lit("Never"))
        .alias("Treatment Arm"),
    )
)

# region 3. primary outcome
################################################################################
# 3. Defining the primary outcome
print("3. Defining the primary outcome")
# ------------------------------------------------------------------------------
# 3.0 Defining data source(s) for outcome definitions

# 3.1 Time of outcome evaluation
# 3.2 Cleaning raw data for clinically implausible values
# 3.3 Quality check

# region 3.4 aggregation
# ------------------------------------------------------------------------------
# 3.4 Aggregation of final outcome variables
# ------------------------------------------------------------------------------
# Primary outcome: in-hospital mortality.
# Also computed: 30-day mortality anchored to T_0 (shock onset), derived from
# hospital mortality flag combined with hospital LOS and T_0 offset.
OUTCOME = (
    reprodICU.utils.mortality.COMMON_MORTALITY_MEASURES()
    .join(
        info.select(
            STAY_KEY,
            pl.col("Hospital Length of Stay (days)").alias("Hospital LOS (days)"),
        ),
        on=STAY_KEY,
        how="left",
    )
    .join(
        T_0.select(STAY_KEY, "T_0"),
        on=STAY_KEY,
        how="inner",
    )
    .with_columns(
        (
            pl.col("Mortality in Hospital")
            & (
                pl.col("Hospital LOS (days)") - pl.col("T_0").truediv(SECONDS_IN_24H)
            ).lt(30)
        ).alias("Mortality 30d After Shock Onset"),
    )
    .select(
        STAY_KEY,
        "Mortality in ICU",
        "Mortality in Hospital",
        "Mortality 30d After Shock Onset",
    )
)

# 3.5 Push aggregated outcome variable to wide format DF for analyses

# region 4. descriptives and confounders
################################################################################
# 4. Defining key descriptives and confounding variables for the primary analysis
print("4. Defining key descriptives and confounding variables for the primary analysis")  # fmt: skip
# ------------------------------------------------------------------------------
# 4.1 Data source(s)
# All covariates are measured strictly in the 0–6h window [T_0, T_LANDMARK]
# to maintain temporal ordering relative to treatment classification.
PATIENT_FEATURES = info.select(
    STAY_KEY,
    "Admission Age (years)",
    "Admission Height (cm)",
    "Admission Weight (kg)",
    "Gender",
    "Ethnicity",
    "Admission Origin",
    "Source Dataset",
)

VITAL_SIGNS = vitals.select(
    STAY_KEY,
    TIME_KEY,
    pl.coalesce(
        pl.col("Invasive mean arterial pressure"),
        pl.col("Non-invasive mean arterial pressure"),
        1 / 3 * pl.col("Invasive systolic arterial pressure")
        + 2 / 3 * pl.col("Invasive diastolic arterial pressure"),
        1 / 3 * pl.col("Non-invasive systolic arterial pressure")
        + 2 / 3 * pl.col("Non-invasive diastolic arterial pressure"),
    ).alias("Mean arterial pressure"),
    pl.col("Heart rate"),
    pl.col("Temperature"),
)

LAB_VALUES = (
    labs.select(
        STAY_KEY,
        TIME_KEY,
        "Creatinine",
        "Lactate",
        "pH",
        "Bicarbonate",
    )
    .with_columns(
        pl.when(
            pl.col(col)
            .struct.field("system")
            .str.contains_any(["Blood", "Serum", "Plasma", "Arterial"])
            | pl.col(col).struct.field("system").is_null()
        )
        .then(pl.col(col).struct.field("value"))
        .otherwise(None)
        .alias(f"{col}")
        for col in ["Creatinine", "Lactate", "pH", "Bicarbonate"]
    )
    .filter(
        pl.any_horizontal(
            pl.col("Creatinine", "Lactate", "pH", "Bicarbonate").is_not_null()
        )
    )
)

# region 4.2 PERIOD OF CONFOUNDING
# ------------------------------------------------------------------------------
# 4.2 Period of interest: strictly [T_0, T_LANDMARK] = first 6h after shock onset
# ------------------------------------------------------------------------------

# 4.3 Cleaning raw data for clinically implausible values
# 4.4 Quality check

# region 4.5 aggregation
# ------------------------------------------------------------------------------
# 4.5 Aggregate covariates over the 0–6h window; one row per patient.
# Physiological extremes (min/max) are used to capture peak severity.
# ------------------------------------------------------------------------------

# Helper: join T_0/T_LANDMARK and filter to the [T_0, T_LANDMARK] window.
LANDMARK_TIMES = T_LANDMARK.select(STAY_KEY, "T_0", "T_LANDMARK")


def in_landmark_window(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.join(LANDMARK_TIMES, on=STAY_KEY, how="inner").filter(
        pl.col(TIME_KEY).is_between(pl.col("T_0"), pl.col("T_LANDMARK"), closed="both")
    )


COVARIATES_VITALS = (
    in_landmark_window(VITAL_SIGNS)
    .group_by(STAY_KEY)
    .agg(
        pl.col("Mean arterial pressure").min().alias("Min MAP (0-6h)"),
        pl.col("Heart rate").max().alias("Max Heart Rate (0-6h)"),
        pl.col("Temperature").mean().alias("Mean Temperature (0-6h)"),
    )
)

COVARIATES_LABS = (
    in_landmark_window(LAB_VALUES)
    .group_by(STAY_KEY)
    .agg(
        pl.col("Lactate").max().alias("Max Lactate (0-6h)"),
        pl.col("Creatinine").max().alias("Max Creatinine (0-6h)"),
        pl.col("pH").min().alias("Min pH (0-6h)"),
        pl.col("Bicarbonate").min().alias("Min Bicarbonate (0-6h)"),
    )
)

# Peak norepinephrine dose in 0–6h (vasoactive intensity before treatment decision)
COVARIATES_NE = (
    get_norepinephrine_data(T_0=T_0)
    .filter(pl.col("Hours Relative to T_0").is_between(0, 6, closed="both"))
    .group_by(STAY_KEY)
    .agg(
        pl.col("Norepinephrine (avg mcg/kg/min during hour)")
        .max()
        .alias("Peak Norepinephrine (0-6h)")
    )
)

# Mechanical ventilation status at T_landmark (hour 6)
COVARIATES_VENT = (
    INCLUDED.join(LANDMARK_TIMES, on=STAY_KEY, how="left")
    .join(
        vent.filter(pl.col("Ventilation Type") == "invasive ventilation"),
        on=STAY_KEY,
        how="left",
    )
    .with_columns(
        pl.col("T_LANDMARK")
        .is_between(
            pl.col("Ventilation Start Relative to Admission (seconds)"),
            pl.col("Ventilation End Relative to Admission (seconds)"),
        )
        .cast(int)
        .alias("ventilated_at_landmark")
    )
    .group_by(STAY_KEY)
    .agg(
        pl.col("ventilated_at_landmark")
        .max()
        .cast(bool)
        .alias("Ventilated at T_landmark")
    )
)

# Net fluid balance in 0–6h window
COVARIATES_FLUIDS = (
    in_landmark_window(
        inout.select(
            STAY_KEY,
            TIME_KEY,
            pl.sum_horizontal("^Fluid intake.*$").alias("Fluid intake (mL)"),
        )
    )
    .group_by(STAY_KEY)
    .agg(
        pl.col("Fluid intake (mL)")
        .sum()
        .clip(lower_bound=0)
        .alias("Fluid Balance 0-6h (mL)")
    )
)

# SOFA score: maximum in 0–6h window (peak severity before treatment decision)
# SOFA is computed relative to T_0 (shock onset)
# -> delete SOFA.parquet if T_0 definition changes to force recomputation
if not os.path.exists(FOLDER + "SOFA.parquet"):
    SOFA(
        patient_information=info,
        timeseries_vitals=vitals,
        timeseries_labs=labs,
        timeseries_resp=resp,
        timeseries_inout=inout,
        medications=meds,
        ventilation=vent,
        t_0_per_stay=T_0,
        window_size=SECONDS_IN_1H,
    ).sink_parquet(FOLDER + "SOFA.parquet")
    print("   -> calculated SOFA score data (relative to shock onset)")

COVARIATES_SOFA = (
    pl.scan_parquet(FOLDER + "SOFA.parquet")
    .filter(pl.col("Hours Relative to T_0").is_between(0, 6))
    .group_by(STAY_KEY)
    .agg(pl.col("SOFA Score").max().alias("Max SOFA (0-6h)"))
)

# APACHE II score: computed at T_0 (shock onset)
# -> delete APACHE2.parquet if T_0 definition changes to force recomputation
if not os.path.exists(FOLDER + "APACHE2.parquet"):
    APACHE2(
        patient_information=info,
        timeseries_vitals=vitals,
        timeseries_labs=labs,
        timeseries_resp=resp,
        diagnoses=diags,
        t_0_per_stay=T_0,
    ).sink_parquet(FOLDER + "APACHE2.parquet")
    print("   -> calculated APACHE II score data (relative to shock onset)")

COVARIATES_APACHE2 = pl.scan_parquet(FOLDER + "APACHE2.parquet").select(
    STAY_KEY, "APACHE II Score"
)

# Pre-existing comorbidities (ICD-10 codes)
DIAGNOSES = diags.join(
    info.select(STAY_KEY, "Admission Age (years)"),
    on=STAY_KEY,
    how="left",
    coalesce=True,
).with_columns(
    pl.when(pl.col("Diagnosis ICD Code Version (source)") == "ICD-9")
    .then(pl.col("Diagnosis ICD-9 Code"))
    .otherwise(pl.col("Diagnosis ICD-10 Code"))
    .alias("Diagnosis ICD Code")
)

ELIXHAUSER = (
    comorbidity(
        score="elixhauser",
        implementation="quan",
        df=DIAGNOSES.collect(),
        id_col=STAY_KEY,
        code_col="Diagnosis ICD Code",
        age_col="Admission Age (years)",
        icd_version="icd9_10",
        icd_version_col="Diagnosis ICD Code Version (source)",
        return_categories=True,
    )
    .lazy()
    .select(
        STAY_KEY,
        # Ischemic heart disease
        "Congestive heart failure",
        "Chronic pulmonary disease",
        "Renal failure",
        pl.max_horizontal("Diabetes uncomplicated", "Diabetes complicated").alias("Diabetes"),
        "Liver disease",
        "Alcohol abuse",
        "Drug abuse",
        pl.max_horizontal("Metastatic cancer", "Solid tumor without metastasis").alias("Cancer"),
        "AIDS/HIV", # proxy for "Immunocompromised" in VASST
        # Solid organ transplant
        # Steroid use
        # Recent trauma
    )
) # fmt: skip

pl.DataFrame(
    {
        "index": [0],
        "category": ["Ischemic heart disease"],
        "icd9_codes": ["410|411|412|413|414"],
        "icd10_codes": ["I20|I21|I22|I23|I24|I25"],
        "weights": [1],
    }
).write_csv(FOLDER + "additional_comorbidities.csv")

ADDITIONAL = (
    comorbidity(
        score="custom",
        df=DIAGNOSES.collect(),
        id_col="Global ICU Stay ID",
        code_col="Diagnosis ICD Code",
        icd_version="icd9_10",
        icd_version_col="Diagnosis ICD Code Version (source)",
        definition_data=Path(FOLDER + "additional_comorbidities.csv"),
        return_categories=True,
    )
    .drop("Custom Comorbidity Score")
    .lazy()
)

# region 4.6 wide format
# ------------------------------------------------------------------------------
# 4.6 Push aggregated variables to wide format DF for analyses
#     Single row per patient; all covariates reflect the 0–6h window.
# ------------------------------------------------------------------------------
COVARIATES = (
    INCLUDED.join(PATIENT_FEATURES,     on=STAY_KEY, how="left")
    .collect()
    .join(COVARIATES_VITALS.collect(),  on=STAY_KEY, how="left")
    .join(COVARIATES_LABS.collect(),    on=STAY_KEY, how="left")
    .join(COVARIATES_NE.collect(),      on=STAY_KEY, how="left")
    .join(COVARIATES_VENT.collect(),    on=STAY_KEY, how="left")
    .join(COVARIATES_FLUIDS.collect(),  on=STAY_KEY, how="left")
    .join(COVARIATES_SOFA.collect(),    on=STAY_KEY, how="left")
    .join(COVARIATES_APACHE2.collect(), on=STAY_KEY, how="left")
    .join(ELIXHAUSER.collect(),         on=STAY_KEY, how="left")
    .join(ADDITIONAL.collect(),         on=STAY_KEY, how="left")
)  # fmt: skip

# region 5. final cohort
################################################################################
# 5. Select study cohort and create Figure 1 / study flow
print("5. Creating final study cohort")
# ------------------------------------------------------------------------------
# Single-row analytical table: one patient = one row, evaluated at T_landmark.
# No time-series structure; suitable for propensity score estimation and
# overlap-weighted logistic regression (B_analysis.py).
STUDY_COHORT = (
    INCLUDED.collect()
    .join(EXPOSURE.collect(), on=STAY_KEY, how="left")
    .join(OUTCOME.collect(),  on=STAY_KEY, how="left")
    .join(COVARIATES,         on=STAY_KEY, how="left")
    .unique()
    .with_columns(
        pl.col(
            "Mortality in ICU",
            "Mortality in Hospital",
            "Mortality 30d After Shock Onset",
            "Early Vasopressin",
            "Delayed Vasopressin",
            "Ventilated at T_landmark",
        ).fill_null(False),
    )
)  # fmt: skip

# ------------------------------------------------------------------------------

STUDY_COHORT.sort(STAY_KEY).lazy().sink_parquet(FOLDER + "STUDY_COHORT.parquet")

# region 6. quality check
################################################################################
# 6. Quality check of study cohort
# Create Data Quality Report based on Y-data profile, assess the following aspects:
# 6.1 Alerts highlighted by the report
# 6.2 Overall cohort size and feasibility, especially focusing on systematic patterns of missingness across years / ICUs / centers
# 6.3 Descriptives for all aggregated variables
# 6.4 Expected interactions and correlation of variables for sanity checks
# 6.5 Patterns of missingness
# 6.6 Select a sample of X observations for front-end double check by clinician (for relevant variables of primary)

# region 7. Table 1
################################################################################
# 7. Create Table 1 across treatment arms and assess carefully with clinicians
print("7. Creating Table 1")
# ------------------------------------------------------------------------------
columns = [
    "Mortality in ICU",
    "Mortality in Hospital",
    "Mortality 30d After Shock Onset",
    "Admission Age (years)",
    "Admission Height (cm)",
    "Admission Weight (kg)",
    "Gender",
    "Ethnicity",
    "Source Dataset",
    "Max SOFA (0-6h)",
    "Peak Norepinephrine (0-6h)",
    "Min MAP (0-6h)",
    "Max Lactate (0-6h)",
    "Max Creatinine (0-6h)",
    "Fluid Balance 0-6h (mL)",
    "Ventilated at T_landmark",
]
categorical = [
    "Source Dataset",
    "Mortality in ICU",
    "Mortality in Hospital",
    "Mortality 30d After Shock Onset",
    "Gender",
    "Ethnicity",
    "Ventilated at T_landmark",
]
nonnormal = [
    "Admission Age (years)",
    "Max SOFA (0-6h)",
    "Peak Norepinephrine (0-6h)",
    "Min MAP (0-6h)",
    "Max Lactate (0-6h)",
    "Max Creatinine (0-6h)",
    "Fluid Balance 0-6h (mL)",
]
table1 = TableOne(
    STUDY_COHORT.with_columns(
        pl.when(
            pl.col("Ethnicity").is_in(
                ["Asian", "Black or African American", "Hispanic or Latino", "White"]
            )
        )
        .then(pl.col("Ethnicity"))
        .otherwise(pl.lit("Other"))
        .alias("Race"),
    ).to_pandas(),
    columns=columns,
    categorical=categorical,
    groupby="Treatment Arm",
    nonnormal=nonnormal,
    pval=False,
    missing=False,
    include_null=True,
    limit={
        "Mortality in ICU": 1,
        "Mortality in Hospital": 1,
        "Mortality 30d After Shock Onset": 1,
        "Gender": 1,
        "Ethnicity": 5,
        "Ventilated at T_landmark": 1,
    },
    order={
        "Mortality in ICU": ["True", "False"],
        "Mortality in Hospital": ["True", "False"],
        "Mortality 30d After Shock Onset": ["True", "False"],
        "Gender": ["Female", "Male"],
        "Ethnicity": [
            "Asian",
            "Black or African American",
            "Hispanic or Latino",
            "White",
            "Other",
        ],
    },
)
table1.tableone.replace("0 (nan)", None, inplace=True)
table1.tableone.replace("0 (0.0)", None, inplace=True)
table1.tableone.replace("nan (nan)", None, inplace=True)
table1.tableone.replace("nan [nan,nan]", None, inplace=True)
table1.to_csv(FOLDER + "table1.csv")

table1_md = (
    table1.tableone.to_markdown()
    .replace("('Grouped by Treatment Arm', '", " " * 31)
    .replace("True", " " * 4)
    .replace("')", " " * 2)
    .replace("('", " " * 2)
    .replace("', '", ", ")
    .replace(".0  ", " " * 4)
    .replace(",   ", " " * 4)
)
with open(FOLDER + "table1.md", "w") as f:
    f.write(table1_md)
