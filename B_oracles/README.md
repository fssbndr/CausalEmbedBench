# Oracle Knobs

Shared infrastructure (`B_oracles/shared/knobs.py`, `oracle_fit.py`) sweeps benchmark difficulty; all knobs default to `1.0` (identity), select a preset via `--knobs-preset <name>` (`oracle_knob_grid.yaml`).

Validate any family's baseline oracle with `.venv/bin/python B_oracles/shared/validate.py <trial>` (`vasst`, `twins`, `ihdp`, `news`, `jobs`, `rctbench`); it writes `VALIDATION_REPORT.txt` next to the parquet and exits non-zero on a gating failure.

- **kappa_overlap (overlap):** scales propensity `e(X)`'s logit relative to the fitted data. `<1` interpolates toward 0.5 (better overlap than reality); `>1` extrapolates further from 0.5 (worse overlap/positivity than reality).
- **gamma_hetero (heterogeneity):** interpolates/extrapolates CATE `tau(X)` relative to the constant ATE. `<1` dampens toward the ATE (less individual-level signal); `>1` exaggerates the spread.
- **xi_effect (effect scale):** scales treatment-effect magnitude; controls causal signal-to-noise.
- **eta_noise (noise):** scales outcome residual variance (SD scales by `sqrt(eta_noise)`); continuous outcomes only.
- **alpha_alignment (alignment):** pre-fit split of covariates between the propensity/outcome models; controls how many covariates are true confounders. Bounded `[0,1]` — `1.0` is a structural ceiling (100% shared), so unlike the other four knobs there's no "beyond baseline" direction.

# Causal Datasets

## IHDP

KNOBS:

- kappa_overlap: not used — `T` is the published, literature-standard assignment; resampling it would break comparability to the canonical IHDP benchmark, and there's no fitted propensity to interpolate.
- gamma_hetero: used — shrinks the published `tau(X)` toward the constant ATE.
- xi_effect: used — scales the published treatment-effect magnitude.
- eta_noise: used — scales residual noise around the published `mu0`/`mu1`.
- alpha_alignment: not used — no fitted propensity model exists to split covariates against.

CITATION:

- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics: A Joint Publication of American Statistical Association, Institute of Mathematical Statistics, Interface Foundation of North America*, 20(1), 217–240. doi:10.1198/jcgs.2010.08162
- Shalit, U., Johansson, F.D. &amp; Sontag, D.. (2017). Estimating individual treatment effect: generalization bounds and algorithms. *Proceedings of the 34th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 70:3076-3085 Available from <https://proceedings.mlr.press/v70/shalit17a.html>.

## JOBS

The randomized subsample (`e==1`) is fit with the same T-learner as RCTBench, giving real `Y0`/`Y1` (`has_individual_cf: true`) instead of the previous `Y0=Y1=yf` placeholder.

KNOBS:

- kappa_overlap: used — fitted propensity `e(X)`, interpolated toward randomization.
- gamma_hetero: used — fitted CATE, shrunk toward the ATE.
- xi_effect: used — fitted treatment effect, scaled directly.
- eta_noise: no effect — JOBS's outcome is binary, so there's no residual variance to scale.
- alpha_alignment: used — pre-fit covariate split between the propensity/outcome models.

CITATION:

- A. Smith, J., & E. Todd, P. (2005). Does matching overcome LaLonde’s critique of nonexperimental estimators? *Journal of Econometrics, 125*(1–2), 305–353. doi:10.1016/j.jeconom.2004.04.011
- Shalit, U., Johansson, F.D. &amp; Sontag, D.. (2017). Estimating individual treatment effect: generalization bounds and algorithms. *Proceedings of the 34th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 70:3076-3085 Available from <https://proceedings.mlr.press/v70/shalit17a.html>.

## NEWS

Same rationale as IHDP: published noise-free `mu0`/`mu1`, `T` preserved exactly.

KNOBS:

- kappa_overlap: not used — `T` is the published assignment; no fitted propensity to interpolate.
- gamma_hetero: used — shrinks the published `tau(X)` toward the constant ATE.
- xi_effect: used — scales the published treatment-effect magnitude.
- eta_noise: used — scales residual noise around the published `mu0`/`mu1`.
- alpha_alignment: not used — no fitted propensity model exists to split covariates against.

CITATION:

- Johansson, F., Shalit, U. &amp; Sontag, D.. (2016). Learning Representations for Counterfactual Inference. *Proceedings of The 33rd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 48:3020-3029 Available from <https://proceedings.mlr.press/v48/johansson16.html>.

## TWINS

`Y0`/`Y1` are real historical outcomes (twin mortality) and are never perturbed.

KNOBS:

- kappa_overlap: used — interpolates the synthetic per-seed CEVAE propensity toward randomization.
- gamma_hetero: not used — `Y0`/`Y1` are real observed data, not a fitted response surface to shrink.
- xi_effect: not used — same reason as gamma_hetero, there's no fitted effect to scale.
- eta_noise: not used — outcomes are real and binary, not sampled from a noise model.
- alpha_alignment: not used — no fitted outcome model consumes covariates to split against.

CITATION:

- Louizos, C., Shalit, U., Mooij, J., Sontag, D., Zemel, R., & Welling, M. (2017). Causal effect inference with deep latent-variable models. doi:10.48550/arXiv.1705.08821

# Real-World Datasets

## VASST

Real ICU covariates (MIMIC-III/IV, eICU-CRD, AmsterdamUMCdb). `e(X)` is fit in-repo via
logistic regression on the observational cohort and used as-is for generation (not
recalibrated) — VASST's own 1:1 randomization ratio is retained only as a diagnostic
constant (`pi_T` in `causal_params.json`), not a target the fit is expected to match.
Baseline risk `mu0` is fit via logistic regression on the control arm alone, then
recalibrated so its mean matches VASST's published control-arm mortality. Heterogeneity
(`log_or` surface) is evidence-anchored to VASST Table 3 subgroup log-ORs (lactate
quartile, NE-dose split) rather than fit to the trial's own patient-level outcomes.

KNOBS:

- kappa_overlap: used — fitted propensity `e(X)`, interpolated toward VASST's 1:1 randomization ratio at `<1`, extrapolated away from it at `>1` (per shared knob semantics above); not gated in oracle generation or validation.
- gamma_hetero: used — evidence-anchored `tau(X)` (Table 3 subgroup surface), shrunk toward the ATE.
- xi_effect: used — evidence-anchored treatment effect, scaled directly.
- eta_noise: no effect — VASST's outcome is binary, so there's no residual variance to scale.
- alpha_alignment: used — pre-fit covariate split between the propensity/outcome models.

CITATION:

- Russell, J. A., Walley, K. R., Singer, J., Gordon, A. C., Hébert, P. C., Cooper, D. J., Holmes, C. L., Mehta, S., Granton, J. T., Storms, M. M., Cook, D. J., Presneill, J. J., & Ayers, D. (2008). Vasopressin versus norepinephrine infusion in patients with septic shock. *The New England Journal of Medicine, 358*(9), 877–887. doi:10.1056/NEJMoa067373

## RCT Bench

Both propensity and outcome models are fit in-repo per trial.

KNOBS:

- kappa_overlap: used — fitted propensity `e(X)`, interpolated toward randomization.
- gamma_hetero: used — fitted CATE, shrunk toward the ATE.
- xi_effect: used — fitted treatment effect, scaled directly.
- eta_noise: used for continuous-outcome trials only — no effect on the binary-outcome trials in the suite (no residual variance to scale).
- alpha_alignment: used — pre-fit covariate split between the propensity/outcome models.

CITATION:

- Shao, Y., Lyu, L., Yu, M., & Wang, B. (2026). How should covariates be handled in randomized trials? Empirical evidence from 50 trials and recommendations for practice. *Journal of Clinical Epidemiology, 197*(112374), 112374. doi:10.1016/j.jclinepi.2026.112374
