# Causal Datasets

sourced from originals, as collected in [doi:10.6084/m9.figshare.30244936](https://figshare.com/articles/dataset/Causal_Machine_Learning_Benchmark_Datasets/30244936).

## IHDP

Variables in the .npz x, t, yf, ycf, mu0, mu1 are covariates, treatment, factual outcome, counterfactual outcome, and noiseless potential outcomes respectively.

Download:

- <https://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip>
- <https://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip>

CITATION:

- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics: A Joint Publication of American Statistical Association, Institute of Mathematical Statistics, Interface Foundation of North America*, 20(1), 217–240. doi:10.1198/jcgs.2010.08162
- Shalit, U., Johansson, F.D. &amp; Sontag, D.. (2017). Estimating individual treatment effect: generalization bounds and algorithms. *Proceedings of the 34th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 70:3076-3085 Available from <https://proceedings.mlr.press/v70/shalit17a.html>.

## JOBS

Variables in the .npz x, t, yf, e are covariates, treatment, factual outcome and indicator for original randomized sample respectively.

Download:

- <https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz>
- <https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz>

CITATION:

- A. Smith, J., & E. Todd, P. (2005). Does matching overcome LaLonde’s critique of nonexperimental estimators? *Journal of Econometrics, 125*(1–2), 305–353. doi:10.1016/j.jeconom.2004.04.011
- Shalit, U., Johansson, F.D. &amp; Sontag, D.. (2017). Estimating individual treatment effect: generalization bounds and algorithms. *Proceedings of the 34th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 70:3076-3085 Available from <https://proceedings.mlr.press/v70/shalit17a.html>.

## NEWS

Download:

- <https://www.fredjo.com/files/NEWS_csv.zip>

CITATION:

- Johansson, F., Shalit, U. &amp; Sontag, D.. (2016). Learning Representations for Counterfactual Inference. *Proceedings of The 33rd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 48:3020-3029 Available from <https://proceedings.mlr.press/v48/johansson16.html>.

## TWINS

Download:

- <https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS>

CITATION:

- Louizos, C., Shalit, U., Mooij, J., Sontag, D., Zemel, R., & Welling, M. (2017). Causal effect inference with deep latent-variable models. doi:10.48550/arXiv.1705.08821

# Real-World Datasets

## RCT Bench

- <https://github.com/syl051088/RCT_Bench>
- <https://github.com/BingkaiWang/RCT_Bench-main>
- <https://bingkaiwang.github.io/RCT_Bench-main/website/>

Download:

- <https://bingkaiwang.github.io/RCT_Bench-main/meta_data.xlsx>
- <https://bingkaiwang.github.io/RCT_Bench-main/data-dictionary.xlsx>
- `for i in {1..125}; do wget "https://bingkaiwang.github.io/RCT_Bench-main/cleaned_data/trial${i}.csv"; done`

CITATION:

- Shao, Y., Lyu, L., Yu, M., & Wang, B. (2026). How should covariates be handled in randomized trials? Empirical evidence from 50 trials and recommendations for practice. *Journal of Clinical Epidemiology, 197*(112374), 112374. doi:10.1016/j.jclinepi.2026.112374
