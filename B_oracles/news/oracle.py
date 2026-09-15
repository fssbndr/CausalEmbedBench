# Semi-synthetic oracle generation for the causal embedding benchmark (NEWS trial).
#
# Reads:  raw/NEWS/csv/topic_doc_mean_n5000_k3477_seed_{1..10}.csv.{x,y}
# Writes: data/oracles/news/semi_synthetic.parquet
#         data/oracles/news/feature_cols.json
#         data/oracles/news/causal_params.json
#
# NEWS (Johansson et al. 2016): semi-synthetic outcome model on real document-topic features.
# Covariates: sparse word/topic counts from 5000 documents, 3477 topics (reconstructed dense).
# Ground truth: Y0 = mu0 (control potential outcome, no noise), Y1 = mu1 (treatment, no noise),
#               Y_factual = noisy realization (yf, ycf based on observed T),
#               true_tau = mu1 - mu0 (ground-truth CATE).
#
# Only gamma/s/nu apply, same rationale as IHDP: mu0/mu1 are a real published response
# surface; T is the published assignment and must stay untouched. At the identity preset,
# output is the exact published data, unchanged.

import os
import sys

import numpy as np
import polars as pl
import scipy.sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from config import SEEDS
from knobs import (
    parse_oracle_args,
    PUBLISHED_KNOBS,
    knob_params,
    resample_published_outcome,
    setup_oracle_dir,
)
from oracle_io import assemble_seed_df, should_log_seed, skip_if_exists, write_causal_params, write_feature_cols

args, knob_config = parse_oracle_args(
    skip_help="Skip regeneration if the output parquet already exists",
    knob_note="Only gamma/s/nu have any effect for NEWS — T is the published assignment and stays untouched.",
)

# ---------------------------------------------------------------------------
NEWS_PATH = "raw/NEWS/csv"
TRIAL = "news"
OUT_DIR = setup_oracle_dir(TRIAL, args.knobs_preset)

skip_if_exists(OUT_DIR, args.skip_existing)

# ---------------------------------------------------------------------------
# Helper: load sparse triplet format from .csv.x file
# ---------------------------------------------------------------------------
def load_sparse_triplet(filepath):
    """Load sparse triplet file. First row: n_rows, n_cols, padding. Rest: i,j,v."""
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
        n_rows, n_cols = int(header[0]), int(header[1])

    data = np.loadtxt(filepath, skiprows=1, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)

    i = data[:, 0].astype(int) - 1  # 1-indexed -> 0-indexed
    j = data[:, 1].astype(int) - 1
    v = data[:, 2]

    sparse_mat = scipy.sparse.coo_matrix((v, (i, j)), shape=(n_rows, n_cols))
    return sparse_mat.toarray().astype(np.float32)

# ---------------------------------------------------------------------------
# 1. Load NEWS data for seeds 1-10
# ---------------------------------------------------------------------------
print("1. Loading NEWS data for seeds 1-10")
news_data = []

for seed in SEEDS:
    seed_num = seed + 1  # NEWS seeds are 1-indexed
    y_file = f"{NEWS_PATH}/topic_doc_mean_n5000_k3477_seed_{seed_num}.csv.y"
    x_file = f"{NEWS_PATH}/topic_doc_mean_n5000_k3477_seed_{seed_num}.csv.x"

    # Dense .y file: [t, yf, ycf, mu0, mu1]
    y_data = np.loadtxt(y_file, delimiter=',')
    t = y_data[:, 0].astype(np.int8)
    yf, ycf, mu0, mu1 = y_data[:, 1], y_data[:, 2], y_data[:, 3], y_data[:, 4]

    print(f"   seed={seed_num:2d}: loading sparse X from {x_file.split('/')[-1]}")
    X = load_sparse_triplet(x_file)

    y0, y1, y_obs = resample_published_outcome(seed, t, mu0, mu1, yf, ycf, knob_config)

    news_data.append({"seed": seed, "X": X, "t": t, "yf": y_obs, "mu0": y0, "mu1": y1})

    if should_log_seed(seed):
        ate_seed = (y1 - y0).mean()
        print(f"      shape: X={X.shape}, ATE={ate_seed:.4f}, mean(mu0)={y0.mean():.3f}, mean(mu1)={y1.mean():.3f}")

n_docs = news_data[0]["X"].shape[0]
n_features = news_data[0]["X"].shape[1]
print(f"   Loaded {len(SEEDS)} seeds: {n_docs} documents × {n_features} features")

# ---------------------------------------------------------------------------
# 2. Write feature columns
# ---------------------------------------------------------------------------
print("2. Extracting ground-truth potential outcomes")

feature_cols = [f"topic_{i}" for i in range(n_features)]
print(f"   Feature columns: {n_features} topics (topic_0 to topic_{n_features-1})")

write_feature_cols(OUT_DIR, feature_cols)

# ---------------------------------------------------------------------------
# 3. Write causal parameters
# ---------------------------------------------------------------------------
print("3. Writing metadata")

global_ate = np.concatenate([d["mu1"] - d["mu0"] for d in news_data]).mean()

causal_params = {
    "n":             int(n_docs),
    "outcome_type":  "continuous",
    "true_ATE":      float(global_ate),
    "ood_sources":   [],
    "x_varies_by_seed": True,  # each seed is a distinct published replication, not a noise redraw
    **knob_params(knob_config, PUBLISHED_KNOBS, "published"),
}
write_causal_params(OUT_DIR, causal_params)

print(f"   Saved {OUT_DIR}/causal_params.json (global ATE={global_ate:.4f})")

# ---------------------------------------------------------------------------
# 4. Assemble and write Parquet
# ---------------------------------------------------------------------------
print(f"4. Writing {OUT_DIR}/semi_synthetic.parquet")

oracle_dfs = []

for data in news_data:
    seed, X, t = data["seed"], data["X"], data["t"]
    yf, mu0, mu1 = data["yf"], data["mu0"], data["mu1"]
    true_tau = mu1 - mu0

    oracle_dfs.append(assemble_seed_df(
        seed, n_docs, t.astype(np.float32), yf.astype(np.float32), mu0.astype(np.float32),
        mu1.astype(np.float32), true_tau.astype(np.float32), "NEWS", X, feature_cols,
    ))

oracle_df = pl.concat(oracle_dfs)
oracle_df.write_parquet(f"{OUT_DIR}/semi_synthetic.parquet")
print(f"   Written {len(oracle_df)} rows × {oracle_df.width} columns "
      f"({len(SEEDS)} seeds × {n_docs} documents, {n_features} features).")
print("   Done.")
