"""
Pipeline configuration: shared constants across all stages.
"""

import os

# Oracle seed replications for stable ranking across methods
# Higher number -> more stable PEHE rankings but longer runtime
SEEDS = list(range(10))

# Training epochs for all embedding models (unsupervised and supervised)
# 100 epochs found to be sufficient for convergence on this scale
EPOCHS = 100

# Cross-validation folds for evaluation (PEHE, ATE, policy value, etc.)
# 5-fold: more training data per fold than 3-fold, particularly for small treatment
# arms (e.g. IHDP's ~125 treated units) where fewer folds starve arm-conditional fits
CV_FOLDS = 5

# Optional: subsample per seed during evaluation (None = full cohort, int = bootstrap n per seed)
SAMPLE_N = None

# Below this many fit samples, GPU kernel-launch overhead outweighs the speedup
CUDA_MIN_N = 2000


def available_cpus() -> int:
    """CPUs actually allocated to this job — SLURM-aware, not the node total.

    os.cpu_count() can report the full node's core count rather than a SLURM
    job's cgroup allocation, which oversubscribes worker pools sized off it.
    """
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4
