"""
PyTorch implementation of SITE (Yao et al. 2018).

Same backbone as CFRNet.py, plus two similarity-preserving terms on six propensity-
selected hard points per batch: a PDDM loss weighted by β_pddm matching predicted to
propensity-based pairwise similarity, and a mid-point distance loss weighted by
β_mid balancing cross-arm representation midpoints.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.optim import Adam

from .base import FC, BaseEstimator, ipw_weights, select_by_treatment


def _find_nearest_point(x, p):
    """Index of the closest value to `p` in `x`, excluding exact matches (util.py find_nearest_point(), lines 197-208)."""
    diff = np.abs(x - p)
    diff = np.where(diff > 0, diff, np.inf)
    return int(np.argmin(diff))


def _find_middle_pair(x, y):
    """Indices (into x, y) of the pair closest to propensity 0.5 jointly (util.py find_middle_pair(), lines 258-269)."""
    grid = np.abs(x[:, None] - 0.5) + np.abs(y[None, :] - 0.5)
    index_1, index_2 = np.unravel_index(np.argmin(grid), grid.shape)
    return int(index_1), int(index_2)


def _find_six_points(prop, t):
    """
    Select the six hard points (i, j, k, l, m, n) per util.py find_three_pairs(), lines 211-256:
    - i, j: treated/control pair closest to propensity 0.5 (near decision boundary)
    - k: control point with propensity farthest from prop[i]; l: control point nearest to k
    - m: treated point with propensity farthest from prop[j]; n: treated point nearest to m

    Args:
        prop: (n,) propensity scores for the batch
        t: (n,) binary treatment for the batch

    Returns: (i, j, k, l, m, n) indices into the batch, or None if either arm is empty
    """
    idx_t = np.where(t > 0.5)[0]
    idx_c = np.where(t <= 0.5)[0]
    if idx_t.size == 0 or idx_c.size == 0:
        return None

    prop_t = prop[idx_t]
    prop_c = prop[idx_c]

    ii, jj = _find_middle_pair(prop_t, prop_c)
    kk = int(np.argmax(np.abs(prop_c - prop_t[ii])))
    ll = _find_nearest_point(prop_c, prop_c[kk])
    mm = int(np.argmax(np.abs(prop_t - prop_c[jj])))
    nn = _find_nearest_point(prop_t, prop_t[mm])

    return idx_t[ii], idx_c[jj], idx_c[kk], idx_c[ll], idx_t[mm], idx_t[nn]


def _similarity_score(s_i, s_j):
    """Linear propensity similarity (util.py similarity_score(), lines 139-148, mode='linear')."""
    mid = (s_i + s_j) / 2.0
    dist = np.abs(s_j - s_i) / 2.0
    return (1.5 * np.abs(mid - 0.5) - 2 * dist + 1) / 2.0


class SITE(BaseEstimator):
    def __init__(self, x_dim, h=200, nh=3, d=200, dim_pddm=200, dim_c=200, dim_s=100,
                 beta_pddm=1.0, beta_mid=1.0, lr=1e-3, outcome_type="binary"):
        """
        Args:
            x_dim: input (covariate) dimension
            h: hidden layer size (encoder trunk and outcome heads)
            nh: number of hidden layers
            d: representation (bottleneck) dimension
            dim_pddm: PDDM unit's u/v projection dimension
            dim_c: PDDM unit's concatenated-projection dimension
            dim_s: PDDM unit's final similarity-embedding dimension
            beta_pddm: PDDM loss weight (site_net.py FLAGS.p_pddm)
            beta_mid: mid-point distance loss weight (site_net.py FLAGS.p_mid_point_mini)
            lr: learning rate for Adam optimizer
            outcome_type: "binary" (log loss) or "continuous" (squared loss)
        """
        super().__init__(outcome_type)
        self.x_dim = x_dim
        self.d = d
        self.beta_pddm = beta_pddm
        self.beta_mid = beta_mid
        self.lr = lr

        # ──────────────────────────────────────────────────────────────────────
        # REPRESENTATION (encoder) — site_net.py lines 84-125, same as CFRNet
        # ──────────────────────────────────────────────────────────────────────
        self.encoder = FC([x_dim] + [h] * (nh - 1) + [d], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # OUTPUT HEADS (split_output=1, the ihdp.txt default) — site_net.py lines 330-342
        # ──────────────────────────────────────────────────────────────────────
        self.y0_head = FC([d] + [h] * nh + [1], activation=nn.ELU)
        self.y1_head = FC([d] + [h] * nh + [1], activation=nn.ELU)

        # ──────────────────────────────────────────────────────────────────────
        # PDDM unit — (site_net.py pddm(), lines 268-325)
        # ──────────────────────────────────────────────────────────────────────
        self.pddm_u = nn.Linear(d, dim_pddm)
        self.pddm_v = nn.Linear(d, dim_pddm)
        self.pddm_c = nn.Linear(2 * dim_pddm, dim_c)
        self.pddm_s = nn.Linear(dim_c, dim_s)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.propensity_model = None

    def pddm(self, r_i, r_j):
        """
        PDDM similarity score for a representation pair (site_net.py pddm(), lines 268-325).

        Args:
            r_i, r_j: (1, d) representation rows

        Returns:
            s: (1, 1) predicted similarity
        """
        u = torch.abs(r_i - r_j)
        v = (r_i + r_j) / 2.0
        u1 = F.elu(self.pddm_u(u))
        v1 = F.elu(self.pddm_v(v))
        u1 = F.normalize(u1, dim=0)
        v1 = F.normalize(v1, dim=0)
        c = F.elu(self.pddm_c(torch.cat([u1, v1], dim=-1)))
        return self.pddm_s(c)

    def _six_point_loss(self, x, r, t, y):
        """
        PDDM + mid-point-distance loss over the batch's six hard points
        (site_net.py lines 172-202, util.py find_three_pairs(), lines 211-256 /
        get_three_pair_simi(), lines 271-282).

        Ground-truth similarity comes from propensity scores of the raw covariates x;
        predicted similarity/midpoints come from the learned representation r.
        Falls back to zero loss if either arm is empty in this batch (as in the
        reference's try/except in find_three_pairs).

        Args:
            x: (n, x_dim) raw covariates (for propensity lookup)
            r: (n, d) representation
            t: (n,) binary treatment
            y: (n,) outcome (unused, kept for call-site symmetry with loss())

        Returns:
            pddm_loss, mid_distance: scalars
        """
        prop = self.propensity_model.predict_proba(x.detach().cpu().numpy())[:, 1]
        points = _find_six_points(prop, t.detach().cpu().numpy())
        if points is None:
            zero = torch.zeros((), dtype=r.dtype)
            return zero, zero
        i, j, k, l, m, n = points

        s_kl = self.pddm(r[k:k+1], r[l:l+1])
        s_mn = self.pddm(r[m:m+1], r[n:n+1])
        s_km = self.pddm(r[k:k+1], r[m:m+1])
        s_ik = self.pddm(r[i:i+1], r[k:k+1])
        s_jm = self.pddm(r[j:j+1], r[m:m+1])

        simi_kl = _similarity_score(prop[k], prop[l])
        simi_mn = _similarity_score(prop[m], prop[n])
        simi_km = _similarity_score(prop[k], prop[m])
        simi_ik = _similarity_score(prop[i], prop[k])
        simi_jm = _similarity_score(prop[j], prop[m])

        pddm_loss = (
            (simi_kl - s_kl) ** 2 + (simi_mn - s_mn) ** 2 + (simi_km - s_km) ** 2
            + (simi_ik - s_ik) ** 2 + (simi_jm - s_jm) ** 2
        ).sum()

        mid_jk = (r[j] + r[k]) / 2.0
        mid_im = (r[i] + r[m]) / 2.0
        mid_distance = ((mid_jk - mid_im) ** 2).sum()

        return pddm_loss, mid_distance

    def loss(self, x, t, y):
        """
        Factual loss + PDDM loss + mid-point distance loss (site_net.py lines 150-214).

        Args:
            x: (n, x_dim) covariates
            t: (n,) binary treatment
            y: (n,) outcome

        Returns:
            total_loss: scalar
        """
        r = self.encode(x)

        y0_pred = self.y0_head(r).squeeze(-1)
        y1_pred = self.y1_head(r).squeeze(-1)
        pred_y = select_by_treatment(t, y0_pred, y1_pred)

        w = ipw_weights(t)  # inverse propensity weights, batch treatment rate stabilized

        if self.outcome_type == "binary":
            per_sample_loss = F.binary_cross_entropy_with_logits(pred_y, y, reduction='none')
        else:
            per_sample_loss = F.mse_loss(pred_y, y, reduction='none')

        factual_loss = (w * per_sample_loss).mean()

        pddm_loss, mid_distance = self._six_point_loss(x, r, t, y)

        return factual_loss + self.beta_pddm * pddm_loss + self.beta_mid * mid_distance

    def _before_fit(self, x, t, y):
        """Fit propensity model once on raw covariates (simi_ite/propensity.py lines 29-30).

        Args:
            x: (n, x_dim) covariates (numpy array)
            t: (n,) binary treatment (numpy array)
            y: (n,) outcome (numpy array, unused)
        """
        self.propensity_model = LogisticRegression(
            penalty="l2", class_weight="balanced", C=3.0, max_iter=1000
        ).fit(x, t)
