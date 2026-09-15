import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EPSILON = 1e-6


class FC(nn.Sequential):
    """
    Fully connected network: Linear -> activation -> ... -> Linear (no final activation).

    Args:
        sizes: list of layer dimensions [in, h1, h2, ..., out]
        activation: activation function class (default: ELU for VAE/CFRNet, ReLU for AE)
    """
    def __init__(self, sizes, activation=nn.ELU):
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:  # not the last (output) layer
                layers.append(activation())
        super().__init__(*layers)


def select_by_treatment(t, y0, y1):
    """Select the treatment-conditioned prediction per sample: t*y1 + (1-t)*y0."""
    return t * y1 + (1 - t) * y0


def ipw_weights(t):
    """Inverse-propensity weights from the batch treatment rate: t/(2u) + (1-t)/(2*(1-u))."""
    u = torch.clamp(t.mean(), min=EPSILON, max=1 - EPSILON)
    return t / (2 * u) + (1 - t) / (2 * (1 - u))


def linear_mmd(r, t):
    """
    Linear MMD^2 balance penalty between treated/control representations.

    Zero if either arm is empty in the batch.

    Args:
        r: (n, d) representation
        t: (n,) binary treatment

    Returns:
        mmd: scalar
    """
    r1 = r[t > 0.5]
    r0 = r[t < 0.5]
    if r1.shape[0] == 0 or r0.shape[0] == 0:
        return r.new_zeros(())
    u = torch.clamp(t.mean(), min=EPSILON, max=1 - EPSILON)
    mean_diff = 2 * u * r1.mean(0) - 2 * (1 - u) * r0.mean(0)
    return (mean_diff ** 2).sum()


class BaseModel(nn.Module):
    """Sklearn-style embedding model interface: encode/decode/fit/transform."""

    def encode(self, x):
        """Encode covariates x -> d-dimensional representation.

        Default implementation calls self.encoder(x). Override when the
        architecture uses a different encoder shape (e.g. VAE with separate
        mu/logvar heads, or disentangled multi-tensor representations)."""
        return self.encoder(x)

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, x, *args, **kwargs):
        raise NotImplementedError

    def transform(self, x):
        """Transform covariates x -> numpy representation array.

        Encodes x via self.encode(), moves result to CPU, detaches, and
        returns as float32 numpy array. Override when encode() returns
        multiple tensors or requires different post-processing."""
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            r = self.encode(x)
        return r.detach().cpu().numpy().astype(np.float32)

    def fit_transform(self, x, *args, **kwargs):
        """Fit then transform (sklearn convenience method)."""
        return self.fit(x, *args, **kwargs).transform(x)


class BaseEstimator(BaseModel):
    """BaseModel + outcome_type-aware treatment-effect estimator interface."""

    def __init__(self, outcome_type="binary"):
        """
        Args:
            outcome_type: "binary" (Bernoulli p(y|...), log loss) or
                          "continuous" (unit-variance Normal p(y|...), squared loss)
        """
        super().__init__()
        if outcome_type not in ("binary", "continuous"):
            raise ValueError(f"outcome_type must be 'binary' or 'continuous', got {outcome_type!r}")
        self.outcome_type = outcome_type

    def predict(self, x, t):
        """Predict potential outcomes y_0(x) and y_1(x), select by treatment.

        Default: encodes x, passes through self.y0_head (control head) and self.y1_head
        (treated head), then selects by treatment indicator t via
        select_by_treatment(t, y0_pred, y1_pred). Override when the model
        has a different head structure (BNN's single shared head, DRCFR's
        alternative encoding, or special outcome logic)."""
        r = self.encode(x)
        y0_pred = self.y0_head(r).squeeze(-1)
        y1_pred = self.y1_head(r).squeeze(-1)
        return select_by_treatment(t, y0_pred, y1_pred)

    def loss(self, x, t, y):
        raise NotImplementedError

    def _y_nll(self, pred, y):
        """Negative log-likelihood of y under p(y|...): Bernoulli (binary) or unit-variance Normal (continuous)."""
        if self.outcome_type == "binary":
            return F.binary_cross_entropy_with_logits(pred, y, reduction='mean')
        return 0.5 * F.mse_loss(pred, y, reduction='mean')

    def _y_mean(self, pred):
        """Mean of p(y|...) in outcome space: sigmoid(pred) (binary) or pred (continuous)."""
        return torch.sigmoid(pred) if self.outcome_type == "binary" else pred

    def _before_fit(self, x, t, y):
        """Hook: called once with raw numpy x, t, y before tensor conversion. No-op by default.

        Args:
            x: (n, x_dim) covariates (numpy array)
            t: (n,) treatment (numpy array)
            y: (n,) outcome (numpy array)
        """
        pass

    def _before_epoch(self, x):
        """Hook: called once per epoch before the batch loop, with tensorized x. No-op by default.

        Args:
            x: (n, x_dim) tensorized covariates
        """
        pass

    def _build_batch(self, idx, x, t, y):
        """Hook: construct (x_batch, t_batch, y_batch) from permutation-index idx. Default: direct slicing.

        Args:
            idx: (b,) permutation-sampled indices into full training set
            x: (n, x_dim) full tensorized covariates
            t: (n,) full tensorized treatment
            y: (n,) full tensorized outcome

        Returns:
            (x_batch, t_batch, y_batch): batched tensors
        """
        return x[idx], t[idx], y[idx]

    def _step(self, x_batch, t_batch, y_batch, optimizer):
        """Hook: one optimizer step on one batch. Default: zero_grad/loss/backward/clip/step.

        Args:
            x_batch: (b, x_dim) batch covariates
            t_batch: (b,) batch treatment
            y_batch: (b,) batch outcome
            optimizer: optimizer instance for this step

        Returns:
            loss_value: scalar loss for epoch accumulation
        """
        optimizer.zero_grad()
        total_loss = self.loss(x_batch, t_batch, y_batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return total_loss.item()

    def _run_sgd(self, x, t, y, optimizer, epochs, batch_size, log_every):
        """Run minibatch SGD loop via _before_epoch, _build_batch, _step hooks.

        Args:
            x: (n, x_dim) tensorized covariates
            t: (n,) tensorized treatment
            y: (n,) tensorized outcome
            optimizer: optimizer instance
            epochs: number of training epochs
            batch_size: batch size per epoch
            log_every: print loss every N epochs (0 = silent)
        """
        n = x.shape[0]
        for epoch in range(epochs):
            self._before_epoch(x)
            epoch_loss = 0.0
            n_batches = 0
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x_batch, t_batch, y_batch = self._build_batch(idx, x, t, y)
                epoch_loss += self._step(x_batch, t_batch, y_batch, optimizer)
                n_batches += 1
            epoch_loss /= max(1, n_batches)
            if log_every > 0 and (epoch + 1) % log_every == 0:
                print(f"   Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

    def fit(self, x, t, y, epochs=200, batch_size=100, log_every=0):
        """Fit via standard single-optimizer minibatch SGD.

        Args:
            x: (n, x_dim) covariates (numpy array)
            t: (n,) treatment (numpy array)
            y: (n,) outcome (numpy array)
            epochs: number of training epochs
            batch_size: batch size for SGD
            log_every: print loss every N epochs (0 = silent)

        Returns: self (for sklearn-like chaining)
        """
        self._before_fit(x, t, y)
        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.train()
        self._run_sgd(x, t, y, getattr(self, "optimizer", None), epochs, batch_size, log_every)
        return self
