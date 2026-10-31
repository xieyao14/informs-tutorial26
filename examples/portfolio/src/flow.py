"""Define the velocity network, Euler sampler, and empirical RBF MMD."""

import math

import torch
from torch import nn


# === Velocity network and time embedding ===
class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal time features for the velocity network."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None] * self.freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FlowMLP(nn.Module):
    """Time-dependent velocity field for flow matching."""

    def __init__(
        self,
        dim: int,
        hidden_width: int,
        hidden_depth: int,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        layers: list[nn.Module] = []
        in_dim = dim + time_embed_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.SiLU())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))


# === Euler sampling ===
@torch.no_grad()
def sample_euler(
    model: nn.Module,
    n_samples: int,
    dim: int,
    device: torch.device,
    n_steps: int,
    seed: int,
) -> torch.Tensor:
    """Integrate the learned velocity field from Gaussian noise at t=1 to t=0."""
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n_samples, dim, device=device, generator=rng)
    dt = 1.0 / n_steps
    for step in range(n_steps, 0, -1):
        t = torch.full((n_samples,), step / n_steps, device=device)
        v = model(x, t)
        x = x - dt * v
    return x


# === Empirical RBF MMD ===
def _rbf_bandwidth_from_samples(z: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(z, z, p=2).pow(2)
    bandwidth = torch.median(dists[dists > 0])
    if not torch.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = torch.tensor(1.0, device=z.device, dtype=z.dtype)
    return bandwidth


def _subsample_rows(z: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if z.shape[0] <= max_rows:
        return z
    rng = torch.Generator(device=z.device)
    rng.manual_seed(seed)
    idx = torch.randperm(z.shape[0], generator=rng, device=z.device)[:max_rows]
    return z[idx]


def _mean_rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    for start in range(0, x.shape[0], chunk_size):
        dist2 = torch.cdist(x[start : start + chunk_size], y, p=2).pow(2)
        total += torch.exp(-dist2 / (2.0 * bandwidth)).sum()
    return total / (x.shape[0] * y.shape[0])


def compute_rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int = 0,
    max_samples: int = 4096,
    max_bandwidth_samples: int = 2048,
) -> float:
    """Compute empirical RBF MMD squared, including diagonal kernel terms.

    The bandwidth is the median positive squared distance between real samples x.
    """
    x = x.float()
    y = y.float()
    x_eval = _subsample_rows(x, max_samples, seed)
    y_eval = _subsample_rows(y, max_samples, seed + 1)
    bandwidth_data = _subsample_rows(x_eval, max_bandwidth_samples, seed + 2)
    bandwidth = _rbf_bandwidth_from_samples(bandwidth_data)
    mmd2 = (
        _mean_rbf_kernel(x_eval, x_eval, bandwidth)
        + _mean_rbf_kernel(y_eval, y_eval, bandwidth)
        - 2.0 * _mean_rbf_kernel(x_eval, y_eval, bandwidth)
    )
    return float(torch.clamp(mmd2, min=0.0).item())
