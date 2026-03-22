"""Synthetic experiment: pointwise vs pairwise reward models under bottleneck.

Two-Gaussian mixture setup: points are drawn from G1 or G2 (50/50).
Cross-cluster pairs always prefer G2. Within-cluster pairs prefer the point
closer to the cluster centroid (Mahalanobis distance), with different
non-isotropic covariance per cluster.

Usage:
    uv run scripts/synthetic-experiments/pointwise_vs_pairwise.py \
        --seeds 5 --output-dir figures/out/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Data generation -- two-Gaussian mixture
# ---------------------------------------------------------------------------

D = 10

_gen = torch.Generator().manual_seed(42)
_dir = torch.randn(D, generator=_gen)
_dir = _dir / _dir.norm()

SEPARATION = 2.0
MU1 = -0.5 * SEPARATION * _dir
MU2 = 0.5 * SEPARATION * _dir

SIGMA1_DIAG = torch.cat([torch.full((5,), 2.0), torch.full((5,), 0.3)])
SIGMA2_DIAG = torch.cat([torch.full((5,), 0.3), torch.full((5,), 2.0)])


def mahalanobis_sq(y: torch.Tensor, mu: torch.Tensor, sigma_diag: torch.Tensor) -> torch.Tensor:
    return ((y - mu) ** 2 / sigma_diag).sum(dim=-1)


def sample_mixture(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    cluster = torch.bernoulli(torch.full((n,), 0.5), generator=generator).long()
    n1 = (cluster == 0).sum().item()
    n2 = n - n1
    y = torch.empty(n, D)
    y[cluster == 0] = torch.randn(n1, D, generator=generator) * SIGMA1_DIAG.sqrt() + MU1
    y[cluster == 1] = torch.randn(n2, D, generator=generator) * SIGMA2_DIAG.sqrt() + MU2
    return y, cluster


def preference_label(
    y_i: torch.Tensor, c_i: torch.Tensor, y_j: torch.Tensor, c_j: torch.Tensor
) -> torch.Tensor:
    cross = c_i != c_j
    labels = torch.zeros(y_i.size(0))
    labels[cross] = (c_j[cross] == 1).float()
    same = ~cross
    if same.any():
        yi_s, yj_s, ci_s = y_i[same], y_j[same], c_i[same]
        is_c0 = ci_s == 0
        is_c1 = ci_s == 1
        dist_i = torch.zeros(same.sum())
        dist_j = torch.zeros(same.sum())
        if is_c0.any():
            dist_i[is_c0] = mahalanobis_sq(yi_s[is_c0], MU1, SIGMA1_DIAG)
            dist_j[is_c0] = mahalanobis_sq(yj_s[is_c0], MU1, SIGMA1_DIAG)
        if is_c1.any():
            dist_i[is_c1] = mahalanobis_sq(yi_s[is_c1], MU2, SIGMA2_DIAG)
            dist_j[is_c1] = mahalanobis_sq(yj_s[is_c1], MU2, SIGMA2_DIAG)
        labels[same] = (dist_i > dist_j).float()
    return labels


def generate_training_pairs(
    n: int, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y, c = sample_mixture(2 * n, generator)
    idx = torch.randperm(2 * n, generator=generator)
    y_i, y_j = y[idx[:n]], y[idx[n:]]
    c_i, c_j = c[idx[:n]], c[idx[n:]]
    labels = preference_label(y_i, c_i, y_j, c_j)
    return y_i, y_j, labels


def generate_across_cluster_pairs(
    n: int, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y1 = torch.randn(n, D, generator=generator) * SIGMA1_DIAG.sqrt() + MU1
    y2 = torch.randn(n, D, generator=generator) * SIGMA2_DIAG.sqrt() + MU2
    c1 = torch.zeros(n, dtype=torch.long)
    c2 = torch.ones(n, dtype=torch.long)
    labels = preference_label(y1, c1, y2, c2)
    return y1, y2, labels


def generate_within_cluster_pairs(
    n: int, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    half = n // 2
    y1_a = torch.randn(half, D, generator=generator) * SIGMA1_DIAG.sqrt() + MU1
    y1_b = torch.randn(half, D, generator=generator) * SIGMA1_DIAG.sqrt() + MU1
    y2_a = torch.randn(n - half, D, generator=generator) * SIGMA2_DIAG.sqrt() + MU2
    y2_b = torch.randn(n - half, D, generator=generator) * SIGMA2_DIAG.sqrt() + MU2
    y_i = torch.cat([y1_a, y2_a])
    y_j = torch.cat([y1_b, y2_b])
    c_i = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(n - half, dtype=torch.long)])
    c_j = c_i.clone()
    perm = torch.randperm(n, generator=generator)
    labels = preference_label(y_i, c_i, y_j, c_j)
    return y_i[perm], y_j[perm], labels[perm]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PointwiseReward(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(y)).squeeze(-1)


class PairwiseReward(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, y_i: torch.Tensor, y_j: torch.Tensor) -> torch.Tensor:
        fwd = self.net(torch.cat([y_i, y_j], dim=-1))
        rev = self.net(torch.cat([y_j, y_i], dim=-1))
        return ((fwd - rev) / 2).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_pointwise(
    model: PointwiseReward,
    y_i: torch.Tensor,
    y_j: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = y_i.size(0)
    losses: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            ri = model(y_i[idx])
            rj = model(y_j[idx])
            logit = ri - rj
            sign = 2 * labels[idx] - 1
            loss = -torch.nn.functional.logsigmoid(sign * logit).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.size(0)
        losses.append(epoch_loss / n)
    return losses


def train_pairwise(
    model: PairwiseReward,
    y_i: torch.Tensor,
    y_j: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = y_i.size(0)
    losses: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logit = model(y_i[idx], y_j[idx])
            sign = 2 * labels[idx] - 1
            loss = -torch.nn.functional.logsigmoid(sign * logit).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.size(0)
        losses.append(epoch_loss / n)
    return losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_accuracy(
    pw_model: PointwiseReward,
    pair_model: PairwiseReward,
    y_i: torch.Tensor,
    y_j: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    pw_logit = pw_model(y_i) - pw_model(y_j)
    pw_pred = (pw_logit > 0).float()
    pw_acc = (pw_pred == labels).float().mean().item()

    pair_logit = pair_model(y_i, y_j)
    pair_pred = (pair_logit > 0).float()
    pair_acc = (pair_pred == labels).float().mean().item()
    return pw_acc, pair_acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _gauss2d(X: np.ndarray, Y: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(cov)
    dx, dy = X - mu[0], Y - mu[1]
    return np.exp(-(inv[0, 0] * dx**2 + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy**2) / 2)


def plot_gaussian_contour(ax: plt.Axes) -> None:
    mu1 = np.array([-1.5, 0.0])
    mu2 = np.array([1.5, 0.0])
    cov1 = np.array([[2.0, 0.6], [0.6, 0.5]])
    cov2 = np.array([[0.5, -0.6], [-0.6, 2.0]])

    xg = np.linspace(-5.5, 5.5, 400)
    yg = np.linspace(-4, 4, 400)
    X, Y = np.meshgrid(xg, yg)
    Z1 = _gauss2d(X, Y, mu1, cov1)
    Z2 = _gauss2d(X, Y, mu2, cov2)

    Z_blend = Z1 - Z2
    vmax = np.abs(Z_blend).max()
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "grp",
        [
            "#0d2b45",
            "#1a4e7a",
            "#3575a8",
            "#6baed6",
            "#b0d5ef",
            "#f0f0e8",
            "#faedcd",
            "#e8d5b5",
            "#d5bdaf",
            "#b09080",
            "#8b6f55",
        ],
    )
    n_lines = 10
    ax.contourf(X, Y, Z_blend, levels=n_lines, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.contour(
        X,
        Y,
        Z_blend,
        levels=n_lines,
        colors="k",
        linewidths=0.9,
        linestyles="solid",
    )

    gx = np.linspace(-4.2, 4.2, 7)
    gy = np.linspace(-2.8, 2.8, 5)
    GX, GY = np.meshgrid(gx, gy)
    gz1 = _gauss2d(GX, GY, mu1, cov1)
    gz2 = _gauss2d(GX, GY, mu2, cov2)

    ratio = np.minimum(gz1, gz2) / np.maximum(gz1, gz2).clip(1e-12)
    near_boundary = ratio > 0.5
    dist_to_mu1 = np.sqrt((GX - mu1[0]) ** 2 + (GY - mu1[1]) ** 2)
    dist_to_mu2 = np.sqrt((GX - mu2[0]) ** 2 + (GY - mu2[1]) ** 2)
    near_centroid = (dist_to_mu1 < 1.2) | (dist_to_mu2 < 1.2)
    label_a = (GX < -3.8) & (GY > 2.4)
    label_b = (GX > 3.3) & (GY < -2.4)
    skip = near_boundary | near_centroid | label_a | label_b

    arrow_len = 0.4
    for is_a, mu, color in [(True, mu1, "#8b6f55"), (False, mu2, "#1a4e7a")]:
        mask = ((gz1 > gz2) if is_a else (gz2 >= gz1)) & ~skip
        px, py = GX[mask], GY[mask]
        dx, dy = mu[0] - px, mu[1] - py
        norm = np.sqrt(dx**2 + dy**2)
        norm = np.maximum(norm, 1e-6)
        ax.quiver(
            px,
            py,
            dx / norm * arrow_len,
            dy / norm * arrow_len,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.006,
            headwidth=3.5,
            headlength=3,
            alpha=0.7,
            zorder=5,
        )

    ax.annotate(
        "",
        xy=mu2,
        xytext=mu1,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5, mutation_scale=18),
        zorder=6,
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-3.5, 3.5)

    ax.text(-4.5, 3.0, "A", fontsize=18, fontweight="bold", color="#8b6f55")
    ax.text(4.0, -3.0, "B", fontsize=18, fontweight="bold", color="#1a4e7a")

    ax.tick_params(labelbottom=False, labelleft=False, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def _save_fig(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        fig.savefig(output.with_suffix(ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_contour_figure(output: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    plot_gaussian_contour(ax)
    fig.tight_layout()
    _save_fig(fig, output)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_single_seed(
    seed: int,
    h: int,
    n_train: int,
    n_test: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[float, float, float, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    y_i_tr, y_j_tr, lab_tr = generate_training_pairs(n_train, generator=gen)
    y_i_across, y_j_across, lab_across = generate_across_cluster_pairs(n_test, generator=gen)
    y_i_within, y_j_within, lab_within = generate_within_cluster_pairs(n_test, generator=gen)

    pw = PointwiseReward(D, h)
    pair = PairwiseReward(D, h)

    train_pointwise(pw, y_i_tr, y_j_tr, lab_tr, epochs=epochs, lr=lr, batch_size=batch_size)
    train_pairwise(pair, y_i_tr, y_j_tr, lab_tr, epochs=epochs, lr=lr, batch_size=batch_size)

    pw_across, pair_across = eval_accuracy(pw, pair, y_i_across, y_j_across, lab_across)
    pw_within, pair_within = eval_accuracy(pw, pair, y_i_within, y_j_within, lab_within)

    return pw_across, pair_across, pw_within, pair_within


def main() -> None:
    args = parse_args()
    global SEPARATION, MU1, MU2
    SEPARATION = args.separation
    MU1 = -0.5 * SEPARATION * _dir
    MU2 = 0.5 * SEPARATION * _dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds):
        pw_a, pair_a, pw_w, pair_w = run_single_seed(
            seed=seed,
            h=args.h,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        print(
            f"seed {seed}: pw_across={pw_a:.3f} pair_across={pair_a:.3f} "
            f"pw_within={pw_w:.3f} pair_within={pair_w:.3f}"
        )

    plot_contour_figure(output_dir / "contour")
    print(f"Saved contour figure to {output_dir / 'contour'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic pointwise vs pairwise experiment")
    p.add_argument("--h", type=int, default=5, help="Bottleneck dimension")
    p.add_argument(
        "--separation", type=float, default=2.0, help="Distance between cluster centroids"
    )
    p.add_argument("--n-train", type=int, default=50_000, help="Training pairs")
    p.add_argument("--n-test", type=int, default=5_000, help="Test pairs per eval set")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size")
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    p.add_argument("--output-dir", type=str, default="figures/out/synthetic")
    return p.parse_args()


if __name__ == "__main__":
    main()
