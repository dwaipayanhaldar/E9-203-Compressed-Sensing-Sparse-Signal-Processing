"""
example.py – synthetic signal demo for l1-K-SVD
================================================
Replicates the synthetic experiment from the paper (Section 3.1):

  * Ground-truth dictionary D of K=50 atoms in R^20
  * N=1500 training signals, each a sparse linear combination of s=3 atoms
  * Additive Laplacian noise at SNR = 20 dB
  * l1-K-SVD is fit and the reconstruction error is visualised

Usage
-----
    python example.py          # run from inside the Pyl1KSVD/ directory
"""

import numpy as np
import matplotlib.pyplot as plt
from pyl1ksvd.pyl1ksvd import L1KSVD

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── experiment parameters (matching the paper's synthetic setup) ──────────────
N       = 1500   # number of training signals
n       = 20     # signal dimension  (m in the paper)
K       = 50     # dictionary atoms
s       = 3      # sparsity (non-zeros per signal)
SNR_dB  = 20     # additive noise SNR

# ── ground-truth dictionary: K random normalised atoms in R^n ─────────────────
D_true = np.random.randn(n, K)
D_true /= np.linalg.norm(D_true, axis=0)

# ── generate training signals ─────────────────────────────────────────────────
#   y_n = D_true x_n + w_n
#   x_n: s non-zeros at random positions, amplitudes ~ N(0,1)
#   w_n: Laplacian noise scaled to achieve the target SNR

X_true = np.zeros((K, N))
for i in range(N):
    idx = np.random.choice(K, s, replace=False)
    X_true[idx, i] = np.random.randn(s)

Y_clean = D_true @ X_true                    # (n, N)  clean signals

# Laplacian noise: draw from Laplace(0, b), scale to match SNR
signal_power = np.mean(np.linalg.norm(Y_clean, axis=0) ** 2)
noise_std    = np.sqrt(signal_power / (10 ** (SNR_dB / 10)))
# Laplace samples: mean=0, scale = std/sqrt(2)
W = np.random.laplace(0, noise_std / np.sqrt(2), size=Y_clean.shape)

Y = Y_clean + W                              # noisy training signals

# ── fit l1-K-SVD ─────────────────────────────────────────────────────────────
model = L1KSVD(
    K=K,
    lam=1.0,            # lambda (paper default for Laplacian noise)
    max_iter=80,
    n_irls_sparse=30,
    n_irls_dict=10,
    threshold_frac=0.03,
)
model.fit(Y, verbose=True)

X_hat = model.X
D_hat = model.D

# ── reconstruction quality ────────────────────────────────────────────────────
Y_reconstructed = D_hat @ X_hat

MAE = np.linalg.norm(Y - Y_reconstructed, ord=1) / np.linalg.norm(Y, ord=1)
MSE = np.linalg.norm(Y - Y_reconstructed, 'fro') ** 2 / np.linalg.norm(Y, 'fro') ** 2
print(f"\nMAE (relative l1): {MAE:.4f}")
print(f"MSE (relative F ): {MSE:.4f}")

# ── atom detection rate (ADR) ─────────────────────────────────────────────────
#   An atom d_hat_j is considered recovered if max_i |d_true_i^T d_hat_j| > 0.99
recovered = 0
for j in range(K):
    inner_products = np.abs(D_true.T @ D_hat[:, j])
    if np.max(inner_products) > 0.99:
        recovered += 1
ADR = recovered / K
print(f"Atom Detection Rate (ADR): {ADR:.2%}")

# ── visualise a few signals and their reconstructions ─────────────────────────
color_palette   = plt.get_cmap("plasma")
N_DISPLAY       = min(6, N)
colors          = [color_palette(i / N_DISPLAY) for i in range(N_DISPLAY)]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for i in range(N_DISPLAY):
    ax = axes[i]
    ax.plot(Y[:, i],               color=colors[i], label="Noisy signal",
            alpha=0.6, linewidth=1.5)
    ax.plot(Y_clean[:, i],         color=colors[i], label="Clean signal",
            linestyle="--", alpha=0.8, linewidth=1.5)
    ax.plot(Y_reconstructed[:, i], color="black",   label="l1-K-SVD recon",
            linestyle=":", linewidth=2)
    ax.set_title(f"Signal {i}")
    if i == 0:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"l1-K-SVD reconstruction  |  N={N}, n={n}, K={K}, s={s}, SNR={SNR_dB} dB\n"
    f"ADR={ADR:.0%}  MAE={MAE:.4f}  MSE={MSE:.4f}",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("./images/l1ksvd_reconstruction.png", dpi=150)
print("\nPlot saved to images/l1ksvd_reconstruction.png")
plt.show()
