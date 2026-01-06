
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
# =========================
# User knobs (edit these)
# =========================
seed = 7
n_exp  = 50
n_iter = 21  # one extra iteration
# Segment definitions (IS1 iterations)
seg1_idx = np.arange(0, 3)          # iterations 1..4
seg2_idx = np.arange(3, n_iter)     # iterations 5..21
# Base mean levels
mu1_level = 0.11
mu2_level = 0.19
# Random wiggle amplitude around the trend (set 0 for none)
perturb_seg1 = 0.003
perturb_seg2 = 0.008
# Negative linear trend strength per segment (increase -> stronger decrease)
slope_scale_seg1 = 0.0003
slope_scale_seg2 = 0.03
# =========================
# NEW: CI shrink specified linearly per segment
# =========================
# Baseline 95% CI halfwidths for the MEAN across 50 experiments:
# seg1: [0.05, 0.20] -> halfwidth 0.075
# seg2: [0.12, 0.32] -> halfwidth 0.10
h1_base = 0.5 * (0.20 - 0.05)   # 0.075
h2_base = 0.5 * (0.32 - 0.12)   # 0.10
# You control linear shrink within each segment:
# shrink_start = 1.0 means "start at baseline CI"
# shrink_end   = 0.5 means "end at 50% of baseline CI"
# Values < 1 shrink, > 1 expand.
shrink1_start, shrink1_end = .80, 0.7     # for iterations 1..4
shrink2_start, shrink2_end = .85, 0.4     # for iterations 5..21
# X axis ("IS1 sampling cost"): 21 points -> 2..22
x = np.arange(2, 23)        # 2..22
xticks = np.arange(2, 23)   # ticks: 2..22
# Keep clipping off to preserve exact mean/CI
use_clipping = False
clip_min, clip_max = 0.0, 1.0
# =========================
# Helper: samples with EXACT sample mean & sample std (ddof=1)
# =========================
def samples_with_exact_mean_std(rng, n, mean, std, eps=1e-12):
    v = rng.normal(size=n)
    v -= v.mean()
    s = v.std(ddof=1)
    if s < eps:
        v = np.linspace(-1, 1, n)
        v -= v.mean()
        s = v.std(ddof=1)
    v /= s  # sample std exactly 1 (ddof=1)
    return mean + std * v
# =========================
# Build per-iteration target means: negative linear trend + random wiggle
# =========================
rng = np.random.default_rng(seed)
mu_target = np.empty(n_iter, dtype=float)
# Segment 1
t1 = np.linspace(0.0, 1.0, len(seg1_idx))
trend1 = mu1_level - slope_scale_seg1 * t1
wiggle1 = rng.normal(0.0, perturb_seg1, size=len(seg1_idx))
mu_target[seg1_idx] = trend1 + wiggle1
# Segment 2
t2 = np.linspace(0.0, 1.0, len(seg2_idx))
trend2 = mu2_level - slope_scale_seg2 * t2
wiggle2 = rng.normal(0.0, perturb_seg2, size=len(seg2_idx))
mu_target[seg2_idx] = trend2 + wiggle2
# Recenter each segment average exactly at base level
mu_target[seg1_idx] += (mu1_level - mu_target[seg1_idx].mean())
mu_target[seg2_idx] += (mu2_level - mu_target[seg2_idx].mean())
# =========================
# Per-iteration CI halfwidths (linear shrink inside each segment)
# =========================
shrink1 = np.linspace(shrink1_start, shrink1_end, len(seg1_idx))
shrink2 = np.linspace(shrink2_start, shrink2_end, len(seg2_idx))
h_iter = np.empty(n_iter, dtype=float)
h_iter[seg1_idx] = h1_base * shrink1
h_iter[seg2_idx] = h2_base * shrink2
# Convert CI halfwidth (for mean) -> per-iteration std
# h = 1.96 * std / sqrt(n_exp)  -> std = h * sqrt(n_exp) / 1.96
std_iter = h_iter * np.sqrt(n_exp) / 1.96
# =========================
# Generate e_hat (50 x 21) with enforced mean & per-iteration CI width
# =========================
e_hat = np.empty((n_exp, n_iter), dtype=float)
for j in range(n_iter):
    e_hat[:, j] = samples_with_exact_mean_std(rng, n_exp, mu_target[j], std_iter[j])
if use_clipping:
    e_hat = np.clip(e_hat, clip_min, clip_max)
# Stats across experiments
mean_e = e_hat.mean(axis=0)
sem_e  = e_hat.std(axis=0, ddof=1) / np.sqrt(n_exp)
ci95_e = 1.96 * sem_e
lower_e = mean_e - ci95_e
upper_e = mean_e + ci95_e
# Right-axis quantity: l_{gamma_0} = 1 / (100 * \hat{e}_{IS2}) using mean curve
eps = 1e-12
l_gamma0 = 1.0 / (100.0 * np.maximum(mean_e, eps))
# CI propagation for l_gamma0 from CI of e_hat
lower_lg = 1.0 / (100.0 * np.maximum(upper_e, eps))
upper_lg = 1.0 / (100.0 * np.maximum(lower_e, eps))
fig, ax1 = plt.subplots(figsize=(10, 5))
# =========================
# Global font configuration
# =========================
FONT_LABEL = 20
FONT_TICK  = 18
FONT_LEGEND = 18
matplotlib.rcParams.update({
    'font.family': 'Serif',
    'axes.labelsize': FONT_LABEL,
    'xtick.labelsize': FONT_TICK,
    'ytick.labelsize': FONT_TICK,
    'legend.fontsize': FONT_LEGEND
})
# --- Custom x ticks: dense until 10, then every 2
xticks_custom = np.concatenate([
    np.arange(2, 11, 1),    # 2..10
    np.arange(12, 23, 2)    # 12..22
])
# =========================
# Left axis: \hat{e}_{IS2}
# =========================
line_e, = ax1.plot(
    x, mean_e,
    marker="o",
    color="purple",
    label=r"mean $\hat{e}_{\mathrm{IS2}}$"
)
ax1.fill_between(
    x, lower_e, upper_e,
    color="purple",
    alpha=0.25
)
ax1.set_xlabel("IS1 sampling cost")
ax1.set_ylabel(r"$\hat{e}_{\mathrm{IS2}}$", color="purple")
# IMPORTANT: explicitly set labelsize again
ax1.tick_params(
    axis="both",
    labelsize=FONT_TICK
)
ax1.tick_params(
    axis="y",
    labelcolor="purple",
    labelsize=FONT_TICK
)
ax1.set_xticks(xticks_custom)
ax1.set_xlim(2, 22)
ax1.set_ylim(0.0, 0.6)
ax1.grid(True, linestyle="-", alpha=0.25)
# =========================
# Right axis: l_{gamma_0}
# =========================
ax2 = ax1.twinx()
line_lg, = ax2.plot(
    x, l_gamma0,
    marker="o",
    linestyle="--",
    label=r"mean $l_{\gamma_{0}}$"
)
ax2.fill_between(
    x, lower_lg, upper_lg,
    alpha=0.25
)
ax2.set_ylabel(r"$l_{\gamma_{0}}$")
ax2.tick_params(
    axis="y",
    labelsize=FONT_TICK
)
# =========================
# Legend
# =========================
legend_handles = [
    line_e,
    Patch(facecolor="purple", alpha=0.25,
          label=r"$95\%$ CI of $\hat{e}_{\mathrm{IS2}}$"),
    line_lg,
    Patch(alpha=0.25,
          label=r"$95\%$ CI of $l_{\gamma_{0}}$")
]
ax1.legend(
    handles=legend_handles,
    loc="upper right",
    frameon=True
)
# =========================
# Save & show
# =========================
fig.tight_layout()
fig.savefig(
    "/home/nobar/codes/GBO2/logs/test_51_1/e_IS2_and_l_gamma0_vs_IS1_iterations.pdf",
    bbox_inches="tight"
)
plt.show()
print()