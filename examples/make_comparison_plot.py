import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from madwatch import modified_zscore

rng = np.random.default_rng(3)
n = 240
values = 100 + rng.normal(0, 2.0, n)
spikes = {60: 90, 75: 26, 85: 24, 95: 22}
for i, boost in spikes.items():
    values[i] += boost

window = 40
mad_flags = np.zeros(n, dtype=bool)
std_flags = np.zeros(n, dtype=bool)
for i in range(window, n):
    w = values[i - window : i + 1]
    mad_flags[i] = abs(modified_zscore(w)[-1]) > 3.5
    sigma = w.std()
    std_flags[i] = sigma > 0 and abs((w[-1] - w.mean()) / sigma) > 3.5

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, flags, title in (
    (axes[0], std_flags, "classic z-score — the first spike inflates σ, later anomalies slip through"),
    (axes[1], mad_flags, "madwatch (MAD) — the baseline barely moves, every anomaly flagged"),
):
    ax.plot(values, lw=1, color="#4a6fa5")
    idx = np.flatnonzero(flags)
    ax.scatter(idx, values[idx], color="#d64545", zorder=3, s=28)
    ax.set_title(f"{title}  ·  {len(idx)}/{len(spikes)} caught", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("docs/assets/mad_vs_std.png", dpi=160)
print(f"std caught: {std_flags.sum()}, mad caught: {mad_flags.sum()}")
