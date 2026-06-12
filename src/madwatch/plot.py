def save_plot(values, flags, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    v = np.asarray(values, dtype=float)
    f = np.asarray(flags, dtype=bool)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(v, lw=1, color="#39d353")
    idx = np.flatnonzero(f)
    ax.scatter(idx, v[idx], color="#ffa657", s=28, zorder=3, label="anomaly")
    if idx.size:
        ax.legend(loc="upper left")
    ax.set_title("madwatch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
