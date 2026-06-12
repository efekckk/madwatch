import numpy as np

from madwatch import RollingDetector
from madwatch.plot import save_plot

rng = np.random.default_rng(7)
values = 100 + 6 * np.sin(np.linspace(0, 2 * np.pi, 300)) + rng.normal(0, 1.5, 300)
values[90] += 60
values[180] -= 45
values[240] += 80

det = RollingDetector(window=40, threshold=3.5, min_samples=10)
flags = [s.is_anomaly for s in det.score(values)]
save_plot(values, flags, "docs/assets/demo.png")
print(f"anomalies: {sum(flags)}")
