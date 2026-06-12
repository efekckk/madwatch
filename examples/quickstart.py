from madwatch import RollingDetector

readings = [102, 99, 101, 103, 100, 98, 102, 101, 99, 100, 104, 100, 740, 101, 99]

det = RollingDetector(window=10, threshold=3.5, min_samples=5)
for i, score in enumerate(det.score(readings)):
    if score.is_anomaly:
        print(f"index {i}: value={score.value:.0f} z={score.z:.1f}")
