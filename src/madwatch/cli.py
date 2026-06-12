import argparse
import sys


def main(argv=None) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("madwatch CLI requires extras: pip install 'madwatch[cli]'", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(
        prog="madwatch",
        description="MAD-based anomaly detection over a CSV column",
    )
    parser.add_argument("csv")
    parser.add_argument("--column", required=True)
    parser.add_argument("--timestamp")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--seasonal", choices=("dow_hour", "dow", "hour"))
    parser.add_argument("--plot")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        print(f"column not found: {args.column}", file=sys.stderr)
        return 2

    series = df[args.column]
    nan_count = int(series.isna().sum())
    if nan_count:
        print(f"warning: skipped {nan_count} NaN rows", file=sys.stderr)
    mask = series.notna()
    values = series[mask].to_numpy(dtype=float)
    labels = list(df.index[mask])

    if args.seasonal:
        if not args.timestamp:
            print("--seasonal requires --timestamp", file=sys.stderr)
            return 2
        if args.timestamp not in df.columns:
            print(f"column not found: {args.timestamp}", file=sys.stderr)
            return 2
        from .seasonal import SeasonalBaseline

        timestamps = list(pd.to_datetime(df.loc[mask, args.timestamp]).dt.to_pydatetime())
        labels = [t.isoformat() for t in timestamps]
        z = SeasonalBaseline(args.seasonal).fit(timestamps, values).score(timestamps, values)
        flags = [abs(s) > args.threshold for s in z]
        zs = list(z)
    else:
        from .rolling import RollingDetector

        det = RollingDetector(
            window=args.window, threshold=args.threshold, min_samples=args.min_samples
        )
        results = det.score(values)
        flags = [r.is_anomaly for r in results]
        zs = [r.z for r in results]

    anomalies = [
        (labels[i], values[i], zs[i]) for i, flagged in enumerate(flags) if flagged
    ]
    if anomalies:
        print(f"{'where':<22}{'value':>12}{'z':>10}")
        for where, value, z in anomalies:
            print(f"{str(where):<22}{value:>12.2f}{z:>10.2f}")
    print(f"{len(anomalies)} anomalies in {len(values)} points", file=sys.stderr)

    if args.plot:
        from .plot import save_plot

        save_plot(values, flags, args.plot)
        print(f"plot saved: {args.plot}", file=sys.stderr)
    return 0
