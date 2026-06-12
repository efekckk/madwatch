# madwatch — Design Document

**Date:** 2026-06-12
**Status:** Approved (brainstorming session with Efe)

## Purpose

Open-source Python library showcasing Efecan Küçük's production anomaly-detection work:
robust anomaly detection built on **MAD (Median Absolute Deviation)** and the
**Modified Z-Score**, with seasonal baselines (day-of-week / hour bucketing) that cut
false positives on weekly-seasonal metrics. First showcase project for the
github.com/efekckk profile; published to PyPI as `madwatch`.

Cross-links: the portfolio site's `anomaly-detection-engine` project page and the
"Why MAD instead of standard deviation?" blog post reference the same math; README
links back to the blog post.

## Package Layout

```
madwatch/
├─ pyproject.toml
├─ README.md            (English)
├─ LICENSE              (MIT)
├─ src/madwatch/
│  ├─ __init__.py       public API re-exports + __version__ = "0.1.0"
│  ├─ core.py           mad(), modified_zscore()
│  ├─ rolling.py        RollingDetector, Score
│  ├─ seasonal.py       SeasonalBaseline
│  ├─ cli.py            console entry point `madwatch`
│  └─ plot.py           save_plot() (matplotlib behind import guard)
├─ tests/
│  ├─ test_core.py
│  ├─ test_rolling.py
│  ├─ test_seasonal.py
│  └─ test_cli.py
└─ .github/workflows/
   ├─ ci.yml            pytest matrix (3.10–3.13) + ruff
   └─ release.yml       tag v* → build → PyPI trusted publishing (OIDC)
```

## Public API

```python
from madwatch import mad, modified_zscore, RollingDetector, SeasonalBaseline

mad(x)                                   # ndarray -> float
modified_zscore(x, scale=0.6745)         # ndarray -> ndarray (z per element, vs whole-array median/MAD)

det = RollingDetector(window=40, threshold=3.5, min_samples=10)
det.update(value)                        # -> Score(value, z, is_anomaly); appends to window
det.score(values)                        # -> list[Score]; batch convenience over update()

sb = SeasonalBaseline(granularity='dow_hour')   # also 'dow', 'hour'
sb.fit(timestamps, values)               # per-bucket median + MAD; returns self
sb.score(timestamps, values)             # -> ndarray of z-scores normalized per bucket
```

`Score` is a frozen dataclass: `value: float, z: float, is_anomaly: bool`.
Timestamps are `datetime` objects (naive or aware; bucketing uses their local fields).

## Behavior Decisions

- **MAD = 0 (constant window/bucket):** z = 0, never an anomaly. No division by zero.
- **Cold start:** before `min_samples` values have been seen, RollingDetector returns
  `Score(value, z=0.0, is_anomaly=False)` and only accumulates the window.
- **Window semantics:** the incoming value is scored against the *previous* `window`
  values (baseline excludes the new point), then appended.
- **NaN input:** core functions and detector raise `ValueError`. CLI counts and skips
  NaN rows, printing a warning to stderr.
- **Unseen bucket at score time (SeasonalBaseline):** falls back to global median/MAD
  computed during fit.
- **Empty input:** `ValueError` everywhere.

## CLI

```
madwatch data.csv --column value [--timestamp ts] [--window 40] [--threshold 3.5]
                  [--seasonal dow_hour] [--plot out.png]
```

- Requires extras: `pip install madwatch[cli]` (pandas + matplotlib). Without them the
  entry point exits with a friendly install hint.
- Default mode: RollingDetector over the column; prints an aligned table of anomalies
  (index/timestamp, value, z) to stdout and a one-line summary to stderr.
- `--seasonal` switches to SeasonalBaseline scoring (requires `--timestamp`).
- `--plot` writes a PNG: the series with anomaly markers.
- Exit code 0 always (analysis tool, not a gate).

## Dependencies

- Core: `numpy>=1.24`, Python `>=3.10`
- Extras `cli`: `pandas>=2.0`, `matplotlib>=3.7`
- Dev: `pytest`, `ruff`

## Testing (TDD)

- core: known-value checks (hand-computed MAD/z), scale constant, NaN/empty raises
- rolling: the **whale test** — one huge outlier in the window does not shift the
  baseline enough to mask the next real anomaly (mirrors the blog post's claim);
  cold start; MAD=0 window; window-exclusion semantics
- seasonal: bucket math (dow_hour key correctness), unseen-bucket fallback,
  fit-then-score round trip on synthetic weekly-seasonal data
- cli: smoke test via `subprocess` on a temp CSV (anomaly row appears in output);
  missing-extras hint path tested with a mocked import failure

## CI / Release

- `ci.yml`: push + PR → ruff check, pytest on 3.10/3.11/3.12/3.13 (ubuntu)
- `release.yml`: tag `v*` → `python -m build` → `pypa/gh-action-pypi-publish` with
  **trusted publishing** (OIDC). One-time manual step for Efe: add the GitHub repo as
  a trusted publisher on PyPI for project `madwatch` (name verified available).
- Version source: `__version__` in `__init__.py`, mirrored in pyproject (static).

## README Outline (English)

1. Badges: CI, PyPI version, Python versions, license
2. Pitch: robust anomaly detection that doesn't panic at paydays — median-based,
   whale-proof
3. Quickstart (pip install, 6-line example)
4. Why MAD? short section + link to the blog post on efekckk.github.io
5. Seasonal baselines: the day-of-week / same-hour idea, before/after false-positive framing
6. CLI usage with example output and plot image (committed under `docs/assets/`)
7. API reference table, contributing, license

## Code Style

- Comments only where the code cannot express a constraint; otherwise none
- Short English docstrings on public API
- No Co-Authored-By lines in any commit
- ruff defaults, line length 100

## Out of Scope (v0.1.0)

- Other estimators (IQR, EWMA, isolation forest)
- Real-time sources (Kafka, websockets) — CSV/arrays only
- Plot customization beyond a single PNG
- Conda packaging
