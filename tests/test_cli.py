import builtins

import pytest

from madwatch import cli


@pytest.fixture
def spike_csv(tmp_path):
    warmup = [str(v) for v in [10, 11, 10, 12, 11] * 4]
    rows = ["value"] + warmup + ["500"] + ["11"] * 5
    p = tmp_path / "data.csv"
    p.write_text("\n".join(rows))
    return p


@pytest.fixture
def seasonal_csv(tmp_path):
    lines = ["ts,value"]
    for day in range(1, 22):
        for hour in (9, 15):
            base = 100 if (day % 7) not in (6, 0) else 20
            lines.append(f"2026-02-{day:02d}T{hour:02d}:00:00,{base}")
    p = tmp_path / "seasonal.csv"
    p.write_text("\n".join(lines))
    return p


def test_detects_spike(spike_csv, capsys):
    rc = cli.main([str(spike_csv), "--column", "value", "--window", "15", "--min-samples", "5"])
    out = capsys.readouterr()
    assert rc == 0
    assert "500" in out.out
    assert "1 anomalies" in out.err


def test_no_anomalies_in_flat_data(tmp_path, capsys):
    p = tmp_path / "flat.csv"
    p.write_text("\n".join(["value"] + ["10"] * 30))
    rc = cli.main([str(p), "--column", "value"])
    out = capsys.readouterr()
    assert rc == 0
    assert "0 anomalies" in out.err


def test_nan_rows_skipped_with_warning(tmp_path, capsys):
    p = tmp_path / "gaps.csv"
    rows = ["idx,value"] + [f"{i},10" for i in range(15)] + ["15,"] + [f"{i},11" for i in range(16, 26)]
    p.write_text("\n".join(rows))
    rc = cli.main([str(p), "--column", "value"])
    out = capsys.readouterr()
    assert rc == 0
    assert "skipped 1 NaN" in out.err


def test_missing_column_errors(spike_csv, capsys):
    rc = cli.main([str(spike_csv), "--column", "nope"])
    assert rc == 2
    assert "column not found" in capsys.readouterr().err


def test_seasonal_requires_timestamp(spike_csv, capsys):
    rc = cli.main([str(spike_csv), "--column", "value", "--seasonal", "dow"])
    assert rc == 2
    assert "--timestamp" in capsys.readouterr().err


def test_seasonal_mode_runs(seasonal_csv, capsys):
    rc = cli.main([
        str(seasonal_csv), "--column", "value", "--timestamp", "ts", "--seasonal", "dow",
    ])
    out = capsys.readouterr()
    assert rc == 0
    assert "anomalies in" in out.err


def test_plot_writes_png(spike_csv, tmp_path):
    png = tmp_path / "out.png"
    rc = cli.main([
        str(spike_csv), "--column", "value", "--window", "15", "--min-samples", "5",
        "--plot", str(png),
    ])
    assert rc == 0
    assert png.stat().st_size > 0


def test_missing_extras_hint(monkeypatch, capsys):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = cli.main(["x.csv", "--column", "v"])
    assert rc == 2
    assert "madwatch[cli]" in capsys.readouterr().err
