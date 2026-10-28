"""Write/read benchmark results as a plain-text file.

Format: `# key: value` metadata lines (run-wide settings), then a CSV table with
one row per measurement. Anything that varies between rows (tensor size, dtype,
batch size, ...) is a column; anything constant across the run is metadata.

    # benchmark: copy
    # iterations: 100
    size_bytes,dtype,mean_ms,p50_ms,p10_ms,p90_ms
    256000,float32,0.0123,0.0121,0.0119,0.0130

`pandas.read_csv(path, comment="#")` reads the table directly.

If the timings carry raw samples, a long-format sibling `<stem>_raw.csv` is also
written, one row per iteration (for histograms etc.):

    size_bytes,dtype,iteration,time_ms
    256000,float32,0,0.0121
"""

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from benchmarks.data.benchmark import BenchmarkTiming

_TIME_FIELDS = [f.name for f in fields(BenchmarkTiming) if f.name != "samples"]


def raw_path(path: Path) -> Path:
    return Path(path).with_name(f"{Path(path).stem}_raw.csv")


@dataclass
class Result:
    params: dict[str, Any]  # what varied: {"size_bytes": 256000, "dtype": "float32"}
    timing: BenchmarkTiming


@dataclass
class Samples:
    params: dict[str, Any]
    times_ms: list[float]  # one per iteration, in run order


def write_results(path: Path, results: list[Result], **metadata: Any) -> None:
    """Write one row per result. Extra kwargs become header metadata."""
    assert results, "no results to write"
    param_names = list(results[0].params)
    for r in results:
        assert list(r.params) == param_names, "all results must have the same params"
    clash = set(param_names) & {f"{n}_ms" for n in _TIME_FIELDS}
    assert not clash, f"param names clash with timing columns: {clash}"

    with open(path, "w", newline="") as f:
        for k, v in metadata.items():
            f.write(f"# {k}: {v}\n")
        writer = csv.writer(f)
        writer.writerow([*param_names, *(f"{n}_ms" for n in _TIME_FIELDS)])
        for r in results:
            writer.writerow([*r.params.values(), *(getattr(r.timing, n) for n in _TIME_FIELDS)])

    if any(r.timing.samples for r in results):
        with open(raw_path(path), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([*param_names, "iteration", "time_ms"])
            for r in results:
                for i, t in enumerate(r.timing.samples):
                    writer.writerow([*r.params.values(), i, t])


def _parse(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def read_results(path: Path) -> tuple[dict[str, str], list[Result]]:
    """Inverse of `write_results`. Metadata comes back as strings; params as int/float/str."""
    metadata: dict[str, str] = {}
    table_lines: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            metadata[key.strip()] = value.strip()
        elif line.strip():
            table_lines.append(line)

    results = []
    for row in csv.DictReader(table_lines):
        timing = BenchmarkTiming(**{n: float(row.pop(f"{n}_ms")) for n in _TIME_FIELDS})
        results.append(Result({k: _parse(v) for k, v in row.items()}, timing))
    return metadata, results


def read_raw(path: Path) -> list[Samples]:
    """Read the `_raw.csv` written next to `path` by `write_results`, grouped per measurement."""
    grouped: dict[tuple[tuple[str, Any], ...], list[float]] = {}
    with open(raw_path(path), newline="") as f:
        for row in csv.DictReader(f):
            time_ms = float(row.pop("time_ms"))
            del row["iteration"]
            key = tuple((k, _parse(v)) for k, v in row.items())
            grouped.setdefault(key, []).append(time_ms)
    return [Samples(dict(key), times) for key, times in grouped.items()]
