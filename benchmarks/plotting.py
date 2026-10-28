"""Generic plotting for saved benchmark results (no GPU needed).

python -m benchmarks.plotting results.txt --x size_bytes --group-by dtype
"""

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.data.benchmark import BenchmarkTiming
from benchmarks.data.io import Result, read_results

# (low, mid, high) for one result; drawn as a median line with a shaded band.
Metric = Callable[[Result], tuple[float, float, float]]


def format_bytes(n: float) -> str:
    """Decimal units, matching the sizes in the data: 256000 -> '256 KB'."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.4g} {unit}"
    return f"{n:g} B"


def time_ms(r: Result) -> tuple[float, float, float]:
    t: BenchmarkTiming = r.timing
    return t.p10, t.p50, t.p90


def plot_timings(
    results: list[Result],
    x: str,
    out_path: Path,
    *,
    group_by: str | None = None,
    metric: Metric = time_ms,
    xlabel: str | None = None,
    ylabel: str = "Time (ms)",
    title: str | None = None,
    log_x: bool = True,
    log_y: bool = False,
    hlines: dict[str, float] | None = None,
    x_format: Callable[[float], str] | None = None,
) -> None:
    """One median line + shaded band per `group_by` value, `x` on the x-axis.

    `x_format` turns x values into tick text and puts a tick at every measured x
    (instead of matplotlib's decades), so exact sizes can be read off the axis.
    """
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        groups[str(r.params[group_by]) if group_by else "all"].append(r)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    all_xs: set[float] = set()
    for name, rs in groups.items():
        rs = sorted(rs, key=lambda r: r.params[x])
        xs = [r.params[x] for r in rs]
        lows, mids, highs = zip(*(metric(r) for r in rs))
        label = name if group_by else "P50"
        (line,) = ax.plot(xs, mids, marker="o", markersize=4, label=label)
        all_xs.update(xs)
        # min/max so the band is right for metrics that invert (e.g. time -> bandwidth)
        band_lo = [min(a, b) for a, b in zip(lows, highs)]
        band_hi = [max(a, b) for a, b in zip(lows, highs)]
        ax.fill_between(xs, band_lo, band_hi, color=line.get_color(), alpha=0.2, linewidth=0)

    for label, y in (hlines or {}).items():
        ax.axhline(y, color="black", linestyle="--", linewidth=1, label=label)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if x_format:
        ticks = sorted(all_xs)
        ax.set_xticks(ticks, [x_format(t) for t in ticks], rotation=45, ha="right")
        ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(title=group_by)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--x", required=True, help="param column for the x-axis")
    parser.add_argument("--group-by", help="param column to draw one line per value of")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    metadata, results = read_results(args.results)
    plot_timings(
        results,
        args.x,
        args.out or args.results.with_suffix(".png"),
        group_by=args.group_by,
        title=metadata.get("benchmark"),
        log_x=not args.linear_x,
        log_y=args.log_y,
    )
