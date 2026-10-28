"""Plot achieved copy bandwidth from saved results (no GPU needed).

    python -m benchmarks.h100_bandwidth.plot_bandwidth [results.txt]
    python -m benchmarks.h100_bandwidth.plot_bandwidth [results.txt] --hist
        [--dtype float32] [--sizes 16000000 32000000]

`--hist` needs the `<results>_raw.csv` file written alongside the results.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.data.io import Result, Samples, read_raw, read_results
from benchmarks.h100_bandwidth.config import PLOTS_DIR, RESULTS_DIR, THEORETICAL_BANDWIDTH_GBPS
from benchmarks.plotting import format_bytes, plot_timings


def copy_bandwidth_gbps_at(size_bytes: int, time_ms: float) -> float:
    """A copy reads and writes every byte, so it moves 2x the tensor size."""
    return 2 * size_bytes / 1e9 / (time_ms / 1e3)


def copy_bandwidth_gbps(r: Result) -> tuple[float, float, float]:
    """(low, mid, high) bandwidth. A slower time (p90) is a lower bandwidth, hence the swap."""
    size, t = r.params["size_bytes"], r.timing
    return (
        copy_bandwidth_gbps_at(size, t.p90),
        copy_bandwidth_gbps_at(size, t.p50),
        copy_bandwidth_gbps_at(size, t.p10),
    )


def plot_bandwidth_histograms(
    samples: list[Samples],
    out_path: Path,
    *,
    title: str | None = None,
    bins: int = 40,
    ncols: int = 4,
) -> None:
    """One bandwidth histogram per measurement (tensor size), with its median marked."""
    samples = sorted(samples, key=lambda s: s.params["size_bytes"])
    nrows = math.ceil(len(samples) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax, s in zip(axes.flat, samples):
        size = s.params["size_bytes"]
        bandwidths = sorted(copy_bandwidth_gbps_at(size, t) for t in s.times_ms)
        ax.hist(bandwidths, bins=bins)
        ax.axvline(bandwidths[len(bandwidths) // 2], color="black", linestyle="--", linewidth=1)
        ax.set_title(format_bytes(size), fontsize=10)
        ax.set_xlabel("Bandwidth (GB/s)")
        ax.set_ylabel("Iterations")
    for ax in axes.flat[len(samples) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="?", default=RESULTS_DIR / "copy_results.txt")
    parser.add_argument("--hist", action="store_true", help="plot per-size histograms of the raw samples")
    parser.add_argument("--dtype", default="float32", help="dtype to plot histograms for")
    parser.add_argument("--sizes", type=int, nargs="+", help="size_bytes to plot histograms for (default: all)")
    args = parser.parse_args()

    metadata, results = read_results(args.results)
    title = f"Copy bandwidth, {metadata.get('device', '')}"

    if args.hist:
        samples = [
            s
            for s in read_raw(args.results)
            if s.params["dtype"] == args.dtype and (not args.sizes or s.params["size_bytes"] in args.sizes)
        ]
        if not samples:
            parser.error(f"no raw samples for dtype={args.dtype} sizes={args.sizes}")
        plot_bandwidth_histograms(
            samples,
            PLOTS_DIR / f"{args.results.stem}_hist_{args.dtype}.png",
            title=f"{title} ({args.dtype})",
        )
        return

    plot_timings(
        results,
        "size_bytes",
        PLOTS_DIR / "bandwidth.png",
        group_by="dtype",
        metric=copy_bandwidth_gbps,
        xlabel="Tensor size",
        ylabel="Bandwidth (GB/s)",
        title=title,
        hlines={"Theoretical limit": THEORETICAL_BANDWIDTH_GBPS},
        x_format=format_bytes,
    )


if __name__ == "__main__":
    main()
