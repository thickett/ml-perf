"""Zoom in on 2-62 MB, where the coarse sweep peaks near 16 MB with high variance and then dips.

    python -m benchmarks.h100_bandwidth.peak_analysis
    python -m benchmarks.h100_bandwidth.peak_analysis --plot-only   # redraw from saved results, no GPU

Writes `results/copy_small_data_sweep_results.txt` and `plots/bandwidth_small_data_sweep.png`, plus
per-dtype histograms of the raw samples across the 16-34 MB transition (raw CSV and histograms are
not committed).
"""

import argparse
from pathlib import Path

from benchmarks.data.io import read_raw, read_results, write_results
from benchmarks.environment import environment_metadata
from benchmarks.h100_bandwidth.config import (
    BYTES_PER_ELEMENT,
    PLOTS_DIR,
    RESULTS_DIR,
    THEORETICAL_BANDWIDTH_GBPS,
)
from benchmarks.h100_bandwidth.data import copy_sweep
from benchmarks.h100_bandwidth.plot_bandwidth import copy_bandwidth_gbps, plot_bandwidth_histograms
from benchmarks.plotting import format_bytes, plot_timings

ITERATIONS, WARMUP_ITERATIONS = 100, 50
SIZES = list(range(2_000_000, 64_000_000, 2_000_000))
# Sizes around the ~16 MB peak and the drop at ~32 MB, for the per-size histograms.
TRANSITION_RANGE_BYTES = (16_000_000, 34_000_000)
RESULTS_PATH = RESULTS_DIR / "copy_small_data_sweep_results.txt"


def plot(out_path: Path) -> None:
    metadata, results = read_results(out_path)
    plot_timings(
        results,
        "size_bytes",
        PLOTS_DIR / "bandwidth_small_data_sweep.png",
        group_by="dtype",
        metric=copy_bandwidth_gbps,
        xlabel="Tensor size",
        ylabel="Bandwidth (GB/s)",
        title=f"Copy bandwidth, {metadata.get('device', '')}",
        hlines={"Theoretical limit": THEORETICAL_BANDWIDTH_GBPS},
        x_format=format_bytes,
    )

    low, high = TRANSITION_RANGE_BYTES
    samples = read_raw(out_path)
    for precision in BYTES_PER_ELEMENT:
        dtype = str(precision).removeprefix("torch.")
        plot_bandwidth_histograms(
            [s for s in samples if s.params["dtype"] == dtype and low <= s.params["size_bytes"] <= high],
            PLOTS_DIR / f"bandwidth_small_data_sweep_hist_{dtype}.png",
            title=f"Copy bandwidth, {metadata.get('device', '')} ({dtype})",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="skip the sweep and redraw from saved results")
    args = parser.parse_args()

    if not args.plot_only:
        results = copy_sweep(SIZES, iterations=ITERATIONS, warmup_iterations=WARMUP_ITERATIONS)
        write_results(
            RESULTS_PATH,
            results,
            benchmark="copy",
            iterations=ITERATIONS,
            warmup_iterations=WARMUP_ITERATIONS,
            **environment_metadata(),
        )
        print(f"Saved results to {RESULTS_PATH}")
    plot(RESULTS_PATH)
