"""Copy bandwidth across tensor sizes from 256 KB to 4 GB (doubling), for each dtype.

    python -m benchmarks.h100_bandwidth.bench_copy

Writes `results/copy_results.txt` (+ an uncommitted `_raw.csv`); plot with `plot_bandwidth`.
"""

from benchmarks.data.io import write_results
from benchmarks.environment import environment_metadata
from benchmarks.h100_bandwidth.config import RESULTS_DIR
from benchmarks.h100_bandwidth.data import copy_sweep

ITERATIONS, WARMUP_ITERATIONS = 100, 50
SIZES = [256_000 << i for i in range(15)]


if __name__ == "__main__":
    results = copy_sweep(SIZES, iterations=ITERATIONS, warmup_iterations=WARMUP_ITERATIONS)
    out_path = RESULTS_DIR / "copy_results.txt"
    write_results(
        out_path,
        results,
        benchmark="copy",
        iterations=ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS,
        **environment_metadata(),
    )
    print(f"Saved results to {out_path}")
