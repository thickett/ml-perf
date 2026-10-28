"""Turn the raw ncu CSVs written by `profile_l2.sh` into one small committed table (no GPU needed).

    python -m benchmarks.h100_bandwidth.summarise_l2_profile [.logs]

Averages each metric over the profiled launches. The raw CSVs stay out of the repo because
they include host details; this keeps only the numbers.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from benchmarks.h100_bandwidth.config import RESULTS_DIR

OUT_PATH = RESULTS_DIR / "l2_profile.txt"


def read_ncu_csv(path: Path) -> tuple[dict[str, float], int]:
    """Mean of each metric over the launches in one ncu `--csv` file, and the launch count."""
    lines = [line for line in path.read_text().splitlines() if line.startswith('"')]
    values: dict[str, list[float]] = defaultdict(list)
    launches = set()
    for row in csv.DictReader(lines):
        values[row["Metric Name"]].append(float(row["Metric Value"].replace(",", "")))
        launches.add(row["ID"])
    return {name: mean(vs) for name, vs in values.items()}, len(launches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", type=Path, nargs="?", default=Path(".logs"))
    args = parser.parse_args()

    paths = sorted(args.log_dir.glob("profile_l2_*mb.csv"), key=lambda p: float(re.findall(r"[\d.]+", p.stem)[-1]))
    if not paths:
        parser.error(f"no profile_l2_*mb.csv files in {args.log_dir}")

    rows = []
    for path in paths:
        size_mb = float(re.findall(r"[\d.]+", path.stem)[-1])
        size = size_mb * 1e6
        m, launches = read_ncu_csv(path)
        seconds = m["gpu__time_duration.sum"] / 1e9
        rows.append(
            {
                "size_mb": f"{size_mb:g}",
                "dram_read_frac": f"{m['dram__bytes_read.sum'] / size:.3f}",
                "dram_write_frac": f"{m['dram__bytes_write.sum'] / size:.3f}",
                "l2_read_hit_pct": f"{m['lts__t_sector_op_read_hit_rate.pct']:.1f}",
                "l2_write_hit_pct": f"{m['lts__t_sector_op_write_hit_rate.pct']:.1f}",
                "kernel_us": f"{seconds * 1e6:.2f}",
                "effective_tbps": f"{2 * size / seconds / 1e12:.2f}",
            }
        )

    with open(OUT_PATH, "w", newline="") as f:
        f.write("# benchmark: l2_profile (ncu, mul(src, 1) on float32, see profile_l2.sh)\n")
        f.write(f"# launches_per_size: {launches}\n")
        f.write("# dram_*_frac: DRAM bytes per kernel / tensor size; effective_tbps: 2 * size / kernel time\n")
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_PATH.read_text())


if __name__ == "__main__":
    main()
