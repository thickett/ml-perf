#!/usr/bin/env bash
# Does the copy bandwidth bump near 16 MB come from L2 residency between iterations?
#
# Profiles a copy-shaped kernel at sizes inside the bump, in the dip, and well past it, with
# ncu's cache flushing and clock locking turned off, so the steady state matches the benchmark.
# If the data stays in L2, DRAM bytes per kernel fall well below the tensor size.
#
#   bash benchmarks/h100_bandwidth/profile_l2.sh        (from the repo root, on a machine with an H100)
#
# Writes one CSV per size to .logs/ (raw ncu output includes host details; summarise before committing).
set -euo pipefail

METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,gpu__time_duration.sum

mkdir -p .logs
for size_mb in 16 18 20 22 24 26 256; do
    # -s 10 skips the randn kernel and warm-up launches; -c 5 profiles five steady-state launches.
    ncu --cache-control none --clock-control none -s 10 -c 5 \
        --metrics "$METRICS" --csv \
        uv run python -m benchmarks.h100_bandwidth.profile_minimal --size-mb "$size_mb" \
        > ".logs/profile_l2_${size_mb}mb.csv"
    echo "done: ${size_mb} MB"
done
