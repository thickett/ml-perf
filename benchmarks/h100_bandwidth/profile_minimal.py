"""Smallest program that reproduces a device copy of a given size, for Nsight Compute.

`dest.copy_(src)` lowers to cudaMemcpyAsync, which ncu cannot profile as a kernel, so this
uses an elementwise `mul(src, 1)` instead: a kernel with the same read+write traffic.

    python -m benchmarks.h100_bandwidth.profile_minimal --size-mb 24

Sizes are decimal MB, matching the benchmark sweeps. See `profile_l2.sh` for the ncu invocation.
"""

import argparse

import torch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--size-mb", type=float, default=24, help="tensor size in MB (1e6 bytes)")
parser.add_argument("--iterations", type=int, default=50)
args = parser.parse_args()

src = torch.randn(int(args.size_mb * 1e6) // 4, device="cuda", dtype=torch.float32)
dest = torch.empty_like(src)
for _ in range(args.iterations):
    torch.mul(src, 1, out=dest)
torch.cuda.synchronize()
