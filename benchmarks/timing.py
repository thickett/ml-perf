"""CUDA-event timing shared by all benchmarks."""

from collections.abc import Callable
from typing import Any

import torch

from benchmarks.data.benchmark import BenchmarkTiming


def time_cuda(
    func: Callable[..., Any],
    *args: Any,
    iterations: int = 100,
    warmup_iterations: int = 50,
    **kwargs: Any,
) -> BenchmarkTiming:
    """Time `func(*args, **kwargs)` with CUDA events. Values are in milliseconds."""
    for _ in range(warmup_iterations):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(start_events, end_events):
        start.record()
        func(*args, **kwargs)
        end.record()
    torch.cuda.synchronize()

    samples = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    elapsed = sorted(samples)
    n = len(elapsed)
    return BenchmarkTiming(
        mean=sum(elapsed) / n,
        p10=elapsed[int(0.1 * n)],
        p50=elapsed[int(0.5 * n)],
        p90=elapsed[int(0.9 * n)],
        samples=samples,
    )
