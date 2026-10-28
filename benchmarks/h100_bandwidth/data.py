import torch

from benchmarks.data.io import Result
from benchmarks.h100_bandwidth.config import BYTES_PER_ELEMENT, DEVICE
from benchmarks.timing import time_cuda


def make_generator(seed: int | None) -> torch.Generator:
    g = torch.Generator(device=DEVICE)
    if seed is not None:
        g.manual_seed(seed)
    return g


def create_data_by_size(n_bytes: int, precision: torch.dtype, seed: int | None = None) -> torch.Tensor:
    assert precision in BYTES_PER_ELEMENT, (
        f"Unsupported precision {precision}. Allowed values: {list(BYTES_PER_ELEMENT)}"
    )
    size = n_bytes // BYTES_PER_ELEMENT[precision]
    if size <= 0:
        raise ValueError(f"Size must be positive, but got {size}. pass a larger byte value")
    return torch.rand(size, device=DEVICE, dtype=precision, generator=make_generator(seed))


def copy_data(tensor: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    """Same-dtype contiguous device copy; PyTorch dispatches this to cudaMemcpyAsync (device to device)."""
    dest.copy_(tensor)
    return dest


def copy_sweep(sizes: list[int], *, iterations: int, warmup_iterations: int) -> list[Result]:
    """Time `copy_data` at every size, for every dtype in `BYTES_PER_ELEMENT`."""
    results: list[Result] = []
    for precision in BYTES_PER_ELEMENT:
        for size in sizes:
            tensor = create_data_by_size(size, precision)
            dest = torch.empty_like(tensor)
            timing = time_cuda(copy_data, tensor, dest, iterations=iterations, warmup_iterations=warmup_iterations)
            results.append(Result({"size_bytes": size, "dtype": str(precision).removeprefix("torch.")}, timing))
            del tensor, dest
    return results
