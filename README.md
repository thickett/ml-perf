# ml-perf

Small GPU performance benchmarks. Each one comes with its code, its raw results and plots, and a write-up of what it found.

| Benchmark | Question | Headline |
| --- | --- | --- |
| [H100 memory bandwidth](benchmarks/h100_bandwidth/) | How close does a device copy get to the 3.35 TB/s HBM3 spec? | 3.03 TB/s (~90%) at ≥1 GB; launch-bound below ~8 MB |

## Setup

Uses [uv](https://docs.astral.sh/uv/). Python 3.13; `uv.lock` pins every dependency, including `torch 2.14.0+cu130` (CUDA 13.0).

```bash
uv sync
uv run pytest      # CPU-only tests for the shared harness
```

Each benchmark's README has the exact commands that reproduce its results.

## Layout

```
benchmarks/
  timing.py         CUDA-event timer (warmup, then each iteration timed separately)
  environment.py    torch/CUDA/driver/git versions, written into every results header
  data/             result types and the results-file format (see data/io.py)
  plotting.py       generic median + p10-p90 band plots from saved results
  <benchmark>/      one directory per benchmark: code and README at the top,
                    measured data in results/, figures drawn from it in plots/
tests/
```

Results files are CSV with `# key: value` metadata lines at the top. Read them with `pandas.read_csv(path, comment="#")`. Per-iteration samples are also written next to them as `<name>_raw.csv`. These are not committed: re-run the benchmark to get them.

## License

[MIT](LICENSE)
