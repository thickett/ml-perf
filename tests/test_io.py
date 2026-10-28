from benchmarks.data.benchmark import BenchmarkTiming
from benchmarks.data.io import Result, read_raw, read_results, write_results


def test_round_trip(tmp_path):
    path = tmp_path / "results.txt"
    results = [
        Result(
            {"size_bytes": 256_000, "dtype": "float32"}, BenchmarkTiming(1.5, 1.4, 1.2, 1.9, samples=[1.2, 1.4, 1.9])
        ),
        Result({"size_bytes": 512_000, "dtype": "bfloat16"}, BenchmarkTiming(2.5, 2.4, 2.2, 2.9, samples=[2.2, 2.9])),
    ]
    write_results(path, results, benchmark="copy", iterations=3)

    metadata, read_back = read_results(path)
    assert metadata == {"benchmark": "copy", "iterations": "3"}
    assert [r.params for r in read_back] == [r.params for r in results]
    assert [r.timing.p50 for r in read_back] == [1.4, 2.4]

    samples = read_raw(path)
    assert [s.times_ms for s in samples] == [[1.2, 1.4, 1.9], [2.2, 2.9]]
