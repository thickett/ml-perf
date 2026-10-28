from dataclasses import dataclass, field


@dataclass
class BenchmarkTiming:
    """Summary statistics in milliseconds."""

    mean: float
    p50: float
    p10: float
    p90: float
    samples: list[float] = field(default_factory=list)  # raw per-iteration times, in run order
