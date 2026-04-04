"""Run the aliased paired-feature benchmark."""

from __future__ import annotations

from temporal_bench.benchmarks import AliasedBenchmarkConfig, run_aliased_benchmark


def main() -> None:
    run_aliased_benchmark(AliasedBenchmarkConfig())


if __name__ == "__main__":
    main()
