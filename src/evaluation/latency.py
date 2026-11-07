"""Latency and throughput measurement."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np


def measure_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 20,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    timings = np.asarray(timings)
    return {
        "n": int(len(timings)),
        "mean_ms": float(timings.mean()),
        "p50_ms": float(np.quantile(timings, 0.50)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "p99_ms": float(np.quantile(timings, 0.99)),
        "throughput_per_s": float(1000.0 / timings.mean()) if timings.mean() > 0 else float("inf"),
    }
