"""CUDA memory profiling utilities.

Provides snapshot-based memory tracking with a context manager interface.
All memory values are in MB unless otherwise noted.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Generator

import torch


@dataclass
class MemorySnapshot:
    """Point-in-time CUDA memory snapshot."""

    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def take_snapshot() -> MemorySnapshot:
    """Read current CUDA memory stats."""
    return MemorySnapshot(
        allocated_mb=torch.cuda.memory_allocated() / (1024**2),
        reserved_mb=torch.cuda.memory_reserved() / (1024**2),
        max_allocated_mb=torch.cuda.max_memory_allocated() / (1024**2),
        max_reserved_mb=torch.cuda.max_memory_reserved() / (1024**2),
    )


def reset_memory_tracking() -> None:
    """Clear CUDA caches and reset peak memory counters."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@contextmanager
def track_memory(label: str) -> Generator[dict[str, Any], None, None]:
    """Context manager that captures before/after CUDA memory snapshots.

    Usage:
        with track_memory("prefill") as mem:
            # ... do work ...
        print(mem["peak_allocated_mb"])

    The yielded dict is populated on exit with:
        label, before, after, peak_allocated_mb, peak_reserved_mb
    """
    reset_memory_tracking()
    before = take_snapshot()
    result: dict[str, Any] = {"label": label}
    yield result
    torch.cuda.synchronize()
    after = take_snapshot()
    result["before"] = before.to_dict()
    result["after"] = after.to_dict()
    result["peak_allocated_mb"] = after.max_allocated_mb
    result["peak_reserved_mb"] = after.max_reserved_mb


def get_gpu_info() -> dict[str, Any]:
    """Return basic GPU information."""
    if not torch.cuda.is_available():
        return {"name": "N/A", "total_mb": 0.0, "compute_capability": (0, 0)}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_mb": props.total_memory / (1024**2),
        "compute_capability": (props.major, props.minor),
    }
