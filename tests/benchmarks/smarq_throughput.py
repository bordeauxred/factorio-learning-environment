#!/usr/bin/env python3
"""SM-ARQ collection/learning throughput probe for fake or live environments."""

from __future__ import annotations

import argparse
import os
import resource
import tempfile
import time
from pathlib import Path

from fle.smarq.train import TrainingConfig, run_training


def _rss_mib() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except ImportError:
        # macOS reports bytes; Linux reports KiB.
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024**2) if value > 10_000_000 else value / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("fake", "live"), default="fake")
    parser.add_argument("--ports", default="27000,27001,27002,27003")
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--replay-ratio", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--raster-tiles", type=int, default=16)
    args = parser.parse_args()
    ports = tuple(int(port) for port in args.ports.split(",") if port)
    # Exclude one-time framework import from the worker-scaling measurements.
    import torch

    _ = torch.backends.mps.is_available()
    print(
        "workers  actions/s  updates/s  game-sec/wall-sec  replay-MiB  CPU%  RAM-MiB"
    )
    with tempfile.TemporaryDirectory(prefix="smarq-throughput-") as temporary:
        for workers in (1, 2, 4):
            cpu_before = time.process_time()
            wall_before = time.perf_counter()
            result = run_training(
                TrainingConfig(
                    run_name=f"w{workers}",
                    env=args.env,
                    ports=ports,
                    workers=workers,
                    replay_ratio=args.replay_ratio,
                    raster_tiles=args.raster_tiles,
                    total_decisions=args.decisions,
                    checkpoint_every=0,
                    run_root=Path(temporary),
                    replay_capacity=min(args.decisions, 128),
                    seed=17,
                )
            )
            wall = max(time.perf_counter() - wall_before, 1e-12)
            cpu = time.process_time() - cpu_before
            rss_after = _rss_mib()
            replay_mib = result.replay_memory_bytes / (1024**2)
            print(
                f"{workers:7d}  {result.decisions / wall:9.2f}  "
                f"{result.updates / wall:9.2f}  {result.simulated_seconds / wall:18.2f}  "
                f"{replay_mib:10.1f}  {100 * cpu / wall:5.1f}  {rss_after:7.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
