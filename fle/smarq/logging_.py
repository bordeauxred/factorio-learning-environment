"""Durable JSONL logging for SM-ARQ experiments.

The logger deliberately accepts plain mappings.  Logging is a boundary of the
runner, not part of the frozen environment contract, and keeping it that way
also makes partially written experiments easy to inspect with standard tools.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np

STEP_FIELDS = (
    "tick",
    "simulated_seconds_elapsed",
    "wall_seconds",
    "game_seconds_per_wall_second",
    "production_score",
    "automated_production_score",
    "delta_production_score",
    "delta_automated_production_score",
    "verb",
    "heads",
    "failure_reason",
    "success",
    "requested_quantity",
    "executed_quantity",
    "q_value",
    "td_error",
    "per_priority",
    "exploratory",
    "epsilon_per_head",
)

EPISODE_FIELDS = (
    "return",
    "final_automated_score",
    "max_automated_score",
    "final_production_score",
    "automated_score_per_game_minute",
    "decisions",
    "simulated_ticks",
    "wall_seconds",
    "failure_reason_histogram",
    "verb_histogram",
    "end_reason",
)

METRIC_FIELDS = (
    "loss",
    "grad_norm",
    "mean_max_q",
    "td_error_mean",
    "td_error_p95",
    "epsilon",
    "replay_size",
    "replay_ratio",
    "updates_per_second",
    "transitions_per_wall_second",
    "simulated_seconds_per_wall_second",
)


def _json_default(value: Any) -> Any:
    """Convert numpy and path values without silently stringifying objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def automated_score_per_game_minute(score: float, simulated_ticks: int) -> float:
    """Return score rate using simulated time, with a well-defined zero case."""
    game_minutes = float(simulated_ticks) / (60.0 * 60.0)
    return float(score) / game_minutes if game_minutes > 0.0 else 0.0


def episode_record(
    *,
    episode_return: float,
    final_automated_score: float,
    max_automated_score: float,
    final_production_score: float,
    decisions: int,
    simulated_ticks: int,
    wall_seconds: float,
    failure_reason_histogram: Mapping[str, int],
    verb_histogram: Mapping[str, int],
    end_reason: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build the canonical episode record.

    In particular, ``wall_seconds`` is not used in the score rate calculation.
    """
    out: dict[str, Any] = {
        "return": float(episode_return),
        "final_automated_score": float(final_automated_score),
        "max_automated_score": float(max_automated_score),
        "final_production_score": float(final_production_score),
        "automated_score_per_game_minute": automated_score_per_game_minute(
            final_automated_score, simulated_ticks
        ),
        "decisions": int(decisions),
        "simulated_ticks": int(simulated_ticks),
        "wall_seconds": float(wall_seconds),
        "failure_reason_histogram": dict(failure_reason_histogram),
        "verb_histogram": dict(verb_histogram),
        "end_reason": end_reason,
    }
    out.update(extra)
    return out


class JsonlLogger:
    """Thread-safe, promptly flushed experiment logger.

    Every line is flushed into the OS after it is written.  ``fsync_every``
    additionally bounds the amount at risk from a kernel or machine crash;
    normal SIGINT/SIGTERM shutdown calls :meth:`close`, which always fsyncs.
    """

    def __init__(self, run_dir: str | Path, *, fsync_every: int = 32) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fsync_every = max(1, int(fsync_every))
        self._lock = threading.RLock()
        self._steps: dict[str, TextIO] = {}
        self._episodes = self._open("episodes.jsonl")
        self._metrics = self._open("metrics.jsonl")
        self._write_counts: dict[int, int] = {}
        self._closed = False

    def _open(self, name: str) -> TextIO:
        return (self.run_dir / name).open("a", encoding="utf-8", buffering=1)

    def _step_file(self, worker: int | str) -> TextIO:
        key = str(worker)
        stream = self._steps.get(key)
        if stream is None:
            stream = self._open(f"steps_{key}.jsonl")
            self._steps[key] = stream
        return stream

    def _write(self, stream: TextIO, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("logger is closed")
        stream.write(json.dumps(dict(record), default=_json_default, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        fileno = stream.fileno()
        count = self._write_counts.get(fileno, 0) + 1
        self._write_counts[fileno] = count
        if count % self.fsync_every == 0:
            os.fsync(fileno)

    @staticmethod
    def _check(record: Mapping[str, Any], required: tuple[str, ...], kind: str) -> None:
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(f"{kind} record missing fields: {', '.join(missing)}")

    def log_step(
        self,
        worker: int | str,
        record: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        payload = dict(record or {})
        payload.update(fields)
        self._check(payload, STEP_FIELDS, "step")
        with self._lock:
            self._write(self._step_file(worker), payload)

    def log_episode(
        self, record: Mapping[str, Any] | None = None, **fields: Any
    ) -> None:
        payload = dict(record or {})
        payload.update(fields)
        self._check(payload, EPISODE_FIELDS, "episode")
        with self._lock:
            self._write(self._episodes, payload)

    def log_metrics(
        self, record: Mapping[str, Any] | None = None, **fields: Any
    ) -> None:
        payload = dict(record or {})
        payload.update(fields)
        self._check(payload, METRIC_FIELDS, "metrics")
        with self._lock:
            self._write(self._metrics, payload)

    def flush(self, *, durable: bool = False) -> None:
        with self._lock:
            streams = [*self._steps.values(), self._episodes, self._metrics]
            for stream in streams:
                stream.flush()
                if durable:
                    os.fsync(stream.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            streams = [*self._steps.values(), self._episodes, self._metrics]
            for stream in streams:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            self._closed = True

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _last_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        pos = stream.tell() - 1
        while pos >= 0:
            stream.seek(pos)
            if stream.read(1) != b"\n":
                break
            pos -= 1
        while pos >= 0:
            stream.seek(pos)
            if stream.read(1) == b"\n":
                pos += 1
                break
            pos -= 1
        stream.seek(max(0, pos))
        line = stream.readline().decode("utf-8").strip()
    return json.loads(line) if line else None


def tail_status(run_dir: str | Path) -> str:
    """Print and return a one-line status assembled from the JSONL tails."""
    root = Path(run_dir)
    metric = _last_json(root / "metrics.jsonl") or {}
    episode = _last_json(root / "episodes.jsonl") or {}
    step_files = sorted(root.glob("steps_*.jsonl"))
    steps = [_last_json(path) for path in step_files]
    steps = [step for step in steps if step is not None]
    decisions = sum(1 for path in step_files for _ in path.open(encoding="utf-8"))
    tick = max((int(step.get("tick", 0)) for step in steps), default=0)
    status = (
        f"decisions={decisions} tick={tick} updates={int(metric.get('updates', 0))} "
        f"replay={int(metric.get('replay_size', 0))} "
        f"loss={float(metric.get('loss', 0.0)):.4g} "
        f"auto={float(episode.get('final_automated_score', 0.0)):.3g}"
    )
    print(status, flush=True)
    return status
