#!/usr/bin/env python3
"""Profile aggregate FleMacroEnv throughput across Factorio server counts."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fle.rl import schema as S


START_RCON_PORT = 27000
RCON_TIMEOUT_S = 600.0
SAMPLE_INTERVAL_S = 5.0
WORKER_EXIT_GRACE_S = 130.0
POINT_FIELDS = (
    "machine",
    "price_per_hour",
    "vcpus",
    "price_per_vcpu_hour",
    "servers",
    "vcpus_used",
    "seconds",
    "regime",
    "mapgen_s",
    "steps_total",
    "steps_per_s",
    "steps_per_s_per_server",
    "episodes_completed",
    "decisions_per_episode",
    "reset_s_mean",
    "step_wall_p50",
    "step_wall_p90",
    "step_wall_p99",
    "ticks_per_s_per_server",
    "cpu_pct_mean",
    "load1_mean",
    "rss_gb_total",
    "rss_gb_per_server",
    "ram_free_gb_min",
    "ok_rate",
    "invalid_rate",
    "deaths",
    "usd_per_1k_steps",
    "steps_per_usd",
    "usd_per_1k_steps_used",
    "marginal_steps_per_s_per_vcpu",
    "log_bytes_total",
    "status",
    "error",
)


class Cluster(Protocol):
    def start(self, servers: int, scenario: str) -> None: ...

    def teardown(self) -> None: ...


class FactorioCluster:
    """Thin wrapper around the implementation used by ``fle cluster start``."""

    def __init__(self) -> None:
        self._manager: Any | None = None

    def _get_manager(self) -> Any:
        if self._manager is None:
            from fle.cluster.run_envs import ClusterManager

            self._manager = ClusterManager()
        return self._manager

    def start(self, servers: int, scenario: str) -> None:
        self._get_manager().start(servers, scenario)

    def teardown(self) -> None:
        try:
            manager = self._get_manager()
            if manager.compose_path.exists():
                manager.stop()
        except (Exception, SystemExit) as exc:  # cleanup is best effort
            print(f"cluster teardown warning: {exc}", file=sys.stderr, flush=True)


class DryRunCluster:
    def start(self, servers: int, scenario: str) -> None:
        del servers, scenario

    def teardown(self) -> None:
        return None


def parse_servers(value: str) -> list[int]:
    try:
        servers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--servers must be comma-separated integers"
        ) from exc
    if not servers or any(count <= 0 for count in servers):
        raise argparse.ArgumentTypeError("--servers entries must be positive")
    if servers != sorted(set(servers)):
        raise argparse.ArgumentTypeError("--servers must be unique and ascending")
    return servers


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--servers", type=parse_servers, default=parse_servers("8,16,24,32,40,48,64")
    )
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--warmup", type=float, default=30.0)
    parser.add_argument("--scenario", default="open_world")
    parser.add_argument("--regime", choices=("macro", "bare"), default="macro")
    parser.add_argument("--speed", type=float, default=40.0)
    parser.add_argument("--machine", default="c3d-highcpu-90")
    parser.add_argument("--price-per-hour", type=float, default=0.75)
    parser.add_argument("--vcpus", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", type=Path, default=Path("runs/throughput/vm-sweep"))
    parser.add_argument(
        "--learner-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=RCON_TIMEOUT_S,
        help="seconds to wait for all RCON ports; raise it for large server counts",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-env",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--keep-logs", action="store_true")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.speed <= 0 or args.price_per_hour <= 0:
        parser.error("--speed and --price-per-hour must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.vcpus is None:
        if not args.dry_run:
            parser.error("--vcpus is required unless --dry-run is used")
        args.vcpus = max(args.servers)
    if args.vcpus <= 0:
        parser.error("--vcpus must be positive")
    return args


def _random_acceptance_action(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mirror fle.rl.env's random policy, including its CRAFT quantity-0 probe."""
    action = S.random_valid_action(obs, rng)
    op = S.OPS[int(action[S.HEADS.index("op")])]
    if op == "CRAFT":
        action[S.HEADS.index("quantity")] = 0
    return action


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _make_worker_env(
    *,
    dry_run: bool,
    port: int,
    speed: float,
    seed: int,
    regime: str,
    max_steps: int | None,
    log_path: Path | None,
    live_env_class: type | None = None,
    fake_env_class: type | None = None,
) -> Any:
    """Construct the worker env without overriding production episode defaults."""
    if dry_run:
        if fake_env_class is None:
            from fle.rl.fake_env import FakeMacroEnv

            fake_env_class = FakeMacroEnv
        kwargs: dict[str, Any] = {"seed": seed, "regime": regime}
        if max_steps is not None:
            kwargs["horizon"] = max_steps
        return fake_env_class(**kwargs)

    if live_env_class is None:
        from fle.rl.env import FleMacroEnv

        live_env_class = FleMacroEnv
    kwargs = {
        "port": port,
        "speed": speed,
        "seed": seed,
        "regime": regime,
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if log_path is not None:
        kwargs["log_path"] = log_path
    return live_env_class(**kwargs)


def _worker_main(
    worker_index: int,
    port: int,
    dry_run: bool,
    regime: str,
    speed: float,
    max_steps: int | None,
    log_env: bool,
    point_dir: str,
    start_event: Any,
    measure_start: Any,
    deadline: Any,
    result_queue: Any,
) -> None:
    """Build one environment and return aggregate measurements, never raw timings."""
    env: Any | None = None
    try:
        seed = 10_000 + worker_index
        rng = np.random.default_rng(seed)
        log_path = Path(point_dir) / f"env_{port}.jsonl" if log_env else None
        env = _make_worker_env(
            dry_run=dry_run,
            port=port,
            speed=speed,
            seed=seed,
            regime=regime,
            max_steps=max_steps,
            log_path=log_path,
        )
        episode = 0
        obs, _ = env.reset(seed=seed)
        deaths_before = int(getattr(env, "deaths", 0))
        result_queue.put(("ready", worker_index, None))
        start_event.wait()

        step_count = 0
        ok_count = 0
        invalid_count = 0
        invalid_denominator = 0
        elapsed_ticks = 0
        tick_baseline: int | None = None
        episode_decisions = 0
        completed_episode_decisions = 0
        episodes_completed = 0
        reset_s_total = 0.0
        # A bounded deterministic reservoir keeps worker memory flat during fast dry runs.
        timing_reservoir: list[float] = []
        timings_seen = 0

        while time.monotonic() < deadline.value:
            step_started = time.monotonic()
            measuring = step_started >= measure_start.value
            if measuring and not dry_run and tick_baseline is None:
                tick_baseline = int(env.instance.get_elapsed_ticks())
            action = _random_acceptance_action(obs, rng)
            obs, _, terminated, truncated, info = env.step(action)
            episode_decisions += 1
            wall_s = time.monotonic() - step_started
            if measuring:
                step_count += 1
                timings_seen += 1
                if len(timing_reservoir) < 4096:
                    timing_reservoir.append(wall_s)
                else:
                    slot = int(rng.integers(timings_seen))
                    if slot < len(timing_reservoir):
                        timing_reservoir[slot] = wall_s
                status = str(info.get("status", ""))
                op = str(info.get("op", ""))
                ok_count += int(status == "ok")
                if op != "WAIT":
                    invalid_denominator += 1
                    invalid_count += int(status in {"tool_rejected", "no_effect"})

            if terminated or truncated:
                if measuring and not dry_run and tick_baseline is not None:
                    current_ticks = int(env.instance.get_elapsed_ticks())
                    elapsed_ticks += max(0, current_ticks - tick_baseline)
                if measuring:
                    episodes_completed += 1
                    completed_episode_decisions += episode_decisions
                episode += 1
                reset_started = time.perf_counter()
                obs, _ = env.reset(seed=seed + episode)
                reset_wall_s = time.perf_counter() - reset_started
                if measuring:
                    reset_s_total += reset_wall_s
                episode_decisions = 0
                tick_baseline = (
                    int(env.instance.get_elapsed_ticks())
                    if measuring and not dry_run
                    else None
                )

        if not dry_run and tick_baseline is not None:
            elapsed_ticks += max(
                0, int(env.instance.get_elapsed_ticks()) - tick_baseline
            )
        payload = {
            "steps": step_count,
            "p50": _percentile(timing_reservoir, 0.50),
            "p90": _percentile(timing_reservoir, 0.90),
            "p99": _percentile(timing_reservoir, 0.99),
            "ticks": elapsed_ticks,
            "ok": ok_count,
            "invalid": invalid_count,
            "invalid_denominator": invalid_denominator,
            "deaths": max(0, int(getattr(env, "deaths", 0)) - deaths_before),
            "episodes_completed": episodes_completed,
            "completed_episode_decisions": completed_episode_decisions,
            "reset_s_total": reset_s_total,
        }
        # Closing before reporting guarantees that the parent observes flushed log bytes.
        env.close()
        env = None
        result_queue.put(("result", worker_index, payload))
    except BaseException:  # a worker failure must become a point record
        result_queue.put(
            ("error", worker_index, traceback.format_exc(limit=12)[-4000:])
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


class ProcCpuSampler:
    def __init__(self) -> None:
        self.previous = self._read()

    @staticmethod
    def _read() -> tuple[int, int] | None:
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            values = [int(value) for value in fields]
        except (OSError, ValueError, IndexError):
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    def sample(self) -> float:
        current = self._read()
        previous = self.previous
        self.previous = current
        if current is None or previous is None:
            return 0.0
        total_delta = current[0] - previous[0]
        idle_delta = current[1] - previous[1]
        if total_delta <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))


def _free_ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _factorio_rss_gb() -> float:
    """Sum host RSS for running containers whose names start with factorio_."""
    if shutil.which("docker") is None:
        return 0.0
    try:
        listed = subprocess.run(
            ["docker", "ps", "--filter", "name=factorio_", "--format", "{{.ID}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        ids = listed.stdout.split()
        if not ids:
            return 0.0
        inspected = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", *ids],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rss_kb = 0
        for raw_pid in inspected.stdout.split():
            status = Path(f"/proc/{int(raw_pid)}/status").read_text()
            vm_rss = next(
                line for line in status.splitlines() if line.startswith("VmRSS:")
            )
            rss_kb += int(vm_rss.split()[1])
        return rss_kb / (1024.0 * 1024.0)
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        return 0.0


def _system_sample(dry_run: bool, cpu: ProcCpuSampler) -> dict[str, float]:
    try:
        load1 = float(os.getloadavg()[0])
    except OSError:
        load1 = 0.0
    return {
        "cpu": cpu.sample(),
        "load1": load1,
        "rss": 0.0 if dry_run else _factorio_rss_gb(),
        "ram_free": _free_ram_gb(),
    }


def wait_for_rcon(servers: int, timeout_s: float = RCON_TIMEOUT_S) -> None:
    from factorio_rcon import RCONClient

    pending = set(range(START_RCON_PORT, START_RCON_PORT + servers))
    deadline = time.monotonic() + timeout_s
    delay = 0.25
    errors: dict[int, str] = {}
    while pending and time.monotonic() < deadline:
        for port in list(pending):
            client = None
            try:
                client = RCONClient("127.0.0.1", port, "factorio", timeout=1)
                response = client.send_command('/sc rcon.print("ready")')
                if response is not None:
                    pending.remove(port)
                    errors.pop(port, None)
            except Exception as exc:  # server may still be generating its map
                errors[port] = str(exc)
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
        if pending:
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(5.0, delay * 1.5)
    if pending:
        detail = "; ".join(
            f"{port}: {errors.get(port, 'no response')}" for port in sorted(pending)
        )
        raise TimeoutError(f"RCON startup timed out after {timeout_s:.0f}s ({detail})")


def cost_metrics(steps_per_s: float, price_per_hour: float) -> tuple[float, float]:
    if steps_per_s <= 0:
        return 0.0, 0.0
    steps_per_usd = steps_per_s * 3600.0 / price_per_hour
    return 1000.0 / steps_per_usd, steps_per_usd


def used_resource_cost(
    steps_per_s: float,
    price_per_hour: float,
    vcpus: int,
    vcpus_used: int,
) -> tuple[float, float]:
    price_per_vcpu_hour = price_per_hour / vcpus
    if steps_per_s <= 0:
        return price_per_vcpu_hour, 0.0
    usd_per_1k_steps_used = vcpus_used * price_per_vcpu_hour / (steps_per_s * 3.6)
    return price_per_vcpu_hour, usd_per_1k_steps_used


def with_marginal_metrics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies with marginals computed between consecutive successful points."""
    annotated: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for original in records:
        record = dict(original)
        record.setdefault("vcpus_used", int(record["servers"]))
        record["marginal_steps_per_s_per_vcpu"] = None
        if record.get("status") == "ok":
            if previous is not None:
                delta_vcpus = int(record["vcpus_used"]) - int(previous["vcpus_used"])
                if delta_vcpus > 0:
                    record["marginal_steps_per_s_per_vcpu"] = (
                        float(record["steps_per_s"]) - float(previous["steps_per_s"])
                    ) / delta_vcpus
            previous = record
        annotated.append(record)
    return annotated


def knee_server_count(records: Sequence[dict[str, Any]]) -> int | None:
    """Return the largest N whose marginal rate is at least half its average rate."""
    eligible = []
    for record in with_marginal_metrics(records):
        if record.get("status") != "ok":
            continue
        marginal = record["marginal_steps_per_s_per_vcpu"]
        vcpus_used = int(record["vcpus_used"])
        if marginal is None or vcpus_used <= 0:
            continue
        average = float(record["steps_per_s"]) / vcpus_used
        if float(marginal) >= 0.5 * average:
            eligible.append(int(record["servers"]))
    return max(eligible, default=None)


def _base_record(
    args: argparse.Namespace, servers: int, mapgen_s: float = 0.0
) -> dict[str, Any]:
    return {
        "machine": str(args.machine),
        "price_per_hour": float(args.price_per_hour),
        "vcpus": int(args.vcpus),
        "price_per_vcpu_hour": float(args.price_per_hour / args.vcpus),
        "servers": int(servers),
        "vcpus_used": int(servers),
        "seconds": float(args.seconds),
        "regime": str(args.regime),
        "mapgen_s": float(mapgen_s),
        "steps_total": 0,
        "steps_per_s": 0.0,
        "steps_per_s_per_server": 0.0,
        "episodes_completed": 0,
        "decisions_per_episode": 0.0,
        "reset_s_mean": 0.0,
        "step_wall_p50": 0.0,
        "step_wall_p90": 0.0,
        "step_wall_p99": 0.0,
        "ticks_per_s_per_server": 0.0,
        "cpu_pct_mean": 0.0,
        "load1_mean": 0.0,
        "rss_gb_total": 0.0,
        "rss_gb_per_server": 0.0,
        "ram_free_gb_min": 0.0,
        "ok_rate": 0.0,
        "invalid_rate": 0.0,
        "deaths": 0,
        "usd_per_1k_steps": 0.0,
        "steps_per_usd": 0.0,
        "usd_per_1k_steps_used": 0.0,
        "marginal_steps_per_s_per_vcpu": None,
        "log_bytes_total": 0,
        "status": "worker_failed",
        "error": "",
    }


def _point_dir(args: argparse.Namespace, servers: int) -> Path:
    return args.out / f"point-{servers}"


def _log_bytes(point_dir: Path) -> int:
    return sum(
        path.stat().st_size for path in point_dir.glob("env_*.jsonl") if path.is_file()
    )


def _remove_env_logs(point_dir: Path) -> None:
    if not point_dir.exists():
        return
    for path in point_dir.glob("env_*.jsonl"):
        if path.is_file():
            path.unlink()
    try:
        point_dir.rmdir()
    except OSError:
        pass


def run_point(
    args: argparse.Namespace,
    servers: int,
    mapgen_s: float = 0.0,
    worker_target: Callable[..., None] = _worker_main,
) -> dict[str, Any]:
    record = _base_record(args, servers, mapgen_s)
    point_dir = _point_dir(args, servers)
    if args.log_env:
        point_dir.mkdir(parents=True, exist_ok=True)
        _remove_env_logs(point_dir)
        point_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    start_event = context.Event()
    measure_start = context.Value("d", 0.0)
    deadline = context.Value("d", 0.0)
    processes = [
        context.Process(
            target=worker_target,
            args=(
                index,
                START_RCON_PORT + index,
                args.dry_run,
                args.regime,
                args.speed,
                args.max_steps,
                args.log_env and not args.dry_run,
                str(point_dir),
                start_event,
                measure_start,
                deadline,
                result_queue,
            ),
            name=f"throughput-{START_RCON_PORT + index}",
        )
        for index in range(servers)
    ]
    results: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    samples: list[dict[str, float]] = []
    try:
        for process in processes:
            process.start()
        ready: set[int] = set()
        ready_deadline = time.monotonic() + RCON_TIMEOUT_S
        while len(ready) < servers and not errors:
            try:
                kind, index, payload = result_queue.get(timeout=0.1)
            except queue.Empty:
                dead = [
                    p.name
                    for p in processes
                    if p.exitcode is not None and p.exitcode != 0
                ]
                if dead:
                    errors.append(f"workers exited before ready: {', '.join(dead)}")
                elif time.monotonic() >= ready_deadline:
                    errors.append(
                        "workers did not become ready before the 600 s timeout"
                    )
                continue
            if kind == "ready":
                ready.add(index)
            elif kind == "error":
                errors.append(f"worker {index}: {payload}")
            elif kind == "result":
                errors.append(f"worker {index} returned before measurement started")
        if errors:
            record["error"] = "\n".join(errors)[-4000:]
            return record

        now = time.monotonic()
        measure_start.value = now + args.warmup
        deadline.value = measure_start.value + args.seconds
        cpu: ProcCpuSampler | None = None
        start_event.set()
        next_sample = measure_start.value
        exit_deadline = deadline.value + WORKER_EXIT_GRACE_S
        while len(results) < servers and not errors:
            now = time.monotonic()
            if now >= next_sample and next_sample <= deadline.value:
                if cpu is None:
                    # Prime the CPU counters at the end of warmup so warmup load
                    # is not included in the measured-window mean.
                    cpu = ProcCpuSampler()
                    next_sample = min(
                        measure_start.value + SAMPLE_INTERVAL_S, deadline.value
                    )
                else:
                    samples.append(_system_sample(args.dry_run, cpu))
                    next_sample = (
                        min(next_sample + SAMPLE_INTERVAL_S, deadline.value)
                        if next_sample < deadline.value
                        else math.inf
                    )
            timeout = max(0.01, min(0.1, exit_deadline - now))
            try:
                kind, index, payload = result_queue.get(timeout=timeout)
            except queue.Empty:
                dead_missing = [
                    p.name
                    for i, p in enumerate(processes)
                    if i not in results and p.exitcode is not None and p.exitcode != 0
                ]
                if dead_missing:
                    errors.append(
                        f"workers exited without results: {', '.join(dead_missing)}"
                    )
                elif time.monotonic() >= exit_deadline:
                    errors.append(
                        "workers did not finish after the measurement deadline"
                    )
                continue
            if kind == "result":
                results[index] = payload
            elif kind == "error":
                errors.append(f"worker {index}: {payload}")
        if errors:
            record["error"] = "\n".join(errors)[-4000:]
            return record

        if not samples:
            if cpu is None:
                cpu = ProcCpuSampler()
            samples.append(_system_sample(args.dry_run, cpu))

        steps = sum(int(item["steps"]) for item in results.values())
        steps_per_s = steps / args.seconds
        weights = max(1, steps)

        def weighted(field: str) -> float:
            return (
                sum(
                    float(item[field]) * int(item["steps"]) for item in results.values()
                )
                / weights
            )

        ok = sum(int(item["ok"]) for item in results.values())
        invalid = sum(int(item["invalid"]) for item in results.values())
        invalid_denominator = sum(
            int(item["invalid_denominator"]) for item in results.values()
        )
        rss_total = max((sample["rss"] for sample in samples), default=0.0)
        usd_per_1k, steps_per_usd = cost_metrics(steps_per_s, args.price_per_hour)
        price_per_vcpu_hour, usd_per_1k_steps_used = used_resource_cost(
            steps_per_s,
            args.price_per_hour,
            args.vcpus,
            servers,
        )
        episodes_completed = sum(
            int(item["episodes_completed"]) for item in results.values()
        )
        completed_episode_decisions = sum(
            int(item["completed_episode_decisions"]) for item in results.values()
        )
        reset_s_total = sum(float(item["reset_s_total"]) for item in results.values())
        record.update(
            price_per_vcpu_hour=price_per_vcpu_hour,
            steps_total=steps,
            steps_per_s=steps_per_s,
            steps_per_s_per_server=steps_per_s / servers,
            episodes_completed=episodes_completed,
            decisions_per_episode=(
                completed_episode_decisions / episodes_completed
                if episodes_completed
                else 0.0
            ),
            reset_s_mean=(
                reset_s_total / episodes_completed if episodes_completed else 0.0
            ),
            step_wall_p50=weighted("p50"),
            step_wall_p90=weighted("p90"),
            step_wall_p99=weighted("p99"),
            ticks_per_s_per_server=(
                sum(int(item["ticks"]) for item in results.values())
                / args.seconds
                / servers
            ),
            cpu_pct_mean=(
                sum(sample["cpu"] for sample in samples) / len(samples)
                if samples
                else 0.0
            ),
            load1_mean=(
                sum(sample["load1"] for sample in samples) / len(samples)
                if samples
                else 0.0
            ),
            rss_gb_total=rss_total,
            rss_gb_per_server=rss_total / servers,
            ram_free_gb_min=min(
                (sample["ram_free"] for sample in samples), default=0.0
            ),
            ok_rate=ok / steps if steps else 0.0,
            invalid_rate=invalid / invalid_denominator if invalid_denominator else 0.0,
            deaths=sum(int(item["deaths"]) for item in results.values()),
            usd_per_1k_steps=usd_per_1k,
            steps_per_usd=steps_per_usd,
            usd_per_1k_steps_used=usd_per_1k_steps_used,
            log_bytes_total=_log_bytes(point_dir),
            status="ok",
            error="",
        )
        return record
    finally:
        start_event.set()
        for process in processes:
            process.join(timeout=0.25)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)
        if args.log_env:
            record["log_bytes_total"] = _log_bytes(point_dir)
        result_queue.close()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in {path} line {line_number}: {exc}"
            ) from exc
    return records


def _append_record(path: Path, record: dict[str, Any]) -> None:
    missing = set(POINT_FIELDS) - record.keys()
    if missing:
        raise ValueError(f"point record is missing fields: {sorted(missing)}")
    with path.open("a", buffering=1) as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()


def run_sweep(
    args: argparse.Namespace,
    cluster: Cluster | None = None,
    point_runner: Callable[..., dict[str, Any]] = run_point,
    worker_target: Callable[..., None] = _worker_main,
) -> list[dict[str, Any]]:
    args.out.mkdir(parents=True, exist_ok=True)
    sweep_path = args.out / "sweep.jsonl"
    existing = _read_records(sweep_path) if args.resume else []
    if not args.resume:
        sweep_path.write_text("")
    completed = {int(record["servers"]) for record in existing}
    cluster = cluster or (DryRunCluster() if args.dry_run else FactorioCluster())

    try:
        for servers in args.servers:
            if servers in completed:
                print(f"servers={servers}: resume skip", flush=True)
                continue
            cluster.teardown()
            record: dict[str, Any] | None = None
            mapgen_s = 0.0
            try:
                if not args.dry_run:
                    mapgen_started = time.perf_counter()
                    try:
                        cluster.start(servers, args.scenario)
                        wait_for_rcon(servers, args.ready_timeout)
                        mapgen_s = time.perf_counter() - mapgen_started
                    except KeyboardInterrupt:
                        raise
                    except BaseException:
                        mapgen_s = time.perf_counter() - mapgen_started
                        record = _base_record(args, servers, mapgen_s)
                        record.update(
                            status="server_start_failed",
                            error=traceback.format_exc(limit=12)[-4000:],
                        )
                if record is None:
                    try:
                        record = point_runner(
                            args,
                            servers,
                            mapgen_s=mapgen_s,
                            worker_target=worker_target,
                        )
                    except KeyboardInterrupt:
                        raise
                    except BaseException:
                        record = _base_record(args, servers, mapgen_s)
                        record.update(
                            status="worker_failed",
                            error=traceback.format_exc(limit=12)[-4000:],
                        )
                history = _read_records(sweep_path)
                record["marginal_steps_per_s_per_vcpu"] = with_marginal_metrics(
                    [*history, record]
                )[-1]["marginal_steps_per_s_per_vcpu"]
                _append_record(sweep_path, record)
                print(
                    f"servers={servers}: status={record['status']} steps/s={record['steps_per_s']:.3f}",
                    flush=True,
                )
            finally:
                try:
                    cluster.teardown()
                finally:
                    if args.log_env and not args.keep_logs:
                        _remove_env_logs(_point_dir(args, servers))
    finally:
        cluster.teardown()
    return _read_records(sweep_path)


def _format_number(record: dict[str, Any], field: str, digits: int = 3) -> str:
    if record.get("status") != "ok":
        return "-"
    value = record.get(field)
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render_summary(
    sweep_path: Path,
    learner_path: Path | None = None,
) -> str:
    records = with_marginal_metrics(_read_records(sweep_path))
    learner: dict[str, Any] = {}
    if learner_path is not None and learner_path.exists():
        learner = json.loads(learner_path.read_text())

    lines = [
        "# VM throughput sweep",
        "",
        "| servers | status | steps/s | steps/s/server | episodes | decisions/episode | reset ms | ticks/s/server | CPU % | RSS GB | marginal steps/s/vCPU | used USD/1k steps | whole-VM USD/1k steps | log MB |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {servers} | {status} | {sps} | {per_server} | {episodes} | {decisions} | {reset_ms} | {ticks} | {cpu} | {rss} | {marginal} | {used_cost} | {whole_cost} | {log_mb} |".format(
                servers=record.get("servers", "?"),
                status=record.get("status", "failed"),
                sps=_format_number(record, "steps_per_s"),
                per_server=_format_number(record, "steps_per_s_per_server"),
                episodes=(
                    str(int(record.get("episodes_completed", 0)))
                    if record.get("status") == "ok"
                    else "-"
                ),
                decisions=_format_number(record, "decisions_per_episode", 1),
                reset_ms=(
                    f"{float(record['reset_s_mean']) * 1000.0:.1f}"
                    if record.get("status") == "ok"
                    else "-"
                ),
                ticks=_format_number(record, "ticks_per_s_per_server"),
                cpu=_format_number(record, "cpu_pct_mean", 1),
                rss=_format_number(record, "rss_gb_total", 2),
                marginal=_format_number(record, "marginal_steps_per_s_per_vcpu"),
                used_cost=_format_number(record, "usd_per_1k_steps_used", 6),
                whole_cost=_format_number(record, "usd_per_1k_steps", 6),
                log_mb=(
                    f"{int(record.get('log_bytes_total', 0)) / (1024 * 1024):.2f}"
                    if record.get("status") == "ok"
                    else "-"
                ),
            )
        )
    ok = [record for record in records if record.get("status") == "ok"]
    lines.append("")
    if ok:
        fastest = max(ok, key=lambda record: float(record["steps_per_s"]))
        whole_vm_cheapest = max(ok, key=lambda record: float(record["steps_per_usd"]))
        used_resource_cheapest = min(
            (record for record in ok if float(record["steps_per_s"]) > 0),
            key=lambda record: float(record["usd_per_1k_steps_used"]),
            default=fastest,
        )
        knee = knee_server_count(ok)
        operating_point = next(
            (record for record in ok if int(record["servers"]) == knee), fastest
        )
        lines.append(
            f"Aggregate throughput is maximal at N={fastest['servers']} "
            f"({float(fastest['steps_per_s']):.3f} steps/s)."
        )
        lines.append(
            f"Used-vCPU cost is minimal at N={used_resource_cheapest['servers']} "
            f"({float(used_resource_cheapest['usd_per_1k_steps_used']):.6f} USD/1k steps)."
        )
        lines.append(
            f"Whole-VM throughput per dollar is maximal at N={whole_vm_cheapest['servers']} "
            f"({float(whole_vm_cheapest['steps_per_usd']):.0f} steps/USD); it matches the "
            "aggregate-throughput optimum because the VM hourly price is fixed."
        )
        if knee is None:
            lines.append(
                "No measured point after the first satisfies the knee threshold."
            )
        else:
            lines.append(
                f"The throughput knee is N={knee}: its marginal steps/s per vCPU is at "
                "least half its average steps/s per vCPU."
            )
        at_eight = next((record for record in ok if int(record["servers"]) == 8), None)
        baseline = (
            f"{float(at_eight['ticks_per_s_per_server']):.3f}"
            if at_eight
            else "not measured"
        )
        lines.append(
            f"At N={fastest['servers']}, game speed is "
            f"{float(fastest['ticks_per_s_per_server']):.3f} ticks/s/server; at N=8 it is {baseline}."
        )
        probes = learner.get("probes", [])
        best_probe = None
        if probes:
            lines.append("")
            lines.append("Learner probe:")
            for probe in probes:
                servers_per_learner = (
                    float(probe["updates_per_s"])
                    / float(operating_point["steps_per_s_per_server"])
                    if float(operating_point["steps_per_s_per_server"]) > 0
                    else 0.0
                )
                lines.append(
                    f"- {int(probe['torch_threads'])} thread(s): "
                    f"{float(probe['updates_per_s']):.3f} updates/s, "
                    f"{float(probe['rss_gb']):.3f} GB RSS, "
                    f"{servers_per_learner:.2f} implied servers/learner."
                )
            best_probe = max(probes, key=lambda probe: float(probe["updates_per_s"]))
        lines.append("")
        if best_probe is None:
            recommendation = (
                "run the learner probe before fixing learner CPU allocation"
            )
        else:
            server_rate = float(operating_point["steps_per_s_per_server"])
            implied = (
                float(best_probe["updates_per_s"]) / server_rate
                if server_rate > 0
                else 0.0
            )
            recommendation = (
                f"use {int(best_probe['torch_threads'])} CPU thread(s) per learner and about "
                f"{implied:.1f} servers per learner"
            )
        lines.append(
            f"Recommendation for compute-recipe.md: use {int(operating_point['servers'])} servers per "
            f"{operating_point['machine']}; {recommendation}."
        )
    else:
        lines.append(
            "No sweep point completed successfully; no server-count recommendation is available."
        )
    return "\n".join(lines) + "\n"


def _proc_rss_gb(pid: int) -> float:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
        return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, StopIteration, IndexError):
        return 0.0


def run_learner_probe(out: Path) -> dict[str, Any]:
    """Run exactly 2,000 Rainbow optimizer updates at each thread count."""
    learner_path = out / "learner.json"
    if learner_path.exists():
        return json.loads(learner_path.read_text())
    probes = []
    for threads in (1, 2, 4):
        probe_out = out / f"learner-t{threads}"
        probe_out.mkdir(parents=True, exist_ok=True)
        log_path = probe_out / "probe.log"
        command = [
            sys.executable,
            "-m",
            "fle.rl.dqn",
            "--algo",
            "dqn",
            "--rainbow",
            "--lr",
            "1e-4",
            "--per-alpha",
            "0.6",
            "--per-optimism",
            "3",
            "--per-episode-return-bonus",
            "2",
            "--n-step",
            "5",
            "--replay-size",
            "100000",
            "--batch-size",
            "64",
            "--update-every",
            "1",
            "--device",
            "cpu",
            "--explore",
            "epsilon",
            "--eps-floor",
            "0.02",
            "--archive-frac",
            "0.25",
            "--archive-mode",
            "uniform",
            "--fake",
            "--total-steps",
            "2499",
            "--torch-threads",
            str(threads),
            "--run-name",
            f"throughput-learner-t{threads}",
            "--out",
            str(probe_out),
        ]
        started = time.perf_counter()
        peak_rss = 0.0
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    peak_rss = max(peak_rss, _proc_rss_gb(process.pid))
                    time.sleep(0.1)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        wall_s = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(
                f"learner probe for {threads} threads failed; see {log_path}"
            )
        # DQN begins optimizing at env step 500.  A 2,499-step run therefore
        # performs exactly the requested 2,000 updates (steps 500..2,499).
        import torch

        checkpoint = torch.load(
            probe_out / "checkpoint-final.pt",
            map_location="cpu",
            weights_only=False,
        )
        updates = int(checkpoint["updates"])
        if updates != 2000:
            raise RuntimeError(
                f"learner probe for {threads} threads made {updates} updates, expected 2000"
            )
        metric_rows = _read_records(probe_out / "metrics.jsonl")
        last_metric = metric_rows[-1] if metric_rows else None
        measured_updates = (
            max(0, int(last_metric["env_steps"]) - 499) if last_metric else 0
        )
        training_wall_s = float(last_metric["wall"]) if last_metric else wall_s
        updates_per_s = (
            measured_updates / training_wall_s
            if measured_updates and training_wall_s > 0
            else updates / wall_s
        )
        probes.append(
            {
                "torch_threads": threads,
                "updates": updates,
                "wall_s": wall_s,
                "updates_per_s": updates_per_s,
                "rss_gb": peak_rss,
            }
        )
    result = {"updates": 2000, "probes": probes}
    learner_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cluster: Cluster = DryRunCluster() if args.dry_run else FactorioCluster()
    interrupted = False
    try:
        run_sweep(args, cluster=cluster)
        if args.learner_probe and not args.dry_run:
            run_learner_probe(args.out)
    except KeyboardInterrupt:
        interrupted = True
        print(
            "interrupted; preserving completed sweep points",
            file=sys.stderr,
            flush=True,
        )
    finally:
        cluster.teardown()
        sweep_path = args.out / "sweep.jsonl"
        if sweep_path.exists():
            summary = render_summary(sweep_path, args.out / "learner.json")
            (args.out / "summary.md").write_text(summary)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
