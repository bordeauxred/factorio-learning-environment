"""Live pre-run diagnostic for the V0 masked action environment."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fle.rl.env import FleMacroEnv
from fle.rl.v0_actions import decode_v0_action, v0_masks_for_prefix
from fle.rl.v0_schema import (
    HEAD_SIZES,
    HEADS,
    LOCAL_CHANNEL_INDEX,
    LOCAL_SIDE,
    PLACEABLE_NAMES,
    VERB_HEADS,
    VERB_INDEX,
    VERBS,
    cell_to_world,
)


@dataclass
class Rate:
    attempts: int = 0
    successes: int = 0

    def add(self, success: bool) -> None:
        self.attempts += 1
        self.successes += int(success)

    def report(self) -> dict[str, int | float | None]:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "rate": self.successes / self.attempts if self.attempts else None,
        }


@dataclass
class DiagnosticStats:
    decisions: int = 0
    successes: int = 0
    wall_s: float = 0.0
    simulated_s: float = 0.0
    verb_histogram: Counter[str] = field(default_factory=Counter)
    failure_reasons: Counter[str] = field(default_factory=Counter)
    phases: Counter[str] = field(default_factory=Counter)
    build_sums: Counter[str] = field(default_factory=Counter)
    place_before: int = 0
    place_after: list[int] = field(default_factory=list)
    place_by_cache: dict[str, Rate] = field(default_factory=lambda: defaultdict(Rate))
    mine: Rate = field(default_factory=Rate)
    insert: Rate = field(default_factory=Rate)
    first_reward: dict[str, Any] | None = None
    stop_conditions: list[str] = field(default_factory=list)


def _harvestable_mask(obs: dict[str, Any]) -> np.ndarray:
    local = obs["local_exact"]
    channels = (
        "iron_ore",
        "copper_ore",
        "coal",
        "stone",
        "uranium_ore",
        "tree",
        "rock_cliff",
    )
    return np.any(
        local[[LOCAL_CHANNEL_INDEX[name] for name in channels]] > 0,
        axis=0,
    ).reshape(-1)


def _nearest_harvestable_distance(obs: dict[str, Any]) -> float | None:
    cells = np.flatnonzero(_harvestable_mask(obs))
    if not cells.size:
        return None
    y, x = np.divmod(cells, LOCAL_SIDE)
    distances = np.hypot(x + 0.5 - LOCAL_SIDE / 2, y + 0.5 - LOCAL_SIDE / 2)
    return float(distances.min())


def _sample_action_once(
    env: FleMacroEnv, rng: np.random.Generator
) -> tuple[np.ndarray | None, float, dict[str, np.ndarray]]:
    assert env._frame is not None
    obs = env._frame.obs
    context = env._v0_context(env._frame)
    prefix: dict[str, int] = {}
    action = np.zeros(len(HEADS), dtype=np.int64)
    mask_wall_s = 0.0
    final_masks: dict[str, np.ndarray] = {}

    started = time.perf_counter()
    masks = v0_masks_for_prefix(prefix, obs, env.world, context)
    mask_wall_s += time.perf_counter() - started
    harvestable = _harvestable_mask(obs)
    if not harvestable.any():
        masks["verb"][VERB_INDEX["MINE"]] = 0
    admitted_verbs = np.flatnonzero(masks["verb"])
    if not admitted_verbs.size:
        raise RuntimeError("V0 diagnostic found no admitted verb")
    verb_index = int(rng.choice(admitted_verbs))
    verb = VERBS[verb_index]
    action[HEADS.index("verb")] = verb_index
    prefix["verb"] = verb_index

    for head in VERB_HEADS[verb]:
        started = time.perf_counter()
        masks = v0_masks_for_prefix(prefix, obs, env.world, context)
        mask_wall_s += time.perf_counter() - started
        if verb == "MINE" and head == "location":
            masks["location"] = harvestable.astype(np.uint8)
        admitted = np.flatnonzero(masks[head])
        if not admitted.size:
            return None, mask_wall_s, masks
        value = int(rng.choice(admitted))
        action[HEADS.index(head)] = value
        prefix[head] = value
        final_masks = masks
    return action, mask_wall_s, final_masks


def _sample_action(
    env: FleMacroEnv, rng: np.random.Generator
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """Rejection-sample complete tuples when a conditional domain is empty."""
    mask_wall_s = 0.0
    for _ in range(128):
        action, attempt_wall_s, final_masks = _sample_action_once(env, rng)
        mask_wall_s += attempt_wall_s
        if action is not None:
            return action, mask_wall_s, final_masks
    raise RuntimeError("could not sample a complete V0 action after 128 attempts")


def _build_fractions(env: FleMacroEnv) -> dict[str, float]:
    cache = env.world.buildability
    values = cache.values
    sampled = cache.sampled_ticks
    tick = env._current.tick
    known = values >= 0
    if sampled.size:
        ages = tick - sampled
        fresh_blocks = (sampled >= 0) & (ages >= 0) & (ages <= env.place_max_age_ticks)
        fresh = fresh_blocks.repeat(8, axis=1).repeat(8, axis=2) & known
    else:
        fresh = np.zeros_like(known)
    stale = known & ~fresh
    return {
        "known": float(known.mean()) if known.size else 0.0,
        "unknown": float((~known).mean()) if known.size else 1.0,
        "fresh": float(fresh.mean()) if fresh.size else 0.0,
        "stale": float(stale.mean()) if stale.size else 0.0,
    }


def _place_local_state(
    env: FleMacroEnv, action: np.ndarray
) -> tuple[str, np.ndarray, np.ndarray]:
    assert env._current is not None
    place_item = PLACEABLE_NAMES[int(action[HEADS.index("place_item")])]
    direction_index = int(action[HEADS.index("direction")])
    direction = (0, 4, 8, 12)[direction_index]
    prototype = env.vocab.place_results.get(place_item, place_item)
    cache = env.world.buildability
    try:
        channel = cache.channel_for(prototype, direction)
    except KeyError:
        return (
            "unknown",
            np.zeros(HEAD_SIZES["location"], dtype=bool),
            np.ones(HEAD_SIZES["location"], dtype=bool),
        )

    px, py = env._current.position
    local_values = np.full(HEAD_SIZES["location"], -1, dtype=np.int8)
    local_ticks = np.full(HEAD_SIZES["location"], -1, dtype=np.int64)
    origin_x, origin_y = cache.origin
    height, width = cache.values.shape[1:]
    for index in range(HEAD_SIZES["location"]):
        world_x, world_y = cell_to_world(index, px, py)
        cache_x, cache_y = world_x - origin_x, world_y - origin_y
        if 0 <= cache_x < width and 0 <= cache_y < height:
            local_values[index] = cache.values[channel, cache_y, cache_x]
            if cache.sampled_ticks.size:
                local_ticks[index] = cache.sampled_ticks[
                    channel, cache_y // 8, cache_x // 8
                ]
    age = env._current.tick - local_ticks
    fresh = (local_ticks >= 0) & (age >= 0) & (age <= env.place_max_age_ticks)
    expected_blocked = (local_values == 0) & fresh
    unknown_or_stale = (local_values < 0) | (local_ticks < 0) | ~fresh
    location = int(action[HEADS.index("location")])
    if local_values[location] < 0 or local_ticks[location] < 0:
        state = "unknown"
    elif not fresh[location]:
        state = "stale"
    elif local_values[location] == 1:
        state = "fresh-legal"
    else:
        state = "fresh-blocked"
    return state, expected_blocked, unknown_or_stale


def _check_place_invariants(
    actual: np.ndarray,
    expected_blocked: np.ndarray,
    unknown_or_stale: np.ndarray,
    stats: DiagnosticStats,
) -> None:
    removed = actual == 0
    if np.any(removed & unknown_or_stale):
        stats.stop_conditions.append(
            "unknown or stale PLACE cells were removed from the action space"
        )
    if not np.array_equal(removed, expected_blocked):
        stats.stop_conditions.append(
            "local frame and PLACE mask disagree on coordinates"
        )


def _json_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return action.astype(int).tolist()
    if isinstance(action, dict):
        return {key: _json_action(value) for key, value in action.items()}
    if isinstance(action, tuple):
        return list(action)
    if isinstance(action, np.generic):
        return action.item()
    return action


def _run(
    env: FleMacroEnv,
    decisions: int,
    seed: int,
    *,
    collect: bool,
) -> tuple[DiagnosticStats, list[float]]:
    rng = np.random.default_rng(seed)
    stats = DiagnosticStats()
    decision_times: list[float] = []
    started_run = time.perf_counter()
    for decision in range(decisions):
        decision_started = time.perf_counter()
        action, mask_wall_s, final_masks = _sample_action(env, rng)
        assert env._frame is not None
        decoded = decode_v0_action(
            action,
            env._frame.obs,
            env._v0_context(env._frame),
        )
        place_state = None
        if decoded.verb == "PLACE":
            place_state, expected_blocked, unknown_or_stale = _place_local_state(
                env, action
            )
            if collect:
                _check_place_invariants(
                    final_masks["location"],
                    expected_blocked,
                    unknown_or_stale,
                    stats,
                )
                stats.place_before += HEAD_SIZES["location"]
                stats.place_after.append(int(final_masks["location"].sum()))

        _obs, reward, terminated, truncated, info = env.step(action)
        elapsed = time.perf_counter() - decision_started
        decision_times.append(elapsed)
        if not collect:
            if terminated or truncated:
                break
            continue

        stats.decisions += 1
        stats.successes += int(info["success"])
        stats.simulated_s += float(info["tau"])
        stats.verb_histogram[decoded.verb] += 1
        stats.phases["mask_construction"] += mask_wall_s
        for name, value in info.get("phase_wall_s", {}).items():
            stats.phases[name] += float(value)
        if not info["success"]:
            stats.failure_reasons[info.get("reason_class") or "untyped"] += 1
        if decoded.verb == "PLACE" and place_state is not None:
            stats.place_by_cache[place_state].add(bool(info["success"]))
        elif decoded.verb == "MINE":
            stats.mine.add(bool(info["success"]))
        elif decoded.verb == "INSERT":
            stats.insert.add(bool(info["success"]))
        requested = info.get("requested_unit_number")
        mutated = info.get("mutated_unit_number")
        if requested is not None and mutated is not None and requested != mutated:
            stats.stop_conditions.append(
                f"entity identity mismatch: requested {requested}, mutated {mutated}"
            )
        fractions = _build_fractions(env)
        for name, value in fractions.items():
            stats.build_sums[name] += value
        if reward > 0 and stats.first_reward is None:
            stats.first_reward = {
                "decision": decision,
                "reward": float(reward),
                "verb": decoded.verb,
                "args": _json_action(decoded.args),
                "heads": action.astype(int).tolist(),
            }
        if stats.stop_conditions:
            break
        if terminated or truncated:
            break
    stats.wall_s = time.perf_counter() - started_run
    return stats, decision_times


def _report(
    stats: DiagnosticStats,
    *,
    nearest_harvestable: float | None,
    before_times: list[float],
    after_times: list[float],
    port: int,
    requested_decisions: int,
) -> dict[str, Any]:
    samples = max(stats.decisions, 1)
    failures = stats.decisions - stats.successes
    place_attempts = len(stats.place_after)
    before_wall = sum(before_times)
    after_wall = sum(after_times)
    return {
        "port": port,
        "requested_decisions": requested_decisions,
        "completed_decisions": stats.decisions,
        "nearest_in_frame_harvestable_distance": nearest_harvestable,
        "place_branching_factor": {
            "samples": place_attempts,
            "before_mean": (
                stats.place_before / place_attempts if place_attempts else None
            ),
            "after_mean": (
                float(np.mean(stats.place_after)) if stats.place_after else None
            ),
            "after_min": min(stats.place_after) if stats.place_after else None,
            "after_max": max(stats.place_after) if stats.place_after else None,
        },
        "buildability_fractions": {
            name: stats.build_sums[name] / samples
            for name in ("known", "unknown", "stale", "fresh")
        },
        "place_success_by_cache_state": {
            name: stats.place_by_cache[name].report()
            for name in ("fresh-legal", "unknown", "stale")
        },
        "mask_construction_latency_ms": {
            "total": stats.phases["mask_construction"] * 1e3,
            "mean_per_decision": stats.phases["mask_construction"] * 1e3 / samples,
        },
        "mine": stats.mine.report(),
        "insert": stats.insert.report(),
        "verb_histogram": dict(sorted(stats.verb_histogram.items())),
        "outcomes": {
            "successes": stats.successes,
            "failures": failures,
            "success_rate": stats.successes / samples,
            "failure_rate": failures / samples,
            "failure_reasons": dict(sorted(stats.failure_reasons.items())),
        },
        "throughput": {
            "unbatched_decisions_per_wall_second": (
                len(before_times) / before_wall if before_wall else 0.0
            ),
            "batched_decisions_per_wall_second": (
                len(after_times) / after_wall if after_wall else 0.0
            ),
            "overall_decisions_per_wall_second": stats.decisions / stats.wall_s,
            "simulated_seconds_per_wall_second": stats.simulated_s / stats.wall_s,
            "wall_seconds": stats.wall_s,
            "simulated_seconds": stats.simulated_s,
        },
        "phase_breakdown_ms_per_decision": {
            name: total * 1e3 / samples for name, total in sorted(stats.phases.items())
        },
        "first_automated_reward": stats.first_reward,
        "hard_stop_conditions": list(dict.fromkeys(stats.stop_conditions)),
        "hard_stop_fired": bool(stats.stop_conditions),
    }


def _print_report(report: dict[str, Any], json_path: Path) -> None:
    print("V0 pre-run diagnostic")
    print(f"port: {report['port']}")
    print(f"decisions: {report['completed_decisions']}/{report['requested_decisions']}")
    print(
        "nearest in-frame harvestable: "
        f"{report['nearest_in_frame_harvestable_distance']} tiles"
    )
    branch = report["place_branching_factor"]
    print(
        "PLACE branching factor: "
        f"{branch['before_mean']} -> {branch['after_mean']} "
        f"(n={branch['samples']}, range={branch['after_min']}..{branch['after_max']})"
    )
    print(
        "buildability fractions: "
        + json.dumps(report["buildability_fractions"], sort_keys=True)
    )
    print("PLACE success by cache state:")
    for name, row in report["place_success_by_cache_state"].items():
        print(f"  {name}: {json.dumps(row, sort_keys=True)}")
    print(
        "mask construction latency: "
        f"{report['mask_construction_latency_ms']['mean_per_decision']:.3f} ms/decision"
    )
    print("MINE: " + json.dumps(report["mine"], sort_keys=True))
    print("INSERT: " + json.dumps(report["insert"], sort_keys=True))
    print("verb histogram: " + json.dumps(report["verb_histogram"], sort_keys=True))
    print("outcomes: " + json.dumps(report["outcomes"], sort_keys=True))
    print("throughput: " + json.dumps(report["throughput"], sort_keys=True))
    print(
        "phase breakdown ms/decision: "
        + json.dumps(report["phase_breakdown_ms_per_decision"], sort_keys=True)
    )
    print(
        "first automated reward: "
        + json.dumps(report["first_automated_reward"], sort_keys=True)
    )
    if report["hard_stop_fired"]:
        print("HARD STOP: " + "; ".join(report["hard_stop_conditions"]))
    else:
        print("hard stops: none")
    print(f"json: {json_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=27001)
    parser.add_argument("--decisions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--benchmark-decisions", type=int, default=32)
    parser.add_argument(
        "--json-out", type=Path, default=Path("/tmp/fle-v0-diagnostic.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.decisions <= 0 or args.benchmark_decisions <= 0:
        raise SystemExit("--decisions and --benchmark-decisions must be positive")
    benchmark_decisions = min(args.benchmark_decisions, args.decisions)
    env = FleMacroEnv(
        port=args.port,
        speed=10,
        max_steps=max(args.decisions, benchmark_decisions) + 1,
        max_ticks=10**9,
        seed=args.seed,
        action_grammar="v0",
        v0_batch_snapshot=False,
    )
    try:
        initial_obs, _ = env.reset(seed=args.seed)
        nearest_harvestable = _nearest_harvestable_distance(initial_obs)
        _before, before_times = _run(env, benchmark_decisions, args.seed, collect=False)

        env.v0_batch_snapshot = True
        initial_obs, _ = env.reset(seed=args.seed)
        nearest_harvestable = _nearest_harvestable_distance(initial_obs)
        stats, decision_times = _run(env, args.decisions, args.seed, collect=True)
        after_times = decision_times[:benchmark_decisions]
        report = _report(
            stats,
            nearest_harvestable=nearest_harvestable,
            before_times=before_times,
            after_times=after_times,
            port=args.port,
            requested_decisions=args.decisions,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _print_report(report, args.json_out)
        return int(report["hard_stop_fired"])
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
