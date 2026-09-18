"""Run a uniform-random census of the twelve RL macro operations."""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from fle.env.instance import FactorioInstance
from fle.rl.ops import (
    OPS,
    ActionSpec,
    AnchorResolutionError,
    NoSupport,
    OperationSnapshot,
    SamplerState,
    VocabData,
    classify_error,
    execute_action,
    inventory_delta,
    sample_action,
    verify_effect,
)
from fle.rl.world import WorldClient

TOOL_TIMEOUT_S = 120
VOCAB_PATH = Path(__file__).parents[2] / "data/rl/vocab/factorio-2.0.73-base-v1.json"
SEEDED_INVENTORY = {
    "coal": 20,
    "iron-ore": 20,
    "iron-plate": 20,
    "stone": 10,
    "transport-belt": 10,
    "burner-inserter": 4,
    "small-electric-pole": 4,
    "pipe": 10,
    "wooden-chest": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=27000)
    parser.add_argument("--regime", choices=("naive", "macro"), required=True)
    parser.add_argument("--fixture", choices=("empty", "seeded"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--speed", type=float, default=10)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def snapshot(world: WorldClient, namespace: Any) -> OperationSnapshot:
    inventory = world.inventory()
    position = world.read_player_pos()
    tick = world.read_tick()
    score_player, score_automated = namespace.score()
    return OperationSnapshot(
        inventory=inventory,
        entities=dict(world.entities),
        position=position,
        tick=tick,
        score_player=score_player,
        score_automated=score_automated,
        current_research=world.research,
    )


def call_with_timeout(function: Callable[[], Any]) -> Any:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="op-census")
    future = executor.submit(function)
    try:
        return future.result(timeout=TOOL_TIMEOUT_S)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutError(f"tool call timed out after {TOOL_TIMEOUT_S}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def apply_seeded_fixture(world: WorldClient, namespace: Any) -> None:
    iron_patches = [patch for patch in world.patches() if patch.name == "iron-ore"]
    if not iron_patches:
        raise RuntimeError("Seeded fixture requires a known iron-ore patch")
    px, py = world.read_player_pos()
    patch = min(
        iron_patches,
        key=lambda candidate: math_distance(candidate.nearest_tile(px, py), (px, py)),
    )
    drill_x, drill_y = patch.nearest_tile(px, py)
    lua = f"""
local surface=game.surfaces[1]
local specs={{
  {{name='stone-furnace',position={{3,0}}}},
  {{name='wooden-chest',position={{3,3}}}},
  {{name='assembling-machine-1',position={{-4,0}}}},
  {{name='transport-belt',position={{0,4}}}},
  {{name='transport-belt',position={{1,4}}}},
  {{name='transport-belt',position={{2,4}}}},
  {{name='burner-inserter',position={{3,-2}}}},
  {{name='burner-mining-drill',position={{{drill_x},{drill_y}}}}}
}}
local made=0
for _,spec in ipairs(specs) do
  local entity=surface.create_entity{{name=spec.name,position=spec.position,force='player',raise_built=true}}
  if entity then made=made+1 end
end
rcon.print(made)
"""
    world.rcon.send_command("/sc " + lua.strip())
    namespace._set_inventory(SEEDED_INVENTORY)
    world.all_drain()


def math_distance(a: tuple[int, int], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def sampler_state(
    world: WorldClient,
    current: OperationSnapshot,
    enabled_recipes: list[str],
    research_state: dict[str, dict[str, Any]],
    vocab: VocabData,
    crafting_categories: list[str],
) -> SamplerState:
    return SamplerState(
        world=world,
        player_pos=current.position,
        inventory=current.inventory,
        enabled_recipes=enabled_recipes,
        research_state=research_state,
        vocab=vocab,
        crafting_categories=tuple(crafting_categories),
    )


def base_record(
    *,
    run_id: str,
    args: argparse.Namespace,
    episode: int,
    step: int,
    op: str,
    action_args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "regime": args.regime,
        "fixture": args.fixture,
        "seed": args.seed,
        "episode": episode,
        "step": step,
        "op": op,
        "args": action_args,
    }


def add_snapshot_fields(
    record: dict[str, Any],
    before: OperationSnapshot,
    after: OperationSnapshot,
) -> None:
    record.update(
        ticks_before=before.tick,
        ticks_after=after.tick,
        player_pos_before=list(before.position),
        player_pos_after=list(after.position),
        inventory_after=after.inventory,
        items_delta=inventory_delta(before.inventory, after.inventory),
        n_entities_after=len(after.entities),
        score_player=after.score_player,
        score_automated=after.score_automated,
    )


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise SystemExit("--episodes and --steps must be positive")
    rng = random.Random(args.seed)
    vocab = VocabData.load(VOCAB_PATH)
    run_id = uuid.uuid4().hex
    ok_counts = {op: 0 for op in OPS}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    instance = FactorioInstance(
        address="localhost",
        tcp_port=args.port,
        fast=True,
        all_technologies_researched=False,
        inventory={},
        reset_speed=args.speed,
    )
    namespace = instance.namespace
    world = WorldClient(instance.rcon_client, namespace)
    instance.set_speed_and_unpause(args.speed)
    reach = world.read_reach()
    crafting_categories = world.read_crafting_categories()
    world.terrain_full_sync()

    with args.out.open("w", buffering=1) as output:
        total_step = 0
        for episode in range(args.episodes):
            instance.reset(
                reset_position=True,
                all_technologies_researched=False,
                clear_entities=True,
            )
            instance.set_speed_and_unpause(args.speed)
            world.entity_full_sync()
            if args.fixture == "seeded":
                apply_seeded_fixture(world, namespace)
            enabled_recipes = world.read_enabled_recipes()
            research_state = world.read_research_state()
            current = snapshot(world, namespace)

            for step in range(args.steps):
                total_step += 1
                op = rng.choice(OPS)
                sampled = sample_action(
                    op,
                    args.regime,
                    sampler_state(world, current, enabled_recipes, research_state, vocab, crafting_categories),
                    rng,
                )
                action_args = sampled.args if isinstance(sampled, ActionSpec) else {}
                record = base_record(
                    run_id=run_id,
                    args=args,
                    episode=episode,
                    step=step,
                    op=op,
                    action_args=action_args,
                )
                if isinstance(sampled, NoSupport):
                    record.update(
                        status="no_support",
                        reason_class=sampled.reason,
                        raw_error="",
                        wall_tool_s=0.0,
                        wall_observe_s=0.0,
                        target_distance=None,
                    )
                    add_snapshot_fields(record, current, current)
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    if total_step % 25 == 0:
                        counts = " ".join(f"{name}={ok_counts[name]}" for name in OPS)
                        print(f"step {total_step}: ok {counts}", flush=True)
                    continue

                observe_start = time.perf_counter()
                before = snapshot(world, namespace)
                wall_observe = time.perf_counter() - observe_start
                tool_start = time.perf_counter()
                status = "no_effect"
                reason_class = "predicate_false"
                raw_error = ""
                execution_result = None
                harness_error = False
                try:
                    execution_result = call_with_timeout(
                        lambda action=sampled: execute_action(
                            action,
                            args.regime,
                            namespace,
                            world,
                            float(reach["resource_reach"]),
                        )
                    )
                except TimeoutError as exc:
                    status, reason_class, raw_error = "error", "timeout", str(exc)[:200]
                    harness_error = True
                except AnchorResolutionError as exc:
                    status = "error"
                    reason_class = "anchor_resolution_failed"
                    raw_error = str(exc)[:200]
                    harness_error = True
                except Exception as exc:  # noqa: BLE001 - tool clients use generic Exception
                    status = "tool_rejected"
                    raw_error = str(exc)[:200]
                    reason_class = classify_error(op, raw_error)
                wall_tool = time.perf_counter() - tool_start

                observe_start = time.perf_counter()
                try:
                    world.all_drain()
                    if harness_error:
                        world.entity_full_sync()
                    after = snapshot(world, namespace)
                    if execution_result and execution_result.status:
                        status = execution_result.status
                        reason_class = execution_result.reason_class or status
                    elif status not in {"error", "tool_rejected"}:
                        ok, effect = verify_effect(sampled, before, after)
                        record["effect"] = effect
                        status = "ok" if ok else "no_effect"
                        reason_class = "effect_verified" if ok else "predicate_false"
                except Exception as exc:  # noqa: BLE001 - observation failures must be logged
                    status = "error"
                    reason_class = "observe_error"
                    trace_lines = traceback.format_exception_only(type(exc), exc)
                    raw_error = trace_lines[-1].strip()[:200]
                    try:
                        world.entity_full_sync()
                        after = snapshot(world, namespace)
                    except Exception:  # noqa: BLE001 - retain the last valid snapshot
                        after = before
                wall_observe += time.perf_counter() - observe_start
                current = after

                if world.techs_finished:
                    enabled_recipes = world.read_enabled_recipes()
                    research_state = world.read_research_state()
                    world.techs_finished.clear()

                target_distance = None
                if args.regime == "macro" and op in {"MOVE", "HARVEST"}:
                    if execution_result is not None:
                        target_distance = execution_result.target_distance
                    if target_distance is None and "target" in sampled.args:
                        target_distance = math_distance(
                            (sampled.args["target"][0], sampled.args["target"][1]),
                            after.position,
                        )
                record.update(
                    status=status,
                    reason_class=reason_class,
                    raw_error=raw_error,
                    wall_tool_s=wall_tool,
                    wall_observe_s=wall_observe,
                    target_distance=target_distance,
                )
                add_snapshot_fields(record, before, after)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                if status == "ok":
                    ok_counts[op] += 1
                if total_step % 25 == 0:
                    counts = " ".join(f"{name}={ok_counts[name]}" for name in OPS)
                    print(f"step {total_step}: ok {counts}", flush=True)


if __name__ == "__main__":
    main()
