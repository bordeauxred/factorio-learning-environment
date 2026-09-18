"""TOY-B: run the scripted burner-automation demonstration on a live server.

This is the swim/drown diagnostic for the action interface itself.  If a script
that uses only the semantic grammar - exact coordinates, no build planner, no
candidate generation - cannot get a drill and a furnace producing, then no
policy can, and the experiment is about the interface rather than the learner.

Every step prints what was requested, what the game did, the simulated duration,
and the raw failure text, because the raw text is what tells us whether a
failure is the agent's fault or ours.

    .venv/bin/python tests/benchmarks/smarq_live_demo.py --port 27004
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from fle.smarq import contract as C
from fle.smarq.env import SemanticEnv


def find_resource_tiles(env: SemanticEnv, name: str, limit: int = 6) -> list[tuple[int, int]]:
    """Exact tiles of a named resource, read from the live world.

    A demonstration script may look at the world to choose its own coordinates;
    what it may not do - and does not do - is hand the policy a shortlist of
    good build positions.  This is used only to write the script.
    """
    lua = (
        "/sc local ch = storage.agent_characters[1] "
        "if not ch then rcon.print('') return end "
        f"local r = game.surfaces[1].find_entities_filtered{{name='{name}', "
        "position=ch.position, radius=400, limit=400} "
        "local out = {} "
        "for _, e in pairs(r) do out[#out+1] = math.floor(e.position.x) .. ',' .. math.floor(e.position.y) end "
        "rcon.print(table.concat(out, ';'))"
    )
    raw = env.instance.rcon_client.send_command(lua) or ""
    tiles: list[tuple[int, int]] = []
    for part in raw.split(";"):
        if "," in part:
            x, y = part.split(",")[:2]
            tiles.append((int(x), int(y)))
    px, py = character_position(env)
    tiles = sorted(set(tiles), key=lambda t: (t[0] - px) ** 2 + (t[1] - py) ** 2)
    return tiles[:limit]


def character_position(env: SemanticEnv) -> tuple[float, float]:
    raw = env.instance.rcon_client.send_command(
        "/sc local ch = storage.agent_characters[1] "
        "rcon.print(ch and (ch.position.x .. ',' .. ch.position.y) or '0,0')"
    )
    x, y = (raw or "0,0").split(",")[:2]
    return float(x), float(y)


def teleport_character(env: SemanticEnv, tile: tuple[int, int]) -> None:
    """Lab setup for TOY-B: start the episode next to ore.

    This is a task definition, not an action-space shortcut: the agent still has
    to choose every intervention and every coordinate itself.
    """
    env.instance.rcon_client.send_command(
        "/sc local ch = storage.agent_characters[1] "
        f"if ch then ch.teleport({{{tile[0]}, {tile[1]}}}) end "
        "rcon.print('ok')"
    )


def act(env: SemanticEnv, verb: str, **kwargs) -> tuple[C.StepResult, C.Action]:
    """Build a contract Action from decoded arguments and execute it."""
    obs = env._observe(full=False) if hasattr(env, "_observe") else None
    heads = C.empty_heads()
    vocab = env.vocab
    if "tile" in kwargs:
        origin = obs.raster_origin if obs is not None else (0, 0)
        tiles = obs.raster_tiles if obs is not None else env.raster_tiles
        pos = C.tile_to_position(kwargs["tile"], origin, tiles)
        if pos is None:
            raise SystemExit(
                f"{verb} target {kwargs['tile']} lies outside the {tiles}-tile window at {origin}; "
                "the script must MOVE_TO closer first"
            )
        heads[C.HEAD_INDEX[C.POSITION]] = pos
    if "prototype" in kwargs:
        heads[C.HEAD_INDEX[C.PROTOTYPE]] = list(vocab.prototypes).index(kwargs["prototype"])
    if "direction" in kwargs:
        heads[C.HEAD_INDEX[C.DIRECTION]] = C.DIRECTIONS.index(kwargs["direction"])
    if "entity_slot" in kwargs:
        heads[C.HEAD_INDEX[C.ENTITY]] = kwargs["entity_slot"]
    if "item" in kwargs:
        heads[C.HEAD_INDEX[C.ITEM]] = list(vocab.items).index(kwargs["item"])
    if "quantity" in kwargs:
        heads[C.HEAD_INDEX[C.QUANTITY]] = C.QUANTITIES.index(kwargs["quantity"])
    if "recipe" in kwargs:
        heads[C.HEAD_INDEX[C.RECIPE]] = list(vocab.recipes).index(kwargs["recipe"])
    if "duration_seconds" in kwargs:
        heads[C.HEAD_INDEX[C.DURATION]] = C.DURATIONS_SECONDS.index(kwargs["duration_seconds"])
    action = C.Action(verb=verb, heads=heads, **kwargs)
    result, _ = env.step(action)
    raw = result.info.get("raw_error") or result.info.get("error") or ""
    print(
        f"  {verb:<12} {str(kwargs):<70.70} -> {'ok' if result.success else result.failure_reason:<22}"
        f" tau={result.duration_ticks:>6}t"
        f" auto={result.automated_production_score:>8.1f} (d={result.delta_automated_production_score:+.1f})"
        f" prod={result.production_score:>7.1f}"
        + (f"\n      raw: {str(raw)[:200]}" if raw else "")
    )
    return result, action


def entity_slot_at(env: SemanticEnv, tile: tuple[int, int]) -> int | None:
    """Which observation row holds the entity standing on this exact tile."""
    obs = env._observe(full=False)
    for slot in range(len(obs.entity_mask)):
        if not obs.entity_mask[slot]:
            continue
        ex = obs.entity_view[slot, 0] + obs.player_tile[0]
        ey = obs.entity_view[slot, 1] + obs.player_tile[1]
        if abs(ex - tile[0]) < 1.5 and abs(ey - tile[1]) < 1.5:
            return slot
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=27004)
    parser.add_argument("--raster-tiles", type=int, default=C.RASTER_TILES_DEFAULT)
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument(
        "--teleport-to-ore",
        action="store_true",
        help="TOY-B lab setup: start the character beside the nearest ore patch",
    )
    args = parser.parse_args()

    wall0 = time.perf_counter()
    env = SemanticEnv(port=args.port, raster_tiles=args.raster_tiles)
    obs, _ = env.reset(0)
    print(f"reset: player={obs.player_tile} tick={obs.tick} "
          f"scores=(prod {obs.production_score}, auto {obs.automated_production_score})")

    iron = find_resource_tiles(env, "iron-ore")
    coal = find_resource_tiles(env, "coal")
    print(f"nearest iron-ore tiles: {iron[:3]}")
    print(f"nearest coal tiles:     {coal[:3]}")
    if not iron:
        raise SystemExit("no iron ore within 200 tiles; nothing to demonstrate")

    target = iron[0]
    if args.teleport_to_ore:
        teleport_character(env, (target[0] + 4, target[1] + 4))
        env.client.obs_all_drain() if hasattr(env.client, "obs_all_drain") else None
        obs = env._observe(full=True)
        print(f"teleported beside ore: player={obs.player_tile}, ore at {target}")

    print("\n-- walking to ore, in window-sized hops --")
    guard = 0
    while guard < 6:
        obs = env._observe(full=False)
        half = obs.raster_tiles // 2 - 2
        dx, dy = target[0] - obs.player_tile[0], target[1] - obs.player_tile[1]
        if abs(dx) <= half and abs(dy) <= half:
            break
        hop = (
            obs.player_tile[0] + int(np.clip(dx, -half, half)),
            obs.player_tile[1] + int(np.clip(dy, -half, half)),
        )
        act(env, "MOVE_TO", tile=hop)
        guard += 1

    print("\n-- placing a burner mining drill on the exact ore tile --")
    act(env, "MOVE_TO", tile=(target[0], target[1] + 2))
    drill, _ = act(env, "PLACE", prototype="burner-mining-drill", tile=target, direction="SOUTH")
    slot = entity_slot_at(env, target)
    print(f"      drill entity slot: {slot}")
    if slot is not None:
        act(env, "INSERT", entity_slot=slot, item="coal", quantity=8)

    print("\n-- a furnace beside it, fuelled --")
    furnace_tile = (target[0], target[1] + 2)
    act(env, "PLACE", prototype="stone-furnace", tile=furnace_tile, direction="NORTH")
    fslot = entity_slot_at(env, furnace_tile)
    print(f"      furnace entity slot: {fslot}")
    if fslot is not None:
        act(env, "INSERT", entity_slot=fslot, item="coal", quantity=8)
        act(env, "INSERT", entity_slot=fslot, item="iron-ore", quantity=16)

    print(f"\n-- letting it run for {args.wait_seconds} simulated seconds --")
    before = env.instance.namespace.score()
    for _ in range(max(1, args.wait_seconds // 60)):
        act(env, "FAST_FORWARD", duration_seconds=60)
    after = env.instance.namespace.score()

    print("\n-- extracting what it made --")
    if fslot is not None:
        act(env, "EXTRACT", entity_slot=fslot, item="iron-plate", quantity="ALL")

    summary = {
        "port": args.port,
        "production_before": before[0],
        "production_after": after[0],
        "automated_before": before[1],
        "automated_after": after[1],
        "automated_delta_over_wait": after[1] - before[1],
        "wall_seconds": round(time.perf_counter() - wall0, 1),
        "episode_ticks": env.clock.tick() - env.episode_start_tick,
    }
    print("\nSUMMARY " + json.dumps(summary))
    verdict = (
        "INTERFACE CAN AUTOMATE" if summary["automated_delta_over_wait"] > 0
        else "NO AUTOMATED SCORE - interface or script is wrong"
    )
    print(verdict)
    env.close()


if __name__ == "__main__":
    main()
