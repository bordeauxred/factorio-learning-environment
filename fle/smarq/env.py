"""Live semi-Markov semantic environment for Factorio."""

from __future__ import annotations

import argparse
import collections
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from fle.env import FactorioInstance
from fle.env.entities import Position
from fle.smarq import contract as C
from fle.smarq.actions import ActionCodec, MaskMetadata, build_masks
from fle.smarq.contract import Action, Masks, Observation, StepResult
from fle.smarq.obs import TensorClient
from fle.smarq.raster import ExactTileRaster
from fle.smarq.sim import MEASURED_EXECUTION_SPEED, ClockSample, SimClock
from fle.smarq.vocab import StableVocab, load_vocab

DEFAULT_INVENTORY = {
    "coal": 50,
    "copper-plate": 50,
    "iron-plate": 50,
    "iron-chest": 2,
    "burner-mining-drill": 3,
    "electric-mining-drill": 1,
    "assembling-machine-1": 1,
    "stone-furnace": 9,
    "transport-belt": 50,
    "boiler": 1,
    "burner-inserter": 32,
    "pipe": 15,
    "steam-engine": 1,
    "small-electric-pole": 10,
}


@dataclass
class ExecutionOutcome:
    success: bool
    failure_reason: str = "ok"
    requested_quantity: int | str | None = None
    executed_quantity: int | None = None
    overshoot_ticks: int = 0
    raw_error: str = ""


def classify_failure(error: BaseException | str, verb: str) -> str:
    text = str(error).lower()
    if any(token in text for token in ("path not found", "could not get path", "too far", "move closer")):
        return "unreachable"
    if verb == "PLACE" and any(
        token in text
        for token in ("blocked", "existing object", "already exists", "water", "cannot place", "suitable position")
    ):
        return "blocked"
    if any(
        token in text
        for token in (
            "no item",
            "in inventory",
            "ingredients",
            "inventory is full",
            "not enough",
            "do not have",
            "does not have",
        )
    ):
        return "insufficient_inventory"
    if any(
        token in text
        for token in ("invalid", "doesn't exist", "isn't something", "cannot be crafted", "already researched")
    ):
        return "invalid_argument"
    return "tool_error"


class FLEActionExecutor:
    """Execute semantic actions with the selected world coordinate unchanged."""

    def __init__(self, instance: Any, clock: SimClock, client: TensorClient) -> None:
        self.instance = instance
        self.clock = clock
        self.client = client
        self.rcon = instance.rcon_client

    def _command(self, lua: str) -> str:
        return self.rcon.send_command("/sc " + lua) or ""

    def _modeled_ticks(self) -> int:
        return int(self._command("rcon.print(storage.elapsed_ticks or 0)"))

    def _controller(self, name: str, *args: Any) -> Any:
        response, _ = self.instance.controllers[name].execute(*args)
        if isinstance(response, str):
            raise RuntimeError(response)
        return response

    def _charge_modeled(self, modeled: int, already_run: int, remaining: int) -> ClockSample:
        additional = min(max(0, modeled - already_run), max(0, remaining))
        return self.clock.advance_ticks(additional)

    def _interaction_position(self, tile: tuple[int, int], exact_destination: bool) -> Position:
        if exact_destination:
            return Position(x=tile[0], y=tile[1])
        px, py = self.client.player_x, self.client.player_y
        tx, ty = float(tile[0]), float(tile[1])
        dx, dy = px - tx, py - ty
        distance = math.hypot(dx, dy)
        if distance <= max(1.0, self.client.build_distance - 1.0):
            return Position(x=px, y=py)
        offset = min(3.0, max(1.0, self.client.build_distance - 1.0))
        return Position(x=tx + dx / distance * offset, y=ty + dy / distance * offset)

    def _navigate(self, tile: tuple[int, int], remaining: int, exact_destination: bool = False) -> int:
        destination = self._interaction_position(tile, exact_destination)
        if math.hypot(destination.x - self.client.player_x, destination.y - self.client.player_y) < 0.3:
            return 0
        before_modeled = self._modeled_ticks()

        def move() -> Any:
            start = Position(x=self.client.player_x, y=self.client.player_y)
            request = self.instance.controllers["request_path"](
                start=start,
                finish=destination,
                allow_paths_through_own_entities=True,
                resolution=-1,
            )
            self.instance.controllers["get_path"](request)
            controller = self.instance.controllers["move_to"]
            return self._controller("move_to", controller.player_index, request, "nil", "nil")

        response, running = self.clock.run_while_executing(move)
        if not isinstance(response, dict) or "x" not in response or "y" not in response:
            raise RuntimeError(f"navigation failed: {response}")
        self.client.player_x = float(response["x"])
        self.client.player_y = float(response["y"])
        modeled = self._modeled_ticks() - before_modeled
        charged = self._charge_modeled(modeled, running.duration_ticks, remaining - running.duration_ticks)
        return running.duration_ticks + charged.duration_ticks

    def _entity(self, action: Action) -> tuple[int, str, float, float] | None:
        unit = action.entity_id
        if not unit and action.entity_slot is not None and 0 <= action.entity_slot < self.client.entity_ids.size:
            unit = int(self.client.entity_ids[action.entity_slot])
        if not unit:
            return None
        response = self._command(
            f"local e=storage.obs_diff and storage.obs_diff.ents[{int(unit)}] "
            "if e and e.valid then rcon.print(e.unit_number..','..e.name..','.."
            "e.position.x..','..e.position.y) else rcon.print('missing') end"
        )
        if response == "missing":
            return None
        unit_text, name, x, y = response.split(",", 3)
        if int(unit_text) != int(unit):
            return None
        return int(unit), name, float(x), float(y)

    def _all_quantity(self, action: Action, entity: tuple[int, str, float, float] | None) -> int:
        if action.verb == "MINE":
            x, y = action.tile
            response = self._command(
                f"local e=game.surfaces[1].find_entities_filtered{{position={{x={x},y={y}}},"
                "type='resource'}[1] rcon.print(e and math.floor(e.amount) or 0)"
            )
        elif action.verb == "CRAFT":
            response = self._command(
                "local c=storage.utils.ensure_valid_character(1) "
                f"rcon.print(c.get_craftable_count('{action.recipe}'))"
            )
        elif action.verb == "INSERT":
            response = self._command(
                "local c=storage.utils.ensure_valid_character(1) "
                f"rcon.print(c.get_item_count('{action.item}'))"
            )
        elif action.verb == "EXTRACT" and entity is not None:
            response = self._command(
                f"local e=storage.obs_diff.ents[{entity[0]}] "
                f"rcon.print(e and e.valid and e.get_item_count('{action.item}') or 0)"
            )
        else:
            response = "0"
        return max(0, int(float(response)))

    def _run_tool(self, name: str, args: tuple[Any, ...], remaining: int) -> tuple[Any, int, int]:
        before_modeled = self._modeled_ticks()
        controller = self.instance.controllers[name]
        response, running = self.clock.run_while_executing(
            lambda: self._controller(name, controller.player_index, *args)
        )
        modeled = self._modeled_ticks() - before_modeled
        charged = self._charge_modeled(modeled, running.duration_ticks, remaining - running.duration_ticks)
        return response, running.duration_ticks + charged.duration_ticks, charged.overshoot_ticks

    def execute(self, action: Action, remaining_ticks: int) -> ExecutionOutcome:
        requested = action.quantity
        executed: int | None = None
        overshoot = 0
        try:
            entity = None
            if action.verb in {"PICKUP", "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE"}:
                entity = self._entity(action)
                if entity is None:
                    return ExecutionOutcome(False, "no_such_entity", requested, 0, 0)

            spent = 0
            if action.verb in {"MOVE_TO", "MINE", "PLACE"}:
                spent = self._navigate(
                    action.tile,
                    remaining_ticks,
                    exact_destination=action.verb == "MOVE_TO",
                )
            elif entity is not None:
                spent = self._navigate(
                    (math.floor(entity[2]), math.floor(entity[3])),
                    remaining_ticks,
                )

            remaining = max(0, remaining_ticks - spent)
            quantity = action.quantity
            if quantity == "ALL":
                quantity = self._all_quantity(action, entity)
            if requested is not None:
                executed = 0
                if int(quantity) <= 0:
                    raise RuntimeError("no acceptable quantity")

            if action.verb == "MOVE_TO":
                return ExecutionOutcome(True, "ok", requested, executed, 0)
            if action.verb == "PLACE":
                if action.prototype == "offshore-pump":
                    # The current FLE controller changes that prototype's target
                    # even when exact=True, so reject it without touching world state.
                    raise ValueError("invalid exact placement for offshore-pump")
                response, _, overshoot = self._run_tool(
                    "place_entity",
                    (
                        action.prototype,
                        C.DIRECTION_TO_FACTORIO[action.direction],
                        action.tile[0],
                        action.tile[1],
                        True,
                    ),
                    remaining,
                )
                position = response.get("position", {}) if isinstance(response, dict) else {}
                placed = (float(position.get("x", math.nan)), float(position.get("y", math.nan)))
                if placed != (float(action.tile[0]), float(action.tile[1])):
                    raise RuntimeError(f"exact placement returned {placed}, requested {action.tile}")
            elif action.verb == "MINE":
                response, _, overshoot = self._run_tool(
                    "harvest_resource",
                    (action.tile[0], action.tile[1], int(quantity), 0.75),
                    remaining,
                )
                executed = int(response)
            elif action.verb == "CRAFT":
                response, _, overshoot = self._run_tool(
                    "craft_item", (action.recipe, int(quantity)), remaining
                )
                executed = int(response)
            elif action.verb == "PICKUP":
                self._run_tool("pickup_entity", (entity[2], entity[3], entity[1]), remaining)
            elif action.verb == "ROTATE":
                self._run_tool(
                    "rotate_entity",
                    (entity[2], entity[3], C.DIRECTION_TO_FACTORIO[action.direction], entity[1]),
                    remaining,
                )
            elif action.verb == "INSERT":
                before = int(
                    self._command(
                        "local c=storage.utils.ensure_valid_character(1) "
                        f"rcon.print(c.get_item_count('{action.item}'))"
                    )
                )
                self._run_tool(
                    "insert_item",
                    (action.item, int(quantity), entity[2], entity[3], entity[1]),
                    remaining,
                )
                after = int(
                    self._command(
                        "local c=storage.utils.ensure_valid_character(1) "
                        f"rcon.print(c.get_item_count('{action.item}'))"
                    )
                )
                executed = max(0, before - after)
            elif action.verb == "EXTRACT":
                response, _, overshoot = self._run_tool(
                    "extract_item",
                    (action.item, int(quantity), entity[2], entity[3], entity[1]),
                    remaining,
                )
                executed = int(response)
            elif action.verb == "SET_RECIPE":
                self._run_tool(
                    "set_entity_recipe", (action.recipe, entity[2], entity[3]), remaining
                )
            elif action.verb == "RESEARCH":
                self._run_tool("set_research", (action.technology,), remaining)
            else:
                raise ValueError(f"unsupported verb {action.verb}")
            if requested is not None and executed is None:
                executed = int(quantity)
            return ExecutionOutcome(True, "ok", requested, executed, overshoot)
        except Exception as error:
            return ExecutionOutcome(
                False,
                classify_failure(error, action.verb),
                requested,
                executed or 0,
                overshoot,
                str(error)[:300],
            )


class SemanticEnv(C.SemanticEnvProtocol):
    """Contract environment backed by one live Factorio server."""

    def __init__(
        self,
        port: int = 27000,
        raster_tiles: int = C.RASTER_TILES_DEFAULT,
        tick_budget: int = C.EPISODE_TICK_BUDGET_DEFAULT,
        decision_cap: int = C.EPISODE_DECISION_CAP_DEFAULT,
        reward_mode: str = "automated",
        execution_speed: float = MEASURED_EXECUTION_SPEED,
        instance: Any | None = None,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError(f"port {port} is not a valid TCP port")
        if reward_mode not in C.REWARD_MODES:
            raise ValueError(reward_mode)
        self.port = port
        self.raster_tiles = int(raster_tiles)
        self.tick_budget = int(tick_budget)
        self.decision_cap = int(decision_cap)
        self.reward_mode = reward_mode
        self.instance = instance or FactorioInstance(
            address="localhost",
            tcp_port=port,
            fast=True,
            cache_scripts=True,
            all_technologies_researched=False,
            inventory=dict(DEFAULT_INVENTORY),
            reset_speed=execution_speed,
        )
        self._owns_instance = instance is None
        self.clock = SimClock(self.instance, execution_speed=execution_speed)
        self._vocab = load_vocab(self.instance.rcon_client)
        self.codec = ActionCodec(self._vocab)
        self.client = TensorClient(vocab=self._vocab)
        self.raster_builder = ExactTileRaster(self.raster_tiles)
        self.executor = FLEActionExecutor(self.instance, self.clock, self.client)
        self.metadata = MaskMetadata()
        self.episode_start_tick = 0
        self.episode_decisions = 0
        self._score = (0.0, 0.0)
        self._closed = False

    @property
    def vocab(self) -> StableVocab:
        return self._vocab

    @property
    def episode_ticks(self) -> int:
        return max(0, self.clock.tick() - self.episode_start_tick)

    def _observe(self, full: bool = False) -> Observation:
        self.clock.boundary()
        if full:
            self.client.full_sync(self.instance.rcon_client)
        else:
            self.client.drain(self.instance.rcon_client)
        distance = self.instance.rcon_client.send_command(
            "/sc local c=storage.utils.ensure_valid_character(1) rcon.print(c.build_distance)"
        )
        if distance:
            self.client.build_distance = float(distance)
        grid, globals_, entity_view, entity_mask, entity_ids = self.client.observation()
        raster, origin, player_tile = self.raster_builder.build(self.client)
        production, automated = self.instance.namespace.score()
        self._score = (float(production), float(automated))
        tick = self.clock.boundary()
        return Observation(
            grid=grid.copy(),
            entity_view=entity_view.copy(),
            entity_mask=entity_mask.copy(),
            entity_ids=entity_ids.copy(),
            globals=globals_.copy(),
            raster=raster,
            raster_origin=origin,
            raster_tiles=self.raster_tiles,
            player_tile=player_tile,
            tick=tick,
            production_score=self._score[0],
            automated_production_score=self._score[1],
        )

    def masks(self) -> Masks:
        self.metadata = MaskMetadata.from_rcon(self.instance.rcon_client, self._vocab)
        return build_masks(self._vocab, self.client, self.metadata)

    def reset(self, seed: int | None = None) -> tuple[Observation, Masks]:
        del seed
        self.instance.reset(
            reset_position=True,
            all_technologies_researched=False,
            clear_entities=True,
        )
        self.episode_decisions = 0
        self.episode_start_tick = self.clock.boundary()
        observation = self._observe(full=True)
        return observation, self.masks()

    def step(self, action: Action) -> tuple[StepResult, Masks]:
        wall_start = time.perf_counter()
        start_tick = self.clock.boundary()
        before_production, before_automated = self.instance.namespace.score()
        elapsed_before = max(0, start_tick - self.episode_start_tick)
        remaining = max(0, self.tick_budget - elapsed_before)
        if action.verb == "FAST_FORWARD":
            requested_ticks = int(action.duration_seconds) * C.TICKS_PER_SECOND
            sample = self.clock.advance_ticks(min(requested_ticks, remaining))
            outcome = ExecutionOutcome(True, "ok", None, None, sample.overshoot_ticks)
        elif remaining == 0:
            outcome = ExecutionOutcome(False, "invalid_argument")
        else:
            outcome = self.executor.execute(action, remaining)
        end_tick = self.clock.boundary()
        self.episode_decisions += 1
        observation = self._observe()
        after_production = observation.production_score
        after_automated = observation.automated_production_score
        delta_production = after_production - float(before_production)
        delta_automated = after_automated - float(before_automated)
        reward = delta_automated if self.reward_mode == "automated" else delta_production
        duration_ticks = end_tick - start_tick
        episode_ticks = max(0, end_tick - self.episode_start_tick)
        out_of_time = episode_ticks >= self.tick_budget
        out_of_decisions = self.episode_decisions >= self.decision_cap
        done = out_of_time or out_of_decisions
        end_reason = "tick_budget" if out_of_time else "decision_cap" if out_of_decisions else None
        result = StepResult(
            observation=observation,
            reward=float(reward),
            done=done,
            duration_ticks=duration_ticks,
            duration_game_seconds=duration_ticks / C.TICKS_PER_SECOND,
            wall_seconds=time.perf_counter() - wall_start,
            success=outcome.success,
            failure_reason=outcome.failure_reason,
            production_score=after_production,
            automated_production_score=after_automated,
            delta_production_score=delta_production,
            delta_automated_production_score=delta_automated,
            episode_ticks=episode_ticks,
            episode_decisions=self.episode_decisions,
            end_reason=end_reason,
            info={
                "requested_quantity": outcome.requested_quantity,
                "executed_quantity": outcome.executed_quantity,
                "advance_overshoot_ticks": outcome.overshoot_ticks,
                # The game's own words. Without these every unrecognised refusal
                # collapses into `tool_error` and the logs cannot say why.
                "raw_error": outcome.raw_error,
            },
        )
        return result, self.masks()

    def close(self) -> None:
        if self._closed:
            return
        self.clock.close()
        if self._owns_instance:
            self.instance.cleanup()
        self._closed = True


def _random_action(
    rng: np.random.Generator,
    codec: ActionCodec,
    observation: Observation,
    masks: Masks,
) -> Action:
    candidates = []
    for verb_index, verb in enumerate(C.VERBS):
        if not masks.verb[verb_index]:
            continue
        if C.ENTITY in C.HEAD_SEQUENCE[verb] and not masks.entity[verb_index].any():
            continue
        candidates.append(verb_index)
    verb_index = int(rng.choice(candidates))
    verb = C.VERBS[verb_index]
    heads = C.empty_heads()
    for head in C.HEAD_SEQUENCE[verb]:
        if head == C.POSITION:
            half = max(2, observation.raster_tiles // 4)
            x = int(rng.integers(observation.raster_tiles // 2 - half, observation.raster_tiles // 2 + half))
            y = int(rng.integers(observation.raster_tiles // 2 - half, observation.raster_tiles // 2 + half))
            value = y * observation.raster_tiles + x
        elif head == C.PROTOTYPE:
            value = int(rng.choice(np.nonzero(masks.prototype)[0]))
        elif head == C.DIRECTION:
            value = int(rng.integers(C.N_DIRECTIONS))
        elif head == C.ENTITY:
            value = int(rng.choice(np.nonzero(masks.entity[verb_index])[0]))
        elif head == C.ITEM:
            value = int(rng.choice(np.nonzero(masks.item)[0]))
        elif head == C.QUANTITY:
            value = int(rng.integers(C.N_QUANTITIES))
        elif head == C.RECIPE:
            legal = masks.craft_recipe
            if verb == "SET_RECIPE":
                legal = masks.recipe_for_entity(int(heads[C.HEAD_INDEX[C.ENTITY]]))
            if legal is None or not legal.any():
                return _random_action(rng, codec, observation, masks)
            value = int(rng.choice(np.nonzero(legal)[0]))
        elif head == C.TECHNOLOGY:
            value = int(rng.choice(np.nonzero(masks.technology)[0]))
        else:
            value = int(rng.integers(C.N_DURATIONS))
        heads[C.HEAD_INDEX[head]] = value
    return codec.decode(verb_index, heads, observation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=27000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--decisions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--raster-tiles", type=int, default=C.RASTER_TILES_DEFAULT)
    parser.add_argument("--reward-mode", choices=C.REWARD_MODES, default="automated")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    env = SemanticEnv(
        port=args.port,
        raster_tiles=args.raster_tiles,
        reward_mode=args.reward_mode,
    )
    try:
        for episode in range(args.episodes):
            observation, masks = env.reset(args.seed + episode)
            paused_before = env.clock.boundary()
            time.sleep(0.05)
            paused_after = env.clock.tick()
            histogram: collections.Counter[str] = collections.Counter()
            wall_start = time.perf_counter()
            result = None
            for _ in range(args.decisions):
                action = _random_action(rng, env.codec, observation, masks)
                result, masks = env.step(action)
                observation = result.observation
                histogram[result.failure_reason] += 1
                if result.done:
                    break
            wall = time.perf_counter() - wall_start
            ticks = result.episode_ticks if result is not None else 0
            production = result.production_score if result is not None else observation.production_score
            automated = (
                result.automated_production_score
                if result is not None
                else observation.automated_production_score
            )
            game_seconds = ticks / C.TICKS_PER_SECOND
            print(
                f"episode={episode + 1} decisions={sum(histogram.values())} "
                f"simulated_ticks={ticks} wall_seconds={wall:.3f} "
                f"game_seconds_per_wall_second={game_seconds / max(wall, 1e-9):.3f} "
                f"production_score={production:.6f} automated_production_score={automated:.6f} "
                f"failure_reasons={dict(sorted(histogram.items()))} "
                f"paused_tick_delta={paused_after - paused_before}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
