"""Learner-facing Factorio macro-action environment.

Run the live random-policy acceptance driver with::

    python -m fle.rl.env --port 27000 --episodes 3 --steps 128 --seed 1
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from fle.commons.models.game_state import GameState
from fle.env.entities import Position
from fle.env.game_types import prototype_by_name
from fle.env.instance import FactorioInstance
from fle.rl import schema as S
from fle.rl.observation import (
    HarvestTarget,
    MacroObservationBuilder,
    ObservationFrame,
    ObservationInput,
    available_technologies,
)
from fle.rl.ops import (
    ActionSpec,
    AnchorResolutionError,
    ExecutionResult,
    OperationSnapshot,
    VocabData,
    _approach_entity,
    classify_error,
    entity_is_rotatable,
    execute_action,
    extractable_items,
    furnace_input_item,
    insertable_items,
    inventory_delta,
    recursively_craftable_recipes,
    verify_effect,
)
from fle.rl.world import EntityRow, WorldClient

TOOL_TIMEOUT_S = 120
WAIT_WALL_CAP_S = 30.0
DELAYED_DRAIN_OPS = frozenset({"PICKUP", "SET_RECIPE", "CONNECT"})
DETERMINISTIC_NO_SUPPORT_COOLDOWNS = frozenset(
    {
        "insert_no_accepted_item",
        "insert_item_not_held_or_accepted",
        "extract_entity_has_no_items",
        "extract_item_not_present",
    }
)
ENTITY_OPS = frozenset({"PICKUP", "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE"})
DEFAULT_MAX_STEPS = 256
DEFAULT_MAX_TICKS = 216_000
PIPE_ENDPOINT_TYPES = frozenset(
    {
        "boiler",
        "generator",
        "offshore-pump",
        "pipe",
        "pipe-to-ground",
        "pump",
        "storage-tank",
    }
)
PIPE_ENDPOINT_NAMES = frozenset(
    {
        "assembling-machine-2",
        "assembling-machine-3",
        "chemical-plant",
        "electric-mining-drill",
        "oil-refinery",
        "pumpjack",
    }
)
BELT_SOURCE_BLOCKED_TYPES = frozenset(
    {
        "container",
        "furnace",
        "infinity-container",
        "linked-container",
        "logistic-container",
    }
)
MOVE_VECTORS: tuple[tuple[float, float], ...] = (
    (0.0, -1.0),
    (math.sqrt(0.5), -math.sqrt(0.5)),
    (1.0, 0.0),
    (math.sqrt(0.5), math.sqrt(0.5)),
    (0.0, 1.0),
    (-math.sqrt(0.5), math.sqrt(0.5)),
    (-1.0, 0.0),
    (-math.sqrt(0.5), -math.sqrt(0.5)),
)
PEACEFUL_RESET_COMMAND = (
    "/sc local surface=game.surfaces[1] surface.peaceful_mode=true "
    'for _,entity in pairs(surface.find_entities_filtered{force="enemy"}) do '
    "entity.destroy{raise_destroy=false} end"
)
V0_SNAPSHOT_COMMAND = (
    "/sc local c=storage.agent_characters[1] local valid=c and c.valid "
    "local inv={} if valid then local i=c.get_main_inventory() "
    "if i and i.valid then for name,count in pairs(storage.utils.get_contents_compat(i)) do "
    "inv[#inv+1]=name..'='..count end end end table.sort(inv) "
    "local q={} for _,t in pairs(game.forces.player.research_queue or {}) do "
    "q[#q+1]=t.name end table.sort(q) "
    "local r=game.forces.player.current_research "
    "local raw=storage.actions.score() or '' "
    "local player=tonumber(string.match(raw, '%[\"player\"%] = ([^,}]+)')) or 0 "
    "local automated=tonumber(string.match(raw, '%[\"automated\"%] = ([^,}]+)')) or 0 "
    "local out={tostring(game.tick),valid and '1' or '0',"
    "valid and tostring(c.position.x) or '',valid and tostring(c.position.y) or '',"
    "tostring(player),tostring(automated),r and r.name or '',"
    "table.concat(q,','),table.concat(inv,',')} "
    "rcon.print(table.concat(out,'\\t'))"
)


def _effective_max_ticks(max_steps: int, max_ticks: int) -> int:
    """Scale the default 256-step budget while preserving explicit custom caps."""
    if max_ticks != DEFAULT_MAX_TICKS:
        return max_ticks
    return max_ticks * max_steps // DEFAULT_MAX_STEPS


_TERRAIN_LOCK = threading.Lock()
_TERRAIN_CACHE: dict[str, Any] | None = None
_TRIGGER_TECH_CACHE: frozenset[str] | None = None


@dataclass(frozen=True)
class FrontierState:
    """A self-reached Factorio state and the macro-episode state around it."""

    game_state: GameState
    step_index: int
    max_steps: int
    episode_ticks: int
    aps_baseline: float
    inventory: dict[str, int]
    entity_hash: str


@dataclass(frozen=True)
class V0ObservationFrame:
    """Observation plus the identity-bearing rows behind its entity slots."""

    obs: dict[str, Any]
    entities: tuple[EntityRow | None, ...]


def _entity_identity_hash(entities: Mapping[int, EntityRow]) -> str:
    """Hash stable entity attributes; volatile contents and recipes are excluded."""
    identities = sorted(
        (
            row.name,
            float(row.x),
            float(row.y),
            int(row.direction),
        )
        for row in entities.values()
    )
    payload = json.dumps(identities, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def call_with_timeout(function: Callable[[], Any]) -> Any:
    """Run one tool/action call with the census harness's wall timeout."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fle-macro-tool")
    future = executor.submit(function)
    try:
        return future.result(timeout=TOOL_TIMEOUT_S)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutError(f"tool call timed out after {TOOL_TIMEOUT_S}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class FleMacroEnv(gym.Env):
    """One live FLE server exposed as a masked macro-action MDP.

    Every reset restores FLE's intended peaceful default and removes existing
    enemy entities because the open-world scenario does not honour the
    ``FactorioInstance(peaceful=True)`` default. Character death remains a
    terminal transition and is counted.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        port: int = 27000,
        speed: float = 40,
        max_steps: int = 256,
        max_ticks: int = 216_000,
        seed: int | None = None,
        log_path: str | Path | None = None,
        quantity_aware_support: bool = True,
        support_cooldowns: bool = True,
        clamp_quantity: bool | None = None,
        clamp_item: bool = False,
        clamp_anchor: bool = False,
        clamp_connect: bool = False,
        regime: str = "macro",
        action_grammar: str = "legacy",
        mask_place_location: bool = True,
        mask_place_item: bool = True,
        mask_inventory_item: bool = True,
        mask_contained_item: bool = True,
        mask_entity: bool = True,
        mask_recipe: bool = True,
        mask_technology: bool = True,
        mask_verb: bool = True,
        place_max_age_ticks: int = 600,
        v0_batch_snapshot: bool = True,
    ) -> None:
        if speed <= 0 or max_steps <= 0 or max_ticks <= 0:
            raise ValueError("speed, max_steps, and max_ticks must be positive")
        if regime not in {"macro", "bare"}:
            raise ValueError(f"Unknown action regime {regime}")
        if action_grammar not in {"legacy", "v0"}:
            raise ValueError(f"Unknown action grammar {action_grammar}")
        if place_max_age_ticks < 0:
            raise ValueError("place_max_age_ticks must be nonnegative")
        self.port = port
        self.speed = speed
        self.max_steps = max_steps
        self.max_ticks = _effective_max_ticks(max_steps, max_ticks)
        self.regime = regime
        self.action_grammar = action_grammar
        self.place_max_age_ticks = place_max_age_ticks
        self.v0_batch_snapshot = v0_batch_snapshot
        self.v0_mask_toggles = {
            "place_location": mask_place_location,
            "place_item": mask_place_item,
            "inventory_item": mask_inventory_item,
            "contained_item": mask_contained_item,
            "entity": mask_entity,
            "recipe": mask_recipe,
            "technology": mask_technology,
            "verb": mask_verb,
        }
        self.v0_mask_counters: Counter[str] = Counter()
        # ``quantity_aware_support`` remains a compatible constructor name.
        # New callers should use the execution-guard name used by the CLI.
        if clamp_quantity is None:
            clamp_quantity = quantity_aware_support
        self.clamp_quantity = clamp_quantity
        self.quantity_aware_support = clamp_quantity
        self.clamp_item = clamp_item
        self.clamp_anchor = clamp_anchor
        self.clamp_connect = clamp_connect
        self.support_cooldowns = support_cooldowns
        if action_grammar == "v0":
            # Lazy by design: package B's offline tests must not depend on the
            # concurrently built V0 observation module.
            from fle.rl import v0_schema as V0

            observation_spaces: dict[str, gym.Space] = {
                name: gym.spaces.Box(
                    low=np.iinfo(dtype).min
                    if np.issubdtype(dtype, np.integer)
                    else -np.inf,
                    high=np.iinfo(dtype).max
                    if np.issubdtype(dtype, np.integer)
                    else np.inf,
                    shape=shape,
                    dtype=dtype,
                )
                for name, (shape, dtype) in V0.OBS_SPEC.items()
            }
            observation_spaces["masks"] = gym.spaces.Dict(
                {
                    head: gym.spaces.MultiBinary(size)
                    for head, size in V0.HEAD_SIZES.items()
                }
            )
            self.observation_space = gym.spaces.Dict(observation_spaces)
            self.action_space = gym.spaces.MultiDiscrete(
                tuple(V0.HEAD_SIZES[head] for head in V0.HEADS)
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(S.OBS_SIZE,),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.MultiDiscrete(S.HEAD_DIMS)
        self.action_space.seed(seed)

        self.vocab = VocabData.load(S.VOCAB_PATH)
        raw_vocab = json.loads(S.VOCAB_PATH.read_text())
        self._status_names = {
            int(row["code"]): row["name"] for row in raw_vocab["statuses"]
        }
        self.instance = FactorioInstance(
            address="localhost",
            tcp_port=port,
            fast=True,
            all_technologies_researched=False,
            inventory={},
            reset_speed=speed,
        )
        self.namespace = self.instance.namespace
        self.world = WorldClient(self.instance.rcon_client, self.namespace)
        if action_grammar == "v0":
            self.world.configure_v0_caches(self.instance.rcon_client)
        self.instance.set_speed_and_unpause(speed)
        self._load_static_world()
        reach = self.world.read_reach()
        self.resource_reach = float(reach["resource_reach"])
        self.build_reach = float(reach["build"])
        self.crafting_categories = tuple(self.world.read_crafting_categories())
        self.trigger_technologies = self._load_trigger_technologies()
        if action_grammar != "v0":
            self.observer = MacroObservationBuilder(
                world=self.world,
                vocab=self.vocab,
                crafting_categories=self.crafting_categories,
                resource_reach=self.resource_reach,
                build_reach=self.build_reach,
                trigger_technologies=self.trigger_technologies,
                status_names=self._status_names,
                max_steps=max_steps,
                max_ticks=self.max_ticks,
                quantity_aware_support=clamp_quantity,
                regime=regime,
            )

        self._log_file = None
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("a", buffering=1)
        self._current: OperationSnapshot | None = None
        self._frame: ObservationFrame | V0ObservationFrame | None = None
        self._enabled_recipes: list[str] = []
        self._research_state: dict[str, dict[str, Any]] = {}
        self._episode_start_tick = 0
        self._steps = 0
        self._episode_max_steps = max_steps
        self._episode_index = -1
        self._episode_done = True
        self._end_reason: str | None = None
        self._last_status: str | None = None
        self._last_op: str | None = None
        self._aps_previous = 0.0
        self._max_aps = 0.0
        self._op_histogram: Counter[str] = Counter()
        self._status_histogram: Counter[str] = Counter()
        self._entities_placed = 0
        self._items_harvested = 0
        self._episode_deaths = 0
        self._excursion = False
        self._restore_hash: str | None = None
        self._excursion_steps = 0
        self._excursion_budget: int | None = None
        self._cooldown_action: tuple[int, ...] | None = None
        self._cooldown_fingerprint: tuple[Any, ...] | None = None
        self._cooldown_entity_unit: int | None = None
        self._cooldown_masked = False
        self.deaths = 0

    def _load_static_world(self) -> None:
        global _TERRAIN_CACHE
        with _TERRAIN_LOCK:
            if _TERRAIN_CACHE is None:
                self.world.terrain_full_sync()
                _TERRAIN_CACHE = {
                    "water": copy.deepcopy(self.world.water),
                    "ores": copy.deepcopy(self.world.ores),
                    "trees": copy.deepcopy(self.world.trees),
                    "obstacles": copy.deepcopy(self.world.obstacles),
                    "nests": copy.deepcopy(self.world.nests),
                }
            else:
                for name, value in _TERRAIN_CACHE.items():
                    setattr(self.world, name, copy.deepcopy(value))
                self.world._terrain_synced = True

    def _load_trigger_technologies(self) -> frozenset[str]:
        global _TRIGGER_TECH_CACHE
        with _TERRAIN_LOCK:
            if _TRIGGER_TECH_CACHE is None:
                _TRIGGER_TECH_CACHE = frozenset(self.world.read_research_triggers())
            return _TRIGGER_TECH_CACHE

    def _snapshot(
        self, fallback: OperationSnapshot | None = None
    ) -> tuple[OperationSnapshot, bool]:
        if getattr(self, "action_grammar", "legacy") == "v0" and getattr(
            self, "v0_batch_snapshot", True
        ):
            return self._snapshot_v0(fallback)
        character_valid = True
        try:
            position = self.world.read_player_pos()
        except Exception:
            if fallback is None:
                raise
            position = fallback.position
            character_valid = False
        try:
            inventory = call_with_timeout(self.world.inventory)
        except Exception:
            if fallback is None:
                raise
            inventory = fallback.inventory
        try:
            tick = self.world.read_tick()
        except Exception:
            if fallback is None:
                raise
            tick = fallback.tick
        try:
            score_player, score_automated = call_with_timeout(self.namespace.score)
        except Exception:
            if fallback is None:
                raise
            score_player = fallback.score_player
            score_automated = fallback.score_automated
        try:
            research_queue = tuple(self.world.read_research_queue())
        except Exception:
            if fallback is None:
                raise
            research_queue = fallback.research_queue
        snapshot = OperationSnapshot(
            inventory=dict(inventory),
            entities=dict(self.world.entities),
            position=position,
            tick=tick,
            score_player=float(score_player),
            score_automated=float(score_automated),
            current_research=self.world.research,
            research_queue=research_queue,
        )
        return snapshot, character_valid

    @staticmethod
    def _parse_v0_items(payload: str) -> dict[str, int]:
        if not payload:
            return {}
        result: dict[str, int] = {}
        for entry in payload.split(","):
            name, separator, count = entry.rpartition("=")
            if not separator:
                raise ValueError(f"invalid V0 inventory entry {entry!r}")
            result[name] = int(count)
        return result

    def _snapshot_v0(
        self, fallback: OperationSnapshot | None = None
    ) -> tuple[OperationSnapshot, bool]:
        """Fetch all scalar V0 step state in one compact RCON response."""
        started = time.perf_counter()
        try:
            response = call_with_timeout(
                lambda: self.instance.rcon_client.send_command(V0_SNAPSHOT_COMMAND)
            )
            # Preserve empty final fields (notably an empty inventory); RCON
            # may append a line terminator, but tabs are part of the payload.
            # Factorio's rcon.print strips trailing whitespace, so when the
            # last fields are all empty (no research, empty queue, empty
            # inventory) they vanish from the wire. Pad rather than fail.
            fields = str(response).rstrip("\r\n").split("\t", 8)
            if len(fields) > 9:
                raise ValueError(
                    f"V0 snapshot expected at most 9 fields, received {len(fields)}"
                )
            if len(fields) < 9:
                fields = fields + [""] * (9 - len(fields))
            (
                tick_text,
                valid_text,
                x_text,
                y_text,
                general_text,
                automated_text,
                research_text,
                queue_text,
                inventory_text,
            ) = fields
            character_valid = valid_text == "1"
            if not character_valid:
                if fallback is None:
                    raise RuntimeError("character is invalid")
                position = fallback.position
            else:
                position = (float(x_text), float(y_text))
            snapshot = OperationSnapshot(
                inventory=self._parse_v0_items(inventory_text),
                entities=dict(self.world.entities),
                position=position,
                tick=int(tick_text),
                score_player=float(general_text)
                - float(getattr(self.instance, "initial_score", 0) or 0),
                score_automated=float(automated_text),
                current_research=research_text or None,
                research_queue=tuple(name for name in queue_text.split(",") if name),
            )
        except Exception:
            if fallback is None:
                raise
            snapshot = fallback
            character_valid = False
        self._v0_last_snapshot_wall_s = time.perf_counter() - started
        return snapshot, character_valid

    def _v0_context(self, frame: V0ObservationFrame | None = None) -> dict[str, Any]:
        """Runtime context consumed by the server-independent V0 action package."""
        assert self._current is not None
        technologies = available_technologies(
            self._research_state,
            trigger_technologies=self.trigger_technologies,
            current_research=self._current.current_research,
            research_queue=self._current.research_queue,
        )
        return {
            "snapshot": self._current,
            "position": self._current.position,
            "tick": self._current.tick,
            "inventory_counts": self._current.inventory,
            "entity_rows": frame.entities if frame is not None else (),
            "vocab": self.vocab,
            "enabled_recipes": self._enabled_recipes,
            "research_state": self._research_state,
            "crafting_categories": self.crafting_categories,
            "available_technologies": technologies,
            "resource_reach": self.resource_reach,
            "build_reach": self.build_reach,
            "buildability": self.world.buildability,
            "place_max_age_ticks": self.place_max_age_ticks,
            "mask_toggles": self.v0_mask_toggles,
            "mask_counters": self.v0_mask_counters,
            "defer_result_measurement": True,
            "defer_final_drain": True,
            "advance_simulated_ticks": lambda ticks: self._advance_simulated_ticks(
                ticks, pause=True
            ),
            "logger": self._write_log,
        }

    def v0_learner_context(self) -> tuple[Any, dict[str, Any]]:
        """Expose the current world and mask context to the V0 learner."""
        if self.action_grammar != "v0":
            raise RuntimeError("V0 learner context requires action_grammar='v0'")
        if getattr(self, "_frame", None) is None:
            raise RuntimeError("V0 learner context is unavailable before reset")
        assert isinstance(self._frame, V0ObservationFrame)
        return self.world, self._v0_context(self._frame)

    @staticmethod
    def _copy_v0_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: (
                {nested: value.copy() for nested, value in block.items()}
                if isinstance(block, Mapping)
                else block.copy()
            )
            for key, block in obs.items()
        }

    def _build_v0_frame(self) -> V0ObservationFrame:
        assert self._current is not None
        # These imports intentionally occur only when the V0 grammar is used.
        from fle.rl.observation import select_entity_slots
        from fle.rl.v0_actions import v0_masks_for_prefix
        from fle.rl.v0_obs import build_v0_observation

        player_x, player_y = self._current.position
        obs = build_v0_observation(
            self.world,
            player_x=player_x,
            player_y=player_y,
            tick=self._current.tick,
            inventory=self._current.inventory,
            enabled_recipes=self._enabled_recipes,
            research_state=self._research_state,
            vocab=self.vocab,
            status_names=self._status_names,
            episode_start_tick=self._episode_start_tick,
            step_count=self._steps,
            max_steps=self._episode_max_steps,
            max_ticks=self.max_ticks,
            last_status=self._last_status,
            last_op=self._last_op,
            current_research=self._current.current_research,
            research_queue=self._current.research_queue,
            automated_score=self._current.score_automated,
            general_score=self._current.score_player,
            resource_reach=self.resource_reach,
            build_fresh_ticks=self.place_max_age_ticks,
            own=True,
        )
        entities = tuple(
            select_entity_slots(self.world.entities, self._current.position)
        )
        frame = V0ObservationFrame(obs=obs, entities=entities)
        obs["masks"] = v0_masks_for_prefix({}, obs, self.world, self._v0_context(frame))
        return frame

    def _build_frame(self) -> ObservationFrame | V0ObservationFrame:
        assert self._current is not None
        if getattr(self, "action_grammar", "legacy") == "v0":
            return self._build_v0_frame()
        self.observer.max_steps = self._episode_max_steps
        frame = self.observer.build(
            ObservationInput(
                snapshot=self._current,
                enabled_recipes=self._enabled_recipes,
                research_state=self._research_state,
                episode_start_tick=self._episode_start_tick,
                step_count=self._steps,
                last_status=self._last_status,
                last_op=self._last_op,
            )
        )
        self._cooldown_masked = self._apply_cooldown_mask(frame)
        return frame

    def snapshot(self) -> FrontierState:
        """Capture a return frontier reached by this environment's agent."""
        if self._current is None:
            raise RuntimeError("snapshot() requires a reset environment")
        return FrontierState(
            game_state=GameState.from_instance(self.instance),
            step_index=self._steps,
            max_steps=self._episode_max_steps,
            episode_ticks=max(0, self._current.tick - self._episode_start_tick),
            aps_baseline=self._current.score_automated,
            inventory=dict(self._current.inventory),
            entity_hash=_entity_identity_hash(self._current.entities),
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = {} if options is None else dict(options)
        restore = options.get("restore")
        if restore is not None and not isinstance(restore, FrontierState):
            raise TypeError("options['restore'] must be a FrontierState")
        if restore is None:
            self.instance.reset(
                reset_position=True,
                all_technologies_researched=False,
                clear_entities=True,
            )
        else:
            budget = options.get("budget")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise ValueError("a restored reset requires a positive integer budget")
            if restore.step_index >= restore.max_steps:
                raise ValueError("frontier has no remaining episode step budget")
            self.instance.reset(
                game_state=restore.game_state,
                reset_position=False,
                all_technologies_researched=False,
                clear_entities=True,
            )
        self.instance.rcon_client.send_command(PEACEFUL_RESET_COMMAND)
        self.world.entity_full_sync()
        self.world.all_drain()
        self.instance.set_speed_and_unpause(self.speed)
        self._enabled_recipes = self.world.read_enabled_recipes()
        self._research_state = self.world.read_research_state()
        self.world.techs_finished.clear()
        self._current, character_valid = self._snapshot()
        if not character_valid:
            raise RuntimeError("character is invalid immediately after reset")
        if restore is not None:
            actual_hash = _entity_identity_hash(self._current.entities)
            inventory_matches = dict(self._current.inventory) == restore.inventory
            if actual_hash != restore.entity_hash or not inventory_matches:
                self._episode_done = True
                self._write_log(
                    {
                        "frontier_restore_refused": {
                            "expected_hash": restore.entity_hash,
                            "actual_hash": actual_hash,
                            "inventory_matches": inventory_matches,
                        }
                    }
                )
                raise RuntimeError(
                    "frontier restore verification failed: entity identity/position "
                    f"hash match={actual_hash == restore.entity_hash}, "
                    f"inventory match={inventory_matches}"
                )
            self._episode_start_tick = self._current.tick - restore.episode_ticks
            self._steps = restore.step_index
            self._episode_max_steps = restore.max_steps
            self._excursion = True
            self._restore_hash = restore.entity_hash
            self._excursion_steps = 0
            self._excursion_budget = min(
                int(options["budget"]), restore.max_steps - restore.step_index
            )
        else:
            self._episode_start_tick = self._current.tick
            self._steps = 0
            self._episode_max_steps = self.max_steps
            self._excursion = False
            self._restore_hash = None
            self._excursion_steps = 0
            self._excursion_budget = None
        self._episode_index += 1
        self._episode_done = False
        self._end_reason = None
        self._last_status = None
        self._last_op = None
        self._aps_previous = self._current.score_automated
        self._max_aps = self._aps_previous
        self._op_histogram.clear()
        self._status_histogram.clear()
        self._entities_placed = 0
        self._items_harvested = 0
        self._episode_deaths = 0
        self._cooldown_action = None
        self._cooldown_fingerprint = None
        self._cooldown_entity_unit = None
        self._cooldown_masked = False
        self._frame = self._build_frame()
        info = {
            "score_player": self._current.score_player,
            "score_automated": self._current.score_automated,
            "excursion": self._excursion,
            "restore_hash": self._restore_hash,
        }
        if getattr(self, "action_grammar", "legacy") == "v0":
            info.update(
                mask_toggles=dict(self.v0_mask_toggles),
                mask_counters=dict(self.v0_mask_counters),
            )
            observation = self._copy_v0_observation(self._frame.obs)
            self.instance.game_control.pause_at_speed(max(self.speed, 10))
        else:
            observation = self._frame.obs.copy()
        return observation, info

    @staticmethod
    def _anchor(row: EntityRow) -> dict[str, Any]:
        return {
            "unit": row.unit,
            "name": row.name,
            "x": row.x,
            "y": row.y,
            "direction": row.direction,
        }

    @classmethod
    def _anchor_log(cls, slot: int, row: EntityRow | None) -> dict[str, Any]:
        result: dict[str, Any] = {"slot": slot}
        if row is not None:
            result.update(cls._anchor(row))
        return result

    def _clamp_entity_anchor(
        self, action: np.ndarray, op: str
    ) -> tuple[
        EntityRow | None,
        dict[str, Any],
        dict[str, Any] | None,
        str | None,
    ]:
        """Resolve the sampled entity slot, optionally using the legacy clamp."""
        assert self._frame is not None
        slot = int(action[S.HEADS.index("entity")])
        requested_row = self._frame.entities[slot]
        requested = self._anchor_log(slot, requested_row)
        live_entities = getattr(self.world, "entities", {})
        candidates: list[tuple[int, EntityRow]] = []
        for candidate_slot, frame_row in enumerate(self._frame.entities):
            if frame_row is None:
                continue
            live_row = live_entities.get(frame_row.unit)
            if live_row is None:
                continue
            if op == "ROTATE" and not entity_is_rotatable(live_row, self.vocab):
                continue
            candidates.append((candidate_slot, live_row))

        requested_live = (
            live_entities.get(requested_row.unit) if requested_row is not None else None
        )
        if requested_live is not None and (
            op != "ROTATE" or entity_is_rotatable(requested_live, self.vocab)
        ):
            return (
                requested_live,
                requested,
                self._anchor_log(slot, requested_live),
                None,
            )
        if not getattr(self, "clamp_anchor", False):
            reason = (
                "no_rotatable_entity" if op == "ROTATE" else "no_live_entity_anchor"
            )
            return None, requested, None, reason
        if not candidates:
            reason = (
                "no_rotatable_entity" if op == "ROTATE" else "no_live_entity_anchor"
            )
            return None, requested, None, reason

        if requested_row is None:
            executed_slot, executed_row = min(
                candidates,
                key=lambda candidate: (abs(candidate[0] - slot), candidate[0]),
            )
        else:
            executed_slot, executed_row = min(
                candidates,
                key=lambda candidate: (
                    math.hypot(
                        candidate[1].x - requested_row.x,
                        candidate[1].y - requested_row.y,
                    ),
                    candidate[0],
                ),
            )
        return (
            executed_row,
            requested,
            self._anchor_log(executed_slot, executed_row),
            None,
        )

    def _nearest_land_in_direction(
        self,
        desired: tuple[float, float],
        vector: tuple[float, float],
        player_pos: tuple[float, float],
    ) -> tuple[float, float]:
        tile_x, tile_y = math.floor(desired[0]), math.floor(desired[1])
        if self.world.is_known(tile_x, tile_y) and not self.world.is_water(
            tile_x, tile_y
        ):
            return desired
        candidates: list[tuple[float, int, int]] = []
        px, py = player_pos
        for radius in range(1, 33):
            for x in range(tile_x - radius, tile_x + radius + 1):
                for y in (tile_y - radius, tile_y + radius):
                    self._append_land_candidate(
                        candidates, x, y, desired, vector, (px, py)
                    )
            for y in range(tile_y - radius + 1, tile_y + radius):
                for x in (tile_x - radius, tile_x + radius):
                    self._append_land_candidate(
                        candidates, x, y, desired, vector, (px, py)
                    )
            if candidates:
                _, x, y = min(candidates)
                return x + 0.5, y + 0.5
        raise RuntimeError("no known land cell in the selected direction")

    def _append_land_candidate(
        self,
        candidates: list[tuple[float, int, int]],
        x: int,
        y: int,
        desired: tuple[float, float],
        vector: tuple[float, float],
        player_pos: tuple[float, float],
    ) -> None:
        if not self.world.is_known(x, y) or self.world.is_water(x, y):
            return
        dx, dy = x + 0.5 - player_pos[0], y + 0.5 - player_pos[1]
        if dx * vector[0] + dy * vector[1] <= 0:
            return
        distance = math.hypot(x + 0.5 - desired[0], y + 0.5 - desired[1])
        candidates.append((distance, x, y))

    def _place_position(
        self,
        item: str,
        offset_index: int,
        direction_index: int,
    ) -> tuple[str, tuple[float, float]]:
        assert self._current is not None
        prototype = self.vocab.place_results.get(item)
        if prototype is None:
            raise ValueError("selected item has no place result")
        info = self.vocab.entity_info.get(prototype)
        if info is None:
            raise ValueError("place result has no entity metadata")
        width = int(info.get("tile_width") or 1)
        height = int(info.get("tile_height") or 1)
        if S.DIRECTIONS[direction_index] in {4, 12}:
            width, height = height, width
        dx, dy = S.offset_to_dxdy(offset_index)
        origin = self._current.position
        raw_x = origin[0] + dx
        raw_y = origin[1] + dy
        x = math.floor(raw_x) + 0.5 if width % 2 else round(raw_x)
        y = math.floor(raw_y) + 0.5 if height % 2 else round(raw_y)
        return prototype, (x, y)

    def _placement_footprint(
        self,
        item: str,
        offset_index: int,
        direction_index: int,
    ) -> tuple[str, tuple[float, float], set[tuple[int, int]]]:
        prototype, centre = self._place_position(item, offset_index, direction_index)
        info = self.vocab.entity_info[prototype]
        width = int(info.get("tile_width") or 1)
        height = int(info.get("tile_height") or 1)
        if S.DIRECTIONS[direction_index] in {4, 12}:
            width, height = height, width
        left = math.floor(centre[0] - width / 2 + 0.5)
        top = math.floor(centre[1] - height / 2 + 0.5)
        footprint = {
            (x, y) for x in range(left, left + width) for y in range(top, top + height)
        }
        return prototype, centre, footprint

    def _placement_is_free(
        self,
        item: str,
        offset_index: int,
        direction_index: int,
    ) -> bool:
        assert self._current is not None
        _, centre, footprint = self._placement_footprint(
            item, offset_index, direction_index
        )
        origin = self._current.position
        if (
            math.hypot(
                centre[0] - origin[0],
                centre[1] - origin[1],
            )
            > self.build_reach
        ):
            return False
        occupied: set[tuple[int, int]] = set()
        for row in getattr(self.world, "entities", self._current.entities).values():
            info = self.vocab.entity_info.get(row.name, {})
            width = row.tile_width or int(info.get("tile_width") or 1)
            height = row.tile_height or int(info.get("tile_height") or 1)
            left = math.floor(row.x - width / 2 + 0.5)
            top = math.floor(row.y - height / 2 + 0.5)
            occupied.update(
                (x, y)
                for x in range(left, left + width)
                for y in range(top, top + height)
            )
        blocked = occupied | set(self.world.trees) | set(self.world.obstacles)
        return all(
            tile not in blocked and not self.world.is_water(*tile) for tile in footprint
        )

    def _decode(
        self,
        action: np.ndarray,
        entity_row: EntityRow | None = None,
    ) -> ActionSpec:
        assert self._current is not None and self._frame is not None
        values = {head: int(action[index]) for index, head in enumerate(S.HEADS)}
        op = S.OPS[values["op"]]
        args: dict[str, Any] = {}
        if op == "WAIT":
            ticks = S.DURATIONS_TICKS[values["duration"]]
            args = {"ticks": ticks, "seconds": ticks / 60.0}
        elif op == "MOVE":
            vector = MOVE_VECTORS[values["move_dir"]]
            desired = (
                self._current.position[0] + vector[0] * S.MOVE_STEP,
                self._current.position[1] + vector[1] * S.MOVE_STEP,
            )
            target = self._nearest_land_in_direction(
                desired, vector, self._current.position
            )
            args = {"target": [target[0], target[1]]}
        elif op == "HARVEST":
            target = self._frame.targets[values["target"]]
            if target is None:
                raise ValueError("selected target slot is empty")
            args = self._harvest_args(target, values["quantity"])
        elif op == "CRAFT":
            recipe = S.RECIPE_NAMES[values["recipe"]]
            args = {
                "recipe": recipe,
                "quantity": S.QUANTITIES[values["quantity"]],
                "main_product": self.vocab.main_products.get(recipe),
            }
        elif op == "PLACE":
            item = S.PLACEABLE_NAMES[values["placeable"]]
            prototype, target = self._place_position(
                item, values["offset"], values["direction"]
            )
            args = {
                "item": item,
                "prototype": prototype,
                "target": [target[0], target[1]],
                "direction": S.DIRECTIONS[values["direction"]],
            }
        elif op in ENTITY_OPS:
            row = entity_row or self._frame.entities[values["entity"]]
            if row is None:
                raise ValueError("selected entity slot is empty")
            args = {"anchor": self._anchor(row)}
            if op == "ROTATE":
                args["direction"] = S.DIRECTIONS[values["direction"]]
            elif op in {"INSERT", "EXTRACT"}:
                args["item"] = S.ITEM_NAMES[values["item"]]
                args["quantity"] = S.QUANTITIES[values["quantity"]]
            elif op == "SET_RECIPE":
                args["recipe"] = S.RECIPE_NAMES[values["recipe"]]
        elif op == "CONNECT":
            source = self._frame.entities[values["entity"]]
            peer = self._frame.entities[values["peer"]]
            if source is None or peer is None:
                raise ValueError("selected CONNECT entity slot is empty")
            args = {
                "source": self._anchor(source),
                "peer": self._anchor(peer),
                "connector": S.CONNECTOR_NAMES[values["connector"]],
            }
        elif op == "RESEARCH":
            args = {"technology": S.TECH_NAMES[values["technology"]]}
        return ActionSpec(op, args)

    @staticmethod
    def _harvest_args(target: HarvestTarget, quantity_index: int) -> dict[str, Any]:
        args: dict[str, Any] = {
            "target_kind": "tree" if target.kind == "tree" else "patch",
            "target": [target.position[0], target.position[1]],
            "quantity": S.QUANTITIES[quantity_index],
        }
        if target.patch_id is not None:
            args["patch_id"] = target.patch_id
        return args

    def _masked_reason(self, action: np.ndarray) -> str | None:
        assert self._frame is not None
        if self._cooldown_hit(action):
            # The observation may have been queued before the learner received
            # the cooldown-adjusted mask. Execute stale exact actions normally.
            return None
        op_index = int(action[S.HEADS.index("op")])
        op = S.OPS[op_index]
        masks = S.split_masks(self._frame.obs)
        if not masks["op"][op_index]:
            if op in ENTITY_OPS or op in {"PLACE", "CONNECT"}:
                return None
            return "masked_op"
        for head in S.OP_HEADS[op]:
            value = int(action[S.HEADS.index(head)])
            if not masks[head][value]:
                if head == "entity" and op in ENTITY_OPS:
                    continue
                if head == "item" and op in {"INSERT", "EXTRACT"}:
                    continue
                if head == "offset" and op == "PLACE":
                    continue
                if head in {"entity", "peer"} and op == "CONNECT":
                    continue
                return f"masked_{head}"
        return None

    def _cooldown_hit(self, action: np.ndarray) -> bool:
        if not getattr(self, "support_cooldowns", False) or self._current is None:
            return False
        fingerprint = self._support_fingerprint(
            self._current,
            self._cooldown_action,
            getattr(self, "_cooldown_entity_unit", None),
        )
        if fingerprint != self._cooldown_fingerprint:
            self._cooldown_action = None
            self._cooldown_fingerprint = None
            self._cooldown_entity_unit = None
            return False
        return self._cooldown_action == self._complete_action_key(action)

    def _apply_cooldown_mask(self, frame: ObservationFrame) -> bool:
        """Expose an exact-action cooldown through an existing per-head mask."""
        if (
            not getattr(self, "support_cooldowns", False)
            or self._current is None
            or self._cooldown_action is None
        ):
            return False
        if (
            self._support_fingerprint(
                self._current,
                self._cooldown_action,
                getattr(self, "_cooldown_entity_unit", None),
            )
            != self._cooldown_fingerprint
        ):
            self._cooldown_action = None
            self._cooldown_fingerprint = None
            self._cooldown_entity_unit = None
            return False

        key = self._cooldown_action
        op_index = key[0]
        op = S.OPS[op_index]
        selected = dict(zip(S.OP_HEADS[op], key[1:], strict=True))
        masks = S.split_masks(frame.obs)
        # Use the last-read argument that still has another completion. If all
        # active arguments are singletons, the op itself has no other admitted
        # completion and is the narrowest representable cooldown mask.
        for head in reversed(S.OP_HEADS[op]):
            value = selected[head]
            if masks[head][value] and int(masks[head].sum()) > 1:
                start, _ = S.MASK_OFFSETS[head]
                frame.obs[start + value] = 0.0
                return True
        start, _ = S.MASK_OFFSETS["op"]
        frame.obs[start + op_index] = 0.0
        return True

    def _support_fingerprint(
        self,
        snapshot: OperationSnapshot,
        action_key: tuple[int, ...] | None = None,
        entity_unit: int | None = None,
    ) -> tuple[Any, ...]:
        """State components whose changes can alter deterministic support."""
        player_tile = tuple(math.floor(value) for value in snapshot.position)
        inventory = tuple(sorted(snapshot.inventory.items()))
        entity_units = tuple(sorted(snapshot.entities))
        relevant_slot: tuple[Any, ...] | None = None
        row = snapshot.entities.get(entity_unit) if entity_unit is not None else None
        if action_key is not None and row is not None:
            op = S.OPS[action_key[0]]
            selected = dict(zip(S.OP_HEADS[op], action_key[1:], strict=True))
            if op == "INSERT" and "item" in selected:
                item = S.ITEM_NAMES[selected["item"]]
                relevant_slot = (
                    "insert",
                    row.unit,
                    furnace_input_item(row, self.vocab),
                    item in insertable_items(row, snapshot.inventory, self.vocab),
                )
            elif op == "EXTRACT" and "item" in selected:
                item = S.ITEM_NAMES[selected["item"]]
                relevant_slot = (
                    "extract",
                    row.unit,
                    row.items.get(item, 0) > 0,
                )
        return player_tile, inventory, entity_units, relevant_slot

    @staticmethod
    def _complete_action_key(action: np.ndarray) -> tuple[int, ...]:
        op_index = int(action[S.HEADS.index("op")])
        op = S.OPS[op_index]
        return (op_index,) + tuple(
            int(action[S.HEADS.index(head)]) for head in S.OP_HEADS[op]
        )

    @staticmethod
    def _item_score_vector(options: Mapping[str, Any] | None) -> np.ndarray | None:
        if not options:
            return None
        for key in ("item_q", "item_logits"):
            if key in options:
                values = np.asarray(options[key], dtype=np.float64)
                return values if values.shape == (S.HEAD_SIZES["item"],) else None
        for key in ("head_q", "head_logits", "q_values", "logits"):
            group = options.get(key)
            if isinstance(group, Mapping) and "item" in group:
                values = np.asarray(group["item"], dtype=np.float64)
                return values if values.shape == (S.HEAD_SIZES["item"],) else None
        return None

    def _clamp_item(
        self,
        action: ActionSpec,
        row: EntityRow,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[ActionSpec, str, str | None, str | None]:
        """Resolve the sampled item, optionally using the legacy item clamp."""
        assert self._current is not None
        requested = str(action.args["item"])
        if action.op == "INSERT":
            admitted = insertable_items(row, self._current.inventory, self.vocab)
            no_support_reason = "insert_no_accepted_item"
        else:
            admitted = extractable_items(row, self.vocab)
            no_support_reason = "extract_entity_has_no_items"
        if not admitted:
            return action, requested, None, no_support_reason
        if requested in admitted:
            return action, requested, requested, None
        if not getattr(self, "clamp_item", False):
            reason = (
                "insert_item_not_held_or_accepted"
                if action.op == "INSERT"
                else "extract_item_not_present"
            )
            return action, requested, None, reason

        scores = self._item_score_vector(options)
        if scores is None:
            executed = admitted[0]
        else:
            executed = max(
                admitted,
                key=lambda name: (scores[S.ITEM_INDEX[name]], -S.ITEM_INDEX[name]),
            )
        args = dict(action.args)
        args["item"] = executed
        return ActionSpec(action.op, args), requested, executed, None

    def _clamp_quantity(
        self, action: ActionSpec
    ) -> tuple[ActionSpec, int | None, int | None]:
        """Clamp only quantities the selected tool call would deterministically refuse."""
        clamp_quantity = getattr(
            self,
            "clamp_quantity",
            getattr(self, "quantity_aware_support", True),
        )
        if not clamp_quantity or action.op not in {
            "CRAFT",
            "INSERT",
            "EXTRACT",
        }:
            quantity = action.args.get("quantity")
            return action, quantity, quantity
        assert self._current is not None
        requested = int(action.args["quantity"])
        executed = requested
        if action.op == "CRAFT":
            recipe = str(action.args["recipe"])
            covered = [
                quantity
                for quantity in S.QUANTITIES
                if quantity <= requested
                and recipe
                in recursively_craftable_recipes(
                    self._current.inventory,
                    self._enabled_recipes,
                    self.vocab,
                    count=quantity,
                    crafting_categories=self.crafting_categories,
                )
            ]
            if covered:
                executed = max(covered)
        elif action.op == "INSERT":
            executed = min(
                requested, self._current.inventory.get(action.args["item"], 0)
            )
        else:
            row = self._entity_for_anchor(action.args["anchor"])
            present = row.items.get(action.args["item"], 0) if row is not None else 0
            executed = min(requested, present)
        if executed == requested:
            return action, requested, executed
        args = dict(action.args)
        args["quantity"] = executed
        return ActionSpec(action.op, args), requested, executed

    def _entity_for_anchor(self, anchor: Mapping[str, Any]) -> EntityRow | None:
        live_entities = getattr(self.world, "entities", {})
        row = live_entities.get(anchor["unit"])
        if row is not None:
            return row
        assert self._current is not None
        return self._current.entities.get(anchor["unit"])

    def _connector_endpoint_compatible(
        self, row: EntityRow, connector: str, *, source: bool
    ) -> bool:
        if connector == "small-electric-pole":
            return True
        entity_type = self.vocab.entity_info.get(row.name, {}).get("type")
        if connector == "pipe":
            return (
                bool(row.fluids)
                or entity_type in PIPE_ENDPOINT_TYPES
                or row.name in PIPE_ENDPOINT_NAMES
            )
        if connector == "transport-belt" and source:
            return entity_type not in BELT_SOURCE_BLOCKED_TYPES
        return True

    def _clamp_connect(
        self, action: np.ndarray
    ) -> tuple[
        ActionSpec | None,
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any],
        dict[str, Any] | None,
        str | None,
        str | None,
    ]:
        """Resolve sampled CONNECT endpoints, optionally using the legacy clamp."""
        assert self._frame is not None
        source_slot = int(action[S.HEADS.index("entity")])
        peer_slot = int(action[S.HEADS.index("peer")])
        connector = S.CONNECTOR_NAMES[int(action[S.HEADS.index("connector")])]
        requested_source_row = self._frame.entities[source_slot]
        requested_peer_row = self._frame.entities[peer_slot]
        requested_source = self._anchor_log(source_slot, requested_source_row)
        requested_peer = self._anchor_log(peer_slot, requested_peer_row)
        live = getattr(self.world, "entities", {})
        requested_source_live = (
            live.get(requested_source_row.unit)
            if requested_source_row is not None
            else None
        )
        requested_peer_live = (
            live.get(requested_peer_row.unit)
            if requested_peer_row is not None
            else None
        )
        if not getattr(self, "clamp_connect", False):
            if requested_source_live is None:
                return (
                    None,
                    requested_source,
                    None,
                    requested_peer,
                    None,
                    "connect_source_stale_or_empty",
                    None,
                )
            if not self._connector_endpoint_compatible(
                requested_source_live, connector, source=True
            ):
                return (
                    None,
                    requested_source,
                    None,
                    requested_peer,
                    None,
                    "connect_source_incompatible",
                    None,
                )
            if requested_peer_live is None:
                return (
                    None,
                    requested_source,
                    self._anchor_log(source_slot, requested_source_live),
                    requested_peer,
                    None,
                    "connect_peer_stale_or_empty",
                    None,
                )
            if requested_peer_live.unit == requested_source_live.unit:
                return (
                    None,
                    requested_source,
                    self._anchor_log(source_slot, requested_source_live),
                    requested_peer,
                    None,
                    "connect_requires_distinct_endpoints",
                    None,
                )
            if not self._connector_endpoint_compatible(
                requested_peer_live, connector, source=False
            ):
                return (
                    None,
                    requested_source,
                    self._anchor_log(source_slot, requested_source_live),
                    requested_peer,
                    None,
                    "connect_peer_incompatible",
                    None,
                )
            spec = ActionSpec(
                "CONNECT",
                {
                    "source": self._anchor(requested_source_live),
                    "peer": self._anchor(requested_peer_live),
                    "connector": connector,
                },
            )
            return (
                spec,
                requested_source,
                self._anchor_log(source_slot, requested_source_live),
                requested_peer,
                self._anchor_log(peer_slot, requested_peer_live),
                None,
                None,
            )
        rows = [
            (slot, live[frame_row.unit])
            for slot, frame_row in enumerate(self._frame.entities)
            if frame_row is not None and frame_row.unit in live
        ]
        sources = [
            candidate
            for candidate in rows
            if self._connector_endpoint_compatible(candidate[1], connector, source=True)
        ]
        if not sources:
            return (
                None,
                requested_source,
                None,
                requested_peer,
                None,
                "connect_no_compatible_source",
                None,
            )
        source_choice = next(
            (candidate for candidate in sources if candidate[0] == source_slot),
            min(
                sources,
                key=lambda candidate: (abs(candidate[0] - source_slot), candidate[0]),
            ),
        )
        peers = [
            candidate
            for candidate in rows
            if candidate[1].unit != source_choice[1].unit
            and self._connector_endpoint_compatible(
                candidate[1], connector, source=False
            )
        ]
        if not peers:
            return (
                None,
                requested_source,
                self._anchor_log(*source_choice),
                requested_peer,
                None,
                "connect_no_distinct_compatible_peer",
                None,
            )
        peer_choice = next(
            (candidate for candidate in peers if candidate[0] == peer_slot),
            min(
                peers,
                key=lambda candidate: (abs(candidate[0] - peer_slot), candidate[0]),
            ),
        )
        reasons = []
        if source_choice[0] != source_slot:
            reasons.append("source_incompatible_or_stale")
        if peer_choice[0] != peer_slot:
            reasons.append("peer_not_distinct_incompatible_or_stale")
        spec = ActionSpec(
            "CONNECT",
            {
                "source": self._anchor(source_choice[1]),
                "peer": self._anchor(peer_choice[1]),
                "connector": connector,
            },
        )
        return (
            spec,
            requested_source,
            self._anchor_log(*source_choice),
            requested_peer,
            self._anchor_log(*peer_choice),
            None,
            ",".join(reasons) or None,
        )

    def _invalid_combination(self, action: ActionSpec) -> str | None:
        assert self._current is not None
        op, args = action.op, action.args
        if op == "PLACE" and self._current.inventory.get(args["item"], 0) <= 0:
            return "place_item_not_held"
        if op == "SET_RECIPE":
            row = self._entity_for_anchor(args["anchor"])
            entity_type = (
                self.vocab.entity_info.get(row.name, {}).get("type") if row else None
            )
            category = (self.vocab.recipe_categories or {}).get(
                args["recipe"], "crafting"
            )
            if entity_type != "assembling-machine" or category != "crafting":
                return "recipe_entity_mismatch"
        return None

    def _wait_true_ticks(self, requested_ticks: int) -> ExecutionResult:
        try:
            self._advance_simulated_ticks(requested_ticks)
        except TimeoutError:
            return ExecutionResult(status="no_effect", reason_class="wait_wall_cap")
        return ExecutionResult()

    def _advance_simulated_ticks(
        self, requested_ticks: int, *, pause: bool = False
    ) -> None:
        """Advance Factorio time by polling simulation ticks without wall sleeps."""
        start_tick = self.world.read_tick()
        target_tick = start_tick + requested_ticks
        deadline = time.monotonic() + WAIT_WALL_CAP_S
        self.instance.set_speed_and_unpause(max(self.speed, 10))
        while self.world.read_tick() < target_tick:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"simulation did not advance {requested_ticks} ticks"
                )
        if pause:
            self.instance.pause()

    def _execute(self, action: ActionSpec) -> ExecutionResult:
        if action.op == "WAIT":
            return self._wait_true_ticks(int(action.args["ticks"]))
        if action.op == "CONNECT":
            return call_with_timeout(lambda: self._execute_connect(action))
        return call_with_timeout(
            lambda: execute_action(
                action,
                getattr(self, "regime", "macro"),
                self.namespace,
                self.world,
                self.resource_reach,
                (
                    self.build_reach
                    if getattr(self, "regime", "macro") == "macro"
                    else None
                ),
            )
        )

    def _execute_connect(self, action: ActionSpec) -> ExecutionResult:
        """Connect slot centres without resolving either endpoint to an entity."""
        source_anchor = action.args["source"]
        source_row = self.world.entities.get(source_anchor["unit"])
        if source_row is None:
            raise AnchorResolutionError(
                "anchor_resolution_failed: CONNECT source row disappeared"
            )
        if getattr(self, "regime", "macro") == "macro":
            approach_result = _approach_entity(
                source_row,
                self.namespace,
                self.world,
                self.build_reach,
            )
            if approach_result is not None:
                return approach_result
        peer_anchor = action.args["peer"]
        source_row = self.world.entities.get(source_anchor["unit"])
        peer_row = self.world.entities.get(peer_anchor["unit"])
        connector_name = action.args["connector"]
        if source_row is None or peer_row is None:
            return ExecutionResult(
                status="no_support",
                reason_class="connect_endpoint_disappeared_after_approach",
            )
        if source_row.unit == peer_row.unit:
            return ExecutionResult(
                status="no_support",
                reason_class="connect_requires_distinct_endpoints",
            )
        if not self._connector_endpoint_compatible(
            source_row, connector_name, source=True
        ) or not self._connector_endpoint_compatible(
            peer_row, connector_name, source=False
        ):
            return ExecutionResult(
                status="no_support",
                reason_class="connect_endpoint_incompatible_after_approach",
            )
        connector = prototype_by_name[connector_name]
        self.namespace.connect_entities(
            Position(x=source_row.x, y=source_row.y),
            Position(x=peer_row.x, y=peer_row.y),
            connection_type=connector,
        )
        return ExecutionResult()

    def _step_v0(
        self, action
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute the strict V0 grammar without entering the legacy action path."""
        from fle.rl import v0_schema as V0
        from fle.rl.v0_actions import decode_v0_action, execute_v0_action

        assert self._current is not None
        assert isinstance(self._frame, V0ObservationFrame)
        decision_started = time.perf_counter()
        action_array = np.asarray(action, dtype=np.int64)
        if action_array.shape != (len(V0.HEADS),) or not self.action_space.contains(
            action_array
        ):
            raise ValueError(
                "action is outside V0 MultiDiscrete"
                f"{tuple(V0.HEAD_SIZES[head] for head in V0.HEADS)}"
            )
        before = self._current
        context = self._v0_context(self._frame)
        decode_started = time.perf_counter()
        decoded = decode_v0_action(action_array, self._frame.obs, context)
        decode_wall_s = time.perf_counter() - decode_started
        self.instance.set_speed_and_unpause(max(self.speed, 10))
        execute_started = time.perf_counter()
        try:
            result = execute_v0_action(
                decoded,
                self.namespace,
                self.world,
                context,
                obs=self._frame.obs,
            )
        finally:
            self.instance.pause()
        execute_wall_s = time.perf_counter() - execute_started

        # A single combined drain is sufficient; unlike the legacy path this
        # never waits in wall time for delayed effects.
        observe_started = time.perf_counter()
        self.world.all_drain()
        after, character_valid = self._snapshot(before)
        observe_wall_s = time.perf_counter() - observe_started
        result = replace(
            result,
            tau=max(0, after.tick - before.tick) / 60.0,
            reward=after.score_automated - before.score_automated,
            general_score=after.score_player,
            general_score_delta=after.score_player - before.score_player,
            tick_before=before.tick,
            tick_after=after.tick,
        )
        self._steps += 1
        if self._excursion:
            self._excursion_steps += 1
        episode_ticks = max(0, after.tick - self._episode_start_tick)
        excursion_done = (
            self._excursion_budget is not None
            and self._excursion_steps >= self._excursion_budget
        )
        terminated = not character_valid
        truncated = (
            self._steps >= self._episode_max_steps
            or episode_ticks >= self.max_ticks
            or excursion_done
        )
        if terminated:
            self._end_reason = "death"
            self._episode_deaths += 1
            self.deaths += 1
        elif excursion_done:
            self._end_reason = "excursion_budget"
        elif self._steps >= self._episode_max_steps:
            self._end_reason = "max_steps"
        elif episode_ticks >= self.max_ticks:
            self._end_reason = "max_ticks"

        self._current = after
        self._aps_previous = after.score_automated
        self._max_aps = max(self._max_aps, after.score_automated)
        self._last_status = result.status
        self._last_op = decoded.verb
        self._op_histogram[decoded.verb] += 1
        self._status_histogram[result.status] += 1
        if decoded.verb == "PLACE" and result.success:
            self._entities_placed += 1
        if self.world.techs_finished:
            self._enabled_recipes = self.world.read_enabled_recipes()
            self._research_state = self.world.read_research_state()
            self.world.techs_finished.clear()
        observation_started = time.perf_counter()
        self._frame = self._build_v0_frame()
        observation_wall_s = time.perf_counter() - observation_started
        decision_wall_s = time.perf_counter() - decision_started
        info = {
            "op": decoded.verb,
            "status": result.status,
            "success": result.success,
            "reason_class": result.reason.value if result.reason else None,
            "engine_message": result.message,
            "tau": result.tau,
            "ticks": result.tick_after - result.tick_before,
            "score_player": result.general_score,
            "score_player_delta": result.general_score_delta,
            "score_automated": after.score_automated,
            "requested_unit_number": result.requested_unit_number,
            "mutated_unit_number": result.mutated_unit_number,
            "mask_toggles": dict(self.v0_mask_toggles),
            "mask_counters": dict(self.v0_mask_counters),
            "action_grammar": "v0",
            "phase_wall_s": {
                "decode": decode_wall_s,
                "execute": execute_wall_s,
                "drain_and_snapshot": observe_wall_s,
                "snapshot": getattr(self, "_v0_last_snapshot_wall_s", 0.0),
                "observation": observation_wall_s,
                "decision": decision_wall_s,
            },
        }
        self._write_log(
            {
                "episode": self._episode_index,
                "step": self._steps - 1,
                **info,
                "reward": result.reward,
                "action": action_array.tolist(),
                "args": decoded.args,
            }
        )
        if terminated or truncated:
            self._episode_done = True
            self._write_episode_end()
        return (
            self._copy_v0_observation(self._frame.obs),
            float(result.reward),
            terminated,
            truncated,
            info,
        )

    def step(self, action, options: Mapping[str, Any] | None = None):
        if self._episode_done or self._current is None or self._frame is None:
            raise RuntimeError(
                "reset() must be called before step(), and after episode end"
            )
        if getattr(self, "action_grammar", "legacy") == "v0":
            return self._step_v0(action)
        action_array = np.asarray(action, dtype=np.int64)
        if action_array.shape != (len(S.HEADS),) or not self.action_space.contains(
            action_array
        ):
            raise ValueError(f"action is outside MultiDiscrete{S.HEAD_DIMS}")
        before = self._current
        step_start = time.perf_counter()
        status = "no_effect"
        reason_class = "predicate_false"
        effect: dict[str, Any] = {}
        raw_error = ""
        execution_result: ExecutionResult | None = None
        requested_quantity: int | None = None
        executed_quantity: int | None = None
        op = S.OPS[int(action_array[S.HEADS.index("op")])]
        used_heads = ("op", *S.OP_HEADS[op])
        preceding_masks = S.split_masks(self._frame.obs)
        admitted_indices = {
            head: [int(index) for index in np.flatnonzero(preceding_masks[head])]
            for head in used_heads
        }
        spec: ActionSpec | None = None
        tool_attempted = False
        requested_item: str | None = None
        executed_item: str | None = None
        requested_anchor: dict[str, Any] | None = None
        executed_anchor: dict[str, Any] | None = None
        requested_peer: dict[str, Any] | None = None
        executed_peer: dict[str, Any] | None = None
        requested_offset: int | None = None
        clamp_reason: str | None = None
        cooldown_hit = self._cooldown_hit(action_array)

        masked_reason = self._masked_reason(action_array)
        if masked_reason is not None:
            status, reason_class = "tool_rejected", masked_reason
        else:
            try:
                entity_row = None
                no_support_reason = None
                if op in ENTITY_OPS:
                    (
                        entity_row,
                        requested_anchor,
                        executed_anchor,
                        no_support_reason,
                    ) = self._clamp_entity_anchor(action_array, op)
                elif op == "PLACE":
                    requested_offset = int(action_array[S.HEADS.index("offset")])
                    item = S.PLACEABLE_NAMES[
                        int(action_array[S.HEADS.index("placeable")])
                    ]
                    if not self._placement_is_free(
                        item,
                        requested_offset,
                        int(action_array[S.HEADS.index("direction")]),
                    ):
                        no_support_reason = "place_offset_unsupported"
                elif op == "CONNECT":
                    (
                        spec,
                        requested_anchor,
                        executed_anchor,
                        requested_peer,
                        executed_peer,
                        no_support_reason,
                        clamp_reason,
                    ) = self._clamp_connect(action_array)
                if execution_result is not None:
                    pass
                elif no_support_reason is not None:
                    status, reason_class = "no_support", no_support_reason
                else:
                    if spec is None:
                        spec = self._decode(action_array, entity_row)
                    if op in {"INSERT", "EXTRACT"}:
                        assert entity_row is not None
                        (
                            spec,
                            requested_item,
                            executed_item,
                            no_support_reason,
                        ) = self._clamp_item(spec, entity_row, options)
                    if no_support_reason is not None:
                        status, reason_class = "no_support", no_support_reason
                    else:
                        spec, requested_quantity, executed_quantity = (
                            self._clamp_quantity(spec)
                        )
                        combination_reason = self._invalid_combination(spec)
                        if combination_reason is not None:
                            status, reason_class = "tool_rejected", combination_reason
                        else:
                            tool_attempted = True
                            execution_result = self._execute(spec)
            except TimeoutError as exc:
                status, reason_class, raw_error = "error", "timeout", str(exc)[:200]
            except AnchorResolutionError as exc:
                status = "error"
                reason_class = "anchor_resolution_failed"
                raw_error = str(exc)[:200]
            except ValueError as exc:
                status, reason_class, raw_error = (
                    "tool_rejected",
                    "invalid_combination",
                    str(exc)[:200],
                )
            except Exception as exc:  # noqa: BLE001 - tool clients raise Exception
                status = "tool_rejected"
                raw_error = str(exc)[:200]
                reason_class = classify_error(op, raw_error)

        try:
            self.world.all_drain()
            if op in DELAYED_DRAIN_OPS:
                self._advance_simulated_ticks(15)
                self.world.all_drain()
            after, character_valid = self._snapshot(before)
        except Exception as exc:  # noqa: BLE001 - preserve a terminal transition
            after = before
            character_valid = False
            status = "error"
            reason_class = "observe_error"
            raw_error = str(exc)[:200]

        if execution_result is not None and execution_result.status is not None:
            status = execution_result.status
            reason_class = execution_result.reason_class or status
        elif spec is not None and tool_attempted:
            ok, effect = verify_effect(spec, before, after)
            if op == "CONNECT" and raw_error and effect.get("new_entities", 0) <= 0:
                ok = False
            if ok:
                reason_class = "ok_after_exception" if raw_error else "effect_verified"
                status = "ok"
            elif status not in {"error", "tool_rejected", "no_support"}:
                status = "no_effect"
                reason_class = "predicate_false"

        jumped_to_spawn = (
            op != "MOVE"
            and math.hypot(*before.position) > 2.5
            and math.hypot(*after.position) <= 1.0
        )
        terminated = not character_valid or jumped_to_spawn
        if terminated:
            reason_class = "character_invalid"
            self._episode_deaths += 1
            self.deaths += 1

        self._steps += 1
        if self._excursion:
            self._excursion_steps += 1
        elapsed_ticks = max(0, after.tick - before.tick)
        episode_ticks = max(0, after.tick - self._episode_start_tick)
        excursion_done = (
            self._excursion_budget is not None
            and self._excursion_steps >= self._excursion_budget
        )
        max_steps_done = self._steps >= self._episode_max_steps
        max_ticks_done = episode_ticks >= self.max_ticks
        truncated = max_steps_done or max_ticks_done or excursion_done
        if terminated:
            self._end_reason = "death"
        elif excursion_done:
            self._end_reason = "excursion_budget"
        elif max_steps_done:
            self._end_reason = "max_steps"
        elif max_ticks_done:
            self._end_reason = "max_ticks"
        reward = after.score_automated - self._aps_previous
        self._aps_previous = after.score_automated
        self._max_aps = max(self._max_aps, after.score_automated)
        delta = inventory_delta(before.inventory, after.inventory)
        transferred_quantity: int | None = None
        if spec is not None and op in {"INSERT", "EXTRACT"}:
            item_delta = delta.get(str(spec.args["item"]), 0)
            transferred_quantity = max(0, -item_delta if op == "INSERT" else item_delta)
        elif op in {"HARVEST", "PICKUP"}:
            transferred_quantity = sum(max(0, count) for count in delta.values())
        self._current = after
        if self.support_cooldowns and (
            status == "no_support"
            and reason_class in DETERMINISTIC_NO_SUPPORT_COOLDOWNS
        ):
            self._cooldown_action = self._complete_action_key(action_array)
            self._cooldown_entity_unit = (
                entity_row.unit if entity_row is not None else None
            )
            self._cooldown_fingerprint = self._support_fingerprint(
                after,
                self._cooldown_action,
                self._cooldown_entity_unit,
            )
        self._last_status = status
        self._last_op = op
        self._op_histogram[op] += 1
        self._status_histogram[status] += 1
        if op == "PLACE" and status == "ok":
            self._entities_placed += 1
        if op == "HARVEST" and status == "ok":
            self._items_harvested += sum(max(0, count) for count in delta.values())

        if self.world.techs_finished:
            self._enabled_recipes = self.world.read_enabled_recipes()
            self._research_state = self.world.read_research_state()
            self.world.techs_finished.clear()
        self._cooldown_masked = False
        self._frame = self._build_frame()
        wall_s = time.perf_counter() - step_start
        info = {
            "op": op,
            "status": status,
            "reason_class": reason_class,
            "score_player": after.score_player,
            "score_automated": after.score_automated,
            "ticks": elapsed_ticks,
            "wall_s": wall_s,
            "items_delta": delta,
            "player_pos_before": list(before.position),
            "player_pos_after": list(after.position),
            "transferred_quantity": transferred_quantity,
            "admitted_indices": admitted_indices,
            "n_entities": len(after.entities),
            "deaths": self.deaths,
            "excursion": self._excursion,
            "restore_hash": self._restore_hash,
            "requested_quantity": requested_quantity,
            "executed_quantity": executed_quantity,
            "requested_item": requested_item,
            "executed_item": executed_item,
            "requested_anchor": requested_anchor,
            "executed_anchor": executed_anchor,
            "requested_peer": requested_peer,
            "executed_peer": executed_peer,
            "requested_offset": requested_offset,
            "executed_offset": (
                requested_offset if op == "PLACE" and status == "ok" else None
            ),
            "clamp_reason": clamp_reason,
            "cooldown_hit": cooldown_hit,
            "cooldown_masked": self._cooldown_masked,
            "regime": getattr(self, "regime", "macro"),
        }
        record = {
            "episode": self._episode_index,
            "step": self._steps - 1,
            **info,
            "reward": reward,
            "raw_error": raw_error,
            "action": action_array.tolist(),
            "args": spec.args if spec is not None else {},
            "effect": effect,
        }
        self._write_log(record)
        if terminated or truncated:
            self._episode_done = True
            self._write_episode_end()
        return self._frame.obs.copy(), float(reward), terminated, truncated, info

    def _write_log(self, record: Mapping[str, Any]) -> None:
        if self._log_file is not None:
            self._log_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_episode_end(self) -> None:
        assert self._current is not None
        end_reason = getattr(self, "_end_reason", None) or "max_steps"
        self._write_log(
            {
                "episode_end": {
                    "episode": self._episode_index,
                    "final_aps": self._current.score_automated,
                    "max_aps": self._max_aps,
                    "player_ps": self._current.score_player,
                    "op_histogram": dict(self._op_histogram),
                    "status_histogram": dict(self._status_histogram),
                    "entities_placed": self._entities_placed,
                    "items_harvested": self._items_harvested,
                    "deaths": self._episode_deaths,
                    "excursion": self._excursion,
                    "restore_hash": self._restore_hash,
                    "max_steps": self._episode_max_steps,
                    "end_reason": end_reason,
                    "regime": getattr(self, "regime", "macro"),
                }
            }
        )

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if hasattr(self, "instance"):
            self.instance.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random valid FLE macro policy")
    parser.add_argument("--port", type=int, default=27000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--speed", type=float, default=40)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--action-regime", choices=("macro", "bare"), default="macro")
    parser.add_argument(
        "--clamp-quantity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clamp-item",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--clamp-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--clamp-connect",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _random_acceptance_action(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a valid action, probing CRAFT at the mask's one-output contract."""
    action = S.random_valid_action(obs, rng)
    op = S.OPS[int(action[S.HEADS.index("op")])]
    if op == "CRAFT":
        action[S.HEADS.index("quantity")] = 0
    return action


def main() -> None:
    args = _parse_args()
    if args.episodes <= 0 or args.steps <= 0:
        raise SystemExit("--episodes and --steps must be positive")
    rng = np.random.default_rng(args.seed)
    env = FleMacroEnv(
        port=args.port,
        speed=args.speed,
        max_steps=args.steps,
        seed=args.seed,
        log_path=args.log_path,
        clamp_quantity=args.clamp_quantity,
        clamp_item=args.clamp_item,
        clamp_anchor=args.clamp_anchor,
        clamp_connect=args.clamp_connect,
        regime=args.action_regime,
    )
    total_harvest_successes = 0
    try:
        for episode in range(args.episodes):
            deaths_before = env.deaths
            obs, _ = env.reset(seed=args.seed + episode)
            episode_start = time.perf_counter()
            infos: list[dict[str, Any]] = []
            for _ in range(args.steps):
                action = _random_acceptance_action(obs, rng)
                obs, _, terminated, truncated, info = env.step(action)
                infos.append(info)
                if terminated or truncated:
                    break
            wall_s = time.perf_counter() - episode_start
            ok_rate = sum(info["status"] == "ok" for info in infos) / len(infos)
            non_wait = [info for info in infos if info["op"] != "WAIT"]
            invalid = sum(
                info["status"] in {"tool_rejected", "no_effect"} for info in non_wait
            )
            invalid_rate = invalid / len(non_wait) if non_wait else 0.0
            harvest_successes = sum(
                info["op"] == "HARVEST" and info["status"] == "ok" for info in infos
            )
            place_infos = [info for info in infos if info["op"] == "PLACE"]
            craft_infos = [info for info in infos if info["op"] == "CRAFT"]
            place_ok = sum(info["status"] == "ok" for info in place_infos)
            craft_ok = sum(info["status"] == "ok" for info in craft_infos)
            place_ok_rate = place_ok / len(place_infos) if place_infos else 0.0
            craft_ok_rate = craft_ok / len(craft_infos) if craft_infos else 0.0
            total_harvest_successes += harvest_successes
            final = infos[-1]
            print(
                f"episode {episode + 1}: final automated PS={final['score_automated']:.3f} "
                f"player PS={final['score_player']:.3f} ok rate={ok_rate:.3f} "
                f"invalid-combination rate={invalid_rate:.3f} "
                f"steps/s={len(infos) / wall_s:.3f} "
                f"deaths={env.deaths - deaths_before} "
                f"PLACE ok rate={place_ok_rate:.3f} ({place_ok}/{len(place_infos)}) "
                f"CRAFT ok rate={craft_ok_rate:.3f} ({craft_ok}/{len(craft_infos)}) "
                f"HARVEST success count={harvest_successes}",
                flush=True,
            )
        print(
            f"HARVEST succeeded at least once: {total_harvest_successes > 0} "
            f"(total={total_harvest_successes})",
            flush=True,
        )
        if total_harvest_successes == 0:
            raise SystemExit("live acceptance failed: no successful HARVEST")
    finally:
        env.close()


if __name__ == "__main__":
    main()
