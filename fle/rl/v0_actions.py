"""V0 autoregressive action decoding, masks, and exact execution.

This module deliberately does not import ``v0_obs``.  The observation package
is built in parallel and is passed here as ordinary mappings/objects, keeping
the action package independently testable without Factorio or RCON.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from fle.env.entities import Direction, Position
from fle.env.game_types import prototype_by_name
from fle.rl.ops import (
    RECIPE_NAMES,
    AnchorResolutionError,
    EntityIdentityError,
    SamplerState,
    _adjacent_position,
    _adjacent_tile_position,
    hand_craftable_recipes,
    resolve_entity_by_unit,
)
from fle.rl.v0_schema import (
    DIRECTIONS,
    FAST_FORWARD_SECONDS,
    HEAD_SIZES,
    HEADS,
    ITEM_NAMES,
    PLACE_MAX_AGE_TICKS,
    PLACEABLE_NAMES,
    QUANTITY_BINS,
    TECH_NAMES,
    VERB_HEADS,
    VERBS,
    blocked_mask_for_local,
    cell_to_world,
)
from fle.rl.v0_schema import RECIPE_NAMES as RECIPE_VOCAB

LOGGER = logging.getLogger(__name__)

MASK_NAMES = (
    "place_location",
    "place_item",
    "inventory_item",
    "contained_item",
    "entity",
    "recipe",
    "technology",
    "verb",
)
ENTITY_VERBS = frozenset({"PICKUP", "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE"})


class V0FailureReason(str, Enum):
    MASKED = "masked"
    INVALID_ACTION = "invalid_action"
    NAVIGATION_FAILED = "navigation_failed"
    TARGET_DISAPPEARED = "target_disappeared"
    IDENTITY_MISMATCH = "identity_mismatch"
    PLACEMENT_TARGET_CHANGED = "placement_target_changed"
    NO_SIM_TIME_ADVANCER = "no_sim_time_advancer"
    ENGINE_REJECTED = "engine_rejected"


@dataclass(frozen=True)
class V0Action:
    """One fully decoded semantic V0 action."""

    verb: str
    args: dict[str, Any]
    head_values: dict[str, int] = field(default_factory=dict)

    @property
    def op(self) -> str:
        """Compatibility alias for operation-oriented logging."""
        return self.verb


@dataclass(frozen=True)
class V0ExecutionResult:
    """SMDP result with learning reward and logging score kept separate."""

    tau: float
    reward: float
    general_score: float
    success: bool
    general_score_delta: float = 0.0
    reason: V0FailureReason | None = None
    message: str = ""
    tick_before: int = 0
    tick_after: int = 0
    requested_unit_number: int | None = None
    mutated_unit_number: int | None = None

    @property
    def status(self) -> str:
        return "ok" if self.success else "failed"

    @property
    def duration(self) -> float:
        return self.tau

    @property
    def reason_class(self) -> str | None:
        return self.reason.value if self.reason is not None else None

    @property
    def engine_message(self) -> str:
        return self.message


def _get(source: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if source is not None and hasattr(source, name):
            return getattr(source, name)
    return default


def _sources(obs: Any, context: Any) -> tuple[Any, ...]:
    snapshot = _get(context, "snapshot", "current")
    return context, snapshot, obs


def _first(sources: Sequence[Any], *names: str, default: Any = None) -> Any:
    for source in sources:
        value = _get(source, *names, default=None)
        if value is not None:
            return value
    return default


def _player_position(obs: Any, context: Any) -> tuple[float, float]:
    value = _first(_sources(obs, context), "player_position", "position")
    if value is None:
        raise ValueError("V0 decoding requires the current player position")
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    return float(value[0]), float(value[1])


def _head_mapping(head_values: Mapping[str, Any] | Sequence[Any]) -> dict[str, int]:
    if isinstance(head_values, Mapping):
        result = {str(key): int(value) for key, value in head_values.items()}
    else:
        values = tuple(int(value) for value in head_values)
        if len(values) == len(HEADS):
            result = dict(zip(HEADS, values, strict=True))
        elif values:
            verb_index = values[0]
            if not 0 <= verb_index < len(VERBS):
                raise ValueError(f"verb index {verb_index} outside V0 grammar")
            names = ("verb", *VERB_HEADS[VERBS[verb_index]])
            if len(values) != len(names):
                raise ValueError(
                    f"{VERBS[verb_index]} expects {len(names)} selected heads, "
                    f"got {len(values)}"
                )
            result = dict(zip(names, values, strict=True))
        else:
            raise ValueError("head_values may not be empty")
    if "verb" not in result:
        raise ValueError("head_values is missing verb")
    return result


def _entity_rows(obs: Any, context: Any) -> tuple[Any | None, ...]:
    rows = _first(
        _sources(obs, context),
        "entity_rows",
        "entities_by_slot",
        "entities",
        default=(),
    )
    if isinstance(rows, Mapping):
        # A world unit-number mapping is not an entity-slot table.
        slots = _first(_sources(obs, context), "entity_slots", default=None)
        if slots is not None:
            return tuple(slots)
        return tuple(rows.values())
    return tuple(rows)


def _row_unit(row: Any) -> int:
    value = _get(row, "unit", "unit_number", "id")
    if value is None:
        raise ValueError("selected entity row has no unit number")
    return int(value)


def _row_anchor(row: Any) -> dict[str, Any]:
    position = _get(row, "position")
    x = _get(row, "x", default=_get(position, "x"))
    y = _get(row, "y", default=_get(position, "y"))
    return {
        "unit": _row_unit(row),
        "unit_number": _row_unit(row),
        "name": str(_get(row, "name", default="")),
        "x": float(x),
        "y": float(y),
        "direction": int(_get(row, "direction", default=0)),
    }


def _place_prototype(item: str, context: Any) -> str:
    place_results = _get(context, "place_results", default=None)
    if place_results is None:
        vocab = _get(context, "vocab")
        place_results = _get(vocab, "place_results", default={})
    return str(place_results.get(item, item))


def _placement_position(
    tile: tuple[int, int], prototype: str, direction: int, context: Any
) -> tuple[float, float]:
    vocab = _get(context, "vocab")
    entity_info = _get(context, "entity_info", default=None)
    if entity_info is None:
        entity_info = _get(vocab, "entity_info", default={})
    info = entity_info.get(prototype, {}) if entity_info is not None else {}
    width = int(info.get("tile_width") or 1)
    height = int(info.get("tile_height") or 1)
    if direction in {4, 12}:
        width, height = height, width
    x = float(tile[0]) + 0.5 if width % 2 else float(tile[0])
    y = float(tile[1]) + 0.5 if height % 2 else float(tile[1])
    return x, y


def decode_v0_action(
    head_values: Mapping[str, Any] | Sequence[Any],
    obs: Any,
    context: Any,
) -> V0Action:
    """Decode selected heads in the fixed ``VERB_HEADS`` order."""
    values = _head_mapping(head_values)
    verb_index = values["verb"]
    if not 0 <= verb_index < len(VERBS):
        raise ValueError(f"verb index {verb_index} outside V0 grammar")
    verb = VERBS[verb_index]
    required = VERB_HEADS[verb]
    missing = [head for head in required if head not in values]
    if missing:
        raise ValueError(f"{verb} is missing heads: {', '.join(missing)}")
    for head in required:
        if not 0 <= values[head] < HEAD_SIZES[head]:
            raise ValueError(f"{head} index {values[head]} outside V0 grammar")

    args: dict[str, Any] = {}
    if "location" in required:
        player_x, player_y = _player_position(obs, context)
        tile = cell_to_world(values["location"], player_x, player_y)
        args["tile"] = tile
        args["world_tile"] = tile
        args["location"] = values["location"]
    if "quantity" in required:
        args["quantity"] = QUANTITY_BINS[values["quantity"]]
    if "direction" in required:
        args["direction"] = DIRECTIONS[values["direction"]]
    if "place_item" in required:
        item = PLACEABLE_NAMES[values["place_item"]]
        prototype = _place_prototype(item, context)
        args.update(item=item, place_item=item, prototype=prototype)
        args["target"] = _placement_position(
            args["tile"], prototype, args["direction"], context
        )
    elif "location" in required:
        args["target"] = args["tile"]
    if "entity" in required:
        rows = _entity_rows(obs, context)
        slot = values["entity"]
        row = rows[slot] if slot < len(rows) else None
        args["entity_slot"] = slot
        args["anchor"] = _row_anchor(row) if row is not None else None
        args["unit_number"] = _row_unit(row) if row is not None else None
    if "inventory_item" in required:
        args["item"] = ITEM_NAMES[values["inventory_item"]]
        args["inventory_item"] = args["item"]
    if "contained_item" in required:
        args["item"] = ITEM_NAMES[values["contained_item"]]
        args["contained_item"] = args["item"]
    if "recipe" in required:
        args["recipe"] = RECIPE_VOCAB[values["recipe"]]
    if "technology" in required:
        args["technology"] = TECH_NAMES[values["technology"]]
    if "duration" in required:
        args["seconds"] = FAST_FORWARD_SECONDS[values["duration"]]
        args["duration"] = args["seconds"]
        args["ticks"] = 60 * args["seconds"]
    selected = {"verb": verb_index, **{head: values[head] for head in required}}
    return V0Action(verb=verb, args=args, head_values=selected)


def _prefix_mapping(prefix: Any) -> dict[str, int]:
    if prefix is None:
        return {}
    if isinstance(prefix, V0Action):
        return dict(prefix.head_values)
    if isinstance(prefix, Mapping):
        return {str(key): int(value) for key, value in prefix.items()}
    values = tuple(int(value) for value in prefix)
    if not values:
        return {}
    verb = VERBS[values[0]]
    names = ("verb", *VERB_HEADS[verb][: len(values) - 1])
    return dict(zip(names, values, strict=True))


def _inventory(obs: Any, context: Any) -> Mapping[str, int | float]:
    inventory = _first(_sources(obs, context), "inventory_counts", "inventory")
    if isinstance(inventory, Mapping):
        return inventory
    if inventory is None:
        return {}
    array = np.asarray(inventory)
    return {name: float(array[index]) for index, name in enumerate(ITEM_NAMES)}


def _row_contents(row: Any) -> Mapping[str, int | float]:
    items = _get(row, "items", "contents", default=None)
    if items is not None:
        return items
    result: dict[str, int | float] = {}
    for inventory in _get(row, "inventories", default={}).values():
        for name, count in inventory.items():
            result[name] = result.get(name, 0) + count
    return result


def _counter(context: Any) -> MutableMapping[str, int] | None:
    counters = _get(context, "mask_counters", "metrics", default=None)
    return counters if isinstance(counters, MutableMapping) else None


def _count(context: Any, key: str, amount: int = 1) -> None:
    counters = _counter(context)
    if counters is not None:
        counters[key] = int(counters.get(key, 0)) + amount


def _mask_enabled(context: Any, name: str) -> bool:
    toggles = _get(context, "mask_toggles", "masks_enabled", default={})
    if isinstance(toggles, Mapping):
        enabled = bool(toggles.get(name, toggles.get(f"mask_{name}", True)))
    else:
        enabled = bool(_get(context, f"mask_{name}", default=True))
    _count(context, f"mask/{name}/{'enabled' if enabled else 'disabled'}")
    return enabled


def _selected_row(values: Mapping[str, int], obs: Any, context: Any) -> Any | None:
    slot = values.get("entity")
    rows = _entity_rows(obs, context)
    return rows[slot] if slot is not None and slot < len(rows) else None


def _world_entities(world: Any) -> Mapping[int, Any]:
    entities = _get(world, "entities", default={})
    return entities if isinstance(entities, Mapping) else {}


def v0_masks_for_prefix(
    prefix: Any,
    obs: Any,
    world: Any,
    context: Any,
) -> dict[str, np.ndarray]:
    """Return allowed-value masks conditioned on the selected prefix.

    Every returned array uses 1 for allowed and 0 for forbidden.  Unknown,
    stale, and out-of-window buildability cells remain allowed by delegating
    the crop/freshness rule to :func:`blocked_mask_for_local`.
    """
    values = _prefix_mapping(prefix)
    masks = {head: np.ones(size, dtype=np.uint8) for head, size in HEAD_SIZES.items()}
    enabled_flags = {name: _mask_enabled(context, name) for name in MASK_NAMES}
    inventory = _inventory(obs, context)

    if enabled_flags["place_item"]:
        masks["place_item"] = np.fromiter(
            (inventory.get(name, 0) > 0 for name in PLACEABLE_NAMES),
            dtype=np.uint8,
            count=len(PLACEABLE_NAMES),
        )
    if enabled_flags["inventory_item"]:
        masks["inventory_item"] = np.fromiter(
            (inventory.get(name, 0) > 0 for name in ITEM_NAMES),
            dtype=np.uint8,
            count=len(ITEM_NAMES),
        )
    if enabled_flags["entity"]:
        rows = _entity_rows(obs, context)
        live = _world_entities(world)
        has_live_cache = isinstance(_get(world, "entities", default=None), Mapping)
        entity_mask = np.zeros(HEAD_SIZES["entity"], dtype=np.uint8)
        for slot, row in enumerate(rows[: HEAD_SIZES["entity"]]):
            if row is not None and (not has_live_cache or _row_unit(row) in live):
                entity_mask[slot] = 1
        masks["entity"] = entity_mask
    if enabled_flags["contained_item"] and "entity" in values:
        row = _selected_row(values, obs, context)
        if row is not None:
            live_row = _world_entities(world).get(_row_unit(row), row)
            contents = _row_contents(live_row)
            masks["contained_item"] = np.fromiter(
                (contents.get(name, 0) > 0 for name in ITEM_NAMES),
                dtype=np.uint8,
                count=len(ITEM_NAMES),
            )
        else:
            masks["contained_item"].fill(0)
    if enabled_flags["recipe"]:
        enabled = _first(
            _sources(obs, context), "hand_craftable_recipes", "craftable_recipes"
        )
        if enabled is None:
            enabled_recipes = _get(context, "enabled_recipes")
            vocab = _get(context, "vocab")
            if enabled_recipes is not None and vocab is not None:
                enabled = hand_craftable_recipes(
                    SamplerState(
                        world=world,
                        player_pos=_player_position(obs, context),
                        inventory=inventory,
                        enabled_recipes=enabled_recipes,
                        research_state=_get(context, "research_state", default={}),
                        vocab=vocab,
                        crafting_categories=_get(
                            context,
                            "crafting_categories",
                            default=("crafting",),
                        ),
                    )
                )
        if enabled is None:
            recipe_array = _get(obs, "recipes_enabled")
            if recipe_array is not None:
                masks["recipe"] = (np.asarray(recipe_array) > 0).astype(np.uint8)
        else:
            enabled_names = set(enabled)
            masks["recipe"] = np.fromiter(
                (name in enabled_names for name in RECIPE_VOCAB),
                dtype=np.uint8,
                count=len(RECIPE_VOCAB),
            )
    if enabled_flags["technology"]:
        available = _first(
            _sources(obs, context),
            "available_technologies",
            "researchable_technologies",
        )
        if available is not None:
            available_names = set(available)
            masks["technology"] = np.fromiter(
                (name in available_names for name in TECH_NAMES),
                dtype=np.uint8,
                count=len(TECH_NAMES),
            )

    if (
        enabled_flags["place_location"]
        and values.get("verb") == VERBS.index("PLACE")
        and "place_item" in values
        and "direction" in values
    ):
        item = PLACEABLE_NAMES[values["place_item"]]
        prototype = _place_prototype(item, context)
        direction = DIRECTIONS[values["direction"]]
        cache = _first((context, world), "buildability", "buildability_cache")
        if cache is not None:
            try:
                channel = cache.channel_for(prototype, direction)
            except KeyError:
                _count(context, "build/mask_lookup_miss")
            else:
                tick = int(_first(_sources(obs, context), "tick", default=0))
                player_x, player_y = _player_position(obs, context)
                max_age = int(
                    _get(context, "place_max_age_ticks", default=PLACE_MAX_AGE_TICKS)
                )
                blocked = blocked_mask_for_local(
                    cache, channel, tick, player_x, player_y, max_age
                )
                masks["location"] = (~blocked.reshape(-1)).astype(np.uint8)

    if enabled_flags["verb"]:
        for verb_index, verb in enumerate(VERBS):
            if any(not masks[head].any() for head in VERB_HEADS[verb]):
                masks["verb"][verb_index] = 0
        if enabled_flags["contained_item"]:
            has_extractable_target = any(
                row is not None
                and any(count > 0 for count in _row_contents(row).values())
                for row in _entity_rows(obs, context)
            )
            if not has_extractable_target:
                masks["verb"][VERBS.index("EXTRACT")] = 0

    for name in MASK_NAMES:
        head = "location" if name == "place_location" else name
        if head in masks:
            _count(context, f"mask/{name}/forbidden", int((masks[head] == 0).sum()))
    return masks


def _read_tick(world: Any) -> int:
    reader = _get(world, "read_tick")
    if callable(reader):
        return int(reader())
    return int(_get(world, "tick", default=0))


def _read_scores(namespace: Any, world: Any, context: Any) -> tuple[float, float]:
    score = _get(namespace, "score")
    if callable(score):
        general, automated = score()
        return float(general), float(automated)
    values = _get(context, "scores")
    if callable(values):
        values = values()
    if values is not None:
        if isinstance(values, Mapping):
            return float(values["general"]), float(values["automated"])
        return float(values[0]), float(values[1])
    general = float(_get(world, "score_player", "general_score", default=0.0))
    automated = float(_get(world, "score_automated", "automated_score", default=0.0))
    return general, automated


def _drain(world: Any) -> None:
    drain = _get(world, "all_drain")
    if callable(drain):
        drain()


def _position_xy(value: Any) -> tuple[float, float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    return float(value[0]), float(value[1])


def _world_player_position(world: Any) -> tuple[float, float]:
    player_x = _get(world, "player_x", default=None)
    player_y = _get(world, "player_y", default=None)
    if player_x is not None and player_y is not None:
        return float(player_x), float(player_y)
    return _position_xy(world.read_player_pos())


def _navigate_tile(
    tile: tuple[int, int], namespace: Any, world: Any, reach: float
) -> None:
    px, py = _world_player_position(world)
    target_x, target_y = tile[0] + 0.5, tile[1] + 0.5
    if math.hypot(px - target_x, py - target_y) <= reach - 0.1:
        return
    destination = _adjacent_tile_position(tile[0], tile[1], world, px, py)
    namespace.move_to(destination)
    _drain(world)
    px, py = _world_player_position(world)
    if math.hypot(px - target_x, py - target_y) > reach - 0.1:
        raise RuntimeError("navigation did not enter interaction range")


def _navigate_entity(row: Any, namespace: Any, world: Any, reach: float) -> None:
    px, py = _world_player_position(world)
    if math.hypot(px - row.x, py - row.y) <= reach - 0.1:
        return
    namespace.move_to(_adjacent_position(row, world, px, py))
    _drain(world)
    live = _world_entities(world).get(row.unit)
    if live is None:
        replacement = next(
            (
                candidate
                for candidate in _world_entities(world).values()
                if candidate.name == row.name
                and math.hypot(candidate.x - row.x, candidate.y - row.y) <= 0.75
            ),
            None,
        )
        if replacement is not None:
            raise EntityIdentityError(
                "entity_identity_mismatch after navigation: "
                f"requested unit {row.unit}, live unit {replacement.unit}",
                requested_unit_number=row.unit,
                mutated_unit_number=replacement.unit,
            )
        raise AnchorResolutionError(
            f"anchor_resolution_failed: unit {row.unit} disappeared after navigation"
        )
    px, py = _world_player_position(world)
    if math.hypot(px - live.x, py - live.y) > reach - 0.1:
        raise RuntimeError("navigation did not enter interaction range")


def _quantity(action: V0Action, row: Any | None, inventory: Mapping[str, Any]) -> int:
    value = action.args.get("quantity", 1)
    if value != "ALL":
        return int(value)
    if action.verb == "INSERT":
        return int(inventory.get(action.args["item"], 0))
    if action.verb == "EXTRACT" and row is not None:
        return int(_row_contents(row).get(action.args["item"], 0))
    if action.verb == "MINE":
        return 2**31 - 1
    return -1


def _advance_simulated_ticks(
    ticks: int, namespace: Any, world: Any, context: Any
) -> None:
    candidates = (
        (_get(context, "advance_simulated_ticks"), (ticks,)),
        (_get(world, "advance_simulated_ticks", "advance_ticks"), (ticks,)),
        (_get(namespace, "advance_simulated_ticks", "advance_ticks"), (ticks,)),
        (_get(namespace, "fast_forward"), (ticks / 60.0,)),
    )
    for function, args in candidates:
        if callable(function):
            function(*args)
            return
    raise RuntimeError("no simulated-time advancement API is available")


def _craft_quantity(
    action: V0Action,
    world: Any,
    context: Any,
    inventory: Mapping[str, int | float],
) -> int:
    value = action.args.get("quantity", 1)
    if value != "ALL":
        return int(value)
    vocab = _get(context, "vocab")
    enabled_recipes = _get(context, "enabled_recipes", default=())
    if vocab is None:
        raise ValueError("CRAFT ALL requires vocabulary metadata")
    state = SamplerState(
        world=world,
        player_pos=_player_position(None, context),
        inventory=inventory,
        enabled_recipes=enabled_recipes,
        research_state=_get(context, "research_state", default={}),
        vocab=vocab,
        crafting_categories=_get(context, "crafting_categories", default=("crafting",)),
    )
    recipe = action.args["recipe"]

    def admitted(count: int) -> bool:
        return recipe in hand_craftable_recipes(state, count=count)

    if not admitted(1):
        return 0
    low, high = 1, 2
    while high < 2**20 and admitted(high):
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if admitted(middle):
            low = middle
        else:
            high = middle
    return low


def _result_unit(result: Any) -> int | None:
    if result is None or isinstance(result, (bool, int, float, str)):
        return None
    value = _get(result, "unit_number", "unit", "id", default=None)
    return int(value) if value is not None else None


def _log_failure(context: Any, action: V0Action, result: V0ExecutionResult) -> None:
    record = {
        "verb": action.verb,
        "reason": result.reason.value if result.reason else None,
        "message": result.message,
        "requested_unit_number": result.requested_unit_number,
        "mutated_unit_number": result.mutated_unit_number,
    }
    logger = _get(context, "log_action", "logger")
    if callable(logger):
        logger(record)
    elif isinstance(logger, list):
        logger.append(record)
    LOGGER.warning("V0 action failed: %s", record)


def _masked_head(action: V0Action, obs: Any, world: Any, context: Any) -> str | None:
    if obs is None or not action.head_values:
        return None
    prefix: dict[str, int] = {}
    selected = (("verb", action.head_values["verb"]),)
    selected += tuple(
        (head, action.head_values[head]) for head in VERB_HEADS[action.verb]
    )
    for head, value in selected:
        masks = v0_masks_for_prefix(prefix, obs, world, context)
        if not masks[head][value]:
            return head
        prefix[head] = value
    return None


def execute_v0_action(
    action: V0Action,
    namespace: Any,
    world: Any,
    context: Any,
    *,
    obs: Any = None,
) -> V0ExecutionResult:
    """Execute one V0 semantic action with strict identity and SMDP timing."""
    deferred_measurement = bool(
        _get(context, "defer_result_measurement", default=False)
    )
    snapshot = _get(context, "snapshot", "current")
    if deferred_measurement and snapshot is not None:
        tick_before = int(_get(snapshot, "tick", default=0))
        _general_before = float(_get(snapshot, "score_player", default=0.0))
        automated_before = float(_get(snapshot, "score_automated", default=0.0))
    else:
        tick_before = _read_tick(world)
        _general_before, automated_before = _read_scores(namespace, world, context)
    requested_unit: int | None = None
    mutated_unit: int | None = None

    def finish(
        success: bool,
        reason: V0FailureReason | None = None,
        message: str = "",
    ) -> V0ExecutionResult:
        if deferred_measurement:
            tick_after = tick_before
            general_after = _general_before
            automated_after = automated_before
        else:
            tick_after = _read_tick(world)
            general_after, automated_after = _read_scores(namespace, world, context)
        result = V0ExecutionResult(
            tau=max(0, tick_after - tick_before) / 60.0,
            reward=automated_after - automated_before,
            general_score=general_after,
            general_score_delta=general_after - _general_before,
            success=success,
            reason=reason,
            message=message,
            tick_before=tick_before,
            tick_after=tick_after,
            requested_unit_number=requested_unit,
            mutated_unit_number=mutated_unit,
        )
        if not success:
            _log_failure(context, action, result)
        return result

    masked = _masked_head(action, obs, world, context)
    if masked is not None:
        return finish(False, V0FailureReason.MASKED, f"masked_{masked}")

    try:
        verb, args = action.verb, action.args
        resource_reach = float(_get(context, "resource_reach", default=2.7))
        build_reach = float(_get(context, "build_reach", default=10.0))
        inventory = _inventory(obs, context)
        if verb == "MOVE_TO":
            namespace.move_to(
                Position(x=args["tile"][0] + 0.5, y=args["tile"][1] + 0.5)
            )
        elif verb == "MINE":
            _navigate_tile(args["tile"], namespace, world, resource_reach)
            tile = tuple(args["tile"])
            if (
                tile not in _get(world, "ores", default={})
                and tile not in _get(world, "trees", default=set())
                and tile not in _get(world, "obstacles", default=set())
            ):
                return finish(
                    False,
                    V0FailureReason.ENGINE_REJECTED,
                    f"no minable target at selected tile {tile}",
                )
            namespace.harvest_resource(
                Position(x=args["tile"][0] + 0.5, y=args["tile"][1] + 0.5),
                _quantity(action, None, inventory),
            )
        elif verb == "PLACE":
            selected_tile = tuple(args["tile"])
            selected_target = tuple(args["target"])
            _navigate_tile(selected_tile, namespace, world, build_reach)
            expected = _placement_position(
                selected_tile, args["prototype"], args["direction"], context
            )
            if selected_target != expected or tuple(args["tile"]) != selected_tile:
                return finish(
                    False,
                    V0FailureReason.PLACEMENT_TARGET_CHANGED,
                    f"placement target changed from {selected_target} to {expected}",
                )
            namespace.place_entity(
                prototype_by_name.get(args["prototype"], args["prototype"]),
                Direction(args["direction"]),
                Position(x=selected_target[0], y=selected_target[1]),
                exact=True,
            )
        elif verb == "CRAFT":
            namespace.craft_item(
                args["recipe"], _craft_quantity(action, world, context, inventory)
            )
        elif verb in ENTITY_VERBS:
            if args["anchor"] is None:
                return finish(
                    False,
                    V0FailureReason.INVALID_ACTION,
                    f"selected entity slot {args['entity_slot']} is empty",
                )
            requested_unit = int(args["unit_number"])
            row = _world_entities(world).get(requested_unit)
            if row is None:
                return finish(
                    False,
                    V0FailureReason.TARGET_DISAPPEARED,
                    f"unit {requested_unit} is not live",
                )
            _navigate_entity(row, namespace, world, build_reach)
            row = _world_entities(world).get(requested_unit)
            if row is None:
                return finish(
                    False,
                    V0FailureReason.TARGET_DISAPPEARED,
                    f"unit {requested_unit} disappeared during navigation",
                )
            entity = resolve_entity_by_unit(namespace, world, requested_unit)
            mutated_unit = int(_get(entity, "unit_number", "unit", "id", default=-1))
            if requested_unit != mutated_unit:
                return finish(
                    False,
                    V0FailureReason.IDENTITY_MISMATCH,
                    f"requested unit {requested_unit}, resolved unit {mutated_unit}",
                )
            tool_result = None
            if verb == "PICKUP":
                tool_result = namespace.pickup_entity(entity)
            elif verb == "ROTATE":
                tool_result = namespace.rotate_entity(
                    entity, Direction(args["direction"])
                )
            elif verb == "INSERT":
                tool_result = namespace.insert_item(
                    prototype_by_name.get(args["item"], args["item"]),
                    entity,
                    _quantity(action, row, inventory),
                )
            elif verb == "EXTRACT":
                tool_result = namespace.extract_item(
                    prototype_by_name.get(args["item"], args["item"]),
                    entity,
                    _quantity(action, row, inventory),
                )
            else:
                recipe = prototype_by_name.get(
                    args["recipe"], RECIPE_NAMES.get(args["recipe"], args["recipe"])
                )
                tool_result = namespace.set_entity_recipe(entity, recipe)
            returned_unit = _result_unit(tool_result)
            if returned_unit is not None:
                mutated_unit = returned_unit
            if requested_unit != mutated_unit:
                raise EntityIdentityError(
                    f"requested unit {requested_unit}, mutated unit {mutated_unit}",
                    requested_unit_number=requested_unit,
                    mutated_unit_number=mutated_unit,
                )
        elif verb == "RESEARCH":
            namespace.set_research(args["technology"])
        elif verb == "FAST_FORWARD":
            _advance_simulated_ticks(args["ticks"], namespace, world, context)
        else:
            return finish(
                False, V0FailureReason.INVALID_ACTION, f"unknown V0 verb {verb}"
            )
        if not _get(context, "defer_final_drain", default=False):
            _drain(world)
        return finish(True)
    except EntityIdentityError as exc:
        if exc.requested_unit_number is not None:
            requested_unit = exc.requested_unit_number
        if exc.mutated_unit_number is not None:
            mutated_unit = exc.mutated_unit_number
        return finish(False, V0FailureReason.IDENTITY_MISMATCH, str(exc))
    except AnchorResolutionError as exc:
        return finish(False, V0FailureReason.TARGET_DISAPPEARED, str(exc))
    except Exception as exc:  # noqa: BLE001 - engine tools raise plain Exception
        reason = (
            V0FailureReason.NO_SIM_TIME_ADVANCER
            if action.verb == "FAST_FORWARD"
            and "simulated-time advancement" in str(exc)
            else V0FailureReason.NAVIGATION_FAILED
            if "navigation" in str(exc)
            else V0FailureReason.ENGINE_REJECTED
        )
        return finish(False, reason, str(exc)[:500])


__all__ = [
    "MASK_NAMES",
    "V0Action",
    "V0ExecutionResult",
    "V0FailureReason",
    "decode_v0_action",
    "execute_v0_action",
    "v0_masks_for_prefix",
]
