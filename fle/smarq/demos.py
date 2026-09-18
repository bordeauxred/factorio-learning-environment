"""Scripted SM-ARQ demonstrations made only from semantic actions.

The scripts choose literal world tiles.  They do not query for placement
candidates or alter a coordinate after choosing it; consequently a bad script
coordinate fails in exactly the same way as a learned policy coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from fle.smarq import contract as C
from fle.smarq.actions import move_position_to_tile, tile_to_move_position


class DemoReplayProtocol(Protocol):
    def add(self, transition: C.Transition, priority: float = ...) -> Any: ...


def _vocab_index(values: Any, value: str, head: str) -> int:
    try:
        return int(tuple(values).index(value))
    except ValueError as exc:
        raise ValueError(f"{value!r} is not in the environment {head} vocabulary") from exc


def semantic_action(
    verb: str,
    observation: C.Observation,
    vocab: C.VocabProtocol,
    *,
    tile: tuple[int, int] | None = None,
    prototype: str | None = None,
    direction: str | None = None,
    entity_slot: int | None = None,
    item: str | None = None,
    quantity: int | str | None = None,
    recipe: str | None = None,
    technology: str | None = None,
    duration_seconds: int | None = None,
    coarse_move: bool = False,
) -> C.Action:
    """Encode one literal semantic action using the frozen head layout."""
    if verb not in C.VERB_INDEX:
        raise ValueError(f"unknown verb {verb!r}")
    heads = C.empty_heads()
    if C.POSITION in C.HEAD_SEQUENCE[verb]:
        if tile is None:
            raise ValueError(f"{verb} requires an exact tile")
        if verb == "MOVE_TO" and coarse_move:
            position = tile_to_move_position(tile, observation.player_tile)
            if position is not None:
                tile = move_position_to_tile(position, observation.player_tile)
        else:
            position = C.tile_to_position(
                tile, observation.raster_origin, observation.raster_tiles
            )
        if position is None:
            raise ValueError(
                f"literal tile {tile} is outside raster at {observation.raster_origin}"
            )
        heads[C.HEAD_INDEX[C.POSITION]] = position
    if C.PROTOTYPE in C.HEAD_SEQUENCE[verb]:
        heads[C.HEAD_INDEX[C.PROTOTYPE]] = _vocab_index(
            vocab.prototypes, prototype, C.PROTOTYPE
        )
    if C.DIRECTION in C.HEAD_SEQUENCE[verb]:
        try:
            heads[C.HEAD_INDEX[C.DIRECTION]] = C.DIRECTIONS.index(direction)
        except ValueError as exc:
            raise ValueError(f"unknown direction {direction!r}") from exc
    entity_id: int | None = None
    if C.ENTITY in C.HEAD_SEQUENCE[verb]:
        if entity_slot is None or not 0 <= entity_slot < len(observation.entity_ids):
            raise ValueError(f"invalid entity slot {entity_slot}")
        heads[C.HEAD_INDEX[C.ENTITY]] = entity_slot
        entity_id = int(observation.entity_ids[entity_slot])
    if C.ITEM in C.HEAD_SEQUENCE[verb]:
        heads[C.HEAD_INDEX[C.ITEM]] = _vocab_index(vocab.items, item, C.ITEM)
    if C.QUANTITY in C.HEAD_SEQUENCE[verb]:
        try:
            heads[C.HEAD_INDEX[C.QUANTITY]] = C.QUANTITIES.index(quantity)
        except ValueError as exc:
            raise ValueError(f"unsupported quantity {quantity!r}") from exc
    if C.RECIPE in C.HEAD_SEQUENCE[verb]:
        heads[C.HEAD_INDEX[C.RECIPE]] = _vocab_index(vocab.recipes, recipe, C.RECIPE)
    if C.TECHNOLOGY in C.HEAD_SEQUENCE[verb]:
        heads[C.HEAD_INDEX[C.TECHNOLOGY]] = _vocab_index(
            vocab.technologies, technology, C.TECHNOLOGY
        )
    if C.DURATION in C.HEAD_SEQUENCE[verb]:
        try:
            heads[C.HEAD_INDEX[C.DURATION]] = C.DURATIONS_SECONDS.index(duration_seconds)
        except ValueError as exc:
            raise ValueError(f"unsupported duration {duration_seconds!r}") from exc
    return C.Action(
        verb=verb,
        heads=heads,
        tile=tile,
        prototype=prototype,
        direction=direction,
        entity_slot=entity_slot,
        entity_id=entity_id,
        item=item,
        quantity=quantity,
        recipe=recipe,
        technology=technology,
        duration_seconds=duration_seconds,
    )


@dataclass(frozen=True)
class _ActionSpec:
    verb: str
    arguments: dict[str, Any]


class ScriptedDemo:
    """Base class for finite policies whose choices use the current observation."""

    def __init__(self, specs: list[_ActionSpec]) -> None:
        self._specs = tuple(specs)
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    @property
    def finished(self) -> bool:
        return self._index >= len(self._specs)

    def next_action(
        self,
        observation: C.Observation,
        masks: C.Masks,
        vocab: C.VocabProtocol,
        *,
        coarse_move: bool = False,
    ) -> C.Action:
        del masks  # Scripts are allowed to fail; masks are structural only.
        if self.finished:
            raise StopIteration
        spec = self._specs[self._index]
        self._index += 1
        return semantic_action(
            spec.verb,
            observation,
            vocab,
            coarse_move=coarse_move,
            **spec.arguments,
        )

    def __len__(self) -> int:
        return len(self._specs)


class BurnerAutomationDemo(ScriptedDemo):
    """Mine ore, place and fuel a furnace, wait, then extract exact output.

    ``ore_tile`` and ``furnace_tile`` are literal world coordinates.  The
    furnace entity is expected in ``furnace_entity_slot`` after placement.
    Set ``craft_furnace=False`` when the initial inventory is known to contain
    one; keeping it true is a safe demonstration of the conditional craft step.
    """

    def __init__(
        self,
        *,
        ore_tile: tuple[int, int] = (2, 0),
        furnace_tile: tuple[int, int] = (3, 0),
        furnace_entity_slot: int = 0,
        craft_furnace: bool = True,
        mine_quantity: int | str = 4,
        coal_quantity: int | str = 4,
        wait_seconds: int = 10,
        extract_quantity: int | str = "ALL",
    ) -> None:
        specs = [
            _ActionSpec("MOVE_TO", {"tile": ore_tile}),
            _ActionSpec("MINE", {"tile": ore_tile, "quantity": mine_quantity}),
        ]
        if craft_furnace:
            specs.append(
                _ActionSpec("CRAFT", {"recipe": "stone-furnace", "quantity": 1})
            )
        specs.extend(
            [
                _ActionSpec(
                    "PLACE",
                    {
                        "prototype": "stone-furnace",
                        "tile": furnace_tile,
                        "direction": "NORTH",
                    },
                ),
                _ActionSpec(
                    "INSERT",
                    {
                        "entity_slot": furnace_entity_slot,
                        "item": "coal",
                        "quantity": coal_quantity,
                    },
                ),
                _ActionSpec("FAST_FORWARD", {"duration_seconds": wait_seconds}),
                _ActionSpec(
                    "EXTRACT",
                    {
                        "entity_slot": furnace_entity_slot,
                        "item": "iron-plate",
                        "quantity": extract_quantity,
                    },
                ),
            ]
        )
        super().__init__(specs)


class HandMiningDemo(ScriptedDemo):
    """Weak baseline that only walks to and hand-mines explicit ore tiles."""

    def __init__(
        self,
        *,
        ore_tiles: tuple[tuple[int, int], ...] = ((2, 0), (3, 0), (4, 0)),
        quantity: int | str = 4,
    ) -> None:
        specs: list[_ActionSpec] = []
        for tile in ore_tiles:
            specs.append(_ActionSpec("MOVE_TO", {"tile": tile}))
            specs.append(_ActionSpec("MINE", {"tile": tile, "quantity": quantity}))
        super().__init__(specs)


def masks_as_dict(masks: C.Masks) -> dict[str, Any]:
    """Store replayable arrays, resolving contextual item/recipe callbacks."""
    result: dict[str, Any] = {
        "verb": masks.verb,
        "prototype": masks.prototype,
        "item": masks.item,
        "craft_recipe": masks.craft_recipe,
        "technology": masks.technology,
        "entity": masks.entity,
    }
    if callable(masks.item_for_entity):
        slots = np.flatnonzero(np.asarray(masks.entity, dtype=bool).any(axis=0))
        items: dict[int, np.ndarray] = {}
        for slot in slots:
            value = masks.item_for_entity(int(slot))
            if value is not None:
                items[int(slot)] = np.asarray(value, dtype=bool)
        result["item_for_entity"] = items
    elif masks.item_for_entity is not None:
        result["item_for_entity"] = masks.item_for_entity
    if callable(masks.recipe_for_entity):
        slots = np.flatnonzero(np.asarray(masks.entity, dtype=bool).any(axis=0))
        recipes: dict[int, np.ndarray] = {}
        for slot in slots:
            value = masks.recipe_for_entity(int(slot))
            if value is not None:
                recipes[int(slot)] = np.asarray(value, dtype=bool)
        result["recipe_for_entity"] = recipes
    elif masks.recipe_for_entity is not None:
        result["recipe_for_entity"] = masks.recipe_for_entity
    return result


def _replay_add(replay: Any, transition: C.Transition, priority: float) -> None:
    add = getattr(replay, "add", None) or getattr(replay, "push", None)
    if add is None:
        raise TypeError("replay must expose add() or push()")
    try:
        add(transition, priority=priority)
    except TypeError:
        try:
            add(transition, priority)
        except TypeError:
            add(transition)


@dataclass
class DemoCollection:
    transitions: list[C.Transition]
    actions: list[C.Action]
    total_reward: float
    initial_priority: float


def collect_demonstration(
    env: C.SemanticEnvProtocol,
    policy: ScriptedDemo,
    replay: DemoReplayProtocol | Any | None = None,
    *,
    seed: int | None = None,
    initial_priority: float = 10.0,
) -> DemoCollection:
    """Execute one script and optionally insert its off-policy transitions."""
    if initial_priority <= 1.0:
        raise ValueError("demonstration priority must be elevated above 1.0")
    observation, masks = env.reset(seed)
    policy.reset()
    transitions: list[C.Transition] = []
    actions: list[C.Action] = []
    total_reward = 0.0
    while not policy.finished:
        action = policy.next_action(
            observation,
            masks,
            env.vocab,
            coarse_move=bool(getattr(env, "coarse_move", False)),
        )
        result, next_masks = env.step(action)
        transition = C.Transition(
            observation=observation.as_dict(),
            action_heads=action.heads.copy(),
            verb=C.VERB_INDEX[action.verb],
            reward=result.reward,
            tau_seconds=result.duration_game_seconds,
            next_observation=result.observation.as_dict(),
            done=result.done,
            success=result.success,
            failure_reason=result.failure_reason,
            masks=masks_as_dict(masks),
            next_masks=masks_as_dict(next_masks),
            is_demo=True,
        )
        transitions.append(transition)
        actions.append(action)
        total_reward += result.reward
        if replay is not None:
            _replay_add(replay, transition, initial_priority)
        observation, masks = result.observation, next_masks
        if result.done:
            break
    return DemoCollection(transitions, actions, total_reward, initial_priority)
