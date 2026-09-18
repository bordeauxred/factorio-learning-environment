"""SM-ARQ action coding and structural legality masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from fle.smarq import contract as C
from fle.smarq.contract import Action, Masks, Observation
from fle.smarq.obs import TensorClient
from fle.smarq.vocab import StableVocab


def _indices(values: Iterable[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


class ActionCodec:
    """Decode policy-head indices without changing spatial selections."""

    def __init__(self, vocab: StableVocab) -> None:
        self.vocab = vocab
        self._prototype = _indices(vocab.prototypes)
        self._item = _indices(vocab.items)
        self._recipe = _indices(vocab.recipes)
        self._technology = _indices(vocab.technologies)

    def decode(self, verb_index: int, heads: np.ndarray, observation: Observation) -> Action:
        if not 0 <= verb_index < C.N_VERBS:
            raise ValueError(f"verb index {verb_index} out of range")
        values = np.asarray(heads, dtype=np.int64)
        if values.shape != (len(C.HEADS),):
            raise ValueError(f"heads must have shape {(len(C.HEADS),)}")
        verb = C.VERBS[verb_index]
        action_heads = C.empty_heads()
        for name in C.HEAD_SEQUENCE[verb]:
            action_heads[C.HEAD_INDEX[name]] = values[C.HEAD_INDEX[name]]

        def selected(name: str, choices: tuple[Any, ...] | list[Any]) -> Any:
            index = int(action_heads[C.HEAD_INDEX[name]])
            if not 0 <= index < len(choices):
                raise ValueError(f"{name} index {index} out of range")
            return choices[index]

        tile = None
        if C.POSITION in C.HEAD_SEQUENCE[verb]:
            tile = C.position_to_tile(
                int(action_heads[C.HEAD_INDEX[C.POSITION]]),
                observation.raster_origin,
                observation.raster_tiles,
            )
        entity_slot = None
        entity_id = None
        if C.ENTITY in C.HEAD_SEQUENCE[verb]:
            entity_slot = int(action_heads[C.HEAD_INDEX[C.ENTITY]])
            if not 0 <= entity_slot < observation.entity_ids.shape[0]:
                raise ValueError(f"entity slot {entity_slot} out of range")
            entity_id = int(observation.entity_ids[entity_slot])
        return Action(
            verb=verb,
            heads=action_heads,
            tile=tile,
            prototype=(
                selected(C.PROTOTYPE, self.vocab.prototypes)
                if C.PROTOTYPE in C.HEAD_SEQUENCE[verb]
                else None
            ),
            direction=(
                selected(C.DIRECTION, C.DIRECTIONS)
                if C.DIRECTION in C.HEAD_SEQUENCE[verb]
                else None
            ),
            entity_slot=entity_slot,
            entity_id=entity_id,
            item=(
                selected(C.ITEM, self.vocab.items)
                if C.ITEM in C.HEAD_SEQUENCE[verb]
                else None
            ),
            quantity=(
                selected(C.QUANTITY, C.QUANTITIES)
                if C.QUANTITY in C.HEAD_SEQUENCE[verb]
                else None
            ),
            recipe=(
                selected(C.RECIPE, self.vocab.recipes)
                if C.RECIPE in C.HEAD_SEQUENCE[verb]
                else None
            ),
            technology=(
                selected(C.TECHNOLOGY, self.vocab.technologies)
                if C.TECHNOLOGY in C.HEAD_SEQUENCE[verb]
                else None
            ),
            duration_seconds=(
                selected(C.DURATION, C.DURATIONS_SECONDS)
                if C.DURATION in C.HEAD_SEQUENCE[verb]
                else None
            ),
        )

    def encode(self, action: Action, observation: Observation) -> tuple[int, np.ndarray]:
        if action.verb not in C.VERB_INDEX:
            raise ValueError(action.verb)
        heads = C.empty_heads()
        sequence = C.HEAD_SEQUENCE[action.verb]
        if C.POSITION in sequence:
            position = C.tile_to_position(action.tile, observation.raster_origin, observation.raster_tiles)
            if position is None:
                raise ValueError(f"tile {action.tile} outside current raster")
            heads[C.HEAD_INDEX[C.POSITION]] = position
        mappings = {
            C.PROTOTYPE: (action.prototype, self._prototype),
            C.DIRECTION: (action.direction, _indices(C.DIRECTIONS)),
            C.ITEM: (action.item, self._item),
            C.QUANTITY: (action.quantity, _indices(C.QUANTITIES)),
            C.RECIPE: (action.recipe, self._recipe),
            C.TECHNOLOGY: (action.technology, self._technology),
            C.DURATION: (action.duration_seconds, _indices(C.DURATIONS_SECONDS)),
        }
        for name, (value, mapping) in mappings.items():
            if name in sequence:
                if value not in mapping:
                    raise ValueError(f"unknown {name}: {value}")
                heads[C.HEAD_INDEX[name]] = mapping[value]
        if C.ENTITY in sequence:
            slot = action.entity_slot
            if action.entity_id:
                matches = np.nonzero(observation.entity_ids == action.entity_id)[0]
                if matches.size:
                    slot = int(matches[0])
            if slot is None or not 0 <= slot < C.ENTITY_SLOTS:
                raise ValueError(f"invalid entity slot: {slot}")
            heads[C.HEAD_INDEX[C.ENTITY]] = slot
        return C.VERB_INDEX[action.verb], heads


@dataclass(frozen=True)
class EntityCapability:
    holds_items: bool = False
    recipe_categories: frozenset[str] = frozenset()


@dataclass
class MaskMetadata:
    """Prototype facts only; never contains quantities or spatial feasibility."""

    placeable: set[str] = field(default_factory=set)
    recipe_category: dict[str, str] = field(default_factory=dict)
    hand_categories: set[str] = field(default_factory=lambda: {"crafting"})
    entity: dict[str, EntityCapability] = field(default_factory=dict)
    researched: set[str] = field(default_factory=set)

    @classmethod
    def from_rcon(cls, rcon: Any, vocab: StableVocab) -> "MaskMetadata":
        def command(lua: str) -> str:
            return rcon.send_command("/sc " + lua) or ""

        recipe_rows = command(
            "local a={} for n,p in pairs(prototypes.recipe) do "
            "a[#a+1]=n..','..p.category end table.sort(a) rcon.print(table.concat(a,'|'))"
        )
        recipe_category = {}
        for row in recipe_rows.split("|"):
            if row:
                name, category = row.split(",", 1)
                recipe_category[name] = category

        hand_rows = command(
            "local a={} local c=storage.utils.ensure_valid_character(1) "
            "for n,v in pairs(c.prototype.crafting_categories or {}) do "
            "if v then a[#a+1]=n end end table.sort(a) rcon.print(table.concat(a,'|'))"
        )
        entity_rows = command(
            "local a={} for n,p in pairs(prototypes.entity) do local c={} "
            "for k,v in pairs(p.crafting_categories or {}) do if v then c[#c+1]=k end end "
            "table.sort(c) a[#a+1]=n..','..p.type..','..table.concat(c,'+') end "
            "table.sort(a) rcon.print(table.concat(a,'|'))"
        )
        holder_types = {
            "container",
            "logistic-container",
            "furnace",
            "assembling-machine",
            "lab",
            "mining-drill",
            "boiler",
            "burner-generator",
            "reactor",
            "rocket-silo",
            "beacon",
            "ammo-turret",
            "artillery-turret",
            "cargo-wagon",
            "car",
            "spider-vehicle",
            "roboport",
        }
        entity = {}
        for row in entity_rows.split("|"):
            if not row:
                continue
            name, entity_type, categories = row.split(",", 2)
            entity[name] = EntityCapability(
                holds_items=entity_type in holder_types,
                recipe_categories=(
                    frozenset(filter(None, categories.split("+")))
                    if entity_type == "assembling-machine"
                    else frozenset()
                ),
            )
        researched_rows = command(
            "local a={} for n,t in pairs(game.forces.player.technologies) do "
            "if t.researched then a[#a+1]=n end end table.sort(a) "
            "rcon.print(table.concat(a,'|'))"
        )
        return cls(
            placeable=set(vocab.prototypes[1:]),
            recipe_category=recipe_category,
            hand_categories=set(filter(None, hand_rows.split("|"))),
            entity=entity,
            researched=set(filter(None, researched_rows.split("|"))),
        )


def build_masks(vocab: StableVocab, client: TensorClient, metadata: MaskMetadata) -> Masks:
    """Build type/syntax masks only; no resource or geometry state is consulted."""
    verb = np.ones(C.N_VERBS, dtype=bool)
    prototype = np.asarray(
        [index > 0 and name in metadata.placeable for index, name in enumerate(vocab.prototypes)],
        dtype=bool,
    )
    item = np.arange(len(vocab.items)) > 0
    craft_recipe = np.asarray(
        [
            index > 0 and metadata.recipe_category.get(name) in metadata.hand_categories
            for index, name in enumerate(vocab.recipes)
        ],
        dtype=bool,
    )
    technology = np.asarray(
        [index > 0 and name not in metadata.researched for index, name in enumerate(vocab.technologies)],
        dtype=bool,
    )
    entity = np.zeros((C.N_VERBS, C.ENTITY_SLOTS), dtype=bool)
    _, _, view, live, _ = client.observation()
    live = np.asarray(live, dtype=bool)
    for action_name in ("PICKUP", "ROTATE"):
        entity[C.VERB_INDEX[action_name]] = live
    recipe_masks: dict[int, np.ndarray] = {}
    for slot in np.nonzero(live)[0]:
        unit = str(int(client.entity_ids[slot]))
        row = client.entities.get(unit)
        if row is None:
            continue
        name = client._parse_row(row)["name"]
        capability = metadata.entity.get(name, EntityCapability())
        if capability.holds_items:
            entity[C.VERB_INDEX["INSERT"], slot] = True
            entity[C.VERB_INDEX["EXTRACT"], slot] = True
        if capability.recipe_categories:
            entity[C.VERB_INDEX["SET_RECIPE"], slot] = True
            recipe_masks[int(slot)] = np.asarray(
                [
                    index > 0
                    and metadata.recipe_category.get(recipe) in capability.recipe_categories
                    for index, recipe in enumerate(vocab.recipes)
                ],
                dtype=bool,
            )

    def recipes_for(slot: int) -> np.ndarray | None:
        return recipe_masks.get(int(slot))

    # ``view`` is intentionally unused beyond forcing the same current ordering
    # used by entity_ids; no feature magnitude affects legality.
    del view
    return Masks(
        verb=verb,
        prototype=prototype,
        item=item,
        craft_recipe=craft_recipe,
        technology=technology,
        entity=entity,
        recipe_for_entity=recipes_for,
    )
