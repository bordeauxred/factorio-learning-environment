"""SM-ARQ action coding and structural legality masks.

With ``coarse_move`` enabled, MOVE_TO reuses the frozen POSITION action slot
but interprets its index over Factorio's existing 96x96 observation grid at
three tiles per cell.  This gives locomotion a distinct 288-tile field without
changing any contract head index or size when the flag is off.  All factory
geometry verbs continue to decode POSITION against the exact-tile raster.
"""

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


def move_position_to_tile(index: int, player_tile: tuple[int, int]) -> tuple[int, int]:
    """Decode one locomotion cell to its absolute three-tile grid anchor."""
    if not 0 <= index < C.GRID_SIZE * C.GRID_SIZE:
        raise ValueError(f"move position index {index} out of range")
    x = index % C.GRID_SIZE
    y = index // C.GRID_SIZE
    half = C.GRID_SIZE // 2
    return (
        int(player_tile[0] + (x - half) * C.GRID_CELL_TILES),
        int(player_tile[1] + (y - half) * C.GRID_CELL_TILES),
    )


def tile_to_move_position(tile: tuple[int, int], player_tile: tuple[int, int]) -> int | None:
    """Encode a world tile to the nearest locomotion cell, if in range."""
    half = C.GRID_SIZE // 2
    dx = round((tile[0] - player_tile[0]) / C.GRID_CELL_TILES)
    dy = round((tile[1] - player_tile[1]) / C.GRID_CELL_TILES)
    x, y = int(dx + half), int(dy + half)
    if not (0 <= x < C.GRID_SIZE and 0 <= y < C.GRID_SIZE):
        return None
    return y * C.GRID_SIZE + x


class ActionCodec:
    """Decode policy-head indices without changing spatial selections."""

    def __init__(self, vocab: StableVocab, *, coarse_move: bool = False) -> None:
        self.vocab = vocab
        self.coarse_move = bool(coarse_move)
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
            position = int(action_heads[C.HEAD_INDEX[C.POSITION]])
            if verb == "MOVE_TO" and self.coarse_move:
                tile = move_position_to_tile(position, observation.player_tile)
            else:
                tile = C.position_to_tile(
                    position,
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
            if action.verb == "MOVE_TO" and self.coarse_move:
                position = tile_to_move_position(action.tile, observation.player_tile)
            else:
                position = C.tile_to_position(
                    action.tile, observation.raster_origin, observation.raster_tiles
                )
            if position is None:
                window = "coarse move grid" if self.coarse_move else "current raster"
                raise ValueError(f"tile {action.tile} outside {window}")
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
    accepted_items: frozenset[str] | None = frozenset()
    recipe_selected_inputs: bool = False


@dataclass
class MaskMetadata:
    """Prototype facts only; never contains quantities or spatial feasibility."""

    placeable: set[str] = field(default_factory=set)
    recipe_category: dict[str, str] = field(default_factory=dict)
    hand_categories: set[str] = field(default_factory=lambda: {"crafting"})
    entity: dict[str, EntityCapability] = field(default_factory=dict)
    researched: set[str] = field(default_factory=set)
    startable_technologies: set[str] = field(default_factory=set)
    recipe_ingredients: dict[str, frozenset[str]] = field(default_factory=dict)

    def refresh_research(self, rcon: Any) -> None:
        """Refresh force state while retaining cached prototype compatibility."""
        response = rcon.send_command(
            "/sc local f=game.forces.player local labs=0 "
            "for _,e in pairs(game.surfaces[1].find_entities_filtered{force=f,type='lab'}) do "
            "if e.valid then labs=labs+1 end end local done={} local start={} "
            "for n,t in pairs(f.technologies) do if t.researched then done[#done+1]=n "
            "elseif labs>0 and t.enabled then local ok=true for _,p in pairs(t.prerequisites) do "
            "if not p.researched then ok=false break end end if ok then start[#start+1]=n end end end "
            "table.sort(done) table.sort(start) "
            "rcon.print(table.concat(done,'|')..';'..table.concat(start,'|'))"
        ) or ";"
        researched, _, startable = response.partition(";")
        self.researched = set(filter(None, researched.split("|")))
        self.startable_technologies = set(filter(None, startable.split("|")))

    @classmethod
    def from_rcon(cls, rcon: Any, vocab: StableVocab) -> "MaskMetadata":
        def command(lua: str) -> str:
            return rcon.send_command("/sc " + lua) or ""

        recipe_rows = command(
            "local a={} for n,p in pairs(prototypes.recipe) do local i={} "
            "for _,v in pairs(p.ingredients or {}) do if v.type=='item' then i[#i+1]=v.name end end "
            "table.sort(i) a[#a+1]=n..','..p.category..','..table.concat(i,'+') end "
            "table.sort(a) rcon.print(table.concat(a,'|'))"
        )
        recipe_category = {}
        recipe_ingredients: dict[str, frozenset[str]] = {}
        for row in recipe_rows.split("|"):
            if row:
                name, category, ingredients = row.split(",", 2)
                recipe_category[name] = category
                recipe_ingredients[name] = frozenset(filter(None, ingredients.split("+")))

        hand_rows = command(
            "local a={} local c=storage.utils.ensure_valid_character(1) "
            "for n,v in pairs(c.prototype.crafting_categories or {}) do "
            "if v then a[#a+1]=n end end table.sort(a) rcon.print(table.concat(a,'|'))"
        )
        technology_input_rows = command(
            "local a={} for _,t in pairs(game.forces.player.technologies) do "
            "for _,v in pairs(t.research_unit_ingredients or {}) do a[#a+1]=v.name end end "
            "table.sort(a) rcon.print(table.concat(a,'|'))"
        )
        item_rows = command(
            "local a={} for n,p in pairs(prototypes.item) do local fuel=p.fuel_category or '' "
            "local ammo='' local module='' if p.type=='ammo' then ammo=p.ammo_category.name "
            "elseif p.type=='module' then module=p.category end "
            "a[#a+1]=n..','..fuel..','..ammo..','..module end "
            "table.sort(a) rcon.print(table.concat(a,'|'))"
        )
        fuel_items: dict[str, set[str]] = {}
        ammo_items: dict[str, set[str]] = {}
        module_items: set[str] = set()
        for row in item_rows.split("|"):
            if not row:
                continue
            name, fuel, ammo, module = row.split(",", 3)
            if fuel:
                fuel_items.setdefault(fuel, set()).add(name)
            if ammo:
                ammo_items.setdefault(ammo, set()).add(name)
            if module:
                module_items.add(name)

        entity_rows = command(
            "local a={} for n,p in pairs(prototypes.entity) do local c={} local f={} local m={} "
            "for k,v in pairs(p.crafting_categories or {}) do if v then c[#c+1]=k end end "
            "if p.burner_prototype then for k,v in pairs(p.burner_prototype.fuel_categories) do "
            "if v then f[#f+1]=k end end end "
            "if p.attack_parameters then for _,v in pairs(p.attack_parameters.ammo_categories or {}) do "
            "m[#m+1]=v end end table.sort(c) table.sort(f) table.sort(m) "
            "a[#a+1]=n..','..p.type..','..table.concat(c,'+')..','..table.concat(f,'+')..','.."
            "table.concat(m,'+')..','..p.module_inventory_size end "
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
        unrestricted_types = {
            "car",
            "cargo-wagon",
            "container",
            "logistic-container",
            "spider-vehicle",
        }
        lab_inputs = set(filter(None, technology_input_rows.split("|")))
        entity = {}
        for row in entity_rows.split("|"):
            if not row:
                continue
            name, entity_type, categories, fuels, ammo, module_slots = row.split(",", 5)
            category_set = frozenset(filter(None, categories.split("+")))
            accepted: set[str] = set()
            for category in filter(None, fuels.split("+")):
                accepted.update(fuel_items.get(category, ()))
            for category in filter(None, ammo.split("+")):
                accepted.update(ammo_items.get(category, ()))
            if int(module_slots) > 0:
                accepted.update(module_items)
            if entity_type == "furnace":
                for recipe, category in recipe_category.items():
                    if category in category_set:
                        accepted.update(recipe_ingredients.get(recipe, ()))
            if entity_type == "lab":
                accepted.update(lab_inputs)
            entity[name] = EntityCapability(
                holds_items=entity_type in holder_types or bool(accepted),
                recipe_categories=category_set if entity_type == "assembling-machine" else frozenset(),
                accepted_items=None if entity_type in unrestricted_types else frozenset(accepted),
                recipe_selected_inputs=entity_type in {"assembling-machine", "rocket-silo"},
            )
        metadata = cls(
            placeable=set(vocab.prototypes[1:]),
            recipe_category=recipe_category,
            hand_categories=set(filter(None, hand_rows.split("|"))),
            entity=entity,
            recipe_ingredients=recipe_ingredients,
        )
        metadata.refresh_research(rcon)
        return metadata


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
        [index > 0 and name in metadata.startable_technologies for index, name in enumerate(vocab.technologies)],
        dtype=bool,
    )
    entity = np.zeros((C.N_VERBS, C.ENTITY_SLOTS), dtype=bool)
    item_masks: dict[int, np.ndarray] = {}
    _, _, view, live, _ = client.observation()
    live = np.asarray(live, dtype=bool)
    # The character occupies a row like any other entity, but it cannot be
    # picked up, rotated, or inserted into: PICKUP on it returns "Unknown item
    # name: character". That is a type impossibility, so it is one of the few
    # things a structural mask may remove.
    for slot in np.nonzero(live)[0]:
        unit = str(int(client.entity_ids[slot]))
        row = client.entities.get(unit)
        if row is not None and client._parse_row(row).get("name") == "character":
            live[slot] = False
    for action_name in ("PICKUP", "ROTATE"):
        entity[C.VERB_INDEX[action_name]] = live
    recipe_masks: dict[int, np.ndarray] = {}
    for slot in np.nonzero(live)[0]:
        unit = str(int(client.entity_ids[slot]))
        row = client.entities.get(unit)
        if row is None:
            continue
        parsed = client._parse_row(row)
        name = parsed["name"]
        capability = metadata.entity.get(name, EntityCapability())
        if capability.holds_items:
            accepted = capability.accepted_items
            accepted_names = set(vocab.items[1:]) if accepted is None else set(accepted)
            if capability.recipe_selected_inputs and parsed.get("recipe"):
                accepted_names.update(metadata.recipe_ingredients.get(parsed["recipe"], ()))
            item_mask = np.asarray(
                [index > 0 and item_name in accepted_names for index, item_name in enumerate(vocab.items)],
                dtype=bool,
            )
            if item_mask.any():
                entity[C.VERB_INDEX["INSERT"], slot] = True
                entity[C.VERB_INDEX["EXTRACT"], slot] = True
                item_masks[int(slot)] = item_mask
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

    def items_for(slot: int) -> np.ndarray | None:
        return item_masks.get(int(slot))

    # ``view`` is intentionally unused beyond forcing the same current ordering
    # used by entity_ids; no feature magnitude affects legality.
    del view

    # A verb whose first required argument has no legal value cannot be
    # constructed at all, so offering it is a syntax error rather than a bad
    # idea: the policy would select it and then face an all-false head. This is
    # the brief's own example ("the entity pointer must point to a live entity")
    # taken to its limit, and it is the only state this function consults - it
    # asks whether an argument exists, never whether it is affordable, reachable
    # or wise. Without it, RESEARCH stayed selectable while 0 of 197
    # technologies were legal, and INSERT stayed selectable with no entity on
    # the map.
    _first_argument = {
        "RESEARCH": technology,
        "CRAFT": craft_recipe,
        "PLACE": prototype,
        "PICKUP": entity[C.VERB_INDEX["PICKUP"]],
        "ROTATE": entity[C.VERB_INDEX["ROTATE"]],
        "INSERT": entity[C.VERB_INDEX["INSERT"]],
        "EXTRACT": entity[C.VERB_INDEX["EXTRACT"]],
        "SET_RECIPE": entity[C.VERB_INDEX["SET_RECIPE"]],
    }
    for name, options in _first_argument.items():
        if not np.any(options):
            verb[C.VERB_INDEX[name]] = False

    return Masks(
        verb=verb,
        prototype=prototype,
        item=item,
        craft_recipe=craft_recipe,
        technology=technology,
        entity=entity,
        item_for_entity=items_for,
        recipe_for_entity=recipes_for,
    )
