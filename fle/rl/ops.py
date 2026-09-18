"""Operation definitions, random samplers, execution, and effect verification."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from fle.env.entities import Direction, Entity, Position
from fle.env.game_types import RecipeName, prototype_by_name
from fle.rl.world import EntityRow, WorldClient

LOGGER = logging.getLogger(__name__)

OPS = (
    "WAIT",
    "MOVE",
    "HARVEST",
    "CRAFT",
    "PLACE",
    "PICKUP",
    "ROTATE",
    "INSERT",
    "EXTRACT",
    "SET_RECIPE",
    "CONNECT",
    "RESEARCH",
)
QUANTITIES_NAIVE_HARVEST = (1, 5, 20, -1)
QUANTITIES = (1, 5, 20)
DIRECTIONS = tuple(direction.value for direction in Direction if direction.value in {0, 4, 8, 12})
CONNECTOR_NAMES = ("transport-belt", "pipe", "small-electric-pole")

SMELTABLE_INSERT_ITEMS = frozenset(
    {"iron-ore", "copper-ore", "stone", "iron-plate"}
)
SMELTABLE_STACK_LIMITS = {
    "iron-ore": 50,
    "copper-ore": 50,
    "stone": 50,
    "iron-plate": 100,
}
ROTATABLE_ENTITY_TYPES = frozenset(
    {
        "transport-belt",
        "underground-belt",
        "splitter",
        "inserter",
        "mining-drill",
        "assembling-machine",
        "boiler",
        "offshore-pump",
        "pipe-to-ground",
    }
)

PLACEABLE_ALLOWLIST = (
    "burner-mining-drill",
    "electric-mining-drill",
    "stone-furnace",
    "steel-furnace",
    "electric-furnace",
    "assembling-machine-1",
    "assembling-machine-2",
    "wooden-chest",
    "iron-chest",
    "steel-chest",
    "transport-belt",
    "fast-transport-belt",
    "underground-belt",
    "splitter",
    "burner-inserter",
    "inserter",
    "long-handed-inserter",
    "fast-inserter",
    "pipe",
    "pipe-to-ground",
    "small-electric-pole",
    "medium-electric-pole",
    "boiler",
    "steam-engine",
    "lab",
    "offshore-pump",
    "pumpjack",
    "chemical-plant",
    "oil-refinery",
)


def _verified_allowlist() -> tuple[str, ...]:
    verified = []
    for name in PLACEABLE_ALLOWLIST:
        if name not in prototype_by_name:
            LOGGER.warning("Dropping unresolved PLACE prototype %s", name)
        else:
            verified.append(name)
    return tuple(verified)


VERIFIED_PLACEABLE_ALLOWLIST = _verified_allowlist()
RECIPE_NAMES = {recipe.value: recipe for recipe in RecipeName}


@dataclass(frozen=True)
class ActionSpec:
    op: str
    args: dict[str, Any]


@dataclass(frozen=True)
class NoSupport:
    reason: str


@dataclass(frozen=True)
class VocabData:
    items: tuple[str, ...]
    recipes: tuple[str, ...]
    technologies: tuple[str, ...]
    place_results: Mapping[str, str]
    entity_info: Mapping[str, Mapping[str, Any]]
    main_products: Mapping[str, str]
    recipe_categories: Mapping[str, str] = field(default_factory=dict)
    recipe_ingredients: Mapping[str, tuple[tuple[str, float], ...]] = field(
        default_factory=dict
    )
    recipe_product_amounts: Mapping[str, float] = field(default_factory=dict)
    item_info: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VocabData:
        items = tuple(row["name"] for row in data["items"])
        recipes = tuple(row["name"] for row in data["recipes"])
        technologies = tuple(row["name"] for row in data["technologies"])
        place_results = {
            row["name"]: row["place_result"]
            for row in data["items"]
            if row.get("place_result")
        }
        entity_info = {row["name"]: row for row in data["entities"]}
        main_products = {
            row["name"]: row["products"][0]["name"]
            for row in data["recipes"]
            if row.get("products")
        }
        recipe_categories = {row["name"]: row.get("category") or "crafting" for row in data["recipes"]}
        recipe_ingredients = {
            row["name"]: tuple(
                (ingredient["name"], float(ingredient["amount"]))
                for ingredient in row.get("ingredients", ())
            )
            for row in data["recipes"]
        }
        recipe_product_amounts = {
            row["name"]: float(row["products"][0]["amount"])
            for row in data["recipes"]
            if row.get("products") and row["products"][0].get("amount")
        }
        return cls(
            items=items,
            recipes=recipes,
            technologies=technologies,
            place_results=place_results,
            entity_info=entity_info,
            main_products=main_products,
            recipe_categories=recipe_categories,
            recipe_ingredients=recipe_ingredients,
            recipe_product_amounts=recipe_product_amounts,
            item_info={row["name"]: row for row in data["items"]},
        )

    @classmethod
    def load(cls, path: str | Path) -> VocabData:
        import json

        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SamplerState:
    world: Any
    player_pos: tuple[float, float]
    inventory: Mapping[str, int]
    enabled_recipes: Sequence[str]
    research_state: Mapping[str, Mapping[str, Any]]
    vocab: VocabData
    crafting_categories: Sequence[str] = ("crafting",)


def hand_craftable_recipes(state: SamplerState, count: int = 1) -> list[str]:
    """Enabled recipes craftable for ``count`` outputs from the inventory."""
    return recursively_craftable_recipes(
        state.inventory,
        state.enabled_recipes,
        state.vocab,
        count=count,
        crafting_categories=state.crafting_categories,
    )


def recursively_craftable_recipes(
    inventory: Mapping[str, int],
    enabled_recipes: Sequence[str] | set[str] | frozenset[str],
    vocab: VocabData,
    *,
    count: int = 1,
    crafting_categories: Sequence[str] = ("crafting",),
) -> list[str]:
    """Return recipes the server can hand-craft for ``count`` output units.

    This is a pure simulation of ``craft_item/server.lua`` ingredient handling:
    recursively craft missing enabled hand-craftable sub-recipes in one shared
    inventory, then recheck and consume the parent's ingredients. Each candidate
    starts from a fresh inventory copy and recursive cycles are rejected.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    enabled = frozenset(enabled_recipes)
    categories = frozenset(crafting_categories)

    def attempt_craft(
        recipe_name: str,
        count: float,
        stock: dict[str, float],
        attempted: set[str],
    ) -> bool:
        if recipe_name in attempted:
            return False
        if recipe_name not in enabled:
            return False
        if vocab.recipe_categories.get(recipe_name, "crafting") not in categories:
            return False
        product_amount = vocab.recipe_product_amounts.get(recipe_name)
        if product_amount is None or product_amount <= 0:
            return False

        attempted.add(recipe_name)
        crafts_needed = math.ceil(count / product_amount)
        requirements = [
            (name, amount * crafts_needed)
            for name, amount in vocab.recipe_ingredients.get(recipe_name, ())
        ]
        for ingredient_name, needed in requirements:
            missing = needed - stock.get(ingredient_name, 0.0)
            if missing > 0 and not attempt_craft(
                ingredient_name, missing, stock, attempted
            ):
                return False

        if any(stock.get(name, 0.0) < needed for name, needed in requirements):
            return False
        for ingredient_name, needed in requirements:
            stock[ingredient_name] = stock.get(ingredient_name, 0.0) - needed
        stock[recipe_name] = stock.get(recipe_name, 0.0) + (
            crafts_needed * product_amount
        )
        return True

    craftable: list[str] = []
    for recipe_name in vocab.recipes:
        if attempt_craft(
            recipe_name,
            float(count),
            {name: float(count) for name, count in inventory.items()},
            set(),
        ):
            craftable.append(recipe_name)
    return craftable


@dataclass(frozen=True)
class OperationSnapshot:
    inventory: dict[str, int]
    entities: dict[int, EntityRow]
    position: tuple[float, float]
    tick: int
    score_player: float
    score_automated: float
    current_research: str | None
    research_queue: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionResult:
    status: str | None = None
    reason_class: str | None = None
    target_distance: float | None = None


class AnchorResolutionError(RuntimeError):
    pass


def is_fuel_item(name: str, vocab: VocabData) -> bool:
    """Return whether the pinned vocabulary gives an item positive fuel value."""
    return float(vocab.item_info.get(name, {}).get("fuel_value") or 0) > 0


def entity_is_rotatable(row: EntityRow, vocab: VocabData) -> bool:
    """State-only ROTATE support derived from prototype type and slot recipe."""
    entity_type = vocab.entity_info.get(row.name, {}).get("type")
    if row.name == "steam-engine":
        return True
    if entity_type == "assembling-machine":
        return bool(row.recipe)
    return entity_type in ROTATABLE_ENTITY_TYPES


def furnace_input_item(row: EntityRow, vocab: VocabData) -> str | None:
    """Return the visible furnace-source item type, ignoring output progress."""
    if vocab.entity_info.get(row.name, {}).get("type") != "furnace":
        return None
    source = row.inventories.get(1, {})
    return next(
        (
            name
            for name, count in sorted(source.items())
            if count > 0 and not is_fuel_item(name, vocab)
        ),
        None,
    )


def insertable_items(
    row: EntityRow,
    inventory: Mapping[str, int],
    vocab: VocabData,
) -> list[str]:
    """Held items admitted by the target's deterministic acceptance rules.

    This is deliberately a support predicate, not a preference. Capacity is
    used only for pinned, visible single-slot furnace inputs. Unknown target
    types admit nothing because the insert tool has no guaranteed accepting
    inventory for them.
    """
    held = [
        name
        for name, count in inventory.items()
        if count > 0 and name in vocab.item_info
    ]
    entity_type = vocab.entity_info.get(row.name, {}).get("type")
    if entity_type == "container":
        return held
    if entity_type == "furnace":
        input_item = furnace_input_item(row, vocab)
        input_count = row.inventories.get(1, {}).get(input_item, 0) if input_item else 0
        return [
            name
            for name in held
            if is_fuel_item(name, vocab)
            or (
                name in SMELTABLE_INSERT_ITEMS
                and (input_item is None or name == input_item)
                and input_count < SMELTABLE_STACK_LIMITS[name]
            )
        ]
    if entity_type == "mining-drill" or row.name == "burner-inserter":
        return [name for name in held if is_fuel_item(name, vocab)]
    if entity_type == "assembling-machine":
        if row.recipe:
            ingredients = {
                name for name, _ in vocab.recipe_ingredients.get(row.recipe, ())
            }
            return [name for name in held if name in ingredients]
        return [name for name in held if not is_fuel_item(name, vocab)]
    return []


def extractable_items(row: EntityRow, vocab: VocabData) -> list[str]:
    """Items the extraction tool can read from this visible entity slot."""
    entity_type = vocab.entity_info.get(row.name, {}).get("type")
    return [
        name
        for name, count in row.items.items()
        if count > 0
        and name in vocab.item_info
        and not (entity_type == "mining-drill" and is_fuel_item(name, vocab))
    ]


def _offset(rng: Random) -> tuple[int, int]:
    return rng.randint(-8, 8), rng.randint(-8, 8)


def _anchor(row: EntityRow) -> dict[str, Any]:
    return {
        "unit": row.unit,
        "name": row.name,
        "x": row.x,
        "y": row.y,
        "direction": row.direction,
    }


def _rows(state: SamplerState) -> list[EntityRow]:
    return [state.world.entities[key] for key in sorted(state.world.entities)]


def _random_row(state: SamplerState, rng: Random) -> EntityRow | NoSupport:
    rows = _rows(state)
    return rng.choice(rows) if rows else NoSupport("no_entity")


def _base_entity_action(op: str, state: SamplerState, rng: Random) -> ActionSpec | NoSupport:
    row = _random_row(state, rng)
    if isinstance(row, NoSupport):
        return row
    return ActionSpec(op, {"anchor": _anchor(row)})


def sample_action(op: str, regime: str, state: SamplerState, rng: Random) -> ActionSpec | NoSupport:
    if op not in OPS:
        raise ValueError(f"Unknown operation {op}")
    if regime not in {"naive", "macro", "bare"}:
        raise ValueError(f"Unknown regime {regime}")
    if regime == "naive":
        return _sample_naive(op, state, rng)
    return _sample_macro(op, state, rng)


def sample_naive(op: str, state: SamplerState, rng: Random) -> ActionSpec | NoSupport:
    return _sample_naive(op, state, rng)


def sample_macro(op: str, state: SamplerState, rng: Random) -> ActionSpec | NoSupport:
    return _sample_macro(op, state, rng)


def _sample_naive(op: str, state: SamplerState, rng: Random) -> ActionSpec | NoSupport:
    px, py = state.player_pos
    if op == "WAIT":
        return ActionSpec(op, {"seconds": rng.choice((1, 5, 15, 60))})
    if op in {"MOVE", "HARVEST"}:
        dx, dy = _offset(rng)
        args: dict[str, Any] = {"dx": dx, "dy": dy, "target": [px + dx, py + dy]}
        if op == "HARVEST":
            args["quantity"] = rng.choice(QUANTITIES_NAIVE_HARVEST)
        return ActionSpec(op, args)
    if op == "CRAFT":
        recipe = rng.choice(state.vocab.recipes)
        return ActionSpec(
            op,
            {
                "recipe": recipe,
                "quantity": rng.choice(QUANTITIES),
                "main_product": state.vocab.main_products.get(recipe),
            },
        )
    if op == "PLACE":
        dx, dy = _offset(rng)
        return ActionSpec(
            op,
            {
                "prototype": rng.choice(VERIFIED_PLACEABLE_ALLOWLIST),
                "dx": dx,
                "dy": dy,
                "target": [px + dx, py + dy],
                "direction": rng.choice(DIRECTIONS),
            },
        )
    if op in {"PICKUP", "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE"}:
        action = _base_entity_action(op, state, rng)
        if isinstance(action, NoSupport):
            return action
        args = dict(action.args)
        if op == "ROTATE":
            args["direction"] = rng.choice(DIRECTIONS)
        elif op in {"INSERT", "EXTRACT"}:
            args.update(item=rng.choice(state.vocab.items), quantity=rng.choice(QUANTITIES))
        elif op == "SET_RECIPE":
            args["recipe"] = rng.choice(state.vocab.recipes)
        return ActionSpec(op, args)
    if op == "CONNECT":
        rows = _rows(state)
        if len(rows) < 2:
            return NoSupport("fewer_than_two_entities")
        first, second = rng.sample(rows, 2)
        return ActionSpec(
            op,
            {
                "source": _anchor(first),
                "target": _anchor(second),
                "connector": rng.choice(CONNECTOR_NAMES),
            },
        )
    return ActionSpec(op, {"technology": rng.choice(state.vocab.technologies)})


def _sample_macro(op: str, state: SamplerState, rng: Random) -> ActionSpec | NoSupport:
    px, py = state.player_pos
    rows = _rows(state)
    if op == "WAIT":
        ticks = rng.choice((60, 300, 900, 3600) if rows else (60,))
        return ActionSpec(op, {"ticks": ticks, "seconds": ticks / 60})
    if op == "MOVE":
        cells = state.world.known_cells(px, py, radius=64)
        if not cells:
            return NoSupport("no_known_land_cell")
        target = rng.choice(cells)
        return ActionSpec(op, {"target": [target[0], target[1]]})
    if op == "HARVEST":
        patches = []
        for patch in state.world.patches():
            tile = patch.nearest_tile(px, py)
            if math.hypot(tile[0] - px, tile[1] - py) <= 160:
                patches.append((patch, tile))
        trees = [
            tree for tree in sorted(state.world.trees) if math.hypot(tree[0] - px, tree[1] - py) <= 160
        ]
        kinds = (["patch"] if patches else []) + (["tree"] if trees else [])
        if not kinds:
            return NoSupport("no_harvest_target")
        kind = rng.choice(kinds)
        if kind == "patch":
            patch, tile = rng.choice(patches)
            args = {"target_kind": kind, "patch_id": patch.id, "target": [tile[0] + 0.5, tile[1] + 0.5]}
        else:
            tree = rng.choice(trees)
            args = {"target_kind": kind, "target": [tree[0] + 0.5, tree[1] + 0.5]}
        args["quantity"] = rng.choice(QUANTITIES)
        return ActionSpec(op, args)
    if op == "CRAFT":
        craftable = hand_craftable_recipes(state)
        if not craftable:
            return NoSupport("no_hand_craftable_recipe")
        recipe = rng.choice(tuple(craftable))
        return ActionSpec(
            op,
            {
                "recipe": recipe,
                "quantity": rng.choice(QUANTITIES),
                "main_product": state.vocab.main_products.get(recipe),
            },
        )
    if op == "PLACE":
        candidates = []
        for item, count in state.inventory.items():
            place_result = state.vocab.place_results.get(item)
            if count > 0 and place_result in prototype_by_name:
                candidates.append((item, place_result))
        if not candidates:
            return NoSupport("no_placeable_in_inventory")
        item, prototype = rng.choice(sorted(candidates))
        dx, dy = _offset(rng)
        info = state.vocab.entity_info[prototype]
        width, height = int(info["tile_width"]), int(info["tile_height"])
        raw_x, raw_y = px + dx, py + dy
        target_x = math.floor(raw_x) + 0.5 if width % 2 else round(raw_x)
        target_y = math.floor(raw_y) + 0.5 if height % 2 else round(raw_y)
        return ActionSpec(
            op,
            {
                "item": item,
                "prototype": prototype,
                "dx": dx,
                "dy": dy,
                "target": [target_x, target_y],
                "direction": rng.choice(DIRECTIONS),
            },
        )
    if op in {"PICKUP", "ROTATE"}:
        if not rows:
            return NoSupport("no_entity")
        candidates = (
            [row for row in rows if entity_is_rotatable(row, state.vocab)]
            if op == "ROTATE"
            else rows
        )
        if not candidates:
            return NoSupport("no_rotatable_entity")
        row = rng.choice(candidates)
        args = {"anchor": _anchor(row)}
        if op == "ROTATE":
            choices = [direction for direction in DIRECTIONS if direction != row.direction]
            args["direction"] = rng.choice(choices)
        return ActionSpec(op, args)
    if op == "INSERT":
        if not rows:
            return NoSupport("no_entity")
        candidates = [
            (row, insertable_items(row, state.inventory, state.vocab))
            for row in rows
        ]
        candidates = [(row, items) for row, items in candidates if items]
        if not candidates:
            return NoSupport("insert_no_accepted_item")
        row, items = rng.choice(candidates)
        return ActionSpec(
            op,
            {
                "anchor": _anchor(row),
                "item": rng.choice(items),
                "quantity": rng.choice(QUANTITIES),
            },
        )
    if op == "EXTRACT":
        if not rows:
            return NoSupport("no_entity")
        candidates = [row for row in rows if row.items]
        if not candidates:
            return NoSupport("entity_has_no_items")
        row = rng.choice(candidates)
        return ActionSpec(
            op,
            {
                "anchor": _anchor(row),
                "item": rng.choice(sorted(row.items)),
                "quantity": rng.choice(QUANTITIES),
            },
        )
    if op == "SET_RECIPE":
        candidates = [
            row
            for row in rows
            if state.vocab.entity_info.get(row.name, {}).get("type") == "assembling-machine"
        ]
        if not candidates:
            return NoSupport("no_assembling_machine")
        if not state.enabled_recipes:
            return NoSupport("no_enabled_recipe")
        return ActionSpec(
            op,
            {
                "anchor": _anchor(rng.choice(candidates)),
                "recipe": rng.choice(tuple(state.enabled_recipes)),
            },
        )
    if op == "CONNECT":
        if len(rows) < 2:
            return NoSupport("fewer_than_two_entities")
        connectors = sorted(
            name for name in CONNECTOR_NAMES if state.inventory.get(name, 0) > 0
        )
        if not connectors:
            return NoSupport("no_connector_in_inventory")
        first, second = rng.sample(rows, 2)
        return ActionSpec(
            op,
            {
                "source": _anchor(first),
                "target": _anchor(second),
                "connector": rng.choice(connectors),
            },
        )
    candidates = []
    for name, tech in state.research_state.items():
        prerequisites = tech.get("prerequisites", [])
        ready = all(state.research_state.get(pre, {}).get("researched", False) for pre in prerequisites)
        if tech.get("enabled") and not tech.get("researched") and ready:
            candidates.append(name)
    if not candidates:
        return NoSupport("no_research_available")
    return ActionSpec(op, {"technology": rng.choice(sorted(candidates))})


def _resolve_entity(namespace: Any, anchor: Mapping[str, Any]) -> Entity:
    position = Position(x=anchor["x"], y=anchor["y"])
    prototype = prototype_by_name.get(anchor["name"])
    if prototype is not None:
        entity = namespace.get_entity(prototype, position)
        if entity is not None and entity.name == anchor["name"]:
            return entity
    else:
        for entity in namespace.get_entities(position=position, radius=0.5):
            if isinstance(entity, Entity) and entity.name == anchor["name"]:
                return entity
    raise AnchorResolutionError(
        f"anchor_resolution_failed: u{anchor['unit']} {anchor['name']} at ({anchor['x']}, {anchor['y']})"
    )


def _tool_prototype(name: str) -> Any:
    return prototype_by_name.get(name, name)


def _occupied_tiles(world: WorldClient) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for row in world.entities.values():
        width = row.tile_width or 1
        height = row.tile_height or 1
        left = math.floor(row.x - width / 2 + 0.5)
        top = math.floor(row.y - height / 2 + 0.5)
        for x in range(left, left + width):
            for y in range(top, top + height):
                occupied.add((x, y))
    return occupied


def _adjacent_tile_position(
    tile_x: int,
    tile_y: int,
    world: WorldClient,
    px: float,
    py: float,
) -> Position:
    """Return the nearest non-colliding working tile around a terrain tile."""
    occupied = _occupied_tiles(world)
    candidates: list[tuple[float, int, int]] = []
    for x in range(tile_x - 1, tile_x + 2):
        for y in range(tile_y - 1, tile_y + 2):
            if (x, y) == (tile_x, tile_y):
                continue
            tile = (x, y)
            if (
                tile in occupied
                or tile in world.trees
                or tile in world.obstacles
                or world.is_water(x, y)
            ):
                continue
            candidates.append((math.hypot(x + 0.5 - px, y + 0.5 - py), x, y))
    if not candidates:
        raise AnchorResolutionError(
            f"approach_failed: no free tile adjacent to ({tile_x}, {tile_y})"
        )
    _, x, y = min(candidates)
    return Position(x=x + 0.5, y=y + 0.5)


def _adjacent_position(row: EntityRow, world: WorldClient, px: float, py: float) -> Position:
    width = row.tile_width or 1
    height = row.tile_height or 1
    left = math.floor(row.x - width / 2 + 0.5)
    top = math.floor(row.y - height / 2 + 0.5)
    right = left + width - 1
    bottom = top + height - 1
    occupied = _occupied_tiles(world)
    candidates: list[tuple[float, int, int]] = []
    for x in range(left - 1, right + 2):
        for y in range(top - 1, bottom + 2):
            if left <= x <= right and top <= y <= bottom:
                continue
            tile = (x, y)
            if (
                tile in occupied
                or tile in world.trees
                or tile in world.obstacles
                or world.is_water(x, y)
            ):
                continue
            candidates.append((math.hypot(x + 0.5 - px, y + 0.5 - py), x, y))
    if not candidates:
        raise AnchorResolutionError(f"approach_failed: no free tile adjacent to u{row.unit}")
    _, x, y = min(candidates)
    return Position(x=x + 0.5, y=y + 0.5)


def _approach_entity(
    row: EntityRow,
    namespace: Any,
    world: WorldClient,
    build_reach: float,
) -> ExecutionResult | None:
    px, py = world.read_player_pos()
    distance = math.hypot(px - row.x, py - row.y)
    if distance <= build_reach - 0.1:
        return None
    namespace.move_to(_adjacent_position(row, world, px, py))
    # Movement advances the simulation and can consume/remove the anchor's
    # contents. Refresh both entity and terrain caches before validation and
    # before any entity tool receives the stale pre-approach row.
    world.all_drain()
    px, py = world.read_player_pos()
    distance = math.hypot(px - row.x, py - row.y)
    if distance > build_reach - 0.1:
        return ExecutionResult(
            status="approach_failed",
            reason_class="approach_failed",
            target_distance=distance,
        )
    return None


def _live_entity_row(
    world: WorldClient, anchor: Mapping[str, Any]
) -> EntityRow | None:
    row = world.entities.get(anchor["unit"])
    if row is not None:
        return row
    return next(
        (
            candidate
            for candidate in world.entities.values()
            if candidate.name == anchor["name"]
            and math.hypot(
                candidate.x - anchor["x"],
                candidate.y - anchor["y"],
            )
            <= 0.75
        ),
        None,
    )


def execute_action(
    action: ActionSpec,
    regime: str,
    namespace: Any,
    world: WorldClient,
    resource_reach: float,
    build_reach: float | None = None,
) -> ExecutionResult:
    if regime not in {"naive", "macro", "bare"}:
        raise ValueError(f"Unknown regime {regime}")
    op, args = action.op, action.args
    if op == "WAIT":
        namespace.sleep(args["seconds"])
    elif op == "MOVE":
        namespace.move_to(Position(x=args["target"][0], y=args["target"][1]))
    elif op == "HARVEST":
        target = Position(x=args["target"][0], y=args["target"][1])
        if regime == "macro":
            namespace.move_to(target)
            px, py = world.read_player_pos()
            distance = math.hypot(px - target.x, py - target.y)
            if distance > resource_reach - 0.1:
                return ExecutionResult(
                    status="approach_failed",
                    reason_class="approach_failed",
                    target_distance=distance,
                )
        namespace.harvest_resource(target, args["quantity"])
    elif op == "CRAFT":
        namespace.craft_item(args["recipe"], args["quantity"])
    elif op == "PLACE":
        namespace.place_entity(
            _tool_prototype(args["prototype"]),
            Direction(args["direction"]),
            Position(x=args["target"][0], y=args["target"][1]),
            exact=True,
        )
    elif op in {"PICKUP", "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE"}:
        row = _live_entity_row(world, args["anchor"])
        if row is None:
            raise AnchorResolutionError("anchor_resolution_failed: entity row disappeared")
        if regime != "bare" and build_reach is not None:
            approach_result = _approach_entity(row, namespace, world, build_reach)
            if approach_result is not None:
                return approach_result
            row = _live_entity_row(world, args["anchor"])
            if row is None:
                raise AnchorResolutionError(
                    "anchor_resolution_failed: entity row disappeared after approach"
                )
        if op == "EXTRACT" and row.items.get(args["item"], 0) <= 0:
            return ExecutionResult(
                status="no_support",
                reason_class="extract_item_disappeared_after_approach",
            )
        entity = _resolve_entity(namespace, args["anchor"])
        if op == "PICKUP":
            namespace.pickup_entity(entity)
        elif op == "ROTATE":
            namespace.rotate_entity(entity, Direction(args["direction"]))
        elif op == "INSERT":
            namespace.insert_item(_tool_prototype(args["item"]), entity, args["quantity"])
        elif op == "EXTRACT":
            namespace.extract_item(_tool_prototype(args["item"]), entity, args["quantity"])
        else:
            recipe = prototype_by_name.get(args["recipe"], RECIPE_NAMES.get(args["recipe"], args["recipe"]))
            namespace.set_entity_recipe(entity, recipe)
    elif op == "CONNECT":
        source = _resolve_entity(namespace, args["source"])
        target = _resolve_entity(namespace, args["target"])
        namespace.connect_entities(
            source,
            target,
            connection_type=_tool_prototype(args["connector"]),
        )
    elif op == "RESEARCH":
        namespace.set_research(args["technology"])
    else:
        raise ValueError(f"Unknown operation {op}")
    target_distance = None
    if op in {"MOVE", "HARVEST"} and "target" in args:
        px, py = world.read_player_pos()
        target_distance = math.hypot(px - args["target"][0], py - args["target"][1])
    return ExecutionResult(target_distance=target_distance)


def inventory_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        name: after.get(name, 0) - before.get(name, 0)
        for name in sorted(set(before) | set(after))
        if after.get(name, 0) != before.get(name, 0)
    }


def _target_row(action: ActionSpec, snapshot: OperationSnapshot) -> EntityRow | None:
    anchor = action.args.get("anchor")
    if not anchor:
        return None
    row = snapshot.entities.get(anchor["unit"])
    if row is not None:
        return row
    return next(
        (
            candidate
            for candidate in snapshot.entities.values()
            if candidate.name == anchor["name"]
            and math.hypot(candidate.x - anchor["x"], candidate.y - anchor["y"]) <= 0.75
        ),
        None,
    )


def verify_effect(
    action: ActionSpec, before: OperationSnapshot, after: OperationSnapshot
) -> tuple[bool, dict[str, Any]]:
    op, args = action.op, action.args
    delta = inventory_delta(before.inventory, after.inventory)
    if op == "WAIT":
        actual = after.tick - before.tick
        return actual >= 1, {"requested_ticks": args.get("ticks", args.get("seconds", 0) * 60), "actual_ticks": actual}
    if op == "MOVE":
        distance = math.hypot(after.position[0] - args["target"][0], after.position[1] - args["target"][1])
        return distance <= 2.5, {"final_distance": distance}
    if op == "HARVEST":
        positive = {name: count for name, count in delta.items() if count > 0}
        total_delta = sum(after.inventory.values()) - sum(before.inventory.values())
        return total_delta > 0, {
            "items_acquired": sum(positive.values()),
            "item_names": sorted(positive),
            "total_inventory_delta": total_delta,
        }
    if op == "CRAFT":
        product = args.get("main_product")
        product_delta = delta.get(product, 0) if product else 0
        return product_delta > 0, {"main_product": product, "product_delta": product_delta}
    if op == "PLACE":
        name = args["prototype"]
        count_before = sum(row.name == name for row in before.entities.values())
        count_after = sum(row.name == name for row in after.entities.values())
        difference = count_after - count_before
        return difference == 1, {"row_count_delta": difference}
    if op == "PICKUP":
        anchor = args["anchor"]
        unit_gone = anchor["unit"] not in after.entities
        item_delta = delta.get(anchor["name"], 0)
        return item_delta > 0, {"unit_gone": unit_gone, "item_delta": item_delta}
    if op == "ROTATE":
        before_row, after_row = _target_row(action, before), _target_row(action, after)
        before_direction = before_row.direction if before_row else None
        after_direction = after_row.direction if after_row else None
        ok = after_direction == args["direction"] and after_direction != before_direction
        return ok, {"direction_before": before_direction, "direction_after": after_direction}
    if op in {"INSERT", "EXTRACT"}:
        item_delta = delta.get(args["item"], 0)
        ok = item_delta < 0 if op == "INSERT" else item_delta > 0
        return ok, {"item_delta": item_delta}
    if op == "SET_RECIPE":
        row = _target_row(action, after)
        recipe = row.recipe if row else None
        return recipe == args["recipe"], {"recipe_after": recipe}
    if op == "CONNECT":
        difference = len(after.entities) - len(before.entities)
        connector = args.get("connector")
        connector_delta = delta.get(connector, 0) if connector else 0
        return difference > 0 or connector_delta < 0, {
            "new_entities": difference,
            "row_count_delta": difference,
            "connector": connector,
            "connector_item_delta": connector_delta,
        }
    if op == "RESEARCH":
        selected = args["technology"]
        return selected == after.current_research or selected in after.research_queue, {
            "research_after": after.current_research,
            "research_queue_after": after.research_queue,
        }
    raise ValueError(f"Unknown operation {op}")


ERROR_PATTERNS = (
    ("nothing_to_harvest", re.compile(r"nothing within reach|nothing to harvest|no matching resources|no harvestable", re.IGNORECASE)),
    ("out_of_reach", re.compile(r"out of reach|too far away|move closer|target position is too far", re.IGNORECASE)),
    ("invalid_quantity", re.compile(r"invalid (?:count|quantity)|count must|quantity must|greater than 0|non.?positive", re.IGNORECASE)),
    ("no_path", re.compile(r"no path|failed to find a path|could not get path|invalid path|path.*not found", re.IGNORECASE)),
    ("cannot_place", re.compile(r"cannot place|could not place|suitable position|create_entity returned nil|already exists at the target", re.IGNORECASE)),
    ("not_in_inventory", re.compile(r"not in (?:the )?inventory|no .+ (?:in inventory|to insert)|do not have (?:the required|enough)", re.IGNORECASE)),
    ("missing_ingredients", re.compile(r"missing ingredients|missing .+ ingredient|sub-ingredient|still missing", re.IGNORECASE)),
    ("recipe_disabled", re.compile(r"recipe.*(?:disabled|not unlocked)|not unlocked.*recipe", re.IGNORECASE)),
    ("not_hand_craftable", re.compile(r"cannot be crafted|requires a crafting machine|smelting in a furnace", re.IGNORECASE)),
    ("tech_researched", re.compile(r"technology .*already researched", re.IGNORECASE)),
    ("tech_prerequisite", re.compile(r"missing prerequisites|technology .*not enabled", re.IGNORECASE)),
    ("tech_unknown", re.compile(r"technology .*does not exist|technology is invalid", re.IGNORECASE)),
    ("unknown_name", re.compile(r"doesn.t exist|isn.t something that exists|invalid (?:item |entity )?prototype|typo", re.IGNORECASE)),
    ("entity_not_found", re.compile(r"no entity|couldn.t find .* at position|no building found|could not find any entities", re.IGNORECASE)),
    ("no_suitable_slot", re.compile(r"inventory is full|no suitable slot|no available space|does not support modules", re.IGNORECASE)),
    ("nothing_to_extract", re.compile(r"no .+ found in any nearby|containing .+|failed to extract|could not extract", re.IGNORECASE)),
    ("connect_failed", re.compile(r"failed to connect|cannot connect|connect.*failed|placement blockage", re.IGNORECASE)),
    ("assertion", re.compile(r"assertion(?:error)?|assert ", re.IGNORECASE)),
)


def classify_error(op: str, message: str) -> str:
    del op
    for name, pattern in ERROR_PATTERNS:
        if pattern.search(message):
            return name
    return "other"
