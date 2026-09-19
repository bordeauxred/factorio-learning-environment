"""Pure observation and support-mask construction for the macro environment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fle.rl import schema as S
from fle.rl.ops import (
    OperationSnapshot,
    SamplerState,
    VocabData,
    entity_is_rotatable,
    extractable_items,
    hand_craftable_recipes,
    insertable_items,
)
from fle.rl.world import EntityRow


@dataclass(frozen=True)
class HarvestTarget:
    """One target slot, including the exact tile centre used by HARVEST."""

    kind: str
    position: tuple[float, float]
    n_tiles: int
    patch_id: str | None = None


@dataclass(frozen=True)
class ObservationInput:
    snapshot: OperationSnapshot
    enabled_recipes: Sequence[str]
    research_state: Mapping[str, Mapping[str, object]]
    episode_start_tick: int
    step_count: int
    last_status: str | None
    last_op: str | None


@dataclass(frozen=True)
class ObservationFrame:
    obs: np.ndarray
    targets: tuple[HarvestTarget | None, ...]
    entities: tuple[EntityRow | None, ...]


def split_observation(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Split one flat observation into the shapes declared by the schema."""
    blocks: dict[str, np.ndarray] = {}
    for name, (start, stop) in S.OBS_LAYOUT.items():
        block = obs[start:stop]
        if name == "targets":
            block = block.reshape(S.N_TARGET_SLOTS, S.TARGET_FEATURES)
        elif name == "entities":
            block = block.reshape(S.N_ENTITY_SLOTS, S.ENTITY_FEATURES)
        elif name == "grid":
            block = block.reshape(len(S.GRID_CHANNELS), S.GRID_SIDE, S.GRID_SIDE)
        blocks[name] = block
    return blocks


def write_globals_vector(
    out: np.ndarray,
    state: ObservationInput,
    *,
    resource_reach: float,
    max_steps: int,
    max_ticks: int,
) -> None:
    """Write the existing 16-value macro global encoding into ``out``."""
    snap = state.snapshot
    elapsed_ticks = max(0, snap.tick - state.episode_start_tick)
    remaining_steps = max(0, max_steps - state.step_count)
    status_index = {
        "ok": 0,
        "no_effect": 1,
    }.get(state.last_status, 2 if state.last_status is not None else None)
    out[0] = elapsed_ticks / max_ticks
    out[1] = remaining_steps / max_steps
    out[2] = snap.position[0] / 256.0
    out[3] = snap.position[1] / 256.0
    out[4] = math.log1p(sum(snap.inventory.values())) / math.log1p(1000)
    out[5] = math.log1p(len(snap.entities)) / 5.0
    out[6] = math.log1p(max(snap.score_player, 0.0)) / 10.0
    out[7] = math.log1p(max(snap.score_automated, 0.0)) / 10.0
    if status_index is not None:
        out[8 + status_index] = 1.0
    if state.last_op in S.OP_INDEX:
        out[11] = S.OP_INDEX[state.last_op] / len(S.OPS)
    out[12] = float(snap.current_research is not None)
    out[13] = float(bool(snap.entities))
    out[14] = resource_reach / 10.0


def write_inventory_vector(out: np.ndarray, inventory: Mapping[str, int]) -> None:
    """Write the existing log-scaled inventory encoding into ``out``."""
    scale = math.log1p(1000)
    for name, count in inventory.items():
        index = S.ITEM_INDEX.get(name)
        if index is not None and count > 0:
            out[index] = math.log1p(count) / scale


def write_research_vector(
    out: np.ndarray, research_state: Mapping[str, Mapping[str, object]]
) -> None:
    """Write the existing binary completed-technology encoding into ``out``."""
    for name, tech in research_state.items():
        index = S.TECH_INDEX.get(name)
        if index is not None:
            out[index] = float(bool(tech.get("researched")))


def write_recipes_vector(out: np.ndarray, enabled_recipes: Sequence[str]) -> None:
    """Write the existing binary enabled-recipe encoding into ``out``."""
    for name in enabled_recipes:
        index = S.RECIPE_INDEX.get(name)
        if index is not None:
            out[index] = 1.0


def select_entity_slots(
    entities: Mapping[int, EntityRow], player_pos: tuple[float, float]
) -> tuple[EntityRow | None, ...]:
    """Return the existing nearest-first, fixed-size entity slot table."""
    px, py = player_pos
    rows = sorted(
        entities.values(),
        key=lambda row: (math.hypot(row.x - px, row.y - py), row.unit),
    )[: S.N_ENTITY_SLOTS]
    return tuple(rows) + (None,) * (S.N_ENTITY_SLOTS - len(rows))


def entity_class(
    row: EntityRow, entity_info: Mapping[str, Mapping[str, object]]
) -> str:
    """Map an entity row to the existing coarse entity class."""
    entity_type = entity_info.get(row.name, {}).get("type")
    return entity_type if entity_type in S.ENTITY_CLASSES else "other"


def status_group(status: int, status_names: Mapping[int, str]) -> str:
    """Map a Factorio status code to the existing status group."""
    name = status_names.get(status, "other")
    if name == "working":
        return "working"
    if name in {"no_fuel", "no_power", "low_power", "no_minable_resources"}:
        return "no_fuel_or_power"
    if name in {
        "no_ingredients",
        "item_ingredient_shortage",
        "waiting_for_source_items",
    }:
        return "no_input"
    if name in {"full_output", "waiting_for_space_in_destination"}:
        return "output_full"
    return "other"


def write_entity_table(
    out: np.ndarray,
    entities: Sequence[EntityRow | None],
    player_pos: tuple[float, float],
    *,
    entity_info: Mapping[str, Mapping[str, object]],
    status_names: Mapping[int, str],
) -> None:
    """Write the existing 24-feature entity table into ``out``."""
    px, py = player_pos
    for index, row in enumerate(entities):
        if row is None:
            continue
        out[index, S.ENTITY_CLASSES.index(entity_class(row, entity_info))] = 1.0
        dx, dy = row.x - px, row.y - py
        offset = len(S.ENTITY_CLASSES)
        out[index, offset] = dx / 32.0
        out[index, offset + 1] = dy / 32.0
        out[index, offset + 2] = min(math.hypot(dx, dy) / 32.0, 1.0)
        out[index, offset + 3] = row.direction / 16.0
        status_start = offset + 4
        group = status_group(row.status, status_names)
        out[index, status_start + S.STATUS_GROUPS.index(group)] = 1.0
        detail_start = status_start + len(S.STATUS_GROUPS)
        if row.recipe:
            out[index, detail_start] = 1.0
            recipe_index = S.RECIPE_INDEX.get(row.recipe)
            if recipe_index is not None:
                out[index, detail_start + 1] = recipe_index / len(S.RECIPE_NAMES)
        items = row.items
        out[index, detail_start + 2] = math.log1p(sum(items.values())) / math.log1p(
            1000
        )
        out[index, detail_start + 3] = math.log1p(
            items.get("coal", 0) + items.get("wood", 0)
        ) / math.log1p(100)
        out[index, detail_start + 4] = 1.0


def entity_dimensions(
    row: EntityRow, entity_info: Mapping[str, Mapping[str, object]]
) -> tuple[int, int]:
    """Resolve the same real footprint used by the macro observation."""
    info = entity_info.get(row.name, {})
    width = row.tile_width or int(info.get("tile_width") or 1)
    height = row.tile_height or int(info.get("tile_height") or 1)
    if row.direction in {4, 12}:
        width, height = height, width
    return width, height


def available_technologies(
    research_state: Mapping[str, Mapping[str, object]],
    *,
    trigger_technologies: set[str] | frozenset[str],
    current_research: str | None,
    research_queue: Sequence[str],
) -> list[str]:
    """Return technologies accepted by the existing prerequisite predicate."""
    available: list[str] = []
    for name, tech in research_state.items():
        prerequisites = tech.get("prerequisites", [])
        ready = all(
            bool(research_state.get(str(pre), {}).get("researched"))
            for pre in prerequisites
        )
        if (
            tech.get("enabled")
            and not tech.get("researched")
            and ready
            and name not in trigger_technologies
            and name != current_research
            and name not in research_queue
            and name in S.TECH_INDEX
        ):
            available.append(name)
    return available


class MacroObservationBuilder:
    """Build observations from a cached world and an immutable live snapshot."""

    def __init__(
        self,
        *,
        world,
        vocab: VocabData,
        crafting_categories: Sequence[str],
        resource_reach: float,
        build_reach: float,
        trigger_technologies: set[str] | frozenset[str],
        status_names: Mapping[int, str],
        max_steps: int,
        max_ticks: int,
        quantity_aware_support: bool = True,
        regime: str = "macro",
    ) -> None:
        if regime not in {"macro", "bare"}:
            raise ValueError(f"Unknown action regime {regime}")
        self.world = world
        self.vocab = vocab
        self.crafting_categories = tuple(crafting_categories)
        self.resource_reach = resource_reach
        self.build_reach = build_reach
        self.trigger_technologies = frozenset(trigger_technologies)
        self.status_names = dict(status_names)
        self.max_steps = max_steps
        self.max_ticks = max_ticks
        self.quantity_aware_support = quantity_aware_support
        self.regime = regime

    def build(self, state: ObservationInput) -> ObservationFrame:
        obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
        targets = self._target_slots(state.snapshot.position)
        entities = self._entity_slots(state.snapshot)
        occupied = self._occupied_tiles(state.snapshot.entities.values())

        self._write_globals(obs, state)
        self._write_inventory(obs, state.snapshot.inventory)
        self._write_tech(obs, state.research_state)
        self._write_recipes(obs, state.enabled_recipes)
        self._write_targets(obs, targets, state.snapshot.position)
        self._write_entities(obs, entities, state.snapshot.position)
        self._write_grid(obs, state.snapshot.position, occupied)
        self._write_masks(obs, state, targets, entities, occupied)
        np.clip(obs, -1.0, 1.0, out=obs)
        return ObservationFrame(obs, targets, entities)

    def _write_globals(self, obs: np.ndarray, state: ObservationInput) -> None:
        start, _ = S.OBS_LAYOUT["globals"]
        values = np.zeros(S.N_GLOBALS, dtype=np.float32)
        write_globals_vector(
            values,
            state,
            resource_reach=self.resource_reach,
            max_steps=self.max_steps,
            max_ticks=self.max_ticks,
        )
        obs[start : start + S.N_GLOBALS] = values

    @staticmethod
    def _write_inventory(obs: np.ndarray, inventory: Mapping[str, int]) -> None:
        start, _ = S.OBS_LAYOUT["inventory"]
        write_inventory_vector(obs[start : start + len(S.ITEM_NAMES)], inventory)

    @staticmethod
    def _write_tech(
        obs: np.ndarray, research_state: Mapping[str, Mapping[str, object]]
    ) -> None:
        start, _ = S.OBS_LAYOUT["tech"]
        write_research_vector(obs[start : start + len(S.TECH_NAMES)], research_state)

    @staticmethod
    def _write_recipes(obs: np.ndarray, enabled_recipes: Sequence[str]) -> None:
        start, _ = S.OBS_LAYOUT["recipes_enabled"]
        write_recipes_vector(obs[start : start + len(S.RECIPE_NAMES)], enabled_recipes)

    def _target_slots(
        self, player_pos: tuple[float, float]
    ) -> tuple[HarvestTarget | None, ...]:
        px, py = player_pos
        patch_targets: list[tuple[float, str, HarvestTarget]] = []
        for patch in self.world.patches():
            if not self._resource_is_hand_harvestable(patch.name):
                continue
            tile = patch.nearest_tile(px, py)
            position = (tile[0] + 0.5, tile[1] + 0.5)
            tile_distance = math.hypot(tile[0] - px, tile[1] - py)
            if tile_distance <= S.TARGET_RADIUS:
                target = HarvestTarget(patch.name, position, patch.n_tiles, patch.id)
                patch_targets.append((tile_distance, patch.id, target))
        patch_targets.sort(key=lambda row: (row[0], row[1]))

        tree_targets: list[HarvestTarget | None] = [None] * S.N_TREE_SLOTS
        tree_distances = [math.inf] * S.N_TREE_SLOTS
        for x, y in sorted(self.world.trees):
            # Terrain drains can remove a tree while this deterministic slot
            # pass is iterating over its sorted snapshot.
            if (x, y) not in self.world.trees:
                continue
            position = (x + 0.5, y + 0.5)
            dx, dy = position[0] - px, position[1] - py
            tile_distance = math.hypot(x - px, y - py)
            if tile_distance > S.TARGET_RADIUS:
                continue
            octant = round(math.atan2(dy, dx) / (math.pi / 4)) % 8
            if tile_distance < tree_distances[octant]:
                tree_distances[octant] = tile_distance
                tree_targets[octant] = HarvestTarget("tree", position, 0)

        slots: list[HarvestTarget | None] = [
            row[2] for row in patch_targets[: S.N_PATCH_SLOTS]
        ]
        slots.extend([None] * (S.N_PATCH_SLOTS - len(slots)))
        slots.extend(tree_targets)
        return tuple(slots)

    def _resource_is_hand_harvestable(self, name: str) -> bool:
        """Return whether the pinned vocab identifies a hand-minable product.

        The current pinned export does not carry ``minable`` prototype fields,
        so a matching item is the conservative evidence that a resource has a
        character-acquirable product. Fluid oil and acid-gated uranium are
        deterministic exclusions.
        """
        if name in {"crude-oil", "uranium-ore"}:
            return False
        info = self.vocab.entity_info.get(name, {})
        return info.get("type") == "resource" and name in self.vocab.item_info

    @staticmethod
    def _entity_slots(snapshot: OperationSnapshot) -> tuple[EntityRow | None, ...]:
        return select_entity_slots(snapshot.entities, snapshot.position)

    def _write_targets(
        self,
        obs: np.ndarray,
        targets: tuple[HarvestTarget | None, ...],
        player_pos: tuple[float, float],
    ) -> None:
        start, _ = S.OBS_LAYOUT["targets"]
        rows = obs[start : start + S.N_TARGET_SLOTS * S.TARGET_FEATURES].reshape(
            S.N_TARGET_SLOTS, S.TARGET_FEATURES
        )
        px, py = player_pos
        for index, target in enumerate(targets):
            if target is None:
                continue
            kind = target.kind if target.kind in S.TARGET_KINDS else "other-resource"
            rows[index, S.TARGET_KINDS.index(kind)] = 1.0
            dx, dy = target.position[0] - px, target.position[1] - py
            offset = len(S.TARGET_KINDS)
            rows[index, offset : offset + 3] = (
                dx / S.TARGET_RADIUS,
                dy / S.TARGET_RADIUS,
                math.hypot(dx, dy) / S.TARGET_RADIUS,
            )
            rows[index, offset + 3] = math.log1p(target.n_tiles) / 10.0
            rows[index, offset + 4] = 1.0

    def _entity_class(self, row: EntityRow) -> str:
        return entity_class(row, self.vocab.entity_info)

    def _status_group(self, status: int) -> str:
        return status_group(status, self.status_names)

    def _write_entities(
        self,
        obs: np.ndarray,
        entities: tuple[EntityRow | None, ...],
        player_pos: tuple[float, float],
    ) -> None:
        start, _ = S.OBS_LAYOUT["entities"]
        rows = obs[start : start + S.N_ENTITY_SLOTS * S.ENTITY_FEATURES].reshape(
            S.N_ENTITY_SLOTS, S.ENTITY_FEATURES
        )
        write_entity_table(
            rows,
            entities,
            player_pos,
            entity_info=self.vocab.entity_info,
            status_names=self.status_names,
        )

    def _entity_dimensions(self, row: EntityRow) -> tuple[int, int]:
        return entity_dimensions(row, self.vocab.entity_info)

    def _occupied_tiles(self, entities: Sequence[EntityRow]) -> set[tuple[int, int]]:
        occupied: set[tuple[int, int]] = set()
        for row in entities:
            width, height = self._entity_dimensions(row)
            left = math.floor(row.x - width / 2 + 0.5)
            top = math.floor(row.y - height / 2 + 0.5)
            for x in range(left, left + width):
                for y in range(top, top + height):
                    occupied.add((x, y))
        return occupied

    def _write_grid(
        self,
        obs: np.ndarray,
        player_pos: tuple[float, float],
        occupied: set[tuple[int, int]],
    ) -> None:
        start, _ = S.OBS_LAYOUT["grid"]
        grid = obs[start : start + len(S.GRID_CHANNELS) * S.GRID_SIDE**2].reshape(
            len(S.GRID_CHANNELS), S.GRID_SIDE, S.GRID_SIDE
        )
        origin_x, origin_y = math.floor(player_pos[0]), math.floor(player_pos[1])
        for row_index, dy in enumerate(range(-S.OFFSET_RADIUS, S.OFFSET_RADIUS + 1)):
            for column_index, dx in enumerate(
                range(-S.OFFSET_RADIUS, S.OFFSET_RADIUS + 1)
            ):
                tile = (origin_x + dx, origin_y + dy)
                grid[0, row_index, column_index] = float(tile in occupied)
                grid[1, row_index, column_index] = float(tile in self.world.ores)
                grid[2, row_index, column_index] = float(
                    tile in self.world.trees or tile in self.world.obstacles
                )
                grid[3, row_index, column_index] = float(self.world.is_water(*tile))

    def _write_masks(
        self,
        obs: np.ndarray,
        state: ObservationInput,
        targets: tuple[HarvestTarget | None, ...],
        entities: tuple[EntityRow | None, ...],
        occupied: set[tuple[int, int]],
    ) -> None:
        masks = {
            head: np.zeros(S.HEAD_SIZES[head], dtype=np.float32) for head in S.HEADS
        }
        snap = state.snapshot
        reach_margin = 0.1
        target_in_reach = [
            target is not None
            and (
                self.regime == "macro"
                or math.hypot(
                    target.position[0] - snap.position[0],
                    target.position[1] - snap.position[1],
                )
                <= self.resource_reach - reach_margin
            )
            for target in targets
        ]
        entity_in_reach = [
            entity is not None
            and (
                self.regime == "macro"
                or math.hypot(
                    entity.x - snap.position[0],
                    entity.y - snap.position[1],
                )
                <= self.build_reach - reach_margin
            )
            for entity in entities
        ]
        entity_rows = [
            row
            for row, supported in zip(entities, entity_in_reach, strict=True)
            if row is not None and supported
        ]
        for index, target in enumerate(targets):
            masks["target"][index] = float(target_in_reach[index])
        for index, entity in enumerate(entities):
            masks["entity"][index] = float(entity_in_reach[index])
            masks["peer"][index] = float(entity_in_reach[index])

        # The macro-v2 item head is shared, but INSERT is the operation that
        # consumes player inventory. Keep its visible support held-only; an
        # EXTRACT pair is resolved against the selected entity at execution.
        for name, count in snap.inventory.items():
            index = S.ITEM_INDEX.get(name)
            if index is not None and count > 0:
                masks["item"][index] = 1.0
        for name, count in snap.inventory.items():
            index = S.PLACEABLE_INDEX.get(name)
            if index is not None and count > 0:
                masks["placeable"][index] = 1.0
            connector_index = S.CONNECTOR_INDEX.get(name)
            if connector_index is not None and count > 0:
                masks["connector"][connector_index] = 1.0

        sampler_state = SamplerState(
            world=self.world,
            player_pos=snap.position,
            inventory=snap.inventory,
            enabled_recipes=state.enabled_recipes,
            research_state=state.research_state,
            vocab=self.vocab,
            crafting_categories=self.crafting_categories,
        )
        craftable_by_quantity = {
            quantity: hand_craftable_recipes(sampler_state, quantity)
            for quantity in S.QUANTITIES
        }
        craftable = craftable_by_quantity[S.QUANTITIES[0]]
        for name in craftable:
            index = S.RECIPE_INDEX.get(name)
            if index is not None:
                masks["recipe"][index] = 1.0

        available_techs = available_technologies(
            state.research_state,
            trigger_technologies=self.trigger_technologies,
            current_research=snap.current_research,
            research_queue=snap.research_queue,
        )
        for name in available_techs:
            masks["technology"][S.TECH_INDEX[name]] = 1.0

        origin_x, origin_y = math.floor(snap.position[0]), math.floor(snap.position[1])
        for index in range(S.N_OFFSETS):
            dx, dy = S.offset_to_dxdy(index)
            tile = (origin_x + dx, origin_y + dy)
            centre = (tile[0] + 0.5, tile[1] + 0.5)
            # The factorised offset head is placeable-independent. Its support
            # therefore guarantees the minimum 1x1 footprint; the executor
            # validates the selected prototype's rotated footprint at this
            # exact offset without relocating it.
            legal = (
                math.hypot(centre[0] - snap.position[0], centre[1] - snap.position[1])
                <= self.build_reach
                and not self.world.is_water(*tile)
                and tile not in occupied
                and tile not in self.world.trees
                and tile not in self.world.obstacles
            )
            masks["offset"][index] = float(legal)
        masks["direction"][:] = 1.0
        if self.quantity_aware_support:
            for index, quantity in enumerate(S.QUANTITIES):
                craft_supported = bool(craftable_by_quantity[quantity])
                insert_supported = any(
                    snap.inventory.get(name, 0) >= quantity
                    for row in entity_rows
                    for name in insertable_items(row, snap.inventory, self.vocab)
                )
                extract_supported = any(
                    row.items.get(name, 0) >= quantity
                    for row in entity_rows
                    for name in extractable_items(row, self.vocab)
                )
                harvest_supported = any(target_in_reach)
                masks["quantity"][index] = float(
                    craft_supported
                    or insert_supported
                    or extract_supported
                    or harvest_supported
                )
        else:
            masks["quantity"][:] = 1.0
        masks["duration"][0] = 1.0
        if entity_rows:
            masks["duration"][:] = 1.0
        masks["move_dir"][:] = 1.0

        placeable = bool(masks["placeable"].any())
        rotatable = [row for row in entity_rows if entity_is_rotatable(row, self.vocab)]
        insert_supported = any(
            insertable_items(row, snap.inventory, self.vocab) for row in entity_rows
        )
        assembling_machine = any(
            self.vocab.entity_info.get(row.name, {}).get("type") == "assembling-machine"
            for row in entity_rows
        )
        crafting_recipes = any(
            (self.vocab.recipe_categories or {}).get(name, "crafting") == "crafting"
            for name in state.enabled_recipes
        )
        op_support = {
            "WAIT": True,
            "MOVE": True,
            "HARVEST": any(target_in_reach),
            "CRAFT": bool(craftable),
            "PLACE": placeable and bool(masks["offset"].any()),
            "PICKUP": bool(entity_rows),
            "ROTATE": bool(rotatable),
            "INSERT": insert_supported,
            "EXTRACT": any(extractable_items(row, self.vocab) for row in entity_rows),
            "SET_RECIPE": assembling_machine and crafting_recipes,
            "RESEARCH": bool(available_techs),
            "CONNECT": len(entity_rows) >= 2 and bool(masks["connector"].any()),
        }
        for op, supported in op_support.items():
            masks["op"][S.OP_INDEX[op]] = float(supported)
        assert masks["op"][S.OP_INDEX["WAIT"]] == 1.0

        for head, mask in masks.items():
            start, stop = S.MASK_OFFSETS[head]
            obs[start:stop] = mask
