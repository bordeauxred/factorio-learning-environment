"""V0 observation and state-only support-mask construction.

The buildability cache is returned by reference when ``own=False`` (the
default). Its array is mutated by the next combined world drain. Pass
``own=True`` when retaining an observation independently of the live caches.
The other spatial blocks are derived tensors and therefore are newly allocated.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fle.rl import schema as legacy_schema
from fle.rl.observation import (
    ObservationInput,
    available_technologies,
    entity_class,
    entity_dimensions,
    select_entity_slots,
    write_entity_table,
    write_globals_vector,
    write_inventory_vector,
    write_recipes_vector,
    write_research_vector,
)
from fle.rl.ops import (
    OperationSnapshot,
    SamplerState,
    VocabData,
    hand_craftable_recipes,
)
from fle.rl.v0_schema import (
    BUILD_BLOCKS_SIDE,
    BUILD_CHANNELS,
    BUILD_NEVER_SAMPLED,
    BUILD_SIDE,
    C_LOCAL,
    HEAD_SIZES,
    ITEM_INDEX,
    LOCAL_CHANNEL_INDEX,
    LOCAL_SIDE,
    MINIMAP_AGE_SCALE,
    MINIMAP_BASE_CHANNELS,
    MINIMAP_LOG1P_CHANNELS,
    MINIMAP_SIDE,
    OBS_SPEC,
    ORE_CHANNEL_FOR,
    PLACE_MAX_AGE_TICKS,
    PLACEABLE_NAMES,
    RECIPE_INDEX,
    TECH_INDEX,
    blocked_mask_for_local,
    local_origin,
)
from fle.rl.world import EntityRow

_RAW_VOCAB = json.loads(legacy_schema.VOCAB_PATH.read_text())
_DEFAULT_VOCAB = VocabData.from_dict(_RAW_VOCAB)
_DEFAULT_STATUS_NAMES = {
    int(row["code"]): str(row["name"]) for row in _RAW_VOCAB["statuses"]
}


@dataclass(frozen=True)
class MaskToggles:
    """Independent switches for every state-only mask built in this package."""

    place_item: bool = True
    inventory_item: bool = True
    entity: bool = True
    recipe: bool = True
    technology: bool = True

    def as_dict(self) -> dict[str, bool]:
        """Return all toggle values in a logging-friendly form."""
        return asdict(self)

    @property
    def active(self) -> tuple[str, ...]:
        """Names of masks that currently constrain their head."""
        return tuple(name for name, enabled in self.as_dict().items() if enabled)

    @property
    def active_masks(self) -> tuple[str, ...]:
        """Logging-friendly alias that states what the tuple contains."""
        return self.active


def _points_in_local(
    points: Sequence[tuple[int, int]], origin_x: int, origin_y: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorise sparse world points into local x/y indices and an in-frame mask."""
    if not points:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=bool)
    coordinates = np.asarray(points, dtype=np.int64)
    cell_x = coordinates[:, 0] - origin_x
    cell_y = coordinates[:, 1] - origin_y
    inside = (
        (cell_x >= 0) & (cell_x < LOCAL_SIDE) & (cell_y >= 0) & (cell_y < LOCAL_SIDE)
    )
    return cell_x, cell_y, inside


def _mark_points(
    plane: np.ndarray,
    points: Sequence[tuple[int, int]],
    origin_x: int,
    origin_y: int,
) -> None:
    cell_x, cell_y, inside = _points_in_local(points, origin_x, origin_y)
    plane[cell_y[inside], cell_x[inside]] = 1.0


def _water_chunk(mask: int) -> np.ndarray:
    """Decode one 32x32 terrain chunk without a per-tile Python loop."""
    words = np.frombuffer(mask.to_bytes(128, "big"), dtype=">u4").astype(np.uint32)
    return ((words[:, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.float16)


def _build_local_exact(
    world: Any,
    player_x: float,
    player_y: float,
    entity_info: Mapping[str, Mapping[str, object]],
) -> np.ndarray:
    local = np.zeros((C_LOCAL, LOCAL_SIDE, LOCAL_SIDE), dtype=np.float16)
    unknown = local[LOCAL_CHANNEL_INDEX["unknown_terrain"]]
    unknown.fill(1)
    origin_x, origin_y = local_origin(player_x, player_y)
    right, bottom = origin_x + LOCAL_SIDE, origin_y + LOCAL_SIDE

    # Terrain sync is chunk-granular. At most nine 32x32 chunks overlap a
    # 64x64 frame, and each chunk is decoded with vectorised bit operations.
    for (chunk_x, chunk_y), mask in world.water.items():
        chunk_left, chunk_top = chunk_x * 32, chunk_y * 32
        left, top = max(origin_x, chunk_left), max(origin_y, chunk_top)
        chunk_right, chunk_bottom = chunk_left + 32, chunk_top + 32
        overlap_right, overlap_bottom = (
            min(right, chunk_right),
            min(bottom, chunk_bottom),
        )
        if left >= overlap_right or top >= overlap_bottom:
            continue
        dst_y = slice(top - origin_y, overlap_bottom - origin_y)
        dst_x = slice(left - origin_x, overlap_right - origin_x)
        src_y = slice(top - chunk_top, overlap_bottom - chunk_top)
        src_x = slice(left - chunk_left, overlap_right - chunk_left)
        unknown[dst_y, dst_x] = 0
        if mask:
            local[LOCAL_CHANNEL_INDEX["water"], dst_y, dst_x] = _water_chunk(mask)[
                src_y, src_x
            ]

    amount_plane = local[LOCAL_CHANNEL_INDEX["resource_amount"]]
    if world.ores:
        ore_points = tuple(world.ores)
        cell_x, cell_y, inside = _points_in_local(ore_points, origin_x, origin_y)
        names = np.fromiter(
            (world.ores[point][0] for point in ore_points),
            dtype=object,
            count=len(ore_points),
        )
        amounts = np.fromiter(
            (world.ores[point][1] for point in ore_points),
            dtype=np.float64,
            count=len(ore_points),
        )
        for name, channel in ORE_CHANNEL_FOR.items():
            selected = inside & (names == name)
            local[channel, cell_y[selected], cell_x[selected]] = 1.0
        amount_plane[cell_y[inside], cell_x[inside]] = (
            np.log1p(np.maximum(amounts[inside], 0)) / 16.0
        )

    tree_plane = local[LOCAL_CHANNEL_INDEX["tree"]]
    _mark_points(tree_plane, tuple(world.trees), origin_x, origin_y)
    obstacle_plane = local[LOCAL_CHANNEL_INDEX["rock_cliff"]]
    _mark_points(obstacle_plane, tuple(world.obstacles), origin_x, origin_y)
    enemy_plane = local[LOCAL_CHANNEL_INDEX["enemy_occupancy"]]
    nest_points = tuple(
        (math.floor(x), math.floor(y)) for _, x, y in world.nests.values()
    )
    _mark_points(enemy_plane, nest_points, origin_x, origin_y)

    friendly = local[LOCAL_CHANNEL_INDEX["friendly_occupancy"]]
    dir_sin = local[LOCAL_CHANNEL_INDEX["dir_sin"]]
    dir_cos = local[LOCAL_CHANNEL_INDEX["dir_cos"]]
    class_plane = local[LOCAL_CHANNEL_INDEX["entity_class"]]
    for row in world.entities.values():
        width, height = entity_dimensions(row, entity_info)
        left = math.floor(row.x - width / 2 + 0.5)
        top = math.floor(row.y - height / 2 + 0.5)
        x0, y0 = max(left, origin_x), max(top, origin_y)
        x1, y1 = min(left + width, right), min(top + height, bottom)
        if x0 >= x1 or y0 >= y1:
            continue
        ys = slice(y0 - origin_y, y1 - origin_y)
        xs = slice(x0 - origin_x, x1 - origin_x)
        angle = row.direction / 16.0 * 2.0 * math.pi
        friendly[ys, xs] = 1.0
        dir_sin[ys, xs] = math.sin(angle)
        dir_cos[ys, xs] = math.cos(angle)
        class_index = legacy_schema.ENTITY_CLASSES.index(entity_class(row, entity_info))
        class_plane[ys, xs] = class_index / len(legacy_schema.ENTITY_CLASSES)

    player_tile_x, player_tile_y = math.floor(player_x), math.floor(player_y)
    player_cell_x = player_tile_x - origin_x
    player_cell_y = player_tile_y - origin_y
    local[LOCAL_CHANNEL_INDEX["player"], player_cell_y, player_cell_x] = 1.0
    return local


def _upsample_ticks(sampled_ticks: np.ndarray) -> np.ndarray:
    if sampled_ticks.shape == (MINIMAP_SIDE, MINIMAP_SIDE):
        return sampled_ticks
    if (
        sampled_ticks.ndim != 2
        or not sampled_ticks.size
        or MINIMAP_SIDE % sampled_ticks.shape[0]
        or MINIMAP_SIDE % sampled_ticks.shape[1]
    ):
        raise ValueError(
            f"minimap sampled_ticks has incompatible shape {sampled_ticks.shape}"
        )
    return sampled_ticks.repeat(MINIMAP_SIDE // sampled_ticks.shape[0], axis=0).repeat(
        MINIMAP_SIDE // sampled_ticks.shape[1], axis=1
    )


def _build_minimap(world: Any, tick: int) -> tuple[np.ndarray, np.ndarray]:
    cache = world.minimap
    minimap = np.empty(OBS_SPEC["minimap"][0], dtype=np.float16)
    if cache.values.shape == (MINIMAP_BASE_CHANNELS, MINIMAP_SIDE, MINIMAP_SIDE):
        # Channels 2-8 and 12 are unbounded sums (resource amounts reach the
        # hundreds of thousands per 4x4 cell) and overflow float16 to +inf,
        # silently destroying the resource signal. Compress them with log1p,
        # which keeps -1 unknown at -1 and lands every real value under 14.
        raw = cache.values
        unbounded = np.asarray(MINIMAP_LOG1P_CHANNELS, dtype=np.intp)
        bounded = np.setdiff1d(
            np.arange(MINIMAP_BASE_CHANNELS, dtype=np.intp), unbounded
        )
        # Bounded channels are fractions, flags and normalised counts, so they
        # cast straight down. The unbounded ones must be compressed BEFORE the
        # cast or they saturate float16 at +inf.
        minimap[bounded] = raw[bounded]
        block = raw[unbounded]
        minimap[unbounded] = np.where(
            block < 0.0, block, np.log1p(np.maximum(block, 0.0))
        )
        sampled_ticks = _upsample_ticks(cache.sampled_ticks)
    elif cache.values.size == 0:
        minimap[:MINIMAP_BASE_CHANNELS].fill(-1)
        sampled_ticks = np.full((MINIMAP_SIDE, MINIMAP_SIDE), -1, dtype=np.int64)
    else:
        raise ValueError(f"minimap cache has incompatible shape {cache.values.shape}")
    known = sampled_ticks >= 0
    minimap[MINIMAP_BASE_CHANNELS] = known
    age = np.clip((tick - sampled_ticks) / MINIMAP_AGE_SCALE, 0.0, 1.0)
    age[~known] = 1.0
    minimap[MINIMAP_BASE_CHANNELS + 1] = age
    return minimap, known


def _buildability_view(world: Any, own: bool) -> np.ndarray:
    values = world.buildability.values
    expected = (BUILD_CHANNELS, BUILD_SIDE, BUILD_SIDE)
    if values.shape == expected:
        return values.copy() if own else values
    if values.size == 0:
        return np.full(expected, -1, dtype=np.int8)
    raise ValueError(f"buildability cache has incompatible shape {values.shape}")


def _build_age(world: Any, tick: int) -> tuple[np.ndarray, np.ndarray]:
    sampled_ticks = world.buildability.sampled_ticks
    expected = (BUILD_CHANNELS, BUILD_BLOCKS_SIDE, BUILD_BLOCKS_SIDE)
    if sampled_ticks.shape != expected:
        if sampled_ticks.size:
            raise ValueError(
                f"buildability sampled_ticks has incompatible shape {sampled_ticks.shape}"
            )
        sampled_ticks = np.full(expected, -1, dtype=np.int64)
    known = sampled_ticks >= 0
    age = np.full(expected, BUILD_NEVER_SAMPLED, dtype=np.uint16)
    age[known] = np.clip(tick - sampled_ticks[known], 0, 65535).astype(np.uint16)
    return age, known


def _record_timing(
    timings: MutableMapping[str, float] | None, name: str, start_ns: int
) -> None:
    if timings is not None:
        timings[name] = (time.perf_counter_ns() - start_ns) / 1000.0


def build_v0_observation(
    world: Any,
    *,
    player_x: float,
    player_y: float,
    tick: int,
    inventory: Mapping[str, int] | None = None,
    enabled_recipes: Sequence[str] = (),
    research_state: Mapping[str, Mapping[str, object]] | None = None,
    vocab: VocabData | None = None,
    status_names: Mapping[int, str] | None = None,
    episode_start_tick: int = 0,
    step_count: int = 0,
    max_steps: int = 256,
    max_ticks: int = 216_000,
    last_status: str | None = None,
    last_op: str | None = None,
    current_research: str | None = None,
    research_queue: Sequence[str] = (),
    automated_score: float = 0.0,
    general_score: float = 0.0,
    resource_reach: float = 0.0,
    build_fresh_ticks: int = PLACE_MAX_AGE_TICKS,
    own: bool = False,
    timings: MutableMapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build one observation matching :data:`fle.rl.v0_schema.OBS_SPEC`.

    With ``own=False``, ``buildability`` aliases the live cache and is mutated
    by the next combined drain. ``own=True`` copies that int8 array. Derived
    tensors are necessarily materialised in their contract dtypes.

    If provided, ``timings`` receives per-block elapsed microseconds.
    """
    if max_steps <= 0 or max_ticks <= 0:
        raise ValueError("max_steps and max_ticks must be positive")
    if build_fresh_ticks < 0:
        raise ValueError("build_fresh_ticks must be nonnegative")
    inventory = inventory or {}
    research_state = research_state or {}
    vocab = vocab or _DEFAULT_VOCAB
    status_names = status_names or _DEFAULT_STATUS_NAMES
    current_research = (
        getattr(world, "research", None)
        if current_research is None
        else current_research
    )

    started = time.perf_counter_ns()
    local = _build_local_exact(world, player_x, player_y, vocab.entity_info)
    _record_timing(timings, "local_exact", started)

    started = time.perf_counter_ns()
    minimap, minimap_known = _build_minimap(world, tick)
    _record_timing(timings, "minimap", started)

    started = time.perf_counter_ns()
    buildability = _buildability_view(world, own)
    _record_timing(timings, "buildability", started)

    started = time.perf_counter_ns()
    build_age, build_known_blocks = _build_age(world, tick)
    _record_timing(timings, "build_age", started)

    started = time.perf_counter_ns()
    entity_slots = select_entity_slots(world.entities, (player_x, player_y))
    entities = np.zeros(OBS_SPEC["entities"][0], dtype=np.float32)
    write_entity_table(
        entities,
        entity_slots,
        (player_x, player_y),
        entity_info=vocab.entity_info,
        status_names=status_names,
    )
    entity_mask = np.fromiter(
        (row is not None for row in entity_slots),
        dtype=np.float32,
        count=HEAD_SIZES["entity"],
    )
    inventory_vector = np.zeros(OBS_SPEC["inventory"][0], dtype=np.float32)
    write_inventory_vector(inventory_vector, inventory)
    research = np.zeros(OBS_SPEC["research"][0], dtype=np.float32)
    write_research_vector(research, research_state)
    recipes = np.zeros(OBS_SPEC["recipes_enabled"][0], dtype=np.float32)
    write_recipes_vector(recipes, enabled_recipes)
    np.clip(entities, -1.0, 1.0, out=entities)
    np.clip(inventory_vector, -1.0, 1.0, out=inventory_vector)
    _record_timing(timings, "state_vectors", started)

    started = time.perf_counter_ns()
    snapshot = OperationSnapshot(
        inventory=dict(inventory),
        entities=world.entities,
        position=(player_x, player_y),
        tick=tick,
        score_player=general_score,
        score_automated=automated_score,
        current_research=current_research,
        research_queue=tuple(research_queue),
    )
    state = ObservationInput(
        snapshot=snapshot,
        enabled_recipes=enabled_recipes,
        research_state=research_state,
        episode_start_tick=episode_start_tick,
        step_count=step_count,
        last_status=last_status,
        last_op=last_op,
    )
    globals_vector = np.zeros(OBS_SPEC["globals"][0], dtype=np.float32)
    write_globals_vector(
        globals_vector[: legacy_schema.N_GLOBALS],
        state,
        resource_reach=resource_reach,
        max_steps=max_steps,
        max_ticks=max_ticks,
    )
    np.clip(
        globals_vector[: legacy_schema.N_GLOBALS],
        -1.0,
        1.0,
        out=globals_vector[: legacy_schema.N_GLOBALS],
    )
    extra = globals_vector[legacy_schema.N_GLOBALS :]
    sampled_ticks = world.buildability.sampled_ticks
    if sampled_ticks.shape == build_known_blocks.shape:
        fresh_blocks = (
            build_known_blocks
            & (tick - sampled_ticks >= 0)
            & (tick - sampled_ticks <= build_fresh_ticks)
        )
    else:
        fresh_blocks = np.zeros_like(build_known_blocks)
    extra[:] = (
        tick / 1e6,
        automated_score / 1e3,
        general_score / 1e3,
        max(0, tick - episode_start_tick) / 60.0 / 3600.0,
        float(np.mean(build_known_blocks)),
        float(np.mean(fresh_blocks)),
        float(np.mean(minimap_known)),
    )
    _record_timing(timings, "globals", started)

    return {
        "local_exact": local,
        "minimap": minimap,
        "buildability": buildability,
        "build_age": build_age,
        "entities": entities,
        "entity_mask": entity_mask,
        "inventory": inventory_vector,
        "research": research,
        "recipes_enabled": recipes,
        "globals": globals_vector,
    }


def build_v0_masks(
    world: Any,
    obs: Mapping[str, np.ndarray],
    *,
    inventory: Mapping[str, int],
    enabled_recipes: Sequence[str],
    research_state: Mapping[str, Mapping[str, object]],
    vocab: VocabData | None = None,
    crafting_categories: Sequence[str] = ("crafting",),
    trigger_technologies: set[str] | frozenset[str] = frozenset(),
    current_research: str | None = None,
    research_queue: Sequence[str] = (),
    toggles: MaskToggles | None = None,
) -> dict[str, np.ndarray]:
    """Build the five state-only V0 support masks owned by this package.

    A disabled toggle returns an all-ones vector for that head. Prefix-dependent
    PLACE-location and contained-item masks are exposed separately below.
    """
    vocab = vocab or _DEFAULT_VOCAB
    toggles = toggles or MaskToggles()
    masks = {
        name: np.ones(HEAD_SIZES[name], dtype=np.uint8) for name in toggles.as_dict()
    }

    if toggles.place_item:
        mask = masks["place_item"]
        mask.fill(0)
        for place_index, name in enumerate(PLACEABLE_NAMES):
            item_index = ITEM_INDEX.get(name)
            if item_index is not None and obs["inventory"][item_index] > 0:
                mask[place_index] = 1
    if toggles.inventory_item:
        masks["inventory_item"][:] = obs["inventory"] > 0
    if toggles.entity:
        masks["entity"][:] = obs["entity_mask"] > 0
    if toggles.recipe:
        masks["recipe"].fill(0)
        sampler_state = SamplerState(
            world=world,
            player_pos=(
                float(getattr(world, "player_x", 0.0)),
                float(getattr(world, "player_y", 0.0)),
            ),
            inventory=inventory,
            enabled_recipes=enabled_recipes,
            research_state=research_state,
            vocab=vocab,
            crafting_categories=crafting_categories,
        )
        for name in hand_craftable_recipes(sampler_state):
            index = RECIPE_INDEX.get(name)
            if index is not None:
                masks["recipe"][index] = 1
    if toggles.technology:
        masks["technology"].fill(0)
        available = available_technologies(
            research_state,
            trigger_technologies=trigger_technologies,
            current_research=(
                getattr(world, "research", None)
                if current_research is None
                else current_research
            ),
            research_queue=research_queue,
        )
        for name in available:
            masks["technology"][TECH_INDEX[name]] = 1
    return masks


def build_place_location_mask(
    world: Any,
    *,
    place_item: str,
    direction: int,
    player_x: float,
    player_y: float,
    tick: int,
    max_age_ticks: int = PLACE_MAX_AGE_TICKS,
) -> np.ndarray:
    """Prefix-dependent PLACE-location support for package B.

    Lookup misses, unknown cells, stale cells and cells outside the cache are
    allowed. Only freshly proven blocked cells are removed.
    """
    try:
        channel = world.buildability.channel_for(place_item, direction)
    except KeyError:
        return np.ones(HEAD_SIZES["location"], dtype=np.uint8)
    blocked = blocked_mask_for_local(
        world.buildability,
        channel,
        tick,
        player_x,
        player_y,
        max_age_ticks,
    )
    return np.logical_not(blocked).reshape(-1).astype(np.uint8)


def build_contained_item_mask(entity: EntityRow | None) -> np.ndarray:
    """Prefix-dependent contained-item support for package B."""
    mask = np.zeros(HEAD_SIZES["contained_item"], dtype=np.uint8)
    if entity is None:
        return mask
    for name, count in entity.items.items():
        index = ITEM_INDEX.get(name)
        if index is not None and count > 0:
            mask[index] = 1
    return mask
