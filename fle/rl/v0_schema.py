"""V0 tensor contract: one exact local frame, three verbs, buildability both ways.

This module is the interface between the V0 observation builder, the V0 action
executor and the learner. It is written once and imported read-only by all of
them, so the coordinate transform and the head sizes cannot drift apart.

See docs/rl/specs/v0-contract.md. Where code and prose disagree, this file is
what runs.
"""

from __future__ import annotations

import numpy as np

from fle.rl.schema import (  # noqa: F401 - re-exported vocab
    ITEM_INDEX,
    ITEM_NAMES,
    N_ENTITY_SLOTS,
    PLACEABLE_INDEX,
    PLACEABLE_NAMES,
    RECIPE_INDEX,
    RECIPE_NAMES,
    TECH_INDEX,
    TECH_NAMES,
)

# --- exact local frame ------------------------------------------------------
LOCAL_SIDE = 64
LOCAL_CELLS = LOCAL_SIDE * LOCAL_SIDE
LOCAL_HALF = LOCAL_SIDE // 2  # the player sits at cell (32, 32)
MOVE_MAX_TILES = LOCAL_HALF  # every in-window cell is within 32 tiles

LOCAL_CHANNELS: tuple[str, ...] = (
    "water",
    "iron_ore",
    "copper_ore",
    "coal",
    "stone",
    "uranium_ore",
    "crude_oil",
    "resource_amount",
    "tree",
    "rock_cliff",
    "friendly_occupancy",
    "enemy_occupancy",
    "player",
    "dir_sin",
    "dir_cos",
    "entity_class",
    "unknown_terrain",
)
C_LOCAL = len(LOCAL_CHANNELS)
LOCAL_CHANNEL_INDEX = {name: i for i, name in enumerate(LOCAL_CHANNELS)}
ORE_CHANNEL_FOR = {
    "iron-ore": LOCAL_CHANNEL_INDEX["iron_ore"],
    "copper-ore": LOCAL_CHANNEL_INDEX["copper_ore"],
    "coal": LOCAL_CHANNEL_INDEX["coal"],
    "stone": LOCAL_CHANNEL_INDEX["stone"],
    "uranium-ore": LOCAL_CHANNEL_INDEX["uranium_ore"],
    "crude-oil": LOCAL_CHANNEL_INDEX["crude_oil"],
}

# --- map views --------------------------------------------------------------
MINIMAP_SIDE = 128
MINIMAP_BASE_CHANNELS = 14  # Jack's channels, verbatim
C_MINIMAP = MINIMAP_BASE_CHANNELS + 2  # + known, + age
MINIMAP_AGE_SCALE = 3600.0
# Unbounded minimap channels: the five ore sums, crude oil, tree count and
# enemy unit count. Stored as log1p because raw amounts overflow float16.
MINIMAP_LOG1P_CHANNELS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 12)
MINIMAP_STRIDE = 4  # world tiles per minimap cell

BUILD_CHANNELS = 63
BUILD_SIDE = 128
BUILD_BLOCK = 8
BUILD_BLOCKS_SIDE = BUILD_SIDE // BUILD_BLOCK  # 16
BUILD_NEVER_SAMPLED = np.uint16(65535)
PLACE_MAX_AGE_TICKS = 600

# --- observation keys, shapes and dtypes ------------------------------------
OBS_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "local_exact": ((C_LOCAL, LOCAL_SIDE, LOCAL_SIDE), np.dtype(np.float16)),
    "minimap": ((C_MINIMAP, MINIMAP_SIDE, MINIMAP_SIDE), np.dtype(np.float16)),
    "buildability": ((BUILD_CHANNELS, BUILD_SIDE, BUILD_SIDE), np.dtype(np.int8)),
    "build_age": (
        (BUILD_CHANNELS, BUILD_BLOCKS_SIDE, BUILD_BLOCKS_SIDE),
        np.dtype(np.uint16),
    ),
    "entities": ((N_ENTITY_SLOTS, 24), np.dtype(np.float32)),
    "entity_mask": ((N_ENTITY_SLOTS,), np.dtype(np.float32)),
    "inventory": ((len(ITEM_NAMES),), np.dtype(np.float32)),
    "research": ((len(TECH_NAMES),), np.dtype(np.float32)),
    "recipes_enabled": ((len(RECIPE_NAMES),), np.dtype(np.float32)),
    "globals": ((23,), np.dtype(np.float32)),
}


def obs_nbytes() -> int:
    """Bytes one stored state occupies, masks excluded."""
    return sum(
        int(np.prod(shape)) * dtype.itemsize for shape, dtype in OBS_SPEC.values()
    )


# --- verbs and heads --------------------------------------------------------
VERBS: tuple[str, ...] = (
    "MOVE_TO",
    "MINE",
    "PLACE",
    "CRAFT",
    "PICKUP",
    "ROTATE",
    "INSERT",
    "EXTRACT",
    "SET_RECIPE",
    "RESEARCH",
    "FAST_FORWARD",
)
VERB_INDEX = {name: i for i, name in enumerate(VERBS)}

DIRECTIONS: tuple[int, ...] = (0, 4, 8, 12)  # engine cardinals; no diagonals
QUANTITY_BINS: tuple[int | str, ...] = (1, 2, 4, 8, 16, 32, "ALL")
FAST_FORWARD_SECONDS: tuple[int, ...] = (1, 10, 60)

HEAD_SIZES: dict[str, int] = {
    "verb": len(VERBS),
    "location": LOCAL_CELLS,
    "place_item": len(PLACEABLE_NAMES),
    "direction": len(DIRECTIONS),
    "quantity": len(QUANTITY_BINS),
    "entity": N_ENTITY_SLOTS,
    "inventory_item": len(ITEM_NAMES),
    "contained_item": len(ITEM_NAMES),
    "recipe": len(RECIPE_NAMES),
    "technology": len(TECH_NAMES),
    "duration": len(FAST_FORWARD_SECONDS),
}
HEADS: tuple[str, ...] = tuple(HEAD_SIZES)

# Decode order per verb. Every later head conditions on the earlier choices.
VERB_HEADS: dict[str, tuple[str, ...]] = {
    "MOVE_TO": ("location",),
    "MINE": ("location", "quantity"),
    "PLACE": ("place_item", "direction", "location"),
    "CRAFT": ("recipe", "quantity"),
    "PICKUP": ("entity",),
    "ROTATE": ("entity", "direction"),
    "INSERT": ("entity", "inventory_item", "quantity"),
    "EXTRACT": ("entity", "contained_item", "quantity"),
    "SET_RECIPE": ("entity", "recipe"),
    "RESEARCH": ("technology",),
    "FAST_FORWARD": ("duration",),
}
assert set(VERB_HEADS) == set(VERBS)
assert all(h in HEAD_SIZES for heads in VERB_HEADS.values() for h in heads)

SMDP_HALF_LIFE_SECONDS = 3600.0


def smdp_discount(tau_seconds):
    """Gamma(tau) = 2 ** (-tau / 3600), elementwise and array-safe."""
    return np.exp2(-np.asarray(tau_seconds, dtype=np.float64) / SMDP_HALF_LIFE_SECONDS)


# --- the one coordinate transform -------------------------------------------
def local_origin(player_x: float, player_y: float) -> tuple[int, int]:
    """World tile at local cell (0, 0)."""
    return int(np.floor(player_x)) - LOCAL_HALF, int(np.floor(player_y)) - LOCAL_HALF


def cell_to_world(cell: int, player_x: float, player_y: float) -> tuple[int, int]:
    """Location head index -> world tile. Inverse of world_to_cell."""
    if not 0 <= cell < LOCAL_CELLS:
        raise ValueError(f"cell {cell} outside the {LOCAL_SIDE}x{LOCAL_SIDE} frame")
    ox, oy = local_origin(player_x, player_y)
    y_cell, x_cell = divmod(cell, LOCAL_SIDE)
    return ox + x_cell, oy + y_cell


def world_to_cell(tile_x: int, tile_y: int, player_x: float, player_y: float) -> int:
    """World tile -> location head index. Raises if outside the frame."""
    ox, oy = local_origin(player_x, player_y)
    x_cell, y_cell = int(tile_x) - ox, int(tile_y) - oy
    if not (0 <= x_cell < LOCAL_SIDE and 0 <= y_cell < LOCAL_SIDE):
        raise ValueError(f"tile ({tile_x}, {tile_y}) outside the local frame")
    return y_cell * LOCAL_SIDE + x_cell


def crop_to_local(
    plane: np.ndarray,
    plane_origin: tuple[int, int],
    player_x: float,
    player_y: float,
    fill,
) -> np.ndarray:
    """Crop a world-anchored plane into the 64x64 local frame.

    `plane` is (H, W) or (C, H, W) with world tile `plane_origin` at index
    [..., 0, 0]. Cells of the local frame that the plane does not cover take
    `fill`. Partial overlap is normal: the buildability window is 8-aligned and
    only recentres after a 24-tile dead zone, so the local frame routinely
    hangs over its edge.
    """
    single = plane.ndim == 2
    src = plane[None] if single else plane
    channels = src.shape[0]
    out = np.full((channels, LOCAL_SIDE, LOCAL_SIDE), fill, dtype=src.dtype)
    ox, oy = local_origin(player_x, player_y)
    px, py = plane_origin
    height, width = src.shape[1], src.shape[2]

    # Intersection in world tiles, then the same window in both index spaces.
    left, top = max(ox, px), max(oy, py)
    right, bottom = min(ox + LOCAL_SIDE, px + width), min(oy + LOCAL_SIDE, py + height)
    if left < right and top < bottom:
        out[:, top - oy : bottom - oy, left - ox : right - ox] = src[
            :, top - py : bottom - py, left - px : right - px
        ]
    return out[0] if single else out


def blocked_mask_for_local(
    cache,
    channel: int,
    current_tick: int,
    player_x: float,
    player_y: float,
    max_age_ticks: int = PLACE_MAX_AGE_TICKS,
) -> np.ndarray:
    """(64, 64) bool: True only where placement is a certain refusal.

    Unknown, stale and outside-the-cache cells are False, i.e. allowed. The
    policy may always attempt them and the engine decides.
    """
    if cache.size == 0:
        return np.zeros((LOCAL_SIDE, LOCAL_SIDE), dtype=bool)
    blocked = cache.certainly_blocked_mask(current_tick, max_age_ticks)[channel]
    return crop_to_local(blocked, cache.origin, player_x, player_y, fill=False)
