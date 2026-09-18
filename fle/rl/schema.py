"""Shared contract between the FLE macro-action environment and its learners.

Everything a learner needs to know about the tensor interface lives here:
head names and sizes, observation block layout, and how action masks are
packed into the observation tail. The environment (``fle/rl/env.py``) and the
learners import these constants; nothing is hard-coded twice.

Design (see docs/rl/specs/learner-env.md):

- Action: ``MultiDiscrete(HEAD_SIZES)``. The policy samples every head; the
  environment reads only the heads the chosen op uses (``OP_HEADS``).
- Masks: per head, a binary vector packed at the end of the observation in
  ``HEADS`` order. A mask is state-only support: it removes what the simulator
  deterministically refuses in the current state, never a preference.
- Observation: float32 ``Box`` of length ``OBS_SIZE``; block offsets in
  ``OBS_LAYOUT``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fle.env.game_types import prototype_by_name

VOCAB_PATH = Path(__file__).resolve().parents[2] / "data/rl/vocab/factorio-2.0.73-base-v1.json"

_vocab = json.loads(VOCAB_PATH.read_text())
ITEM_NAMES: tuple[str, ...] = tuple(row["name"] for row in _vocab["items"])
PLACEABLE_NAMES: tuple[str, ...] = tuple(
    row["name"]
    for row in _vocab["items"]
    if row.get("place_result") in prototype_by_name
)
CONNECTOR_NAMES: tuple[str, ...] = tuple(
    name
    for name in (
        "transport-belt",
        "fast-transport-belt",
        "pipe",
        "small-electric-pole",
        "medium-electric-pole",
    )
    if name in prototype_by_name
)
RECIPE_NAMES: tuple[str, ...] = tuple(row["name"] for row in _vocab["recipes"])
TECH_NAMES: tuple[str, ...] = tuple(row["name"] for row in _vocab["technologies"])
ITEM_INDEX = {name: i for i, name in enumerate(ITEM_NAMES)}
PLACEABLE_INDEX = {name: i for i, name in enumerate(PLACEABLE_NAMES)}
CONNECTOR_INDEX = {name: i for i, name in enumerate(CONNECTOR_NAMES)}
RECIPE_INDEX = {name: i for i, name in enumerate(RECIPE_NAMES)}
TECH_INDEX = {name: i for i, name in enumerate(TECH_NAMES)}

# --- operations -----------------------------------------------------------
OPS: tuple[str, ...] = (
    "WAIT", "MOVE", "HARVEST", "CRAFT", "PLACE", "PICKUP",
    "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE", "RESEARCH", "CONNECT",
)
OP_INDEX = {name: i for i, name in enumerate(OPS)}

# --- argument heads ---------------------------------------------------------
N_TARGET_SLOTS = 24          # 16 nearest ore patches + 8 nearest trees (one per compass octant)
N_PATCH_SLOTS = 16
N_TREE_SLOTS = 8
N_ENTITY_SLOTS = 32          # nearest tracked entities
OFFSET_RADIUS = 8            # PLACE offset dx, dy in [-8, 8] relative to the player
N_OFFSETS = (2 * OFFSET_RADIUS + 1) ** 2   # 289
DIRECTIONS: tuple[int, ...] = (0, 4, 8, 12)              # Factorio 2.0 cardinal codes
QUANTITIES: tuple[int, ...] = (1, 5, 20)
DURATIONS_TICKS: tuple[int, ...] = (60, 300, 900, 3600)
MOVE_DIRS = 8                # compass octants, MOVE_STEP tiles each
MOVE_STEP = 16
TARGET_RADIUS = 160          # tiles; targets beyond this are not offered

HEADS: tuple[str, ...] = (
    "op", "target", "entity", "item", "placeable", "recipe", "technology",
    "offset", "direction", "quantity", "duration", "move_dir", "peer", "connector",
)
HEAD_SIZES: dict[str, int] = {
    "op": len(OPS),
    "target": N_TARGET_SLOTS,
    "entity": N_ENTITY_SLOTS,
    "item": len(ITEM_NAMES),
    "placeable": len(PLACEABLE_NAMES),
    "recipe": len(RECIPE_NAMES),
    "technology": len(TECH_NAMES),
    "offset": N_OFFSETS,
    "direction": len(DIRECTIONS),
    "quantity": len(QUANTITIES),
    "duration": len(DURATIONS_TICKS),
    "move_dir": MOVE_DIRS,
    "peer": N_ENTITY_SLOTS,
    "connector": len(CONNECTOR_NAMES),
}
HEAD_DIMS: tuple[int, ...] = tuple(HEAD_SIZES[h] for h in HEADS)
MASK_SIZE = sum(HEAD_DIMS)

# Which argument heads each op reads. Other heads are ignored for that op.
OP_HEADS: dict[str, tuple[str, ...]] = {
    "WAIT": ("duration",),
    "MOVE": ("move_dir",),
    "HARVEST": ("target", "quantity"),
    "CRAFT": ("recipe", "quantity"),
    "PLACE": ("placeable", "offset", "direction"),
    "PICKUP": ("entity",),
    "ROTATE": ("entity", "direction"),
    "INSERT": ("entity", "item", "quantity"),
    "EXTRACT": ("entity", "item", "quantity"),
    "SET_RECIPE": ("entity", "recipe"),
    "RESEARCH": ("technology",),
    "CONNECT": ("entity", "peer", "connector"),
}


def offset_to_dxdy(index: int) -> tuple[int, int]:
    side = 2 * OFFSET_RADIUS + 1
    return index % side - OFFSET_RADIUS, index // side - OFFSET_RADIUS


def dxdy_to_offset(dx: int, dy: int) -> int:
    side = 2 * OFFSET_RADIUS + 1
    return (dy + OFFSET_RADIUS) * side + (dx + OFFSET_RADIUS)


# --- observation layout -----------------------------------------------------
N_GLOBALS = 16
TARGET_KINDS: tuple[str, ...] = ("iron-ore", "copper-ore", "stone", "coal", "other-resource", "tree")
TARGET_FEATURES = len(TARGET_KINDS) + 6      # kind one-hot, dx, dy, dist, log size, valid, pad = 12
ENTITY_CLASSES: tuple[str, ...] = (
    "mining-drill", "furnace", "container", "assembling-machine",
    "transport-belt", "inserter", "electric-pole", "pipe", "other",
)
STATUS_GROUPS: tuple[str, ...] = ("working", "no_fuel_or_power", "no_input", "output_full", "other")
# class one-hot 9, dx, dy, dist, dir, status one-hot 5, has_recipe, recipe_id,
# log item total, log fuel count, valid, pad = 24
ENTITY_FEATURES = len(ENTITY_CLASSES) + 4 + len(STATUS_GROUPS) + 6
GRID_CHANNELS: tuple[str, ...] = ("entity", "resource", "obstacle", "water")
GRID_SIDE = 2 * OFFSET_RADIUS + 1             # 17, same window as the PLACE offset head

_blocks = [
    ("globals", N_GLOBALS),
    ("inventory", len(ITEM_NAMES)),
    ("tech", len(TECH_NAMES)),
    ("recipes_enabled", len(RECIPE_NAMES)),
    ("targets", N_TARGET_SLOTS * TARGET_FEATURES),
    ("entities", N_ENTITY_SLOTS * ENTITY_FEATURES),
    ("grid", len(GRID_CHANNELS) * GRID_SIDE * GRID_SIDE),
    ("masks", MASK_SIZE),
]
OBS_LAYOUT: dict[str, tuple[int, int]] = {}
_cursor = 0
for _name, _size in _blocks:
    OBS_LAYOUT[_name] = (_cursor, _cursor + _size)
    _cursor += _size
OBS_SIZE = _cursor

MASK_OFFSETS: dict[str, tuple[int, int]] = {}
_cursor = OBS_LAYOUT["masks"][0]
for _head in HEADS:
    MASK_OFFSETS[_head] = (_cursor, _cursor + HEAD_SIZES[_head])
    _cursor += HEAD_SIZES[_head]
assert _cursor == OBS_SIZE

SCHEMA_VERSION = "macro-v2"


def split_masks(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-head boolean masks from one observation or a batch (..., OBS_SIZE)."""
    return {head: obs[..., a:b] > 0.5 for head, (a, b) in MASK_OFFSETS.items()}


def random_valid_action(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random action respecting the packed masks. Index 0 if a head is empty."""
    masks = split_masks(obs)
    action = np.zeros(len(HEADS), dtype=np.int64)
    for i, head in enumerate(HEADS):
        valid = np.flatnonzero(masks[head])
        action[i] = int(rng.choice(valid)) if valid.size else 0
    return action


def describe() -> str:
    lines = [f"schema {SCHEMA_VERSION}: obs {OBS_SIZE} floats, {len(HEADS)} heads, mask {MASK_SIZE}"]
    for name, (a, b) in OBS_LAYOUT.items():
        lines.append(f"  {name:16s} [{a:5d}, {b:5d})  {b - a}")
    lines.append("  heads: " + ", ".join(f"{h}={HEAD_SIZES[h]}" for h in HEADS))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
