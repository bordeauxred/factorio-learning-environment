"""SM-ARQ interface contract: the frozen boundary between environment, network,
learner and training loop.

Every parallel implementation session imports names from this module and from
nothing else of each other's code.  Nothing here may import torch or the FLE
runtime: the learner side must be able to import this module without a running
Factorio server, and the environment side without torch.

Design commitments this file encodes (do not relax them without changing the
spec first):

* The policy owns geometry.  ``POSITION`` is an exact-tile head over an
  egocentric window; the executor never moves a requested coordinate.
* Locomotion is abstract.  Navigation happens inside an action, costs simulated
  game time, and never changes the requested target.
* Time is simulated.  Every duration in a transition is measured in Factorio
  ticks; wall-clock time is a performance metric only.
* Action construction is a sequence of zero-time internal decisions.  Only a
  completed action advances the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "sm-arq-1"

TICKS_PER_SECOND = 60

# --------------------------------------------------------------------------
# Verbs
# --------------------------------------------------------------------------

VERBS: tuple[str, ...] = (
    "MOVE_TO",
    "MINE",
    "CRAFT",
    "PLACE",
    "PICKUP",
    "ROTATE",
    "INSERT",
    "EXTRACT",
    "SET_RECIPE",
    "RESEARCH",
    "FAST_FORWARD",
)
VERB_INDEX: dict[str, int] = {v: i for i, v in enumerate(VERBS)}
N_VERBS = len(VERBS)

# --------------------------------------------------------------------------
# Heads
#
# A head is one autoregressive decision.  ``HEAD_SEQUENCE[verb]`` is the exact
# order in which heads are decided for that verb; every later head conditions on
# all earlier selections.  Heads absent from a verb's sequence are not decided
# and must be recorded as -1 in an encoded action.
# --------------------------------------------------------------------------

POSITION = "position"
PROTOTYPE = "prototype"
DIRECTION = "direction"
ENTITY = "entity"
ITEM = "item"
QUANTITY = "quantity"
RECIPE = "recipe"
TECHNOLOGY = "technology"
DURATION = "duration"

HEADS: tuple[str, ...] = (
    POSITION,
    PROTOTYPE,
    DIRECTION,
    ENTITY,
    ITEM,
    QUANTITY,
    RECIPE,
    TECHNOLOGY,
    DURATION,
)
HEAD_INDEX: dict[str, int] = {h: i for i, h in enumerate(HEADS)}

HEAD_SEQUENCE: dict[str, tuple[str, ...]] = {
    "MOVE_TO": (POSITION,),
    "MINE": (POSITION, QUANTITY),
    "CRAFT": (RECIPE, QUANTITY),
    "PLACE": (PROTOTYPE, POSITION, DIRECTION),
    "PICKUP": (ENTITY,),
    "ROTATE": (ENTITY, DIRECTION),
    "INSERT": (ENTITY, ITEM, QUANTITY),
    "EXTRACT": (ENTITY, ITEM, QUANTITY),
    "SET_RECIPE": (ENTITY, RECIPE),
    "RESEARCH": (TECHNOLOGY,),
    "FAST_FORWARD": (DURATION,),
}

# Directions are the four cardinals only, in Factorio's 2.0 numbering.
DIRECTIONS: tuple[str, ...] = ("NORTH", "EAST", "SOUTH", "WEST")
DIRECTION_TO_FACTORIO: dict[str, int] = {
    "NORTH": 0,
    "EAST": 4,
    "SOUTH": 8,
    "WEST": 12,
}
N_DIRECTIONS = len(DIRECTIONS)

# Quantity buckets.  ``ALL`` means "as many as the game will give for this
# action", resolved by the executor at execution time (all of a stack, the whole
# inventory slot, the full craftable count).  It is a genuine policy choice, not
# a clamp: the requested bucket is always logged next to what executed.
QUANTITIES: tuple[int | str, ...] = (1, 2, 4, 8, 16, 32, "ALL")
N_QUANTITIES = len(QUANTITIES)

# FAST_FORWARD durations in simulated seconds.  These are simulated, executed as
# fast as the simulator manages; no intentional wall-clock sleeping.
DURATIONS_SECONDS: tuple[int, ...] = (1, 10, 60)
DURATIONS_TICKS: tuple[int, ...] = tuple(s * TICKS_PER_SECOND for s in DURATIONS_SECONDS)
N_DURATIONS = len(DURATIONS_SECONDS)

# --------------------------------------------------------------------------
# Observation shapes
# --------------------------------------------------------------------------

GRID_CHANNELS = 17
GRID_SIZE = 96  # cells per side, 3 tiles per cell (PR #414 tiered client)
GRID_CELL_TILES = 3

ENTITY_SLOTS = 2048
ENTITY_FEATURES = 38

N_GLOBALS = 10

# Exact-tile egocentric raster used by the spatial head.  Side length in tiles;
# configurable, 288 is the target for full runs if it is affordable.
RASTER_TILES_DEFAULT = 96
RASTER_CHANNELS = 8
RASTER_CHANNEL_NAMES: tuple[str, ...] = (
    "occupied",  # any entity footprint covers the tile
    "impassable",  # water, cliff, or other non-walkable terrain
    "resource",  # ore / resource entity present
    "belt",  # transport belt footprint
    "machine",  # crafting machine / furnace footprint
    "tree_rock",  # trees and rocks (minable obstruction)
    "player",  # the character's own tile
    "in_build_reach",  # within build_distance of the character right now
)

# The tile addressed by raster index ``p`` for a raster of side ``R`` whose
# window origin (top-left, inclusive) is ``(ox, oy)``:
#     x = ox + (p % R)
#     y = oy + (p // R)
# The origin is reported in every observation as ``raster_origin`` so that a
# replayed transition resolves to the same absolute tiles it was chosen for.


@dataclass(frozen=True)
class Observation:
    """One decision-boundary observation.  All arrays are float32 except masks.

    ``raster_origin`` is the absolute tile coordinate of raster index 0 and is
    required to decode a POSITION selection into an exact world tile.
    """

    grid: np.ndarray  # (GRID_CHANNELS, GRID_SIZE, GRID_SIZE) float32
    entity_view: np.ndarray  # (ENTITY_SLOTS, ENTITY_FEATURES) float32
    entity_mask: np.ndarray  # (ENTITY_SLOTS,) bool - slot holds a live entity
    entity_ids: np.ndarray  # (ENTITY_SLOTS,) int64 - stable unit numbers, 0 = empty
    globals: np.ndarray  # (N_GLOBALS,) float32
    raster: np.ndarray  # (RASTER_CHANNELS, R, R) float32
    raster_origin: tuple[int, int]  # absolute tile of raster index 0
    raster_tiles: int  # R
    player_tile: tuple[int, int]
    tick: int
    production_score: float
    automated_production_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid,
            "entity_view": self.entity_view,
            "entity_mask": self.entity_mask,
            "entity_ids": self.entity_ids,
            "globals": self.globals,
            "raster": self.raster,
            "raster_origin": self.raster_origin,
            "raster_tiles": self.raster_tiles,
            "player_tile": self.player_tile,
            "tick": self.tick,
            "production_score": self.production_score,
            "automated_production_score": self.automated_production_score,
        }


# --------------------------------------------------------------------------
# Masks
#
# Structural masks only.  A mask may remove a choice that is a type or syntax
# impossibility, never one that is merely a bad idea, unaffordable, occupied,
# out of reach, or strategically pointless.  In particular:
#   * the POSITION head is NEVER masked;
#   * inventory contents never mask PROTOTYPE, ITEM, RECIPE or QUANTITY;
#   * occupancy, reachability and layout quality never mask anything.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Masks:
    """Per-head legality, valid at this decision boundary.

    ``entity`` masks depend on the verb being constructed, so they are supplied
    per verb: ``entity[verb_index]`` is the (ENTITY_SLOTS,) mask for that verb.
    ``recipe_for_entity`` is consulted only by SET_RECIPE, after the entity has
    been chosen: ``recipe_for_entity(slot) -> (n_recipes,) bool``.
    """

    verb: np.ndarray  # (N_VERBS,) bool
    prototype: np.ndarray  # (n_prototypes,) bool - placeable prototypes
    item: np.ndarray  # (n_items,) bool - items that exist in the vocab
    craft_recipe: np.ndarray  # (n_recipes,) bool - hand-craftable category
    technology: np.ndarray  # (n_technologies,) bool - exists and not researched
    entity: np.ndarray  # (N_VERBS, ENTITY_SLOTS) bool
    recipe_for_entity: Any = None  # callable(slot:int) -> np.ndarray | None


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """A complete semantic action, as chosen by the policy.

    ``heads`` holds the raw index selected at each head, -1 where the head does
    not participate in this verb; it is what the learner replays.  The decoded
    fields are what the executor uses and what the logs record.
    """

    verb: str
    heads: np.ndarray  # (len(HEADS),) int64, -1 where unused

    # Decoded arguments, present only where the verb uses them.
    tile: tuple[int, int] | None = None  # exact absolute tile from POSITION
    prototype: str | None = None
    direction: str | None = None
    entity_slot: int | None = None
    entity_id: int | None = None  # stable unit number of the pointed entity
    item: str | None = None
    quantity: int | str | None = None
    recipe: str | None = None
    technology: str | None = None
    duration_seconds: int | None = None

    def head(self, name: str) -> int:
        return int(self.heads[HEAD_INDEX[name]])


FAILURE_REASONS: tuple[str, ...] = (
    "ok",
    "blocked",  # the exact requested tile cannot take the entity
    "unreachable",  # no interaction position could be reached
    "insufficient_inventory",
    "no_such_entity",  # the pointed slot no longer holds a live entity
    "invalid_argument",  # the game refused the combination outright
    "tool_error",  # anything else the game reported
)


@dataclass
class StepResult:
    """What one completed semantic action produced.

    ``duration_ticks`` is the simulated time the action consumed, including any
    navigation, and is the tau of the SMDP.  ``wall_seconds`` is a performance
    metric and must never enter a learning target.
    """

    observation: Observation
    reward: float  # delta automated_production_score by default
    done: bool
    duration_ticks: int
    duration_game_seconds: float
    wall_seconds: float
    success: bool
    failure_reason: str
    production_score: float
    automated_production_score: float
    delta_production_score: float
    delta_automated_production_score: float
    episode_ticks: int
    episode_decisions: int
    end_reason: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Episode and discounting
# --------------------------------------------------------------------------

EPISODE_GAME_MINUTES_DEFAULT = 30
EPISODE_TICK_BUDGET_DEFAULT = EPISODE_GAME_MINUTES_DEFAULT * 60 * TICKS_PER_SECOND
EPISODE_DECISION_CAP_DEFAULT = 1000

# Continuous-time discount horizon, in simulated seconds.
DISCOUNT_HORIZON_SECONDS_DEFAULT = 3600.0


def smdp_discount(tau_seconds: float, horizon_seconds: float = DISCOUNT_HORIZON_SECONDS_DEFAULT) -> float:
    """Gamma(tau) = 2 ** (-tau / H).  tau is simulated game time, never wall time."""
    if tau_seconds < 0:
        raise ValueError("tau must be non-negative")
    return float(2.0 ** (-tau_seconds / horizon_seconds))


REWARD_MODES: tuple[str, ...] = ("automated", "production")


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------


@dataclass
class Transition:
    """One SMDP transition (s, a, r, tau, s').

    Internal autoregressive choices are not stored as transitions: a stored
    transition always corresponds to a completed semantic action.  The learner
    reconstructs the internal chain from ``action.heads`` and the stored masks.
    """

    observation: Mapping[str, Any]
    action_heads: np.ndarray  # (len(HEADS),) int64
    verb: int
    reward: float
    tau_seconds: float
    next_observation: Mapping[str, Any]
    done: bool
    success: bool
    failure_reason: str
    # Masks are stored so that replayed maxima respect the legality that held
    # at collection time.
    masks: Mapping[str, Any] | None = None
    next_masks: Mapping[str, Any] | None = None
    is_demo: bool = False


# --------------------------------------------------------------------------
# Environment protocol
# --------------------------------------------------------------------------


class SemanticEnvProtocol:
    """What ``fle.smarq.env.SemanticEnv`` and ``fle.smarq.fake.FakeSemanticEnv``
    both provide.  The training loop and the tests are written against this and
    nothing else.
    """

    def reset(self, seed: int | None = None) -> tuple[Observation, Masks]:
        raise NotImplementedError

    def step(self, action: Action) -> tuple[StepResult, Masks]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def vocab(self) -> "VocabProtocol":
        raise NotImplementedError


class VocabProtocol:
    """Stable integer vocabularies shared by env and network.

    Index 0 of every vocabulary is the reserved ``<none>`` symbol so that an
    absent argument embeds to a learned null rather than to a real symbol.
    """

    prototypes: Sequence[str]
    items: Sequence[str]
    recipes: Sequence[str]
    technologies: Sequence[str]
    entity_types: Sequence[str]

    def head_sizes(self, raster_tiles: int) -> dict[str, int]:
        return {
            POSITION: raster_tiles * raster_tiles,
            PROTOTYPE: len(self.prototypes),
            DIRECTION: N_DIRECTIONS,
            ENTITY: ENTITY_SLOTS,
            ITEM: len(self.items),
            QUANTITY: N_QUANTITIES,
            RECIPE: len(self.recipes),
            TECHNOLOGY: len(self.technologies),
            DURATION: N_DURATIONS,
        }


def empty_heads() -> np.ndarray:
    return np.full(len(HEADS), -1, dtype=np.int64)


def position_to_tile(index: int, origin: tuple[int, int], raster_tiles: int) -> tuple[int, int]:
    if not 0 <= index < raster_tiles * raster_tiles:
        raise ValueError(f"position index {index} out of range for raster {raster_tiles}")
    return (origin[0] + index % raster_tiles, origin[1] + index // raster_tiles)


def tile_to_position(tile: tuple[int, int], origin: tuple[int, int], raster_tiles: int) -> int | None:
    dx, dy = tile[0] - origin[0], tile[1] - origin[1]
    if not (0 <= dx < raster_tiles and 0 <= dy < raster_tiles):
        return None
    return int(dy * raster_tiles + dx)
