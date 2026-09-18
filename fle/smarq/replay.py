"""Proportional prioritized replay with compact observation storage.

Only live entity rows are retained (float16 rows, uint16 slot indices, int64
unit IDs). Grid channels use affine uint8 quantization with one min/scale pair
per channel and level-1 zlib compression; sparse early-game grids are normally
1--20 KiB. The contract's binary raster is bit-packed to one bit per cell.

With 64 live entities, a conservative 32 KiB compressed-grid estimate is about
93 KiB per raster-96 transition, including both observations but excluding
small Python/mask overhead. The 30k default is therefore roughly 2.7 GiB of
array payload and remains below 6 GiB even near the 200 KiB design ceiling.
Actual bytes depend on grid entropy and live entity count; use
``packed_transition_bytes`` to measure collected data.
"""

from __future__ import annotations

import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from fle.smarq import contract as C


DEFAULT_CAPACITY = 30_000


def _pack_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    grid = np.asarray(observation["grid"], dtype=np.float32)
    flat_grid = grid.reshape(grid.shape[0], -1)
    grid_min = flat_grid.min(axis=1).astype(np.float32)
    grid_max = flat_grid.max(axis=1).astype(np.float32)
    grid_scale = (grid_max - grid_min) / 255.0
    safe_scale = np.where(grid_scale > 0, grid_scale, 1.0)
    quantized_grid = np.rint(
        (grid - grid_min[:, None, None]) / safe_scale[:, None, None]
    ).clip(0, 255).astype(np.uint8)

    entity_mask = np.asarray(observation["entity_mask"], dtype=bool)
    live_slots = np.flatnonzero(entity_mask)
    if entity_mask.size > np.iinfo(np.uint16).max:
        raise ValueError("entity slot index no longer fits uint16")
    entity_view = np.asarray(observation["entity_view"], dtype=np.float32)
    entity_ids = np.asarray(observation["entity_ids"], dtype=np.int64)

    raster = np.asarray(observation["raster"])
    binary_raster = bool(np.all((raster == 0) | (raster == 1)))
    if binary_raster:
        raster_payload: Any = np.packbits(
            raster.astype(bool).reshape(-1), bitorder="little"
        )
    else:
        quantized_raster = np.rint(np.clip(raster, 0, 1) * 255).astype(np.uint8)
        raster_payload = zlib.compress(quantized_raster.tobytes(), level=1)

    packed: dict[str, Any] = {
        "grid": zlib.compress(quantized_grid.tobytes(), level=1),
        "_grid_shape": grid.shape,
        "_grid_min": grid_min,
        "_grid_scale": grid_scale,
        "entity_view": entity_view[live_slots].astype(np.float16),
        "entity_mask": live_slots.astype(np.uint16),
        "entity_ids": entity_ids[live_slots].copy(),
        "_entity_slots": entity_mask.size,
        "globals": np.asarray(observation["globals"], dtype=np.float16),
        "raster": raster_payload,
        "_raster_shape": raster.shape,
        "_raster_binary": binary_raster,
    }
    for key, value in observation.items():
        if key in packed or key in {
            "grid",
            "entity_view",
            "entity_mask",
            "entity_ids",
            "globals",
            "raster",
        }:
            continue
        array = np.asarray(value) if isinstance(value, (np.ndarray, list)) else None
        if array is not None:
            packed[key] = array.copy()
        else:
            packed[key] = value
    return packed


def _unpack_observation(packed: Mapping[str, Any]) -> dict[str, Any]:
    grid_shape = tuple(packed["_grid_shape"])
    quantized_grid = np.frombuffer(
        zlib.decompress(packed["grid"]), dtype=np.uint8
    ).reshape(grid_shape)
    grid_min = np.asarray(packed["_grid_min"], dtype=np.float32)
    grid_scale = np.asarray(packed["_grid_scale"], dtype=np.float32)
    grid = (
        grid_min[:, None, None]
        + quantized_grid.astype(np.float32) * grid_scale[:, None, None]
    )

    entity_slots = int(packed["_entity_slots"])
    live_slots = np.asarray(packed["entity_mask"], dtype=np.uint16).astype(np.int64)
    entity_mask = np.zeros(entity_slots, dtype=bool)
    entity_mask[live_slots] = True
    entity_view = np.zeros((entity_slots, C.ENTITY_FEATURES), dtype=np.float32)
    entity_view[live_slots] = np.asarray(packed["entity_view"], dtype=np.float32)
    entity_ids = np.zeros(entity_slots, dtype=np.int64)
    entity_ids[live_slots] = np.asarray(packed["entity_ids"], dtype=np.int64)

    raster_shape = tuple(packed["_raster_shape"])
    if packed["_raster_binary"]:
        raster_size = int(np.prod(raster_shape))
        raster = np.unpackbits(
            packed["raster"], bitorder="little", count=raster_size
        ).reshape(raster_shape).astype(np.float32)
    else:
        raster = (
            np.frombuffer(zlib.decompress(packed["raster"]), dtype=np.uint8)
            .reshape(raster_shape)
            .astype(np.float32)
            / 255.0
        )

    observation: dict[str, Any] = {
        "grid": grid,
        "entity_view": entity_view,
        "entity_mask": entity_mask,
        "entity_ids": entity_ids,
        "globals": np.asarray(packed["globals"], dtype=np.float32),
        "raster": raster,
    }
    for key, value in packed.items():
        if key.startswith("_") or key in observation:
            continue
        if isinstance(value, np.ndarray):
            observation[key] = value.copy()
        else:
            observation[key] = value
    return observation


def _pack_masks(
    masks: C.Masks | Mapping[str, Any] | None,
    action_heads: np.ndarray | None = None,
    verb: int | None = None,
) -> dict[str, Any] | None:
    if masks is None:
        return None
    names = ("verb", "prototype", "item", "craft_recipe", "technology", "entity")
    result: dict[str, Any] = {}
    entity_array: np.ndarray | None = None
    for name in names:
        value = masks.get(name) if isinstance(masks, Mapping) else getattr(masks, name, None)
        if value is not None:
            array = np.asarray(value, dtype=bool)
            if name == "entity":
                entity_array = array
                result[name] = np.packbits(array.reshape(-1), bitorder="little")
                result["_entity_shape"] = array.shape
            else:
                result[name] = array.copy()
    provider = (
        masks.get("recipe_for_entity")
        if isinstance(masks, Mapping)
        else getattr(masks, "recipe_for_entity", None)
    )
    if provider is not None:
        if callable(provider):
            resolved: dict[int, np.ndarray] = {}
            slots: np.ndarray
            if entity_array is not None:
                slots = np.flatnonzero(entity_array.any(axis=0))
            else:
                slots = np.empty(0, dtype=np.int64)
            # Always include the replayed entity, even if its live mask changed.
            if action_heads is not None and verb is not None and C.ENTITY in C.HEAD_SEQUENCE[C.VERBS[verb]]:
                slot = int(action_heads[C.HEAD_INDEX[C.ENTITY]])
                if slot >= 0:
                    slots = np.unique(np.append(slots, slot))
            for slot in slots:
                value = provider(int(slot))
                if value is not None:
                    resolved[int(slot)] = np.asarray(value, dtype=bool).copy()
            result["recipe_for_entity"] = resolved
        elif isinstance(provider, Mapping):
            result["recipe_for_entity"] = {
                int(key): np.asarray(value, dtype=bool).copy() for key, value in provider.items()
            }
        else:
            result["recipe_for_entity"] = np.asarray(provider, dtype=bool).copy()
    # A pre-resolved mask is also accepted by policy.mask_for_head.
    if isinstance(masks, Mapping) and "recipe" in masks:
        result["recipe"] = np.asarray(masks["recipe"], dtype=bool).copy()
    return result


def _unpack_masks(packed: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if packed is None:
        return None
    result = {key: value for key, value in packed.items() if not key.startswith("_")}
    if "_entity_shape" in packed:
        shape = tuple(packed["_entity_shape"])
        count = int(np.prod(shape))
        result["entity"] = np.unpackbits(
            packed["entity"], bitorder="little", count=count
        ).reshape(shape).astype(bool)
    return result


@dataclass
class _PackedTransition:
    observation: dict[str, Any]
    action_heads: np.ndarray
    verb: int
    reward: float
    tau_seconds: float
    next_observation: dict[str, Any]
    done: bool
    success: bool
    failure_reason: str
    masks: dict[str, Any] | None
    next_masks: dict[str, Any] | None
    is_demo: bool

    @classmethod
    def from_transition(cls, transition: C.Transition) -> "_PackedTransition":
        heads = np.asarray(transition.action_heads, dtype=np.int64).copy()
        return cls(
            _pack_observation(transition.observation),
            heads,
            int(transition.verb),
            float(transition.reward),
            float(transition.tau_seconds),
            _pack_observation(transition.next_observation),
            bool(transition.done),
            bool(transition.success),
            str(transition.failure_reason),
            _pack_masks(transition.masks, heads, int(transition.verb)),
            _pack_masks(transition.next_masks),
            bool(transition.is_demo),
        )

    def unpack(self) -> C.Transition:
        return C.Transition(
            observation=_unpack_observation(self.observation),
            action_heads=self.action_heads.copy(),
            verb=self.verb,
            reward=self.reward,
            tau_seconds=self.tau_seconds,
            next_observation=_unpack_observation(self.next_observation),
            done=self.done,
            success=self.success,
            failure_reason=self.failure_reason,
            masks=_unpack_masks(self.masks),
            next_masks=_unpack_masks(self.next_masks),
            is_demo=self.is_demo,
        )


@dataclass
class ReplayBatch:
    transitions: list[C.Transition]
    indices: np.ndarray
    weights: np.ndarray
    probabilities: np.ndarray
    beta: float

    def __iter__(self) -> Iterator[Any]:
        # Common PER call sites expect exactly these three values.
        yield self.transitions
        yield self.indices
        yield self.weights


class PrioritizedReplay:
    """Array-backed sum tree implementing proportional PER."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 1_000_000,
        priority_epsilon: float = 1e-6,
        demo_bonus: float = 1.0,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = max(1, int(beta_steps))
        self.priority_epsilon = float(priority_epsilon)
        self.demo_bonus = float(demo_bonus)
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self._tree_capacity = tree_capacity
        self._tree = np.zeros(2 * tree_capacity, dtype=np.float64)
        self._storage: list[_PackedTransition | None] = [None] * capacity
        self._next = 0
        self._size = 0
        self._samples = 0
        self._max_priority = 1.0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def total_priority(self) -> float:
        return float(self._tree[1])

    @property
    def beta(self) -> float:
        fraction = min(1.0, self._samples / self.beta_steps)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def _set_leaf(self, index: int, value: float) -> None:
        node = self._tree_capacity + index
        change = value - self._tree[node]
        while node:
            self._tree[node] += change
            node //= 2

    def _scaled_priority(self, priority: float, is_demo: bool) -> float:
        raw = abs(float(priority)) + self.priority_epsilon
        if is_demo:
            raw += self.demo_bonus
        return raw**self.alpha

    def add(self, transition: C.Transition, priority: float | None = None) -> int:
        index = self._next
        packed = _PackedTransition.from_transition(transition)
        self._storage[index] = packed
        raw_priority = self._max_priority if priority is None else abs(float(priority))
        self._max_priority = max(self._max_priority, raw_priority)
        self._set_leaf(index, self._scaled_priority(raw_priority, packed.is_demo))
        self._next = (self._next + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)
        return index

    def _find_prefix(self, mass: float) -> int:
        node = 1
        while node < self._tree_capacity:
            left = node * 2
            if mass < self._tree[left]:
                node = left
            else:
                mass -= self._tree[left]
                node = left + 1
        return node - self._tree_capacity

    def sample(self, batch_size: int, beta: float | None = None) -> ReplayBatch:
        if self._size == 0:
            raise ValueError("cannot sample empty replay")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        use_beta = self.beta if beta is None else float(beta)
        total = self.total_priority
        segment = total / batch_size
        masses = (np.arange(batch_size) + self._rng.random(batch_size)) * segment
        indices = np.asarray([self._find_prefix(float(mass)) for mass in masses], dtype=np.int64)
        leaf = self._tree[self._tree_capacity + indices]
        probabilities = leaf / total
        weights = (self._size * probabilities) ** (-use_beta)
        weights /= weights.max()
        transitions = []
        for index in indices:
            packed = self._storage[int(index)]
            if packed is None:  # pragma: no cover - tree/storage invariant
                raise RuntimeError("sum tree referenced an empty replay slot")
            transitions.append(packed.unpack())
        self._samples += batch_size
        return ReplayBatch(
            transitions,
            indices,
            weights.astype(np.float32),
            probabilities.astype(np.float64),
            use_beta,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")
        for index, priority in zip(indices, priorities, strict=True):
            index = int(index)
            if not 0 <= index < self._size or self._storage[index] is None:
                raise IndexError(index)
            raw = abs(float(priority))
            self._max_priority = max(self._max_priority, raw)
            self._set_leaf(index, self._scaled_priority(raw, self._storage[index].is_demo))

    def save(self, path: str | Path) -> None:
        state = {
            "version": 1,
            "config": {
                "capacity": self.capacity,
                "alpha": self.alpha,
                "beta_start": self.beta_start,
                "beta_steps": self.beta_steps,
                "priority_epsilon": self.priority_epsilon,
                "demo_bonus": self.demo_bonus,
            },
            "tree_capacity": self._tree_capacity,
            "tree": self._tree,
            "storage": self._storage,
            "next": self._next,
            "size": self._size,
            "samples": self._samples,
            "max_priority": self._max_priority,
            "rng_state": self._rng.bit_generator.state,
        }
        with Path(path).open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PrioritizedReplay":
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)  # noqa: S301 - trusted local checkpoint
        if state.get("version") != 1:
            raise ValueError("unsupported replay checkpoint")
        replay = cls(**state["config"])
        replay._tree_capacity = state["tree_capacity"]
        replay._tree = state["tree"]
        replay._storage = state["storage"]
        replay._next = state["next"]
        replay._size = state["size"]
        replay._samples = state["samples"]
        replay._max_priority = state["max_priority"]
        replay._rng.bit_generator.state = state["rng_state"]
        return replay


def observation_payload_bytes(
    raster_tiles: int,
    entity_slots: int = C.ENTITY_SLOTS,
    *,
    live_entities: int = 64,
    compressed_grid_bytes: int = 32 * 1024,
) -> int:
    """Conservative estimate; compressed grid size is data-dependent."""
    del entity_slots  # Empty rows cost nothing in the sparse entity encoding.
    return (
        compressed_grid_bytes
        + C.GRID_CHANNELS * 2 * 4
        + live_entities * (C.ENTITY_FEATURES * 2 + 2 + 8)
        + C.N_GLOBALS * 2
        + (C.RASTER_CHANNELS * raster_tiles * raster_tiles + 7) // 8
    )


def transition_payload_bytes(
    raster_tiles: int,
    entity_slots: int = C.ENTITY_SLOTS,
    *,
    live_entities: int = 64,
    compressed_grid_bytes: int = 32 * 1024,
) -> int:
    """Estimated payload, excluding small metadata and masks."""
    observation = observation_payload_bytes(
        raster_tiles,
        entity_slots,
        live_entities=live_entities,
        compressed_grid_bytes=compressed_grid_bytes,
    )
    return 2 * observation + len(C.HEADS) * 8


def _object_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_object_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_object_nbytes(item) for item in value)
    return 0


def packed_transition_bytes(transition: C.Transition) -> int:
    """Measure compressed arrays/bytes for one transition, including masks."""
    packed = _PackedTransition.from_transition(transition)
    return sum(_object_nbytes(value) for value in vars(packed).values())


PrioritizedReplayBuffer = PrioritizedReplay
