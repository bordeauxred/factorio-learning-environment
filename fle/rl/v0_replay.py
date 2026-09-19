"""Memory-conscious ring replay and SMDP n-step returns for V0."""

from __future__ import annotations

import weakref
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from fle.rl.v0_schema import HEAD_SIZES, HEADS, OBS_SPEC, obs_nbytes, smdp_discount


def _validate_observation(obs: Mapping[str, np.ndarray]) -> None:
    missing = set(OBS_SPEC) - set(obs)
    if missing:
        raise KeyError(f"observation missing keys: {sorted(missing)}")
    for key, (shape, dtype) in OBS_SPEC.items():
        value = np.asarray(obs[key])
        if value.shape != shape:
            raise ValueError(f"{key} shape {value.shape} != {shape}")
        if value.dtype != dtype:
            raise TypeError(f"{key} dtype {value.dtype} != {dtype}")


def _action_array(action: Mapping[str, int] | np.ndarray) -> np.ndarray:
    if isinstance(action, Mapping):
        result = np.zeros(len(HEADS), dtype=np.int16)
        for head, value in action.items():
            result[HEADS.index(head)] = int(value)
        return result
    result = np.asarray(action)
    if result.shape != (len(HEADS),):
        raise ValueError(f"action shape {result.shape} != ({len(HEADS)},)")
    return result.astype(np.int16, copy=False)


def full_masks() -> dict[str, np.ndarray]:
    """Return an unrestricted, batchless mask dictionary."""
    return {head: np.ones(size, dtype=np.uint8) for head, size in HEAD_SIZES.items()}


@dataclass(frozen=True)
class V0Transition:
    obs: Mapping[str, np.ndarray]
    action: Mapping[str, int] | np.ndarray
    reward: float
    next_obs: Mapping[str, np.ndarray]
    duration: float
    done: bool
    masks: Mapping[str, np.ndarray] | None = None
    next_masks: Mapping[str, np.ndarray] | None = None
    discount: float | None = None


@dataclass(frozen=True)
class V0Batch:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: dict[str, torch.Tensor]
    durations: torch.Tensor
    discounts: torch.Tensor
    masks: dict[str, torch.Tensor]
    next_masks: dict[str, torch.Tensor]
    indices: np.ndarray
    action_rows: tuple[torch.Tensor | None, ...] | None = None


@dataclass(frozen=True)
class _Step:
    obs: Mapping[str, np.ndarray]
    action: Mapping[str, int] | np.ndarray
    reward: float
    next_obs: Mapping[str, np.ndarray]
    duration: float
    done: bool
    masks: Mapping[str, np.ndarray] | None
    next_masks: Mapping[str, np.ndarray] | None


class V0NStepAccumulator:
    """Compose rewards and discounts in simulated time, not decision count."""

    def __init__(self, n: int = 3, *, use_smdp: bool = True, gamma: float = 0.99):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.use_smdp = use_smdp
        self.gamma = gamma
        self.pending: deque[_Step] = deque()

    def _step_discount(self, step: _Step) -> float:
        if self.use_smdp:
            return float(smdp_discount(step.duration))
        return self.gamma

    def _emit(self, length: int, *, boundary: bool = False) -> V0Transition:
        steps = list(self.pending)[:length]
        reward = 0.0
        cumulative_discount = 1.0
        duration = 0.0
        for step in steps:
            reward += cumulative_discount * step.reward
            duration += step.duration
            cumulative_discount *= self._step_discount(step)
        last = steps[-1]
        discount = 0.0 if last.done or boundary else cumulative_discount
        first = self.pending.popleft()
        return V0Transition(
            obs=first.obs,
            action=first.action,
            reward=reward,
            next_obs=last.next_obs,
            duration=duration,
            done=last.done or boundary,
            masks=first.masks,
            next_masks=last.next_masks,
            discount=discount,
        )

    def append(
        self,
        obs: Mapping[str, np.ndarray],
        action: Mapping[str, int] | np.ndarray,
        reward: float,
        next_obs: Mapping[str, np.ndarray],
        duration: float,
        done: bool,
        masks: Mapping[str, np.ndarray] | None = None,
        next_masks: Mapping[str, np.ndarray] | None = None,
    ) -> list[V0Transition]:
        if duration < 0:
            raise ValueError("duration cannot be negative")
        self.pending.append(
            _Step(
                obs,
                action,
                float(reward),
                next_obs,
                float(duration),
                bool(done),
                masks,
                next_masks,
            )
        )
        emitted = []
        if done:
            while self.pending:
                emitted.append(self._emit(min(self.n, len(self.pending))))
        elif len(self.pending) >= self.n:
            emitted.append(self._emit(self.n))
        return emitted

    def flush(self) -> list[V0Transition]:
        """Flush an excursion boundary without bootstrapping across it."""
        emitted = []
        while self.pending:
            emitted.append(self._emit(min(self.n, len(self.pending)), boundary=True))
        return emitted


class _ChunkedArray:
    """A lazily allocated first dimension, avoiding a 13 GiB constructor."""

    def __init__(
        self,
        length: int,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunk_size: int,
    ) -> None:
        self.length = length
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.chunk_size = chunk_size
        self.chunks: dict[int, np.ndarray] = {}

    def _chunk(self, index: int) -> tuple[np.ndarray, int]:
        chunk_index, offset = divmod(int(index), self.chunk_size)
        chunk = self.chunks.get(chunk_index)
        if chunk is None:
            count = min(self.chunk_size, self.length - chunk_index * self.chunk_size)
            chunk = np.empty((count, *self.shape), dtype=self.dtype)
            self.chunks[chunk_index] = chunk
        return chunk, offset

    def set(self, index: int, value: np.ndarray) -> None:
        array = np.asarray(value)
        if array.shape != self.shape or array.dtype != self.dtype:
            raise TypeError(
                f"stored value shape/dtype {array.shape}/{array.dtype} != "
                f"{self.shape}/{self.dtype}"
            )
        chunk, offset = self._chunk(index)
        np.copyto(chunk[offset], array, casting="no")

    def get(self, index: int) -> np.ndarray:
        chunk, offset = self._chunk(index)
        return chunk[offset].copy()

    def gather(self, indices: np.ndarray) -> np.ndarray:
        output = np.empty((len(indices), *self.shape), dtype=self.dtype)
        self.gather_into(indices, output)
        return output

    def gather_into(self, indices: np.ndarray, output: np.ndarray) -> None:
        for row, index in enumerate(indices):
            chunk, offset = self._chunk(int(index))
            np.copyto(output[row], chunk[offset], casting="unsafe")

    @property
    def allocated_nbytes(self) -> int:
        return sum(chunk.nbytes for chunk in self.chunks.values())


class V0ReplayBuffer:
    """State-interning ring replay with exact observation storage dtypes.

    Transition metadata points into a reference-counted state pool. Identity
    interning reuses an n-step successor when it later becomes another
    transition's current state. The pool is lazy; two slots per transition also
    support unrelated insertions without duplicating any individual state.
    """

    def __init__(self, capacity: int = 4096, *, chunk_size: int = 4) -> None:
        if capacity <= 0 or chunk_size <= 0:
            raise ValueError("capacity and chunk_size must be positive")
        self.capacity = capacity
        self.state_slots = capacity * 2
        self.chunk_size = chunk_size
        self.states = {
            key: _ChunkedArray(self.state_slots, shape, dtype, chunk_size)
            for key, (shape, dtype) in OBS_SPEC.items()
        }
        self.actions = np.empty((capacity, len(HEADS)), dtype=np.int16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.durations = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.obs_indices = np.empty(capacity, dtype=np.int32)
        self.next_indices = np.empty(capacity, dtype=np.int32)
        self.valid = np.zeros(capacity, dtype=bool)
        self.mask_storage = {
            head: np.ones((self.state_slots, size), dtype=np.uint8)
            for head, size in HEAD_SIZES.items()
        }
        self._state_references = np.zeros(self.state_slots, dtype=np.int32)
        self._free_states = list(reversed(range(self.state_slots)))
        self._identity_slots: dict[
            int, tuple[int, weakref.ReferenceType[np.ndarray]]
        ] = {}
        self._slot_identities = np.full(self.state_slots, -1, dtype=np.int64)
        self.position = 0
        self.size = 0
        self._build_buffers: dict[str, np.ndarray] = {}

    @property
    def bytes_per_transition(self) -> int:
        """Observation payload bytes: one schema state per transition."""
        return obs_nbytes()

    @property
    def nbytes(self) -> int:
        """Configured observation-payload budget (metadata excluded)."""
        return self.capacity * self.bytes_per_transition

    @property
    def allocated_nbytes(self) -> int:
        state_bytes = sum(value.allocated_nbytes for value in self.states.values())
        metadata = (
            self.actions.nbytes
            + self.rewards.nbytes
            + self.durations.nbytes
            + self.discounts.nbytes
            + self.obs_indices.nbytes
            + self.next_indices.nbytes
            + self.valid.nbytes
            + self._state_references.nbytes
            + self._slot_identities.nbytes
            + sum(value.nbytes for value in self.mask_storage.values())
            + sum(value.nbytes for value in self._build_buffers.values())
        )
        return state_bytes + metadata

    def _write_state(self, index: int, obs: Mapping[str, np.ndarray]) -> None:
        _validate_observation(obs)
        for key in OBS_SPEC:
            self.states[key].set(index, obs[key])

    def _write_masks(self, index: int, masks: Mapping[str, np.ndarray] | None) -> None:
        values = full_masks() if masks is None else masks
        for head, size in HEAD_SIZES.items():
            value = np.asarray(values.get(head, np.ones(size)), dtype=np.uint8)
            if value.shape != (size,):
                raise ValueError(f"mask {head} shape {value.shape} != ({size},)")
            self.mask_storage[head][index] = value

    def _acquire_state(
        self,
        obs: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray] | None,
    ) -> int:
        identity_object = np.asarray(obs["local_exact"])
        identity = id(identity_object)
        existing = self._identity_slots.get(identity)
        if existing is not None:
            slot, reference = existing
            if reference() is identity_object and self._state_references[slot] > 0:
                self._state_references[slot] += 1
                return slot
        if not self._free_states:
            raise RuntimeError("replay state pool exhausted")
        slot = self._free_states.pop()
        self._write_state(slot, obs)
        self._write_masks(slot, masks)
        self._state_references[slot] = 1
        self._slot_identities[slot] = identity
        self._identity_slots[identity] = (slot, weakref.ref(identity_object))
        return slot

    def _release_state(self, slot: int) -> None:
        self._state_references[slot] -= 1
        if self._state_references[slot] > 0:
            return
        identity = int(self._slot_identities[slot])
        existing = self._identity_slots.get(identity)
        if existing is not None and existing[0] == slot:
            del self._identity_slots[identity]
        self._slot_identities[slot] = -1
        self._free_states.append(slot)

    def add(self, transition: V0Transition) -> int:
        index = self.position
        if self.valid[index]:
            self._release_state(int(self.obs_indices[index]))
            self._release_state(int(self.next_indices[index]))
        obs_index = self._acquire_state(transition.obs, transition.masks)
        next_index = self._acquire_state(transition.next_obs, transition.next_masks)
        self.obs_indices[index] = obs_index
        self.next_indices[index] = next_index
        self.actions[index] = _action_array(transition.action)
        self.rewards[index] = transition.reward
        self.durations[index] = transition.duration
        self.discounts[index] = (
            float(transition.discount)
            if transition.discount is not None
            else (0.0 if transition.done else float(smdp_discount(transition.duration)))
        )
        self.valid[index] = True
        self.size = min(self.size + 1, self.capacity)
        self.position = (index + 1) % self.capacity
        return index

    def get_state(self, index: int) -> dict[str, np.ndarray]:
        if not self.valid[index]:
            raise IndexError(f"transition slot {index} is not valid")
        state_index = int(self.obs_indices[index])
        return {key: value.get(state_index) for key, value in self.states.items()}

    def _cast_buildability(self, label: str, indices: np.ndarray) -> np.ndarray:
        shape = (len(indices), *OBS_SPEC["buildability"][0])
        buffer = self._build_buffers.get(label)
        if buffer is None or buffer.shape[0] < len(indices):
            buffer = np.empty(shape, dtype=np.float32)
            self._build_buffers[label] = buffer
        view = buffer[: len(indices)]
        self.states["buildability"].gather_into(indices, view)
        return view

    def _batch_states(
        self, indices: np.ndarray, device: torch.device, label: str
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key in OBS_SPEC:
            if key == "buildability":
                array = self._cast_buildability(label, indices)
            else:
                array = self.states[key].gather(indices)
            # PyTorch/MPS does not consistently support uint16 tensors. The
            # replay dtype remains uint16; conversion happens only in batching.
            if key == "build_age":
                array = array.astype(np.float32)
            result[key] = torch.as_tensor(array, device=device)
        return result

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> V0Batch:
        available = np.flatnonzero(self.valid)
        if not len(available):
            raise ValueError("cannot sample empty replay")
        selected = rng.integers(len(available), size=batch_size)
        indices = available[selected]
        obs_indices = self.obs_indices[indices]
        next_indices = self.next_indices[indices]
        masks = {
            head: torch.as_tensor(values[obs_indices].astype(bool), device=device)
            for head, values in self.mask_storage.items()
        }
        next_masks = {
            head: torch.as_tensor(values[next_indices].astype(bool), device=device)
            for head, values in self.mask_storage.items()
        }
        host_actions = self.actions[indices].astype(np.int64)
        verb_column = HEADS.index("verb")
        action_rows = tuple(
            (
                torch.as_tensor(rows, dtype=torch.long, device=device)
                if len(rows)
                else None
            )
            for verb in range(HEAD_SIZES["verb"])
            for rows in [np.flatnonzero(host_actions[:, verb_column] == verb)]
        )
        return V0Batch(
            obs=self._batch_states(obs_indices, device, "obs"),
            actions=torch.as_tensor(host_actions, device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=self._batch_states(next_indices, device, "next_obs"),
            durations=torch.as_tensor(self.durations[indices], device=device),
            discounts=torch.as_tensor(self.discounts[indices], device=device),
            masks=masks,
            next_masks=next_masks,
            indices=indices,
            action_rows=action_rows,
        )
