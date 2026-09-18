"""Autoregressive greedy and per-head epsilon-greedy policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fle.smarq import contract as C
from fle.smarq.net import EncodedState, SMARQNetwork


def _field(masks: C.Masks | Mapping[str, Any] | None, name: str, default: Any = None) -> Any:
    if masks is None:
        return default
    if isinstance(masks, Mapping):
        return masks.get(name, default)
    return getattr(masks, name, default)


def mask_for_head(
    masks: C.Masks | Mapping[str, Any] | None,
    head: str,
    verb: int,
    selected: Mapping[str, int],
    size: int,
) -> np.ndarray:
    """Return the structural legality mask for one concrete prefix."""
    if head in (C.POSITION, C.DIRECTION, C.QUANTITY, C.DURATION):
        return np.ones(size, dtype=bool)
    if head == C.PROTOTYPE:
        value = _field(masks, "prototype")
    elif head == C.ITEM:
        provider = _field(masks, "item_for_entity")
        slot = selected.get(C.ENTITY, -1)
        if callable(provider):
            value = provider(slot)
        elif isinstance(provider, Mapping):
            value = provider.get(slot)
        elif provider is not None and np.asarray(provider).ndim == 2 and slot >= 0:
            value = np.asarray(provider)[slot]
        else:
            value = None
        if value is None:
            # Verbs without an entity target retain the global item mask.
            value = _field(masks, "item")
    elif head == C.TECHNOLOGY:
        value = _field(masks, "technology")
    elif head == C.ENTITY:
        table = _field(masks, "entity")
        value = None if table is None else np.asarray(table, dtype=bool)[verb]
    elif head == C.RECIPE:
        if C.VERBS[verb] == "CRAFT":
            value = _field(masks, "craft_recipe")
        else:
            provider = _field(masks, "recipe_for_entity")
            slot = selected.get(C.ENTITY, -1)
            if callable(provider):
                value = provider(slot)
            elif isinstance(provider, Mapping):
                value = provider.get(slot)
            elif provider is not None and np.asarray(provider).ndim == 2 and slot >= 0:
                value = np.asarray(provider)[slot]
            else:
                # Stored transitions may only carry the already-resolved mask.
                value = _field(masks, "recipe")
    else:  # pragma: no cover - guarded by contract HEADS
        raise KeyError(head)
    if value is None:
        return np.ones(size, dtype=bool)
    array = np.asarray(value, dtype=bool).reshape(-1)
    if array.size < size:
        array = np.pad(array, (0, size - array.size), constant_values=False)
    return array[:size]


def masked_argmax(q_values: Tensor, legal: np.ndarray) -> int:
    legal = np.asarray(legal, dtype=bool)
    if legal.size != q_values.numel():
        raise ValueError(f"mask has {legal.size} entries for {q_values.numel()} Q values")
    indices = np.flatnonzero(legal)
    if indices.size == 0:
        raise ValueError("no legal choice for autoregressive head")
    legal_tensor = torch.as_tensor(legal, device=q_values.device)
    return int(q_values.masked_fill(~legal_tensor, -torch.inf).argmax().item())


class AutoregressivePolicy:
    """Construct complete contract actions one legal head at a time."""

    def __init__(
        self,
        network: SMARQNetwork,
        vocab: C.VocabProtocol,
        *,
        seed: int = 0,
    ) -> None:
        self.network = network
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _epsilon(epsilon: float | Mapping[str, float], head: str) -> float:
        if isinstance(epsilon, Mapping):
            return float(epsilon.get(head, epsilon.get("default", 0.0)))
        return float(epsilon)

    def _constructible_verbs(
        self, masks: C.Masks | Mapping[str, Any] | None, entity_slots: int
    ) -> np.ndarray:
        value = _field(masks, "verb")
        legal = (
            np.ones(C.N_VERBS, dtype=bool)
            if value is None
            else np.asarray(value, dtype=bool).copy()
        )
        # A verb whose first structurally constrained head has no choice cannot
        # produce a contract.Action. This matters in an empty fake world.
        for verb, name in enumerate(C.VERBS):
            selected: dict[str, int] = {}
            for head in C.HEAD_SEQUENCE[name]:
                size = entity_slots if head == C.ENTITY else self.network.head_sizes[head]
                choices = mask_for_head(masks, head, verb, selected, size)
                if not choices.any():
                    legal[verb] = False
                    break
                # Later contextual masks cannot generally be known yet.
                if head in (C.ENTITY, C.PROTOTYPE):
                    break
        return legal

    def _choose(self, q_values: Tensor, legal: np.ndarray, epsilon: float) -> int:
        indices = np.flatnonzero(legal)
        if indices.size == 0:
            raise ValueError("no legal action choice")
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(indices))
        return masked_argmax(q_values.reshape(-1), legal)

    def select_indices(
        self,
        encoded: EncodedState,
        masks: C.Masks | Mapping[str, Any] | None,
        epsilon: float | Mapping[str, float] = 0.0,
    ) -> tuple[int, np.ndarray, dict[str, int], list[float]]:
        if encoded.z_state.shape[0] != 1:
            raise ValueError("action selection expects one encoded observation")
        verb_q = self.network.q_verbs(encoded)[0]
        verb_mask = self._constructible_verbs(masks, encoded.entities.shape[1])
        verb = self._choose(verb_q, verb_mask, self._epsilon(epsilon, "verb"))
        heads = C.empty_heads()
        selected: dict[str, int] = {}
        chosen_q = [float(verb_q[verb].item())]
        for head in C.HEAD_SEQUENCE[C.VERBS[verb]]:
            q_values = self.network.q_head(encoded, head, verb, selected)[0]
            size = q_values.numel()
            legal = mask_for_head(masks, head, verb, selected, size)
            choice = self._choose(q_values, legal, self._epsilon(epsilon, head))
            heads[C.HEAD_INDEX[head]] = choice
            selected[head] = choice
            chosen_q.append(float(q_values[choice].item()))
        return verb, heads, selected, chosen_q

    @torch.no_grad()
    def action(
        self,
        observation: C.Observation | Mapping[str, Any],
        masks: C.Masks | Mapping[str, Any] | None,
        epsilon: float | Mapping[str, float] = 0.0,
    ) -> C.Action:
        encoded = self.network.encode(observation)
        verb, heads, selected, _ = self.select_indices(encoded, masks, epsilon)
        obs = observation.as_dict() if isinstance(observation, C.Observation) else observation
        return self._decode(verb, heads, selected, obs)

    __call__ = action

    def greedy(
        self,
        observation: C.Observation | Mapping[str, Any],
        masks: C.Masks | Mapping[str, Any] | None,
    ) -> C.Action:
        return self.action(observation, masks, epsilon=0.0)

    def _decode(
        self,
        verb: int,
        heads: np.ndarray,
        selected: Mapping[str, int],
        observation: Mapping[str, Any],
    ) -> C.Action:
        kwargs: dict[str, Any] = {}
        if C.POSITION in selected:
            if C.VERBS[verb] == "MOVE_TO" and self.network.coarse_move:
                from fle.smarq.actions import move_position_to_tile

                kwargs["tile"] = move_position_to_tile(
                    selected[C.POSITION], tuple(observation["player_tile"])
                )
            else:
                kwargs["tile"] = C.position_to_tile(
                    selected[C.POSITION],
                    tuple(observation["raster_origin"]),
                    self.network.raster_tiles,
                )
        if C.PROTOTYPE in selected:
            kwargs["prototype"] = self.vocab.prototypes[selected[C.PROTOTYPE]]
        if C.DIRECTION in selected:
            kwargs["direction"] = C.DIRECTIONS[selected[C.DIRECTION]]
        if C.ENTITY in selected:
            slot = selected[C.ENTITY]
            kwargs["entity_slot"] = slot
            ids = np.asarray(observation["entity_ids"])
            kwargs["entity_id"] = int(ids[slot]) if slot < len(ids) else 0
        if C.ITEM in selected:
            kwargs["item"] = self.vocab.items[selected[C.ITEM]]
        if C.QUANTITY in selected:
            kwargs["quantity"] = C.QUANTITIES[selected[C.QUANTITY]]
        if C.RECIPE in selected:
            kwargs["recipe"] = self.vocab.recipes[selected[C.RECIPE]]
        if C.TECHNOLOGY in selected:
            kwargs["technology"] = self.vocab.technologies[selected[C.TECHNOLOGY]]
        if C.DURATION in selected:
            kwargs["duration_seconds"] = C.DURATIONS_SECONDS[selected[C.DURATION]]
        return C.Action(C.VERBS[verb], heads.copy(), **kwargs)

    @torch.no_grad()
    def partial_value(
        self,
        observation: C.Observation | Mapping[str, Any],
        masks: C.Masks | Mapping[str, Any] | None,
        *,
        verb: int | str | None = None,
        heads: np.ndarray | None = None,
    ) -> float:
        """Maximum Q over the next decision after a supplied prefix."""
        encoded = self.network.encode(observation)
        if verb is None:
            q_values = self.network.q_verbs(encoded)[0]
            legal = self._constructible_verbs(masks, encoded.entities.shape[1])
            return float(q_values[torch.as_tensor(legal, device=q_values.device)].max().item())
        verb_index = C.VERB_INDEX[verb] if isinstance(verb, str) else int(verb)
        heads = C.empty_heads() if heads is None else np.asarray(heads)
        selected: dict[str, int] = {}
        next_head: str | None = None
        for head in C.HEAD_SEQUENCE[C.VERBS[verb_index]]:
            choice = int(heads[C.HEAD_INDEX[head]])
            if choice < 0:
                next_head = head
                break
            selected[head] = choice
        if next_head is None:
            raise ValueError("partial action is already complete")
        q_values = self.network.q_head(encoded, next_head, verb_index, selected)[0]
        legal = mask_for_head(masks, next_head, verb_index, selected, q_values.numel())
        return float(q_values[torch.as_tensor(legal, device=q_values.device)].max().item())


Policy = AutoregressivePolicy
