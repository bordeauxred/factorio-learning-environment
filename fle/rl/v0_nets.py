"""Structured encoders and masked autoregressive Double-DQN utilities for V0."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fle.rl.dqn import masked_argmax
from fle.rl.nets import MaskedRowEncoder
from fle.rl.v0_schema import HEAD_SIZES, HEADS, VERB_HEADS, VERBS

MaskValue = torch.Tensor | np.ndarray
MaskResolver = Callable[[str, Mapping[str, torch.Tensor]], MaskValue]
MaskSource = Mapping[str, MaskValue | MaskResolver] | MaskResolver


@dataclass(frozen=True)
class EncodedV0:
    state: torch.Tensor
    local_spatial: torch.Tensor
    build_spatial: torch.Tensor


@dataclass(frozen=True)
class DecodedActions:
    """A complete grammar action and its online Q value."""

    actions: torch.Tensor
    q_values: torch.Tensor


class V0Encoder(nn.Module):
    """One tower per V0 observation family, fused to a 384-wide state."""

    output_size = 384

    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(17, 24, 3, padding=1),
            nn.ReLU(),
        )
        self.local_pool = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 96),
            nn.ReLU(),
        )
        self.minimap = nn.Sequential(
            nn.Conv2d(16, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 96),
            nn.ReLU(),
        )
        # The first/third convolutions preserve 128/64 resolution; the middle
        # convolution is the contract's stride-2 stage.
        self.buildability = nn.Sequential(
            nn.Conv2d(63, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.build_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 96),
            nn.ReLU(),
        )
        self.build_age = nn.Sequential(
            nn.Conv2d(63, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, 48),
            nn.ReLU(),
        )
        self.entities = MaskedRowEncoder(24, 64)
        self.dense = nn.Sequential(
            nn.Linear(251 + 196 + 217 + 23, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(96 + 96 + 96 + 48 + 128 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
            nn.ReLU(),
        )

    def forward(self, obs: Mapping[str, torch.Tensor]) -> EncodedV0:
        local_spatial = self.local(obs["local_exact"].float())
        local = self.local_pool(local_spatial)
        minimap = self.minimap(obs["minimap"].float())
        build_spatial = self.buildability(obs["buildability"].float())
        build = self.build_pool(build_spatial)
        # uint16 is divided before entering the age tower so never-sampled has
        # a stable value of one and freshness remains observable.
        age = self.build_age(obs["build_age"].float() / 65535.0)
        entities = self.entities(obs["entities"].float(), obs["entity_mask"] > 0.5)
        dense = self.dense(
            torch.cat(
                (
                    obs["inventory"].float(),
                    obs["research"].float(),
                    obs["recipes_enabled"].float(),
                    obs["globals"].float(),
                ),
                dim=-1,
            )
        )
        state = self.fusion(
            torch.cat((local, minimap, build, age, entities, dense), dim=-1)
        )
        return EncodedV0(state, local_spatial, build_spatial)


class V0DQN(nn.Module):
    """Dueling autoregressive Q network with a convolutional location head."""

    prefix_size = 64

    def __init__(self) -> None:
        super().__init__()
        self.encoder = V0Encoder()
        self.value = nn.Linear(self.encoder.output_size, 1)
        self.verb = nn.Linear(self.encoder.output_size, HEAD_SIZES["verb"])
        self.prefix_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, self.prefix_size)
                for name, size in HEAD_SIZES.items()
            }
        )
        input_size = self.encoder.output_size + self.prefix_size
        self.argument_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_size, 192),
                    nn.ReLU(),
                    nn.Linear(192, size),
                )
                for name, size in HEAD_SIZES.items()
                if name not in {"verb", "location"}
            }
        )
        self.location_condition = nn.Linear(input_size, 24)
        self.location_decoder = nn.Sequential(
            nn.Conv2d(24 + 32, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 1, 1),
        )

    def forward(self, obs: Mapping[str, torch.Tensor]) -> EncodedV0:
        return self.encoder(obs)

    def initial_prefix(self, verb: torch.Tensor) -> torch.Tensor:
        return self.prefix_embeddings["verb"](verb)

    def extend_prefix(
        self, prefix: torch.Tensor, head: str, choice: torch.Tensor
    ) -> torch.Tensor:
        return prefix + self.prefix_embeddings[head](choice)

    def scores(
        self, encoded: EncodedV0, head: str, prefix: torch.Tensor
    ) -> torch.Tensor:
        conditioned = torch.cat((encoded.state, prefix), dim=-1)
        if head != "location":
            return self.argument_heads[head](conditioned)
        condition = self.location_condition(conditioned).unsqueeze(-1).unsqueeze(-1)
        local = encoded.local_spatial + condition
        build = encoded.build_spatial
        if build.shape[-2:] != local.shape[-2:]:
            build = F.interpolate(build, size=local.shape[-2:], mode="bilinear")
        return self.location_decoder(torch.cat((local, build), dim=1)).flatten(1)

    def parameter_counts(self) -> dict[str, int]:
        groups: dict[str, nn.Module] = {
            "local": nn.ModuleList([self.encoder.local, self.encoder.local_pool]),
            "minimap": self.encoder.minimap,
            "buildability": nn.ModuleList(
                [self.encoder.buildability, self.encoder.build_pool]
            ),
            "build_age": self.encoder.build_age,
            "entities": self.encoder.entities,
            "dense": self.encoder.dense,
            "fusion": self.encoder.fusion,
            "autoregressive_heads": nn.ModuleList(
                [
                    self.value,
                    self.verb,
                    self.prefix_embeddings,
                    self.argument_heads,
                    self.location_condition,
                    self.location_decoder,
                ]
            ),
        }
        return {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }


def _resolve_mask(
    masks: MaskSource,
    head: str,
    prefix: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if callable(masks):
        value = masks(head, prefix)
    else:
        value: Any = masks.get(head)
        if callable(value):
            value = value(head, prefix)
        if value is None:
            value = torch.ones((batch_size, HEAD_SIZES[head]), dtype=torch.bool)
    result = torch.as_tensor(value, dtype=torch.bool, device=device)
    if result.ndim == 1:
        result = result.unsqueeze(0).expand(batch_size, -1)
    if result.shape != (batch_size, HEAD_SIZES[head]):
        raise ValueError(
            f"mask {head!r} shape {tuple(result.shape)} != "
            f"({batch_size}, {HEAD_SIZES[head]})"
        )
    if not callable(masks) and "verb" in prefix:
        for verb_index, verb_name in enumerate(VERBS):
            keyed = masks.get(f"{verb_name}:{head}")
            if keyed is None:
                continue
            if callable(keyed):
                keyed = keyed(head, prefix)
            keyed_mask = torch.as_tensor(keyed, dtype=torch.bool, device=device)
            if keyed_mask.ndim == 1:
                keyed_mask = keyed_mask.unsqueeze(0).expand(batch_size, -1)
            if keyed_mask.shape != result.shape:
                raise ValueError(
                    f"mask {verb_name}:{head} shape {tuple(keyed_mask.shape)} "
                    f"!= {tuple(result.shape)}"
                )
            use_keyed = prefix["verb"] == verb_index
            result = torch.where(use_keyed[:, None], keyed_mask, result)
    return result


def _choose(
    scores: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    greedy = masked_argmax(scores, mask, empty_value=-1)
    if epsilon <= 0:
        return greedy
    explore = (
        torch.rand(scores.shape[0], device=scores.device, generator=generator) < epsilon
    )
    random_scores = torch.rand(scores.shape, device=scores.device, generator=generator)
    random_choice = masked_argmax(random_scores, mask, empty_value=-1)
    return torch.where(explore, random_choice, greedy)


def _stage_groups() -> tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]:
    """Group verbs which score the same head at each prefix depth."""
    depth_count = max(len(heads) for heads in VERB_HEADS.values())
    stages = []
    for depth in range(depth_count):
        groups = []
        for head in HEADS:
            if head == "verb":
                continue
            verbs = tuple(
                verb_index
                for verb_index, verb_name in enumerate(VERBS)
                if len(VERB_HEADS[verb_name]) > depth
                and VERB_HEADS[verb_name][depth] == head
            )
            if verbs:
                groups.append((head, verbs))
        stages.append(tuple(groups))
    return tuple(stages)


_DECODE_STAGES = _stage_groups()


def _expand_encoded(encoded: EncodedV0, count: int, *, spatial: bool) -> EncodedV0:
    batch = encoded.state.shape[0]

    def expand(value: torch.Tensor) -> torch.Tensor:
        return (
            value[:, None]
            .expand(batch, count, *value.shape[1:])
            .reshape(batch * count, *value.shape[1:])
        )

    return EncodedV0(
        expand(encoded.state),
        expand(encoded.local_spatial) if spatial else encoded.local_spatial,
        expand(encoded.build_spatial) if spatial else encoded.build_spatial,
    )


def _group_prefix(
    actions: torch.Tensor, verb_indices: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    return {
        head: actions[:, verb_indices, column].reshape(-1)
        for column, head in enumerate(HEADS)
    }


def _group_masks(
    masks: MaskSource,
    head: str,
    prefix: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    group_size: int,
) -> torch.Tensor:
    """Expand a state mask across a fixed verb group without scalar reads."""
    flat_batch = batch_size * group_size
    if callable(masks):
        value: Any = masks(head, prefix)
    else:
        value = masks.get(head)
        if callable(value):
            value = value(head, prefix)
        if value is None:
            value = torch.ones((flat_batch, HEAD_SIZES[head]), dtype=torch.bool)

    def expand(value: MaskValue) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=torch.bool, device=device)
        if result.ndim == 1:
            return result.unsqueeze(0).expand(flat_batch, -1)
        if result.shape[0] == batch_size:
            return (
                result[:, None]
                .expand(batch_size, group_size, result.shape[1])
                .reshape(flat_batch, result.shape[1])
            )
        if result.shape[0] != flat_batch:
            raise ValueError(
                f"mask {head!r} batch {result.shape[0]} not in "
                f"{{{batch_size}, {flat_batch}}}"
            )
        return result

    result = expand(value)
    if result.shape[1] != HEAD_SIZES[head]:
        raise ValueError(f"mask {head!r} width {result.shape[1]} != {HEAD_SIZES[head]}")
    if not callable(masks):
        for verb_index, verb_name in enumerate(VERBS):
            keyed = masks.get(f"{verb_name}:{head}")
            if keyed is None:
                continue
            if callable(keyed):
                keyed = keyed(head, prefix)
            keyed_mask = expand(keyed)
            result = torch.where(
                (prefix["verb"] == verb_index)[:, None], keyed_mask, result
            )
    return result


def _score_group(
    network: V0DQN,
    encoded: EncodedV0,
    head: str,
    verb_indices: tuple[int, ...],
    prefixes: torch.Tensor,
) -> torch.Tensor:
    batch = encoded.state.shape[0]
    count = len(verb_indices)
    group_encoded = _expand_encoded(encoded, count, spatial=head == "location")
    scores = network.scores(
        group_encoded, head, prefixes[:, verb_indices].reshape(batch * count, -1)
    )
    return scores.reshape(batch, count, HEAD_SIZES[head])


def _initial_candidates(
    network: V0DQN, encoded: EncodedV0
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = encoded.state.shape[0]
    device = encoded.state.device
    verbs = torch.arange(len(VERBS), device=device)
    actions = torch.zeros(
        (batch, len(VERBS), len(HEADS)), dtype=torch.long, device=device
    )
    actions[:, :, HEADS.index("verb")] = verbs
    prefixes = network.initial_prefix(verbs).unsqueeze(0).expand(batch, -1, -1)
    return actions, prefixes.clone()


def _decode_candidates(
    network: V0DQN,
    encoded: EncodedV0,
    masks: MaskSource,
    epsilon: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode every verb in fixed, head-batched autoregressive stages."""
    batch = encoded.state.shape[0]
    device = encoded.state.device
    actions, prefixes = _initial_candidates(network, encoded)
    advantages = torch.zeros((batch, len(VERBS)), device=device)
    viable = torch.ones((batch, len(VERBS)), dtype=torch.bool, device=device)
    for stage in _DECODE_STAGES:
        for head, verb_indices in stage:
            count = len(verb_indices)
            scores = _score_group(network, encoded, head, verb_indices, prefixes)
            flat_prefix = _group_prefix(actions, verb_indices)
            mask = _group_masks(
                masks,
                head,
                flat_prefix,
                device=device,
                batch_size=batch,
                group_size=count,
            ).reshape(batch, count, -1)
            choice = _choose(
                scores.reshape(batch * count, -1),
                mask.reshape(batch * count, -1),
                epsilon,
                generator,
            ).reshape(batch, count)
            viable[:, verb_indices] &= choice >= 0
            safe_choice = choice.clamp_min(0)
            actions[:, verb_indices, HEADS.index(head)] = safe_choice
            advantages[:, verb_indices] += scores.gather(
                2, safe_choice.unsqueeze(-1)
            ).squeeze(-1)
            prefixes[:, verb_indices] += network.prefix_embeddings[head](safe_choice)
    return actions, advantages, viable


@torch.no_grad()
def decode_actions(
    network: V0DQN,
    obs: Mapping[str, torch.Tensor],
    masks: MaskSource,
    *,
    epsilon: float = 0.0,
    generator: torch.Generator | None = None,
) -> DecodedActions:
    """The single sequential decoder used by acting and Double-DQN selection."""
    encoded = network(obs)
    batch = encoded.state.shape[0]
    device = encoded.state.device
    empty_prefix: dict[str, torch.Tensor] = {}
    verb_mask = _resolve_mask(
        masks, "verb", empty_prefix, device=device, batch_size=batch
    ).clone()

    candidate_actions, candidate_arguments, viable = _decode_candidates(
        network, encoded, masks, 0.0, generator
    )
    verb_mask &= viable

    verb_scores = network.verb(encoded.state)
    chosen_verb = _choose(verb_scores, verb_mask, epsilon, generator)
    if bool((chosen_verb < 0).any()):
        rows = (chosen_verb < 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"no viable verb for batch rows: {rows}")

    if epsilon > 0:
        explored_actions, explored_arguments, explored_viable = _decode_candidates(
            network, encoded, masks, epsilon, generator
        )
        candidate_actions = torch.where(
            explored_viable[:, :, None], explored_actions, candidate_actions
        )
        candidate_arguments = torch.where(
            explored_viable, explored_arguments, candidate_arguments
        )

    rows = torch.arange(batch, device=device)
    actions = candidate_actions[rows, chosen_verb]
    argument_q = candidate_arguments[rows, chosen_verb]
    q_values = (
        network.value(encoded.state).squeeze(-1)
        + verb_scores[rows, chosen_verb]
        + argument_q
    )
    return DecodedActions(actions, q_values)


def _argument_q_for_known_actions(
    network: V0DQN, encoded: EncodedV0, actions: torch.Tensor
) -> torch.Tensor:
    """Score known tuples without host reads or device-side dynamic branches."""
    batch = actions.shape[0]
    verbs = actions[:, HEADS.index("verb")]
    device = encoded.state.device
    advantages = torch.zeros(batch, dtype=encoded.state.dtype, device=device)
    location_prefix = torch.zeros(
        (batch, network.prefix_size), dtype=encoded.state.dtype, device=device
    )
    location_active = torch.zeros(batch, dtype=torch.bool, device=device)
    for verb_index, verb_name in enumerate(VERBS):
        active = verbs == verb_index
        verb = torch.full((batch,), verb_index, dtype=torch.long, device=device)
        prefix = network.initial_prefix(verb)
        for head in VERB_HEADS[verb_name]:
            choice = actions[:, HEADS.index(head)]
            if head == "location":
                location_prefix = torch.where(active[:, None], prefix, location_prefix)
                location_active |= active
            else:
                scores = network.scores(encoded, head, prefix)
                selected = scores.gather(1, choice[:, None]).squeeze(1)
                advantages += torch.where(active, selected, 0.0)
            prefix = network.extend_prefix(prefix, head, choice)
    location_scores = network.scores(encoded, "location", location_prefix)
    location_choice = actions[:, HEADS.index("location")]
    selected_location = location_scores.gather(1, location_choice[:, None]).squeeze(1)
    advantages += torch.where(location_active, selected_location, 0.0)
    return advantages


def _argument_q_for_grouped_actions(
    network: V0DQN,
    encoded: EncodedV0,
    actions: torch.Tensor,
    action_rows: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    """Score replay actions using row groups assembled before device transfer."""
    advantages = torch.zeros(
        actions.shape[0], dtype=encoded.state.dtype, device=encoded.state.device
    )
    for verb_index, (verb_name, rows) in enumerate(zip(VERBS, action_rows)):
        if rows is None:
            continue
        subset = EncodedV0(
            encoded.state[rows],
            encoded.local_spatial[rows],
            encoded.build_spatial[rows],
        )
        verb = torch.full(
            (rows.shape[0],),
            verb_index,
            dtype=torch.long,
            device=encoded.state.device,
        )
        prefix = network.initial_prefix(verb)
        verb_advantage = torch.zeros(
            rows.shape[0], dtype=encoded.state.dtype, device=encoded.state.device
        )
        for head in VERB_HEADS[verb_name]:
            choice = actions[rows, HEADS.index(head)]
            scores = network.scores(subset, head, prefix)
            verb_advantage += scores.gather(1, choice[:, None]).squeeze(1)
            prefix = network.extend_prefix(prefix, head, choice)
        advantages = advantages.index_add(0, rows, verb_advantage)
    return advantages


def action_q(
    network: V0DQN,
    obs: Mapping[str, torch.Tensor],
    actions: torch.Tensor,
    action_rows: tuple[torch.Tensor | None, ...] | None = None,
) -> torch.Tensor:
    """Evaluate V + A_verb + SUM of prefix-conditioned argument advantages."""
    encoded = network(obs)
    batch = actions.shape[0]
    device = encoded.state.device
    rows = torch.arange(batch, device=device)
    verbs = actions[:, HEADS.index("verb")]
    result = network.value(encoded.state).squeeze(-1)
    result = result + network.verb(encoded.state)[rows, verbs]
    if action_rows is None:
        argument_q = _argument_q_for_known_actions(network, encoded, actions)
    else:
        argument_q = _argument_q_for_grouped_actions(
            network, encoded, actions, action_rows
        )
    return result + argument_q


def behaviour_actions(
    network: V0DQN,
    obs: Mapping[str, torch.Tensor],
    masks: MaskSource,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> DecodedActions:
    """Behaviour wrapper intentionally delegates to the shared decoder."""
    return decode_actions(network, obs, masks, epsilon=epsilon, generator=generator)


def double_dqn_loss(
    online: V0DQN,
    target: V0DQN,
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked autoregressive Double-DQN Huber loss."""
    predicted = action_q(
        online,
        batch.obs,
        batch.actions,
        action_rows=getattr(batch, "action_rows", None),
    )
    with torch.no_grad():
        # This is deliberately the exact acting decoder, with epsilon disabled.
        selected = decode_actions(online, batch.next_obs, batch.next_masks)
        next_q = action_q(target, batch.next_obs, selected.actions)
        td_target = batch.rewards + batch.discounts * next_q
    loss = F.smooth_l1_loss(predicted, td_target)
    return loss, (td_target - predicted.detach()).abs().mean()
