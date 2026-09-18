"""Branching Double DQN for the learner-facing FLE macro environment.

This is deliberately a single-file trainer in the CleanRL style.  It keeps a
local encoder until the PPO and DQN implementations can share one without
coupling their development environments.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fle.rl import schema as S
from fle.rl.fake_env import FakeMacroEnv

MASKED_VALUE = -1e8
ENV_TIMEOUT_SECONDS = 90
MAX_ARCHIVE_EPISODES_PER_RUNG = 16
MAX_ARCHIVE_TOP_EPISODES = 32
RUNG_NAMES = (
    "furnace_fuelled_and_fed",
    "plates_extracted",
    "gears_crafted",
    "drill_crafted",
    "drill_placed",
    "drill_fuelled",
)
PRIMARY_HEADS = {
    "CRAFT": "recipe",
    "PLACE": "placeable",
    "INSERT": "item",
    "EXTRACT": "item",
    "SET_RECIPE": "recipe",
    "RESEARCH": "technology",
    "WAIT": "duration",
    "MOVE": "move_dir",
    "CONNECT": "connector",
}
FURNACE_INPUTS = frozenset({"iron-ore", "copper-ore", "stone"})
_VOCAB_DATA = json.loads(S.VOCAB_PATH.read_text())
FUEL_ITEMS = frozenset(
    str(row["name"])
    for row in _VOCAB_DATA["items"]
    if float(row.get("fuel_value") or 0) > 0
)
_ENTITY_TYPES = {
    str(row["name"]): str(row.get("type") or "") for row in _VOCAB_DATA["entities"]
}
RESOURCE_REQUIRING_PLACEABLES = frozenset(
    str(row["name"])
    for row in _VOCAB_DATA["items"]
    if _ENTITY_TYPES.get(str(row.get("place_result"))) == "mining-drill"
)
DRILL_RECIPE = "burner-mining-drill"
DRILL_CRAFTED_RUNG = RUNG_NAMES.index("drill_crafted")
_MISSING_PRIMARY_WARNED: set[str] = set()


def observation_masks(obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Recover boolean action masks from a float observation batch."""
    return {head: obs[..., a:b] > 0.5 for head, (a, b) in S.MASK_OFFSETS.items()}


def masked_argmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax over admitted indices, returning zero for an empty mask."""
    return values.masked_fill(~mask.bool(), MASKED_VALUE).argmax(dim=-1)


class RowEncoder(nn.Module):
    """Encode variable-cardinality rows with masked mean/max pooling."""

    def __init__(self, features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        valid = rows[..., -2] > 0.5
        encoded = self.mlp(rows)
        weights = valid.unsqueeze(-1)
        mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        maximum = encoded.masked_fill(~weights, MASKED_VALUE).amax(dim=1)
        maximum = torch.where(valid.any(dim=1, keepdim=True), maximum, 0.0)
        return torch.cat((mean, maximum), dim=-1)


class ObservationEncoder(nn.Module):
    """Block-aware encoder matching the macro observation schema."""

    def __init__(self):
        super().__init__()
        dense_size = sum(
            S.OBS_LAYOUT[name][1] - S.OBS_LAYOUT[name][0]
            for name in ("globals", "inventory", "tech", "recipes_enabled")
        )
        self.dense = nn.Sequential(
            nn.Linear(dense_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.targets = RowEncoder(S.TARGET_FEATURES)
        self.entities = RowEncoder(S.ENTITY_FEATURES)
        self.grid = nn.Sequential(
            nn.Conv2d(len(S.GRID_CHANNELS), 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(nn.Linear(256 + 128 * 3, 512), nn.ReLU())

    @staticmethod
    def _block(obs: torch.Tensor, name: str) -> torch.Tensor:
        start, stop = S.OBS_LAYOUT[name]
        return obs[..., start:stop]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dense = torch.cat(
            [
                self._block(obs, name)
                for name in ("globals", "inventory", "tech", "recipes_enabled")
            ],
            dim=-1,
        )
        targets = self._block(obs, "targets").reshape(
            -1, S.N_TARGET_SLOTS, S.TARGET_FEATURES
        )
        entities = self._block(obs, "entities").reshape(
            -1, S.N_ENTITY_SLOTS, S.ENTITY_FEATURES
        )
        grid = self._block(obs, "grid").reshape(
            -1, len(S.GRID_CHANNELS), S.GRID_SIDE, S.GRID_SIDE
        )
        encoded = torch.cat(
            (self.dense(dense), self.targets(targets), self.entities(entities), self.grid(grid)),
            dim=-1,
        )
        return self.trunk(encoded)


class BranchingDQN(nn.Module):
    """Dueling value function with one centred advantage per action head."""

    def __init__(self):
        super().__init__()
        self.encoder = ObservationEncoder()
        self.value = nn.Linear(512, 1)
        self.advantages = nn.ModuleDict(
            {head: nn.Linear(512, S.HEAD_SIZES[head]) for head in S.HEADS}
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encoder(obs.float())
        value = self.value(hidden).squeeze(-1)
        advantages = {}
        for head, layer in self.advantages.items():
            advantage = layer(hidden)
            advantages[head] = advantage - advantage.mean(dim=-1, keepdim=True)
        return value, advantages


class DuelingHead(nn.Module):
    """One complete branching dueling head operating on a shared encoding."""

    def __init__(self):
        super().__init__()
        self.value = nn.Linear(512, 1)
        self.advantages = nn.ModuleDict(
            {head: nn.Linear(512, S.HEAD_SIZES[head]) for head in S.HEADS}
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = self.value(hidden).squeeze(-1)
        advantages = {}
        for name, layer in self.advantages.items():
            advantage = layer(hidden)
            advantages[name] = advantage - advantage.mean(dim=-1, keepdim=True)
        return value, advantages


class BootstrappedDQN(nn.Module):
    """Shared encoder with independent dueling heads and frozen random priors."""

    def __init__(self, num_heads: int = 8, prior_scale: float = 1.0):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = num_heads
        self.prior_scale = prior_scale
        self.encoder = ObservationEncoder()
        self.heads = nn.ModuleList(DuelingHead() for _ in range(num_heads))
        self.prior_networks = nn.ModuleList(BranchingDQN() for _ in range(num_heads))
        for parameter in self.prior_networks.parameters():
            parameter.requires_grad_(False)

    def learned_outputs(
        self, obs: torch.Tensor
    ) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        hidden = self.encoder(obs.float())
        return [head(hidden) for head in self.heads]

    def prior_outputs(
        self, obs: torch.Tensor
    ) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        with torch.no_grad():
            return [network(obs) for network in self.prior_networks]

    def forward(
        self, obs: torch.Tensor
    ) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """Return learned plus prior Q outputs for behavior exploration."""
        learned = self.learned_outputs(obs)
        priors = self.prior_outputs(obs)
        outputs = []
        for (value, advantages), (prior_value, prior_advantages) in zip(
            learned, priors
        ):
            outputs.append(
                (
                    value + self.prior_scale * prior_value,
                    {
                        name: advantages[name] + self.prior_scale * prior_advantages[name]
                        for name in S.HEADS
                    },
                )
            )
        return outputs


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer with explicitly managed noise."""

    def __init__(
        self, in_features: int, out_features: int, sigma0: float = 0.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            "weight_epsilon", torch.empty(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.noise_enabled = True
        self.reset_parameters()
        self.reset_noise()

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_parameters(self) -> None:
        bound = 1.0 / np.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma0 / np.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma0 / np.sqrt(self.out_features))

    def reset_noise(self) -> None:
        input_noise = self._scale_noise(self.in_features, self.weight_mu.device)
        output_noise = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(output_noise.outer(input_noise))
        self.bias_epsilon.copy_(output_noise)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.noise_enabled:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(inputs, weight, bias)


class NoisyObservationEncoder(ObservationEncoder):
    """Existing observation encoder with only the trunk output made noisy."""

    def __init__(self):
        super().__init__()
        self.trunk[0] = NoisyLinear(256 + 128 * 3, 512)


class NoisyBranchingDQN(nn.Module):
    """Scalar branching DQN whose trunk and output layers use NoisyNet."""

    dist = "none"
    noisy = True

    def __init__(self):
        super().__init__()
        self.encoder = NoisyObservationEncoder()
        self.value = NoisyLinear(512, 1)
        self.advantages = nn.ModuleDict(
            {
                head: NoisyLinear(512, S.HEAD_SIZES[head])
                for head in S.HEADS
            }
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encoder(obs.float())
        value = self.value(hidden).squeeze(-1)
        advantages = {}
        for head, layer in self.advantages.items():
            advantage = layer(hidden)
            advantages[head] = advantage - advantage.mean(dim=-1, keepdim=True)
        return value, advantages


class QuantileOutputLayer(nn.Module):
    """Low-rank action-by-quantile output with an explicit expected-value path."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_quantiles: int,
        noisy: bool,
        rank: int = 4,
    ) -> None:
        super().__init__()
        layer = NoisyLinear if noisy else nn.Linear
        self.out_features = out_features
        self.num_quantiles = num_quantiles
        self.mean_layer = layer(in_features, out_features)
        if num_quantiles > 1:
            self.rank = min(rank, num_quantiles)
            self.residual_layer = layer(in_features, out_features * self.rank)
            self.quantile_basis = nn.Parameter(torch.empty(self.rank, num_quantiles))
            nn.init.normal_(self.quantile_basis, std=1.0 / np.sqrt(self.rank))
        else:
            self.rank = 0
            self.register_parameter("quantile_basis", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self.mean_layer(inputs).unsqueeze(-1)
        if self.num_quantiles == 1:
            return mean
        factors = self.residual_layer(inputs).reshape(
            -1, self.out_features, self.rank
        )
        residual = torch.einsum("bar,rq->baq", factors, self.quantile_basis)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        return mean + residual

    def expected(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mean_layer(inputs)


class QuantileBranchingDQN(nn.Module):
    """Branching QR-DQN with a shared quantile value and per-head advantages."""

    dist = "qr"

    def __init__(self, num_quantiles: int = 51, noisy: bool = False):
        super().__init__()
        if num_quantiles <= 0:
            raise ValueError("num_quantiles must be positive")
        self.num_quantiles = num_quantiles
        self.noisy = noisy
        self.encoder = NoisyObservationEncoder() if noisy else ObservationEncoder()
        self.value = QuantileOutputLayer(512, 1, num_quantiles, noisy)
        self.advantages = nn.ModuleDict(
            {
                head: QuantileOutputLayer(
                    512, S.HEAD_SIZES[head], num_quantiles, noisy
                )
                for head in S.HEADS
            }
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encoder(obs.float())
        value = self.value(hidden).squeeze(1)
        advantages = {}
        for head, layer in self.advantages.items():
            advantage = layer(hidden)
            advantages[head] = advantage - advantage.mean(dim=1, keepdim=True)
        return value, advantages

    def expected_forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encoder(obs.float())
        value = self.value.expected(hidden).squeeze(-1)
        advantages = {}
        for head, layer in self.advantages.items():
            advantage = layer.expected(hidden)
            advantages[head] = advantage - advantage.mean(dim=-1, keepdim=True)
        return value, advantages


DQNNetwork = BranchingDQN | NoisyBranchingDQN | QuantileBranchingDQN


def reset_network_noise(network: nn.Module) -> None:
    for module in network.modules():
        if isinstance(module, NoisyLinear):
            module.reset_noise()


def set_network_noise_enabled(network: nn.Module, enabled: bool) -> None:
    for module in network.modules():
        if isinstance(module, NoisyLinear):
            module.noise_enabled = enabled


def noisy_sigma_means(network: nn.Module) -> dict[str, float] | None:
    if not any(isinstance(module, NoisyLinear) for module in network.modules()):
        return None
    result = {}
    trunk = network.encoder.trunk[0]
    if isinstance(trunk, NoisyLinear):
        result["trunk"] = float(trunk.weight_sigma.detach().abs().mean())
    def sigma_mean(module: nn.Module) -> float | None:
        sigmas = [
            child.weight_sigma.detach().abs().mean()
            for child in module.modules()
            if isinstance(child, NoisyLinear)
        ]
        if not sigmas:
            return None
        return float(torch.stack(sigmas).mean())

    value_sigma = sigma_mean(network.value)
    if value_sigma is not None:
        result["value"] = value_sigma
    for head, layer in network.advantages.items():
        head_sigma = sigma_mean(layer)
        if head_sigma is not None:
            result[head] = head_sigma
    return result


def complete_action_q(
    value: torch.Tensor,
    advantages: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> torch.Tensor:
    """Evaluate V + op advantage + mean advantages of the op's arguments."""
    op_column = S.HEADS.index("op")
    chosen_ops = actions[:, op_column]
    op_advantage = advantages["op"].gather(1, chosen_ops[:, None]).squeeze(1)
    argument_advantage = torch.zeros_like(value)
    for op_index, op_name in enumerate(S.OPS):
        selected = chosen_ops == op_index
        if not selected.any():
            continue
        per_head = []
        for head in S.OP_HEADS[op_name]:
            column = S.HEADS.index(head)
            per_head.append(
                advantages[head][selected]
                .gather(1, actions[selected, column, None])
                .squeeze(1)
            )
        argument_advantage[selected] = torch.stack(per_head).mean(dim=0)
    return value + op_advantage + argument_advantage


def complete_action_quantiles(
    value: torch.Tensor,
    advantages: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> torch.Tensor:
    """Apply the existing branching action combination per quantile."""
    op_column = S.HEADS.index("op")
    chosen_ops = actions[:, op_column]
    rows = torch.arange(len(actions), device=actions.device)
    op_advantage = advantages["op"][rows, chosen_ops]
    argument_advantage = torch.zeros_like(value)
    for op_index, op_name in enumerate(S.OPS):
        selected = chosen_ops == op_index
        if not selected.any():
            continue
        per_head = []
        for head in S.OP_HEADS[op_name]:
            column = S.HEADS.index(head)
            head_actions = actions[selected, column]
            selected_rows = torch.arange(
                len(head_actions), device=actions.device
            )
            per_head.append(advantages[head][selected][selected_rows, head_actions])
        argument_advantage[selected] = torch.stack(per_head).mean(dim=0)
    return value + op_advantage + argument_advantage


def quantile_mean_outputs(
    value: torch.Tensor, advantages: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce a QR-DQN output to expected values for masked action selection."""
    return value.mean(dim=-1), {
        head: branch.mean(dim=-1) for head, branch in advantages.items()
    }


def expected_outputs(
    network: DQNNetwork, obs: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(network, QuantileBranchingDQN):
        return network.expected_forward(obs)
    return network(obs)


def greedy_actions_from_outputs(
    value: torch.Tensor,
    advantages: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Maximize the complete action Q while respecting every branch mask."""
    actions = torch.stack(
        [masked_argmax(advantages[head], masks[head]) for head in S.HEADS], dim=1
    )
    op_scores = []
    for op_index, op_name in enumerate(S.OPS):
        active_maxima = []
        for head in S.OP_HEADS[op_name]:
            selected = actions[:, S.HEADS.index(head)]
            active_maxima.append(advantages[head].gather(1, selected[:, None]).squeeze(1))
        op_scores.append(
            value
            + advantages["op"][:, op_index]
            + torch.stack(active_maxima).mean(dim=0)
        )
    scores = torch.stack(op_scores, dim=1)
    actions[:, S.HEADS.index("op")] = masked_argmax(scores, masks["op"])
    return actions


def greedy_actions(network: DQNNetwork, obs: torch.Tensor) -> torch.Tensor:
    value, advantages = expected_outputs(network, obs)
    return greedy_actions_from_outputs(value, advantages, observation_masks(obs))


def epsilon_greedy_actions(
    network: DQNNetwork,
    obs: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    exploration_flags: np.ndarray | None = None,
) -> np.ndarray:
    """Select each branch epsilon-greedily from its admitted indices."""
    with torch.no_grad():
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        actions = greedy_actions(network, tensor).cpu().numpy()
    return randomize_actions(actions, obs, epsilon, rng, exploration_flags)


def randomize_actions(
    actions: np.ndarray,
    obs: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    exploration_flags: np.ndarray | None = None,
) -> np.ndarray:
    """Apply independent masked epsilon exploration to complete actions."""
    masks = S.split_masks(obs)
    for row in range(len(obs)):
        for column, head in enumerate(S.HEADS):
            if rng.random() >= epsilon:
                continue
            if exploration_flags is not None:
                exploration_flags[row] = True
            admitted = np.flatnonzero(masks[head][row])
            actions[row, column] = int(rng.choice(admitted)) if admitted.size else 0
    return actions


def averaged_bootstrap_outputs(
    outputs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average complete dueling outputs across bootstrap heads for evaluation."""
    values = torch.stack([value for value, _ in outputs]).mean(dim=0)
    advantages = {
        name: torch.stack([head_advantages[name] for _, head_advantages in outputs]).mean(dim=0)
        for name in S.HEADS
    }
    return values, advantages


def bootstrapped_actions(
    network: BootstrappedDQN,
    obs: np.ndarray,
    acting_heads: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    exploration_flags: np.ndarray | None = None,
) -> np.ndarray:
    """Act with one fixed bootstrap head per environment episode."""
    with torch.no_grad():
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        masks = observation_masks(tensor)
        outputs = network(tensor)
        per_head_actions = torch.stack(
            [greedy_actions_from_outputs(value, advantages, masks) for value, advantages in outputs]
        )
        row_indices = torch.arange(len(obs), device=device)
        head_indices = torch.as_tensor(acting_heads, dtype=torch.long, device=device)
        actions = per_head_actions[head_indices, row_indices].cpu().numpy()
    return randomize_actions(actions, obs, epsilon, rng, exploration_flags)


def greedy_policy_actions(
    network: DQNNetwork | BootstrappedDQN,
    obs: torch.Tensor,
) -> torch.Tensor:
    """Greedy action, averaging Q outputs across heads for Bootstrapped DQN."""
    if isinstance(network, BootstrappedDQN):
        value, advantages = averaged_bootstrap_outputs(network.learned_outputs(obs))
        return greedy_actions_from_outputs(value, advantages, observation_masks(obs))
    return greedy_actions(network, obs)


@dataclass(frozen=True)
class PairCandidate:
    action: np.ndarray
    op: str
    primary_id: int
    primary: str
    q: float
    standardized_q: float = 0.0


def _row_block(obs: np.ndarray, name: str, rows: int, features: int) -> np.ndarray:
    start, stop = S.OBS_LAYOUT[name]
    return obs[start:stop].reshape(rows, features)


def _target_kind(obs: np.ndarray, slot: int) -> tuple[int, str]:
    row = _row_block(obs, "targets", S.N_TARGET_SLOTS, S.TARGET_FEATURES)[slot]
    kind = int(np.argmax(row[: len(S.TARGET_KINDS)]))
    return kind, S.TARGET_KINDS[kind]


def _entity_class(obs: np.ndarray, slot: int) -> tuple[int, str]:
    row = _row_block(obs, "entities", S.N_ENTITY_SLOTS, S.ENTITY_FEATURES)[slot]
    entity_class = int(np.argmax(row[: len(S.ENTITY_CLASSES)]))
    return entity_class, S.ENTITY_CLASSES[entity_class]


def _direct_primary_label(head: str, index: int) -> str:
    if head == "recipe":
        return S.RECIPE_NAMES[index]
    if head == "placeable":
        return S.PLACEABLE_NAMES[index]
    if head == "item":
        return S.ITEM_NAMES[index]
    if head == "technology":
        return S.TECH_NAMES[index]
    if head == "duration":
        return str(S.DURATIONS_TICKS[index])
    if head == "move_dir":
        return str(index)
    if head == "connector":
        return S.CONNECTOR_NAMES[index]
    raise KeyError(head)


def _require_primary(
    op: str, primary_id: int | None, primary: str | None
) -> tuple[int, str]:
    """Reject incomplete UCB keys, emitting at most one warning per operation."""
    if primary_id is not None and primary is not None:
        return primary_id, primary
    if op not in _MISSING_PRIMARY_WARNED:
        print(
            json.dumps(
                {
                    "event": "ucb_primary_missing",
                    "level": "warning",
                    "op": op,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        _MISSING_PRIMARY_WARNED.add(op)
    raise RuntimeError(f"UCB primary resolved to None for {op}")


def _primary_head(op: str) -> str:
    head = PRIMARY_HEADS.get(op)
    if head is None:
        _require_primary(op, None, None)
        raise AssertionError("unreachable")
    return head


def action_details(obs: np.ndarray, action: np.ndarray) -> dict[str, str | int]:
    """Decode the chosen primary and milestone-relevant arguments offline."""
    op_index = int(action[S.HEADS.index("op")])
    op = S.OPS[op_index]
    details: dict[str, str | int] = {"op": op}
    if op == "HARVEST":
        slot = int(action[S.HEADS.index("target")])
        primary_id, primary = _target_kind(obs, slot)
        details.update(primary_id=primary_id, primary=primary, target_slot=slot)
    elif op in {"PICKUP", "ROTATE"}:
        slot = int(action[S.HEADS.index("entity")])
        primary_id, primary = _entity_class(obs, slot)
        details.update(primary_id=primary_id, primary=primary, entity_slot=slot)
    else:
        head = _primary_head(op)
        primary_id = int(action[S.HEADS.index(head)])
        details.update(
            primary_id=primary_id,
            primary=_direct_primary_label(head, primary_id),
        )
    primary_id, primary = _require_primary(
        op,
        details.get("primary_id"),
        details.get("primary"),
    )
    details.update(primary_id=primary_id, primary=primary)
    if op in {"INSERT", "EXTRACT"}:
        slot = int(action[S.HEADS.index("entity")])
        _, entity_class = _entity_class(obs, slot)
        details["entity_class"] = entity_class
        details["item"] = S.ITEM_NAMES[int(action[S.HEADS.index("item")])]
    return details


def _primary_options(
    obs: np.ndarray, masks: dict[str, np.ndarray], op: str
) -> list[tuple[int, str, str, int]]:
    """Return (pair id, label, action head, action index) for supported primaries."""
    if op == "HARVEST":
        options = []
        for slot in np.flatnonzero(masks["target"]):
            primary_id, primary = _target_kind(obs, int(slot))
            options.append((primary_id, primary, "target", int(slot)))
        return options
    if op in {"PICKUP", "ROTATE"}:
        options = []
        for slot in np.flatnonzero(masks["entity"]):
            primary_id, primary = _entity_class(obs, int(slot))
            options.append((primary_id, primary, "entity", int(slot)))
        return options
    head = _primary_head(op)
    return [
        (int(index), _direct_primary_label(head, int(index)), head, int(index))
        for index in np.flatnonzero(masks[head])
    ]


def pair_candidates_from_outputs(
    obs: np.ndarray,
    value: torch.Tensor,
    advantages: dict[str, torch.Tensor],
) -> list[PairCandidate]:
    """Enumerate best greedy-secondary completion for every admitted primary pair."""
    tensor_obs = torch.as_tensor(obs[None], dtype=torch.float32, device=value.device)
    tensor_masks = observation_masks(tensor_obs)
    base = greedy_actions_from_outputs(value, advantages, tensor_masks)[0].cpu().numpy()
    masks = S.split_masks(obs)
    raw: list[tuple[np.ndarray, str, int, str]] = []
    op_column = S.HEADS.index("op")
    for op_index in np.flatnonzero(masks["op"]):
        op = S.OPS[int(op_index)]
        for primary_id, primary, head, action_index in _primary_options(obs, masks, op):
            action = base.copy()
            action[op_column] = int(op_index)
            action[S.HEADS.index(head)] = action_index
            raw.append((action, op, primary_id, primary))
    if not raw:
        return []
    actions = torch.as_tensor(
        np.stack([row[0] for row in raw]), dtype=torch.long, device=value.device
    )
    repeated_value = value.expand(len(raw))
    repeated_advantages = {
        head: branch.expand(len(raw), -1) for head, branch in advantages.items()
    }
    q_values = (
        complete_action_q(repeated_value, repeated_advantages, actions)
        .detach()
        .cpu()
        .numpy()
    )
    best_by_pair: dict[tuple[str, int], PairCandidate] = {}
    for (action, op, primary_id, primary), q_value in zip(raw, q_values):
        candidate = PairCandidate(action, op, primary_id, primary, float(q_value))
        key = (op, primary_id)
        if key not in best_by_pair or candidate.q > best_by_pair[key].q:
            best_by_pair[key] = candidate
    candidates = list(best_by_pair.values())
    q_array = np.asarray([candidate.q for candidate in candidates], dtype=np.float64)
    std = float(q_array.std())
    z_values = (q_array - q_array.mean()) / std if std > 1e-8 else np.zeros_like(q_array)
    return [
        PairCandidate(
            candidate.action,
            candidate.op,
            candidate.primary_id,
            candidate.primary,
            candidate.q,
            float(z_value),
        )
        for candidate, z_value in zip(candidates, z_values)
    ]


def sample_secondary_heads(
    action: np.ndarray,
    obs: np.ndarray,
    op: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample admitted secondary heads, preserving PLACE's greedy offset."""
    sampled = action.copy()
    masks = S.split_masks(obs)
    if op == "HARVEST":
        primary_head = "target"
    elif op in {"PICKUP", "ROTATE"}:
        primary_head = "entity"
    else:
        primary_head = _primary_head(op)
    for head in S.OP_HEADS[op]:
        if head == primary_head:
            continue
        if op == "PLACE" and head != "direction":
            continue
        admitted = np.flatnonzero(masks[head])
        if admitted.size:
            sampled[S.HEADS.index(head)] = int(rng.choice(admitted))
    return sampled


def _resource_offsets(obs: np.ndarray) -> np.ndarray:
    """Return admitted PLACE offsets whose matching observation tile has ore."""
    masks = S.split_masks(obs)
    grid = _row_block(
        obs,
        "grid",
        len(S.GRID_CHANNELS),
        S.GRID_SIDE * S.GRID_SIDE,
    ).reshape(len(S.GRID_CHANNELS), S.GRID_SIDE, S.GRID_SIDE)
    resource = grid[S.GRID_CHANNELS.index("resource")]
    return np.asarray(
        [
            int(index)
            for index in np.flatnonzero(masks["offset"])
            if resource[
                S.OFFSET_RADIUS + S.offset_to_dxdy(int(index))[1],
                S.OFFSET_RADIUS + S.offset_to_dxdy(int(index))[0],
            ]
            > 0.5
        ],
        dtype=np.int64,
    )


def _sample_resource_place_offset(
    action: np.ndarray,
    obs: np.ndarray,
    placeable: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an ore-backed admitted offset without changing a fallback action."""
    sampled = action.copy()
    if placeable not in RESOURCE_REQUIRING_PLACEABLES:
        return sampled
    offsets = _resource_offsets(obs)
    if offsets.size:
        sampled[S.HEADS.index("offset")] = int(rng.choice(offsets))
    return sampled


def _inventory_has(obs: np.ndarray, item: str) -> bool:
    start, _ = S.OBS_LAYOUT["inventory"]
    return bool(obs[start + S.ITEM_INDEX[item]] > 0)


def _entity_slots_of_class(obs: np.ndarray, entity_class: str) -> list[int]:
    rows = _row_block(obs, "entities", S.N_ENTITY_SLOTS, S.ENTITY_FEATURES)
    class_index = S.ENTITY_CLASSES.index(entity_class)
    return [
        slot
        for slot, row in enumerate(rows)
        if row[-2] > 0.5 and int(np.argmax(row[: len(S.ENTITY_CLASSES)])) == class_index
    ]


def _frontier_force_candidates(
    obs: np.ndarray,
    candidates: Sequence[PairCandidate],
    explorer: UCBExplorer,
    rung: int,
    max_count: int,
    tried_actions: set[tuple] | None = None,
) -> list[PairCandidate]:
    """Return distinct under-tried full drill-sequence actions."""
    masks = S.split_masks(obs)
    drill_slots = _entity_slots_of_class(obs, "mining-drill")
    entity_mask = masks["entity"]
    drill_slots = [slot for slot in drill_slots if entity_mask[slot]]
    tried = tried_actions or set()
    eligible: list[PairCandidate] = []
    for candidate in candidates:
        if explorer._pair_count(rung, candidate) > max_count:
            continue
        if (
            candidate.op == "PLACE"
            and candidate.primary == DRILL_RECIPE
            and _inventory_has(obs, DRILL_RECIPE)
        ):
            offsets = _resource_offsets(obs)
            if not offsets.size:
                offsets = np.asarray(
                    [candidate.action[S.HEADS.index("offset")]], dtype=np.int64
                )
            directions = np.flatnonzero(masks["direction"])
            for offset in offsets:
                for direction in directions:
                    action = candidate.action.copy()
                    action[S.HEADS.index("offset")] = int(offset)
                    action[S.HEADS.index("direction")] = int(direction)
                    completed = PairCandidate(
                        action,
                        candidate.op,
                        candidate.primary_id,
                        candidate.primary,
                        candidate.q,
                        candidate.standardized_q,
                    )
                    if frontier_trial_key(completed) not in tried:
                        eligible.append(completed)
        elif (
            candidate.op == "INSERT"
            and drill_slots
            and candidate.primary in FUEL_ITEMS
            and _inventory_has(obs, candidate.primary)
        ):
            inventory_start, _ = S.OBS_LAYOUT["inventory"]
            held_scaled = obs[inventory_start + candidate.primary_id]
            held = round(np.expm1(float(held_scaled) * np.log1p(1000)))
            quantities = [
                index
                for index in np.flatnonzero(masks["quantity"])
                if S.QUANTITIES[int(index)] <= held
            ]
            for slot in drill_slots:
                for quantity in quantities:
                    action = candidate.action.copy()
                    action[S.HEADS.index("entity")] = int(slot)
                    action[S.HEADS.index("quantity")] = int(quantity)
                    completed = PairCandidate(
                        action,
                        candidate.op,
                        candidate.primary_id,
                        candidate.primary,
                        candidate.q,
                        candidate.standardized_q,
                    )
                    if frontier_trial_key(completed) not in tried:
                        eligible.append(completed)
        else:
            continue
    return eligible


def frontier_trial_key(candidate: PairCandidate) -> tuple:
    """Identify one complete forced trial without conflating its secondaries."""
    primary_head = _primary_head(candidate.op)
    secondary = tuple(
        int(candidate.action[S.HEADS.index(head)])
        for head in S.OP_HEADS[candidate.op]
        if head != primary_head
    )
    return candidate.op, candidate.primary_id, secondary


class UCBExplorer:
    """Global count-UCB behavior policy over supported op/primary pairs."""

    def __init__(self, c: float = 1.5, key_mode: str = "global"):
        if key_mode not in {"global", "rung"}:
            raise ValueError("UCB key mode must be 'global' or 'rung'")
        self.c = c
        self.key_mode = key_mode
        self.counts: Counter[tuple] = Counter()
        self.total = 0

    def _key(self, rung: int, candidate: PairCandidate) -> tuple:
        pair = (candidate.op, candidate.primary_id)
        if self.key_mode == "rung" and candidate.op == "CRAFT":
            quantity_index = int(candidate.action[S.HEADS.index("quantity")])
            return (rung, *pair, S.QUANTITIES[quantity_index])
        return (rung, *pair) if self.key_mode == "rung" else pair

    def _pair_count(self, rung: int, candidate: PairCandidate) -> int:
        if self.key_mode != "rung" or candidate.op != "CRAFT":
            return self.counts[self._key(rung, candidate)]
        prefix = (rung, candidate.op, candidate.primary_id)
        return sum(
            count for key, count in self.counts.items() if key[:3] == prefix
        )

    def reconcile_executed_quantity(
        self, diagnostic: dict, executed_quantity: int | None
    ) -> None:
        """Move a rung-keyed CRAFT count from requested to executed quantity."""
        if (
            self.key_mode != "rung"
            or diagnostic.get("op") != "CRAFT"
            or executed_quantity is None
        ):
            return
        old_key = tuple(diagnostic["_count_key"])
        new_key = (
            int(diagnostic["rung"]),
            "CRAFT",
            int(diagnostic["primary_id"]),
            int(executed_quantity),
        )
        if new_key == old_key:
            return
        self.counts[old_key] -= 1
        if self.counts[old_key] <= 0:
            del self.counts[old_key]
        self.counts[new_key] += 1
        diagnostic["_count_key"] = new_key

    def replace_last_selection(
        self,
        diagnostic: dict,
        obs: np.ndarray,
        action: np.ndarray,
        rung: int,
    ) -> dict:
        """Replace a just-counted behavior selection with a forced action."""
        old_key = tuple(diagnostic["_count_key"])
        self.counts[old_key] -= 1
        if self.counts[old_key] <= 0:
            del self.counts[old_key]
        self.total -= 1
        details = action_details(obs, action)
        candidate = PairCandidate(
            action.copy(),
            str(details["op"]),
            int(details["primary_id"]),
            str(details["primary"]),
            0.0,
            0.0,
        )
        key = self._key(rung, candidate)
        n_before = self.counts[key]
        total_before = self.total
        self.counts[key] += 1
        self.total += 1
        return {
            "exploratory": True,
            "op": candidate.op,
            "primary": candidate.primary,
            "primary_id": candidate.primary_id,
            "standardized_q": 0.0,
            "ucb_bonus": 0.0,
            "ucb_c": self.c,
            "rung": rung,
            "n": n_before,
            "N": total_before,
            "secondary_heads_sampled": False,
            "forced_frontier_action": True,
            "frontier_forced": False,
            "frontier_trial": None,
            "admitted_pair_count": diagnostic.get("admitted_pair_count"),
            "n0_pair_count": diagnostic.get("n0_pair_count"),
            "_count_key": key,
        }

    def select(
        self,
        network: DQNNetwork,
        obs: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
        device: torch.device,
        rungs: np.ndarray | None = None,
        coefficient: float | None = None,
        forced_arg_eps: float = 0.0,
        retry_pairs: Sequence[tuple[int, str, int] | None] | None = None,
        frontier_forcing: Sequence[bool] | None = None,
        frontier_tried_actions: Sequence[set[tuple]] | None = None,
        frontier_force_max_count: int = 2,
    ) -> tuple[np.ndarray, list[dict]]:
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            values, advantages = expected_outputs(network, tensor)
        actions = []
        diagnostics = []
        for row in range(len(obs)):
            rung = int(rungs[row]) if rungs is not None else -1
            candidates = pair_candidates_from_outputs(
                obs[row],
                values[row : row + 1],
                {head: branch[row : row + 1] for head, branch in advantages.items()},
            )
            if not candidates:
                action = S.random_valid_action(obs[row], rng)
                details = action_details(obs[row], action)
                candidates = [
                    PairCandidate(
                        action,
                        str(details["op"]),
                        int(details["primary_id"]),
                        str(details["primary"]),
                        0.0,
                        0.0,
                    )
                ]
            random_floor = rng.random() < epsilon
            force_frontier = bool(
                frontier_forcing is not None and frontier_forcing[row]
            )
            frontier_candidates = (
                _frontier_force_candidates(
                    obs[row],
                    candidates,
                    self,
                    rung,
                    frontier_force_max_count,
                    (
                        None
                        if frontier_tried_actions is None
                        else frontier_tried_actions[row]
                    ),
                )
                if force_frontier
                else []
            )
            unseen = [
                candidate
                for candidate in candidates
                if self._pair_count(rung, candidate) == 0
            ]
            active_c = self.c if coefficient is None else coefficient
            bonuses = {
                self._key(rung, candidate): active_c
                * np.sqrt(
                    np.log(self.total + 1) /
                    (self.counts[self._key(rung, candidate)] + 1)
                )
                for candidate in candidates
            }
            frontier_forced = bool(frontier_candidates)
            if frontier_forced:
                chosen = frontier_candidates[int(rng.integers(len(frontier_candidates)))]
                action = chosen.action.copy()
                key = self._key(rung, chosen)
                random_floor = False
            elif unseen:
                chosen = max(unseen, key=lambda candidate: candidate.standardized_q)
                action = chosen.action.copy()
                key = self._key(rung, chosen)
                random_floor = False
            elif random_floor:
                action = S.random_valid_action(obs[row], rng)
                details = action_details(obs[row], action)
                pair = (str(details["op"]), int(details["primary_id"]))
                chosen = next(
                    (
                        candidate
                        for candidate in candidates
                        if (candidate.op, candidate.primary_id) == pair
                    ),
                    PairCandidate(
                        action,
                        pair[0],
                        pair[1],
                        str(details["primary"]),
                        0.0,
                        0.0,
                    ),
                )
                key = self._key(rung, chosen)
            else:
                chosen = max(
                    candidates,
                    key=lambda candidate: candidate.standardized_q
                    + bonuses[self._key(rung, candidate)],
                )
                action = chosen.action.copy()
                key = self._key(rung, chosen)
            pair_n_before = self._pair_count(rung, chosen)
            total_before = self.total
            retry_pair = None if retry_pairs is None else retry_pairs[row]
            force_secondary = bool(
                not frontier_forced
                and forced_arg_eps > 0
                and (
                    pair_n_before == 0
                    or retry_pair == (rung, chosen.op, chosen.primary_id)
                )
                and rng.random() < forced_arg_eps
            )
            if force_secondary:
                action = sample_secondary_heads(
                    action, obs[row], chosen.op, rng
                )
            if (
                not frontier_forced
                and pair_n_before == 0
                and chosen.op == "PLACE"
            ):
                action = _sample_resource_place_offset(
                    action, obs[row], chosen.primary, rng
                )
            if pair_n_before == 0 and "quantity" in S.OP_HEADS[chosen.op]:
                quantity_mask = S.split_masks(obs[row])["quantity"]
                admitted = np.flatnonzero(quantity_mask)
                if admitted.size:
                    action[S.HEADS.index("quantity")] = int(admitted[0])
            chosen = PairCandidate(
                action,
                chosen.op,
                chosen.primary_id,
                chosen.primary,
                chosen.q,
                chosen.standardized_q,
            )
            key = self._key(rung, chosen)
            n_before = self.counts[key]
            selected_bonus = active_c * np.sqrt(
                np.log(self.total + 1) / (n_before + 1)
            )
            greedy_pair = max(candidates, key=lambda candidate: candidate.standardized_q)
            exploratory = (
                random_floor
                or n_before == 0
                or (chosen.op, chosen.primary_id)
                != (greedy_pair.op, greedy_pair.primary_id)
            )
            self.counts[key] += 1
            self.total += 1
            actions.append(action)
            diagnostics.append(
                {
                    "exploratory": exploratory,
                    "op": chosen.op,
                    "primary": chosen.primary,
                    "primary_id": chosen.primary_id,
                    "standardized_q": chosen.standardized_q,
                    "ucb_bonus": selected_bonus,
                    "ucb_c": active_c,
                    "rung": rung,
                    "n": n_before,
                    "N": total_before,
                    "secondary_heads_sampled": force_secondary,
                    "frontier_forced": frontier_forced,
                    "frontier_trial": (
                        frontier_trial_key(chosen) if frontier_forced else None
                    ),
                    "admitted_pair_count": len(candidates),
                    "n0_pair_count": len(unseen),
                    "_count_key": key,
                }
            )
        return np.stack(actions), diagnostics

    def state_dict(self) -> dict:
        counts = []
        for key, count in sorted(self.counts.items()):
            counts.append((*key, count))
        return {
            "counts": counts,
            "total": self.total,
            "key_mode": self.key_mode,
        }

    def load_state_dict(self, state: dict) -> None:
        saved_mode = str(state.get("key_mode", "global"))
        if saved_mode != self.key_mode:
            print(
                "resume_initialized_fresh incompatible=ucb_counts_key_mode "
                f"saved={saved_mode} requested={self.key_mode}",
                flush=True,
            )
            self.counts.clear()
            self.total = 0
            return
        counts: Counter[tuple] = Counter()
        dropped_keys = 0
        dropped_selections = 0
        for row in state["counts"]:
            if len(row) == 3:
                op, primary_id, count = row
                if str(op) == "PLACE" and primary_id is None:
                    dropped_keys += 1
                    dropped_selections += int(count)
                    continue
                key = (str(op), int(primary_id))
            elif len(row) == 4:
                rung, op, primary_id, count = row
                if str(op) == "PLACE" and primary_id is None:
                    dropped_keys += 1
                    dropped_selections += int(count)
                    continue
                key = (int(rung), str(op), int(primary_id))
            elif len(row) == 5:
                rung, op, primary_id, quantity, count = row
                if str(op) == "PLACE" and primary_id is None:
                    dropped_keys += 1
                    dropped_selections += int(count)
                    continue
                key = (int(rung), str(op), int(primary_id), int(quantity))
            else:
                raise ValueError("invalid UCB checkpoint count row")
            counts[key] = int(count)
        if dropped_keys:
            print(
                "warning=resume_dropped_legacy_ucb_place_none_keys "
                f"keys={dropped_keys} selections={dropped_selections}",
                flush=True,
            )
        self.counts = counts
        self.total = max(0, int(state["total"]) - dropped_selections)


def action_diagnostics(
    network: DQNNetwork | BootstrappedDQN,
    obs: np.ndarray,
    actions: np.ndarray,
    exploratory: np.ndarray,
    device: torch.device,
    acting_heads: np.ndarray | None = None,
) -> list[dict]:
    """Describe epsilon/boot actions in the same format as UCB action logs."""
    with torch.no_grad():
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if isinstance(network, BootstrappedDQN):
            all_outputs = network(tensor)
        else:
            values, advantages = expected_outputs(network, tensor)
            all_outputs = [(values, advantages)]
    rows = []
    for row, (observation, action) in enumerate(zip(obs, actions)):
        head_index = int(acting_heads[row]) if acting_heads is not None else 0
        values, advantages = all_outputs[head_index]
        candidates = pair_candidates_from_outputs(
            observation,
            values[row : row + 1],
            {head: branch[row : row + 1] for head, branch in advantages.items()},
        )
        details = action_details(observation, action)
        key = (str(details["op"]), int(details["primary_id"]))
        chosen = next(
            (candidate for candidate in candidates if (candidate.op, candidate.primary_id) == key),
            None,
        )
        rows.append(
            {
                "exploratory": bool(exploratory[row]),
                "op": key[0],
                "primary": str(details["primary"]),
                "primary_id": key[1],
                "standardized_q": chosen.standardized_q if chosen is not None else 0.0,
                "ucb_bonus": None,
                "n": None,
                "N": None,
                "acting_head": head_index if acting_heads is not None else None,
                "secondary_heads_sampled": False,
            }
        )
    return rows


@dataclass(frozen=True)
class NStepTransition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    discount: float
    bootstrap_mask: np.ndarray | None = None
    behaviour: str = "unknown"
    excursion: bool = False


class EpisodeMilestones:
    """Track the deepest overnight-plan ladder rung reached in one episode."""

    def __init__(self):
        self.furnace_fuelled = False
        self.furnace_fed = False
        self.reached = [False] * len(RUNG_NAMES)
        self.first_attainment: list[int | None] = [None] * len(RUNG_NAMES)

    @classmethod
    def from_frontier(cls, rung: int) -> EpisodeMilestones:
        """Start excursion tracking at the rung already present in a restore."""
        milestones = cls()
        for index in range(min(rung + 1, len(RUNG_NAMES))):
            milestones.reached[index] = True
        if rung >= 0:
            milestones.furnace_fuelled = True
            milestones.furnace_fed = True
        return milestones

    def observe(
        self,
        details: dict[str, str | int],
        status: str,
        step: int | None = None,
    ) -> int:
        if status != "ok":
            return self.deepest
        before = self.reached.copy()
        op = str(details["op"])
        primary = str(details["primary"])
        entity_class = str(details.get("entity_class", ""))
        if op == "INSERT" and entity_class == "furnace":
            self.furnace_fuelled |= primary in FUEL_ITEMS
            self.furnace_fed |= primary in FURNACE_INPUTS
            self.reached[0] |= self.furnace_fuelled and self.furnace_fed
        elif op == "EXTRACT" and primary in {"iron-plate", "copper-plate"}:
            self.reached[1] = True
        elif op == "CRAFT" and primary == "iron-gear-wheel":
            self.reached[2] = True
        elif op == "CRAFT" and primary == "burner-mining-drill":
            self.reached[3] = True
        elif op == "PLACE" and primary == "burner-mining-drill":
            self.reached[4] = True
        elif op == "INSERT" and entity_class == "mining-drill" and primary in FUEL_ITEMS:
            self.reached[5] = True
        if step is not None:
            for index, (was_reached, is_reached) in enumerate(
                zip(before, self.reached)
            ):
                if not was_reached and is_reached:
                    self.first_attainment[index] = step
        return self.deepest

    @property
    def deepest(self) -> int:
        return max((index for index, reached in enumerate(self.reached) if reached), default=-1)


def is_new_return_frontier(previous_rung: int, current_rung: int, return_rung: int) -> bool:
    """Whether this transition newly crossed the configured return threshold."""
    return current_rung >= return_rung and current_rung > previous_rung


def advance_frontier_force_window(
    remaining: int,
    previous_rung: int,
    current_rung: int,
    tries: int,
    trial_was_new: bool = False,
) -> int:
    """Count distinct forced trials, resetting after a new drill-or-deeper rung."""
    if tries > 0 and current_rung > previous_rung and current_rung >= DRILL_CRAFTED_RUNG:
        return tries
    return max(0, remaining - int(trial_was_new))


def drill_recipe_admitted(obs: np.ndarray) -> bool:
    """Whether the current support masks admit hand-crafting the burner drill."""
    masks = S.split_masks(obs)
    return bool(
        masks["op"][S.OP_INDEX["CRAFT"]]
        and masks["recipe"][S.RECIPE_INDEX[DRILL_RECIPE]]
    )


def drill_place_admitted(obs: np.ndarray) -> bool:
    """Whether the held burner drill can be selected for PLACE."""
    masks = S.split_masks(obs)
    return bool(
        _inventory_has(obs, DRILL_RECIPE)
        and masks["op"][S.OP_INDEX["PLACE"]]
        and masks["placeable"][S.PLACEABLE_INDEX[DRILL_RECIPE]]
    )


def forced_drill_action(
    obs: np.ndarray,
    base: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build the first excursion action: PLACE held drill, otherwise CRAFT it."""
    if not drill_place_admitted(obs) and not drill_recipe_admitted(obs):
        raise ValueError("burner-mining-drill PLACE and CRAFT are not admitted")
    rng = np.random.default_rng(0) if rng is None else rng
    action = (
        S.random_valid_action(obs, rng)
        if base is None
        else base.copy()
    )
    if drill_place_admitted(obs):
        action[S.HEADS.index("op")] = S.OP_INDEX["PLACE"]
        action[S.HEADS.index("placeable")] = S.PLACEABLE_INDEX[DRILL_RECIPE]
        action = _sample_resource_place_offset(action, obs, DRILL_RECIPE, rng)
        directions = np.flatnonzero(S.split_masks(obs)["direction"])
        if directions.size:
            action[S.HEADS.index("direction")] = int(rng.choice(directions))
        return action
    action[S.HEADS.index("op")] = S.OP_INDEX["CRAFT"]
    action[S.HEADS.index("recipe")] = S.RECIPE_INDEX[DRILL_RECIPE]
    quantities = np.flatnonzero(S.split_masks(obs)["quantity"])
    if not quantities.size:
        raise ValueError("drill CRAFT has no admitted quantity")
    action[S.HEADS.index("quantity")] = int(quantities[0])
    return action


@dataclass(frozen=True)
class ArchivedFrontier:
    """Runtime return state plus the checkpoint-safe metadata describing it."""

    state: Any | None
    restore_hash: str
    rung: int
    step: int


@dataclass(frozen=True)
class ArchivedEpisode:
    episode_id: int
    rung: int
    aps: float
    transitions: tuple[NStepTransition, ...]
    first_attainment: int | None = None
    frontier: ArchivedFrontier | None = None


class MilestoneArchive:
    """Bounded complete-episode archive for ladder milestones and high APS."""

    def __init__(
        self,
        per_rung: int = MAX_ARCHIVE_EPISODES_PER_RUNG,
        top_aps: int = MAX_ARCHIVE_TOP_EPISODES,
    ):
        self.per_rung = per_rung
        self.top_aps = top_aps
        self.next_episode_id = 0
        self.episodes: dict[int, ArchivedEpisode] = {}
        self.rung_ids: dict[int, list[int]] = {index: [] for index in range(len(RUNG_NAMES))}
        self.top_ids: list[int] = []
        self._flat: list[tuple[int, int]] = []
        self.draw_counts: Counter[str] = Counter()

    @staticmethod
    def _compact_transition(transition: NStepTransition) -> NStepTransition:
        return NStepTransition(
            obs=np.asarray(transition.obs, dtype=np.float16).copy(),
            action=np.asarray(transition.action, dtype=np.int16).copy(),
            reward=float(transition.reward),
            next_obs=np.asarray(transition.next_obs, dtype=np.float16).copy(),
            discount=float(transition.discount),
            bootstrap_mask=(
                None
                if transition.bootstrap_mask is None
                else np.asarray(transition.bootstrap_mask, dtype=np.uint8).copy()
            ),
            behaviour=transition.behaviour,
            excursion=transition.excursion,
        )

    def add_episode(
        self,
        transitions: Sequence[NStepTransition],
        rung: int,
        aps: float,
        first_attainment: int | None = None,
        frontier: ArchivedFrontier | None = None,
        comparable_aps: bool = True,
    ) -> None:
        if not transitions:
            return
        qualifies_top = comparable_aps and (
            len(self.top_ids) < self.top_aps
            or any(aps > self.episodes[episode_id].aps for episode_id in self.top_ids)
        )
        if rung < 0 and not qualifies_top:
            return
        episode_id = self.next_episode_id
        self.next_episode_id += 1
        self.episodes[episode_id] = ArchivedEpisode(
            episode_id,
            rung,
            float(aps),
            tuple(self._compact_transition(transition) for transition in transitions),
            first_attainment,
            frontier,
        )
        if rung >= 0:
            bucket = self.rung_ids[rung]
            bucket.append(episode_id)
            if len(bucket) > self.per_rung:
                bucket.pop(0)
        if comparable_aps:
            self.top_ids.append(episode_id)
            self.top_ids.sort(key=lambda key: self.episodes[key].aps, reverse=True)
            del self.top_ids[self.top_aps :]
        referenced = set(self.top_ids)
        for bucket in self.rung_ids.values():
            referenced.update(bucket)
        for key in tuple(self.episodes):
            if key not in referenced:
                del self.episodes[key]
        self._rebuild_flat()

    def _rebuild_flat(self) -> None:
        self._flat = [
            (episode_id, transition_index)
            for episode_id, episode in self.episodes.items()
            for transition_index in range(len(episode.transitions))
        ]

    @property
    def transition_count(self) -> int:
        return len(self._flat)

    def sizes(self) -> dict:
        return {
            "episodes": len(self.episodes),
            "transitions": self.transition_count,
            "top_aps": len(self.top_ids),
            "per_rung": {
                RUNG_NAMES[index]: len(self.rung_ids[index])
                for index in range(len(RUNG_NAMES))
            },
        }

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
        num_bootstrap_heads: int,
        mode: str = "uniform",
    ) -> ReplayBatch:
        if not self._flat:
            raise ValueError("cannot sample empty milestone archive")
        if mode == "uniform":
            selected = rng.integers(len(self._flat), size=batch_size)
            references = [self._flat[index] for index in selected]
        elif mode == "frontier":
            references = self._sample_frontier(batch_size, rng)
        else:
            raise ValueError(f"unknown archive mode {mode!r}")
        transitions = []
        for episode_id, transition_index in references:
            episode = self.episodes[episode_id]
            transitions.append(episode.transitions[transition_index])
            self._record_draw(episode, transition_index, mode)
        masks = np.ones((batch_size, num_bootstrap_heads), dtype=np.uint8)
        for row, transition in enumerate(transitions):
            if transition.bootstrap_mask is not None:
                masks[row] = transition.bootstrap_mask
        return ReplayBatch(
            obs=torch.as_tensor(
                np.stack([transition.obs for transition in transitions]),
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                np.stack([transition.action for transition in transitions]),
                dtype=torch.long,
                device=device,
            ),
            rewards=torch.as_tensor(
                [transition.reward for transition in transitions], device=device
            ),
            next_obs=torch.as_tensor(
                np.stack([transition.next_obs for transition in transitions]),
                dtype=torch.float32,
                device=device,
            ),
            discounts=torch.as_tensor(
                [transition.discount for transition in transitions], device=device
            ),
            weights=torch.ones(batch_size, device=device),
            indices=np.full(batch_size, -1, dtype=np.int64),
            bootstrap_masks=torch.as_tensor(masks, dtype=torch.bool, device=device),
            excursions=torch.as_tensor(
                [transition.excursion for transition in transitions],
                dtype=torch.bool,
                device=device,
            ),
        )

    def _sample_frontier(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        rung_values = [episode.rung for episode in self.episodes.values() if episode.rung >= 0]
        if not rung_values:
            groups = [list(self.episodes)] * batch_size
        else:
            deepest = max(rung_values)
            deepest_ids = [
                episode_id
                for episode_id, episode in self.episodes.items()
                if episode.rung == deepest
            ]
            preceding_ids = [
                episode_id
                for episode_id, episode in self.episodes.items()
                if episode.rung == deepest - 1
            ]
            other_ids = [
                episode_id
                for episode_id, episode in self.episodes.items()
                if episode.rung not in {deepest, deepest - 1}
            ]
            all_ids = list(self.episodes)
            deepest_count = round(batch_size * 0.50)
            preceding_count = round(batch_size * 0.25)
            other_count = batch_size - deepest_count - preceding_count
            groups = (
                [(deepest_ids or all_ids)] * deepest_count
                + [(preceding_ids or all_ids)] * preceding_count
                + [(other_ids or all_ids)] * other_count
            )
        references = []
        for episode_ids in groups:
            episode_id = int(rng.choice(episode_ids))
            episode = self.episodes[episode_id]
            use_window = episode.first_attainment is not None and rng.random() < 0.75
            if use_window:
                low = max(0, episode.first_attainment - 32)
                high = min(len(episode.transitions), episode.first_attainment + 9)
                transition_index = int(rng.integers(low, max(low + 1, high)))
            else:
                transition_index = int(rng.integers(len(episode.transitions)))
            references.append((episode_id, transition_index))
            self.draw_counts[f"source:{'frontier_window' if use_window else 'episode_uniform'}"] += 1
        return references

    def _record_draw(
        self, episode: ArchivedEpisode, transition_index: int, mode: str
    ) -> None:
        rung = RUNG_NAMES[episode.rung] if episode.rung >= 0 else "none"
        self.draw_counts[f"rung:{rung}"] += 1
        if episode.first_attainment is None:
            offset = "unknown"
        else:
            delta = transition_index - episode.first_attainment
            if -32 <= delta < 0:
                offset = "pre_-32_-1"
            elif delta == 0:
                offset = "attainment"
            elif 0 < delta <= 8:
                offset = "post_1_8"
            else:
                offset = "outside_window"
        self.draw_counts[f"offset:{offset}"] += 1
        self.draw_counts[f"mode:{mode}"] += 1

    def consume_draw_counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {
            "by_rung": {},
            "by_offset": {},
            "by_source": {},
            "by_mode": {},
        }
        categories = {
            "rung": "by_rung",
            "offset": "by_offset",
            "source": "by_source",
            "mode": "by_mode",
        }
        for key, count in self.draw_counts.items():
            prefix, label = key.split(":", 1)
            result[categories[prefix]][label] = count
        self.draw_counts.clear()
        return result

    def state_dict(self) -> dict:
        episodes = []
        for episode in self.episodes.values():
            episodes.append(
                {
                    "episode_id": episode.episode_id,
                    "rung": episode.rung,
                    "aps": episode.aps,
                    "first_attainment": episode.first_attainment,
                    "frontier": (
                        None
                        if episode.frontier is None
                        else {
                            "hash": episode.frontier.restore_hash,
                            "rung": episode.frontier.rung,
                            "step": episode.frontier.step,
                        }
                    ),
                    "transitions": [
                        {
                            "obs": torch.from_numpy(transition.obs.copy()),
                            "action": torch.from_numpy(transition.action.copy()),
                            "reward": transition.reward,
                            "next_obs": torch.from_numpy(transition.next_obs.copy()),
                            "discount": transition.discount,
                            "bootstrap_mask": (
                                None
                                if transition.bootstrap_mask is None
                                else torch.from_numpy(transition.bootstrap_mask.copy())
                            ),
                            "behaviour": transition.behaviour,
                            "excursion": transition.excursion,
                        }
                        for transition in episode.transitions
                    ],
                }
            )
        return {
            "per_rung": self.per_rung,
            "top_aps": self.top_aps,
            "next_episode_id": self.next_episode_id,
            "episodes": episodes,
            "rung_ids": self.rung_ids,
            "top_ids": self.top_ids,
        }

    def load_state_dict(self, state: dict) -> None:
        self.per_rung = int(state["per_rung"])
        self.top_aps = int(state["top_aps"])
        self.next_episode_id = int(state["next_episode_id"])
        self.episodes.clear()
        for row in state["episodes"]:
            transitions = tuple(
                NStepTransition(
                    obs=transition["obs"].cpu().numpy(),
                    action=transition["action"].cpu().numpy(),
                    reward=float(transition["reward"]),
                    next_obs=transition["next_obs"].cpu().numpy(),
                    discount=float(transition["discount"]),
                    bootstrap_mask=(
                        None
                        if transition["bootstrap_mask"] is None
                        else transition["bootstrap_mask"].cpu().numpy()
                    ),
                    behaviour=str(transition.get("behaviour", "unknown")),
                    excursion=bool(transition.get("excursion", False)),
                )
                for transition in row["transitions"]
            )
            frontier_metadata = row.get("frontier")
            frontier = (
                None
                if frontier_metadata is None
                else ArchivedFrontier(
                    state=None,
                    restore_hash=str(frontier_metadata["hash"]),
                    rung=int(frontier_metadata["rung"]),
                    step=int(frontier_metadata["step"]),
                )
            )
            episode = ArchivedEpisode(
                int(row["episode_id"]),
                int(row["rung"]),
                float(row["aps"]),
                transitions,
                (
                    None
                    if row.get("first_attainment") is None
                    else int(row["first_attainment"])
                ),
                frontier,
            )
            self.episodes[episode.episode_id] = episode
        self.rung_ids = {
            int(index): [int(episode_id) for episode_id in episode_ids]
            for index, episode_ids in state["rung_ids"].items()
        }
        self.top_ids = [int(episode_id) for episode_id in state["top_ids"]]
        self._rebuild_flat()


@dataclass(frozen=True)
class _OneStepTransition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    bootstrap_mask: np.ndarray | None
    behaviour: str
    excursion: bool


class NStepAccumulator:
    """Convert a stream of one-step transitions into n-step transitions."""

    def __init__(self, n: int = 3, gamma: float = 0.99):
        self.n = n
        self.gamma = gamma
        self.pending: deque[_OneStepTransition] = deque()

    def _emit(self, length: int, *, boundary: bool = False) -> NStepTransition:
        steps = list(self.pending)[:length]
        reward = sum((self.gamma**index) * step.reward for index, step in enumerate(steps))
        last = steps[-1]
        discount = 0.0 if last.done or boundary else self.gamma**length
        first = self.pending.popleft()
        return NStepTransition(
            first.obs,
            first.action,
            reward,
            last.next_obs,
            discount,
            first.bootstrap_mask,
            first.behaviour,
            first.excursion,
        )

    def append(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        bootstrap_mask: np.ndarray | None = None,
        behaviour: str = "unknown",
        excursion: bool = False,
    ) -> list[NStepTransition]:
        emitted = []
        if self.pending and self.pending[0].excursion != excursion:
            while self.pending:
                emitted.append(
                    self._emit(min(self.n, len(self.pending)), boundary=True)
                )
        self.pending.append(
            _OneStepTransition(
                obs,
                action,
                reward,
                next_obs,
                done,
                bootstrap_mask,
                behaviour,
                excursion,
            )
        )
        if done:
            while self.pending:
                emitted.append(self._emit(min(self.n, len(self.pending))))
        elif len(self.pending) >= self.n:
            emitted.append(self._emit(self.n))
        return emitted


@dataclass(frozen=True)
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    discounts: torch.Tensor
    weights: torch.Tensor | None = None
    indices: np.ndarray | None = None
    bootstrap_masks: torch.Tensor | None = None
    excursions: torch.Tensor | None = None


class ReplayBuffer:
    """Fixed-size uniform or proportional-priority replay."""

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        prioritized: bool = False,
        alpha: float = 0.6,
        num_bootstrap_heads: int = 1,
        priority_cap_percentile: float | None = None,
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.num_bootstrap_heads = num_bootstrap_heads
        self.priority_cap_percentile = priority_cap_percentile
        self.priority_history: deque[float] = deque(maxlen=100_000)
        self.obs = np.empty((capacity, S.OBS_SIZE), dtype=np.float16)
        self.next_obs = np.empty((capacity, S.OBS_SIZE), dtype=np.float16)
        self.actions = np.empty((capacity, len(S.HEADS)), dtype=np.int16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.bootstrap_masks = np.ones(
            (capacity, num_bootstrap_heads), dtype=np.uint8
        )
        self.behaviours = np.full(capacity, "unknown", dtype=object)
        self.excursions = np.zeros(capacity, dtype=bool)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, transition: NStepTransition) -> int:
        index = self.position
        self.obs[index] = transition.obs
        self.next_obs[index] = transition.next_obs
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.discounts[index] = transition.discount
        if transition.bootstrap_mask is None:
            self.bootstrap_masks[index] = 1
        else:
            mask = np.asarray(transition.bootstrap_mask, dtype=np.uint8)
            if mask.shape != (self.num_bootstrap_heads,):
                raise ValueError(
                    "bootstrap mask shape "
                    f"{mask.shape} != ({self.num_bootstrap_heads},)"
                )
            self.bootstrap_masks[index] = mask
        self.behaviours[index] = transition.behaviour
        self.excursions[index] = transition.excursion
        if self.size:
            self.priorities[index] = self.priorities[: self.size].max()
        else:
            self.priorities[index] = 1.0
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
        beta: float = 0.4,
    ) -> ReplayBatch:
        if not self.size:
            raise ValueError("cannot sample empty replay")
        if self.prioritized:
            scaled = self.priorities[: self.size].astype(np.float64)
            probabilities = scaled / scaled.sum()
            indices = rng.choice(self.size, size=batch_size, p=probabilities)
            weights = (self.size * probabilities[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = rng.integers(self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float64)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.long, device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            discounts=torch.as_tensor(self.discounts[indices], device=device),
            weights=torch.as_tensor(weights, dtype=torch.float32, device=device),
            indices=np.asarray(indices),
            bootstrap_masks=torch.as_tensor(
                self.bootstrap_masks[indices], dtype=torch.bool, device=device
            ),
            excursions=torch.as_tensor(
                self.excursions[indices], dtype=torch.bool, device=device
            ),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update sampled priorities, retaining the largest value for duplicates."""
        if not self.prioritized:
            return
        valid_priorities = [
            max(float(priority), 1e-6)
            for index, priority in zip(indices, priorities)
            if int(index) >= 0
        ]
        self.priority_history.extend(valid_priorities)
        cap = None
        if self.priority_cap_percentile is not None and self.priority_history:
            cap = float(
                np.percentile(self.priority_history, self.priority_cap_percentile)
            )
        updates: dict[int, float] = {}
        for index, priority in zip(indices, priorities):
            key = int(index)
            if key < 0:
                continue
            value = max(float(priority), 1e-6)
            if cap is not None:
                value = min(value, cap)
            updates[key] = max(updates.get(key, 0.0), value)
        for index, priority in updates.items():
            self.priorities[index] = priority

    def boost_priorities(self, indices: Sequence[int], multiplier: float) -> int:
        """Multiply each referenced stored priority once and return the count."""
        if not self.prioritized or multiplier <= 1.0:
            return 0
        unique_indices = {
            int(index) for index in indices if 0 <= int(index) < self.size
        }
        for index in unique_indices:
            self.priorities[index] *= multiplier
        return len(unique_indices)


def per_priorities_from_deltas(
    td_deltas: torch.Tensor,
    alpha: float,
    optimism: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute proportional replay priorities from signed TD errors."""
    optimism_factors = torch.where(
        td_deltas > 0,
        torch.as_tensor(optimism, dtype=td_deltas.dtype, device=td_deltas.device),
        torch.ones((), dtype=td_deltas.dtype, device=td_deltas.device),
    )
    return (td_deltas.abs() * optimism_factors).pow(alpha) + epsilon


def _concatenate_batches(first: ReplayBatch, second: ReplayBatch) -> ReplayBatch:
    assert first.weights is not None and second.weights is not None
    assert first.indices is not None and second.indices is not None
    assert first.bootstrap_masks is not None and second.bootstrap_masks is not None
    return ReplayBatch(
        obs=torch.cat((first.obs, second.obs)),
        actions=torch.cat((first.actions, second.actions)),
        rewards=torch.cat((first.rewards, second.rewards)),
        next_obs=torch.cat((first.next_obs, second.next_obs)),
        discounts=torch.cat((first.discounts, second.discounts)),
        weights=torch.cat((first.weights, second.weights)),
        indices=np.concatenate((first.indices, second.indices)),
        bootstrap_masks=torch.cat((first.bootstrap_masks, second.bootstrap_masks)),
        excursions=(
            torch.cat((first.excursions, second.excursions))
            if first.excursions is not None and second.excursions is not None
            else None
        ),
    )


def sample_training_batch(
    replay: ReplayBuffer,
    archive: MilestoneArchive,
    batch_size: int,
    archive_fraction: float,
    rng: np.random.Generator,
    device: torch.device,
    beta: float,
    archive_mode: str = "uniform",
) -> ReplayBatch:
    archive_size = 0
    if archive.transition_count:
        archive_size = min(batch_size - 1, round(batch_size * archive_fraction))
    replay_batch = replay.sample(batch_size - archive_size, rng, device, beta=beta)
    if not archive_size:
        return replay_batch
    archive_batch = archive.sample(
        archive_size,
        rng,
        device,
        replay.num_bootstrap_heads,
        mode=archive_mode,
    )
    return _concatenate_batches(replay_batch, archive_batch)


def compute_td_loss(
    online: BranchingDQN,
    target: BranchingDQN,
    batch: ReplayBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Double-DQN Huber loss and the batch mean greedy Q."""
    loss, mean_max_q, _ = compute_td_loss_details(online, target, batch)
    return loss, mean_max_q


def compute_td_loss_details(
    online: BranchingDQN,
    target: BranchingDQN,
    batch: ReplayBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted Double-DQN loss, greedy Q, and signed TD errors."""
    value, advantages = online(batch.obs)
    chosen_q = complete_action_q(value, advantages, batch.actions)
    with torch.no_grad():
        next_online_value, next_online_advantages = online(batch.next_obs)
        next_actions = greedy_actions_from_outputs(
            next_online_value,
            next_online_advantages,
            observation_masks(batch.next_obs),
        )
        next_target_value, next_target_advantages = target(batch.next_obs)
        next_q = complete_action_q(next_target_value, next_target_advantages, next_actions)
        td_target = batch.rewards + batch.discounts * next_q
        greedy = greedy_actions_from_outputs(value, advantages, observation_masks(batch.obs))
        mean_max_q = complete_action_q(value, advantages, greedy).mean()
    per_sample_loss = F.smooth_l1_loss(chosen_q, td_target, reduction="none")
    weights = batch.weights
    if weights is None:
        weights = torch.ones_like(per_sample_loss)
    loss = (weights * per_sample_loss).mean()
    return loss, mean_max_q, td_target - chosen_q.detach()


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Return the per-sample QR-DQN Huber quantile regression loss."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    delta = target[:, None, :] - predicted[:, :, None]
    absolute_delta = delta.abs()
    huber = torch.where(
        absolute_delta <= kappa,
        0.5 * delta.square(),
        kappa * (absolute_delta - 0.5 * kappa),
    )
    count = predicted.shape[1]
    taus = (
        (torch.arange(count, device=predicted.device, dtype=predicted.dtype) + 0.5)
        / count
    ).view(1, count, 1)
    quantile_weights = (taus - (delta.detach() < 0).to(predicted.dtype)).abs()
    return (quantile_weights * huber / kappa).mean(dim=(1, 2))


def compute_quantile_td_loss(
    online: QuantileBranchingDQN,
    target: QuantileBranchingDQN,
    batch: ReplayBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return QR Double-DQN loss, expected greedy Q, and signed TD errors."""
    value, advantages = online(batch.obs)
    chosen_quantiles = complete_action_quantiles(value, advantages, batch.actions)
    with torch.no_grad():
        next_mean_value, next_mean_advantages = online.expected_forward(
            batch.next_obs
        )
        next_actions = greedy_actions_from_outputs(
            next_mean_value,
            next_mean_advantages,
            observation_masks(batch.next_obs),
        )
        next_target_value, next_target_advantages = target(batch.next_obs)
        next_quantiles = complete_action_quantiles(
            next_target_value, next_target_advantages, next_actions
        )
        target_quantiles = (
            batch.rewards[:, None] + batch.discounts[:, None] * next_quantiles
        )
        mean_value, mean_advantages = quantile_mean_outputs(value, advantages)
        greedy = greedy_actions_from_outputs(
            mean_value, mean_advantages, observation_masks(batch.obs)
        )
        mean_max_q = complete_action_quantiles(
            value, advantages, greedy
        ).mean(dim=-1).mean()
    per_sample_loss = quantile_huber_loss(chosen_quantiles, target_quantiles)
    weights = batch.weights
    if weights is None:
        weights = torch.ones_like(per_sample_loss)
    loss = (weights * per_sample_loss).mean()
    td_deltas = target_quantiles.mean(dim=-1) - chosen_quantiles.detach().mean(
        dim=-1
    )
    return loss, mean_max_q, td_deltas


def compute_bootstrapped_td_loss(
    online: BootstrappedDQN,
    target: BootstrappedDQN,
    batch: ReplayBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute masked per-head Double-DQN losses and signed TD errors."""
    online_outputs = online.learned_outputs(batch.obs)
    with torch.no_grad():
        next_online_outputs = online.learned_outputs(batch.next_obs)
        next_target_outputs = target.learned_outputs(batch.next_obs)
    if batch.bootstrap_masks is None:
        bootstrap_masks = torch.ones(
            (len(batch.obs), online.num_heads), dtype=torch.bool, device=batch.obs.device
        )
    else:
        bootstrap_masks = batch.bootstrap_masks
    if bootstrap_masks.shape[1] != online.num_heads:
        raise ValueError("replay bootstrap mask width does not match network heads")
    weights = batch.weights
    if weights is None:
        weights = torch.ones(len(batch.obs), device=batch.obs.device)

    losses = []
    max_qs = []
    signed_errors = []
    obs_masks = observation_masks(batch.obs)
    next_masks = observation_masks(batch.next_obs)
    for head_index in range(online.num_heads):
        value, advantages = online_outputs[head_index]
        chosen_q = complete_action_q(value, advantages, batch.actions)
        with torch.no_grad():
            next_online_value, next_online_advantages = next_online_outputs[head_index]
            next_actions = greedy_actions_from_outputs(
                next_online_value, next_online_advantages, next_masks
            )
            next_target_value, next_target_advantages = next_target_outputs[head_index]
            next_q = complete_action_q(
                next_target_value, next_target_advantages, next_actions
            )
            td_target = batch.rewards + batch.discounts * next_q
            greedy = greedy_actions_from_outputs(value, advantages, obs_masks)
            max_qs.append(complete_action_q(value, advantages, greedy).mean())
        per_sample_loss = F.smooth_l1_loss(chosen_q, td_target, reduction="none")
        active = bootstrap_masks[:, head_index].float()
        weighted_active = weights * active
        losses.append(
            (weighted_active * per_sample_loss).sum() / weighted_active.sum().clamp_min(1.0)
        )
        signed_errors.append(td_target - chosen_q.detach())

    head_losses = torch.stack(losses)
    error_matrix = torch.stack(signed_errors, dim=1)
    masked_errors = error_matrix.abs().masked_fill(~bootstrap_masks, 0.0)
    active_counts = bootstrap_masks.sum(dim=1).clamp_min(1)
    mean_absolute_errors = masked_errors.sum(dim=1) / active_counts
    mean_signed_errors = (
        error_matrix.masked_fill(~bootstrap_masks, 0.0).sum(dim=1) / active_counts
    )
    td_deltas = torch.where(
        mean_signed_errors >= 0, mean_absolute_errors, -mean_absolute_errors
    )
    return head_losses.mean(), torch.stack(max_qs), td_deltas, head_losses


@dataclass(frozen=True)
class VectorStep:
    env_indices: np.ndarray
    action_rows: np.ndarray
    transition_obs: np.ndarray
    rollout_obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: list[dict]


class ThreadVectorEnv:
    """Minimal thread-backed vector environment for network-bound live envs."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], object]],
        labels: Sequence[str] | None = None,
        timeout_seconds: float = ENV_TIMEOUT_SECONDS,
    ):
        self.envs = [env_fn() for env_fn in env_fns]
        self.labels = list(labels) if labels is not None else [str(i) for i in range(len(self.envs))]
        self.timeout_seconds = timeout_seconds
        self.alive = np.ones(len(self.envs), dtype=bool)
        self.dead_events: list[dict] = []
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.envs), thread_name_prefix="fle-dqn-env"
        )

    def reset(self, seed: int) -> np.ndarray:
        observations = np.zeros((len(self.envs), S.OBS_SIZE), dtype=np.float32)
        futures = {
            index: self.executor.submit(env.reset, seed=seed + index)
            for index, env in enumerate(self.envs)
        }
        deadline = time.monotonic() + self.timeout_seconds
        for index, future in futures.items():
            try:
                observations[index] = future.result(
                    timeout=max(0.0, deadline - time.monotonic())
                )[0]
            except FutureTimeout:
                self._mark_dead(index, "reset", "timeout")
            except Exception as exc:  # noqa: BLE001 - isolate failed live workers
                self._mark_dead(index, "reset", repr(exc))
        if not self.alive.any():
            raise RuntimeError("all environments are dead after reset")
        return observations

    def _mark_dead(self, index: int, phase: str, error: str) -> None:
        if not self.alive[index]:
            return
        self.alive[index] = False
        event = {
            "event": "env_dead",
            "env": self.labels[index],
            "phase": phase,
            "error": error,
            "timeout_seconds": self.timeout_seconds,
        }
        self.dead_events.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)

    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.alive)

    def step(
        self,
        actions: np.ndarray,
        count: int | None = None,
        env_indices: np.ndarray | None = None,
        auto_reset: bool = True,
    ) -> VectorStep:
        if env_indices is None:
            active = self.active_indices()
            count = len(active) if count is None else min(count, len(active))
            env_indices = active[:count]
        futures = {
            (row, int(index)): self.executor.submit(self.envs[int(index)].step, actions[row])
            for row, index in enumerate(env_indices)
        }
        deadline = time.monotonic() + self.timeout_seconds
        results = []
        for (action_row, index), future in futures.items():
            try:
                next_obs, reward, terminated, truncated, info = future.result(
                    timeout=max(0.0, deadline - time.monotonic())
                )
            except FutureTimeout:
                self._mark_dead(index, "step", "timeout")
                continue
            except Exception as exc:  # noqa: BLE001 - isolate failed live workers
                self._mark_dead(index, "step", repr(exc))
                continue
            done = terminated or truncated
            rollout_obs = next_obs
            if done and auto_reset:
                reset_future = self.executor.submit(self.envs[index].reset)
                try:
                    rollout_obs = reset_future.result(timeout=self.timeout_seconds)[0]
                except FutureTimeout:
                    self._mark_dead(index, "reset", "timeout")
                except Exception as exc:  # noqa: BLE001 - isolate failed live workers
                    self._mark_dead(index, "reset", repr(exc))
            results.append(
                (index, action_row, next_obs, rollout_obs, reward, done, info)
            )
        if not results:
            empty_obs = np.empty((0, S.OBS_SIZE), dtype=np.float32)
            return VectorStep(
                env_indices=np.empty(0, dtype=np.int64),
                action_rows=np.empty(0, dtype=np.int64),
                transition_obs=empty_obs,
                rollout_obs=empty_obs.copy(),
                rewards=np.empty(0, dtype=np.float32),
                dones=np.empty(0, dtype=bool),
                infos=[],
            )
        return VectorStep(
            env_indices=np.asarray([result[0] for result in results], dtype=np.int64),
            action_rows=np.asarray([result[1] for result in results], dtype=np.int64),
            transition_obs=np.stack([result[2] for result in results]),
            rollout_obs=np.stack([result[3] for result in results]),
            rewards=np.asarray([result[4] for result in results], dtype=np.float32),
            dones=np.asarray([result[5] for result in results], dtype=bool),
            infos=[result[6] for result in results],
        )

    def reset_one(
        self, index: int, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset one worker, allowing the trainer to choose a frontier first."""
        future = self.executor.submit(self.envs[index].reset, options=options)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(
                f"environment {self.labels[index]} reset timed out"
            ) from exc

    def close(self) -> None:
        for index, env in enumerate(self.envs):
            if self.alive[index]:
                env.close()
        self.executor.shutdown(wait=False, cancel_futures=True)


def linear_epsilon(
    step: int,
    total_steps: int,
    start: float = 1.0,
    end: float = 0.1,
    decay_steps: int | None = None,
) -> float:
    duration = max(1, decay_steps if decay_steps is not None else int(total_steps * 0.3))
    fraction = min(step / duration, 1.0)
    return start + fraction * (end - start)


def linear_beta(step: int, total_steps: int, beta0: float = 0.4) -> float:
    """Anneal prioritized-replay importance correction from 0.4 to 1.0."""
    return beta0 + (1.0 - beta0) * min(step / max(total_steps, 1), 1.0)


def linear_ucb_coefficient(
    step: int,
    start: float,
    end: float | None,
    decay_steps: int | None,
) -> float:
    """Linearly decay the behavior-only UCB coefficient when configured."""
    if end is None or decay_steps is None:
        return start
    fraction = min(step / max(decay_steps, 1), 1.0)
    return start + fraction * (end - start)


def _has_working_mining_drill(obs: np.ndarray) -> bool:
    rows = _row_block(obs, "entities", S.N_ENTITY_SLOTS, S.ENTITY_FEATURES)
    class_column = S.ENTITY_CLASSES.index("mining-drill")
    status_column = len(S.ENTITY_CLASSES) + 4 + S.STATUS_GROUPS.index("working")
    return bool(np.any((rows[:, class_column] > 0.5) & (rows[:, status_column] > 0.5)))


def _raw_ore_inventory_increased(before: np.ndarray, after: np.ndarray) -> bool:
    start, _ = S.OBS_LAYOUT["inventory"]
    for item in ("iron-ore", "copper-ore", "stone", "coal"):
        index = S.ITEM_INDEX[item]
        if after[start + index] > before[start + index] + 1e-8:
            return True
    return False


def record_episode_counters(
    milestones: EpisodeMilestones,
    automated_ore: bool,
    excursion: bool,
    reward: float,
    root_counts: Counter,
    conversion_counts: Counter,
    excursion_counts: Counter,
    root_rewards: deque[float],
) -> None:
    """Record comparable root outcomes separately from restored excursions."""
    if excursion:
        if milestones.first_attainment[4] is not None:
            excursion_counts["drill_placed"] += 1
        if milestones.first_attainment[5] is not None:
            excursion_counts["drill_fuelled"] += 1
        if automated_ore:
            excursion_counts["automated_ore_produced"] += 1
        return

    root_counts["episodes"] += 1
    if milestones.reached[4]:
        root_counts["drill_placed"] += 1
    if milestones.reached[5]:
        root_counts["drill_fuelled"] += 1
    if automated_ore:
        root_counts["automated_ore_produced"] += 1
    if milestones.reached[1]:
        conversion_counts["plates"] += 1
    if milestones.reached[2]:
        conversion_counts["gears"] += 1
    if milestones.reached[3]:
        conversion_counts["drills"] += 1
    if milestones.reached[1] and milestones.reached[2]:
        conversion_counts["plates_to_gear"] += 1
    if milestones.reached[2] and milestones.reached[3]:
        conversion_counts["gear_to_drill"] += 1
    if milestones.reached[3] and milestones.reached[4]:
        conversion_counts["drill_to_placed"] += 1
    root_rewards.append(reward)


def evaluate_random(episodes: int, seed: int) -> float:
    env = FakeMacroEnv(seed=seed)
    rng = np.random.default_rng(seed)
    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            action = S.random_valid_action(obs, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def evaluate_greedy(
    network: DQNNetwork | BootstrappedDQN,
    episodes: int,
    seed: int,
    device: torch.device,
) -> float:
    envs = [FakeMacroEnv(seed=seed + episode) for episode in range(episodes)]
    observations = np.stack(
        [env.reset(seed=seed + episode)[0] for episode, env in enumerate(envs)]
    )
    rewards = np.zeros(episodes, dtype=np.float64)
    done = np.zeros(episodes, dtype=bool)
    network.eval()
    has_noisy_layers = any(
        isinstance(module, NoisyLinear) for module in network.modules()
    )
    set_network_noise_enabled(network, False)
    while not done.all():
        active = np.flatnonzero(~done)
        with torch.no_grad():
            tensor = torch.as_tensor(observations[active], dtype=torch.float32, device=device)
            actions = greedy_policy_actions(network, tensor).cpu().numpy()
        for row, episode in enumerate(active):
            obs, reward, terminated, truncated, _ = envs[episode].step(actions[row])
            observations[episode] = obs
            rewards[episode] += reward
            done[episode] = terminated or truncated
    network.train()
    if has_noisy_layers:
        set_network_noise_enabled(network, True)
    for env in envs:
        env.close()
    return float(np.mean(rewards))


def save_checkpoint(
    path: Path,
    online: DQNNetwork | BootstrappedDQN,
    target: DQNNetwork | BootstrappedDQN,
    optimizer: torch.optim.Optimizer,
    env_steps: int,
    updates: int,
    explorer: UCBExplorer | None = None,
    archive: MilestoneArchive | None = None,
    behaviour_steps: dict[str, int] | None = None,
    checkpoint_args: Mapping[str, Any] | None = None,
) -> None:
    is_bootstrapped = isinstance(online, BootstrappedDQN)
    torch.save(
        {
            "online": online.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "env_steps": env_steps,
            "updates": updates,
            "schema_version": S.SCHEMA_VERSION,
            "head_sizes": {head: S.HEAD_SIZES[head] for head in S.HEADS},
            "algo": "bootdqn" if is_bootstrapped else "dqn",
            "heads": online.num_heads if is_bootstrapped else 1,
            "prior_scale": online.prior_scale if is_bootstrapped else 0.0,
            "dist": getattr(online, "dist", "none"),
            "quantiles": getattr(online, "num_quantiles", 1),
            "noisy": bool(getattr(online, "noisy", False)),
            "ucb_state": explorer.state_dict() if explorer is not None else None,
            "archive": archive.state_dict() if archive is not None else None,
            "behaviour_steps": behaviour_steps,
            "args": dict(checkpoint_args or {}),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    online: DQNNetwork | BootstrappedDQN,
    target: DQNNetwork | BootstrappedDQN,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    explorer: UCBExplorer | None = None,
    archive: MilestoneArchive | None = None,
    behaviour_steps: dict[str, int] | None = None,
    checkpoint_args: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Restore networks, optimizer, and counters after validating the schema."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != S.SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema {checkpoint.get('schema_version')!r} != {S.SCHEMA_VERSION!r}"
        )
    expected_sizes = {head: S.HEAD_SIZES[head] for head in S.HEADS}
    saved_sizes = checkpoint.get("head_sizes", expected_sizes)
    if saved_sizes != expected_sizes:
        raise ValueError("checkpoint action heads do not match the runtime schema")
    expected_algo = "bootdqn" if isinstance(online, BootstrappedDQN) else "dqn"
    if checkpoint.get("algo", "dqn") != expected_algo:
        raise ValueError("checkpoint algorithm does not match --algo")
    if isinstance(online, BootstrappedDQN):
        if checkpoint.get("heads") != online.num_heads:
            raise ValueError("checkpoint head count does not match --heads")
        if checkpoint.get("prior_scale") != online.prior_scale:
            raise ValueError("checkpoint prior scale does not match --prior-scale")
    else:
        expected_dist = getattr(online, "dist", "none")
        expected_quantiles = getattr(online, "num_quantiles", 1)
        expected_noisy = bool(getattr(online, "noisy", False))
        if checkpoint.get("dist", "none") != expected_dist:
            raise ValueError("checkpoint distribution does not match --dist")
        if int(checkpoint.get("quantiles", 1)) != expected_quantiles:
            raise ValueError("checkpoint quantile count does not match --quantiles")
        if bool(checkpoint.get("noisy", False)) != expected_noisy:
            raise ValueError("checkpoint noisy setting does not match --noisy")
    online.load_state_dict(checkpoint["online"])
    target.load_state_dict(checkpoint["target"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    missing = []
    if checkpoint.get("ucb_state") is None:
        missing.append("ucb_counts")
    elif explorer is not None:
        explorer.load_state_dict(checkpoint["ucb_state"])
    if checkpoint.get("archive") is None:
        missing.append("milestone_archive")
    elif archive is not None:
        archive.load_state_dict(checkpoint["archive"])
    saved_behaviour_steps = checkpoint.get("behaviour_steps")
    if behaviour_steps is not None:
        if isinstance(saved_behaviour_steps, dict):
            behaviour_steps.update(
                {
                    str(label): max(0, int(step))
                    for label, step in saved_behaviour_steps.items()
                }
            )
        else:
            missing.append("behaviour_steps")
    if checkpoint_args is not None:
        checkpoint_args.update(checkpoint.get("args", {}))
    if missing:
        print(
            "resume_initialized_fresh missing=" + ",".join(missing),
            flush=True,
        )
    return int(checkpoint["env_steps"]), int(checkpoint["updates"])


def policy_from_checkpoint(
    path: str | Path,
    device: torch.device,
    noisy_eval: bool = False,
) -> tuple[DQNNetwork | BootstrappedDQN, dict]:
    """Construct an evaluation network from checkpoint metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != S.SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema {checkpoint.get('schema_version')!r} != {S.SCHEMA_VERSION!r}"
        )
    expected_sizes = {head: S.HEAD_SIZES[head] for head in S.HEADS}
    if checkpoint.get("head_sizes", expected_sizes) != expected_sizes:
        raise ValueError("checkpoint action heads do not match the runtime schema")
    algo = checkpoint.get("algo", "dqn")
    if algo == "bootdqn":
        network: DQNNetwork | BootstrappedDQN = BootstrappedDQN(
            num_heads=int(checkpoint["heads"]),
            prior_scale=float(checkpoint["prior_scale"]),
        )
    elif algo == "dqn":
        dist = str(checkpoint.get("dist", "none"))
        noisy = bool(checkpoint.get("noisy", False))
        if dist == "qr":
            network = QuantileBranchingDQN(
                int(checkpoint.get("quantiles", 51)), noisy=noisy
            )
        elif dist == "none" and noisy:
            network = NoisyBranchingDQN()
        elif dist == "none":
            network = BranchingDQN()
        else:
            raise ValueError(f"unknown checkpoint distribution {dist!r}")
    else:
        raise ValueError(f"unknown checkpoint algorithm {algo!r}")
    network.load_state_dict(checkpoint["online"])
    network.to(device).eval()
    set_network_noise_enabled(network, noisy_eval)
    if noisy_eval:
        reset_network_noise(network)
    return network, checkpoint


def _make_env_fns(args: argparse.Namespace) -> list[Callable[[], object]]:
    ports = [int(port) for port in args.ports.split(",") if port.strip()]
    count = max(1, len(ports))
    max_steps = (
        [None] * count
        if args.max_steps_by_port is None
        else [
            int(value.strip())
            for value in args.max_steps_by_port.split(",")
            if value.strip()
        ]
    )
    if args.fake:
        return [
            lambda index=index, steps=steps: FakeMacroEnv(
                seed=args.seed + index,
                regime=args.action_regime,
                **({} if steps is None else {"horizon": steps}),
            )
            for index, steps in enumerate(max_steps)
        ]
    if not ports:
        raise ValueError("--ports must name at least one RCON port unless --fake is used")
    from fle.rl.env import (
        FleMacroEnv,  # Lazy: importing this may load live-server clients.
    )

    out = Path(args.out) if args.out else Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    return [
        lambda port=port, steps=steps: FleMacroEnv(
            port=port,
            speed=40,
            log_path=out / f"env_{port}.jsonl",
            clamp_quantity=args.clamp_quantity,
            clamp_item=args.clamp_item,
            clamp_anchor=args.clamp_anchor,
            clamp_connect=args.clamp_connect,
            support_cooldowns=args.support_cooldowns,
            regime=args.action_regime,
            **({} if steps is None else {"max_steps": steps}),
        )
        for port, steps in zip(ports, max_steps)
    ]


def _env_labels(args: argparse.Namespace, count: int) -> list[str]:
    ports = [port.strip() for port in args.ports.split(",") if port.strip()]
    if not ports:
        return [str(index) for index in range(count)]
    return [ports[index] if index < len(ports) else str(index) for index in range(count)]


def _behaviours_by_env(args: argparse.Namespace, count: int) -> list[str]:
    if args.behaviour_by_port is None:
        return [args.explore] * count
    behaviours = [
        value.strip() for value in args.behaviour_by_port.split(",") if value.strip()
    ]
    if len(behaviours) != count:
        raise ValueError(
            f"--behaviour-by-port has {len(behaviours)} entries for {count} environments"
        )
    return behaviours


def train(
    args: argparse.Namespace,
) -> tuple[DQNNetwork | BootstrappedDQN, float | None]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    return_rung = (
        None
        if args.return_rung == "drill_admitted"
        else RUNG_NAMES.index(args.return_rung)
    )

    out = Path(args.out) if args.out else Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    print(
        "train_start "
        + json.dumps(
            {"regime": args.action_regime, "schema_version": S.SCHEMA_VERSION},
            sort_keys=True,
        ),
        flush=True,
    )
    metrics_path = out / "metrics.jsonl"
    rng = np.random.default_rng(args.seed)
    baseline = evaluate_random(100, args.seed + 10_000) if args.fake else None
    if baseline is not None:
        print(f"random_baseline episodes=100 mean_episode_reward={baseline:.6f}", flush=True)

    if args.algo == "bootdqn":
        online: DQNNetwork | BootstrappedDQN = BootstrappedDQN(
            args.heads, args.prior_scale
        ).to(device)
        target: DQNNetwork | BootstrappedDQN = BootstrappedDQN(
            args.heads, args.prior_scale
        ).to(device)
    elif args.dist == "qr":
        online = QuantileBranchingDQN(args.quantiles, noisy=args.noisy).to(device)
        target = QuantileBranchingDQN(args.quantiles, noisy=args.noisy).to(device)
    elif args.noisy:
        online = NoisyBranchingDQN().to(device)
        target = NoisyBranchingDQN().to(device)
    else:
        online = BranchingDQN().to(device)
        target = BranchingDQN().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    replay = ReplayBuffer(
        capacity=args.replay_size,
        prioritized=args.per,
        alpha=args.per_alpha,
        num_bootstrap_heads=args.heads if args.algo == "bootdqn" else 1,
        priority_cap_percentile=args.per_priority_cap_pct,
    )
    explorer = UCBExplorer(args.ucb_c, args.ucb_key)
    archive = MilestoneArchive()
    env_fns = _make_env_fns(args)
    labels = _env_labels(args, len(env_fns))
    behaviours = _behaviours_by_env(args, len(env_fns))
    restored_behaviour_steps: dict[str, int] = {}
    updates = 0
    env_steps = 0
    if args.resume:
        resumed_args: dict[str, Any] = {}
        env_steps, updates = load_checkpoint(
            args.resume,
            online,
            target,
            optimizer,
            device,
            explorer=explorer,
            archive=archive,
            behaviour_steps=restored_behaviour_steps,
            checkpoint_args=resumed_args,
        )
        saved_regime = resumed_args.get("action_regime")
        if saved_regime is not None and saved_regime != args.action_regime:
            raise ValueError(
                "checkpoint action regime does not match --action-regime: "
                f"{saved_regime!r} != {args.action_regime!r}"
            )
        print(
            f"resumed checkpoint={args.resume} env_steps={env_steps} updates={updates}",
            flush=True,
        )
        if args.frontier_return:
            print(
                "resume_frontier_return_disabled_until_new_snapshot",
                flush=True,
            )
    envs = ThreadVectorEnv(env_fns, labels=labels)
    accumulators = [
        NStepAccumulator(n=args.n_step, gamma=args.gamma) for _ in envs.envs
    ]
    try:
        observations = envs.reset(args.seed)
    except Exception:
        envs.close()
        raise
    episode_rewards = np.zeros(len(envs.envs), dtype=np.float64)
    episode_ops = [Counter() for _ in envs.envs]
    episode_transitions: list[list[NStepTransition]] = [[] for _ in envs.envs]
    episode_replay_indices: list[list[int]] = [[] for _ in envs.envs]
    milestones = [EpisodeMilestones() for _ in envs.envs]
    latest_frontier: ArchivedFrontier | None = None
    episode_frontiers: list[ArchivedFrontier | None] = [None for _ in envs.envs]
    episode_excursions = np.zeros(len(envs.envs), dtype=bool)
    drill_frontier_seen = np.zeros(len(envs.envs), dtype=bool)
    force_drill_first_action = np.zeros(len(envs.envs), dtype=bool)
    frontier_force_remaining = np.zeros(len(envs.envs), dtype=np.int64)
    frontier_force_trials: list[set[tuple]] = [set() for _ in envs.envs]
    rejected_pairs: list[tuple[int, str, int] | None] = [None for _ in envs.envs]
    episode_indices = np.zeros(len(envs.envs), dtype=np.int64)
    action_steps = np.zeros(len(envs.envs), dtype=np.int64)
    episode_steps = np.zeros(len(envs.envs), dtype=np.int64)
    ucb_env_count = sum(behaviour == "ucb" for behaviour in behaviours)
    legacy_ucb_steps = explorer.total // max(ucb_env_count, 1)
    behaviour_steps = np.asarray(
        [
            restored_behaviour_steps.get(
                label,
                legacy_ucb_steps if args.resume and behaviour == "ucb" else 0,
            )
            for label, behaviour in zip(labels, behaviours)
        ],
        dtype=np.int64,
    )
    latest_epsilons = np.zeros(len(envs.envs), dtype=np.float64)
    latest_ucb_coefficients = np.zeros(len(envs.envs), dtype=np.float64)
    episode_automated_ore = np.zeros(len(envs.envs), dtype=bool)
    recent_episode_rewards: deque[float] = deque(maxlen=100)
    finished_since_log: list[dict] = []
    window_ops: Counter = Counter()
    losses: deque[float] = deque(maxlen=100)
    max_qs: deque[float] = deque(maxlen=100)
    positive_delta_priorities: list[float] = []
    negative_delta_priorities: list[float] = []
    positive_delta_samples = 0
    per_samples = 0
    per_head_max_qs = [deque(maxlen=100) for _ in range(args.heads)]
    per_head_losses = [deque(maxlen=100) for _ in range(args.heads)]
    acting_heads = (
        rng.integers(args.heads, size=len(envs.envs))
        if args.algo == "bootdqn"
        else np.zeros(len(envs.envs), dtype=np.int64)
    )
    acting_head_counts: Counter = Counter()
    transition_provenance: Counter = Counter()
    root_episode_counts: Counter = Counter()
    excursion_episode_counts: Counter = Counter()
    conversion_counts: Counter = Counter()
    invalid = 0
    eligible = 0
    started = time.monotonic()
    action_files = []
    if args.log_actions:
        action_files = [
            (out / f"actions_{label}.jsonl").open("a", buffering=1)
            for label in labels
        ]

    try:
        metrics_mode = "a" if args.resume else "w"
        with metrics_path.open(metrics_mode) as metrics_file:
            while env_steps < args.total_steps:
                active = envs.active_indices()
                if not active.size:
                    save_checkpoint(
                        out / "checkpoint-all-envs-dead.pt",
                        online,
                        target,
                        optimizer,
                        env_steps,
                        updates,
                        explorer,
                        archive,
                        {
                            label: int(behaviour_steps[index])
                            for index, label in enumerate(labels)
                        },
                        checkpoint_args=vars(args),
                    )
                    raise RuntimeError("all environments are dead")
                count = min(len(active), args.total_steps - env_steps)
                selected_envs = active[:count]
                selected_obs = observations[selected_envs]
                if args.frontier_return and args.return_rung == "drill_admitted":
                    for row, env_index in enumerate(selected_envs):
                        if drill_frontier_seen[env_index] or not drill_recipe_admitted(
                            selected_obs[row]
                        ):
                            continue
                        drill_frontier_seen[env_index] = True
                        try:
                            state = envs.envs[env_index].snapshot()
                            frontier = ArchivedFrontier(
                                state=state,
                                restore_hash=str(state.entity_hash),
                                rung=milestones[env_index].deepest,
                                step=int(state.step_index),
                            )
                            latest_frontier = frontier
                            episode_frontiers[env_index] = frontier
                        except Exception as exc:  # noqa: BLE001 - disable return only
                            latest_frontier = None
                            print(
                                json.dumps(
                                    {
                                        "event": "frontier_snapshot_failed",
                                        "env": labels[env_index],
                                        "error": repr(exc),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                exploration_flags = np.zeros(count, dtype=bool)
                if args.noisy:
                    reset_network_noise(online)
                if args.behaviour_by_port is not None:
                    actions = np.zeros((count, len(S.HEADS)), dtype=np.int64)
                    diagnostics: list[dict] = [{} for _ in range(count)]
                    for row, env_index in enumerate(selected_envs):
                        behaviour = behaviours[env_index]
                        if behaviour == "ucb":
                            assert not isinstance(online, BootstrappedDQN)
                            epsilon = args.eps_floor
                            coefficient = linear_ucb_coefficient(
                                int(behaviour_steps[env_index]),
                                args.ucb_c,
                                args.ucb_c_end,
                                args.ucb_c_decay_steps,
                            )
                            row_actions, row_diagnostics = explorer.select(
                                online,
                                selected_obs[row : row + 1],
                                epsilon,
                                rng,
                                device,
                                rungs=np.asarray([milestones[env_index].deepest]),
                                coefficient=coefficient,
                                forced_arg_eps=args.forced_arg_eps,
                                retry_pairs=[rejected_pairs[env_index]],
                                frontier_forcing=[
                                    frontier_force_remaining[env_index] > 0
                                ],
                                frontier_tried_actions=[
                                    frontier_force_trials[env_index]
                                ],
                                frontier_force_max_count=args.frontier_force_max_count,
                            )
                            actions[row] = row_actions[0]
                            diagnostics[row] = row_diagnostics[0]
                            exploration_flags[row] = bool(
                                diagnostics[row]["exploratory"]
                            )
                            latest_ucb_coefficients[env_index] = coefficient
                        else:
                            epsilon = (
                                args.eps_floor
                                if args.noisy
                                else linear_epsilon(
                                    int(behaviour_steps[env_index]),
                                    args.total_steps,
                                    args.eps_start,
                                    args.eps_end,
                                    args.eps_decay_steps,
                                )
                            )
                            if args.algo == "bootdqn":
                                assert isinstance(online, BootstrappedDQN)
                                row_actions = bootstrapped_actions(
                                    online,
                                    selected_obs[row : row + 1],
                                    acting_heads[env_index : env_index + 1],
                                    epsilon,
                                    rng,
                                    device,
                                    exploration_flags[row : row + 1],
                                )
                                row_heads = acting_heads[env_index : env_index + 1]
                            else:
                                assert not isinstance(online, BootstrappedDQN)
                                row_actions = epsilon_greedy_actions(
                                    online,
                                    selected_obs[row : row + 1],
                                    epsilon,
                                    rng,
                                    device,
                                    exploration_flags[row : row + 1],
                                )
                                row_heads = None
                            actions[row] = row_actions[0]
                            if args.log_actions:
                                diagnostics[row] = action_diagnostics(
                                    online,
                                    selected_obs[row : row + 1],
                                    row_actions,
                                    exploration_flags[row : row + 1],
                                    device,
                                    row_heads,
                                )[0]
                            latest_epsilons[env_index] = epsilon
                elif args.algo == "bootdqn":
                    assert isinstance(online, BootstrappedDQN)
                    epsilon = args.eps_floor
                    latest_epsilons[selected_envs] = epsilon
                    actions = bootstrapped_actions(
                        online,
                        selected_obs,
                        acting_heads[selected_envs],
                        epsilon,
                        rng,
                        device,
                        exploration_flags,
                    )
                    diagnostics = (
                        action_diagnostics(
                            online,
                            selected_obs,
                            actions,
                            exploration_flags,
                            device,
                            acting_heads[selected_envs],
                        )
                        if args.log_actions
                        else [{} for _ in range(count)]
                    )
                elif args.explore == "ucb":
                    assert not isinstance(online, BootstrappedDQN)
                    epsilon = args.eps_floor
                    coefficient = linear_ucb_coefficient(
                        env_steps,
                        args.ucb_c,
                        args.ucb_c_end,
                        args.ucb_c_decay_steps,
                    )
                    latest_ucb_coefficients[selected_envs] = coefficient
                    actions, diagnostics = explorer.select(
                        online,
                        selected_obs,
                        epsilon,
                        rng,
                        device,
                        rungs=np.asarray(
                            [milestones[index].deepest for index in selected_envs]
                        ),
                        coefficient=coefficient,
                        forced_arg_eps=args.forced_arg_eps,
                        retry_pairs=[rejected_pairs[index] for index in selected_envs],
                        frontier_forcing=[
                            frontier_force_remaining[index] > 0
                            for index in selected_envs
                        ],
                        frontier_tried_actions=[
                            frontier_force_trials[index] for index in selected_envs
                        ],
                        frontier_force_max_count=args.frontier_force_max_count,
                    )
                else:
                    assert not isinstance(online, BootstrappedDQN)
                    epsilon = (
                        args.eps_floor
                        if args.noisy
                        else linear_epsilon(
                            env_steps,
                            args.total_steps,
                            args.eps_start,
                            args.eps_end,
                            args.eps_decay_steps,
                        )
                    )
                    latest_epsilons[selected_envs] = epsilon
                    actions = epsilon_greedy_actions(
                        online,
                        selected_obs,
                        epsilon,
                        rng,
                        device,
                        exploration_flags,
                    )
                    diagnostics = (
                        action_diagnostics(
                            online,
                            selected_obs,
                            actions,
                            exploration_flags,
                            device,
                        )
                        if args.log_actions
                        else [{} for _ in range(count)]
                    )
                for row, env_index in enumerate(selected_envs):
                    if not force_drill_first_action[env_index]:
                        continue
                    forced = forced_drill_action(
                        selected_obs[row], actions[row], rng
                    )
                    if behaviours[env_index] == "ucb":
                        diagnostics[row] = explorer.replace_last_selection(
                            diagnostics[row],
                            selected_obs[row],
                            forced,
                            milestones[env_index].deepest,
                        )
                        exploration_flags[row] = True
                    actions[row] = forced
                    force_drill_first_action[env_index] = False
                behaviour_steps[selected_envs] += 1
                old_observations = selected_obs.copy()
                vector_step = envs.step(
                    actions,
                    env_indices=selected_envs,
                    auto_reset=not args.frontier_return,
                )
                observations[vector_step.env_indices] = vector_step.rollout_obs

                if args.log_actions:
                    for row, env_index in enumerate(selected_envs):
                        action_steps[env_index] += 1
                        diagnostic = diagnostics[row]
                        op_name = str(diagnostic["op"])
                        read_heads = ("op", *S.OP_HEADS[op_name])
                        masks = S.split_masks(selected_obs[row])
                        admitted_indices = {
                            head: [int(index) for index in np.flatnonzero(masks[head])]
                            for head in read_heads
                        }
                        chosen_indices = {
                            head: int(actions[row, S.HEADS.index(head)])
                            for head in read_heads
                        }
                        action_files[env_index].write(
                            json.dumps(
                                {
                                    "step": int(action_steps[env_index]),
                                    "episode": int(episode_indices[env_index]),
                                    "behaviour": behaviours[env_index],
                                    "exploratory": bool(diagnostic["exploratory"]),
                                    "chosen": {
                                        "op": diagnostic["op"],
                                        "primary": diagnostic["primary"],
                                    },
                                    "chosen_indices": chosen_indices,
                                    "admitted_indices": admitted_indices,
                                    "standardized_q": diagnostic["standardized_q"],
                                    "ucb_bonus": diagnostic.get("ucb_bonus"),
                                    "ucb_c": diagnostic.get("ucb_c"),
                                    "rung": diagnostic.get(
                                        "rung", milestones[env_index].deepest
                                    ),
                                    "acting_head": diagnostic.get("acting_head"),
                                    "n": diagnostic.get("n"),
                                    "N": diagnostic.get("N"),
                                    "secondary_heads_sampled": bool(
                                        diagnostic.get(
                                            "secondary_heads_sampled", False
                                        )
                                    ),
                                    "frontier_forced": bool(
                                        diagnostic.get("frontier_forced", False)
                                    ),
                                    "admitted_pair_count": diagnostic.get(
                                        "admitted_pair_count"
                                    ),
                                    "n0_pair_count": diagnostic.get(
                                        "n0_pair_count"
                                    ),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )

                for result_row, env_index in enumerate(vector_step.env_indices):
                    action_row = int(vector_step.action_rows[result_row])
                    env_index = int(env_index)
                    env_steps += 1
                    reward = float(vector_step.rewards[result_row])
                    done = bool(vector_step.dones[result_row])
                    info = vector_step.infos[result_row]
                    action = actions[action_row]
                    old_observation = old_observations[action_row]
                    details = action_details(old_observation, action)
                    op_column = S.HEADS.index("op")
                    op_name = str(
                        info.get("op", S.OPS[int(action[op_column])])
                    )
                    status = str(info.get("status", ""))
                    behaviour = behaviours[env_index]
                    if behaviour == "ucb":
                        explorer.reconcile_executed_quantity(
                            diagnostics[action_row], info.get("executed_quantity")
                        )
                    is_excursion = bool(info.get("excursion", False))
                    episode_excursions[env_index] |= is_excursion
                    transition_provenance[behaviour] += 1
                    if is_excursion:
                        transition_provenance["excursion"] += 1
                    episode_head = int(acting_heads[env_index])
                    if args.algo == "bootdqn":
                        acting_head_counts[episode_head] += 1
                    episode_rewards[env_index] += reward
                    episode_ops[env_index][op_name] += 1
                    window_ops[op_name] += 1
                    if op_name != "WAIT":
                        eligible += 1
                        invalid += status in {"tool_rejected", "no_effect"}

                    previous_rung = milestones[env_index].deepest
                    current_rung = milestones[env_index].observe(
                        details, status, int(episode_steps[env_index])
                    )
                    frontier_trial = diagnostics[action_row].get("frontier_trial")
                    trial_was_new = bool(
                        frontier_trial is not None
                        and frontier_trial not in frontier_force_trials[env_index]
                    )
                    if trial_was_new:
                        frontier_force_trials[env_index].add(frontier_trial)
                    reset_force_window = bool(
                        args.frontier_force_tries > 0
                        and current_rung > previous_rung
                        and current_rung >= DRILL_CRAFTED_RUNG
                    )
                    if reset_force_window:
                        frontier_force_trials[env_index].clear()
                    frontier_force_remaining[env_index] = (
                        advance_frontier_force_window(
                            int(frontier_force_remaining[env_index]),
                            previous_rung,
                            current_rung,
                            (
                                args.frontier_force_tries
                                if behaviour == "ucb"
                                else 0
                            ),
                            trial_was_new=trial_was_new,
                        )
                    )
                    if (
                        args.frontier_return
                        and (
                            (
                                return_rung is not None
                                and is_new_return_frontier(
                                    previous_rung, current_rung, return_rung
                                )
                            )
                            or (
                                args.return_rung == "drill_admitted"
                                and status == "ok"
                                and str(details["op"]) == "CRAFT"
                                and str(details["primary"]) == DRILL_RECIPE
                                and previous_rung < DRILL_CRAFTED_RUNG
                                and current_rung >= DRILL_CRAFTED_RUNG
                            )
                        )
                    ):
                        try:
                            state = envs.envs[env_index].snapshot()
                            frontier = ArchivedFrontier(
                                state=state,
                                restore_hash=str(state.entity_hash),
                                rung=current_rung,
                                step=int(state.step_index),
                            )
                            latest_frontier = frontier
                            episode_frontiers[env_index] = frontier
                        except Exception as exc:  # noqa: BLE001 - disable return only
                            latest_frontier = None
                            print(
                                json.dumps(
                                    {
                                        "event": "frontier_snapshot_failed",
                                        "env": labels[env_index],
                                        "error": repr(exc),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                    rejected_pairs[env_index] = (
                        (
                            current_rung,
                            str(details["op"]),
                            int(details["primary_id"]),
                        )
                        if status == "tool_rejected"
                        else None
                    )
                    transition_observation = vector_step.transition_obs[result_row]
                    episode_automated_ore[env_index] |= (
                        milestones[env_index].reached[4]
                        and _has_working_mining_drill(transition_observation)
                    ) or (
                        op_name != "HARVEST"
                        and _raw_ore_inventory_increased(
                            old_observation, transition_observation
                        )
                    )
                    episode_steps[env_index] += 1
                    emitted = accumulators[env_index].append(
                        old_observation,
                        action,
                        reward,
                        transition_observation,
                        done,
                        (
                            rng.random(args.heads) < 0.5
                            if args.algo == "bootdqn"
                            else None
                        ),
                        behaviour,
                        is_excursion,
                    )
                    for transition in emitted:
                        replay_index = replay.add(transition)
                        episode_replay_indices[env_index].append(replay_index)
                        episode_transitions[env_index].append(transition)

                    if done:
                        total_reward = float(episode_rewards[env_index])
                        final_aps = float(info.get("score_automated", 0.0))
                        priority_multiplier = 1.0 + (
                            args.per_episode_return_bonus
                            * float(np.clip(final_aps / 1000.0, 0.0, 1.0))
                        )
                        boosted_transitions = replay.boost_priorities(
                            episode_replay_indices[env_index], priority_multiplier
                        )
                        episode_max_steps = int(
                            getattr(
                                envs.envs[env_index],
                                "_episode_max_steps",
                                getattr(envs.envs[env_index], "max_steps", 0),
                            )
                        )
                        archive.add_episode(
                            episode_transitions[env_index],
                            milestones[env_index].deepest,
                            final_aps,
                            (
                                milestones[env_index].first_attainment[
                                    milestones[env_index].deepest
                                ]
                                if milestones[env_index].deepest >= 0
                                else None
                            ),
                            frontier=episode_frontiers[env_index],
                            comparable_aps=not episode_excursions[env_index],
                        )
                        record_episode_counters(
                            milestones[env_index],
                            bool(episode_automated_ore[env_index]),
                            bool(episode_excursions[env_index]),
                            total_reward,
                            root_episode_counts,
                            conversion_counts,
                            excursion_episode_counts,
                            recent_episode_rewards,
                        )
                        finished_since_log.append(
                            {
                                "reward": total_reward,
                                "score_automated": final_aps,
                                "score_player": float(info.get("score_player", 0.0)),
                                "per_boosted_transitions": boosted_transitions,
                                "op_histogram": dict(episode_ops[env_index]),
                                "acting_head": (
                                    episode_head if args.algo == "bootdqn" else None
                                ),
                                "behaviour": behaviour,
                                "regime": args.action_regime,
                                "excursion": bool(episode_excursions[env_index]),
                                "restore_hash": info.get("restore_hash"),
                                "max_steps": episode_max_steps,
                                "automated_ore_produced": bool(
                                    episode_automated_ore[env_index]
                                ),
                                "deepest_rung": (
                                    RUNG_NAMES[milestones[env_index].deepest]
                                    if milestones[env_index].deepest >= 0
                                    else None
                                ),
                            }
                        )
                        episode_rewards[env_index] = 0.0
                        episode_ops[env_index].clear()
                        episode_transitions[env_index] = []
                        episode_replay_indices[env_index] = []
                        episode_frontiers[env_index] = None
                        milestones[env_index] = EpisodeMilestones()
                        episode_steps[env_index] = 0
                        episode_automated_ore[env_index] = False
                        episode_excursions[env_index] = False
                        drill_frontier_seen[env_index] = False
                        force_drill_first_action[env_index] = False
                        frontier_force_remaining[env_index] = 0
                        frontier_force_trials[env_index].clear()
                        rejected_pairs[env_index] = None
                        episode_indices[env_index] += 1
                        if args.algo == "bootdqn":
                            acting_heads[env_index] = int(rng.integers(args.heads))

                        if args.frontier_return:
                            frontier = latest_frontier
                            use_frontier = (
                                behaviours[env_index] == "ucb"
                                and frontier is not None
                                and frontier.state is not None
                                and frontier.step < int(frontier.state.max_steps)
                                and rng.random() < args.return_prob
                            )
                            reset_options = (
                                {
                                    "restore": frontier.state,
                                    "budget": args.return_budget,
                                }
                                if use_frontier
                                else None
                            )
                            try:
                                reset_obs, reset_info = envs.reset_one(
                                    env_index, reset_options
                                )
                            except Exception as exc:  # noqa: BLE001 - fallback to root
                                if not use_frontier:
                                    envs._mark_dead(env_index, "reset", repr(exc))
                                    continue
                                latest_frontier = None
                                print(
                                    json.dumps(
                                        {
                                            "event": "frontier_return_disabled",
                                            "env": labels[env_index],
                                            "error": repr(exc),
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                                try:
                                    reset_obs, reset_info = envs.reset_one(env_index)
                                except Exception as root_exc:  # noqa: BLE001
                                    envs._mark_dead(
                                        env_index, "reset", repr(root_exc)
                                    )
                                    continue
                                use_frontier = False
                            observations[env_index] = reset_obs
                            if use_frontier:
                                assert frontier is not None
                                milestones[env_index] = EpisodeMilestones.from_frontier(
                                    frontier.rung
                                )
                                episode_excursions[env_index] = bool(
                                    reset_info.get("excursion", True)
                                )
                                drill_frontier_seen[env_index] = True
                                force_drill_first_action[env_index] = (
                                    args.return_rung == "drill_admitted"
                                )
                                excursion_episode_counts["run"] += 1

                    if (
                        env_steps >= 500
                        and replay.size >= args.batch_size
                        and env_steps % args.update_every == 0
                    ):
                        batch = sample_training_batch(
                            replay,
                            archive,
                            args.batch_size,
                            args.archive_frac,
                            rng,
                            device,
                            beta=linear_beta(
                                env_steps, args.total_steps, args.per_beta0
                            ),
                            archive_mode=args.archive_mode,
                        )
                        if args.noisy:
                            reset_network_noise(online)
                            reset_network_noise(target)
                        if args.algo == "bootdqn":
                            assert isinstance(online, BootstrappedDQN)
                            assert isinstance(target, BootstrappedDQN)
                            loss, head_max_q, td_deltas, head_loss = (
                                compute_bootstrapped_td_loss(online, target, batch)
                            )
                            mean_max_q = head_max_q.mean()
                            for head_index in range(args.heads):
                                per_head_max_qs[head_index].append(
                                    float(head_max_q[head_index])
                                )
                                per_head_losses[head_index].append(
                                    float(head_loss[head_index].detach())
                                )
                        elif args.dist == "qr":
                            assert isinstance(online, QuantileBranchingDQN)
                            assert isinstance(target, QuantileBranchingDQN)
                            loss, mean_max_q, td_deltas = compute_quantile_td_loss(
                                online, target, batch
                            )
                        else:
                            assert not isinstance(online, BootstrappedDQN)
                            assert not isinstance(target, BootstrappedDQN)
                            loss, mean_max_q, td_deltas = compute_td_loss_details(
                                online, target, batch
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        if args.per:
                            assert batch.indices is not None
                            priorities = per_priorities_from_deltas(
                                td_deltas, args.per_alpha, args.per_optimism
                            )
                            replay.update_priorities(
                                batch.indices, priorities.cpu().numpy()
                            )
                            positive = td_deltas > 0
                            negative = td_deltas < 0
                            positive_delta_priorities.extend(
                                priorities[positive].detach().cpu().tolist()
                            )
                            negative_delta_priorities.extend(
                                priorities[negative].detach().cpu().tolist()
                            )
                            positive_delta_samples += int(positive.sum())
                            per_samples += len(td_deltas)
                        updates += 1
                        losses.append(float(loss.detach()))
                        max_qs.append(float(mean_max_q))
                        if updates % args.target_period == 0:
                            target.load_state_dict(online.state_dict())

                    if env_steps % 100 == 0:
                        record = {
                            "env_steps": env_steps,
                            "regime": args.action_regime,
                            "wall": time.monotonic() - started,
                            "epsilon": (
                                None
                                if args.behaviour_by_port is not None
                                else (
                                    args.eps_floor
                                    if args.algo == "bootdqn"
                                    or args.explore == "ucb"
                                    or args.noisy
                                    else linear_epsilon(
                                        env_steps,
                                        args.total_steps,
                                        args.eps_start,
                                        args.eps_end,
                                        args.eps_decay_steps,
                                    )
                                )
                            ),
                            "behaviour_by_port": {
                                label: {
                                    "behaviour": behaviours[index],
                                    "steps": int(behaviour_steps[index]),
                                    "epsilon": (
                                        float(latest_epsilons[index])
                                        if behaviours[index] == "epsilon"
                                        else args.eps_floor
                                    ),
                                    "ucb_c": (
                                        float(latest_ucb_coefficients[index])
                                        if behaviours[index] == "ucb"
                                        else None
                                    ),
                                }
                                for index, label in enumerate(labels)
                            },
                            "transition_provenance": dict(transition_provenance),
                            "per_beta": (
                                linear_beta(
                                    env_steps, args.total_steps, args.per_beta0
                                )
                                if args.per
                                else None
                            ),
                            "per_positive_delta_mean_priority": (
                                float(np.mean(positive_delta_priorities))
                                if positive_delta_priorities
                                else None
                            ),
                            "per_negative_delta_mean_priority": (
                                float(np.mean(negative_delta_priorities))
                                if negative_delta_priorities
                                else None
                            ),
                            "per_positive_delta_fraction": (
                                positive_delta_samples / per_samples
                                if per_samples
                                else None
                            ),
                            "td_loss": float(np.mean(losses)) if losses else None,
                            "mean_max_q": float(np.mean(max_qs)) if max_qs else None,
                            "mean_episode_reward": (
                                float(np.mean(recent_episode_rewards))
                                if recent_episode_rewards
                                else None
                            ),
                            "episodes": finished_since_log,
                            "op_histogram": dict(window_ops),
                            "invalid_combination_rate": invalid / eligible if eligible else 0.0,
                            "acting_head_histogram": dict(acting_head_counts),
                            "per_head_mean_max_q": (
                                [
                                    float(np.mean(values)) if values else None
                                    for values in per_head_max_qs
                                ]
                                if args.algo == "bootdqn"
                                else None
                            ),
                            "per_head_td_loss": (
                                [
                                    float(np.mean(values)) if values else None
                                    for values in per_head_losses
                                ]
                                if args.algo == "bootdqn"
                                else None
                            ),
                            "mean_abs_sigma": noisy_sigma_means(online),
                            "ucb_pairs": len(explorer.counts),
                            "ucb_total": explorer.total,
                            "ucb_key": explorer.key_mode,
                            "archive_sizes": archive.sizes(),
                            "archive_draws": archive.consume_draw_counts(),
                            "root_episode_counts": {
                                "episodes": root_episode_counts["episodes"],
                                "drill_placed": root_episode_counts[
                                    "drill_placed"
                                ],
                                "drill_fuelled": root_episode_counts[
                                    "drill_fuelled"
                                ],
                                "automated_ore_produced": root_episode_counts[
                                    "automated_ore_produced"
                                ],
                            },
                            "excursion_episode_counts": {
                                "run": excursion_episode_counts["run"],
                                "drill_placed": excursion_episode_counts[
                                    "drill_placed"
                                ],
                                "drill_fuelled": excursion_episode_counts[
                                    "drill_fuelled"
                                ],
                                "automated_ore_produced": excursion_episode_counts[
                                    "automated_ore_produced"
                                ],
                            },
                            "conversion_counts": {
                                "plates": conversion_counts["plates"],
                                "gears": conversion_counts["gears"],
                                "drills": conversion_counts["drills"],
                                "plates_to_gear": conversion_counts[
                                    "plates_to_gear"
                                ],
                                "gear_to_drill": conversion_counts[
                                    "gear_to_drill"
                                ],
                                "drill_to_placed": conversion_counts[
                                    "drill_to_placed"
                                ],
                            },
                            "dead_envs": envs.dead_events,
                        }
                        metrics_file.write(json.dumps(record) + "\n")
                        metrics_file.flush()
                        finished_since_log = []
                        window_ops.clear()
                        acting_head_counts.clear()
                        transition_provenance.clear()
                        positive_delta_priorities.clear()
                        negative_delta_priorities.clear()
                        positive_delta_samples = 0
                        per_samples = 0

                    if env_steps % 2000 == 0:
                        save_checkpoint(
                            out / f"checkpoint-{env_steps}.pt",
                            online,
                            target,
                            optimizer,
                            env_steps,
                            updates,
                            explorer,
                            archive,
                            {
                                label: int(behaviour_steps[index])
                                for index, label in enumerate(labels)
                            },
                            checkpoint_args=vars(args),
                        )
    finally:
        envs.close()
        for action_file in action_files:
            action_file.close()

    save_checkpoint(
        out / "checkpoint-final.pt",
        online,
        target,
        optimizer,
        env_steps,
        updates,
        explorer,
        archive,
        {
            label: int(behaviour_steps[index])
            for index, label in enumerate(labels)
        },
        checkpoint_args=vars(args),
    )

    trained = evaluate_greedy(online, 100, args.seed + 20_000, device) if args.fake else None
    if baseline is not None and trained is not None:
        print(
            "acceptance "
            f"baseline_mean_episode_reward={baseline:.6f} "
            f"trained_mean_episode_reward={trained:.6f} "
            f"episodes=100 steps={env_steps}",
            flush=True,
        )
    return online, trained


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", default="27002")
    parser.add_argument("--action-regime", choices=("macro", "bare"), default="macro")
    parser.add_argument("--algo", choices=("dqn", "bootdqn"), default="dqn")
    parser.add_argument("--explore", choices=("epsilon", "ucb"), default="epsilon")
    parser.add_argument("--ucb-c", type=float, default=1.5)
    parser.add_argument("--forced-arg-eps", type=float, default=0.0)
    parser.add_argument("--frontier-force-tries", type=int, default=0)
    parser.add_argument("--frontier-force-max-count", type=int, default=2)
    parser.add_argument("--ucb-key", choices=("global", "rung"), default="global")
    parser.add_argument("--ucb-c-end", type=float)
    parser.add_argument("--ucb-c-decay-steps", type=int)
    parser.add_argument("--behaviour-by-port")
    parser.add_argument("--max-steps-by-port")
    per_group = parser.add_mutually_exclusive_group()
    per_group.add_argument("--per", dest="per", action="store_true")
    per_group.add_argument("--no-per", dest="per", action="store_false")
    parser.set_defaults(per=False)
    parser.add_argument("--dist", choices=("none", "qr"), default="none")
    parser.add_argument("--quantiles", type=int, default=51)
    noisy_group = parser.add_mutually_exclusive_group()
    noisy_group.add_argument("--noisy", dest="noisy", action="store_true")
    noisy_group.add_argument("--no-noisy", dest="noisy", action="store_false")
    parser.set_defaults(noisy=False)
    parser.add_argument("--rainbow", action="store_true")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--eps-floor", type=float, default=0.02)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.1)
    parser.add_argument("--eps-decay-steps", type=int)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-period", type=int, default=1000)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta0", "--per-beta", dest="per_beta0", type=float, default=0.4)
    parser.add_argument("--per-optimism", type=float, default=1.0)
    parser.add_argument("--per-episode-return-bonus", type=float, default=0.0)
    parser.add_argument("--per-priority-cap-pct", type=float)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--archive-frac", type=float, default=0.25)
    parser.add_argument(
        "--archive-mode", choices=("uniform", "frontier"), default="uniform"
    )
    parser.add_argument("--frontier-return", action="store_true")
    parser.add_argument(
        "--clamp-quantity",
        "--quantity-aware-support",
        dest="clamp_quantity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clamp-item",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--clamp-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--clamp-connect",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--support-cooldowns",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--return-prob", type=float, default=0.5)
    parser.add_argument("--return-budget", type=int, default=64)
    parser.add_argument(
        "--return-rung",
        choices=("gears_crafted", "drill_admitted", "drill_crafted"),
        default="drill_crafted",
    )
    parser.add_argument("--log-actions", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--run-name", default="dqn-v0")
    parser.add_argument("--out")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(raw_argv)
    # Preserve the old parsed attribute for launch scripts and tests that read it.
    args.quantity_aware_support = args.clamp_quantity

    def explicitly_set(*flags: str) -> bool:
        return any(
            argument == flag or argument.startswith(flag + "=")
            for argument in raw_argv
            for flag in flags
        )

    if args.rainbow:
        if not explicitly_set("--per", "--no-per"):
            args.per = True
        if not explicitly_set("--dist"):
            args.dist = "qr"
        if not explicitly_set("--quantiles"):
            args.quantiles = 51
        if not explicitly_set("--noisy", "--no-noisy"):
            args.noisy = True
        if not explicitly_set("--n-step"):
            args.n_step = 3
        if not explicitly_set("--gamma"):
            args.gamma = 0.99
        if not explicitly_set("--target-period"):
            args.target_period = 1000
        if not explicitly_set("--lr"):
            args.lr = 6.25e-5
        if not explicitly_set("--per-alpha"):
            args.per_alpha = 0.5
        if not explicitly_set("--per-beta0", "--per-beta"):
            args.per_beta0 = 0.4
    if args.heads <= 0:
        parser.error("--heads must be positive")
    if args.prior_scale < 0:
        parser.error("--prior-scale must be non-negative")
    if args.quantiles <= 0:
        parser.error("--quantiles must be positive")
    if not 0 <= args.eps_floor <= 1:
        parser.error("--eps-floor must be between 0 and 1")
    if not 0 <= args.eps_start <= 1 or not 0 <= args.eps_end <= 1:
        parser.error("--eps-start and --eps-end must be between 0 and 1")
    if args.eps_decay_steps is not None and args.eps_decay_steps <= 0:
        parser.error("--eps-decay-steps must be positive")
    if args.n_step <= 0 or args.lr <= 0:
        parser.error("--n-step and --lr must be positive")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be between 0 and 1")
    if min(
        args.target_period,
        args.replay_size,
        args.batch_size,
        args.update_every,
    ) <= 0:
        parser.error("period, replay, batch, and update sizes must be positive")
    if not 0 <= args.per_alpha <= 1 or not 0 <= args.per_beta0 <= 1:
        parser.error("PER alpha and beta must be between 0 and 1")
    if args.per_optimism < 1:
        parser.error("--per-optimism must be at least 1")
    if args.per_episode_return_bonus < 0:
        parser.error("--per-episode-return-bonus must be non-negative")
    if args.per_priority_cap_pct is not None and not (
        0 < args.per_priority_cap_pct <= 100
    ):
        parser.error("--per-priority-cap-pct must be in (0, 100]")
    if not 0 <= args.archive_frac < 1:
        parser.error("--archive-frac must be in [0, 1)")
    if not 0 <= args.return_prob <= 1:
        parser.error("--return-prob must be between 0 and 1")
    if not 0 <= args.forced_arg_eps <= 1:
        parser.error("--forced-arg-eps must be between 0 and 1")
    if args.frontier_force_tries < 0:
        parser.error("--frontier-force-tries must be non-negative")
    if args.frontier_force_max_count < 0:
        parser.error("--frontier-force-max-count must be non-negative")
    if args.return_budget <= 0:
        parser.error("--return-budget must be positive")
    if args.ucb_c < 0:
        parser.error("--ucb-c must be non-negative")
    if args.ucb_c_end is not None and args.ucb_c_end < 0:
        parser.error("--ucb-c-end must be non-negative")
    if (args.ucb_c_end is None) != (args.ucb_c_decay_steps is None):
        parser.error("--ucb-c-end and --ucb-c-decay-steps must be set together")
    if args.ucb_c_decay_steps is not None and args.ucb_c_decay_steps <= 0:
        parser.error("--ucb-c-decay-steps must be positive")
    assigned_behaviours = None
    ports = [port for port in args.ports.split(",") if port.strip()]
    expected_port_values = max(1, len(ports))
    if args.max_steps_by_port is not None:
        try:
            max_steps_values = [
                int(value.strip())
                for value in args.max_steps_by_port.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--max-steps-by-port entries must be integers")
        if len(max_steps_values) != expected_port_values:
            parser.error("--max-steps-by-port must have one entry per port")
        if any(value <= 0 for value in max_steps_values):
            parser.error("--max-steps-by-port entries must be positive")
    if args.behaviour_by_port is not None:
        assigned_behaviours = args.behaviour_by_port.split(",")
        if len(assigned_behaviours) != expected_port_values:
            parser.error("--behaviour-by-port must have one entry per port")
        if any(value not in {"epsilon", "ucb"} for value in assigned_behaviours):
            parser.error("--behaviour-by-port entries must be epsilon or ucb")
    uses_ucb = args.explore == "ucb" or (
        assigned_behaviours is not None and "ucb" in assigned_behaviours
    )
    if uses_ucb and args.algo != "dqn":
        parser.error("--explore ucb is only supported with --algo dqn")
    if args.algo != "dqn" and (args.dist != "none" or args.noisy):
        parser.error("--dist and --noisy are only supported with --algo dqn")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
