"""Shared neural-network encoders for macro-action learners."""

from __future__ import annotations

import torch
from torch import nn

from fle.rl import schema as S


def _mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.ReLU(),
    )


class MaskedRowEncoder(nn.Module):
    """Encode rows, then concatenate masked mean and max pooling.

    Keeping this primitive here lets both the legacy flat encoder and the V0
    structured encoder use the same padding-safe pooling semantics.
    """

    def __init__(self, features: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.output_size = hidden_size * 2
        self.row = _mlp(features, hidden_size, hidden_size)

    @staticmethod
    def pool(encoded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(encoded.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weights).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid.unsqueeze(-1), -1e8).amax(dim=1)
        maximum = torch.where(
            valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        return torch.cat((mean, maximum), dim=-1)

    def forward(self, rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.pool(self.row(rows), valid.bool())


class MacroEncoder(nn.Module):
    """Encode the structured blocks in a flat macro observation.

    The masks in the observation tail are deliberately excluded. Target and
    entity rows are pooled with their per-row validity flag, so padding cannot
    affect the representation.
    """

    output_size = 512

    def __init__(self) -> None:
        super().__init__()
        base_start = S.OBS_LAYOUT["globals"][0]
        base_end = S.OBS_LAYOUT["recipes_enabled"][1]
        self.base_slice = slice(base_start, base_end)
        self.target_slice = slice(*S.OBS_LAYOUT["targets"])
        self.entity_slice = slice(*S.OBS_LAYOUT["entities"])
        self.grid_slice = slice(*S.OBS_LAYOUT["grid"])

        self.base = _mlp(base_end - base_start, 256, 256)
        self.target_row = _mlp(S.TARGET_FEATURES, 64, 64)
        self.entity_row = _mlp(S.ENTITY_FEATURES, 64, 64)
        self.grid = nn.Sequential(
            nn.Conv2d(len(S.GRID_CHANNELS), 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
        )
        self.trunk = _mlp(256 + 128 + 128 + 128, 512, self.output_size)

    @staticmethod
    def _masked_pool(encoded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1.0)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(valid == 0, -1e8).amax(dim=1)
        any_valid = valid.any(dim=1)
        maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float()
        batch = observations.shape[0]

        base = self.base(observations[:, self.base_slice])

        target_rows = observations[:, self.target_slice].reshape(
            batch, S.N_TARGET_SLOTS, S.TARGET_FEATURES
        )
        targets = self.target_row(target_rows)
        targets = self._masked_pool(targets, target_rows[..., -2] > 0.5)

        entity_rows = observations[:, self.entity_slice].reshape(
            batch, S.N_ENTITY_SLOTS, S.ENTITY_FEATURES
        )
        entities = self.entity_row(entity_rows)
        entities = self._masked_pool(entities, entity_rows[..., -2] > 0.5)

        grid = observations[:, self.grid_slice].reshape(
            batch, len(S.GRID_CHANNELS), S.GRID_SIDE, S.GRID_SIDE
        )
        grid = self.grid(grid)
        return self.trunk(torch.cat((base, targets, entities, grid), dim=-1))
