"""Neural network for semi-Markov autoregressive Q-learning.

The spatial decoder uses FiLM rather than concatenating a broadcast action
vector.  FiLM keeps the native-resolution activation small (eight channels),
which matters at a 288-tile raster, while still making every spatial score
conditional on the verb, prototype, and all other preceding arguments.

When ``coarse_move`` is enabled, MOVE_TO has a separate spatial decoder over
the retained 96x96 coarse grid.  It still records its choice in the frozen
POSITION action slot; exact POSITION decoding and every existing head size and
index are unchanged when the flag is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fle.smarq import contract as C


_CAT_COLUMNS = (4, 6, 18, 20, 22, 24, 26, 28, 30, 32, 34)
_ITEM_COLUMNS = (20, 22, 24, 26, 28, 30, 32, 34)
_NUMERIC_COLUMNS = tuple(i for i in range(C.ENTITY_FEATURES) if i not in _CAT_COLUMNS)
_COUNT_COLUMNS = (19, 21, 23, 25, 27, 29, 31, 33, 35, 36, 37)


def _as_mapping(observation: C.Observation | Mapping[str, Any]) -> Mapping[str, Any]:
    return observation.as_dict() if isinstance(observation, C.Observation) else observation


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


@dataclass
class EncodedState:
    """Features retained for all heads of one or more observations."""

    z_state: Tensor
    spatial: Tensor
    raster: Tensor
    entities: Tensor
    entity_mask: Tensor
    raster_tiles: int

    def select(self, index: int) -> "EncodedState":
        sl = slice(index, index + 1)
        return EncodedState(
            self.z_state[sl],
            self.spatial[sl],
            self.raster[sl],
            self.entities[sl],
            self.entity_mask[sl],
            self.raster_tiles,
        )

    def index_select(self, indices: Tensor) -> "EncodedState":
        return EncodedState(
            self.z_state.index_select(0, indices),
            self.spatial.index_select(0, indices),
            self.raster.index_select(0, indices),
            self.entities.index_select(0, indices),
            self.entity_mask.index_select(0, indices),
            self.raster_tiles,
        )

    def detached(self) -> "EncodedState":
        return EncodedState(
            self.z_state.detach(),
            self.spatial.detach(),
            self.raster.detach(),
            self.entities.detach(),
            self.entity_mask.detach(),
            self.raster_tiles,
        )


class SMARQNetwork(nn.Module):
    """Compact convolutional/entity encoder with autoregressive Q heads."""

    context_dim = 32

    def __init__(
        self,
        vocab: C.VocabProtocol,
        raster_tiles: int = C.RASTER_TILES_DEFAULT,
        *,
        entity_slots: int = 512,
        entity_id_capacity: int = 4096,
        state_dim: int = 256,
        coarse_move: bool = False,
    ) -> None:
        super().__init__()
        if raster_tiles <= 0:
            raise ValueError("raster_tiles must be positive")
        self.vocab = vocab
        self.raster_tiles = raster_tiles
        self.coarse_move = bool(coarse_move)
        if entity_slots <= 0 or entity_slots > C.ENTITY_SLOTS:
            raise ValueError(f"entity_slots must be in [1, {C.ENTITY_SLOTS}]")
        self.entity_slots = int(entity_slots)
        self.head_sizes = vocab.head_sizes(raster_tiles)
        self.head_sizes[C.ENTITY] = self.entity_slots

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(C.GRID_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.grid_pool = nn.Sequential(nn.Linear(128, 192), nn.ReLU())

        cap = max(32, entity_id_capacity)
        self.entity_type_embedding = nn.Embedding(cap, 8, padding_idx=0)
        self.entity_recipe_embedding = nn.Embedding(cap, 12, padding_idx=0)
        self.entity_fluid_embedding = nn.Embedding(cap, 8, padding_idx=0)
        self.entity_item_embedding = nn.Embedding(cap, 12, padding_idx=0)
        entity_in = len(_NUMERIC_COLUMNS) + 8 + 12 + 8 + 8 * 12
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_in, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
        )
        self.entity_pool = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.globals_encoder = nn.Sequential(
            nn.Linear(C.N_GLOBALS, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(192 + 256 + 64, 384),
            nn.ReLU(),
            nn.Linear(384, state_dim),
            nn.ReLU(),
        )

        self.verb_embedding = nn.Embedding(C.N_VERBS, self.context_dim)
        discrete_sizes = {
            C.PROTOTYPE: len(vocab.prototypes),
            C.DIRECTION: C.N_DIRECTIONS,
            C.ITEM: len(vocab.items),
            C.QUANTITY: C.N_QUANTITIES,
            C.RECIPE: len(vocab.recipes),
            C.TECHNOLOGY: len(vocab.technologies),
            C.DURATION: C.N_DURATIONS,
        }
        self.argument_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, self.context_dim) for name, size in discrete_sizes.items()}
        )
        self.position_embedding = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, self.context_dim)
        )
        self.entity_argument = nn.Linear(128, self.context_dim)

        # One fixed slot per possible prior decision makes the dependency on
        # every selected argument explicit, without recurrence.
        context_in = state_dim + self.context_dim * (1 + len(C.HEADS))
        self.context_encoder = nn.Sequential(
            nn.Linear(context_in, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU()
        )
        self.verb_head = nn.Sequential(
            nn.Linear(state_dim, 192), nn.ReLU(), nn.Linear(192, C.N_VERBS)
        )
        self.discrete_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, size))
                for name, size in discrete_sizes.items()
            }
        )

        self.spatial_grid = nn.Conv2d(128, 8, 1)
        self.raster_encoder = nn.Sequential(
            nn.Conv2d(C.RASTER_CHANNELS, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1)
        )
        self.spatial_film = nn.Linear(256, 16)
        self.spatial_out = nn.Sequential(
            nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1)
        )
        if self.coarse_move:
            self.move_spatial_film = nn.Linear(256, 16)
            self.move_spatial_out = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, 1),
            )
        self.pointer_context = nn.Linear(256, 128)
        self.pointer_entity = nn.Linear(128, 128)
        self.pointer_bias = nn.Linear(128, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _prepare_batch(
        self, observations: C.Observation | Mapping[str, Any] | Sequence[C.Observation | Mapping[str, Any]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if isinstance(observations, (C.Observation, Mapping)):
            obs_list = [observations]
        else:
            obs_list = list(observations)
        if not obs_list:
            raise ValueError("cannot encode an empty observation batch")
        maps = [_as_mapping(obs) for obs in obs_list]
        device = next(self.parameters()).device

        def stack(name: str, dtype: torch.dtype, limit: int | None = None) -> Tensor:
            values = [obs[name] for obs in maps]
            if all(isinstance(value, Tensor) for value in values):
                if limit is not None:
                    values = [value[:limit] for value in values]
                return torch.stack(values).to(device=device, dtype=dtype)
            # A single contiguous copy and device transfer is dramatically
            # cheaper on MPS than 128 individual as_tensor transfers.
            if limit is None:
                array = np.stack([np.asarray(value) for value in values])
            else:
                array = np.stack([np.asarray(value)[:limit] for value in values])
            return torch.from_numpy(array).to(device=device, dtype=dtype)

        grid = stack("grid", torch.float32)
        entities = stack("entity_view", torch.float32, self.entity_slots)
        entity_mask = stack("entity_mask", torch.bool, self.entity_slots)
        globals_ = stack("globals", torch.float32)
        raster = stack("raster", torch.float32)
        if raster.shape[-2:] != (self.raster_tiles, self.raster_tiles):
            raise ValueError(
                f"network raster is {self.raster_tiles}, observation is {tuple(raster.shape[-2:])}"
            )
        return grid, entities, entity_mask, globals_, raster

    def prepare_batch(
        self,
        observations: C.Observation
        | Mapping[str, Any]
        | Sequence[C.Observation | Mapping[str, Any]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Collate and transfer observations once for reuse by both networks."""
        return self._prepare_batch(observations)

    def _entity_features(self, raw: Tensor) -> Tensor:
        cap = self.entity_type_embedding.num_embeddings

        def indices(column: int) -> Tensor:
            return raw[..., column].round().long().clamp_(0, cap - 1)

        numeric = raw[..., list(_NUMERIC_COLUMNS)].clone()
        numeric_columns = {column: offset for offset, column in enumerate(_NUMERIC_COLUMNS)}
        # Coordinates are already player-relative. Signed log compression keeps
        # exact direction while limiting remote-entity scale.
        for column in (0, 1):
            offset = numeric_columns[column]
            value = numeric[..., offset]
            numeric[..., offset] = torch.sign(value) * torch.log1p(value.abs())
        for column in _COUNT_COLUMNS:
            if column in numeric_columns:
                offset = numeric_columns[column]
                numeric[..., offset] = torch.log1p(numeric[..., offset].clamp_min(0))
        # Joules span many orders of magnitude; progress is contractually [0,1].
        numeric[..., numeric_columns[7]] = torch.log1p(numeric[..., numeric_columns[7]].clamp_min(0))
        numeric[..., numeric_columns[8]] = numeric[..., numeric_columns[8]].clamp(0, 1)
        cats = [
            self.entity_type_embedding(indices(4)),
            self.entity_recipe_embedding(indices(6)),
            self.entity_fluid_embedding(indices(18)),
            *(self.entity_item_embedding(indices(column)) for column in _ITEM_COLUMNS),
        ]
        return torch.cat([numeric, *cats], dim=-1)

    def encode(
        self, observations: C.Observation | Mapping[str, Any] | Sequence[C.Observation | Mapping[str, Any]]
    ) -> EncodedState:
        return self.encode_prepared(self._prepare_batch(observations))

    def encode_prepared(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> EncodedState:
        grid, entity_raw, entity_mask, globals_, raster = batch
        spatial = self.grid_encoder(grid)
        z_grid = self.grid_pool(F.adaptive_avg_pool2d(spatial, 1).flatten(1))

        # Padding is the common case (early games often have <100 live rows in
        # a 512/2048-slot table). Avoid running the shared MLP and eleven
        # embedding lookups on empty rows; scatter preserves pointer slot IDs.
        live_entities = self.entity_mlp(self._entity_features(entity_raw[entity_mask]))
        entities = live_entities.new_zeros((*entity_raw.shape[:2], 128))
        entities[entity_mask] = live_entities
        mask_f = entity_mask.unsqueeze(-1).to(entities.dtype)
        count = mask_f.sum(dim=1).clamp_min(1)
        mean = (entities * mask_f).sum(dim=1) / count
        neg_inf = torch.finfo(entities.dtype).min
        maximum = entities.masked_fill(~entity_mask.unsqueeze(-1), neg_inf).max(dim=1).values
        maximum = torch.where(entity_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        z_entities = self.entity_pool(torch.cat([mean, maximum], dim=-1))
        z_globals = self.globals_encoder(globals_)
        z_state = self.state_encoder(torch.cat([z_grid, z_entities, z_globals], dim=-1))
        return EncodedState(z_state, spatial, raster, entities, entity_mask, self.raster_tiles)

    def forward(
        self, observations: C.Observation | Mapping[str, Any] | Sequence[C.Observation | Mapping[str, Any]]
    ) -> EncodedState:
        return self.encode(observations)

    def q_verbs(self, encoded: EncodedState) -> Tensor:
        return self.verb_head(encoded.z_state)

    def _selection_tensor(self, value: int | Tensor, batch: int, device: torch.device) -> Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=torch.long)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch)
        return tensor

    def action_context(
        self,
        encoded: EncodedState,
        verb: int | Tensor,
        selected: Mapping[str, int | Tensor] | None = None,
    ) -> Tensor:
        selected = selected or {}
        batch = encoded.z_state.shape[0]
        device = encoded.z_state.device
        verb_tensor = self._selection_tensor(verb, batch, device)
        slots = [self.verb_embedding(verb_tensor)]
        for head in C.HEADS:
            if head not in selected:
                slots.append(torch.zeros(batch, self.context_dim, device=device))
                continue
            index = self._selection_tensor(selected[head], batch, device)
            present = index >= 0
            safe_index = index.clamp_min(0)
            if head == C.POSITION:
                y = torch.div(safe_index, self.raster_tiles, rounding_mode="floor")
                x = safe_index.remainder(self.raster_tiles)
                denom = max(1, self.raster_tiles - 1)
                xy = torch.stack([x, y], dim=-1).float().div(denom).mul(2).sub(1)
                slots.append(self.position_embedding(xy) * present[:, None])
            elif head == C.ENTITY:
                safe = safe_index.clamp(0, encoded.entities.shape[1] - 1)
                row = encoded.entities[torch.arange(batch, device=device), safe]
                slots.append(self.entity_argument(row) * present[:, None])
            else:
                slots.append(self.argument_embeddings[head](safe_index) * present[:, None])
        return self.context_encoder(torch.cat([encoded.z_state, *slots], dim=-1))

    def q_head(
        self,
        encoded: EncodedState,
        head: str,
        verb: int | Tensor,
        selected: Mapping[str, int | Tensor] | None = None,
    ) -> Tensor:
        if head not in C.HEADS:
            raise KeyError(head)
        context = self.action_context(encoded, verb, selected)
        if head == C.POSITION:
            if self.coarse_move:
                verb_tensor = self._selection_tensor(
                    verb, encoded.z_state.shape[0], encoded.z_state.device
                )
                move_rows = verb_tensor == C.VERB_INDEX["MOVE_TO"]
                if move_rows.all():
                    grid = F.interpolate(
                        self.spatial_grid(encoded.spatial),
                        size=(C.GRID_SIZE, C.GRID_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )
                    scale, shift = self.move_spatial_film(context).chunk(2, dim=-1)
                    grid = grid * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
                    return self.move_spatial_out(grid).flatten(1)
                if move_rows.any():
                    raise ValueError("mixed MOVE_TO and exact POSITION batch")
            grid = F.interpolate(
                self.spatial_grid(encoded.spatial),
                size=(self.raster_tiles, self.raster_tiles),
                mode="bilinear",
                align_corners=False,
            )
            features = grid + self.raster_encoder(encoded.raster)
            scale, shift = self.spatial_film(context).chunk(2, dim=-1)
            features = features * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            return self.spatial_out(features).flatten(1)
        if head == C.ENTITY:
            query = self.pointer_context(context).unsqueeze(1)
            keys = self.pointer_entity(encoded.entities)
            return ((query * keys).sum(dim=-1) / (128**0.5)) + self.pointer_bias(
                encoded.entities
            ).squeeze(-1)
        return self.discrete_heads[head](context)


# A short alias is convenient in training scripts and keeps older experiments
# readable without creating a second implementation.
QNetwork = SMARQNetwork
