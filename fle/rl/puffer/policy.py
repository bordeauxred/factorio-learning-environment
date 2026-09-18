"""Masked multi-head actor-critic policy for PufferLib 3.0."""

from __future__ import annotations

import torch
from torch import nn

from fle.rl import schema as S
from fle.rl.nets import MacroEncoder


class MaskedMultiHeadPolicy(nn.Module):
    """Independent categorical heads with masks reconstructed from the input."""

    def __init__(self, env=None) -> None:
        super().__init__()
        if env is not None:
            observed = tuple(env.single_observation_space.shape)
            action_dims = tuple(int(v) for v in env.single_action_space.nvec)
            if observed != (S.OBS_SIZE,) or action_dims != S.HEAD_DIMS:
                raise ValueError("environment spaces do not match fle.rl.schema")

        self.encoder = MacroEncoder()
        self.action_heads = nn.ModuleDict(
            {head: nn.Linear(self.encoder.output_size, S.HEAD_SIZES[head]) for head in S.HEADS}
        )
        self.value_head = nn.Linear(self.encoder.output_size, 1)
        self._initialize_output_layers()

    def _initialize_output_layers(self) -> None:
        for layer in self.action_heads.values():
            nn.init.orthogonal_(layer.weight, 0.01)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward_eval(self, observations: torch.Tensor, state=None):
        return self.forward(observations, state)

    def forward(self, observations: torch.Tensor, state=None):
        del state
        hidden = self.encoder(observations)
        logits: list[torch.Tensor] = []
        for head in S.HEADS:
            start, end = S.MASK_OFFSETS[head]
            mask = observations[:, start:end] > 0.5
            # A head no enabled op reads may be entirely zero (spec: learner-env.md,
            # masks). Admit index 0 there so the Categorical stays finite; the
            # environment ignores that head for the chosen op.
            empty = ~mask.any(dim=-1)
            if bool(empty.any()):
                mask = mask.clone()
                mask[empty, 0] = True
            head_logits = self.action_heads[head](hidden)
            logits.append(head_logits.masked_fill(~mask, -1e8))
        return logits, self.value_head(hidden)

    @staticmethod
    def entropy_by_head(logits: list[torch.Tensor]) -> dict[str, float]:
        return {
            head: float(torch.distributions.Categorical(logits=head_logits).entropy().mean())
            for head, head_logits in zip(S.HEADS, logits, strict=True)
        }
