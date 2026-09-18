from __future__ import annotations

import numpy as np
import pytest
import torch

from fle.rl import schema as S
from fle.rl.fake_env import FakeMacroEnv
from fle.rl.puffer.policy import MaskedMultiHeadPolicy


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for offline tests."""
    yield


def test_policy_masks_every_head() -> None:
    observation, _ = FakeMacroEnv(seed=1).reset()
    policy = MaskedMultiHeadPolicy()
    logits, value = policy(torch.as_tensor(observation).unsqueeze(0))

    assert value.shape == (1, 1)
    for head, head_logits in zip(S.HEADS, logits, strict=True):
        start, end = S.MASK_OFFSETS[head]
        mask = observation[start:end] > 0.5
        assert torch.all(head_logits[0, ~torch.as_tensor(mask)] == -1e8)
        assert torch.all(torch.isfinite(head_logits))


def test_policy_uses_zero_fallback_for_fully_masked_unused_head() -> None:
    observation, _ = FakeMacroEnv(seed=2).reset()
    start, end = S.MASK_OFFSETS["target"]
    observation[start:end] = 0

    logits, _ = MaskedMultiHeadPolicy()(
        torch.as_tensor(observation).unsqueeze(0)
    )
    target_logits = logits[S.HEADS.index("target")][0]

    assert target_logits[0] > -1e8
    assert torch.all(target_logits[1:] == -1e8)


def test_encoder_batch_is_finite_with_empty_rows() -> None:
    observation = np.zeros((2, S.OBS_SIZE), dtype=np.float32)
    for start, _ in S.MASK_OFFSETS.values():
        observation[:, start] = 1
    policy = MaskedMultiHeadPolicy()
    logits, value = policy(torch.as_tensor(observation))
    loss = value.square().mean() + sum(head.square().mean() for head in logits)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
        for parameter in policy.parameters()
    )
