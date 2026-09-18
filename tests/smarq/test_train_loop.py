from __future__ import annotations

import json

import numpy as np
import pytest

from fle.smarq.fake import FakeSemanticEnv
from fle.smarq.logging_ import EPISODE_FIELDS, METRIC_FIELDS, STEP_FIELDS
from fle.smarq.train import (
    FallbackRandomPolicy,
    TrainingConfig,
    load_checkpoint,
    run_training,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for offline tests."""
    yield


def _config(tmp_path, **changes) -> TrainingConfig:
    values = {
        "run_name": "test",
        "workers": 2,
        "replay_ratio": 4,
        "raster_tiles": 8,
        "episode_game_minutes": 0.2,
        "decision_cap": 3,
        "total_decisions": 8,
        "checkpoint_every": 0,
        "run_root": tmp_path,
        "replay_capacity": 6,
        "seed": 7,
        "force_fallback": True,
    }
    values.update(changes)
    return TrainingConfig(**values)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_training_loop_writes_complete_jsonl_and_respects_ratio(tmp_path):
    config = _config(tmp_path)
    result = run_training(config)
    assert result.decisions == 8
    assert result.updates == result.decisions * config.replay_ratio
    step_files = sorted(result.run_dir.glob("steps_*.jsonl"))
    steps = [record for path in step_files for record in _records(path)]
    episodes = _records(result.run_dir / "episodes.jsonl")
    metrics = _records(result.run_dir / "metrics.jsonl")
    assert len(steps) == result.decisions
    assert episodes and metrics
    assert set(STEP_FIELDS) <= steps[0].keys()
    assert set(EPISODE_FIELDS) <= episodes[0].keys()
    assert set(METRIC_FIELDS) <= metrics[0].keys()
    assert metrics[-1]["replay_ratio"] == 4


def test_checkpoint_restores_same_next_random_action(tmp_path):
    env = FakeSemanticEnv(raster_tiles=8, seed=11)
    observation, masks = env.reset(11)
    policy = FallbackRandomPolicy(env.vocab, seed=91)
    policy.select(observation, masks, 0)
    checkpoint = tmp_path / "policy.pkl"
    save_checkpoint(checkpoint, {"policy": policy.state_dict()})
    expected = policy.select(observation, masks, 1).action

    resumed = FallbackRandomPolicy(env.vocab, seed=0)
    resumed.load_state_dict(load_checkpoint(checkpoint)["policy"])
    actual = resumed.select(observation, masks, 1).action
    assert actual.verb == expected.verb
    assert np.array_equal(actual.heads, expected.heads)
    assert actual.quantity == expected.quantity
    assert actual.recipe == expected.recipe


def test_demonstrations_enter_training_replay(tmp_path):
    result = run_training(
        _config(tmp_path, run_name="demo", workers=1, demos=2, replay_capacity=32)
    )
    payload = load_checkpoint(result.checkpoint)
    replay = payload["backend_state"]["replay"]
    demo_flags = [transition.is_demo for transition in replay["transitions"]]
    assert result.demo_transitions > 0
    assert any(demo_flags)
