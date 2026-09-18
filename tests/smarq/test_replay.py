import numpy as np
import pytest

from fle.smarq import contract as C
from fle.smarq.fake import FakeSemanticEnv
from fle.smarq.replay import (
    DEFAULT_CAPACITY,
    PrioritizedReplay,
    observation_payload_bytes,
    packed_transition_bytes,
    transition_payload_bytes,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Shadow the repository fixture that requires a live Factorio server."""
    yield


def _transition(reward=0.0, is_demo=False, raster_tiles=16, entity_slots=8):
    env = FakeSemanticEnv(raster_tiles=raster_tiles, entity_slots=entity_slots)
    observation, masks = env.reset(0)
    heads = C.empty_heads()
    heads[C.HEAD_INDEX[C.DURATION]] = 0
    return C.Transition(
        observation.as_dict(),
        heads,
        C.VERB_INDEX["FAST_FORWARD"],
        reward,
        1.0,
        observation.as_dict(),
        False,
        True,
        "ok",
        masks,
        masks,
        is_demo,
    )


def test_sampling_probabilities_and_importance_weights():
    replay = PrioritizedReplay(capacity=4, alpha=0.6, demo_bonus=0, seed=3)
    raw = np.asarray([1.0, 2.0, 4.0, 8.0])
    for priority in raw:
        replay.add(_transition(priority), priority=priority)
    expected = raw**0.6 / np.sum(raw**0.6)
    counts = np.zeros(4)
    for _ in range(8_000):
        batch = replay.sample(1, beta=0.4)
        counts[batch.indices[0]] += 1
        probability = batch.probabilities[0]
        assert np.isclose(batch.weights[0], 1.0)  # normalized within a size-one batch
        assert np.isclose(probability, expected[batch.indices[0]], atol=1e-6)
    assert np.allclose(counts / counts.sum(), expected, atol=0.025)

    batch = replay.sample(4, beta=0.7)
    unnormalized = (len(replay) * batch.probabilities) ** -0.7
    assert np.allclose(batch.weights, unnormalized / unnormalized.max())


def test_priority_updates_and_demo_bonus_take_effect():
    replay = PrioritizedReplay(capacity=2, demo_bonus=5.0, seed=1)
    replay.add(_transition(), priority=1.0)
    replay.add(_transition(is_demo=True), priority=1.0)
    before = replay._tree[replay._tree_capacity : replay._tree_capacity + 2].copy()
    assert before[1] > before[0]
    replay.update_priorities(np.asarray([0]), np.asarray([100.0]))
    after = replay._tree[replay._tree_capacity : replay._tree_capacity + 2]
    assert after[0] > after[1]


def test_replay_disk_round_trip_is_lossless_after_compaction(tmp_path):
    replay = PrioritizedReplay(capacity=4, seed=9)
    replay.add(_transition(3.25), priority=7.0)
    before = replay.sample(1, beta=0.5)
    path = tmp_path / "replay.pkl"
    replay.save(path)
    restored = PrioritizedReplay.load(path)
    after = restored.sample(1, beta=0.5)
    left, right = before.transitions[0], after.transitions[0]
    assert left.reward == right.reward
    assert left.tau_seconds == right.tau_seconds
    assert np.array_equal(left.action_heads, right.action_heads)
    for key in ("grid", "entity_view", "entity_mask", "entity_ids", "globals", "raster"):
        assert np.array_equal(left.observation[key], right.observation[key])
    assert replay.total_priority == restored.total_priority


def test_documented_payload_numbers():
    per_96 = transition_payload_bytes(96)
    per_288 = transition_payload_bytes(288)
    assert per_96 == 2 * observation_payload_bytes(96) + len(C.HEADS) * 8
    assert per_96 < 100 * 1024
    assert per_288 < 250 * 1024
    assert DEFAULT_CAPACITY * 200 * 1024 < 6 * 1024**3
    actual = packed_transition_bytes(
        _transition(raster_tiles=96, entity_slots=C.ENTITY_SLOTS)
    )
    assert actual < 200 * 1024
