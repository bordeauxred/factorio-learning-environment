from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fle.rl.dqn import masked_argmax
from fle.rl.v0_nets import (
    V0DQN,
    behaviour_actions,
    decode_actions,
    double_dqn_loss,
)
from fle.rl.v0_replay import (
    V0NStepAccumulator,
    V0ReplayBuffer,
    V0Transition,
    full_masks,
)
from fle.rl.v0_schema import (
    HEAD_SIZES,
    HEADS,
    OBS_SPEC,
    VERB_INDEX,
    obs_nbytes,
    smdp_discount,
)
from fle.rl.v0_train import (
    ObservationBatcher,
    live_decision_step,
    main,
    parse_args,
    synthetic_observation,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture."""
    yield


def one_choice_masks(
    verb: str = "FAST_FORWARD", head: str = "duration", choice: int = 0
) -> dict[str, np.ndarray]:
    masks = {name: np.zeros(size, dtype=np.uint8) for name, size in HEAD_SIZES.items()}
    masks["verb"][VERB_INDEX[verb]] = 1
    masks[head][choice] = 1
    return masks


def tensor_observation(
    obs: dict[str, np.ndarray], batch: int = 1
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.repeat(value[None], batch, axis=0))
        for key, value in obs.items()
    }


def tensor_masks(
    masks: dict[str, np.ndarray], batch: int = 1
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.repeat(value[None], batch, axis=0)).bool()
        for key, value in masks.items()
    }


def test_maximally_masked_batch_has_finite_loss_and_gradients() -> None:
    torch.manual_seed(0)
    obs = synthetic_observation()
    masks = one_choice_masks(choice=1)
    replay = V0ReplayBuffer(capacity=2, chunk_size=1)
    action = np.zeros(len(HEADS), dtype=np.int16)
    action[HEADS.index("verb")] = VERB_INDEX["FAST_FORWARD"]
    action[HEADS.index("duration")] = 1
    for reward in (1.0, -0.5):
        replay.add(V0Transition(obs, action, reward, obs, 10.0, False, masks, masks))
    batch = replay.sample(2, np.random.default_rng(0), torch.device("cpu"))
    online = V0DQN()
    target = V0DQN()
    target.load_state_dict(online.state_dict())

    loss, td_error = double_dqn_loss(online, target, batch)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in online.parameters()
        if parameter.grad is not None
    ]
    assert torch.isfinite(loss)
    assert torch.isfinite(td_error)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_masked_action_is_never_selected_by_behaviour_or_target_decode() -> None:
    torch.manual_seed(1)
    network = V0DQN()
    obs = tensor_observation(synthetic_observation())
    masks = tensor_masks(one_choice_masks(choice=2))

    target_choice = decode_actions(network, obs, masks).actions
    for _ in range(10):
        behaviour_choice = behaviour_actions(network, obs, masks, epsilon=1.0).actions
        assert int(behaviour_choice[0, HEADS.index("duration")]) == 2
    assert int(target_choice[0, HEADS.index("duration")]) == 2


def test_behaviour_and_double_dqn_reference_the_same_decode_function() -> None:
    assert behaviour_actions.__globals__["decode_actions"] is decode_actions
    assert double_dqn_loss.__globals__["decode_actions"] is decode_actions
    assert "decode_actions(" in inspect.getsource(behaviour_actions)
    assert "decode_actions(" in inspect.getsource(double_dqn_loss)


def test_smdp_n_step_discount_is_product_of_per_step_discounts() -> None:
    accumulator = V0NStepAccumulator(n=3, use_smdp=True)
    action = np.zeros(len(HEADS), dtype=np.int16)
    durations = (1.0, 10.0, 60.0)
    rewards = (2.0, 3.0, 5.0)
    assert accumulator.append({}, action, rewards[0], {}, durations[0], False) == []
    assert accumulator.append({}, action, rewards[1], {}, durations[1], False) == []
    emitted = accumulator.append({}, action, rewards[2], {}, durations[2], False)

    transition = emitted[0]
    discounts = [float(smdp_discount(duration)) for duration in durations]
    assert transition.discount == pytest.approx(np.prod(discounts))
    assert transition.reward == pytest.approx(
        rewards[0]
        + discounts[0] * rewards[1]
        + discounts[0] * discounts[1] * rewards[2]
    )
    assert transition.duration == sum(durations)


def test_replay_round_trips_schema_dtypes_and_casts_only_batch_buildability() -> None:
    obs = synthetic_observation()
    obs["buildability"][0, 0, 0] = -1
    obs["buildability"][0, 0, 1] = 1
    replay = V0ReplayBuffer(capacity=2, chunk_size=1)
    replay.add(
        V0Transition(
            obs,
            np.zeros(len(HEADS), dtype=np.int16),
            1.0,
            obs,
            5.0,
            False,
            full_masks(),
            full_masks(),
        )
    )

    restored = replay.get_state(0)
    for key, (_, dtype) in OBS_SPEC.items():
        assert restored[key].dtype == dtype
        np.testing.assert_array_equal(restored[key], obs[key])
    assert restored["buildability"].dtype == np.int8
    batch = replay.sample(1, np.random.default_rng(0), torch.device("cpu"))
    assert batch.obs["buildability"].dtype == torch.float32
    assert batch.obs["buildability"][0, 0, 0, 0] == -1


def test_replay_byte_measurement_matches_schema_array_sizes() -> None:
    replay = V0ReplayBuffer(capacity=7)
    measured = sum(
        np.empty(shape, dtype=dtype).nbytes for shape, dtype in OBS_SPEC.values()
    )
    assert measured == obs_nbytes()
    assert replay.bytes_per_transition == measured
    assert replay.nbytes == 7 * measured


def test_all_zero_mask_does_not_silently_select_index_zero() -> None:
    values = torch.tensor([[100.0, 2.0, 1.0]])
    mask = torch.zeros_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="all-zero"):
        masked_argmax(values, mask)
    assert masked_argmax(values, mask, empty_value=-1).item() == -1


def test_fake_training_loop_runs_for_a_few_hundred_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--fake",
            "--total-steps",
            "200",
            "--wandb-mode",
            "disabled",
            "--batch-size",
            "1",
            "--fake-replay-size",
            "24",
            "--learning-starts",
            "8",
            "--update-every",
            "32",
            "--log-interval",
            "100",
        ]
    )
    output = capsys.readouterr().out
    assert result["updates"] > 0
    assert np.isfinite(float(result["final_loss"]))
    assert "step=1 loss=" not in output


def test_default_replay_capacity_is_memory_safe() -> None:
    assert parse_args(["--fake"]).replay_size == 4096


def test_live_inner_step_masks_tau_replay_and_survives_step_exception() -> None:
    class StubLiveEnv:
        def __init__(self) -> None:
            self.fail_next = True
            self.actions: list[np.ndarray] = []
            self.world = SimpleNamespace(entities={})
            self.context = {
                "mask_toggles": {
                    "place_location": False,
                    "place_item": False,
                    "inventory_item": False,
                    "contained_item": False,
                    "entity": False,
                    "recipe": False,
                    "technology": False,
                    "verb": False,
                },
                "mask_counters": {},
            }

        def reset(self, *, seed: int | None = None):
            del seed
            return live_obs(), {}

        def v0_learner_context(self):
            return self.world, self.context

        def step(self, action: np.ndarray):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("synthetic per-step failure")
            assert action.dtype == np.int64
            self.actions.append(action.copy())
            return live_obs(), 1.25, False, False, {"ticks": 120, "success": True}

    def live_obs() -> dict[str, object]:
        obs: dict[str, object] = synthetic_observation()
        obs["masks"] = one_choice_masks(choice=2)
        return obs

    device = torch.device("cpu")
    env = StubLiveEnv()
    obs, _ = env.reset(seed=7)
    network = V0DQN().to(device)
    replay = V0ReplayBuffer(capacity=2, chunk_size=1)
    accumulator = V0NStepAccumulator(n=1)
    batcher = ObservationBatcher(device)

    failed = live_decision_step(
        env, obs, network, replay, accumulator, batcher, device, epsilon=1.0
    )
    assert isinstance(failed.error, RuntimeError)
    assert failed.obs is obs
    assert replay.size == 0
    assert not accumulator.pending

    succeeded = live_decision_step(
        env, obs, network, replay, accumulator, batcher, device, epsilon=1.0
    )
    assert succeeded.error is None
    assert succeeded.transitions_added == 1
    assert succeeded.duration == pytest.approx(2.0)
    assert replay.size == 1
    assert replay.durations[0] == pytest.approx(2.0)
    action = env.actions[0]
    assert action[HEADS.index("verb")] == VERB_INDEX["FAST_FORWARD"]
    assert action[HEADS.index("duration")] == 2
