import copy

import numpy as np
import pytest
import torch

from fle.smarq import contract as C
from fle.smarq.fake import FakeSemanticEnv
from fle.smarq.learner import DoubleDQNLearner
from fle.smarq.net import SMARQNetwork
from fle.smarq.policy import AutoregressivePolicy
from fle.smarq.replay import PrioritizedReplay


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Shadow the repository fixture that requires a live Factorio server."""
    yield


def _fast_forward_transition(env, duration_index=0, reward=None, tau=None, done=False):
    observation, masks = env.reset(0)
    heads = C.empty_heads()
    heads[C.HEAD_INDEX[C.DURATION]] = duration_index
    action = C.Action(
        "FAST_FORWARD",
        heads,
        duration_seconds=C.DURATIONS_SECONDS[duration_index],
    )
    result, next_masks = env.step(action)
    return C.Transition(
        observation.as_dict(),
        heads,
        C.VERB_INDEX["FAST_FORWARD"],
        result.reward if reward is None else reward,
        result.duration_game_seconds if tau is None else tau,
        result.observation.as_dict(),
        done,
        result.success,
        result.failure_reason,
        masks,
        next_masks,
    )


def test_smdp_target_uses_simulated_duration_by_hand():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    learner = DoubleDQNLearner(SMARQNetwork(env.vocab, 16), horizon_seconds=100.0)
    short = learner.semantic_target(3.0, 10.0, False, 8.0)
    long = learner.semantic_target(3.0, 50.0, False, 8.0)
    assert np.isclose(short, 3.0 + 2 ** (-10 / 100) * 8.0)
    assert np.isclose(long, 3.0 + 2 ** (-50 / 100) * 8.0)
    assert long < short


def test_internal_transition_has_unit_discount_and_max_bootstrap():
    q_values = torch.tensor([-2.0, 7.0, 100.0, 3.0])
    legal = np.asarray([True, True, False, True])
    target = DoubleDQNLearner.internal_target(q_values, legal)
    assert target.item() == 7.0  # r=0 + Gamma(0)*max_legal Q; Gamma(0)=1


def test_gradient_step_is_finite_changes_parameters_and_clips():
    torch.manual_seed(0)
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    network = SMARQNetwork(env.vocab, 16)
    learner = DoubleDQNLearner(network, batch_size=2, grad_clip=1e-4)
    transitions = [
        _fast_forward_transition(env, reward=1_000.0),
        _fast_forward_transition(env, duration_index=1, reward=-1_000.0),
    ]
    before = copy.deepcopy(learner.online.state_dict())
    stats = learner.train_batch(transitions)
    assert np.isfinite(stats.loss)
    assert stats.grad_clipped
    assert any(not torch.equal(before[name], value) for name, value in learner.online.state_dict().items())


def test_target_network_updates_exactly_on_schedule():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    learner = DoubleDQNLearner(
        SMARQNetwork(env.vocab, 16), batch_size=1, target_update_interval=2
    )
    transition = _fast_forward_transition(env, reward=10.0)
    target_initial = copy.deepcopy(learner.target.state_dict())
    first = learner.train_batch([transition])
    assert not first.target_updated
    assert all(torch.equal(target_initial[name], value) for name, value in learner.target.state_dict().items())
    second = learner.train_batch([transition])
    assert second.target_updated
    assert all(
        torch.equal(learner.online.state_dict()[name], value)
        for name, value in learner.target.state_dict().items()
    )


def test_n_step_uses_product_of_per_step_discounts():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    learner = DoubleDQNLearner(SMARQNetwork(env.vocab, 16), horizon_seconds=100, n_step=3)
    transitions = [
        _fast_forward_transition(env, reward=1.0, tau=10.0),
        _fast_forward_transition(env, reward=2.0, tau=20.0),
        _fast_forward_transition(env, reward=4.0, tau=30.0),
    ]
    combined = learner.make_n_step_transition(transitions)
    expected = 1.0 + 2 ** (-10 / 100) * 2.0 + 2 ** (-30 / 100) * 4.0
    assert np.isclose(combined.reward, expected)
    assert combined.tau_seconds == 60.0
    assert np.isclose(
        2 ** (-combined.tau_seconds / 100),
        np.prod([2 ** (-transition.tau_seconds / 100) for transition in transitions]),
    )


def test_learner_policy_can_evaluate_partial_chain():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    observation, masks = env.reset(1)
    network = SMARQNetwork(env.vocab, 16)
    policy = AutoregressivePolicy(network, env.vocab)
    value = policy.partial_value(observation, masks, verb="FAST_FORWARD")
    assert np.isfinite(value)


@pytest.mark.slow
def test_vectorized_batch_matches_per_sample_reference():
    torch.manual_seed(2)
    env = FakeSemanticEnv(
        raster_tiles=16, entity_slots=8, tick_budget=1_000, decision_cap=5
    )
    network = SMARQNetwork(env.vocab, 16, entity_slots=8)
    behavior = AutoregressivePolicy(network, env.vocab, seed=3)
    transitions = []
    observation, masks = env.reset(1)
    while len(transitions) < 6:
        action = behavior.action(observation, masks, epsilon=1.0)
        result, next_masks = env.step(action)
        transitions.append(
            C.Transition(
                observation.as_dict(),
                action.heads,
                C.VERB_INDEX[action.verb],
                result.reward,
                result.duration_game_seconds,
                result.observation.as_dict(),
                result.done,
                result.success,
                result.failure_reason,
                masks,
                next_masks,
            )
        )
        if result.done:
            observation, masks = env.reset(len(transitions) + 1)
        else:
            observation, masks = result.observation, next_masks

    learner = DoubleDQNLearner(network, batch_size=len(transitions), device="cpu")
    comparison = learner.compare_vectorized_reference(transitions)
    for vectorized, reference in comparison.values():
        assert np.allclose(vectorized, reference, rtol=2e-5, atol=2e-6)


def _working_furnace_state(env, seed):
    _, masks = env.reset(seed)
    heads = C.empty_heads()
    heads[C.HEAD_INDEX[C.PROTOTYPE]] = 1
    heads[C.HEAD_INDEX[C.POSITION]] = 8 * 16 + 10
    heads[C.HEAD_INDEX[C.DIRECTION]] = 0
    place = C.Action(
        "PLACE",
        heads,
        tile=(2, 0),
        prototype="stone-furnace",
        direction="NORTH",
    )
    result, masks = env.step(place)

    heads = C.empty_heads()
    heads[C.HEAD_INDEX[C.ENTITY]] = 0
    heads[C.HEAD_INDEX[C.ITEM]] = 1
    heads[C.HEAD_INDEX[C.QUANTITY]] = C.N_QUANTITIES - 1
    insert = C.Action(
        "INSERT",
        heads,
        entity_slot=0,
        entity_id=1,
        item="coal",
        quantity="ALL",
    )
    result, masks = env.step(insert)
    assert env.entities[0]["working"]
    return result.observation, masks


def test_learning_smoke_improves_over_random_policy():
    """Dense offline toy task: choose one action with a working furnace."""
    torch.manual_seed(7)
    env = FakeSemanticEnv(
        raster_tiles=16,
        entity_slots=8,
        tick_budget=10_000,
        decision_cap=3,
    )
    behavior = AutoregressivePolicy(SMARQNetwork(env.vocab, 16), env.vocab, seed=9)
    replay = PrioritizedReplay(capacity=512, demo_bonus=0, seed=11)
    for index in range(2_000):
        observation, masks = _working_furnace_state(env, index)
        action = behavior.action(observation, masks, epsilon=1.0)
        result, next_masks = env.step(action)
        replay.add(
            C.Transition(
                observation.as_dict(),
                action.heads,
                C.VERB_INDEX[action.verb],
                result.reward,
                result.duration_game_seconds,
                result.observation.as_dict(),
                result.done,
                result.success,
                result.failure_reason,
                masks,
                next_masks,
            ),
            priority=abs(result.reward) + 0.1,
        )

    learner = DoubleDQNLearner(
        SMARQNetwork(env.vocab, 16),
        batch_size=16,
        target_update_interval=10,
        learning_rate=3e-4,
    )
    stats = None
    for _ in range(80):
        stats = learner.update_from_replay(replay)
    assert stats is not None

    def average_return(policy, epsilon):
        returns = []
        for index in range(40):
            observation, masks = _working_furnace_state(env, 10_000 + index)
            action = policy.action(observation, masks, epsilon=epsilon)
            result, _ = env.step(action)
            returns.append(result.reward)
        return float(np.mean(returns))

    random_return = average_return(behavior, 1.0)
    trained_return = average_return(
        AutoregressivePolicy(learner.online, env.vocab, seed=13), 0.0
    )
    print(
        "SMOKE "
        f"collected=2000 random_policy_return={random_return:.3f} "
        f"trained_return={trained_return:.3f} loss={stats.loss:.6f} "
        f"mean_q={stats.mean_q:.6f}"
    )
    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.mean_q)
    assert trained_return > random_return + 20.0

