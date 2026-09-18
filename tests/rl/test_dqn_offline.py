from __future__ import annotations

import time
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fle.rl import schema as S
from fle.rl.dqn import (
    RUNG_NAMES,
    ArchivedFrontier,
    BootstrappedDQN,
    BranchingDQN,
    EpisodeMilestones,
    MilestoneArchive,
    NoisyBranchingDQN,
    NStepAccumulator,
    NStepTransition,
    QuantileBranchingDQN,
    ReplayBatch,
    ReplayBuffer,
    ThreadVectorEnv,
    UCBExplorer,
    _make_env_fns,
    action_details,
    advance_frontier_force_window,
    complete_action_q,
    complete_action_quantiles,
    compute_bootstrapped_td_loss,
    compute_quantile_td_loss,
    compute_td_loss,
    drill_recipe_admitted,
    forced_drill_action,
    is_new_return_frontier,
    linear_ucb_coefficient,
    load_checkpoint,
    masked_argmax,
    parse_args,
    per_priorities_from_deltas,
    record_episode_counters,
    reset_network_noise,
    sample_secondary_heads,
    save_checkpoint,
    set_network_noise_enabled,
    train,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for this offline module."""
    yield


def maximally_masked_observations(batch_size: int) -> torch.Tensor:
    obs = torch.zeros((batch_size, S.OBS_SIZE), dtype=torch.float32)
    for start, _ in S.MASK_OFFSETS.values():
        obs[:, start] = 1.0
    return obs


def test_maximally_masked_batch_has_finite_loss_and_gradients() -> None:
    torch.manual_seed(0)
    online = BranchingDQN()
    target = BranchingDQN()
    target.load_state_dict(online.state_dict())
    obs = maximally_masked_observations(4)
    batch = ReplayBatch(
        obs=obs,
        actions=torch.zeros((4, len(S.HEADS)), dtype=torch.long),
        rewards=torch.tensor([1.0, 0.0, -1.0, 0.5]),
        next_obs=obs.clone(),
        discounts=torch.full((4,), 0.99**3),
    )

    loss, mean_max_q = compute_td_loss(online, target, batch)
    loss.backward()

    gradients = [parameter.grad for parameter in online.parameters() if parameter.grad is not None]
    assert torch.isfinite(loss)
    assert torch.isfinite(mean_max_q)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_every_bootstrap_head_has_finite_loss_on_maximally_masked_batch() -> None:
    torch.manual_seed(0)
    online = BootstrappedDQN(num_heads=4)
    target = BootstrappedDQN(num_heads=4)
    target.load_state_dict(online.state_dict())
    obs = maximally_masked_observations(4)
    batch = ReplayBatch(
        obs=obs,
        actions=torch.zeros((4, len(S.HEADS)), dtype=torch.long),
        rewards=torch.tensor([1.0, 0.0, -1.0, 0.5]),
        next_obs=obs.clone(),
        discounts=torch.full((4,), 0.99**3),
        weights=torch.ones(4),
        bootstrap_masks=torch.eye(4, dtype=torch.bool),
    )

    loss, mean_max_q, priorities, head_losses = compute_bootstrapped_td_loss(
        online, target, batch
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(mean_max_q).all()
    assert torch.isfinite(priorities).all()
    assert torch.isfinite(head_losses).all()
    for head in online.heads:
        gradients = [
            parameter.grad for parameter in head.parameters() if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_quantile_loss_is_finite_and_gradient_carrying_when_maximally_masked() -> None:
    torch.manual_seed(8)
    online = QuantileBranchingDQN(num_quantiles=5)
    target = QuantileBranchingDQN(num_quantiles=5)
    target.load_state_dict(online.state_dict())
    obs = maximally_masked_observations(3)
    batch = ReplayBatch(
        obs=obs,
        actions=torch.zeros((3, len(S.HEADS)), dtype=torch.long),
        rewards=torch.tensor([-8.0, 0.0, 1700.0]),
        next_obs=obs.clone(),
        discounts=torch.full((3,), 0.99**3),
        weights=torch.ones(3),
    )

    loss, mean_max_q, priorities = compute_quantile_td_loss(online, target, batch)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in online.parameters()
        if parameter.grad is not None
    ]
    assert torch.isfinite(loss)
    assert torch.isfinite(mean_max_q)
    assert torch.isfinite(priorities).all()
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_single_quantile_mean_matches_plain_complete_action_q() -> None:
    torch.manual_seed(9)
    plain = BranchingDQN()
    quantile = QuantileBranchingDQN(num_quantiles=1)
    quantile.encoder.load_state_dict(plain.encoder.state_dict())
    quantile.value.mean_layer.load_state_dict(plain.value.state_dict())
    for head in S.HEADS:
        quantile.advantages[head].mean_layer.load_state_dict(
            plain.advantages[head].state_dict()
        )
    obs = maximally_masked_observations(2)
    actions = torch.zeros((2, len(S.HEADS)), dtype=torch.long)

    plain_value, plain_advantages = plain(obs)
    quantile_value, quantile_advantages = quantile(obs)
    plain_q = complete_action_q(plain_value, plain_advantages, actions)
    quantile_q = complete_action_quantiles(
        quantile_value, quantile_advantages, actions
    ).squeeze(-1)

    assert torch.allclose(quantile_q, plain_q)


def test_noisy_layers_resample_and_are_deterministic_when_disabled() -> None:
    torch.manual_seed(10)
    network = NoisyBranchingDQN()
    obs = maximally_masked_observations(2)

    first_value, first_advantages = network(obs)
    reset_network_noise(network)
    second_value, second_advantages = network(obs)
    set_network_noise_enabled(network, False)
    deterministic_value, deterministic_advantages = network(obs)
    repeated_value, repeated_advantages = network(obs)

    assert not torch.equal(first_value, second_value)
    assert any(
        not torch.equal(first_advantages[head], second_advantages[head])
        for head in S.HEADS
    )
    assert torch.equal(deterministic_value, repeated_value)
    assert all(
        torch.equal(deterministic_advantages[head], repeated_advantages[head])
        for head in S.HEADS
    )


def test_masked_argmax_never_selects_a_masked_index() -> None:
    values = torch.tensor([[100.0, 3.0, 2.0], [-1.0, 50.0, 4.0]])
    masks = torch.tensor([[False, True, False], [True, False, True]])
    selected = masked_argmax(values, masks)
    assert selected.tolist() == [1, 2]
    assert masks.gather(1, selected[:, None]).all()


def test_three_step_return_matches_hand_computation() -> None:
    accumulator = NStepAccumulator(n=3, gamma=0.99)
    observations = [np.full(S.OBS_SIZE, index, dtype=np.float32) for index in range(4)]
    action = np.zeros(len(S.HEADS), dtype=np.int64)

    assert accumulator.append(observations[0], action, 1.0, observations[1], False) == []
    assert accumulator.append(observations[1], action, 2.0, observations[2], False) == []
    emitted = accumulator.append(observations[2], action, 3.0, observations[3], False)

    assert len(emitted) == 1
    transition = emitted[0]
    assert transition.reward == pytest.approx(1.0 + 0.99 * 2.0 + 0.99**2 * 3.0)
    assert transition.discount == pytest.approx(0.99**3)
    np.testing.assert_array_equal(transition.obs, observations[0])
    np.testing.assert_array_equal(transition.next_obs, observations[3])


def test_ten_step_return_truncates_at_episode_end() -> None:
    accumulator = NStepAccumulator(n=10, gamma=0.5)
    observations = [np.full(S.OBS_SIZE, index, dtype=np.float32) for index in range(7)]
    action = np.zeros(len(S.HEADS), dtype=np.int64)

    assert accumulator.append(observations[0], action, 1.0, observations[1], False) == []
    assert accumulator.append(observations[1], action, 2.0, observations[2], False) == []
    emitted = accumulator.append(observations[2], action, 4.0, observations[3], True)

    assert [transition.reward for transition in emitted] == pytest.approx(
        [3.0, 4.0, 4.0]
    )
    assert [transition.discount for transition in emitted] == [0.0, 0.0, 0.0]
    assert all(np.array_equal(transition.next_obs, observations[3]) for transition in emitted)

    assert accumulator.append(observations[3], action, 8.0, observations[4], False) == []
    second_episode = accumulator.append(
        observations[4], action, 16.0, observations[5], True
    )
    assert [transition.reward for transition in second_episode] == pytest.approx(
        [16.0, 16.0]
    )
    assert all(
        np.array_equal(transition.next_obs, observations[5])
        for transition in second_episode
    )


def _transition(value: float, bootstrap_mask: np.ndarray | None = None) -> NStepTransition:
    obs = np.full(S.OBS_SIZE, value, dtype=np.float32)
    for start, _ in S.MASK_OFFSETS.values():
        obs[start] = 1.0
    return NStepTransition(
        obs=obs,
        action=np.zeros(len(S.HEADS), dtype=np.int64),
        reward=value,
        next_obs=obs.copy(),
        discount=0.99,
        bootstrap_mask=bootstrap_mask,
    )


def test_per_importance_weights_have_sane_sum() -> None:
    replay = ReplayBuffer(capacity=4, prioritized=True)
    for value in range(4):
        replay.add(_transition(float(value)))
    replay.priorities[:4] = np.asarray([1.0, 2.0, 4.0, 8.0])

    batch = replay.sample(256, np.random.default_rng(3), torch.device("cpu"), beta=0.7)

    assert batch.weights is not None
    assert torch.isfinite(batch.weights).all()
    assert (batch.weights > 0).all()
    assert batch.weights.max() == pytest.approx(1.0)
    assert 0.0 < float(batch.weights.sum()) <= 256.0


def test_symmetric_per_priorities_match_existing_formula_bit_for_bit() -> None:
    deltas = torch.tensor([-8.0, -1.0, 0.0, 1.0, 8.0])
    alpha = 0.6
    expected = deltas.abs().pow(alpha) + 1e-6

    actual = per_priorities_from_deltas(deltas, alpha, optimism=1.0)

    assert torch.equal(actual, expected)


def test_optimistic_per_scales_equal_positive_error_by_k_to_alpha() -> None:
    alpha = 0.6
    priorities = per_priorities_from_deltas(
        torch.tensor([-2.0, 2.0]), alpha, optimism=3.0
    )

    ratio = (priorities[1] - 1e-6) / (priorities[0] - 1e-6)
    assert ratio == pytest.approx(3.0**alpha)


def test_episode_return_bonus_multiplies_exactly_episode_indices() -> None:
    replay = ReplayBuffer(capacity=5, prioritized=True)
    for value in range(5):
        replay.add(_transition(float(value)))
    replay.priorities[:] = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    boosted = replay.boost_priorities([1, 3], multiplier=2.5)

    assert boosted == 2
    np.testing.assert_array_equal(
        replay.priorities, np.asarray([1.0, 5.0, 3.0, 10.0, 5.0])
    )


def test_rung_ucb_counts_and_schedule_round_trip() -> None:
    explorer = UCBExplorer(c=0.75, key_mode="rung")
    explorer.counts[(-1, "CRAFT", 3)] = 1
    explorer.counts[(2, "CRAFT", 3)] = 4
    explorer.total = 5

    restored = UCBExplorer(c=0.75, key_mode="rung")
    restored.load_state_dict(explorer.state_dict())

    assert restored.counts == explorer.counts
    assert restored.total == 5
    assert linear_ucb_coefficient(0, 0.75, 0.15, 40_000) == pytest.approx(0.75)
    assert linear_ucb_coefficient(20_000, 0.75, 0.15, 40_000) == pytest.approx(0.45)
    assert linear_ucb_coefficient(50_000, 0.75, 0.15, 40_000) == pytest.approx(0.15)
    assert linear_ucb_coefficient(50_000, 0.75, None, None) == pytest.approx(0.75)


def test_every_schema_op_decodes_a_non_none_primary() -> None:
    obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
    target_start, _ = S.OBS_LAYOUT["targets"]
    obs[target_start] = 1.0
    entity_start, _ = S.OBS_LAYOUT["entities"]
    obs[entity_start] = 1.0
    action = np.zeros(len(S.HEADS), dtype=np.int64)

    for op_index, op in enumerate(S.OPS):
        action[S.HEADS.index("op")] = op_index
        details = action_details(obs, action)
        assert details["op"] == op
        assert details["primary_id"] is not None
        assert details["primary"] is not None


def _place_pair_observation() -> np.ndarray:
    obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
    op_start, _ = S.MASK_OFFSETS["op"]
    obs[op_start + S.OP_INDEX["PLACE"]] = 1.0
    placeable_start, _ = S.MASK_OFFSETS["placeable"]
    obs[
        placeable_start + S.PLACEABLE_INDEX["burner-mining-drill"]
    ] = 1.0
    for head, indices in {
        "offset": (S.dxdy_to_offset(1, 0), S.dxdy_to_offset(2, 0)),
        "direction": (0, 1),
    }.items():
        start, _ = S.MASK_OFFSETS[head]
        obs[start + np.asarray(indices)] = 1.0
    return obs


def _drill_frontier_observation() -> np.ndarray:
    obs = _place_pair_observation()
    op_start, _ = S.MASK_OFFSETS["op"]
    obs[op_start + S.OP_INDEX["INSERT"]] = 1.0
    item_start, _ = S.MASK_OFFSETS["item"]
    obs[item_start + S.ITEM_INDEX["coal"]] = 1.0
    obs[item_start + S.ITEM_INDEX["wood"]] = 1.0
    entity_start, _ = S.OBS_LAYOUT["entities"]
    drill_slot = 2
    row_start = entity_start + drill_slot * S.ENTITY_FEATURES
    obs[row_start + S.ENTITY_CLASSES.index("mining-drill")] = 1.0
    obs[row_start + S.ENTITY_FEATURES - 2] = 1.0
    entity_mask_start, _ = S.MASK_OFFSETS["entity"]
    obs[entity_mask_start + drill_slot] = 1.0
    quantity_start, quantity_stop = S.MASK_OFFSETS["quantity"]
    obs[quantity_start:quantity_stop] = 1.0
    inventory_start, _ = S.OBS_LAYOUT["inventory"]
    obs[inventory_start + S.ITEM_INDEX["burner-mining-drill"]] = 0.1
    scale = np.log1p(1000)
    obs[inventory_start + S.ITEM_INDEX["coal"]] = np.log1p(20) / scale
    obs[inventory_start + S.ITEM_INDEX["wood"]] = np.log1p(20) / scale
    return obs


def test_frontier_force_window_starts_on_drill_rung_and_expires() -> None:
    remaining = advance_frontier_force_window(0, 2, 3, 3)
    assert remaining == 3

    remaining = advance_frontier_force_window(
        remaining, 3, 3, 3, trial_was_new=True
    )
    remaining = advance_frontier_force_window(remaining, 3, 3, 3)
    assert remaining == 2
    remaining = advance_frontier_force_window(
        remaining, 3, 3, 3, trial_was_new=True
    )
    remaining = advance_frontier_force_window(
        remaining, 3, 3, 3, trial_was_new=True
    )

    assert remaining == 0
    assert advance_frontier_force_window(0, 1, 2, 3) == 0


def test_frontier_force_only_selects_pairs_at_or_below_max_count() -> None:
    obs = _drill_frontier_observation()
    rung = RUNG_NAMES.index("drill_crafted")
    drill = S.PLACEABLE_INDEX["burner-mining-drill"]
    coal = S.ITEM_INDEX["coal"]
    wood = S.ITEM_INDEX["wood"]
    explorer = UCBExplorer(key_mode="rung")
    explorer.counts[(rung, "PLACE", drill)] = 3
    explorer.counts[(rung, "INSERT", coal)] = 2
    explorer.counts[(rung, "INSERT", wood)] = 3
    explorer.total = 8

    actions, diagnostics = explorer.select(
        BranchingDQN(),
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(31),
        device=torch.device("cpu"),
        rungs=np.asarray([rung]),
        frontier_forcing=[True],
        frontier_force_max_count=2,
    )

    assert diagnostics[0]["frontier_forced"] is True
    assert diagnostics[0]["op"] == "INSERT"
    assert diagnostics[0]["primary"] == "coal"
    assert diagnostics[0]["n"] == 2
    assert actions[0, S.HEADS.index("entity")] == 2
    assert S.QUANTITIES[actions[0, S.HEADS.index("quantity")]] <= 20
    assert diagnostics[0]["admitted_pair_count"] >= 3
    assert diagnostics[0]["n0_pair_count"] >= 0


def test_frontier_force_does_not_repeat_a_complete_insert_trial() -> None:
    obs = _drill_frontier_observation()
    rung = RUNG_NAMES.index("drill_crafted")
    explorer = UCBExplorer(key_mode="rung")
    network = BranchingDQN()

    first_actions, first_diagnostics = explorer.select(
        network,
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(8),
        device=torch.device("cpu"),
        rungs=np.asarray([rung]),
        frontier_forcing=[True],
        frontier_tried_actions=[set()],
        frontier_force_max_count=10,
    )
    first_trial = first_diagnostics[0]["frontier_trial"]
    second_actions, second_diagnostics = explorer.select(
        network,
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(8),
        device=torch.device("cpu"),
        rungs=np.asarray([rung]),
        frontier_forcing=[True],
        frontier_tried_actions=[{first_trial}],
        frontier_force_max_count=10,
    )

    assert first_diagnostics[0]["frontier_forced"] is True
    assert second_diagnostics[0]["frontier_forced"] is True
    assert second_diagnostics[0]["frontier_trial"] != first_trial
    assert not np.array_equal(first_actions, second_actions)


def test_first_drill_place_samples_only_admitted_resource_offsets() -> None:
    obs = _place_pair_observation()
    grid_start, _ = S.OBS_LAYOUT["grid"]
    grid = obs[grid_start : grid_start + len(S.GRID_CHANNELS) * S.GRID_SIDE**2]
    grid = grid.reshape(len(S.GRID_CHANNELS), S.GRID_SIDE, S.GRID_SIDE)
    grid[S.GRID_CHANNELS.index("resource"), S.OFFSET_RADIUS, S.OFFSET_RADIUS + 2] = 1

    actions, diagnostics = UCBExplorer(key_mode="rung").select(
        BranchingDQN(),
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(4),
        device=torch.device("cpu"),
        rungs=np.asarray([3]),
    )

    assert diagnostics[0]["n"] == 0
    assert actions[0, S.HEADS.index("offset")] == S.dxdy_to_offset(2, 0)


def test_frontier_insert_candidates_use_only_held_positive_fuel_items() -> None:
    obs = _drill_frontier_observation()
    item_start, _ = S.MASK_OFFSETS["item"]
    inventory_start, _ = S.OBS_LAYOUT["inventory"]
    obs[item_start + S.ITEM_INDEX["stone"]] = 1.0
    obs[inventory_start + S.ITEM_INDEX["stone"]] = np.log1p(20) / np.log1p(1000)
    explorer = UCBExplorer(key_mode="rung")
    rung = RUNG_NAMES.index("drill_crafted")
    drill = S.PLACEABLE_INDEX["burner-mining-drill"]
    explorer.counts[(rung, "PLACE", drill)] = 3
    explorer.total = 3

    actions, diagnostics = explorer.select(
        BranchingDQN(),
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(5),
        device=torch.device("cpu"),
        rungs=np.asarray([rung]),
        frontier_forcing=[True],
        frontier_force_max_count=2,
    )

    item = S.ITEM_NAMES[actions[0, S.HEADS.index("item")]]
    assert diagnostics[0]["frontier_forced"] is True
    assert diagnostics[0]["op"] == "INSERT"
    assert item in {"coal", "wood"}


def test_forced_drill_place_keeps_sampled_offset_when_no_ore_is_admitted() -> None:
    obs = _place_pair_observation()
    inventory_start, _ = S.OBS_LAYOUT["inventory"]
    obs[inventory_start + S.ITEM_INDEX["burner-mining-drill"]] = 0.1
    base = np.zeros(len(S.HEADS), dtype=np.int64)
    base[S.HEADS.index("offset")] = S.dxdy_to_offset(1, 0)

    action = forced_drill_action(obs, base, np.random.default_rng(3))

    assert action[S.HEADS.index("offset")] == S.dxdy_to_offset(1, 0)


def test_forced_place_pair_preserves_offset_and_samples_direction() -> None:
    obs = _place_pair_observation()
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["PLACE"]
    action[S.HEADS.index("placeable")] = S.PLACEABLE_INDEX[
        "burner-mining-drill"
    ]
    greedy_offset = S.dxdy_to_offset(1, 0)
    action[S.HEADS.index("offset")] = greedy_offset
    seen_offsets = set()
    seen_directions = set()
    rng = np.random.default_rng(12)
    for _ in range(100):
        sampled = sample_secondary_heads(action, obs, "PLACE", rng)
        seen_offsets.add(int(sampled[S.HEADS.index("offset")]))
        seen_directions.add(int(sampled[S.HEADS.index("direction")]))
        assert sampled[S.HEADS.index("placeable")] == action[
            S.HEADS.index("placeable")
        ]
    assert seen_offsets == {greedy_offset}
    assert seen_directions == {0, 1}

    explorer = UCBExplorer(key_mode="rung")
    network = BranchingDQN()
    first_actions, first_diagnostics = explorer.select(
        network,
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(3),
        device=torch.device("cpu"),
        rungs=np.asarray([3]),
        forced_arg_eps=0.0,
    )
    primary = S.PLACEABLE_INDEX["burner-mining-drill"]
    _, retry_diagnostics = explorer.select(
        network,
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(4),
        device=torch.device("cpu"),
        rungs=np.asarray([3]),
        forced_arg_eps=1.0,
        retry_pairs=[(3, "PLACE", primary)],
    )

    assert first_actions[0, S.HEADS.index("placeable")] == primary
    assert first_diagnostics[0]["secondary_heads_sampled"] is False
    assert retry_diagnostics[0]["n"] == 1
    assert retry_diagnostics[0]["secondary_heads_sampled"] is True


def test_newly_admitted_placeable_is_forced_at_current_rung() -> None:
    obs = _place_pair_observation()
    placeable_start, _ = S.MASK_OFFSETS["placeable"]
    admitted = (
        S.PLACEABLE_INDEX["stone-furnace"],
        S.PLACEABLE_INDEX["wooden-chest"],
        S.PLACEABLE_INDEX["burner-mining-drill"],
    )
    obs[placeable_start + np.asarray(admitted)] = 1.0
    rung = 3
    drill = S.PLACEABLE_INDEX["burner-mining-drill"]
    explorer = UCBExplorer(key_mode="rung")
    network = BranchingDQN()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    for primary_id in admitted:
        if primary_id != drill:
            explorer.counts[(rung, "PLACE", primary_id)] = 1
            explorer.total += 1

    actions, diagnostics = explorer.select(
        network,
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(21),
        device=torch.device("cpu"),
        rungs=np.asarray([rung]),
        forced_arg_eps=1.0,
    )

    assert actions[0, S.HEADS.index("placeable")] == drill
    assert actions[0, S.HEADS.index("offset")] == S.dxdy_to_offset(1, 0)
    assert diagnostics[0]["primary"] == "burner-mining-drill"
    assert diagnostics[0]["n"] == 0


def test_resume_drops_legacy_place_none_count_keys(capsys: pytest.CaptureFixture[str]) -> None:
    explorer = UCBExplorer(key_mode="rung")
    explorer.load_state_dict(
        {
            "key_mode": "rung",
            "counts": [
                (3, "PLACE", None, 7),
                (3, "CRAFT", S.RECIPE_INDEX["burner-mining-drill"], 2),
            ],
            "total": 9,
        }
    )

    assert explorer.counts == {
        (3, "CRAFT", S.RECIPE_INDEX["burner-mining-drill"]): 2
    }
    assert explorer.total == 2
    assert "keys=1 selections=7" in capsys.readouterr().out


def _craft_drill_observation() -> np.ndarray:
    obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
    op_start, _ = S.MASK_OFFSETS["op"]
    obs[op_start + S.OP_INDEX["CRAFT"]] = 1.0
    recipe_start, _ = S.MASK_OFFSETS["recipe"]
    obs[recipe_start + S.RECIPE_INDEX["burner-mining-drill"]] = 1.0
    quantity_start, quantity_stop = S.MASK_OFFSETS["quantity"]
    obs[quantity_start:quantity_stop] = 1.0
    return obs


def test_first_ucb_pair_uses_smallest_quantity_and_counts_executed_craft():
    obs = _craft_drill_observation()
    explorer = UCBExplorer(key_mode="rung")

    actions, diagnostics = explorer.select(
        BranchingDQN(),
        obs[None],
        epsilon=0.0,
        rng=np.random.default_rng(17),
        device=torch.device("cpu"),
        rungs=np.asarray([2]),
        forced_arg_eps=1.0,
    )

    assert actions[0, S.HEADS.index("quantity")] == 0
    recipe = S.RECIPE_INDEX["burner-mining-drill"]
    assert diagnostics[0]["_count_key"] == (2, "CRAFT", recipe, 1)
    explorer.reconcile_executed_quantity(diagnostics[0], 5)
    assert explorer.counts[(2, "CRAFT", recipe, 1)] == 0
    assert explorer.counts[(2, "CRAFT", recipe, 5)] == 1


def test_drill_admission_forces_smallest_supported_craft():
    obs = _craft_drill_observation()

    action = forced_drill_action(obs)

    assert drill_recipe_admitted(obs)
    assert action[S.HEADS.index("op")] == S.OP_INDEX["CRAFT"]
    assert action[S.HEADS.index("recipe")] == S.RECIPE_INDEX[
        "burner-mining-drill"
    ]
    assert action[S.HEADS.index("quantity")] == 0


def test_crafted_drill_frontier_forces_place_with_greedy_offset():
    obs = _drill_frontier_observation()
    base = np.zeros(len(S.HEADS), dtype=np.int64)
    greedy_offset = S.dxdy_to_offset(2, 0)
    base[S.HEADS.index("offset")] = greedy_offset

    action = forced_drill_action(obs, base, np.random.default_rng(7))

    assert action[S.HEADS.index("op")] == S.OP_INDEX["PLACE"]
    assert action[S.HEADS.index("placeable")] == S.PLACEABLE_INDEX[
        "burner-mining-drill"
    ]
    assert action[S.HEADS.index("offset")] == greedy_offset


def test_drill_craft_adds_snapshot_and_restored_action_places(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DrillFrontierEnv:
        def __init__(self) -> None:
            self._episode_max_steps = 10
            self.step_index = 0
            self.snapshots: list[int] = []
            self.restores: list[int] = []
            self.actions: list[str] = []
            self.observation = _craft_drill_observation()

        def reset(self, *, seed=None, options=None):
            del seed
            restore = None if options is None else options.get("restore")
            if restore is None:
                self.step_index = 0
                self.observation = _craft_drill_observation()
                return self.observation.copy(), {"excursion": False}
            self.restores.append(int(restore.step_index))
            self.step_index = int(restore.step_index)
            self.observation = _drill_frontier_observation()
            return self.observation.copy(), {"excursion": True}

        def snapshot(self):
            self.snapshots.append(self.step_index)
            return SimpleNamespace(
                entity_hash=f"snapshot-{self.step_index}",
                step_index=self.step_index,
                max_steps=self._episode_max_steps,
            )

        def step(self, action):
            details = action_details(self.observation, action)
            self.actions.append(str(details["op"]))
            self.step_index += 1
            self.observation = _drill_frontier_observation()
            return self.observation.copy(), 1.0, False, True, {
                "op": details["op"],
                "status": "ok",
                "score_automated": 0.0,
                "score_player": 0.0,
                "excursion": bool(self.restores),
            }

        def close(self) -> None:
            pass

    env = DrillFrontierEnv()
    monkeypatch.setattr("fle.rl.dqn._make_env_fns", lambda args: [lambda: env])
    args = parse_args(
        [
            "--ports",
            "27000",
            "--explore",
            "ucb",
            "--frontier-return",
            "--return-rung",
            "drill_admitted",
            "--return-prob",
            "1",
            "--total-steps",
            "2",
            "--out",
            str(tmp_path),
        ]
    )

    train(args)

    assert env.snapshots[:2] == [0, 1]
    assert env.restores[0] == 1
    assert env.actions == ["CRAFT", "PLACE"]


def test_frontier_archive_balances_rungs_and_records_offset_draws() -> None:
    archive = MilestoneArchive()
    for rung in (1, 2, 3):
        transitions = [_transition(float(rung)) for _ in range(100)]
        archive.add_episode(transitions, rung, float(rung), first_attainment=50)

    batch = archive.sample(
        40,
        np.random.default_rng(9),
        torch.device("cpu"),
        num_bootstrap_heads=1,
        mode="frontier",
    )
    draws = archive.consume_draw_counts()

    rewards = batch.rewards.tolist()
    assert rewards.count(3.0) == 20
    assert rewards.count(2.0) == 10
    assert rewards.count(1.0) == 10
    assert sum(draws["by_rung"].values()) == 40
    assert sum(draws["by_offset"].values()) == 40
    assert draws["by_mode"] == {"frontier": 40}
    assert draws["by_source"]["frontier_window"] > draws["by_source"]["episode_uniform"]


def test_milestones_record_first_attainment_step() -> None:
    milestones = EpisodeMilestones()
    details = {"op": "CRAFT", "primary": "iron-gear-wheel"}

    milestones.observe(details, "ok", step=17)
    milestones.observe(details, "ok", step=18)

    assert milestones.deepest == 2
    assert milestones.first_attainment[2] == 17


def test_bootstrap_masks_are_stored_and_applied() -> None:
    replay = ReplayBuffer(capacity=2, num_bootstrap_heads=2)
    replay.add(_transition(1.0, np.asarray([True, False])))
    stored = replay.sample(1, np.random.default_rng(0), torch.device("cpu"))
    assert stored.bootstrap_masks is not None
    assert stored.bootstrap_masks.tolist() == [[True, False]]

    torch.manual_seed(1)
    online = BootstrappedDQN(num_heads=2)
    target = BootstrappedDQN(num_heads=2)
    target.load_state_dict(online.state_dict())
    obs = maximally_masked_observations(2)
    batch = ReplayBatch(
        obs=obs,
        actions=torch.zeros((2, len(S.HEADS)), dtype=torch.long),
        rewards=torch.ones(2),
        next_obs=obs.clone(),
        discounts=torch.full((2,), 0.99),
        weights=torch.ones(2),
        bootstrap_masks=torch.tensor([[True, False], [True, False]]),
    )
    loss, _, _, head_losses = compute_bootstrapped_td_loss(online, target, batch)
    loss.backward()

    assert head_losses[0] > 0
    assert head_losses[1] == 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in online.heads[1].parameters()
    )


def test_randomized_priors_do_not_enter_replay_loss_or_targets() -> None:
    torch.manual_seed(4)
    online = BootstrappedDQN(num_heads=2)
    target = BootstrappedDQN(num_heads=2)
    target.load_state_dict(online.state_dict())
    obs = maximally_masked_observations(2)
    batch = ReplayBatch(
        obs=obs,
        actions=torch.zeros((2, len(S.HEADS)), dtype=torch.long),
        rewards=torch.tensor([1.0, -1.0]),
        next_obs=obs.clone(),
        discounts=torch.full((2,), 0.99),
        bootstrap_masks=torch.ones((2, 2), dtype=torch.bool),
    )
    before = compute_bootstrapped_td_loss(online, target, batch)
    with torch.no_grad():
        for network in (online, target):
            for parameter in network.prior_networks.parameters():
                parameter.add_(100.0)
    after = compute_bootstrapped_td_loss(online, target, batch)

    for before_value, after_value in zip(before, after):
        assert torch.equal(before_value, after_value)


def test_resume_round_trip_restores_networks_optimizer_and_counters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(2)
    online = BranchingDQN()
    target = BranchingDQN()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=3e-4)
    with torch.no_grad():
        next(online.parameters()).fill_(0.125)
        next(target.parameters()).fill_(-0.25)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, online, target, optimizer, env_steps=1234, updates=567)
    old_format = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("ucb_state", "archive", "dist", "quantiles", "noisy"):
        del old_format[key]
    torch.save(old_format, checkpoint)

    restored_online = BranchingDQN()
    restored_target = BranchingDQN()
    restored_optimizer = torch.optim.Adam(restored_online.parameters(), lr=0.1)
    counters = load_checkpoint(
        checkpoint,
        restored_online,
        restored_target,
        restored_optimizer,
        torch.device("cpu"),
        explorer=UCBExplorer(),
        archive=MilestoneArchive(),
    )

    assert counters == (1234, 567)
    assert torch.equal(next(restored_online.parameters()), next(online.parameters()))
    assert torch.equal(next(restored_target.parameters()), next(target.parameters()))
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert (
        "resume_initialized_fresh missing=ucb_counts,milestone_archive"
        in capsys.readouterr().out
    )


def test_rainbow_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(11)
    online = QuantileBranchingDQN(num_quantiles=3, noisy=True)
    target = QuantileBranchingDQN(num_quantiles=3, noisy=True)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=6.25e-5)
    checkpoint = tmp_path / "rainbow.pt"
    explorer = UCBExplorer(key_mode="rung")
    explorer.counts[(-1, "WAIT", 0)] = 2
    explorer.total = 2
    archive = MilestoneArchive()
    save_checkpoint(
        checkpoint,
        online,
        target,
        optimizer,
        env_steps=321,
        updates=123,
        explorer=explorer,
        archive=archive,
        behaviour_steps={"27000": 111, "27001": 222},
        checkpoint_args={"action_regime": "bare"},
    )

    restored_online = QuantileBranchingDQN(num_quantiles=3, noisy=True)
    restored_target = QuantileBranchingDQN(num_quantiles=3, noisy=True)
    restored_optimizer = torch.optim.Adam(restored_online.parameters(), lr=0.1)
    restored_explorer = UCBExplorer(key_mode="rung")
    restored_behaviour_steps: dict[str, int] = {}
    restored_checkpoint_args: dict[str, object] = {}
    counters = load_checkpoint(
        checkpoint,
        restored_online,
        restored_target,
        restored_optimizer,
        torch.device("cpu"),
        explorer=restored_explorer,
        archive=MilestoneArchive(),
        behaviour_steps=restored_behaviour_steps,
        checkpoint_args=restored_checkpoint_args,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert counters == (321, 123)
    assert payload["dist"] == "qr"
    assert payload["quantiles"] == 3
    assert payload["noisy"] is True
    assert payload["args"]["action_regime"] == "bare"
    assert restored_explorer.counts == explorer.counts
    assert restored_behaviour_steps == {"27000": 111, "27001": 222}
    assert restored_checkpoint_args["action_regime"] == "bare"
    assert linear_ucb_coefficient(
        restored_behaviour_steps["27001"], 0.75, 0.15, 400
    ) == pytest.approx(0.417)
    assert all(
        torch.equal(restored, original)
        for restored, original in zip(
            restored_online.state_dict().values(), online.state_dict().values()
        )
    )


class _SlowStepEnv:
    def reset(self, *, seed: int | None = None):
        return maximally_masked_observations(1)[0].numpy(), {}

    def step(self, action):
        time.sleep(0.05)
        obs = maximally_masked_observations(1)[0].numpy()
        return obs, 0.0, False, False, {}

    def close(self) -> None:
        pass


def test_worker_timeout_kills_only_the_stuck_environment() -> None:
    envs = ThreadVectorEnv(
        [lambda: _SlowStepEnv(), lambda: _SlowStepEnv()],
        labels=["slow", "healthy"],
        timeout_seconds=0.01,
    )
    envs.reset(1)
    envs.envs[1].step = lambda action: (
        maximally_masked_observations(1)[0].numpy(),
        0.0,
        False,
        False,
        {},
    )
    actions = np.zeros((2, len(S.HEADS)), dtype=np.int64)

    result = envs.step(actions)

    assert result.env_indices.tolist() == [1]
    assert envs.alive.tolist() == [False, True]
    assert envs.dead_events[0]["env"] == "slow"
    envs.close()


def test_excursion_transitions_enter_replay_with_tag() -> None:
    accumulator = NStepAccumulator(n=1)
    observation = maximally_masked_observations(1)[0].numpy()
    transition = accumulator.append(
        observation,
        np.zeros(len(S.HEADS), dtype=np.int64),
        3.5,
        observation,
        True,
        behaviour="ucb",
        excursion=True,
    )[0]
    replay = ReplayBuffer(capacity=2)
    replay.add(transition)

    batch = replay.sample(1, np.random.default_rng(0), torch.device("cpu"))

    assert transition.excursion is True
    assert replay.excursions[: replay.size].tolist() == [True]
    assert batch.excursions is not None
    assert batch.excursions.tolist() == [True]
    assert batch.rewards.tolist() == [3.5]


def test_excursion_episode_is_excluded_from_root_counters_and_median() -> None:
    milestones = EpisodeMilestones.from_frontier(3)
    milestones.reached[4] = True
    milestones.reached[5] = True
    milestones.first_attainment[4] = 2
    milestones.first_attainment[5] = 4
    root_counts = Counter()
    conversion_counts = Counter()
    excursion_counts = Counter()
    root_rewards: deque[float] = deque(maxlen=100)

    record_episode_counters(
        milestones,
        automated_ore=True,
        excursion=True,
        reward=999.0,
        root_counts=root_counts,
        conversion_counts=conversion_counts,
        excursion_counts=excursion_counts,
        root_rewards=root_rewards,
    )

    assert root_counts == Counter()
    assert conversion_counts == Counter()
    assert list(root_rewards) == []
    assert excursion_counts == Counter(
        drill_placed=1,
        drill_fuelled=1,
        automated_ore_produced=1,
    )


def test_checkpoint_keeps_only_frontier_metadata_and_resume_has_no_state(
    tmp_path: Path,
) -> None:
    online = BranchingDQN()
    target = BranchingDQN()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters())
    archive = MilestoneArchive()
    frontier = ArchivedFrontier(
        state=object(), restore_hash="abc123", rung=3, step=47
    )
    archive.add_episode([_transition(1.0)], 3, 1.0, frontier=frontier)
    checkpoint = tmp_path / "frontier.pt"

    save_checkpoint(
        checkpoint,
        online,
        target,
        optimizer,
        env_steps=10,
        updates=2,
        archive=archive,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_frontier = payload["archive"]["episodes"][0]["frontier"]
    restored_archive = MilestoneArchive()
    load_checkpoint(
        checkpoint,
        BranchingDQN(),
        BranchingDQN(),
        None,
        torch.device("cpu"),
        archive=restored_archive,
    )
    resumed_frontier = next(iter(restored_archive.episodes.values())).frontier

    assert saved_frontier == {"hash": "abc123", "rung": 3, "step": 47}
    assert resumed_frontier is not None
    assert resumed_frontier.state is None


def test_frontier_return_cli_defaults_off_and_accepts_controls() -> None:
    defaults = parse_args([])
    bare = parse_args(["--action-regime", "bare"])
    clamp_overrides = parse_args(
        [
            "--no-clamp-quantity",
            "--clamp-item",
            "--clamp-anchor",
            "--clamp-connect",
        ]
    )
    legacy_quantity_off = parse_args(["--no-quantity-aware-support"])
    enabled = parse_args(
        [
            "--frontier-return",
            "--return-prob",
            "0.25",
            "--return-budget",
            "12",
            "--return-rung",
            "gears_crafted",
            "--forced-arg-eps",
            "0.3",
            "--frontier-force-tries",
            "24",
            "--frontier-force-max-count",
            "4",
        ]
    )

    assert defaults.frontier_return is False
    assert defaults.action_regime == "macro"
    assert bare.action_regime == "bare"
    assert defaults.return_prob == pytest.approx(0.5)
    assert defaults.return_budget == 64
    assert defaults.return_rung == "drill_crafted"
    assert defaults.clamp_quantity is True
    assert defaults.quantity_aware_support is True
    assert defaults.clamp_item is False
    assert defaults.clamp_anchor is False
    assert defaults.clamp_connect is False
    assert defaults.support_cooldowns is True
    assert defaults.forced_arg_eps == 0.0
    assert defaults.frontier_force_tries == 0
    assert defaults.frontier_force_max_count == 2
    assert clamp_overrides.clamp_quantity is False
    assert clamp_overrides.quantity_aware_support is False
    assert clamp_overrides.clamp_item is True
    assert clamp_overrides.clamp_anchor is True
    assert clamp_overrides.clamp_connect is True
    assert legacy_quantity_off.clamp_quantity is False
    assert enabled.frontier_return is True
    assert enabled.return_prob == pytest.approx(0.25)
    assert enabled.return_budget == 12
    assert enabled.return_rung == "gears_crafted"
    assert enabled.forced_arg_eps == pytest.approx(0.3)
    assert enabled.frontier_force_tries == 24
    assert enabled.frontier_force_max_count == 4
    assert is_new_return_frontier(1, 2, RUNG_NAMES.index("gears_crafted"))
    assert not is_new_return_frontier(1, 2, RUNG_NAMES.index("drill_crafted"))


def test_max_steps_by_port_builds_distinct_fake_horizons() -> None:
    args = parse_args(
        [
            "--fake",
            "--ports",
            "27000,27001",
            "--max-steps-by-port",
            "256,512",
            "--behaviour-by-port",
            "epsilon,ucb",
        ]
    )

    envs = [make_env() for make_env in _make_env_fns(args)]

    assert [env.horizon for env in envs] == [256, 512]
    assert [env._episode_max_steps for env in envs] == [256, 512]
