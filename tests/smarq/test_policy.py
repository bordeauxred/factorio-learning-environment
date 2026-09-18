import numpy as np
import pytest
import torch

from fle.smarq import contract as C
from fle.smarq.fake import FakeSemanticEnv
from fle.smarq.net import SMARQNetwork
from fle.smarq.policy import AutoregressivePolicy


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Shadow the repository fixture that requires a live Factorio server."""
    yield


def _zero_network(network):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()


def test_greedy_walks_contract_sequence_and_decodes_action():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    observation, masks = env.reset(0)
    network = SMARQNetwork(env.vocab, 16)
    _zero_network(network)
    with torch.no_grad():
        network.verb_head[-1].bias[C.VERB_INDEX["PLACE"]] = 3
        network.discrete_heads[C.PROTOTYPE][-1].bias[2] = 4
        network.discrete_heads[C.DIRECTION][-1].bias[3] = 2
    policy = AutoregressivePolicy(network, env.vocab)
    action = policy.greedy(observation, masks)

    assert action.verb == "PLACE"
    assert action.prototype == "burner-mining-drill"
    assert action.tile == observation.raster_origin
    assert action.direction == "WEST"
    assert action.heads[C.HEAD_INDEX[C.PROTOTYPE]] == 2
    assert action.heads[C.HEAD_INDEX[C.POSITION]] == 0
    assert action.heads[C.HEAD_INDEX[C.DIRECTION]] == 3
    unused = set(C.HEADS) - set(C.HEAD_SEQUENCE["PLACE"])
    assert all(action.head(head) == -1 for head in unused)


def test_per_head_epsilon_explores_position_and_not_whole_tuple():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    observation, masks = env.reset(0)
    network = SMARQNetwork(env.vocab, 16)
    _zero_network(network)
    with torch.no_grad():
        network.verb_head[-1].bias[C.VERB_INDEX["MOVE_TO"]] = 2
    policy = AutoregressivePolicy(network, env.vocab, seed=4)
    epsilon = {"verb": 0.0, C.POSITION: 1.0}
    actions = [policy.action(observation, masks, epsilon) for _ in range(30)]
    assert {action.verb for action in actions} == {"MOVE_TO"}
    assert len({action.head(C.POSITION) for action in actions}) > 10


def test_partial_value_is_maximum_over_next_legal_head():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    observation, masks = env.reset(0)
    network = SMARQNetwork(env.vocab, 16)
    _zero_network(network)
    with torch.no_grad():
        network.discrete_heads[C.PROTOTYPE][-1].bias.copy_(
            torch.tensor([100.0, 1.0, 5.0, 3.0, 2.0])
        )
    policy = AutoregressivePolicy(network, env.vocab)
    value = policy.partial_value(observation, masks, verb="PLACE")
    # Prototype zero has the largest raw value but is structurally masked.
    assert np.isclose(value, 5.0)
