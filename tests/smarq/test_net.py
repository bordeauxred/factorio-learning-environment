import pytest
import torch

from fle.smarq import contract as C
from fle.smarq.fake import FakeSemanticEnv
from fle.smarq.net import SMARQNetwork


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Shadow the repository fixture that requires a live Factorio server."""
    yield


@pytest.mark.parametrize("raster_tiles", [16, 96])
def test_network_shapes_and_parameter_budget(raster_tiles):
    env = FakeSemanticEnv(raster_tiles=raster_tiles, entity_slots=32)
    observation, _ = env.reset(0)
    network = SMARQNetwork(env.vocab, raster_tiles)
    encoded = network(observation)

    assert network.parameter_count < 10_000_000
    assert encoded.z_state.shape == (1, 256)
    assert encoded.spatial.shape == (1, 128, 24, 24)
    assert encoded.entities.shape == (1, 32, 128)
    assert network.q_verbs(encoded).shape == (1, C.N_VERBS)
    assert network.q_head(encoded, C.POSITION, C.VERB_INDEX["PLACE"], {}).shape == (
        1,
        raster_tiles**2,
    )
    assert network.q_head(encoded, C.ENTITY, C.VERB_INDEX["INSERT"], {}).shape == (1, 32)
    assert network.q_head(encoded, C.PROTOTYPE, C.VERB_INDEX["PLACE"], {}).shape == (
        1,
        len(env.vocab.prototypes),
    )


def test_all_later_heads_depend_on_the_full_prefix():
    torch.manual_seed(2)
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    observation, _ = env.reset(0)
    network = SMARQNetwork(env.vocab, 16)
    encoded = network(observation)
    verb = C.VERB_INDEX["PLACE"]
    q_without = network.q_head(encoded, C.DIRECTION, verb, {C.PROTOTYPE: 1, C.POSITION: 0})
    q_changed_position = network.q_head(
        encoded, C.DIRECTION, verb, {C.PROTOTYPE: 1, C.POSITION: 17}
    )
    q_changed_prototype = network.q_head(
        encoded, C.DIRECTION, verb, {C.PROTOTYPE: 2, C.POSITION: 0}
    )
    assert not torch.allclose(q_without, q_changed_position)
    assert not torch.allclose(q_without, q_changed_prototype)


def test_entity_categorical_fields_use_embeddings():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=8)
    network = SMARQNetwork(env.vocab, 16)
    assert isinstance(network.entity_type_embedding, torch.nn.Embedding)
    assert isinstance(network.entity_recipe_embedding, torch.nn.Embedding)
    assert isinstance(network.entity_fluid_embedding, torch.nn.Embedding)
    assert isinstance(network.entity_item_embedding, torch.nn.Embedding)


def test_entity_capacity_is_configurable_and_2048_still_works():
    env = FakeSemanticEnv(raster_tiles=16, entity_slots=C.ENTITY_SLOTS)
    observation, _ = env.reset(0)
    compact = SMARQNetwork(env.vocab, 16, entity_slots=512)
    full = SMARQNetwork(env.vocab, 16, entity_slots=C.ENTITY_SLOTS)
    assert compact.encode(observation).entities.shape == (1, 512, 128)
    assert full.encode(observation).entities.shape == (1, C.ENTITY_SLOTS, 128)
