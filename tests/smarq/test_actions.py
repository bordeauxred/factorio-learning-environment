from pathlib import Path

import numpy as np

from fle.smarq import contract as C
from fle.smarq.actions import (
    ActionCodec,
    EntityCapability,
    MaskMetadata,
    build_masks,
)
from fle.smarq.contract import Observation
from fle.smarq.obs import F_INV_TOTAL, TensorClient
from fle.smarq.raster import ExactTileRaster
from fle.smarq.vocab import StableVocab


def vocab() -> StableVocab:
    return StableVocab(
        prototypes=("<none>", "stone-furnace", "wooden-chest"),
        items=("<none>", "coal", "iron-ore"),
        recipes=("<none>", "iron-gear-wheel", "iron-plate"),
        technologies=("<none>", "automation", "logistics"),
        entity_types=("<none>", "stone-furnace", "wooden-chest"),
        game_version="test",
    )


def observation(origin=(-48, -48), side=96) -> Observation:
    ids = np.zeros(C.ENTITY_SLOTS, dtype=np.int64)
    ids[0] = 1234
    return Observation(
        grid=np.zeros((C.GRID_CHANNELS, C.GRID_SIZE, C.GRID_SIZE), np.float32),
        entity_view=np.zeros((C.ENTITY_SLOTS, C.ENTITY_FEATURES), np.float32),
        entity_mask=ids != 0,
        entity_ids=ids,
        globals=np.zeros(C.N_GLOBALS, np.float32),
        raster=np.zeros((C.RASTER_CHANNELS, side, side), np.float32),
        raster_origin=origin,
        raster_tiles=side,
        player_tile=(0, 0),
        tick=10,
        production_score=0.0,
        automated_production_score=0.0,
    )


def test_exact_position_round_trip() -> None:
    obs = observation()
    codec = ActionCodec(vocab())
    heads = C.empty_heads()
    heads[C.HEAD_INDEX[C.PROTOTYPE]] = 1
    heads[C.HEAD_INDEX[C.POSITION]] = 17 * 96 + 73
    heads[C.HEAD_INDEX[C.DIRECTION]] = 2
    action = codec.decode(C.VERB_INDEX["PLACE"], heads, obs)
    assert action.tile == (25, -31)
    verb, encoded = codec.encode(action, obs)
    assert verb == C.VERB_INDEX["PLACE"]
    np.testing.assert_array_equal(encoded, heads)


def test_entity_id_is_captured_when_pointer_is_decoded() -> None:
    obs = observation()
    heads = C.empty_heads()
    heads[C.HEAD_INDEX[C.ENTITY]] = 0
    action = ActionCodec(vocab()).decode(C.VERB_INDEX["PICKUP"], heads, obs)
    assert action.entity_slot == 0
    assert action.entity_id == 1234


def test_masks_are_structural_and_position_has_no_mask() -> None:
    stable_vocab = vocab()
    client = TensorClient(vocab=stable_vocab)
    client.apply_entity("h1:-:0:0:0;u1234,stone-furnace,3,4,0,1,W2:2")
    metadata = MaskMetadata(
        placeable={"stone-furnace", "wooden-chest"},
        recipe_category={"iron-gear-wheel": "crafting", "iron-plate": "smelting"},
        hand_categories={"crafting"},
        entity={
            "stone-furnace": EntityCapability(
                holds_items=True,
                recipe_categories=frozenset({"smelting"}),
                accepted_items=frozenset({"coal", "iron-ore"}),
            )
        },
        researched={"automation"},
        startable_technologies={"logistics"},
    )
    first = build_masks(stable_vocab, client, metadata)
    assert not hasattr(first, C.POSITION)
    assert first.prototype.tolist() == [False, True, True]
    assert first.craft_recipe.tolist() == [False, True, False]
    assert first.technology.tolist() == [False, False, True]
    assert first.entity[C.VERB_INDEX["INSERT"], 0]
    assert first.recipe_for_entity(0).tolist() == [False, False, True]

    # Feature magnitudes and every raster plane are deliberately absent inputs.
    client.table[:, F_INV_TOTAL] = 1_000_000
    second = build_masks(stable_vocab, client, metadata)
    np.testing.assert_array_equal(first.prototype, second.prototype)
    np.testing.assert_array_equal(first.item, second.item)
    np.testing.assert_array_equal(first.entity, second.entity)


def test_entity_item_mask_uses_type_compatibility_not_contents() -> None:
    stable_vocab = StableVocab(
        prototypes=("<none>", "burner-mining-drill"),
        items=("<none>", "coal", "copper-plate", "iron-plate"),
        recipes=("<none>",),
        technologies=("<none>",),
        entity_types=("<none>", "burner-mining-drill"),
        game_version="recorded-2.0.73",
    )
    client = TensorClient(vocab=stable_vocab)
    client.apply_entity("h1:-:0:0:0;u9,burner-mining-drill,1,2,0,1,W2:2")
    metadata = MaskMetadata(
        placeable={"burner-mining-drill"},
        entity={
            "burner-mining-drill": EntityCapability(
                holds_items=True,
                accepted_items=frozenset({"coal"}),
            )
        },
    )

    masks = build_masks(stable_vocab, client, metadata)
    for verb in ("INSERT", "EXTRACT"):
        assert masks.entity[C.VERB_INDEX[verb], 0]
    assert masks.item_for_entity(0).tolist() == [False, True, False, False]
    # The global item head remains independent of inventory/state contents.
    assert masks.item.tolist() == [False, True, True, True]


def test_unstartable_technologies_are_masked() -> None:
    stable_vocab = vocab()
    client = TensorClient(vocab=stable_vocab)
    metadata = MaskMetadata(
        startable_technologies={"logistics"},
        researched={"automation"},
    )
    masks = build_masks(stable_vocab, client, metadata)
    assert masks.technology.tolist() == [False, False, True]


def test_coarse_move_off_preserves_frozen_head_layout_and_sizes() -> None:
    stable_vocab = vocab()
    expected_heads = (
        "position",
        "prototype",
        "direction",
        "entity",
        "item",
        "quantity",
        "recipe",
        "technology",
        "duration",
    )
    expected_sizes = stable_vocab.head_sizes(96)
    codec = ActionCodec(stable_vocab, coarse_move=False)

    assert C.HEADS == expected_heads
    assert C.HEAD_INDEX == {name: index for index, name in enumerate(expected_heads)}
    assert stable_vocab.head_sizes(96) == expected_sizes
    assert not codec.coarse_move


def test_coarse_move_reaches_three_tile_grid_without_changing_place_decode() -> None:
    obs = observation()
    move_heads = C.empty_heads()
    move_heads[C.HEAD_INDEX[C.POSITION]] = 31 * C.GRID_SIZE + 43
    codec = ActionCodec(vocab(), coarse_move=True)
    move = codec.decode(C.VERB_INDEX["MOVE_TO"], move_heads, obs)
    assert move.tile == (-15, -51)

    place_heads = C.empty_heads()
    place_heads[C.HEAD_INDEX[C.PROTOTYPE]] = 1
    place_heads[C.HEAD_INDEX[C.POSITION]] = 31 * 96 + 43
    place_heads[C.HEAD_INDEX[C.DIRECTION]] = 0
    place = codec.decode(C.VERB_INDEX["PLACE"], place_heads, obs)
    assert place.tile == (-5, -17)


def test_executor_module_has_no_geometry_substitution_helpers() -> None:
    source = (Path(__file__).parents[2] / "fle/smarq/env.py").read_text(encoding="utf-8")
    forbidden = ("nearest_buildable", "place_entity_next_to", "projection")
    assert all(name not in source for name in forbidden)


def test_exact_raster_supports_target_288_side() -> None:
    client = TensorClient(vocab=vocab())
    client.apply_entity("h1:-:0:12.5:-3.5;u9,stone-furnace,15.5,-2.5,0,1,W2:2")
    raster, origin, player = ExactTileRaster(288).build(client)
    assert raster.shape == (len(C.RASTER_CHANNEL_NAMES), 288, 288)
    assert player == (12, -4)
    assert origin == (-132, -148)
    assert raster[C.RASTER_CHANNEL_NAMES.index("player"), 144, 144] == 1
