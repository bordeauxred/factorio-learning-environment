"""Offline contract tests for the V0 observation pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fle.commons.observation.buildability import MANIFEST
from fle.rl import schema as legacy_schema
from fle.rl.ops import VocabData
from fle.rl.v0_obs import (
    MaskToggles,
    build_contained_item_mask,
    build_place_location_mask,
    build_v0_masks,
    build_v0_observation,
)
from fle.rl.v0_schema import (
    BUILD_NEVER_SAMPLED,
    HEAD_SIZES,
    ITEM_INDEX,
    LOCAL_CHANNEL_INDEX,
    OBS_SPEC,
    PLACEABLE_INDEX,
    TECH_INDEX,
    obs_nbytes,
    world_to_cell,
)
from fle.rl.world import EntityRow, WorldClient


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for this module."""
    yield


class _NoFloat32Cast(np.ndarray):
    """Array view that fails if the build path regresses to a float32 cast."""

    def astype(self, dtype, *args, **kwargs):
        assert np.dtype(dtype) != np.dtype(np.float32)
        return super().astype(dtype, *args, **kwargs)


class _Rcon:
    def __init__(self, response: str = ""):
        self.response = response
        self.commands: list[str] = []

    def send_command(self, command: str) -> str:
        self.commands.append(command)
        return self.response


def _build_meta() -> str:
    return f"B{MANIFEST['id']},1,1,1,-64,-64,128"


def _water_mask(x: int, y: int) -> int:
    bit = (31 - y % 32) * 32 + x % 32
    return 1 << bit


def _world() -> WorldClient:
    world = WorldClient(_Rcon(), object())
    world.apply_terrain(_build_meta())
    # Centre block is fresh; its eastern neighbour is known but stale.
    world.apply_terrain("b1,0,0,0,90,0000000180000000")
    world.apply_terrain("b1,0,1,0,1,ffffffffffffffff")
    world.apply_terrain("M1,1,1,1,-256,-256,100,0,0,0,generated")
    world.apply_terrain("C1,0,0,80,0:1,128:200")
    world.apply_terrain(f"c0:0:{_water_mask(8, 9):x}")
    world.apply_terrain("oiron-ore:11:9:255;t12:9;k14:9;n5,biter-spawner,13.5,9.5")
    world.apply_entity("u1,assembling-machine-1,16.5,9.5,4,1,W3:3")
    world.player_x, world.player_y, world.tick = 10.5, 10.5, 100
    return world


def _observation(world: WorldClient, *, own: bool = False) -> dict[str, np.ndarray]:
    return build_v0_observation(
        world,
        player_x=10.5,
        player_y=10.5,
        tick=100,
        inventory={"stone-furnace": 2, "iron-plate": 5},
        enabled_recipes=("iron-gear-wheel",),
        research_state={
            "automation": {
                "enabled": True,
                "researched": False,
                "prerequisites": [],
            }
        },
        episode_start_tick=40,
        automated_score=12.0,
        general_score=34.0,
        build_fresh_ticks=60,
        own=own,
    )


def test_every_observation_block_matches_contract():
    world = _world()
    obs = _observation(world)

    assert set(obs) == set(OBS_SPEC)
    for name, (shape, dtype) in OBS_SPEC.items():
        assert obs[name].shape == shape, name
        assert obs[name].dtype == dtype, name


def test_buildability_stays_int8_unknown_and_zero_copy():
    world = _world()
    world.buildability.values = world.buildability.values.view(_NoFloat32Cast)

    borrowed = _observation(world)
    assert borrowed["buildability"] is world.buildability.values
    assert borrowed["buildability"].dtype == np.int8
    assert np.any(borrowed["buildability"] == -1)

    owned = _observation(world, own=True)
    assert owned["buildability"].dtype == np.int8
    assert not np.shares_memory(owned["buildability"], world.buildability.values)


def test_build_age_preserves_never_sampled_and_known_ages():
    obs = _observation(_world())

    assert obs["build_age"][0, 8, 8] == 10
    assert obs["build_age"][0, 8, 9] == 99
    assert obs["build_age"][0, 0, 0] == BUILD_NEVER_SAMPLED
    assert len(np.unique(obs["build_age"])) >= 3


def test_globals_append_the_seven_contract_values():
    obs = _observation(_world())
    extra = obs["globals"][-7:]

    np.testing.assert_allclose(
        extra[:4],
        (100 / 1e6, 12 / 1e3, 34 / 1e3, 60 / 60 / 3600),
    )
    assert np.isclose(extra[4], 128 / (63 * 128 * 128))
    assert np.isclose(extra[5], 1 / (63 * 16 * 16))
    assert np.isclose(extra[6], 64 / (128 * 128))


def test_local_exact_coordinates_amount_and_real_footprint():
    obs = _observation(_world())
    local = obs["local_exact"]

    def value(channel: str, x: int, y: int) -> np.float16:
        cell = world_to_cell(x, y, 10.5, 10.5)
        row, column = divmod(cell, 64)
        return local[LOCAL_CHANNEL_INDEX[channel], row, column]

    assert value("water", 8, 9) == 1
    assert value("iron_ore", 11, 9) == 1
    assert np.isclose(value("resource_amount", 11, 9), math.log1p(255) / 16)
    assert value("tree", 12, 9) == 1
    assert value("rock_cliff", 14, 9) == 1
    assert value("enemy_occupancy", 13, 9) == 1
    assert value("player", 10, 10) == 1
    assert value("unknown_terrain", -1, 10) == 1
    assert value("unknown_terrain", 8, 9) == 0

    # A 3x3 entity centred on (16.5, 9.5) occupies x=15..17, y=8..10.
    for y in range(8, 11):
        for x in range(15, 18):
            assert value("friendly_occupancy", x, y) == 1
    assert value("friendly_occupancy", 14, 9) == 0
    assert value("friendly_occupancy", 18, 9) == 0


def test_rectangular_footprint_rotates_with_entity_direction():
    world = _world()
    world.entities[1] = EntityRow(
        unit=1,
        name="boiler",
        x=16.0,
        y=10.5,
        direction=4,
        status=1,
        tile_width=3,
        tile_height=2,
    )
    local = _observation(world)["local_exact"]
    occupied = local[LOCAL_CHANNEL_INDEX["friendly_occupancy"]]

    # East-facing swaps the prototype's 3x2 footprint to 2x3.
    for y in range(9, 12):
        for x in range(15, 17):
            cell = world_to_cell(x, y, 10.5, 10.5)
            row, column = divmod(cell, 64)
            assert occupied[row, column] == 1


def test_minimap_known_and_age_include_never_sampled_cells():
    obs = _observation(_world())
    minimap = obs["minimap"]

    assert minimap[14, 64, 64] == 1
    assert np.isclose(minimap[15, 64, 64], 20 / 3600, atol=1e-5)
    assert minimap[14, 0, 0] == 0
    assert minimap[15, 0, 0] == 1


def test_masks_toggles_and_prefix_helpers():
    world = _world()
    obs = _observation(world)
    raw = VocabData.load(legacy_schema.VOCAB_PATH)
    masks = build_v0_masks(
        world,
        obs,
        inventory={"stone-furnace": 2, "iron-plate": 5},
        enabled_recipes=("iron-gear-wheel",),
        research_state={
            "automation": {
                "enabled": True,
                "researched": False,
                "prerequisites": [],
            }
        },
        vocab=raw,
    )
    assert set(masks) == set(MaskToggles().active)
    for name, mask in masks.items():
        assert mask.shape == (HEAD_SIZES[name],)
        assert mask.dtype == np.uint8
    assert masks["place_item"][PLACEABLE_INDEX["stone-furnace"]] == 1
    assert masks["inventory_item"][ITEM_INDEX["iron-plate"]] == 1
    assert masks["entity"].sum() == 1
    assert masks["recipe"][legacy_schema.RECIPE_INDEX["iron-gear-wheel"]] == 1
    assert masks["technology"][TECH_INDEX["automation"]] == 1

    disabled = build_v0_masks(
        world,
        obs,
        inventory={},
        enabled_recipes=(),
        research_state={},
        toggles=MaskToggles(recipe=False),
    )
    assert disabled["recipe"].all()
    assert MaskToggles(recipe=False).as_dict()["recipe"] is False

    location = build_place_location_mask(
        world,
        place_item="stone-furnace",
        direction=0,
        player_x=10.5,
        player_y=10.5,
        tick=100,
    )
    assert location.shape == (HEAD_SIZES["location"],)
    assert location.dtype == np.uint8
    contained = build_contained_item_mask(
        EntityRow(
            unit=9,
            name="iron-chest",
            x=0.5,
            y=0.5,
            direction=0,
            status=1,
            inventories={1: {"iron-plate": 3}},
        )
    )
    assert contained[ITEM_INDEX["iron-plate"]] == 1


def test_combined_drain_routes_caches_before_terrain_records():
    response = (
        "h100:-:0:1.5:2.5~"
        + _build_meta()
        + ";M1,1,1,1,-256,-256,100,0,0,0,generated;c0:0:"
    )
    rcon = _Rcon(response)
    world = WorldClient(rcon, object())

    assert world.obs_all_drain() == 4
    assert world.buildability.values.shape == (63, 128, 128)
    assert world.minimap.values.shape == (14, 128, 128)
    assert (0, 0) in world.water

    config_rcon = _Rcon("ok")
    assert world.configure_v0_caches(config_rcon) == ("ok", "ok")
    assert "size=128" in config_rcon.commands[0]
    assert 'visibility="generated"' in config_rcon.commands[1]


def test_obs_nbytes_matches_a_built_observation():
    obs = _observation(_world())
    assert obs_nbytes() == sum(block.nbytes for block in obs.values())
