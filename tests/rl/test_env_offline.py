from __future__ import annotations

import inspect
import math
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from fle.env.entities import Position
from fle.env.game_types import Prototype
from fle.rl import schema as S
from fle.rl.env import (
    DELAYED_DRAIN_OPS,
    PEACEFUL_RESET_COMMAND,
    FleMacroEnv,
    FrontierState,
    _effective_max_ticks,
    _random_acceptance_action,
)
from fle.rl.fake_env import FakeMacroEnv
from fle.rl.observation import (
    HarvestTarget,
    MacroObservationBuilder,
    ObservationInput,
    split_observation,
)
from fle.rl.ops import (
    ActionSpec,
    OperationSnapshot,
    VocabData,
    entity_is_rotatable,
    execute_action,
    insertable_items,
    verify_effect,
)
from fle.rl.world import EntityRow, WorldClient


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for this module."""
    yield


class FakeWorld:
    def __init__(self, *, entities=None, ores=None, trees=None, water=None, obstacles=None):
        self.entities = entities or {}
        self.ores = ores or {}
        self.trees = trees or set()
        self.water = water or {}
        self.obstacles = obstacles or set()

    def patches(self):
        return WorldClient.patches(self)

    def is_water(self, x, y):
        return (x, y) in self.water


def _vocab() -> VocabData:
    return VocabData.from_dict(
        {
            "items": [
                {"name": "coal", "place_result": None},
                {"name": "iron-ore", "place_result": None},
                {"name": "wood", "place_result": None},
                {"name": "wooden-chest", "place_result": "wooden-chest"},
            ],
            "recipes": [
                {
                    "name": "wooden-chest",
                    "category": "crafting",
                    "ingredients": [{"name": "wood", "amount": 2}],
                    "products": [{"name": "wooden-chest", "amount": 1}],
                },
                {
                    "name": "iron-plate",
                    "category": "smelting",
                    "products": [{"name": "iron-plate", "amount": 1}],
                },
            ],
            "technologies": [
                {"name": "automation-science-pack", "prerequisites": []},
                {"name": "electronics", "prerequisites": []},
            ],
            "entities": [
                {
                    "name": "iron-ore",
                    "type": "resource",
                    "tile_width": 1,
                    "tile_height": 1,
                },
                {
                    "name": "mystery-resource",
                    "type": "resource",
                    "tile_width": 1,
                    "tile_height": 1,
                },
                {
                    "name": "wooden-chest",
                    "type": "container",
                    "tile_width": 1,
                    "tile_height": 1,
                },
                {
                    "name": "assembling-machine-1",
                    "type": "assembling-machine",
                    "tile_width": 3,
                    "tile_height": 3,
                },
            ],
        }
    )


def _snapshot(*, inventory=None, entities=None, position=(0.0, 0.0)) -> OperationSnapshot:
    return OperationSnapshot(
        inventory=inventory or {},
        entities=entities or {},
        position=position,
        tick=100,
        score_player=0.0,
        score_automated=0.0,
        current_research=None,
    )


def _research_state():
    return {
        "automation-science-pack": {
            "enabled": True,
            "researched": False,
            "prerequisites": [],
        },
        "electronics": {
            "enabled": True,
            "researched": False,
            "prerequisites": [],
        },
    }


def _builder(world: FakeWorld, *, regime: str = "macro") -> MacroObservationBuilder:
    return MacroObservationBuilder(
        world=world,
        vocab=_vocab(),
        crafting_categories=("crafting",),
        resource_reach=2.7,
        build_reach=10.0,
        trigger_technologies={"electronics"},
        status_names={1: "working", 53: "no_fuel"},
        max_steps=256,
        max_ticks=216_000,
        regime=regime,
    )


def _frame(
    world: FakeWorld,
    *,
    snapshot: OperationSnapshot | None = None,
    enabled_recipes=(),
    regime: str = "macro",
):
    return _builder(world, regime=regime).build(
        ObservationInput(
            snapshot=snapshot or _snapshot(entities=world.entities),
            enabled_recipes=enabled_recipes,
            research_state=_research_state(),
            episode_start_tick=100,
            step_count=0,
            last_status=None,
            last_op=None,
        )
    )


def _valid_indices(obs: np.ndarray, head: str) -> set[int]:
    a, b = S.MASK_OFFSETS[head]
    return set(np.flatnonzero(obs[a:b] > 0.5))


def _place_step_env(
    item: str,
    *,
    ores: dict[tuple[int, int], tuple[str, int]],
    place_error: str | None = None,
):
    before = _snapshot(inventory={item: 1}, position=(0.5, 0.5))
    frame = SimpleNamespace(
        obs=np.ones(S.OBS_SIZE, dtype=np.float32),
        targets=(None,) * S.N_TARGET_SLOTS,
        entities=(None,) * S.N_ENTITY_SLOTS,
    )
    world = FakeWorld(ores=ores)
    live_position = [*before.position]
    placed_position: list[Position] = []

    def place_entity(_prototype, _direction, position, *, exact):
        assert exact is True
        if place_error is not None:
            raise RuntimeError(place_error)
        placed_position.append(position)

    namespace = SimpleNamespace(
        move_to=Mock(),
        place_entity=Mock(side_effect=place_entity),
    )
    world.all_drain = Mock()
    world.read_player_pos = Mock(side_effect=lambda: tuple(live_position))
    world.techs_finished = []

    env = FleMacroEnv.__new__(FleMacroEnv)
    env.action_space = gym.spaces.MultiDiscrete(S.HEAD_DIMS)
    env.vocab = VocabData.load(S.VOCAB_PATH)
    env.namespace = namespace
    env.world = world
    env.resource_reach = 2.7
    env.build_reach = 10.0
    env._episode_done = False
    env._current = before
    env._frame = frame
    env.quantity_aware_support = True
    env.support_cooldowns = True
    env._cooldown_action = None
    env._cooldown_fingerprint = None
    env._episode_start_tick = 100
    env._steps = 0
    env._episode_max_steps = 10
    env.max_ticks = 1_000
    env._excursion = False
    env._excursion_steps = 0
    env._excursion_budget = None
    env._aps_previous = 0.0
    env._max_aps = 0.0
    env._last_status = None
    env._last_op = None
    env._op_histogram = Counter()
    env._status_histogram = Counter()
    env._entities_placed = 0
    env._items_harvested = 0
    env._episode_deaths = 0
    env._end_reason = None
    env.deaths = 0
    env._enabled_recipes = []
    env._research_state = {}
    env._restore_hash = None
    env._episode_index = 0
    env._log_file = None

    def snapshot(_fallback):
        prototype = env.vocab.place_results[item]
        entities = {}
        inventory = {item: 1}
        if placed_position:
            position = placed_position[0]
            info = env.vocab.entity_info[prototype]
            entities[99] = EntityRow(
                99,
                prototype,
                position.x,
                position.y,
                0,
                1,
                tile_width=int(info.get("tile_width") or 1),
                tile_height=int(info.get("tile_height") or 1),
            )
            inventory = {}
        return (
            OperationSnapshot(
                inventory=inventory,
                entities=entities,
                position=tuple(live_position),
                tick=120,
                score_player=0.0,
                score_automated=0.0,
                current_research=None,
            ),
            True,
        )

    env._snapshot = Mock(side_effect=snapshot)
    env._build_frame = Mock(return_value=frame)
    records = []
    env._write_log = records.append
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["PLACE"]
    action[S.HEADS.index("placeable")] = S.PLACEABLE_INDEX[item]
    action[S.HEADS.index("offset")] = S.dxdy_to_offset(0, 0)
    return env, namespace, action, records


def test_empty_world_admits_only_wait_move_and_non_trigger_research():
    frame = _frame(FakeWorld())

    assert {S.OPS[index] for index in _valid_indices(frame.obs, "op")} == {
        "WAIT",
        "MOVE",
        "RESEARCH",
    }
    assert _valid_indices(frame.obs, "target") == set()
    assert _valid_indices(frame.obs, "entity") == set()
    assert _valid_indices(frame.obs, "item") == set()
    assert _valid_indices(frame.obs, "placeable") == set()
    assert _valid_indices(frame.obs, "duration") == {0}


def test_one_ore_patch_admits_harvest_and_populates_a_target_slot():
    world = FakeWorld(ores={(20, 0): ("iron-ore", 7), (21, 0): ("iron-ore", 7)})
    frame = _frame(world)

    assert S.OP_INDEX["HARVEST"] in _valid_indices(frame.obs, "op")
    assert _valid_indices(frame.obs, "target") == {0}
    assert frame.targets[0] is not None
    assert frame.targets[0].kind == "iron-ore"
    assert frame.targets[0].position == (20.5, 0.5)


def test_harvest_targets_exclude_oil_uranium_and_resources_without_item_product():
    world = FakeWorld(
        ores={
            (10, 0): ("crude-oil", 1),
            (11, 0): ("uranium-ore", 1),
            (12, 0): ("mystery-resource", 1),
            (13, 0): ("iron-ore", 1),
        }
    )

    frame = _frame(world)

    assert [target.kind for target in frame.targets if target is not None] == [
        "iron-ore"
    ]


def test_tree_target_is_dropped_if_terrain_cache_removed_it_during_slot_build():
    class StaleTreeSet(set):
        def __contains__(self, value):
            return False

    world = FakeWorld(trees=StaleTreeSet({(2, 0)}))

    frame = _frame(world)

    assert all(target is None for target in frame.targets)


def test_offsets_beyond_build_reach_are_masked():
    frame = _frame(FakeWorld(), snapshot=_snapshot(position=(0.5, 0.5)))
    mask = S.split_masks(frame.obs)["offset"]

    assert mask[S.dxdy_to_offset(6, 8)]  # exactly reach 10
    assert not mask[S.dxdy_to_offset(8, 8)]
    for index in np.flatnonzero(mask):
        dx, dy = S.offset_to_dxdy(int(index))
        assert math.hypot(dx, dy) <= 10.0


def test_offset_mask_blocks_water_trees_and_obstacles_for_one_tile_support():
    world = FakeWorld(
        water={(1, 0): 1},
        trees={(2, 0)},
        obstacles={(3, 0)},
    )
    frame = _frame(world, snapshot=_snapshot(position=(0.5, 0.5)))
    mask = S.split_masks(frame.obs)["offset"]

    for dx in (1, 2, 3):
        assert not mask[S.dxdy_to_offset(dx, 0)]


def test_resource_grid_and_offset_head_share_origin_and_dxdy_convention():
    player = (10.5, 20.5)
    world = FakeWorld(ores={(13, 22): ("iron-ore", 10)})
    frame = _frame(world, snapshot=_snapshot(position=player))
    grid = split_observation(frame.obs)["grid"]
    resource = S.GRID_CHANNELS.index("resource")
    offset = S.dxdy_to_offset(3, 2)
    dx, dy = S.offset_to_dxdy(offset)

    assert (dx, dy) == (3, 2)
    assert grid[resource, S.OFFSET_RADIUS + 2, S.OFFSET_RADIUS + 3] == 1
    assert grid[
        resource,
        S.OFFSET_RADIUS + dy,
        S.OFFSET_RADIUS + dx,
    ] == 1


def test_place_offset_support_rejects_sampled_full_footprint_without_relocation():
    world = FakeWorld(obstacles={(1, -1)})
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.vocab = VocabData.load(S.VOCAB_PATH)
    env.world = world
    env._current = _snapshot(
        inventory={"stone-furnace": 1}, position=(0.5, 0.5)
    )
    env.build_reach = 10.0
    requested = S.dxdy_to_offset(2, 0)

    assert not env._placement_is_free(
        "stone-furnace", requested, S.DIRECTIONS.index(4)
    )
    assert env._placement_is_free(
        "stone-furnace", S.dxdy_to_offset(-2, 0), S.DIRECTIONS.index(4)
    )


def test_burner_drill_support_ignores_ore_coverage():
    world = FakeWorld(ores={(4, 0): ("iron-ore", 10)})
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.vocab = VocabData.load(S.VOCAB_PATH)
    env.world = world
    env._current = _snapshot(
        inventory={"burner-mining-drill": 1}, position=(0.5, 0.5)
    )
    env.build_reach = 10.0
    requested = S.dxdy_to_offset(0, 2)

    assert env._placement_is_free(
        "burner-mining-drill", requested, S.DIRECTIONS.index(0)
    )


def test_resource_place_does_not_approach_ore_or_change_sampled_offset():
    env, namespace, action, records = _place_step_env(
        "burner-mining-drill",
        ores={(60, 0): ("iron-ore", 10)},
    )
    requested = S.dxdy_to_offset(3, 2)
    action[S.HEADS.index("offset")] = requested
    _, expected = env._place_position(
        "burner-mining-drill", requested, S.DIRECTIONS.index(0)
    )

    _, _, _, _, info = env.step(action)

    assert info["status"] == "ok"
    assert info["requested_offset"] == requested
    assert info["executed_offset"] == requested
    assert "approach_tiles" not in info
    namespace.move_to.assert_not_called()
    placed = namespace.place_entity.call_args.args[2]
    assert (placed.x, placed.y) == expected
    assert records[0]["requested_offset"] == requested
    assert records[0]["executed_offset"] == requested


def test_drill_without_ore_is_an_ordinary_tool_rejection_with_game_message():
    message = (
        "Could not place burner-mining-drill at (0, 0), "
        "something is in the way or terrain is unplaceable"
    )
    env, namespace, action, records = _place_step_env(
        "burner-mining-drill",
        ores={},
        place_error=message,
    )

    _, _, _, _, info = env.step(action)

    assert info["status"] == "tool_rejected"
    assert info["requested_offset"] == S.dxdy_to_offset(0, 0)
    namespace.move_to.assert_not_called()
    namespace.place_entity.assert_called_once()
    assert records[0]["raw_error"] == message


def test_place_with_no_free_full_footprint_logs_no_support_without_tool_call():
    before = _snapshot(
        inventory={"stone-furnace": 1}, position=(0.5, 0.5)
    )
    frame = SimpleNamespace(
        obs=np.ones(S.OBS_SIZE, dtype=np.float32),
        targets=(None,) * S.N_TARGET_SLOTS,
        entities=(None,) * S.N_ENTITY_SLOTS,
    )
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.action_space = gym.spaces.MultiDiscrete(S.HEAD_DIMS)
    env.vocab = VocabData.load(S.VOCAB_PATH)
    env.build_reach = 10.0
    env._episode_done = False
    env._current = before
    env._frame = frame
    env.quantity_aware_support = True
    env.support_cooldowns = True
    env._cooldown_action = None
    env._cooldown_fingerprint = None
    env._episode_start_tick = 100
    env._steps = 0
    env._episode_max_steps = 10
    env.max_ticks = 1_000
    env._excursion = False
    env._excursion_steps = 0
    env._excursion_budget = None
    env._aps_previous = 0.0
    env._max_aps = 0.0
    env._last_status = None
    env._last_op = None
    env._op_histogram = Counter()
    env._status_histogram = Counter()
    env._entities_placed = 0
    env._items_harvested = 0
    env._episode_deaths = 0
    env._end_reason = None
    env.deaths = 0
    env._enabled_recipes = []
    env._research_state = {}
    env._restore_hash = None
    env._episode_index = 0
    env._log_file = None
    env.world = SimpleNamespace(
        entities={},
        trees=set(),
        obstacles=set(),
        is_water=lambda x, y: True,
        all_drain=Mock(),
        techs_finished=[],
    )
    env.namespace = SimpleNamespace(move_to=Mock())
    env._snapshot = Mock(return_value=(before, True))
    env._build_frame = Mock(return_value=frame)
    env._execute = Mock()
    records = []
    env._write_log = records.append
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["PLACE"]
    action[S.HEADS.index("placeable")] = S.PLACEABLE_INDEX["stone-furnace"]

    _, _, _, _, info = env.step(action)

    assert info["status"] == "no_support"
    assert info["reason_class"] == "place_offset_unsupported"
    assert info["requested_offset"] == 0
    assert info["executed_offset"] is None
    assert records[0]["requested_offset"] == 0
    assert records[0]["executed_offset"] is None
    assert "approach_tiles" not in info
    env.namespace.move_to.assert_not_called()
    env._execute.assert_not_called()


def test_offset_mask_allows_player_tile_but_excludes_entity_footprint():
    furnace = EntityRow(
        unit=8,
        name="stone-furnace",
        x=13.0,
        y=10.0,
        direction=0,
        status=1,
        tile_width=2,
        tile_height=2,
    )
    world = FakeWorld(entities={8: furnace})
    frame = _frame(
        world,
        snapshot=_snapshot(entities=world.entities, position=(10.5, 10.5)),
    )
    mask = S.split_masks(frame.obs)["offset"]

    # place_entity's manual check sees the character collision, then its
    # avoid_entity helper moves the character aside before creating the build.
    assert mask[S.dxdy_to_offset(0, 0)]
    for tile in ((12, 9), (13, 9), (12, 10), (13, 10)):
        dx, dy = tile[0] - 10, tile[1] - 10
        assert not mask[S.dxdy_to_offset(dx, dy)]


def test_trigger_technologies_are_masked_even_when_enabled_and_ready():
    frame = _frame(FakeWorld())
    tech_mask = S.split_masks(frame.obs)["technology"]

    assert tech_mask[S.TECH_INDEX["automation-science-pack"]]
    assert not tech_mask[S.TECH_INDEX["electronics"]]


def test_entity_and_inventory_support_rules_are_state_only():
    row = EntityRow(
        unit=7,
        name="wooden-chest",
        x=1.5,
        y=0.5,
        direction=0,
        status=53,
        inventories={1: {"iron-ore": 3}},
    )
    world = FakeWorld(entities={7: row})
    frame = _frame(
        world,
        snapshot=_snapshot(
            inventory={"coal": 2, "wood": 2, "wooden-chest": 1},
            entities=world.entities,
        ),
        enabled_recipes=("wooden-chest", "iron-plate"),
    )

    admitted_ops = {S.OPS[index] for index in _valid_indices(frame.obs, "op")}
    assert {"CRAFT", "PLACE", "PICKUP", "INSERT", "EXTRACT"} <= admitted_ops
    assert "ROTATE" not in admitted_ops
    assert "SET_RECIPE" not in admitted_ops
    assert _valid_indices(frame.obs, "item") == {
        S.ITEM_INDEX["coal"],
        S.ITEM_INDEX["wood"],
        S.ITEM_INDEX["wooden-chest"],
    }
    assert _valid_indices(frame.obs, "placeable") == {
        S.PLACEABLE_INDEX["wooden-chest"]
    }
    assert _valid_indices(frame.obs, "recipe") == {S.RECIPE_INDEX["wooden-chest"]}
    assert _valid_indices(frame.obs, "duration") == {0, 1, 2, 3}


def test_bare_masks_drop_out_of_reach_targets_and_entity_slots_only() -> None:
    entities = {
        1: EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1),
        2: EntityRow(
            2,
            "wooden-chest",
            20.5,
            0.5,
            0,
            1,
            inventories={1: {"iron-ore": 1}},
        ),
    }
    world = FakeWorld(
        entities=entities,
        ores={(1, 0): ("iron-ore", 5)},
        trees={(-20, 0)},
    )
    snapshot = _snapshot(entities=entities)
    macro = _frame(world, snapshot=snapshot, regime="macro")
    bare = _frame(world, snapshot=snapshot, regime="bare")
    macro_masks = S.split_masks(macro.obs)
    bare_masks = S.split_masks(bare.obs)
    near_target = next(
        index
        for index, target in enumerate(bare.targets)
        if target is not None and target.kind == "iron-ore"
    )
    far_target = next(
        index
        for index, target in enumerate(bare.targets)
        if target is not None and target.kind == "tree"
    )

    assert macro_masks["target"][[near_target, far_target]].tolist() == [True, True]
    assert bare_masks["target"][[near_target, far_target]].tolist() == [True, False]
    assert macro_masks["entity"][:2].tolist() == [True, True]
    assert bare_masks["entity"][:2].tolist() == [True, False]
    assert macro_masks["op"][S.OP_INDEX["EXTRACT"]]
    assert not bare_masks["op"][S.OP_INDEX["EXTRACT"]]


def test_bare_masks_ops_when_all_targets_and_entities_are_out_of_reach() -> None:
    far = EntityRow(1, "wooden-chest", 20.5, 0.5, 0, 1)
    world = FakeWorld(
        entities={1: far},
        ores={(20, 0): ("iron-ore", 5)},
    )
    snapshot = _snapshot(
        inventory={"coal": 1},
        entities=world.entities,
    )
    macro_masks = S.split_masks(_frame(world, snapshot=snapshot).obs)
    bare_masks = S.split_masks(
        _frame(world, snapshot=snapshot, regime="bare").obs
    )

    assert macro_masks["op"][S.OP_INDEX["HARVEST"]]
    assert macro_masks["op"][S.OP_INDEX["PICKUP"]]
    assert macro_masks["op"][S.OP_INDEX["INSERT"]]
    assert not bare_masks["op"][S.OP_INDEX["HARVEST"]]
    assert not bare_masks["op"][S.OP_INDEX["PICKUP"]]
    assert not bare_masks["op"][S.OP_INDEX["INSERT"]]


def test_extract_mask_drops_for_empty_entities_and_drill_fuel_only() -> None:
    vocab = VocabData.load(S.VOCAB_PATH)
    builder = MacroObservationBuilder(
        world=FakeWorld(),
        vocab=vocab,
        crafting_categories=("crafting",),
        resource_reach=2.7,
        build_reach=10.0,
        trigger_technologies=frozenset(),
        status_names={},
        max_steps=256,
        max_ticks=216_000,
    )
    entities = {
        1: EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1),
        2: EntityRow(
            2,
            "burner-mining-drill",
            3.5,
            0.5,
            0,
            1,
            inventories={1: {"coal": 5}},
        ),
    }
    builder.world.entities = entities

    frame = builder.build(
        ObservationInput(
            snapshot=_snapshot(entities=entities),
            enabled_recipes=(),
            research_state={},
            episode_start_tick=100,
            step_count=0,
            last_status=None,
            last_op=None,
        )
    )

    assert S.OP_INDEX["EXTRACT"] not in _valid_indices(frame.obs, "op")


def test_insert_quantity_mask_excludes_counts_above_held_inventory() -> None:
    row = EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1)
    world = FakeWorld(entities={1: row})
    frame = _frame(
        world,
        snapshot=_snapshot(inventory={"coal": 1}, entities=world.entities),
    )

    assert _valid_indices(frame.obs, "quantity") == {S.QUANTITIES.index(1)}


@pytest.mark.parametrize(
    ("source_count", "inventory", "supported"),
    [
        (20, {"copper-ore": 1}, False),
        (20, {"iron-ore": 1}, True),
        (20, {"coal": 1}, True),
        (50, {"iron-ore": 1}, False),
    ],
)
def test_insert_op_mask_respects_visible_furnace_input_and_capacity(
    source_count: int,
    inventory: dict[str, int],
    supported: bool,
) -> None:
    furnace = EntityRow(
        1,
        "stone-furnace",
        1.5,
        0.5,
        0,
        1,
        inventories={
            1: {"iron-ore": source_count},
            2: {"iron-plate": 7},
        },
    )
    world = FakeWorld(entities={1: furnace})
    builder = MacroObservationBuilder(
        world=world,
        vocab=VocabData.load(S.VOCAB_PATH),
        crafting_categories=("crafting",),
        resource_reach=2.7,
        build_reach=10.0,
        trigger_technologies=frozenset(),
        status_names={},
        max_steps=256,
        max_ticks=216_000,
    )
    frame = builder.build(
        ObservationInput(
            snapshot=_snapshot(inventory=inventory, entities=world.entities),
            enabled_recipes=(),
            research_state={},
            episode_start_tick=100,
            step_count=0,
            last_status=None,
            last_op=None,
        )
    )

    assert bool(S.split_masks(frame.obs)["op"][S.OP_INDEX["INSERT"]]) is supported


@pytest.mark.parametrize(
    ("entities", "inventory"),
    [
        ({}, {"transport-belt": 1}),
        (
            {
                1: EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1),
            },
            {"transport-belt": 1},
        ),
        (
            {
                1: EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1),
                2: EntityRow(2, "wooden-chest", 3.5, 0.5, 0, 1),
            },
            {},
        ),
    ],
)
def test_connect_mask_requires_two_entities_and_a_held_connector(
    entities, inventory
):
    world = FakeWorld(entities=entities)
    frame = _frame(
        world,
        snapshot=_snapshot(entities=entities, inventory=inventory),
    )

    assert S.OP_INDEX["CONNECT"] not in _valid_indices(frame.obs, "op")


def test_connect_masks_peer_slots_and_held_connectors():
    entities = {
        1: EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1),
        2: EntityRow(2, "wooden-chest", 3.5, 0.5, 0, 1),
    }
    world = FakeWorld(entities=entities)
    frame = _frame(
        world,
        snapshot=_snapshot(
            entities=entities,
            inventory={"transport-belt": 2},
        ),
    )

    assert S.OP_INDEX["CONNECT"] in _valid_indices(frame.obs, "op")
    assert _valid_indices(frame.obs, "entity") == {0, 1}
    assert _valid_indices(frame.obs, "peer") == {0, 1}
    assert _valid_indices(frame.obs, "connector") == {
        S.CONNECTOR_INDEX["transport-belt"]
    }


def test_connect_rejects_sampled_incompatible_endpoints_unless_clamp_enabled():
    furnace = EntityRow(1, "stone-furnace", 1.0, 1.0, 0, 1)
    belt = EntityRow(2, "transport-belt", 3.5, 0.5, 0, 1)
    inserter = EntityRow(3, "inserter", 5.5, 0.5, 0, 1)
    env = _argument_clamp_env((furnace, belt, inserter), {"transport-belt": 2})
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("entity")] = 0
    action[S.HEADS.index("peer")] = 0
    action[S.HEADS.index("connector")] = S.CONNECTOR_INDEX["transport-belt"]

    spec, requested, executed, requested_peer, executed_peer, reason, clamp = (
        env._clamp_connect(action)
    )

    assert spec is None
    assert reason == "connect_source_incompatible"
    assert executed is None
    assert executed_peer is None

    env.clamp_connect = True
    spec, requested, executed, requested_peer, executed_peer, reason, clamp = (
        env._clamp_connect(action)
    )

    assert reason is None
    assert spec is not None
    assert requested["unit"] == furnace.unit
    assert requested_peer["unit"] == furnace.unit
    assert executed["unit"] == belt.unit
    assert executed_peer["unit"] != belt.unit
    assert spec.args["source"]["unit"] != spec.args["peer"]["unit"]
    assert clamp == "source_incompatible_or_stale"


def test_connect_pipe_requires_fluid_box_endpoint_support():
    chest = EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1)
    belt = EntityRow(2, "transport-belt", 3.5, 0.5, 0, 1)
    env = _argument_clamp_env((chest, belt), {"pipe": 2})
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("connector")] = S.CONNECTOR_INDEX["pipe"]

    spec, _, _, _, _, reason, _ = env._clamp_connect(action)

    assert spec is None
    assert reason == "connect_source_incompatible"


def test_connect_executor_passes_position_endpoints():
    source = EntityRow(1, "transport-belt", 1.5, 0.5, 0, 1)
    peer = EntityRow(2, "inserter", 3.5, 0.5, 0, 1)
    namespace = SimpleNamespace(connect_entities=Mock())
    world = SimpleNamespace(
        entities={1: source, 2: peer},
        read_player_pos=Mock(return_value=(0.0, 0.0)),
    )
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.namespace = namespace
    env.world = world
    env.build_reach = 10.0
    env.vocab = VocabData.load(S.VOCAB_PATH)
    action = ActionSpec(
        "CONNECT",
        {
            "source": FleMacroEnv._anchor(source),
            "peer": FleMacroEnv._anchor(peer),
            "connector": "transport-belt",
        },
    )

    result = env._execute(action)

    assert result.status is None
    (source_arg, peer_arg), kwargs = namespace.connect_entities.call_args
    assert isinstance(source_arg, Position)
    assert isinstance(peer_arg, Position)
    assert (source_arg.x, source_arg.y) == (1.5, 0.5)
    assert (peer_arg.x, peer_arg.y) == (3.5, 0.5)
    assert kwargs["connection_type"] is Prototype.TransportBelt


def test_bare_harvest_and_entity_ops_never_navigate() -> None:
    row = EntityRow(
        1,
        "wooden-chest",
        20.5,
        0.5,
        0,
        1,
        inventories={1: {"iron-ore": 1}},
    )
    resolved = SimpleNamespace(name="wooden-chest")
    namespace = SimpleNamespace(
        move_to=Mock(),
        harvest_resource=Mock(),
        get_entity=Mock(return_value=resolved),
        pickup_entity=Mock(),
        rotate_entity=Mock(),
        insert_item=Mock(),
        extract_item=Mock(),
        set_entity_recipe=Mock(),
    )
    world = SimpleNamespace(
        entities={1: row},
        read_player_pos=Mock(return_value=(0.0, 0.0)),
    )
    anchor = FleMacroEnv._anchor(row)
    actions = (
        ActionSpec("HARVEST", {"target": [20.5, 0.5], "quantity": 1}),
        ActionSpec("PICKUP", {"anchor": anchor}),
        ActionSpec("ROTATE", {"anchor": anchor, "direction": 4}),
        ActionSpec("INSERT", {"anchor": anchor, "item": "coal", "quantity": 1}),
        ActionSpec(
            "EXTRACT", {"anchor": anchor, "item": "iron-ore", "quantity": 1}
        ),
        ActionSpec("SET_RECIPE", {"anchor": anchor, "recipe": "iron-gear-wheel"}),
    )

    for action in actions:
        result = execute_action(action, "bare", namespace, world, 2.7, 10.0)
        assert result.status is None

    namespace.move_to.assert_not_called()
    namespace.harvest_resource.assert_called_once()
    namespace.pickup_entity.assert_called_once()
    namespace.rotate_entity.assert_called_once()
    namespace.insert_item.assert_called_once()
    namespace.extract_item.assert_called_once()
    namespace.set_entity_recipe.assert_called_once()


def test_bare_connect_does_not_approach_endpoints() -> None:
    source = EntityRow(1, "transport-belt", 20.5, 0.5, 0, 1)
    peer = EntityRow(2, "inserter", 30.5, 0.5, 0, 1)
    namespace = SimpleNamespace(move_to=Mock(), connect_entities=Mock())
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.regime = "bare"
    env.namespace = namespace
    env.world = SimpleNamespace(entities={1: source, 2: peer})
    env.build_reach = 10.0
    env.vocab = VocabData.load(S.VOCAB_PATH)
    action = ActionSpec(
        "CONNECT",
        {
            "source": FleMacroEnv._anchor(source),
            "peer": FleMacroEnv._anchor(peer),
            "connector": "transport-belt",
        },
    )

    result = env._execute(action)

    assert result.status is None
    namespace.move_to.assert_not_called()
    namespace.connect_entities.assert_called_once()


def test_extract_revalidates_contents_after_approach_before_tool_call():
    before_row = EntityRow(
        1,
        "wooden-chest",
        20.5,
        0.5,
        0,
        1,
        inventories={1: {"iron-ore": 1}},
    )
    after_row = EntityRow(1, "wooden-chest", 20.5, 0.5, 0, 1)
    namespace = SimpleNamespace(
        move_to=Mock(),
        get_entity=Mock(),
        extract_item=Mock(),
    )
    world = SimpleNamespace(
        entities={1: before_row},
        trees=set(),
        obstacles=set(),
        is_water=lambda x, y: False,
        read_player_pos=Mock(side_effect=[(0.5, 0.5), (19.5, 0.5)]),
    )

    def drain():
        world.entities[1] = after_row

    world.all_drain = Mock(side_effect=drain)
    action = ActionSpec(
        "EXTRACT",
        {
            "anchor": FleMacroEnv._anchor(before_row),
            "item": "iron-ore",
            "quantity": 1,
        },
    )

    result = execute_action(action, "macro", namespace, world, 2.7, 10.0)

    assert result.status == "no_support"
    assert result.reason_class == "extract_item_disappeared_after_approach"
    world.all_drain.assert_called_once_with()
    namespace.extract_item.assert_not_called()


def test_connect_effect_accepts_new_entities_or_consumed_connector_items():
    source = EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1)
    peer = EntityRow(2, "wooden-chest", 3.5, 0.5, 0, 1)
    connector = EntityRow(3, "transport-belt", 2.5, 0.5, 0, 1)
    action = ActionSpec("CONNECT", {"connector": "transport-belt"})
    before = _snapshot(
        inventory={"transport-belt": 6}, entities={1: source, 2: peer}
    )

    unchanged, unchanged_effect = verify_effect(action, before, before)
    consumed, consumed_effect = verify_effect(
        action,
        before,
        _snapshot(
            inventory={"transport-belt": 5}, entities={1: source, 2: peer}
        ),
    )
    changed, changed_effect = verify_effect(
        action,
        before,
        _snapshot(entities={1: source, 2: peer, 3: connector}),
    )

    assert unchanged is False
    assert unchanged_effect["new_entities"] == 0
    assert consumed is True
    assert consumed_effect["connector_item_delta"] == -1
    assert changed is True
    assert changed_effect["new_entities"] == 1
    assert "CONNECT" in DELAYED_DRAIN_OPS


def test_craft_quantity_support_requires_inputs_for_the_requested_count():
    world = FakeWorld()
    vocab = VocabData.load(S.VOCAB_PATH)
    builder = MacroObservationBuilder(
        world=world,
        vocab=vocab,
        crafting_categories=("crafting",),
        resource_reach=2.7,
        build_reach=10.0,
        trigger_technologies=frozenset(),
        status_names={},
        max_steps=256,
        max_ticks=216_000,
    )

    def quantities(plates: int) -> set[int]:
        frame = builder.build(
            ObservationInput(
                snapshot=_snapshot(inventory={"iron-plate": plates}),
                enabled_recipes=("iron-gear-wheel",),
                research_state={},
                episode_start_tick=100,
                step_count=0,
                last_status=None,
                last_op=None,
            )
        )
        assert S.RECIPE_INDEX["iron-gear-wheel"] in _valid_indices(
            frame.obs, "recipe"
        )
        return _valid_indices(frame.obs, "quantity")

    assert S.QUANTITIES.index(5) in quantities(10)
    assert S.QUANTITIES.index(5) not in quantities(8)


def test_drill_quantity_request_clamps_to_one_supported_craft():
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.quantity_aware_support = True
    env.vocab = VocabData.load(S.VOCAB_PATH)
    env.crafting_categories = ("crafting",)
    env._enabled_recipes = ["burner-mining-drill"]
    env._current = _snapshot(
        inventory={
            "iron-plate": 3,
            "iron-gear-wheel": 3,
            "stone-furnace": 1,
        }
    )
    requested = ActionSpec(
        "CRAFT",
        {
            "recipe": "burner-mining-drill",
            "quantity": 20,
            "main_product": "burner-mining-drill",
        },
    )

    clamped, requested_quantity, executed_quantity = env._clamp_quantity(requested)

    assert requested_quantity == 20
    assert executed_quantity == 1
    assert clamped.args["quantity"] == 1

    env.clamp_quantity = False
    unclamped, requested_quantity, executed_quantity = env._clamp_quantity(requested)

    assert requested_quantity == 20
    assert executed_quantity == 20
    assert unclamped.args["quantity"] == 20


def test_research_masks_current_and_queued_technologies():
    snapshot = OperationSnapshot(
        **{
            **_snapshot().__dict__,
            "current_research": "automation-science-pack",
            "research_queue": ("electronics",),
        }
    )
    frame = _frame(FakeWorld(), snapshot=snapshot)

    assert not S.split_masks(frame.obs)["technology"].any()
    assert S.OP_INDEX["RESEARCH"] not in _valid_indices(frame.obs, "op")


def test_cooldown_masks_only_the_exact_complete_action_until_state_changes():
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.support_cooldowns = True
    env._current = _snapshot(inventory={"iron-plate": 2})
    obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
    for start, stop in S.MASK_OFFSETS.values():
        obs[start:stop] = 1.0
    env._frame = SimpleNamespace(obs=obs)
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["CRAFT"]
    action[S.HEADS.index("recipe")] = S.RECIPE_INDEX["iron-gear-wheel"]
    env._cooldown_action = env._complete_action_key(action)
    env._cooldown_fingerprint = env._support_fingerprint(env._current)
    env._apply_cooldown_mask(env._frame)

    masks = S.split_masks(env._frame.obs)
    assert masks["op"][S.OP_INDEX["CRAFT"]]
    assert not masks["quantity"][action[S.HEADS.index("quantity")]]
    assert env._cooldown_hit(action)
    assert env._masked_reason(action) is None
    other = action.copy()
    other[S.HEADS.index("quantity")] = 1
    assert env._masked_reason(other) is None
    env._current = _snapshot(inventory={"iron-plate": 3})
    assert not env._cooldown_hit(action)
    quantity_start, _ = S.MASK_OFFSETS["quantity"]
    env._frame.obs[quantity_start + action[S.HEADS.index("quantity")]] = 1.0
    assert env._masked_reason(action) is None


def test_insert_cooldown_survives_smelt_progress_in_unrelated_entity_contents():
    before_furnace = EntityRow(
        7,
        "stone-furnace",
        1.5,
        0.5,
        0,
        1,
        inventories={1: {"stone": 20}, 2: {"stone-brick": 3}},
    )
    after_furnace = EntityRow(
        7,
        "stone-furnace",
        1.5,
        0.5,
        0,
        1,
        inventories={1: {"stone": 19}, 2: {"stone-brick": 4}},
    )
    world = FakeWorld(entities={7: before_furnace})
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.vocab = VocabData.load(S.VOCAB_PATH)
    env.support_cooldowns = True
    env._current = _snapshot(
        inventory={"iron-ore": 20},
        entities=world.entities,
        position=(0.5, 0.5),
    )
    obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
    for start, stop in S.MASK_OFFSETS.values():
        obs[start:stop] = 1.0
    frame = SimpleNamespace(obs=obs)
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["INSERT"]
    action[S.HEADS.index("entity")] = 0
    action[S.HEADS.index("item")] = S.ITEM_INDEX["iron-ore"]
    action[S.HEADS.index("quantity")] = S.QUANTITIES.index(1)
    env._cooldown_action = env._complete_action_key(action)
    env._cooldown_entity_unit = before_furnace.unit
    env._cooldown_fingerprint = env._support_fingerprint(
        env._current,
        env._cooldown_action,
        env._cooldown_entity_unit,
    )

    world.entities[7] = after_furnace
    env._current = _snapshot(
        inventory={"iron-ore": 20},
        entities=world.entities,
        position=(0.5, 0.5),
    )

    assert env._cooldown_hit(action)
    assert env._apply_cooldown_mask(frame)
    assert not S.split_masks(frame.obs)["quantity"][S.QUANTITIES.index(1)]


def test_cooldown_masks_op_when_the_rejected_action_has_no_other_completion():
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.support_cooldowns = True
    env._current = _snapshot(inventory={"iron-plate": 2})
    obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["CRAFT"]
    action[S.HEADS.index("recipe")] = S.RECIPE_INDEX["iron-gear-wheel"]
    for head in ("op", "recipe", "quantity"):
        start, _ = S.MASK_OFFSETS[head]
        obs[start + action[S.HEADS.index(head)]] = 1.0
    env._frame = SimpleNamespace(obs=obs)
    env._cooldown_action = env._complete_action_key(action)
    env._cooldown_fingerprint = env._support_fingerprint(env._current)

    env._apply_cooldown_mask(env._frame)

    assert not S.split_masks(obs)["op"][S.OP_INDEX["CRAFT"]]
    assert env._masked_reason(action) is None


@pytest.mark.parametrize(
    ("name", "recipe", "expected"),
    [
        ("transport-belt", None, True),
        ("burner-mining-drill", None, True),
        ("stone-furnace", None, False),
        ("wooden-chest", None, False),
        ("assembling-machine-1", None, False),
        ("assembling-machine-1", "iron-gear-wheel", True),
        ("pipe", None, False),
        ("pipe-to-ground", None, True),
        ("steam-engine", None, True),
    ],
)
def test_rotatable_support_uses_entity_type_and_assembler_recipe(
    name, recipe, expected
):
    vocab = VocabData.load(S.VOCAB_PATH)
    row = EntityRow(1, name, 1.5, 0.5, 0, 1, recipe=recipe)

    assert entity_is_rotatable(row, vocab) is expected


def _argument_clamp_env(
    rows: tuple[EntityRow | None, ...], inventory: dict[str, int]
) -> FleMacroEnv:
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.vocab = VocabData.load(S.VOCAB_PATH)
    env._current = _snapshot(
        inventory=inventory,
        entities={row.unit: row for row in rows if row is not None},
    )
    env.world = SimpleNamespace(entities=dict(env._current.entities))
    env._frame = SimpleNamespace(
        entities=rows + (None,) * (S.N_ENTITY_SLOTS - len(rows))
    )
    return env


def test_rotate_anchor_is_no_support_unless_legacy_clamp_enabled():
    chest = EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1)
    belt = EntityRow(2, "transport-belt", 2.5, 0.5, 0, 1)
    env = _argument_clamp_env((chest, belt), {})
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("entity")] = 0

    row, requested, executed, reason = env._clamp_entity_anchor(action, "ROTATE")

    assert row is None
    assert requested["unit"] == 1
    assert executed is None
    assert reason == "no_rotatable_entity"

    env.clamp_anchor = True
    row, requested, executed, reason = env._clamp_entity_anchor(action, "ROTATE")

    assert row == belt
    assert requested["unit"] == 1
    assert executed == {"slot": 1, **env._anchor(belt)}
    assert reason is None


def test_stale_anchor_is_no_support_unless_legacy_clamp_enabled():
    stale = EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1)
    live = EntityRow(2, "wooden-chest", 3.5, 0.5, 0, 1)
    env = _argument_clamp_env((stale, live), {})
    env.world.entities.pop(stale.unit)
    action = np.zeros(len(S.HEADS), dtype=np.int64)

    row, requested, executed, reason = env._clamp_entity_anchor(action, "PICKUP")

    assert row is None
    assert requested["unit"] == stale.unit
    assert executed is None
    assert reason == "no_live_entity_anchor"

    env.clamp_anchor = True
    row, requested, executed, reason = env._clamp_entity_anchor(action, "PICKUP")

    assert row == live
    assert requested["unit"] == stale.unit
    assert executed == {"slot": 1, **env._anchor(live)}
    assert reason is None

    env.world.entities.clear()
    row, _, executed, reason = env._clamp_entity_anchor(action, "PICKUP")
    assert row is None
    assert executed is None
    assert reason == "no_live_entity_anchor"


def test_insert_item_is_no_support_unless_legacy_clamp_enabled():
    furnace = EntityRow(1, "stone-furnace", 1.5, 0.5, 0, 1)
    inventory = {"stone-furnace": 1, "coal": 2, "iron-ore": 3}
    env = _argument_clamp_env((furnace,), inventory)
    requested = ActionSpec(
        "INSERT",
        {
            "anchor": env._anchor(furnace),
            "item": "stone-furnace",
            "quantity": 1,
        },
    )
    logits = np.zeros(S.HEAD_SIZES["item"], dtype=np.float32)
    logits[S.ITEM_INDEX["iron-ore"]] = 5.0

    clamped, requested_item, executed_item, reason = env._clamp_item(
        requested, furnace, {"head_logits": {"item": logits}}
    )

    assert insertable_items(furnace, inventory, env.vocab) == ["coal", "iron-ore"]
    assert requested_item == "stone-furnace"
    assert executed_item is None
    assert clamped.args["item"] == "stone-furnace"
    assert reason == "insert_item_not_held_or_accepted"

    env.clamp_item = True
    clamped, requested_item, executed_item, reason = env._clamp_item(
        requested, furnace, {"head_logits": {"item": logits}}
    )

    assert executed_item == "iron-ore"
    assert clamped.args["item"] == "iron-ore"
    assert reason is None


def test_extract_item_is_no_support_unless_legacy_clamp_enabled():
    chest = EntityRow(
        1,
        "wooden-chest",
        1.5,
        0.5,
        0,
        1,
        inventories={1: {"iron-ore": 2, "coal": 1}},
    )
    env = _argument_clamp_env((chest,), {})
    requested = ActionSpec(
        "EXTRACT",
        {"anchor": env._anchor(chest), "item": "wood", "quantity": 1},
    )

    clamped, requested_item, executed_item, reason = env._clamp_item(
        requested, chest
    )

    assert requested_item == "wood"
    assert executed_item is None
    assert clamped.args["item"] == "wood"
    assert reason == "extract_item_not_present"

    env.clamp_item = True
    clamped, requested_item, executed_item, reason = env._clamp_item(
        requested, chest
    )

    assert executed_item == "iron-ore"
    assert clamped.args["item"] == "iron-ore"
    assert reason is None

    empty = EntityRow(2, "wooden-chest", 2.5, 0.5, 0, 1)
    _, _, executed_item, reason = env._clamp_item(requested, empty)
    assert executed_item is None
    assert reason == "extract_entity_has_no_items"


def test_extract_drill_fuel_is_deterministic_no_support() -> None:
    drill = EntityRow(
        1,
        "burner-mining-drill",
        1.5,
        0.5,
        0,
        1,
        inventories={1: {"coal": 5}},
    )
    env = _argument_clamp_env((drill,), {})
    requested = ActionSpec(
        "EXTRACT",
        {"anchor": env._anchor(drill), "item": "coal", "quantity": 1},
    )

    _, requested_item, executed_item, reason = env._clamp_item(requested, drill)

    assert requested_item == "coal"
    assert executed_item is None
    assert reason == "extract_entity_has_no_items"


def test_extract_empty_entity_step_marks_cooldown_on_first_no_support():
    chest = EntityRow(1, "wooden-chest", 1.5, 0.5, 0, 1)
    env = _argument_clamp_env((chest,), {})
    obs = np.ones(S.OBS_SIZE, dtype=np.float32)
    env._frame.obs = obs
    env.action_space = gym.spaces.MultiDiscrete(S.HEAD_DIMS)
    env._episode_done = False
    env.quantity_aware_support = True
    env.support_cooldowns = True
    env._cooldown_action = None
    env._cooldown_fingerprint = None
    env._episode_start_tick = 100
    env._steps = 0
    env._episode_max_steps = 10
    env.max_ticks = 1_000
    env._excursion = False
    env._excursion_steps = 0
    env._excursion_budget = None
    env._aps_previous = 0.0
    env._max_aps = 0.0
    env._last_status = None
    env._last_op = None
    env._op_histogram = Counter()
    env._status_histogram = Counter()
    env._entities_placed = 0
    env._items_harvested = 0
    env._episode_deaths = 0
    env.deaths = 0
    env._enabled_recipes = []
    env._research_state = {}
    env._restore_hash = None
    env._episode_index = 0
    env._log_file = None
    env.world.all_drain = Mock()
    env.world.techs_finished = []
    env._snapshot = Mock(return_value=(env._current, True))

    def build_frame():
        env._cooldown_masked = env._apply_cooldown_mask(env._frame)
        return env._frame

    env._build_frame = Mock(side_effect=build_frame)
    env._execute = Mock()
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["EXTRACT"]
    action[S.HEADS.index("item")] = S.ITEM_INDEX["coal"]

    _, _, _, _, info = env.step(action)

    assert info["status"] == "no_support"
    assert info["reason_class"] == "extract_entity_has_no_items"
    assert info["requested_item"] == "coal"
    assert info["executed_item"] is None
    assert info["cooldown_masked"] is True
    assert env._cooldown_action == env._complete_action_key(action)
    assert not S.split_masks(env._frame.obs)["quantity"][0]
    env._execute.assert_not_called()


def test_tool_exception_is_reconciled_as_ok_when_effect_happened_and_logged():
    before = _snapshot(position=(1.0, 1.0))
    after = OperationSnapshot(
        inventory={"wood": 10},
        entities={},
        position=(2.0, 2.0),
        tick=110,
        score_player=0.0,
        score_automated=0.0,
        current_research=None,
    )
    obs = np.ones(S.OBS_SIZE, dtype=np.float32)
    frame = SimpleNamespace(
        obs=obs,
        targets=(HarvestTarget("tree", (1.5, 1.5), 0),)
        + (None,) * (S.N_TARGET_SLOTS - 1),
        entities=(None,) * S.N_ENTITY_SLOTS,
    )
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.action_space = gym.spaces.MultiDiscrete(S.HEAD_DIMS)
    env._episode_done = False
    env._current = before
    env._frame = frame
    env.quantity_aware_support = True
    env.support_cooldowns = True
    env._cooldown_action = None
    env._cooldown_fingerprint = None
    env._episode_start_tick = 100
    env._steps = 0
    env._episode_max_steps = 10
    env.max_ticks = 1_000
    env._excursion = False
    env._excursion_steps = 0
    env._excursion_budget = None
    env._aps_previous = 0.0
    env._max_aps = 0.0
    env._last_status = None
    env._last_op = None
    env._op_histogram = Counter()
    env._status_histogram = Counter()
    env._entities_placed = 0
    env._items_harvested = 0
    env._episode_deaths = 0
    env._end_reason = None
    env.deaths = 0
    env._enabled_recipes = []
    env._research_state = {}
    env._restore_hash = None
    env._episode_index = 0
    env._log_file = None
    env.world = SimpleNamespace(
        all_drain=Mock(),
        techs_finished=[],
    )
    env._snapshot = Mock(return_value=(after, True))
    env._build_frame = Mock(return_value=frame)
    env._execute = Mock(side_effect=RuntimeError("LuaEntity invalid for read"))
    records = []
    env._write_log = records.append
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    action[S.HEADS.index("op")] = S.OP_INDEX["HARVEST"]

    _, _, _, _, info = env.step(action)

    assert info["status"] == "ok"
    assert info["reason_class"] == "ok_after_exception"
    assert info["player_pos_before"] == [1.0, 1.0]
    assert info["player_pos_after"] == [2.0, 2.0]
    assert info["transferred_quantity"] == 10
    assert set(info["admitted_indices"]) == {"op", "target", "quantity"}
    assert records[0]["raw_error"] == "LuaEntity invalid for read"


def test_observation_layout_round_trips_and_stays_in_space():
    frame = _frame(FakeWorld())
    blocks = split_observation(frame.obs)
    reconstructed = np.concatenate([blocks[name].reshape(-1) for name in S.OBS_LAYOUT])

    assert frame.obs.shape == (S.OBS_SIZE,)
    assert frame.obs.dtype == np.float32
    assert np.array_equal(reconstructed, frame.obs)
    assert np.all(frame.obs >= -1.0)
    assert np.all(frame.obs <= 1.0)
    assert blocks["globals"].shape == (S.N_GLOBALS,)
    assert blocks["targets"].shape == (S.N_TARGET_SLOTS, S.TARGET_FEATURES)
    assert blocks["entities"].shape == (S.N_ENTITY_SLOTS, S.ENTITY_FEATURES)
    assert blocks["grid"].shape == (len(S.GRID_CHANNELS), S.GRID_SIDE, S.GRID_SIDE)


def test_random_valid_action_never_selects_a_masked_index():
    row = EntityRow(
        unit=3,
        name="wooden-chest",
        x=2.5,
        y=2.5,
        direction=0,
        status=1,
        inventories={1: {"iron-ore": 1}},
    )
    world = FakeWorld(
        entities={3: row},
        ores={(30, 0): ("iron-ore", 5)},
        trees={(0, 20)},
    )
    frame = _frame(
        world,
        snapshot=_snapshot(inventory={"coal": 1}, entities=world.entities),
        enabled_recipes=("wooden-chest",),
    )
    masks = S.split_masks(frame.obs)
    rng = np.random.default_rng(4)

    for _ in range(2_000):
        action = S.random_valid_action(frame.obs, rng)
        for index, head in enumerate(S.HEADS):
            if masks[head].any():
                assert masks[head][action[index]]
            else:
                assert action[index] == 0


def test_macro_v2_schema_and_default_horizons():
    parameters = inspect.signature(FleMacroEnv).parameters

    assert S.SCHEMA_VERSION == "macro-v2"
    assert S.HEADS[S.HEADS.index("item") + 1] == "placeable"
    assert S.HEAD_SIZES["placeable"] == len(S.PLACEABLE_NAMES)
    assert S.HEAD_DIMS == tuple(S.HEAD_SIZES[head] for head in S.HEADS)
    assert S.OP_HEADS["PLACE"] == ("placeable", "offset", "direction")
    assert S.OP_HEADS["INSERT"] == ("entity", "item", "quantity")
    assert S.OP_HEADS["EXTRACT"] == ("entity", "item", "quantity")
    assert S.OP_HEADS["CONNECT"] == ("entity", "peer", "connector")
    assert parameters["max_steps"].default == 256
    assert parameters["max_ticks"].default == 216_000
    assert FakeMacroEnv().horizon == 256


def test_default_tick_cap_scales_with_step_horizon_but_custom_cap_is_absolute():
    assert _effective_max_ticks(256, 216_000) == 216_000
    assert _effective_max_ticks(512, 216_000) == 432_000
    assert _effective_max_ticks(512, 123_000) == 123_000


def test_peaceful_reset_command_clears_enemy_force_without_destroy_events():
    assert "surface.peaceful_mode=true" in PEACEFUL_RESET_COMMAND
    assert 'find_entities_filtered{force="enemy"}' in PEACEFUL_RESET_COMMAND
    assert "entity.destroy{raise_destroy=false}" in PEACEFUL_RESET_COMMAND


def test_acceptance_driver_probes_craft_mask_at_one_output():
    obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
    for start, _ in S.MASK_OFFSETS.values():
        obs[start] = 1.0
    op_start, op_stop = S.MASK_OFFSETS["op"]
    obs[op_start:op_stop] = 0.0
    obs[op_start + S.OP_INDEX["CRAFT"]] = 1.0

    action = _random_acceptance_action(obs, np.random.default_rng(1))

    assert action[S.HEADS.index("op")] == S.OP_INDEX["CRAFT"]
    assert action[S.HEADS.index("quantity")] == 0


def _restore_harness(monkeypatch, *, mismatched=False):
    row = EntityRow(
        unit=7,
        name="wooden-chest",
        x=1.5,
        y=2.5,
        direction=4,
        status=1,
        inventories={},
    )
    captured = OperationSnapshot(
        inventory={"coal": 3},
        entities={7: row},
        position=(3.0, 4.0),
        tick=150,
        score_player=2.0,
        score_automated=11.0,
        current_research=None,
    )
    restored_row = EntityRow(
        unit=99,
        name="wooden-chest",
        x=2.5 if mismatched else 1.5,
        y=2.5,
        direction=4,
        status=1,
        inventories={1: {"coal": 1}},
    )
    restored = OperationSnapshot(
        inventory={"coal": 3},
        entities={99: restored_row},
        position=(3.0, 4.0),
        tick=1_000,
        score_player=0.0,
        score_automated=0.0,
        current_research=None,
    )
    game_state = object()
    monkeypatch.setattr(
        "fle.rl.env.GameState.from_instance", lambda instance: game_state
    )

    env = FleMacroEnv.__new__(FleMacroEnv)
    env.action_space = gym.spaces.MultiDiscrete(S.HEAD_DIMS)
    env.max_steps = 20
    env.max_ticks = 2_000
    env.speed = 40
    env._current = captured
    env._episode_start_tick = 100
    env._steps = 10
    env._episode_max_steps = 20
    env._episode_done = False
    env._episode_index = 0
    env._log_file = None
    env._op_histogram = Counter()
    env._status_histogram = Counter()
    env._max_aps = captured.score_automated
    env._entities_placed = 0
    env._items_harvested = 0
    env._episode_deaths = 0
    env._excursion = False
    env._restore_hash = None
    env._snapshot = Mock(return_value=(restored, True))
    env._build_frame = Mock(
        return_value=SimpleNamespace(obs=np.zeros(S.OBS_SIZE, dtype=np.float32))
    )
    env.instance = SimpleNamespace(
        reset=Mock(),
        rcon_client=SimpleNamespace(send_command=Mock()),
        set_speed_and_unpause=Mock(),
    )
    env.world = SimpleNamespace(
        entity_full_sync=Mock(),
        all_drain=Mock(),
        read_enabled_recipes=Mock(return_value=[]),
        read_research_state=Mock(return_value={}),
        techs_finished=[],
    )
    return env, game_state


def test_frontier_snapshot_restore_round_trip_uses_game_state(monkeypatch):
    env, game_state = _restore_harness(monkeypatch)
    frontier = env.snapshot()
    env.max_steps = 100

    observation, info = env.reset(options={"restore": frontier, "budget": 64})

    assert isinstance(frontier, FrontierState)
    assert frontier.step_index == 10
    assert frontier.max_steps == 20
    assert frontier.episode_ticks == 50
    env.instance.reset.assert_called_once_with(
        game_state=game_state,
        reset_position=False,
        all_technologies_researched=False,
        clear_entities=True,
    )
    assert env._steps == 10
    assert env._episode_max_steps == 20
    assert env._episode_start_tick == 950
    assert env._aps_previous == 0.0
    assert env._excursion_budget == 10
    assert info == {
        "score_player": 0.0,
        "score_automated": 0.0,
        "excursion": True,
        "restore_hash": frontier.entity_hash,
    }
    assert observation.shape == (S.OBS_SIZE,)


def test_frontier_restore_refuses_entity_hash_mismatch(monkeypatch):
    env, _ = _restore_harness(monkeypatch, mismatched=True)
    records = []
    env._write_log = records.append
    frontier = env.snapshot()

    with pytest.raises(RuntimeError, match="entity identity/position hash"):
        env.reset(options={"restore": frontier, "budget": 4})

    assert records[0]["frontier_restore_refused"]["inventory_matches"] is True
    assert env._episode_done is True


def test_episode_end_records_episode_max_steps(monkeypatch):
    env, _ = _restore_harness(monkeypatch)
    records = []
    env._write_log = records.append
    env._episode_max_steps = 512

    env._write_episode_end()

    assert records == [
        {
            "episode_end": {
                "episode": 0,
                "final_aps": 11.0,
                "max_aps": 11.0,
                "player_ps": 2.0,
                "op_histogram": {},
                "status_histogram": {},
                "entities_placed": 0,
                "items_harvested": 0,
                "deaths": 0,
                "excursion": False,
                "restore_hash": None,
                "max_steps": 512,
                "end_reason": "max_steps",
                "regime": "macro",
            }
        }
    ]


def test_fake_frontier_restore_caps_excursion_at_original_remaining_budget():
    env = FakeMacroEnv(horizon=6, seed=4)
    env.reset()
    action = np.zeros(len(S.HEADS), dtype=np.int64)
    for _ in range(4):
        env.step(action)
    frontier = env.snapshot()

    _, reset_info = env.reset(options={"restore": frontier, "budget": 64})
    _, _, _, first_truncated, first_info = env.step(action)
    _, _, _, second_truncated, second_info = env.step(action)

    assert reset_info["excursion"] is True
    assert first_truncated is False
    assert second_truncated is True
    assert first_info["excursion"] is True
    assert second_info["restore_hash"] == frontier.entity_hash
