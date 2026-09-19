from __future__ import annotations

import math
import time
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fle.commons.observation.buildability import BuildabilityCache
from fle.rl import schema as legacy_schema
from fle.rl.env import FleMacroEnv
from fle.rl.ops import VocabData
from fle.rl.v0_actions import (
    MASK_NAMES,
    V0FailureReason,
    decode_v0_action,
    execute_v0_action,
    v0_masks_for_prefix,
)
from fle.rl.v0_schema import (
    DIRECTIONS,
    HEAD_SIZES,
    ITEM_INDEX,
    PLACEABLE_INDEX,
    TECH_NAMES,
    VERB_INDEX,
    cell_to_world,
)
from fle.rl.world import EntityRow


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for this module."""
    yield


def _row(unit: int, *, items: dict[str, int] | None = None) -> EntityRow:
    inventories = {1: items or {}}
    return EntityRow(
        unit=unit,
        name="wooden-chest",
        x=20.5,
        y=0.5,
        direction=0,
        status=1,
        tile_width=1,
        tile_height=1,
        inventories=inventories,
    )


def _context(**updates):
    result = {
        "position": (0.25, 0.75),
        "tick": 1_000,
        "inventory_counts": {"wooden-chest": 2, "iron-plate": 3},
        "entity_rows": (),
        "mask_toggles": {},
        "mask_counters": Counter(),
        "vocab": VocabData.load(legacy_schema.VOCAB_PATH),
    }
    result.update(updates)
    return result


class FakeWorld:
    def __init__(self, *, buildability=None, entities=None, position=(0.5, 0.5)):
        self.buildability = buildability
        self.entities = entities or {}
        self.position = position
        self.tick = 100
        self.trees = set()
        self.obstacles = set()
        self.water = {}
        self.score_player = 7.0
        self.score_automated = 11.0

    def read_tick(self):
        return self.tick

    def read_player_pos(self):
        return self.position

    def all_drain(self):
        return None

    def is_water(self, x, y):
        return (x, y) in self.water

    def advance_ticks(self, ticks):
        self.tick += ticks


def _namespace(world: FakeWorld):
    placed = []

    def move_to(position):
        world.position = (position.x, position.y)
        world.tick += 30

    def place_entity(prototype, direction, position, *, exact):
        placed.append((prototype, direction, position, exact))
        world.tick += 6
        world.score_automated += 2.5

    namespace = SimpleNamespace(
        move_to=Mock(side_effect=move_to),
        place_entity=Mock(side_effect=place_entity),
        score=lambda: (world.score_player, world.score_automated),
    )
    return namespace, placed


def _empty_cache(origin=(-64, -64)) -> BuildabilityCache:
    cache = BuildabilityCache()
    cache.size = 128
    cache.origin = origin
    cache.values = np.full((63, 128, 128), -1, dtype=np.int8)
    cache.sampled_ticks = np.full((63, 16, 16), -1, dtype=np.int64)
    return cache


def test_three_location_verbs_decode_the_same_cell() -> None:
    cell = 17 * 64 + 41
    context = _context()
    move = decode_v0_action(
        {"verb": VERB_INDEX["MOVE_TO"], "location": cell}, {}, context
    )
    mine = decode_v0_action(
        {"verb": VERB_INDEX["MINE"], "location": cell, "quantity": 0},
        {},
        context,
    )
    place = decode_v0_action(
        {
            "verb": VERB_INDEX["PLACE"],
            "place_item": PLACEABLE_INDEX["wooden-chest"],
            "direction": 0,
            "location": cell,
        },
        {},
        context,
    )
    expected = cell_to_world(cell, *context["position"])
    assert move.args["tile"] == mine.args["tile"] == place.args["tile"] == expected


def test_move_to_is_bounded_by_the_exact_local_frame() -> None:
    player = (15.9, -20.1)
    for cell in range(64 * 64):
        tile = cell_to_world(cell, *player)
        assert (
            max(
                abs(tile[0] - math.floor(player[0])),
                abs(tile[1] - math.floor(player[1])),
            )
            <= 32
        )


def test_place_mask_only_forbids_fresh_proven_blockage() -> None:
    cache = _empty_cache(origin=(-40, -40))
    item = "wooden-chest"
    direction = DIRECTIONS[0]
    channel = cache.channel_for(item, direction)
    # local cells map to world tiles; the deliberately offset cache exercises
    # the crop rather than accidentally comparing equal array coordinates.
    cases = {
        (2, 10): (0, 995, False),  # fresh blocked
        (10, 10): (1, 995, True),  # fresh legal
        (18, 10): (-1, 995, True),  # unknown
        (26, 10): (0, 100, True),  # stale blocked
    }
    for (cell_x, cell_y), (value, sampled, _allowed) in cases.items():
        world_x, world_y = cell_to_world(cell_y * 64 + cell_x, 0.25, 0.75)
        cache_x, cache_y = world_x - cache.origin[0], world_y - cache.origin[1]
        cache.values[channel, cache_y, cache_x] = value
        cache.sampled_ticks[channel, cache_y // 8, cache_x // 8] = sampled
    world = FakeWorld(buildability=cache)
    prefix = {
        "verb": VERB_INDEX["PLACE"],
        "place_item": PLACEABLE_INDEX[item],
        "direction": 0,
    }
    mask = v0_masks_for_prefix(prefix, {}, world, _context(buildability=cache))[
        "location"
    ]
    for (cell_x, cell_y), (_value, _sampled, allowed) in cases.items():
        assert bool(mask[cell_y * 64 + cell_x]) is allowed
    # The local corner is outside this cache window and therefore allowed.
    partial = _empty_cache(origin=(0, 0))
    partial.values[channel].fill(0)
    partial.sampled_ticks[channel].fill(1_000)
    mask = v0_masks_for_prefix(
        prefix, {}, FakeWorld(buildability=partial), _context(buildability=partial)
    )["location"]
    assert mask[0] == 1


def test_place_mask_selects_prototype_direction_and_lookup_miss_is_allow_all() -> None:
    cache = _empty_cache()
    seen = []

    def channel_for(name, direction):
        seen.append((name, direction))
        return BuildabilityCache.channel_for(name, direction)

    cache.channel_for = channel_for
    prefix = {
        "verb": VERB_INDEX["PLACE"],
        "place_item": PLACEABLE_INDEX["wooden-chest"],
        "direction": 2,
    }
    context = _context(buildability=cache)
    v0_masks_for_prefix(prefix, {}, FakeWorld(buildability=cache), context)
    assert seen == [("wooden-chest", DIRECTIONS[2])]

    cache.channel_for = Mock(side_effect=KeyError("missing channel"))
    mask = v0_masks_for_prefix(prefix, {}, FakeWorld(buildability=cache), context)[
        "location"
    ]
    assert mask.all()
    assert context["mask_counters"]["build/mask_lookup_miss"] == 1


def test_place_is_exact_and_navigation_preserves_requested_tile() -> None:
    world = FakeWorld(position=(0.5, 0.5))
    namespace, placed = _namespace(world)
    cell = 32 * 64 + 52
    context = _context(position=(0.5, 0.5), build_reach=10.0)
    action = decode_v0_action(
        {
            "verb": VERB_INDEX["PLACE"],
            "place_item": PLACEABLE_INDEX["wooden-chest"],
            "direction": 0,
            "location": cell,
        },
        {},
        context,
    )
    selected_tile = action.args["tile"]
    result = execute_v0_action(action, namespace, world, context)
    assert result.success
    assert result.tau == pytest.approx(36 / 60)
    assert result.reward == pytest.approx(2.5)
    assert namespace.move_to.called
    assert placed[0][3] is True
    assert (placed[0][2].x, placed[0][2].y) == action.args["target"]
    assert action.args["tile"] == selected_tile


def test_entity_identity_change_during_navigation_fails_without_mutation() -> None:
    requested = _row(101)
    neighbour = _row(202)
    world = FakeWorld(entities={requested.unit: requested}, position=(0.5, 0.5))
    pickup = Mock()

    def move_to(position):
        world.position = (position.x, position.y)
        world.entities = {neighbour.unit: neighbour}

    namespace = SimpleNamespace(
        move_to=Mock(side_effect=move_to),
        pickup_entity=pickup,
        get_entity=Mock(),
        score=lambda: (world.score_player, world.score_automated),
    )
    context = _context(entity_rows=(requested,), build_reach=3.0)
    action = decode_v0_action({"verb": VERB_INDEX["PICKUP"], "entity": 0}, {}, context)
    result = execute_v0_action(action, namespace, world, context)
    assert not result.success
    assert result.reason is V0FailureReason.IDENTITY_MISMATCH
    assert result.requested_unit_number == requested.unit
    assert result.mutated_unit_number == neighbour.unit
    pickup.assert_not_called()
    namespace.get_entity.assert_not_called()


def test_insert_uses_carried_items_and_extract_uses_target_contents() -> None:
    row = _row(101, items={"copper-plate": 4})
    world = FakeWorld(entities={row.unit: row})
    context = _context(
        entity_rows=(row,),
        inventory_counts={"iron-plate": 3},
    )
    insert_masks = v0_masks_for_prefix(
        {"verb": VERB_INDEX["INSERT"], "entity": 0}, {}, world, context
    )
    extract_masks = v0_masks_for_prefix(
        {"verb": VERB_INDEX["EXTRACT"], "entity": 0}, {}, world, context
    )
    assert insert_masks["inventory_item"][ITEM_INDEX["iron-plate"]] == 1
    assert insert_masks["inventory_item"][ITEM_INDEX["copper-plate"]] == 0
    assert extract_masks["contained_item"][ITEM_INDEX["copper-plate"]] == 1
    assert extract_masks["contained_item"][ITEM_INDEX["iron-plate"]] == 0


def test_every_mask_is_individually_toggleable_and_reported() -> None:
    cache = _empty_cache()
    row = _row(101, items={})
    world = FakeWorld(buildability=cache, entities={row.unit: row})
    counters = Counter()
    toggles = {name: False for name in MASK_NAMES}
    context = _context(
        entity_rows=(row,),
        inventory_counts={},
        buildability=cache,
        craftable_recipes=(),
        available_technologies=(),
        mask_toggles=toggles,
        mask_counters=counters,
    )
    prefix = {
        "verb": VERB_INDEX["PLACE"],
        "place_item": PLACEABLE_INDEX["wooden-chest"],
        "direction": 0,
        "entity": 0,
    }
    masks = v0_masks_for_prefix(prefix, {}, world, context)
    assert all(mask.all() for mask in masks.values())
    assert all(counters[f"mask/{name}/disabled"] == 1 for name in MASK_NAMES)


def test_v0_execution_never_calls_time_sleep(monkeypatch) -> None:
    def forbidden_sleep(_seconds):
        raise AssertionError("wall-clock sleep reached")

    monkeypatch.setattr(time, "sleep", forbidden_sleep)
    world = FakeWorld()
    namespace, _placed = _namespace(world)
    action = decode_v0_action(
        {"verb": VERB_INDEX["MOVE_TO"], "location": 32 * 64 + 33},
        {},
        _context(position=world.position),
    )
    result = execute_v0_action(action, namespace, world, _context())
    assert result.success


def test_fast_forward_uses_simulated_ticks_and_separates_scores() -> None:
    world = FakeWorld()
    namespace, _placed = _namespace(world)
    action = decode_v0_action(
        {"verb": VERB_INDEX["FAST_FORWARD"], "duration": 1}, {}, _context()
    )
    result = execute_v0_action(action, namespace, world, _context())
    assert result.success
    assert result.tau == 10.0
    assert result.reward == 0.0
    assert result.general_score == 7.0


def test_v0_snapshot_batches_scalar_reads_into_one_rcon_call() -> None:
    response = (
        "1234\t1\t2.5\t-3.25\t19.5\t7\tautomation\tlogistics,steel-processing"
        "\tcoal=4,iron-plate=12"
    )
    send_command = Mock(return_value=response)
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.action_grammar = "v0"
    env.v0_batch_snapshot = True
    env.instance = SimpleNamespace(
        rcon_client=SimpleNamespace(send_command=send_command), initial_score=1.5
    )
    env.world = SimpleNamespace(entities={})

    snapshot, valid = env._snapshot()

    assert valid
    assert send_command.call_count == 1
    assert snapshot.tick == 1234
    assert snapshot.position == (2.5, -3.25)
    assert snapshot.inventory == {"coal": 4, "iron-plate": 12}
    assert snapshot.score_player == 18.0
    assert snapshot.score_automated == 7.0
    assert snapshot.current_research == "automation"
    assert snapshot.research_queue == ("logistics", "steel-processing")


def test_v0_snapshot_preserves_empty_trailing_inventory_field() -> None:
    env = FleMacroEnv.__new__(FleMacroEnv)
    env.action_grammar = "v0"
    env.v0_batch_snapshot = True
    env.world = SimpleNamespace(entities={})
    env.instance = SimpleNamespace(
        initial_score=0,
        rcon_client=SimpleNamespace(
            send_command=Mock(return_value="9\t1\t0\t0\t0\t0\t\t\t\n")
        ),
    )

    snapshot, valid = env._snapshot()

    assert valid
    assert snapshot.inventory == {}
    assert snapshot.research_queue == ()


def test_synthetic_place_branching_factor(capsys) -> None:
    cache = _empty_cache()
    channel = cache.channel_for("wooden-chest", 0)
    cache.values[channel].fill(1)
    cache.sampled_ticks[channel].fill(1_000)
    cache.values[channel, 32:40, 32:40] = 0
    prefix = {
        "verb": VERB_INDEX["PLACE"],
        "place_item": PLACEABLE_INDEX["wooden-chest"],
        "direction": 0,
    }
    mask = v0_masks_for_prefix(
        prefix, {}, FakeWorld(buildability=cache), _context(buildability=cache)
    )["location"]
    before, after = HEAD_SIZES["location"], int(mask.sum())
    print(f"PLACE branching factor: {before} -> {after}")
    assert (before, after) == (4096, 4032)
    assert "4096 -> 4032" in capsys.readouterr().out
    assert len(TECH_NAMES) == HEAD_SIZES["technology"]
