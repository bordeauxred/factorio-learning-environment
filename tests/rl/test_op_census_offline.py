import random

import pytest

from fle.rl import schema as S
from fle.rl.ops import (
    ActionSpec,
    NoSupport,
    OperationSnapshot,
    SamplerState,
    VocabData,
    classify_error,
    recursively_craftable_recipes,
    sample_action,
    verify_effect,
)
from fle.rl.world import EntityRow, WorldClient, parse_entity_row


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for this offline module."""
    yield


def _row(
    unit: int,
    name: str = "wooden-chest",
    x: float = 1.5,
    y: float = 2.5,
    direction: int = 0,
    inventories: dict[int, dict[str, int]] | None = None,
) -> EntityRow:
    return EntityRow(
        unit=unit,
        name=name,
        x=x,
        y=y,
        direction=direction,
        status=1,
        inventories=inventories or {},
    )


def _vocab() -> VocabData:
    return VocabData.from_dict(
        {
            "items": [
                {"name": "coal", "place_result": None},
                {"name": "wooden-chest", "place_result": "wooden-chest"},
                {"name": "transport-belt", "place_result": "transport-belt"},
            ],
            "recipes": [
                {
                    "name": "wooden-chest",
                    "products": [{"name": "wooden-chest", "amount": 1}],
                },
                {
                    "name": "iron-gear-wheel",
                    "products": [{"name": "iron-gear-wheel", "amount": 1}],
                },
            ],
            "technologies": [
                {"name": "automation", "prerequisites": ["steam-power"]},
                {"name": "steam-power", "prerequisites": []},
            ],
            "entities": [
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


class FakeWorld:
    def __init__(self, entities=None, ores=None, trees=None):
        self.entities = entities or {}
        self.ores = ores or {}
        self.trees = trees or set()

    def patches(self):
        return WorldClient.patches(self)

    def known_cells(self, px, py, radius=64):
        return [(2.0, 2.0), (6.0, 2.0)]


def _state(world=None, inventory=None) -> SamplerState:
    return SamplerState(
        world=world or FakeWorld(),
        player_pos=(0.0, 0.0),
        inventory=inventory or {},
        enabled_recipes=("wooden-chest", "iron-gear-wheel"),
        research_state={
            "steam-power": {
                "researched": True,
                "enabled": True,
                "prerequisites": [],
            },
            "automation": {
                "researched": False,
                "enabled": True,
                "prerequisites": ["steam-power"],
            },
        },
        vocab=_vocab(),
    )


def _snapshot(
    *,
    inventory=None,
    entities=None,
    position=(0.0, 0.0),
    tick=10,
    research=None,
) -> OperationSnapshot:
    return OperationSnapshot(
        inventory=inventory or {},
        entities=entities or {},
        position=position,
        tick=tick,
        score_player=0,
        score_automated=0,
        current_research=research,
    )


def test_parse_rich_entity_row_all_tag_types():
    row = parse_entity_row(
        "u42,assembling-machine-1,1.5,-2.5,4,7,E1200000,P37,"
        "Riron-gear-wheel,H299,W3:3,D2.5:-2.5,K0.5:-2.5,N91,T165,"
        "I1.iron-plate:12,I2.iron-gear-wheel:3,Fwater:99"
    )

    assert row.unit == 42
    assert (row.name, row.x, row.y, row.direction, row.status) == (
        "assembling-machine-1",
        1.5,
        -2.5,
        4,
        7,
    )
    assert row.energy == 1_200_000
    assert row.progress == 37
    assert row.recipe == "iron-gear-wheel"
    assert row.health == 299
    assert (row.tile_width, row.tile_height) == (3, 3)
    assert row.drop == (2.5, -2.5)
    assert row.pickup == (0.5, -2.5)
    assert row.network_id == 91
    assert row.temperature == 165
    assert row.inventories == {
        1: {"iron-plate": 12},
        2: {"iron-gear-wheel": 3},
    }
    assert row.fluids == {"water": 99.0}


def test_apply_wire_records_and_terrain_patches():
    world = WorldClient.__new__(WorldClient)
    WorldClient._init_state(world)
    world.apply_entity(
        "h123:automation:17:3.25:-4.5;u6,character,3.25,-4.5,0,1;"
        "u7,stone-furnace,3,0,0,1,W2:2"
    )
    world.apply_terrain(
        "c0:0:;oiron-ore:1:1:5;oiron-ore:2:1:5;oiron-ore:2:2:5;"
        "oiron-ore:9:9:5;ocoal:-1:0:2;t4:5;k6:7;n8,biter-spawner,9,10"
    )

    assert world.tick == 123
    assert world.research == "automation"
    assert world.entities[7].name == "stone-furnace"
    assert 6 not in world.entities
    assert (4, 5) in world.trees
    assert (6, 7) in world.obstacles
    assert world.nests[8] == ("biter-spawner", 9.0, 10.0)

    patches = world.patches()
    assert [(p.id, p.name, p.n_tiles, p.bbox) for p in patches] == [
        ("coal:-1:0", "coal", 1, (-1, 0, -1, 0)),
        ("iron-ore:1:1", "iron-ore", 3, (1, 1, 2, 2)),
        ("iron-ore:9:9", "iron-ore", 1, (9, 9, 9, 9)),
    ]
    assert patches[1].nearest_tile(3, 2) == (2, 2)

    world.apply_entity("r7")
    world.apply_terrain("d2:2;x4:5;m8")
    assert not world.entities
    assert (2, 2) not in world.ores
    assert (4, 5) not in world.trees
    assert not world.nests


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Nothing within reach to harvest", "nothing_to_harvest"),
        ("target position is too far away", "out_of_reach"),
        ("Invalid count: must be greater than 0", "invalid_quantity"),
        ("Failed to find a path", "no_path"),
        ("Cannot place stone-furnace", "cannot_place"),
        ("No coal to insert from your inventory", "not_in_inventory"),
        ("still missing ingredients", "missing_ingredients"),
        ("recipe is not unlocked yet", "recipe_disabled"),
        ("requires a crafting machine", "not_hand_craftable"),
        ("Technology nope does not exist", "tech_unknown"),
        ("Technology automation is already researched", "tech_researched"),
        ("Missing prerequisites: steam-power", "tech_prerequisite"),
        ("No entity to rotate", "entity_not_found"),
        ("inventory is full", "no_suitable_slot"),
        ("No iron-plate found in any nearby entities", "nothing_to_extract"),
        ("Failed to connect pipes", "connect_failed"),
        ("AssertionError: bad type", "assertion"),
        ("an entirely new failure", "other"),
    ],
)
def test_classify_error(message, expected):
    assert classify_error("HARVEST", message) == expected


@pytest.mark.parametrize(
    ("regime", "op", "reason"),
    [
        (regime, op, "no_entity")
        for regime in ("naive", "macro")
        for op in ("PICKUP", "ROTATE", "INSERT", "EXTRACT")
    ]
    + [
        ("naive", "SET_RECIPE", "no_entity"),
        ("macro", "SET_RECIPE", "no_assembling_machine"),
        ("naive", "CONNECT", "fewer_than_two_entities"),
        ("macro", "CONNECT", "fewer_than_two_entities"),
    ],
)
def test_empty_world_entity_ops_have_no_support(regime, op, reason):
    sampled = sample_action(op, regime, _state(), random.Random(3))
    assert sampled == NoSupport(reason)


def test_macro_patch_produces_harvest_spec():
    world = FakeWorld(
        ores={(50, 0): ("iron-ore", 5), (51, 0): ("iron-ore", 5)}
    )
    sampled = sample_action("HARVEST", "macro", _state(world), random.Random(1))

    assert isinstance(sampled, ActionSpec)
    assert sampled.op == "HARVEST"
    assert sampled.args["target_kind"] == "patch"
    assert sampled.args["target"] == [50.5, 0.5]
    assert sampled.args["quantity"] in (1, 5, 20)


def test_naive_harvest_offsets_stay_in_contract_square():
    rng = random.Random(9)
    samples = [sample_action("HARVEST", "naive", _state(), rng) for _ in range(500)]
    assert all(isinstance(sample, ActionSpec) for sample in samples)
    assert all(-8 <= sample.args["dx"] <= 8 for sample in samples)
    assert all(-8 <= sample.args["dy"] <= 8 for sample in samples)
    assert {sample.args["quantity"] for sample in samples} == {1, 5, 20, -1}


def test_recursive_craft_support_matches_server_ingredient_accounting():
    vocab = VocabData.load(S.VOCAB_PATH)
    enabled = {
        "wooden-chest",
        "iron-gear-wheel",
        "stone-furnace",
        "burner-mining-drill",
    }

    assert "wooden-chest" in recursively_craftable_recipes(
        {"wood": 2}, enabled, vocab
    )
    assert "burner-mining-drill" in recursively_craftable_recipes(
        {"iron-plate": 9, "stone": 5}, enabled, vocab
    )
    assert "burner-mining-drill" not in recursively_craftable_recipes(
        {"iron-plate": 8, "stone": 5}, enabled, vocab
    )
    assert "stone-furnace" not in recursively_craftable_recipes(
        {"stone": 4}, enabled, vocab
    )
    assert "recipe-unknown" not in recursively_craftable_recipes(
        {}, {"recipe-unknown"}, vocab
    )


@pytest.mark.parametrize(
    ("action", "before", "after", "expected", "detail_key"),
    [
        (
            ActionSpec("WAIT", {"ticks": 60}),
            _snapshot(tick=10),
            _snapshot(tick=11),
            True,
            "actual_ticks",
        ),
        (
            ActionSpec("MOVE", {"target": [5.0, 0.0]}),
            _snapshot(),
            _snapshot(position=(4.25, 0.0)),
            True,
            "final_distance",
        ),
        (
            ActionSpec("HARVEST", {"target": [1, 1], "quantity": 1}),
            _snapshot(inventory={}),
            _snapshot(inventory={"wood": 1}),
            True,
            "items_acquired",
        ),
        (
            ActionSpec(
                "CRAFT",
                {"recipe": "iron-gear-wheel", "main_product": "iron-gear-wheel"},
            ),
            _snapshot(inventory={"iron-gear-wheel": 1}),
            _snapshot(inventory={"iron-gear-wheel": 2}),
            True,
            "product_delta",
        ),
        (
            ActionSpec("PLACE", {"prototype": "wooden-chest"}),
            _snapshot(entities={}),
            _snapshot(entities={2: _row(2)}),
            True,
            "row_count_delta",
        ),
        (
            ActionSpec(
                "PICKUP",
                {"anchor": {"unit": 1, "name": "wooden-chest", "x": 1.5, "y": 2.5}},
            ),
            _snapshot(inventory={}, entities={1: _row(1)}),
            _snapshot(inventory={"wooden-chest": 1}, entities={}),
            True,
            "unit_gone",
        ),
        (
            ActionSpec(
                "ROTATE",
                {
                    "anchor": {"unit": 1, "name": "wooden-chest", "x": 1.5, "y": 2.5},
                    "direction": 4,
                },
            ),
            _snapshot(entities={1: _row(1, direction=0)}),
            _snapshot(entities={1: _row(1, direction=4)}),
            True,
            "direction_after",
        ),
        (
            ActionSpec("INSERT", {"item": "coal"}),
            _snapshot(inventory={"coal": 2}),
            _snapshot(inventory={"coal": 1}),
            True,
            "item_delta",
        ),
        (
            ActionSpec("EXTRACT", {"item": "coal"}),
            _snapshot(inventory={"coal": 1}),
            _snapshot(inventory={"coal": 2}),
            True,
            "item_delta",
        ),
        (
            ActionSpec(
                "SET_RECIPE",
                {
                    "anchor": {"unit": 1, "name": "assembling-machine-1", "x": 1.5, "y": 2.5},
                    "recipe": "iron-gear-wheel",
                },
            ),
            _snapshot(entities={1: _row(1, name="assembling-machine-1")}),
            _snapshot(
                entities={
                    1: EntityRow(
                        unit=1,
                        name="assembling-machine-1",
                        x=1.5,
                        y=2.5,
                        direction=0,
                        status=1,
                        recipe="iron-gear-wheel",
                    )
                }
            ),
            True,
            "recipe_after",
        ),
        (
            ActionSpec("CONNECT", {}),
            _snapshot(entities={1: _row(1)}),
            _snapshot(entities={1: _row(1), 2: _row(2)}),
            True,
            "row_count_delta",
        ),
        (
            ActionSpec("RESEARCH", {"technology": "automation"}),
            _snapshot(research=None),
            _snapshot(research="automation"),
            True,
            "research_after",
        ),
    ],
)
def test_effect_verifier_successes(action, before, after, expected, detail_key):
    ok, details = verify_effect(action, before, after)
    assert ok is expected
    assert detail_key in details


def test_effect_verifier_reports_no_effect():
    action = ActionSpec("HARVEST", {"target": [1, 1], "quantity": 1})
    ok, details = verify_effect(
        action,
        _snapshot(inventory={"wood": 2}),
        _snapshot(inventory={"wood": 2}, tick=12),
    )
    assert not ok
    assert details["items_acquired"] == 0
