from types import SimpleNamespace

import numpy as np

from fle.smarq import contract as C
from fle.smarq.actions import ActionCodec
from fle.smarq.contract import Action, Masks, Observation
from fle.smarq.env import ExecutionOutcome, FLEActionExecutor, SemanticEnv
from fle.smarq.vocab import StableVocab


def make_action(verb: str, **kwargs) -> Action:
    return Action(verb=verb, heads=C.empty_heads(), **kwargs)


class RecordingExecutor(FLEActionExecutor):
    def __init__(self, response=None, error=None):
        self.client = SimpleNamespace(entity_ids=np.zeros(C.ENTITY_SLOTS, dtype=np.int64))
        self.response = response
        self.error = error
        self.navigated = []
        self.calls = []

    def _navigate(self, tile, remaining, exact_destination=False):
        self.navigated.append((tile, exact_destination))
        return 17

    def _run_tool(self, name, args, remaining):
        self.calls.append((name, args, remaining))
        if self.error:
            raise RuntimeError(self.error)
        return self.response, 3, 0


def test_exact_placement_coordinate_reaches_tool_unchanged() -> None:
    executor = RecordingExecutor(response={"position": {"x": 11, "y": -7}})
    obs = empty_observation(0)
    obs = Observation(**{**obs.as_dict(), "raster_origin": (7, -11)})
    stable_vocab = StableVocab(
        prototypes=("<none>", "stone-furnace"),
        items=("<none>", "coal"),
        recipes=("<none>", "stone-furnace"),
        technologies=("<none>", "automation"),
        entity_types=("<none>", "stone-furnace"),
    )
    heads = C.empty_heads()
    heads[C.HEAD_INDEX[C.PROTOTYPE]] = 1
    heads[C.HEAD_INDEX[C.POSITION]] = 4 * obs.raster_tiles + 4
    heads[C.HEAD_INDEX[C.DIRECTION]] = C.DIRECTIONS.index("WEST")
    action = ActionCodec(stable_vocab).decode(C.VERB_INDEX["PLACE"], heads, obs)
    outcome = executor.execute(action, 1000)
    assert outcome.success
    assert executor.navigated == [((11, -7), False)]
    assert executor.calls[0][0] == "place_entity"
    assert executor.calls[0][1][2:4] == (11, -7)
    assert executor.calls[0][1][-1] is True


def test_invalid_placement_fails_instead_of_trying_another_tile() -> None:
    executor = RecordingExecutor(error="Cannot place stone-furnace - blocked by water")
    action = make_action(
        "PLACE",
        tile=(8, 9),
        prototype="stone-furnace",
        direction="NORTH",
    )
    outcome = executor.execute(action, 1000)
    assert outcome == ExecutionOutcome(False, "blocked", None, 0, 0)
    assert executor.navigated == [((8, 9), False)]
    assert len(executor.calls) == 1
    assert executor.calls[0][1][2:4] == (8, 9)


def test_navigation_inside_move_preserves_requested_target() -> None:
    executor = RecordingExecutor()
    target = (-19, 37)
    outcome = executor.execute(make_action("MOVE_TO", tile=target), 1000)
    assert outcome.success
    assert executor.navigated == [(target, True)]
    assert executor.calls == []


def empty_observation(tick: int) -> Observation:
    return Observation(
        grid=np.zeros((C.GRID_CHANNELS, C.GRID_SIZE, C.GRID_SIZE), np.float32),
        entity_view=np.zeros((C.ENTITY_SLOTS, C.ENTITY_FEATURES), np.float32),
        entity_mask=np.zeros(C.ENTITY_SLOTS, bool),
        entity_ids=np.zeros(C.ENTITY_SLOTS, np.int64),
        globals=np.zeros(C.N_GLOBALS, np.float32),
        raster=np.zeros((C.RASTER_CHANNELS, 8, 8), np.float32),
        raster_origin=(-4, -4),
        raster_tiles=8,
        player_tile=(0, 0),
        tick=tick,
        production_score=0,
        automated_production_score=0,
    )


class BudgetClock:
    def __init__(self):
        self.now = 100

    def tick(self):
        return self.now

    def boundary(self):
        return self.now

    def advance_ticks(self, ticks):
        self.now += ticks
        return SimpleNamespace(overshoot_ticks=0)


class BudgetEnv(SemanticEnv):
    def __init__(self):
        self.clock = BudgetClock()
        self.instance = SimpleNamespace(namespace=SimpleNamespace(score=lambda: (0.0, 0.0)))
        self.episode_start_tick = 100
        self.tick_budget = 650
        self.decision_cap = 1000
        self.episode_decisions = 0
        self.reward_mode = "automated"

    def _observe(self, full=False):
        return empty_observation(self.clock.tick())

    def masks(self):
        return Masks(
            verb=np.ones(C.N_VERBS, bool),
            prototype=np.ones(1, bool),
            item=np.ones(1, bool),
            craft_recipe=np.ones(1, bool),
            technology=np.ones(1, bool),
            entity=np.zeros((C.N_VERBS, C.ENTITY_SLOTS), bool),
        )


def test_repeated_fast_forward_cannot_exceed_episode_budget() -> None:
    env = BudgetEnv()
    action = make_action("FAST_FORWARD", duration_seconds=60)
    first, _ = env.step(action)
    assert first.done
    assert first.episode_ticks == 650
    assert first.duration_ticks == 650
    second, _ = env.step(action)
    assert second.episode_ticks == 650
    assert second.duration_ticks == 0
