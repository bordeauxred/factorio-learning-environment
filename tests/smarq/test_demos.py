from __future__ import annotations

import numpy as np
import pytest

from fle.smarq import contract as C
from fle.smarq.demos import BurnerAutomationDemo, HandMiningDemo, collect_demonstration
from fle.smarq.fake import FakeSemanticEnv


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for offline tests."""
    yield


class CaptureReplay:
    def __init__(self) -> None:
        self.values: list[tuple[C.Transition, float]] = []

    def add(self, transition: C.Transition, priority: float = 1.0) -> None:
        self.values.append((transition, priority))


def test_burner_demo_preserves_script_coordinates_and_heads():
    env = FakeSemanticEnv(raster_tiles=16, decision_cap=20)
    replay = CaptureReplay()
    demo = BurnerAutomationDemo(
        ore_tile=(2, -1),
        furnace_tile=(5, 1),
        craft_furnace=False,
    )
    result = collect_demonstration(env, demo, replay, seed=3, initial_priority=12.0)
    assert all(isinstance(action, C.Action) for action in result.actions)
    assert [action.tile for action in result.actions if action.verb == "MINE"] == [(2, -1)]
    placed = next(action for action in result.actions if action.verb == "PLACE")
    assert placed.tile == (5, 1)
    position = placed.head(C.POSITION)
    # The encoded head decodes to the exact requested world tile.
    place_transition_index = [a.verb for a in result.actions].index("PLACE")
    origin = tuple(result.transitions[place_transition_index].observation["raster_origin"])
    assert C.position_to_tile(position, origin, 16) == (5, 1)
    assert all(transition.is_demo for transition, _ in replay.values)
    assert all(priority == 12.0 and priority > 1.0 for _, priority in replay.values)
    assert all(action.heads.dtype == np.int64 for action in result.actions)


def test_hand_mining_is_valid_weak_baseline():
    env = FakeSemanticEnv(raster_tiles=16, decision_cap=20)
    result = collect_demonstration(
        env,
        HandMiningDemo(ore_tiles=((2, 0), (3, 0))),
        seed=4,
    )
    assert [action.verb for action in result.actions] == [
        "MOVE_TO",
        "MINE",
        "MOVE_TO",
        "MINE",
    ]
    assert [action.tile for action in result.actions] == [(2, 0), (2, 0), (3, 0), (3, 0)]
    assert result.transitions[-1].observation["tick"] < result.transitions[-1].next_observation[
        "tick"
    ]
