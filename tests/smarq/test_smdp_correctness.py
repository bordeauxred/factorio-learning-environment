"""TOY-A: does the semi-Markov arithmetic do what the brief says it should?

The question this file answers is narrow and worth answering before any expensive
learning: when two choices have a similar nominal outcome but different simulated
durations, does the longer one bootstrap with a smaller discount, and do the
Bellman targets match hand-computed numbers exactly?

Everything here is deterministic and runs offline in well under a second.  The
learner is used when it is importable; the arithmetic assertions stand on their
own either way, which is the point - this is a correctness test, not a learning
test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fle.smarq import contract as C
from fle.smarq.fake import FakeSemanticEnv

H = C.DISCOUNT_HORIZON_SECONDS_DEFAULT


def test_discount_is_halving_per_horizon():
    assert C.smdp_discount(0.0) == 1.0
    assert C.smdp_discount(H) == pytest.approx(0.5)
    assert C.smdp_discount(2 * H) == pytest.approx(0.25)
    # And it is exactly 2**(-tau/H), not an approximation of e**(-tau/H).
    for tau in (1.0, 60.0, 600.0, 3600.0, 7200.0):
        assert C.smdp_discount(tau) == pytest.approx(2.0 ** (-tau / H))


def test_longer_simulated_duration_discounts_more():
    """The core SMDP property: time is the cost, not the number of decisions."""
    short, long_ = C.smdp_discount(60.0), C.smdp_discount(600.0)
    assert short > long_
    # Ten FAST_FORWARD(60s) and one FAST_FORWARD(600s) cover the same game time,
    # so they must bootstrap identically.  If they do not, the discount is being
    # applied per decision rather than per simulated second.
    assert C.smdp_discount(60.0) ** 10 == pytest.approx(C.smdp_discount(600.0))


def test_bellman_target_matches_hand_computation():
    r, tau, q_next = 7.0, 600.0, 3.0
    target = r + C.smdp_discount(tau) * q_next
    assert target == pytest.approx(7.0 + (2.0 ** (-600.0 / 3600.0)) * 3.0)
    assert target == pytest.approx(7.0 + 0.8908987181403393 * 3.0)


def test_internal_autoregressive_steps_are_timeless():
    """Choosing PLACE, then a prototype, then a tile costs no game time."""
    assert C.smdp_discount(0.0) == 1.0
    chain = ("PLACE", "stone-furnace", "tile", "EAST")
    total = 1.0
    for _ in chain:
        total *= C.smdp_discount(0.0)
    assert total == 1.0


def test_two_routes_to_the_same_outcome_differ_only_by_simulated_time():
    """The toy case the brief asks for.

    Two action sequences reach the same factory state and the same automated
    score.  One spends its simulated time walking, the other does not.  The
    shorter one must be preferred by the SMDP return even though the nominal
    reward is identical.
    """
    near, far = FakeSemanticEnv(seed=0), FakeSemanticEnv(seed=0)
    near.reset(0)
    far.reset(0)

    def place_and_feed(env, tile):
        obs = env.observe()
        heads = C.empty_heads()
        heads[C.HEAD_INDEX[C.PROTOTYPE]] = 1
        pos = C.tile_to_position(tile, obs.raster_origin, obs.raster_tiles)
        assert pos is not None, "test tile must lie inside the window"
        heads[C.HEAD_INDEX[C.POSITION]] = pos
        heads[C.HEAD_INDEX[C.DIRECTION]] = 0
        res, _ = env.step(
            C.Action(
                verb="PLACE",
                heads=heads,
                tile=tile,
                prototype="stone-furnace",
                direction="NORTH",
            )
        )
        assert res.success, res.failure_reason
        feed = C.empty_heads()
        feed[C.HEAD_INDEX[C.ENTITY]] = 0
        feed[C.HEAD_INDEX[C.ITEM]] = 1
        feed[C.HEAD_INDEX[C.QUANTITY]] = 3
        res2, _ = env.step(
            C.Action(verb="INSERT", heads=feed, entity_slot=0, item="coal", quantity=8)
        )
        assert res2.success, res2.failure_reason
        return res.duration_ticks + res2.duration_ticks

    # Both furnaces sit on ore; the only difference is how far the character walks.
    ticks_near = place_and_feed(near, (3, 0))
    ticks_far = place_and_feed(far, (5, 1))
    assert ticks_far > ticks_near

    # Now both fast-forward the same amount and earn the same reward.
    ff = C.empty_heads()
    ff[C.HEAD_INDEX[C.DURATION]] = 2
    action = C.Action(verb="FAST_FORWARD", heads=ff, duration_seconds=60)
    r_near, _ = near.step(action)
    r_far, _ = far.step(action)
    # Nominally the same outcome: the longer route burns a little more fuel while
    # walking, so allow a few percent rather than demanding bit equality.
    assert r_near.reward == pytest.approx(r_far.reward, rel=0.05)

    # Discounted from the episode start, the cheaper route is worth more, purely
    # because it spent less simulated time getting there.
    value_near = C.smdp_discount(ticks_near / 60.0) * r_near.reward
    value_far = C.smdp_discount(ticks_far / 60.0) * r_far.reward
    assert value_near > value_far


def test_fast_forward_cannot_buy_extra_episode_time():
    """Waiting is not free: it consumes the same finite simulated budget."""
    env = FakeSemanticEnv(tick_budget=3600, decision_cap=1000, seed=0)
    env.reset(0)
    ff = C.empty_heads()
    ff[C.HEAD_INDEX[C.DURATION]] = 2
    action = C.Action(verb="FAST_FORWARD", heads=ff, duration_seconds=60)
    steps, done = 0, False
    while not done and steps < 50:
        res, _ = env.step(action)
        done, steps = res.done, steps + 1
    assert done
    assert steps == 1  # one 3600-tick wait exhausts a 3600-tick budget
    assert env.tick <= 3600


def test_episode_budget_is_simulated_not_wall_time():
    env = FakeSemanticEnv(tick_budget=600, decision_cap=1000, seed=0)
    env.reset(0)
    ff = C.empty_heads()
    ff[C.HEAD_INDEX[C.DURATION]] = 1
    res, _ = env.step(C.Action(verb="FAST_FORWARD", heads=ff, duration_seconds=10))
    assert res.duration_ticks == 600
    assert res.duration_game_seconds == pytest.approx(10.0)
    # The toy env returns in microseconds; simulated time is what ended the episode.
    assert res.wall_seconds < 1.0
    assert res.done and res.end_reason == "tick_budget"


@pytest.mark.parametrize("horizon", [600.0, 1800.0, 3600.0])
def test_horizon_sweep_orders_values_consistently(horizon):
    """A sweep over H must not reorder two options that differ only in duration."""
    fast, slow = 30.0, 300.0
    assert C.smdp_discount(fast, horizon) > C.smdp_discount(slow, horizon)
    ratio = C.smdp_discount(slow, horizon) / C.smdp_discount(fast, horizon)
    assert ratio == pytest.approx(2.0 ** (-(slow - fast) / horizon))


def test_learner_target_uses_simulated_duration_if_learner_is_available():
    """Bridge to S2's learner: the same arithmetic, through the real code path."""
    learner_mod = pytest.importorskip("fle.smarq.learner")
    fn = None
    for name in ("smdp_target", "compute_target", "bellman_target"):
        fn = getattr(learner_mod, name, None)
        if callable(fn):
            break
    if fn is None:
        pytest.skip("learner exposes no standalone target function to check")
    r, q_next = 2.0, 5.0
    short = fn(np.float32(r), np.float32(60.0), np.float32(q_next))
    long_ = fn(np.float32(r), np.float32(600.0), np.float32(q_next))
    assert float(short) > float(long_)
    assert float(short) == pytest.approx(r + math.pow(2.0, -60.0 / H) * q_next, rel=1e-5)
