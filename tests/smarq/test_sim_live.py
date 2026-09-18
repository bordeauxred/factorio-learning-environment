import os
import time

import pytest

from fle.env import FactorioInstance
from fle.smarq.sim import MEASURED_EXECUTION_SPEED, SimClock


@pytest.fixture
def live_clock():
    port = int(os.environ.get("FACTORIO_RCON_PORT", "27000"))
    assert port in {27000, 27001}
    instance = FactorioInstance(
        address="localhost",
        tcp_port=port,
        fast=True,
        cache_scripts=True,
        all_technologies_researched=False,
        inventory={},
        reset_speed=MEASURED_EXECUTION_SPEED,
    )
    clock = SimClock(instance)
    try:
        yield clock
    finally:
        clock.close()
        instance.cleanup()


@pytest.mark.live
def test_pause_freezes_ticks_across_compute_stall(live_clock) -> None:
    before = live_clock.boundary()
    time.sleep(2.0)
    after = live_clock.tick()
    assert after == before


@pytest.mark.live
@pytest.mark.parametrize("ticks", [60, 600])
def test_fast_forward_is_exact_and_faster_than_simulated_time(live_clock, ticks) -> None:
    wall_start = time.perf_counter()
    sample = live_clock.advance_ticks(ticks)
    wall = time.perf_counter() - wall_start
    assert sample.duration_ticks == ticks
    assert sample.overshoot_ticks == 0
    assert wall < ticks / 60
