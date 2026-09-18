"""Exact simulated-time control for a live Factorio instance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from fle.smarq import contract as C

T = TypeVar("T")

# The controlled measurement in docs/rl/specs/sim-probe.md found that requested
# speed 40 was the ceiling (1165.77 UPS); larger settings were slightly slower.
MEASURED_EXECUTION_SPEED = 40.0


@dataclass(frozen=True)
class ClockSample:
    duration_ticks: int
    duration_game_seconds: float
    wall_seconds: float
    overshoot_ticks: int = 0


class SimClock:
    """Pause at decisions and advance Factorio by observed game ticks."""

    def __init__(
        self,
        instance: Any,
        execution_speed: float = MEASURED_EXECUTION_SPEED,
        poll_interval: float = 0.002,
        timeout_seconds: float = 120.0,
    ) -> None:
        if execution_speed <= 0:
            raise ValueError("execution speed must be positive")
        self.instance = instance
        self.rcon = instance.rcon_client
        self.execution_speed = float(execution_speed)
        self.poll_interval = float(poll_interval)
        self.timeout_seconds = float(timeout_seconds)
        self.last = ClockSample(0, 0.0, 0.0, 0)
        self.overshoots: list[int] = []
        self._handler_installed = False
        self._install_exact_handler()

    def _command(self, lua: str) -> str:
        return self.rcon.send_command("/sc " + lua) or ""

    def tick(self) -> int:
        return int(self._command("rcon.print(game.tick)"))

    def _install_exact_handler(self) -> None:
        response = self._command(
            "storage.__smarq_target=nil storage.__smarq_hit=nil "
            "script.on_nth_tick(1,function(event) "
            "if storage.__smarq_target and event.tick>=storage.__smarq_target then "
            "storage.__smarq_hit=game.tick game.tick_paused=true "
            "storage.__smarq_target=nil end end) rcon.print('registered')"
        )
        if response != "registered":
            raise RuntimeError(f"could not register exact tick handler: {response}")
        self._handler_installed = True

    @staticmethod
    def _sample(start_tick: int, end_tick: int, wall: float, overshoot: int = 0) -> ClockSample:
        ticks = end_tick - start_tick
        return ClockSample(ticks, ticks / C.TICKS_PER_SECOND, wall, overshoot)

    def boundary(self) -> int:
        """Synchronously pause the simulation and return its authoritative tick."""
        response = self._command("game.tick_paused=true rcon.print(game.tick)")
        return int(response)

    def run_while_executing(self, function: Callable[[], T]) -> tuple[T, ClockSample]:
        """Run one tool callable with Factorio unpaused, then re-pause."""
        start_tick = self.boundary()
        wall_start = time.perf_counter()
        self._command(f"game.speed={self.execution_speed:g} game.tick_paused=false")
        try:
            result = function()
        except BaseException:
            end_tick = int(self._command("game.tick_paused=true rcon.print(game.tick)"))
            self.last = self._sample(start_tick, end_tick, time.perf_counter() - wall_start)
            raise
        end_tick = int(self._command("game.tick_paused=true rcon.print(game.tick)"))
        self.last = self._sample(start_tick, end_tick, time.perf_counter() - wall_start)
        return result, self.last

    def advance_ticks(self, ticks: int) -> ClockSample:
        """Advance exactly ``ticks`` with a Lua tick handler and re-pause."""
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        start_tick = self.boundary()
        if ticks == 0:
            self.last = ClockSample(0, 0.0, 0.0, 0)
            self.overshoots.append(0)
            return self.last
        target = start_tick + int(ticks)
        wall_start = time.perf_counter()
        self._command(
            f"storage.__smarq_target={target} storage.__smarq_hit=nil "
            f"game.speed={self.execution_speed:g} game.tick_paused=false"
        )
        deadline = wall_start + self.timeout_seconds
        while True:
            response = self._command(
                "rcon.print(tostring(game.tick_paused)..','..game.tick..','.."
                "tostring(storage.__smarq_hit))"
            )
            paused, final_text, _ = response.split(",", 2)
            final_tick = int(final_text)
            if paused == "true":
                overshoot = final_tick - target
                self.last = self._sample(
                    start_tick,
                    final_tick,
                    time.perf_counter() - wall_start,
                    overshoot,
                )
                self.overshoots.append(overshoot)
                return self.last
            if time.perf_counter() >= deadline:
                self.boundary()
                raise TimeoutError(f"Factorio did not pause at tick {target}: {response}")
            # This is a short status poll only while the simulator is running.
            time.sleep(self.poll_interval)

    def run_until(self, predicate: Callable[[], bool], max_ticks: int) -> ClockSample:
        """Run until a polled predicate succeeds or an exact tick cap is hit."""
        if max_ticks < 0:
            raise ValueError("max_ticks must be non-negative")
        start_tick = self.boundary()
        if max_ticks == 0:
            self.last = ClockSample(0, 0.0, 0.0, 0)
            return self.last
        target = start_tick + max_ticks
        wall_start = time.perf_counter()
        self._command(
            f"storage.__smarq_target={target} storage.__smarq_hit=nil "
            f"game.speed={self.execution_speed:g} game.tick_paused=false"
        )
        deadline = wall_start + self.timeout_seconds
        try:
            while not predicate():
                status = self._command("rcon.print(tostring(game.tick_paused)..','..game.tick)")
                paused, _ = status.split(",", 1)
                if paused == "true":
                    break
                if time.perf_counter() >= deadline:
                    raise TimeoutError("run_until timed out")
                time.sleep(self.poll_interval)
        finally:
            final_tick = self.boundary()
        overshoot = max(0, final_tick - target)
        self.last = self._sample(
            start_tick,
            final_tick,
            time.perf_counter() - wall_start,
            overshoot,
        )
        return self.last

    def close(self) -> None:
        self.boundary()
        if self._handler_installed:
            self._command(
                "storage.__smarq_target=nil storage.__smarq_hit=nil "
                "script.on_nth_tick(1,nil) rcon.print('removed')"
            )
            self._handler_installed = False
