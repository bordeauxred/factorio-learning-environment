"""Live simulated-time measurements for the semi-Markov RL design.

Run from the repository root:
    .venv/bin/python tests/benchmarks/smarq_sim_probe.py --port 27004

Only ports 27004 and 27005 are accepted.  The probe restores the selected
server to an unpaused, speed-10, freshly reset state in its finalizer.
"""

import argparse
import math
import statistics
import sys
import time
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_tensor_obs import TensorClient  # noqa: E402
from fle.env import FactorioInstance  # noqa: E402
from fle.env.entities import Position  # noqa: E402
from fle.env.game_types import Prototype, Resource  # noqa: E402


ALLOWED_PORTS = {27004, 27005}
SPEEDS = (1, 10, 40, 100, 400, 1000)
SAMPLES = 30
RCON_NOOP = "/sc rcon.print(1)"


def percentile95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def summary_ms(values):
    return statistics.mean(values) * 1000, percentile95(values) * 1000


def print_rows(headers, rows):
    print(" | ".join(headers), flush=True)
    print(" | ".join("---" for _ in headers), flush=True)
    for row in rows:
        print(" | ".join(str(value) for value in row), flush=True)


class Probe:
    def __init__(self, port):
        self.port = port
        self.instance = FactorioInstance(
            address="localhost",
            tcp_port=port,
            fast=True,
            cache_scripts=True,
            all_technologies_researched=False,
            inventory={},
            reset_speed=10,
        )
        self.ns = self.instance.namespace
        self.rc = self.instance.rcon_client
        self.best_speed = 10
        self.handler_installed = False

    def command(self, lua):
        return self.rc.send_command("/sc " + lua)

    def tick(self):
        return int(self.command("rcon.print(game.tick)"))

    def reset(self):
        self.instance.reset(
            reset_position=True,
            all_technologies_researched=False,
        )

    def pause(self):
        self.command("game.tick_paused=true")

    def unpause_at(self, speed):
        self.command(f"game.speed={speed} game.tick_paused=false")

    def setup_furnace(self):
        response = self.command(
            "local s=game.surfaces[1] "
            "local c=nil for _,ch in pairs(storage.agent_characters or {}) do "
            "if ch and ch.valid then c=ch break end end "
            "if not c then error('no valid FLE agent character') end "
            "local p=s.find_non_colliding_position('stone-furnace', "
            "{c.position.x+4,c.position.y}, 24, 0.5) "
            "local f=s.create_entity{name='stone-furnace',position=p,"
            "force='player',raise_built=true} "
            "local fuel=f.get_inventory(defines.inventory.fuel) "
            "local source=f.get_inventory(defines.inventory.furnace_source) "
            "local nf=fuel.insert{name='coal',count=50} "
            "local no=source.insert{name='iron-ore',count=50} "
            "storage.__smarq_furnace=f "
            "rcon.print(f.position.x..','..f.position.y..','..nf..','..no)"
        )
        return response

    def furnace_state(self):
        return self.command(
            "local f=storage.__smarq_furnace "
            "if f and f.valid then "
            "local out=f.get_inventory(defines.inventory.furnace_result) "
            "rcon.print(f.status..','..f.crafting_progress..','..out.get_item_count('iron-plate')) "
            "else rcon.print('missing') end"
        )

    def install_exact_handler(self):
        response = self.command(
            "storage.__smarq_target=nil storage.__smarq_hit=nil "
            "script.on_nth_tick(1,function(event) "
            "if storage.__smarq_target and event.tick>=storage.__smarq_target then "
            "storage.__smarq_hit=game.tick game.tick_paused=true "
            "storage.__smarq_target=nil end end) "
            "rcon.print('registered')"
        )
        if response != "registered":
            raise RuntimeError(f"exact handler registration failed: {response}")
        self.handler_installed = True
        return response

    def remove_exact_handler(self):
        if self.handler_installed:
            self.command(
                "storage.__smarq_target=nil storage.__smarq_hit=nil "
                "script.on_nth_tick(1,nil) rcon.print('removed')"
            )
            self.handler_installed = False

    def exact_advance(self, ticks, speed=None):
        speed = speed or self.best_speed
        base = self.tick()
        target = base + ticks
        started = time.perf_counter()
        self.command(
            f"storage.__smarq_target={target} game.speed={speed} "
            "game.tick_paused=false rcon.print(game.tick)"
        )
        deadline = time.perf_counter() + 120
        while True:
            status = self.command(
                "rcon.print(tostring(game.tick_paused)..','..game.tick..','.."
                "tostring(storage.__smarq_hit))"
            )
            paused, final_tick, hit = status.split(",", 2)
            if paused == "true":
                elapsed = time.perf_counter() - started
                return {
                    "base": base,
                    "target": target,
                    "final": int(final_tick),
                    "hit": int(hit),
                    "overshoot": int(final_tick) - target,
                    "wall": elapsed,
                }
            if time.perf_counter() > deadline:
                raise TimeoutError(f"exact advance did not pause at {target}: {status}")

    def q1_pause_correctness(self):
        print("\nQ1 PAUSE CORRECTNESS", flush=True)
        rows = []
        self.reset()
        self.pause()
        tick0 = self.tick()
        t0 = time.perf_counter()
        time.sleep(3.0)
        tick1 = self.tick()
        rows.append(("fresh map", tick0, tick1, tick1 - tick0, f"{time.perf_counter()-t0:.4f}"))

        self.reset()
        setup = self.setup_furnace()
        self.unpause_at(10)
        time.sleep(0.10)
        self.pause()
        before = self.furnace_state()
        tick0 = self.tick()
        t0 = time.perf_counter()
        time.sleep(3.0)
        tick1 = self.tick()
        after = self.furnace_state()
        pong = self.command("rcon.print('pong@'..game.tick)")
        rows.append(("fuelled furnace", tick0, tick1, tick1 - tick0, f"{time.perf_counter()-t0:.4f}"))
        print(f"furnace setup x,y,coal,ore: {setup}")
        print(f"furnace status,progress,plates before: {before}; after: {after}")
        print(f"RCON while paused: {pong}")
        print_rows(("state", "tick before", "tick after", "delta", "wall s"), rows)

    def q2_speed_ceiling(self):
        print("\nQ2 SPEED CEILING", flush=True)
        self.reset()
        rows = []
        rates = []
        for requested in SPEEDS:
            self.unpause_at(requested)
            time.sleep(0.10)
            actual = float(self.command("rcon.print(game.speed)"))
            tick0 = self.tick()
            started = time.perf_counter()
            time.sleep(3.0)
            tick1 = self.tick()
            wall = time.perf_counter() - started
            rate = (tick1 - tick0) / wall
            game_seconds = rate / 60.0
            rates.append(rate)
            rows.append(
                (
                    requested,
                    f"{actual:g}",
                    tick1 - tick0,
                    f"{wall:.4f}",
                    f"{rate:.2f}",
                    f"{game_seconds:.2f}",
                )
            )
        self.best_speed = SPEEDS[max(range(len(rates)), key=rates.__getitem__)]
        print_rows(
            ("requested", "engine speed", "ticks", "wall s", "ticks/wall-s", "game-s/wall-s"),
            rows,
        )
        print(f"measured best requested speed: {self.best_speed} (maximum realized tick rate)")

    def polling_advance(self, ticks):
        self.pause()
        base = self.tick()
        target = base + ticks
        started = time.perf_counter()
        self.command(f"game.speed={self.best_speed} game.tick_paused=false")
        observed = base
        polls = 0
        while observed < target:
            observed = self.tick()
            polls += 1
        final = int(self.command("game.tick_paused=true rcon.print(game.tick)"))
        return {
            "overshoot": final - target,
            "wall": time.perf_counter() - started,
            "polls": polls,
            "final": final,
            "target": target,
        }

    def q3_exact_tick_advance(self):
        print("\nQ3 EXACT TICK ADVANCE", flush=True)
        self.reset()
        poll_rows = []
        for ticks in (600, 3600):
            trials = [self.polling_advance(ticks) for _ in range(10)]
            overshoots = [r["overshoot"] for r in trials]
            walls = [r["wall"] for r in trials]
            poll_rows.append(
                (
                    "RCON poll",
                    ticks,
                    f"{statistics.mean(overshoots):.2f}",
                    max(overshoots),
                    f"{statistics.mean(walls):.4f}",
                    f"{max(walls):.4f}",
                )
            )
            print(f"poll {ticks} trial overshoots: {overshoots}")
            print(f"poll {ticks} trial wall s: {[round(v, 6) for v in walls]}")

        response = self.install_exact_handler()
        print(f"Lua handler registration: {response}")
        try:
            for ticks in (600, 3600):
                trials = [self.exact_advance(ticks) for _ in range(10)]
                overshoots = [r["overshoot"] for r in trials]
                walls = [r["wall"] for r in trials]
                poll_rows.append(
                    (
                        "Lua on_nth_tick(1)",
                        ticks,
                        f"{statistics.mean(overshoots):.2f}",
                        max(overshoots),
                        f"{statistics.mean(walls):.4f}",
                        f"{max(walls):.4f}",
                    )
                )
                print(f"handler {ticks} trial overshoots: {overshoots}")
                print(f"handler {ticks} trial wall s: {[round(v, 6) for v in walls]}")
        finally:
            self.remove_exact_handler()
        print_rows(("method", "ticks", "mean overshoot", "max overshoot", "mean wall s", "max wall s"), poll_rows)

    def q4_factory_cost(self):
        print("\nQ4 FAST-FORWARD COST VS FACTORY SIZE", flush=True)
        rows = []
        for state in ("fresh reset", "one fuelled furnace"):
            self.reset()
            setup = "none"
            if state != "fresh reset":
                setup = self.setup_furnace()
            self.pause()
            self.install_exact_handler()
            try:
                for ticks in (3600, 36000):
                    result = self.exact_advance(ticks)
                    rows.append(
                        (
                            state,
                            ticks,
                            f"{result['wall']:.4f}",
                            f"{ticks / 60 / result['wall']:.2f}",
                            result["overshoot"],
                        )
                    )
            finally:
                self.remove_exact_handler()
            furnace_end = self.furnace_state() if state != "fresh reset" else "n/a"
            print(f"{state} setup: {setup}; end furnace status,progress,plates: {furnace_end}")
        print_rows(("state", "ticks", "wall s", "game-s/wall-s", "overshoot"), rows)

    def timed_score(self):
        started = time.perf_counter()
        value = self.ns.score()
        return value, time.perf_counter() - started

    def q5_score_semantics(self):
        print("\nQ5 SCORE SEMANTICS", flush=True)
        self.reset()
        self.unpause_at(self.best_speed)
        self.command(
            "local c=nil for _,ch in pairs(storage.agent_characters or {}) do "
            "if ch and ch.valid then c=ch break end end "
            "if not c then error('no valid FLE agent character') end "
            "c.insert{name='iron-plate',count=2} "
            "rcon.print(2)"
        )
        baseline, baseline_wall = self.timed_score()
        iron = self.ns.nearest(Resource.IronOre)
        self.ns.move_to(iron)
        mined = self.ns.harvest_resource(iron, quantity=1, radius=10)
        after_mine, mine_score_wall = self.timed_score()
        crafted = self.ns.craft_item(Prototype.IronGearWheel, quantity=1)
        after_craft, craft_score_wall = self.timed_score()

        furnace_setup = self.setup_furnace()
        self.pause()
        self.install_exact_handler()
        try:
            advance = self.exact_advance(600)
        finally:
            self.remove_exact_handler()
        after_auto, auto_score_wall = self.timed_score()
        furnace_end = self.furnace_state()

        stages = (
            ("baseline (2 supplied plates)", baseline, baseline_wall),
            (f"after hand-mine {mined} iron ore", after_mine, mine_score_wall),
            (f"after hand-craft {crafted} gear", after_craft, craft_score_wall),
            ("after furnace +600 ticks", after_auto, auto_score_wall),
        )
        rows = []
        previous = baseline
        for label, value, wall in stages:
            rows.append(
                (
                    label,
                    value[0],
                    value[1],
                    value[0] - previous[0],
                    value[1] - previous[1],
                    f"{wall * 1000:.3f}",
                )
            )
            previous = value
        print(f"autonomous furnace setup x,y,coal,ore: {furnace_setup}")
        print(f"autonomous exact advance: {advance}")
        print(f"autonomous furnace status,progress,plates: {furnace_end}")
        print_rows(
            ("stage", "production", "automated", "delta production", "delta automated", "score ms"),
            rows,
        )

    def tensor_cycle(self, client):
        response = self.command("obs_all_drain()") or ""
        entity_part, _, terrain_part = response.partition("~")
        client.apply_entity(entity_part)
        client.apply_terrain(terrain_part)
        _ = client.observation()

    def q6_latency_budget(self):
        print("\nQ6 LATENCY BUDGET", flush=True)
        self.reset()
        self.pause()
        client = TensorClient()
        client.apply_entity(self.command("obs_diff_full_sync()") or "")
        client.apply_terrain(self.command("obs_terrain_full_sync()") or "")
        self.tensor_cycle(client)
        _ = self.ns.score()

        noop_times = []
        tensor_times = []
        score_times = []
        total_times = []
        for _ in range(SAMPLES):
            t0 = time.perf_counter()
            self.rc.send_command(RCON_NOOP)
            noop_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            self.tensor_cycle(client)
            tensor_times.append(time.perf_counter() - t0)

            _, elapsed = self.timed_score()
            score_times.append(elapsed)

            t0 = time.perf_counter()
            self.tensor_cycle(client)
            _ = self.ns.score()
            total_times.append(time.perf_counter() - t0)

        rows = []
        for label, values in (
            ("trivial RCON round trip", noop_times),
            ("Tensor drain+reconcile+view", tensor_times),
            ("namespace.score()", score_times),
            ("full tensor+score boundary", total_times),
        ):
            mean, p95 = summary_ms(values)
            rows.append((label, len(values), f"{mean:.3f}", f"{p95:.3f}", f"{mean / 1000:.6f}"))
        print_rows(("operation", "n", "mean ms", "p95 ms", "mean wall s"), rows)

    def q7_navigation_cost(self):
        print("\nQ7 NAVIGATION COST", flush=True)
        rows = []
        for distance in (10, 30, 60):
            self.reset()
            self.unpause_at(self.best_speed)
            start = self.ns.player_location
            target = Position(x=start.x + distance, y=start.y)
            game_tick0 = self.tick()
            elapsed0 = self.instance.get_elapsed_ticks()
            started = time.perf_counter()
            final = self.ns.move_to(target)
            game_tick1 = self.tick()
            wall = time.perf_counter() - started
            elapsed1 = self.instance.get_elapsed_ticks()
            displacement = math.hypot(final.x - start.x, final.y - start.y)
            rows.append(
                (
                    distance,
                    f"{displacement:.2f}",
                    game_tick1 - game_tick0,
                    elapsed1 - elapsed0,
                    f"{wall:.4f}",
                    f"({final.x:.2f},{final.y:.2f})",
                )
            )
        print_rows(
            ("requested tiles", "actual displacement", "game.tick delta", "modeled action ticks", "wall s", "final position"),
            rows,
        )

    def cleanup(self):
        try:
            self.remove_exact_handler()
            self.reset()
            self.command(
                "storage.__smarq_target=nil storage.__smarq_hit=nil "
                "game.speed=10 game.tick_paused=false rcon.print('restored')"
            )
            state = self.command(
                "rcon.print(game.tick..','..tostring(game.tick_paused)..','..game.speed)"
            )
            print(f"\nFINAL SERVER STATE port {self.port}: {state} (tick,paused,speed)", flush=True)
        finally:
            self.instance.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.port not in ALLOWED_PORTS:
        parser.error("--port must be 27004 or 27005")

    command = f".venv/bin/python tests/benchmarks/smarq_sim_probe.py --port {args.port}"
    print(f"EXACT COMMAND: {command}", flush=True)
    probe = Probe(args.port)
    try:
        probe.q1_pause_correctness()
        probe.q2_speed_ceiling()
        probe.q3_exact_tick_advance()
        probe.q4_factory_cost()
        probe.q5_score_semantics()
        probe.q6_latency_budget()
        probe.q7_navigation_cost()
    finally:
        probe.cleanup()


if __name__ == "__main__":
    main()
