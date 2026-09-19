"""Measure bounded progressive buildability overhead with 1,000 entities."""

import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

from benchmark_tensor_obs import TensorClient

ROOT = Path(__file__).resolve().parents[2]


def main():
    spec = importlib.util.spec_from_file_location(
        "isolated_fixture", ROOT / "tests/observation_diff/conftest.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    server = fixture.tiered_rcon.__wrapped__()
    rc = next(server)
    results = []
    try:
        count = rc.send_command(
            """/sc game.tick_paused=true local s=game.surfaces[1]
for _,e in pairs(s.find_entities_filtered{area={{-64,-64},{64,64}}}) do e.destroy{raise_destroy=true} end
local tiles={} for y=-64,63 do for x=-64,63 do tiles[#tiles+1]={name="grass-1",position={x,y}} end end s.set_tiles(tiles)
local n=0 for i=0,999 do local e=s.create_entity{name="iron-chest",position={-60+(i%40)*3+.5,-36+math.floor(i/40)*3+.5},force="player",raise_built=true} if e then n=n+1 end end rcon.print(n)"""
        )
        assert int(count) == 1000
        client = TensorClient()
        client.apply_entity(rc.send_command("/sc obs_diff_full_sync()"))
        client.apply_terrain(rc.send_command("/sc obs_terrain_full_sync()"))
        for budget in (0, 128, 256, 1024, 0):
            if budget:
                rc.send_command(
                    f"/sc obs_buildability_configure{{size=128,check_budget={budget},center={{x=0,y=0}}}}"
                )
            else:
                rc.send_command("/sc obs_buildability_configure{enabled=false}")
            timings, payloads = [], []
            for i in range(120):
                start = time.perf_counter()
                response = rc.send_command("/sc obs_all_drain()") or ""
                entities, _, terrain = response.partition("~")
                client.apply_entity(entities)
                client.apply_terrain(terrain)
                client.observation(include_buildability=True)
                elapsed = (time.perf_counter() - start) * 1000
                if i >= 20:
                    timings.append(elapsed)
                    payloads.append(len(response))
            row = {
                "check_budget": budget,
                "median_ms": statistics.median(timings),
                "p95_ms": sorted(timings)[94],
                "median_payload_bytes": statistics.median(payloads),
                "sampled_cells": int((client.buildability.values >= 0).sum()),
                "ms": timings,
            }
            results.append(row)
            print({k: v for k, v in row.items() if k != "ms"}, flush=True)
        output = (
            ROOT
            / "tests/benchmarks/results/progressive-buildability-128-2026-09-18.json"
        )
        output.write_text(
            json.dumps(
                {
                    "platform": platform.platform(),
                    "factorio": "2.0.73",
                    "entities": 1000,
                    "world_paused": True,
                    "size": 128,
                    "channels": 63,
                    "warmup_polls": 20,
                    "measured_polls": 100,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        server.close()


if __name__ == "__main__":
    main()
