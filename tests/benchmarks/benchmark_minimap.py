"""Isolated minimap, narrow-only and combined polling/training profiles."""

import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np

from benchmark_tensor_obs import TensorClient
from benchmark_training_rasterisation import measure, snapshot, stats
from fle.commons.observation.minimap import MinimapCache

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests/benchmarks/results/minimap-2026-09-18.json"


def owned_minimap(observation):
    return {
        **observation,
        "values": observation["values"].copy(),
        "sampled_ticks": observation["sampled_ticks"].copy(),
    }


def main():
    spec = importlib.util.spec_from_file_location(
        "isolated_fixture", ROOT / "tests/observation_diff/conftest.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    server = fixture.tiered_rcon.__wrapped__()
    rc = next(server)
    report = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "factorio": "2.0.73",
        "structures": 1000,
        "characters": 1,
        "minimap_shape": [14, 128, 128],
        "stride": 4,
        "world_tiles": 512,
        "minimap_chunk_budget": 4,
        "buildability_check_budget": 256,
        "warmup": 20,
        "samples": 100,
        "scope": "CPU NumPy; no model/GPU",
        "visibility": "generated",
        "cases": {},
    }
    try:
        result = rc.send_command(
            """/sc game.tick_paused=true local s=game.surfaces[1]
s.request_to_generate_chunks({0,0},9) s.force_generate_chunk_requests()
for _,e in pairs(s.find_entities_filtered{area={{-64,-64},{64,64}}}) do e.destroy{raise_destroy=true} end
local tiles={} for y=-64,63 do for x=-64,63 do tiles[#tiles+1]={name="grass-1",position={x,y}} end end s.set_tiles(tiles)
local n=0 for i=0,999 do local e=s.create_entity{name="iron-chest",position={-60+(i%40)*3+.5,-36+math.floor(i/40)*3+.5},force="player",raise_built=true} if e then n=n+1 end end
storage.obs_char=s.create_entity{name="character",position={.5,.5},force="player"}
assert(n==1000 and storage.obs_char) game.forces.player.chart(s,{{-256,-256},{255,255}}) rcon.print("ready")"""
        )
        assert result.strip() == "ready", result
        # Generated visibility matches the existing FLE terrain stream.
        for mode in ("minimap_only", "narrow_only", "combined", "minimap_only_repeat"):
            mini = mode != "narrow_only"
            narrow = mode in ("narrow_only", "combined")
            rc.send_command(
                "/sc obs_buildability_configure{enabled=false} obs_minimap_configure{enabled=false}"
            )
            client = TensorClient() if narrow else MinimapCache()
            cache = client.minimap if narrow else client
            if narrow:
                client.apply_entity(rc.send_command("/sc obs_diff_full_sync()") or "")
                client.apply_terrain(
                    rc.send_command("/sc obs_terrain_full_sync()") or ""
                )
            case = {}
            if mini:
                cache.configure(rc, visibility="generated")
                start = time.perf_counter_ns()
                fill_latencies = []
                for i in range(200):
                    t0 = time.perf_counter_ns()
                    cache.apply(rc.send_command("/sc obs_minimap_drain()") or "")
                    fill_latencies.append((time.perf_counter_ns() - t0) / 1e6)
                    if cache.enabled and np.all(cache.sampled_ticks >= 0):
                        break
                assert np.all(cache.sampled_ticks >= 0)
                assert np.all(cache.values[0] == 1)
                case["initial_fill"] = {
                    "polls": i + 1,
                    "elapsed_ms": (time.perf_counter_ns() - start) / 1e6,
                    **stats(fill_latencies),
                }
                assert cache.values.shape == (14, 128, 128)
                assert cache.values[13].sum() == 1
            if narrow:
                client.buildability.configure(rc)
            phases = {
                k: []
                for k in (
                    "rcon",
                    "decode_apply",
                    "observation",
                    "owned_snapshot",
                    "client_total",
                    "total",
                )
            }
            payloads = []
            for i in range(120):
                t0 = time.perf_counter_ns()
                response = (
                    rc.send_command(
                        "/sc obs_all_drain()" if narrow else "/sc obs_minimap_drain()"
                    )
                    or ""
                )
                t1 = time.perf_counter_ns()
                if narrow:
                    entities, _, terrain = response.partition("~")
                    client.apply_entity(entities)
                    client.apply_terrain(terrain)
                else:
                    cache.apply(response)
                t2 = time.perf_counter_ns()
                observation = (
                    client.observation(include_buildability=True) if narrow else None
                )
                minimap = cache.observation() if mini else None
                t3 = time.perf_counter_ns()
                owned = snapshot(observation) if narrow else None
                owned_map = owned_minimap(minimap) if mini else None
                t4 = time.perf_counter_ns()
                if i >= 20:
                    for k, dt in zip(
                        phases, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t1, t4 - t0)
                    ):
                        phases[k].append(dt / 1e6)
                    payloads.append(len(response.encode()))
                del owned, owned_map
            case["steady_poll"] = {k: stats(v) for k, v in phases.items()}
            case["median_payload_bytes"] = float(np.median(payloads))
            if mini:
                case["minimap_copy"] = measure(
                    lambda: owned_minimap(cache.observation())
                )
                case["minimap_sample_bytes"] = (
                    cache.values.nbytes + cache.sampled_ticks.nbytes
                )
            if narrow:
                case["narrow_sample"] = measure(
                    lambda: snapshot(client.observation(include_buildability=True))
                )
                case["cached_entities"] = len(client.entities)
            if mini and narrow:
                case["combined_sample"] = measure(
                    lambda: (
                        snapshot(client.observation(include_buildability=True)),
                        owned_minimap(cache.observation()),
                    )
                )
            report["cases"][mode] = case
            print(
                mode,
                {k: round(v["median_ms"], 3) for k, v in case["steady_poll"].items()},
                flush=True,
            )
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        server.close()


if __name__ == "__main__":
    main()
