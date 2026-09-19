"""Profile CPU rasterisation and owned float32 training samples (no model/GPU).

Run from the checkout root with PYTHONPATH=.; uses a disposable Factorio server.
RCON timings are separated from client work. No PyTorch dependency is required.
"""

import cProfile
import importlib.util
import io
import json
import platform
import pstats
import time
from pathlib import Path

import numpy as np

from benchmark_tensor_obs import TensorClient

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests/benchmarks/results/training-rasterisation-128-2026-09-18"
WARMUP, SAMPLES = 20, 100


def stats(samples):
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


def measure(fn, samples=SAMPLES):
    for _ in range(WARMUP):
        fn()
    timings = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = fn()
        timings.append((time.perf_counter_ns() - start) / 1e6)
        del result
    return stats(timings)


def snapshot(obs):
    """Owned arrays for a rollout buffer; keep coordinate/freshness metadata.

    Existing context is flattened float32; entities/globals/mask are float32.
    Buildability remains spatial but is cast to float32, preserving -1 unknown.
    No resizing mixes the 288-tile context with the 128-tile buildability view.
    """
    *base, build = obs
    return {
        "base": tuple(a.copy() for a in base),
        "buildability": build["values"].astype(np.float32),
        "sampled_ticks": build["sampled_ticks"].copy(),
        "origin": build["origin"],
        "tick": build["tick"],
        "surface": build["surface"],
        "force": build["force"],
        "generation": build["generation"],
        "manifest": build["manifest"],
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
        "processor": platform.processor(),
        "numpy": np.__version__,
        "factorio": "2.0.73",
        "world_paused": True,
        "buildability_size": 128,
        "check_budget": 256,
        "warmup": WARMUP,
        "samples": SAMPLES,
        "scope": "CPU NumPy; no torch conversion, GPU transfer, batching or model",
    }
    try:
        count = rc.send_command(
            """/sc game.tick_paused=true local s=game.surfaces[1]
for _,e in pairs(s.find_entities_filtered{area={{-64,-64},{64,64}}}) do e.destroy{raise_destroy=true} end
local tiles={} for y=-64,63 do for x=-64,63 do tiles[#tiles+1]={name="grass-1",position={x,y}} end end s.set_tiles(tiles)
local n=0 for i=0,999 do local e=s.create_entity{name="iron-chest",position={-60+(i%40)*3+.5,-36+math.floor(i/40)*3+.5},force="player",raise_built=true} if e then n=n+1 end end rcon.print(n)"""
        )
        assert int(count) == 1000
        entity_sync = rc.send_command("/sc obs_diff_full_sync()") or ""
        terrain_sync = rc.send_command("/sc obs_terrain_full_sync()") or ""
        client = TensorClient()
        client.apply_entity(entity_sync)
        client.apply_terrain(terrain_sync)
        rc.send_command(
            "/sc obs_buildability_configure{size=128,check_budget=256,center={x=0,y=0}}"
        )
        phases = {
            k: []
            for k in (
                "rcon",
                "decode_apply",
                "observation",
                "owned_float32_snapshot",
                "client_total",
                "total_with_rcon",
            )
        }
        for i in range(WARMUP + SAMPLES):
            t0 = time.perf_counter_ns()
            response = rc.send_command("/sc obs_all_drain()") or ""
            t1 = time.perf_counter_ns()
            entities, _, terrain = response.partition("~")
            client.apply_entity(entities)
            client.apply_terrain(terrain)
            t2 = time.perf_counter_ns()
            obs = client.observation(include_buildability=True)
            t3 = time.perf_counter_ns()
            owned = snapshot(obs)
            t4 = time.perf_counter_ns()
            if i >= WARMUP:
                for key, elapsed in zip(
                    phases,
                    (
                        t1 - t0,
                        t2 - t1,
                        t3 - t2,
                        t4 - t3,
                        t4 - t1,
                        t4 - t0,
                    ),
                ):
                    phases[key].append(elapsed / 1e6)
            del owned
        report["live_poll"] = {k: stats(v) for k, v in phases.items()}
        report["cached_entities"] = len(client.entities)
        report["terrain_counts"] = {
            "water_chunks": len(client.water),
            "ore_tiles": len(client.ores),
            "trees": len(client.trees),
            "obstacles": len(client.obstacles),
        }
        obs = client.observation(include_buildability=True)
        report["arrays"] = {
            name: {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes}
            for name, a in zip(
                (
                    "context",
                    "globals",
                    "entities",
                    "entity_mask",
                    "buildability",
                    "sampled_ticks",
                ),
                (*obs[:4], obs[4]["values"], obs[4]["sampled_ticks"]),
            )
        }
        report["sampled_buildability_cells"] = int((obs[4]["values"] >= 0).sum())
        report["total_buildability_cells"] = obs[4]["values"].size
        owned = snapshot(obs)
        assert owned["buildability"].dtype == np.float32
        np.testing.assert_array_equal(owned["buildability"], obs[4]["values"])
        for src, dst in zip(obs[:4], owned["base"]):
            np.testing.assert_array_equal(src, dst)
            assert not np.shares_memory(src, dst)
        assert not np.shares_memory(owned["buildability"], obs[4]["values"])
        report["owned_sample_bytes"] = (
            sum(a.nbytes for a in owned["base"])
            + owned["buildability"].nbytes
            + owned["sampled_ticks"].nbytes
        )
        del owned

        # CPU-only measurements over captured state, with no RCON inside timers.
        measurements = {
            "observation": measure(
                lambda: client.observation(include_buildability=True)
            ),
            "snapshot_existing_observation": measure(lambda: snapshot(obs)),
            "buildability_float32_cast": measure(
                lambda: obs[4]["values"].astype(np.float32)
            ),
            "fresh_legal_mask": measure(
                lambda: client.buildability.legal_mask(client.tick, 120)
            ),
            "observation_and_snapshot": measure(
                lambda: snapshot(client.observation(include_buildability=True))
            ),
            "full_grid_rebuild": measure(client._rebuild_grid),
            "full_grid_and_table_rebuild": measure(client.rebuild),
        }
        # Replay real captured entity rows as forced upserts. This measures update
        # count scaling, not a claim about real factory change frequency.
        rows = ["u" + key + "," + row for key, row in client.entities.items()]
        for n in (0, 1, 10, 100, 1000):
            assert len(rows) >= n
            delta = ";".join(rows[:n])
            measurements[f"apply_{n}_entity_upserts"] = measure(
                lambda: client.apply_entity(delta)
            )
            measurements[f"train_sample_{n}_entity_upserts"] = measure(
                lambda: (
                    client.apply_entity(delta),
                    snapshot(client.observation(include_buildability=True)),
                )
            )
        # One contiguous feature vector is optional; metadata must be retained
        # separately and categorical features still need policy-specific encoding.
        measurements["flat_float32_feature_vector"] = measure(
            lambda: np.concatenate(
                [*(a.reshape(-1) for a in obs[:4]), obs[4]["values"].reshape(-1)],
                dtype=np.float32,
            )
        )
        report["cpu_only"] = measurements
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(50):
            client.apply_entity(";".join(rows[:100]))
            snapshot(client.observation(include_buildability=True))
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(30)
        OUTPUT.with_suffix(".profile.txt").write_text(stream.getvalue().rstrip() + "\n")
        OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        for group in ("live_poll", "cpu_only"):
            print(group, flush=True)
            for name, row in report[group].items():
                print(
                    f"  {name}: median {row['median_ms']:.3f} ms; p95 {row['p95_ms']:.3f} ms",
                    flush=True,
                )
        print(f"Owned sample: {report['owned_sample_bytes']} bytes", flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
