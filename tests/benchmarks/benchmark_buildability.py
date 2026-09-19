"""Measure engine-authoritative spatial masks on an isolated Factorio server.

Run with the repository Python environment. Results include raw timings and the
resolved FLE build-item catalog. No existing Factorio server is touched.
These are manual-build spatial checks, excluding inventory and player reach.
"""

import argparse
import ast
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests/benchmarks/results/buildability-2026-09-18.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        help="Measure selected-entity and full-catalog reach windows",
    )
    args = parser.parse_args()
    output = (
        OUTPUT.with_name("buildability-local-2026-09-18.json") if args.local else OUTPUT
    )
    tree = ast.parse((ROOT / "fle/env/game_types.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Prototype"
    )
    names = sorted(
        {
            n.value.elts[0].value
            for n in cls.body
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Tuple)
            and isinstance(n.value.elts[0], ast.Constant)
        }
    )
    spec = importlib.util.spec_from_file_location(
        "isolated_fixture", ROOT / "tests/observation_diff/conftest.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    server = fixture.tiered_rcon.__wrapped__()
    rc = next(server)
    result = {
        "date": "2026-09-18",
        "host": platform.platform(),
        "factorio": "2.0.73",
        "speed": 10,
        "samples": [],
    }
    try:

        def lua(code):
            return rc.send_command("/sc " + code)

        # Resolve item -> place_result: excludes resources, ammunition and
        # virtual groups, and correctly resolves the rail item to straight-rail.
        catalog = lua(
            "local out={} for _,name in ipairs("
            + "{"
            + ",".join(json.dumps(n) for n in names)
            + "}"
            + ") do local i=prototypes.item[name] if i and i.place_result then "
            "local p=i.place_result out[#out+1]={item=name,name=p.name,type=p.type,"
            "width=p.tile_width,height=p.tile_height} end end "
            "rcon.print(helpers.table_to_json(out))"
        )
        result["catalog"] = json.loads(catalog)
        print("Buildable catalog:", len(result["catalog"]), flush=True)
        lua(
            "local s=game.surfaces[1] s.request_to_generate_chunks({0,0},6) s.force_generate_chunk_requests() "
            "game.forces.player.research_all_technologies() game.speed=10"
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            start = time.perf_counter()
            lua("rcon.print(1)")
            if time.perf_counter() - start < 0.005:
                break
            time.sleep(1)
        # Frozen world avoids changing masks between timing samples. All RCON
        # timings still include server scheduling; compare paired drain baseline.
        lua("game.tick_paused=true obs_diff_full_sync()")
        lua("obs_terrain_full_sync()")
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_tensor_obs import TensorClient

        client = TensorClient()
        client.apply_entity(lua("obs_diff_full_sync()") or "")
        client.apply_terrain(lua("obs_terrain_full_sync()") or "")

        def consume(resp):
            entities, _, terrain = resp.partition("~")
            client.apply_entity(entities)
            client.apply_terrain(terrain)
            client.observation()

        # One channel per tested (entity,direction); anchors use tile dimensions
        # to select the normal integer/half-integer placement lattice.
        lua(
            "function bench_build_mask(n,step,specs) local s=game.surfaces[1] local out={} "
            "for _,a in ipairs(specs) do local p=prototypes.entity[a[1]] "
            "local w,h=p.tile_width,p.tile_height if a[2]==4 or a[2]==12 then w,h=h,w end "
            "local ox,oy=(w%2)*0.5,(h%2)*0.5 "
            "for y=0,n-1 do local row={} for x=0,n-1 do "
            "row[#row+1]=s.can_place_entity{name=a[1],direction=a[2],"
            "position={x=x*step-math.floor(n*step/2)+ox,y=y*step-math.floor(n*step/2)+oy},"
            'force="player",build_check_type=defines.build_check_type.manual} and "1" or "0" '
            "end out[#out+1]=table.concat(row) end end return table.concat(out) end"
        )
        specs = [
            ["iron-chest", 0],
            ["burner-mining-drill", 0],
            ["electric-mining-drill", 0],
            ["pumpjack", 0],
        ] + [["offshore-pump", d] for d in (0, 4, 8, 12)]
        result["representative_channels"] = specs
        if args.local:
            specs = [[p["name"], d] for p in result["catalog"] for d in (0, 4, 8, 12)]
            result["representative_channels"] = specs
            cases = [(21, 1, 4), (21, 1, len(specs))]
        else:
            cases = [
                (21, 1, 1),
                (21, 1, 8),
                (96, 3, 1),
                (96, 3, 8),
                (96, 1, 8),
                (288, 1, 1),
                (288, 1, 8),
            ]
        for size, step, channels in cases:
            spec_lua = (
                "{"
                + ",".join(
                    "{" + json.dumps(n) + "," + str(d) + "}"
                    for n, d in specs[:channels]
                )
                + "}"
            )
            command = f'local masks=bench_build_mask({size},{step},{spec_lua}) obs_all_drain() rcon.print("|"..masks)'
            samples, baselines = [], []
            for repeat in range(7):
                start = time.perf_counter()
                consume(lua("obs_all_drain()") or "")
                baseline = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                response = lua(command)
                obs, _, masks = response.partition("|")
                consume(obs.strip())
                tensor = (
                    np.frombuffer(masks.strip().encode(), dtype=np.uint8) == ord("1")
                ).astype(np.float32)
                assert tensor.size == size * size * channels
                elapsed = (time.perf_counter() - start) * 1000
                if repeat:
                    samples.append(elapsed)
                    baselines.append(baseline)
            row = dict(
                size=size,
                tile_step=step,
                channels=channels,
                checks=size * size * channels,
                ms=samples,
                baseline_ms=baselines,
                median_ms=statistics.median(samples),
                baseline_median_ms=statistics.median(baselines),
            )
            result["samples"].append(row)
            print(json.dumps(row), flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        server.close()


if __name__ == "__main__":
    main()
