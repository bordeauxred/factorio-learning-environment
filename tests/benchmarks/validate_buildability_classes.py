"""Adversarial live comparisons of proposed shared buildability channels."""

import argparse
import importlib.util
import json
from pathlib import Path

from buildability_classes import classes

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "tests/benchmarks/results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-type", choices=("manual", "ghost_revive", "new_only"), default="manual"
    )
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--fractional", action="store_true")
    args = parser.parse_args()
    _, groups = classes()
    spec = importlib.util.spec_from_file_location(
        "isolated_fixture", ROOT / "tests/observation_diff/conftest.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    server = fixture.tiered_rcon.__wrapped__()
    rc = next(server)
    results = []
    signatures = ["" for _ in groups]
    variants = [a for group in groups for a in group]
    variant_signatures = ["" for _ in variants]
    try:

        def lua(code):
            return rc.send_command("/sc " + code)

        encoded = (
            "{"
            + ",".join(
                "{"
                + ",".join("{" + json.dumps(n) + "," + str(d) + "}" for n, d in g)
                + "}"
                for g in groups
            )
            + "}"
        )
        lua(
            "game.tick_paused=true storage.mask_groups=" + encoded + " "
            'local s=game.create_surface("mask-parity",{width=64,height=64,autoplace_controls={}}) '
            "s.request_to_generate_chunks({0,0},2) s.force_generate_chunk_requests() "
            'function reset_mask_scene() local s=game.surfaces["mask-parity"] '
            "for _,e in pairs(s.find_entities()) do e.destroy() end "
            'local t={} for y=-20,20 do for x=-20,20 do t[#t+1]={name="grass-1",position={x,y}} end end '
            "s.set_tiles(t) end "
            'function test_mask_scene() local s=game.surfaces["mask-parity"] local out={mismatches={},masks={},variants={},checks=0} local vi=0 '
            "for gi,g in ipairs(storage.mask_groups) do local bits={} local allbits={} for ai=1,#g do allbits[ai]={} end "
            "for y=-4,4 do for x=-4,4 do local expected=nil "
            "for ai,a in ipairs(g) do local p=prototypes.entity[a[1]] local w,h=p.tile_width,p.tile_height "
            "if a[2]==4 or a[2]==12 then w,h=h,w end "
            'local v=s.can_place_entity{name=a[1],direction=a[2],force="player",'
            "position={x+(w%2)*.5,y+(h%2)*.5},build_check_type=defines.build_check_type."
            + ("manual" if args.check_type == "new_only" else args.check_type)
            + "} "
            + (
                'if v then v=s.can_place_entity{name=a[1],direction=a[2],force="player",position={x+(w%2)*.5,y+(h%2)*.5},build_check_type=defines.build_check_type.ghost_revive} end '
                if args.check_type == "new_only"
                else ""
            )
            + 'allbits[ai][#allbits[ai]+1]=v and "1" or "0" '
            "out.checks=out.checks+1 "
            'if expected==nil then expected=v bits[#bits+1]=v and "1" or "0" '
            "elseif expected~=v and #out.mismatches<20 then out.mismatches[#out.mismatches+1]={group=gi,entity=a,x=x,y=y,expected=expected,actual=v} end "
            "end end end out.masks[gi]=table.concat(bits) for ai=1,#g do vi=vi+1 out.variants[vi]=table.concat(allbits[ai]) end end "
            "rcon.print(helpers.table_to_json(out)) end"
        )

        def scene(label, code):
            lua('reset_mask_scene() local s=game.surfaces["mask-parity"] ' + code)
            r = json.loads(lua("test_mask_scene()"))
            for i, bits in enumerate(r.pop("masks")):
                signatures[i] += bits
            for i, bits in enumerate(r.pop("variants")):
                variant_signatures[i] += bits
            results.append({"scene": label, **r})
            if len(results) % 10 == 0:
                print("Scenes checked:", len(results), flush=True)

        scene("empty-ground", "")
        for resource in ("iron-ore", "uranium-ore", "crude-oil"):
            scene(
                resource,
                "s.create_entity{name="
                + json.dumps(resource)
                + ",position={.5,.5},amount=10000}",
            )
        for d in range(4):
            conditions = ["y<0", "x>0", "y>0", "x<0"]
            scene(
                "shore-" + str(d),
                "local t={} for y=-20,20 do for x=-20,20 do if "
                + conditions[d]
                + ' then t[#t+1]={name="water",position={x,y}} end end end s.set_tiles(t)',
            )
        scene(
            "all-water",
            'local t={} for y=-20,20 do for x=-20,20 do t[#t+1]={name="water",position={x,y}} end end s.set_tiles(t)',
        )
        for direction in (0, 4):
            scene(
                "rail-" + str(direction),
                'for i=-16,16,2 do s.create_entity{name="straight-rail",position={'
                + ("0,i" if direction == 0 else "i,0")
                + "},direction="
                + str(direction)
                + ',force="player"} end',
            )
        catalog = json.loads(
            (RESULTS / "buildability-prototypes-2.0.73.json").read_text()
        )
        for p in catalog:
            scene(
                "blocker-" + p["name"],
                "s.create_entity{name="
                + json.dumps(p["name"])
                + ',position={.5,.5},force="player"}',
            )
        for name in ("character", "small-biter", "tree-01", "rock-big"):
            scene(
                "natural-" + name,
                "s.create_entity{name="
                + json.dumps(name)
                + ',position={.5,.5},force="neutral"}',
            )
        if args.extended:
            for p in catalog:
                for direction in (0, 4, 8, 12):
                    w, h = p["tile_width"], p["tile_height"]
                    if direction in (4, 12):
                        w, h = h, w
                    scene(
                        f"aligned-{p['name']}-{direction}",
                        "s.create_entity{name="
                        + json.dumps(p["name"])
                        + ",direction="
                        + str(direction)
                        + ",position={"
                        + str(w % 2 * 0.5)
                        + ","
                        + str(h % 2 * 0.5)
                        + '},force="player"}',
                    )
            for name in (
                "pipe",
                "pipe-to-ground",
                "storage-tank",
                "pump",
                "offshore-pump",
                "boiler",
                "heat-exchanger",
                "steam-engine",
                "steam-turbine",
                "chemical-plant",
                "oil-refinery",
                "assembling-machine-2",
            ):
                for direction in (0, 4, 8, 12):
                    for fluid in ("water", "steam", "crude-oil"):
                        scene(
                            f"fluid-{name}-{direction}-{fluid}",
                            "local e=s.create_entity{name="
                            + json.dumps(name)
                            + ",direction="
                            + str(direction)
                            + ',position={.5,.5},force="player"} '
                            'if e then pcall(function() e.set_recipe("sulfuric-acid") end) '
                            "pcall(function() for i=1,#e.fluidbox do e.fluidbox[i]={name="
                            + json.dumps(fluid)
                            + ",amount=100,temperature=100} end end) end",
                        )
        # Sub-tile blockers distinguish almost equal collision boxes, e.g.
        # a pipe (0.2890625 half-width) and speaker (0.296875 half-width).
        radii = sorted(
            {
                abs(p["collision_box"][corner][axis])
                for p in catalog
                for corner in ("left_top", "right_bottom")
                for axis in ("x", "y")
            }
        )
        for a, b in zip(radii, radii[1:]):
            offset = 0.5 + 0.19921875 + (a + b) / 2
            for axis in (0, 1):
                pos = [offset, 0.5] if axis == 0 else [0.5, offset]
                scene(
                    f"fractional-character-{axis}-{a}-{b}",
                    's.create_entity{name="character",position={'
                    + ",".join(map(str, pos))
                    + '},force="player"}',
                )
                if args.fractional:
                    offset = 0.5 + 0.34765625 + (a + b) / 2
                    pos = [offset, 0.5] if axis == 0 else [0.5, offset]
                    scene(
                        f"fractional-chest-{axis}-{a}-{b}",
                        'local e=s.create_entity{name="iron-chest",position={10,10},force="player"} e.teleport({'
                        + ",".join(map(str, pos))
                        + "})",
                    )
        indistinguishable = []
        for i, a in enumerate(signatures):
            for j in range(i):
                if a == signatures[j]:
                    indistinguishable.append([j, i])
        report = {
            "channels": groups,
            "scenes": results,
            "check_type": args.check_type,
            "checks": sum(r["checks"] for r in results),
            "mismatch_count": sum(len(r["mismatches"]) for r in results),
            "indistinguishable_pairs": indistinguishable,
        }
        if args.fractional:
            report["rail_probe"] = json.loads(
                lua(
                    'reset_mask_scene() local s=game.surfaces["mask-parity"] local out={} '
                    'for i=-16,16,2 do s.create_entity{name="straight-rail",position={0,i},direction=0,force="player"} end '
                    'for _,name in ipairs({"locomotive","cargo-wagon","fluid-wagon"}) do '
                    "local r={name=name,manual=0,ghost_revive=0,examples={}} "
                    "for x=-12,12 do for y=-12,12 do for _,d in ipairs({0,4,8,12}) do "
                    'for _,mode in ipairs({"manual","ghost_revive"}) do '
                    'local ok=s.can_place_entity{name=name,position={x*.25,y*.25},direction=d,force="player",build_check_type=defines.build_check_type[mode]} '
                    "if ok then r[mode]=r[mode]+1 if #r.examples<3 then r.examples[#r.examples+1]={x*.25,y*.25,d,mode} end end "
                    "end end end end "
                    'local e=s.create_entity{name=name,position={0,0},direction=0,force="player"} r.create_succeeded=e~=nil if e then r.position=e.position e.destroy() end '
                    "out[#out+1]=r end rcon.print(helpers.table_to_json(out))"
                )
            )
        partitions = {}
        for variant, signature in zip(variants, variant_signatures):
            partitions.setdefault(signature, []).append(variant)
        report["observed_distinct_channels"] = list(partitions.values())
        entity_partition = {}
        lookup = {tuple(v): s for v, s in zip(variants, variant_signatures)}
        for name in sorted({v[0] for v in variants}):
            key = tuple(lookup[name, d] for d in (0, 4, 8, 12))
            entity_partition.setdefault(key, []).append(name)
        report["observed_entity_classes"] = list(entity_partition.values())
        (
            RESULTS
            / (
                "buildability-class-validation-"
                + args.check_type
                + ("-extended" if args.extended else "")
                + ("-fractional" if args.fractional else "")
                + "-2.0.73.json"
            )
        ).write_text(json.dumps(report, indent=2) + "\n")
        print(
            "Distinct empirical entity classes / directional channels:",
            len(entity_partition),
            len(partitions),
            flush=True,
        )
        print(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "scenes",
                    "channels",
                    "observed_distinct_channels",
                    "observed_entity_classes",
                )
            },
            flush=True,
        )
    finally:
        server.close()


if __name__ == "__main__":
    main()
