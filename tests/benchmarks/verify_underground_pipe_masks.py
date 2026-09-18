"""Find directional placement witnesses for underground pipes in Factorio 2.0.73.

All setup calls return checked JSON; blocker positions and fluid contents are
recorded so silent snapping/failed setup cannot masquerade as mask equivalence.
Uses an isolated container, never an existing FLE server.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "tests/benchmarks/results"
DIRECTIONS = (0, 4, 8, 12)


def main():
    spec = importlib.util.spec_from_file_location(
        "isolated_fixture", ROOT / "tests/observation_diff/conftest.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    server = fixture.tiered_rcon.__wrapped__()
    rc = next(server)
    cases = []
    try:

        def lua(code):
            response = rc.send_command(
                "/sc local ok,result=pcall(function() "
                + code
                + " end) rcon.print(helpers.table_to_json({ok=ok,result=result}))"
            )
            result = json.loads(response)
            assert result["ok"], result
            return result.get("result")

        metadata = lua(
            'game.tick_paused=true local s=game.create_surface("underground-mask",{width=64,height=64}) '
            "s.request_to_generate_chunks({0,0},2) s.force_generate_chunk_requests() "
            "for _,e in pairs(s.find_entities()) do e.destroy() end "
            "local tiles={} for y=-24,24 do for x=-24,24 do "
            'tiles[#tiles+1]={name="grass-1",position={x,y}} end end s.set_tiles(tiles) '
            'local p=prototypes.entity["pipe-to-ground"] local f={} '
            "for _,b in pairs(p.fluidbox_prototypes) do f[#f+1]={production_type=b.production_type,connections=b.pipe_connections} end "
            "return {box=p.collision_box,mask=p.collision_mask,fluidboxes=f}"
        )

        def case(name, setup, rotation=0):
            code = (
                'local s=game.surfaces["underground-mask"] '
                "for _,e in pairs(s.find_entities()) do assert(e.destroy()) end "
                "local entities={} "
                "local function pos(x,y) "
                + f"for i=1,{rotation} do x,y=-y,x end "
                + "return {x=x+.5,y=y+.5} end "
                "local function put(name,x,y,d,fluid) local p=pos(x,y) "
                "local e=s.create_entity{name=name,position=p,direction=((d or 0)+"
                + str(rotation * 4)
                + ')%16,force="player"} '
                'assert(e,"failed to create "..name) '
                'assert(math.abs(e.position.x-p.x)<.004 and math.abs(e.position.y-p.y)<.004,"position snapped for "..name) '
                "if fluid then e.fluidbox[1]={name=fluid,amount=50,temperature=100} "
                'assert(e.fluidbox[1] and e.fluidbox[1].name==fluid,"fluid assignment failed") end '
                "entities[#entities+1]=e return e end "
                + setup
                + " local out={entities={},directions={}} "
                "for _,e in ipairs(entities) do local fluids={} "
                "pcall(function() for i=1,#e.fluidbox do if e.fluidbox[i] then fluids[i]=e.fluidbox[i] end end end) "
                "out.entities[#out.entities+1]={name=e.name,position=e.position,direction=e.direction,box=e.bounding_box,fluids=fluids} end "
                'for _,d in ipairs({0,4,8,12}) do local q={name="pipe-to-ground",position={.5,.5},direction=d,force="player"} '
                "q.build_check_type=defines.build_check_type.manual local m=s.can_place_entity(q) "
                "q.build_check_type=defines.build_check_type.ghost_revive local g=s.can_place_entity(q) "
                "out.directions[#out.directions+1]={direction=d,manual=m,collision=g,new_only=m and g} end return out"
            )
            row = {"name": name, "rotation": rotation, **lua(code)}
            cases.append(row)
            values = [v["new_only"] for v in row["directions"]]
            if len(set(values)) > 1:
                print(name, rotation, values, flush=True)

        case("empty", "")
        case("occupied", 'put("iron-chest",0,0,0)')
        # A car has an off-grid position and collides with the pipe's car layer.
        # Its bounding box ends between the pipe's short and long face.
        for rotation in range(4):
            case("fractional-car", 'put("car",0,-1.25,0)', rotation)
        # Normal aligned construction: neighboring ordinary/underground pipes,
        # splitters and belts, with every orientation.
        for name in ("pipe", "pipe-to-ground", "transport-belt", "iron-chest"):
            for d in DIRECTIONS:
                for rotation in range(4):
                    case(
                        "adjacent-" + name + "-" + str(d),
                        f"put({json.dumps(name)},0,-1,{d})",
                        rotation,
                    )
        # A single new underground segment can join separate fluid systems via
        # its surface and underground ends. Sweep direction/range and contents.
        for distance in (2, 5, 10, 11):
            for other_d in DIRECTIONS:
                for pair in (
                    ("water", "water"),
                    ("water", "steam"),
                    ("water", "crude-oil"),
                ):
                    setup = (
                        f'put("pipe",0,-1,0,{json.dumps(pair[0])}) '
                        f'put("pipe-to-ground",0,{distance},{other_d},{json.dumps(pair[1])})'
                    )
                    for rotation in range(4):
                        case(
                            f"fluid-bridge-{distance}-{other_d}-{pair[0]}-{pair[1]}",
                            setup,
                            rotation,
                        )

        assert all(v["new_only"] for v in cases[0]["directions"])
        assert not any(v["new_only"] for v in cases[1]["directions"])
        fluid_cases = [r for r in cases if r["name"].startswith("fluid-bridge")]
        # These decisive witnesses leave the entire candidate tile empty,
        # satisfying even a whole-tile interpretation of occupied-site exclusion.
        for row in fluid_cases:
            for entity in row["entities"]:
                box = entity["box"]
                assert (
                    box["right_bottom"]["x"] <= 0
                    or box["left_top"]["x"] >= 1
                    or box["right_bottom"]["y"] <= 0
                    or box["left_top"]["y"] >= 1
                ), row
        for rotation in range(4):
            row = next(
                r
                for r in cases
                if r["name"] == "fluid-bridge-2-8-water-steam"
                and r["rotation"] == rotation
            )
            assert [v["new_only"] for v in row["directions"]] == [
                d != rotation for d in range(4)
            ]
        witnesses = []
        for a in range(4):
            for b in range(a):
                witness = next(
                    (
                        r
                        for r in fluid_cases
                        if r["directions"][a]["new_only"]
                        != r["directions"][b]["new_only"]
                    ),
                    None,
                )
                assert witness is not None, (a, b)
                witnesses.append(
                    {
                        "directions": [DIRECTIONS[b], DIRECTIONS[a]],
                        "case": (
                            None
                            if witness is None
                            else {
                                "name": witness["name"],
                                "rotation": witness["rotation"],
                            }
                        ),
                    }
                )
        report = {
            "factorio": "2.0.73",
            "predicate": "manual AND ghost_revive",
            "metadata": metadata,
            "cases": cases,
            "pairwise_witnesses": witnesses,
        }
        path = RESULTS / "underground-pipe-verification-2.0.73.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"cases": len(cases), "witnesses": witnesses}), flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
