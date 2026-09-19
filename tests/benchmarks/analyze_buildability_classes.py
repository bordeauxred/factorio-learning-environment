"""Export version-matched placement metadata from an isolated Factorio server."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "tests/benchmarks/results"
FIELDS = """collision_box secondary_collision_box collision_mask
collision_mask_collides_with_self collision_mask_collides_with_tiles_only
collision_mask_considers_tile_transitions tile_width tile_height
building_grid_bit_shift flags tile_buildability_rules mining_drill_radius
resource_categories fluid_source_offset vector_to_place_result
fast_replaceable_group is_building rotation_snap_angle""".split()


def main():
    catalog = json.loads((RESULTS / "buildability-2026-09-18.json").read_text())[
        "catalog"
    ]
    spec = importlib.util.spec_from_file_location(
        "isolated_fixture", ROOT / "tests/observation_diff/conftest.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    server = fixture.tiered_rcon.__wrapped__()
    rc = next(server)
    try:
        names = "{" + ",".join(json.dumps(p["name"]) for p in catalog) + "}"
        fields = "{" + ",".join(json.dumps(f) for f in FIELDS) + "}"
        code = (
            "local out={} for _,name in ipairs(" + names + ") do "
            "local p=prototypes.entity[name] local row={name=name,type=p.type,errors={}} "
            "for _,key in ipairs(" + fields + ") do "
            "local ok,value=pcall(function() return p[key] end) "
            "if ok then row[key]=value else row.errors[key]=tostring(value) end end "
            "out[#out+1]=row end rcon.print(helpers.table_to_json(out))"
        )
        rows = json.loads(rc.send_command("/sc " + code))
        (RESULTS / "buildability-prototypes-2.0.73.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n"
        )
        print(f"Exported {len(rows)} prototypes", flush=True)
        collision_catalog = rc.send_command(
            "/sc local out={entities={},tiles={}} "
            "for name,p in pairs(prototypes.entity) do out.entities[name]={mask=p.collision_mask,box=p.collision_box} end "
            "for name,p in pairs(prototypes.tile) do out.tiles[name]=p.collision_mask end "
            "rcon.print(helpers.table_to_json(out))"
        )
        (RESULTS / "buildability-collision-catalog-2.0.73.json").write_text(
            json.dumps(json.loads(collision_catalog), indent=2, sort_keys=True) + "\n"
        )
    finally:
        server.close()


if __name__ == "__main__":
    main()
