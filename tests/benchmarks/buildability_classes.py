"""Candidate ordinary-build equivalence classes for the frozen base-game catalog.

This is analysis code, not a production placement validator. Grouping excludes
inventory, reach, fast replacement, ghosts, mods and non-cardinal directions.
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def rotate_box(box, direction):
    if not box:
        return None
    points = [
        (box[a]["x"], box[b]["y"])
        for a in ("left_top", "right_bottom")
        for b in ("left_top", "right_bottom")
    ]
    for _ in range(direction // 4):
        points = [(-y, x) for x, y in points]
    return [
        min(x for x, y in points),
        min(y for x, y in points),
        max(x for x, y in points),
        max(y for x, y in points),
    ]


def classes():
    rows = json.loads((RESULTS / "buildability-prototypes-2.0.73.json").read_text())
    catalog = json.loads(
        (RESULTS / "buildability-collision-catalog-2.0.73.json").read_text()
    )
    obstacles = [p["mask"] for p in catalog["entities"].values()] + list(
        catalog["tiles"].values()
    )
    groups = defaultdict(list)
    entity_groups = defaultdict(list)
    for p in rows:
        layers = set(p["collision_mask"]["layers"])
        # Quotient only redundant layer names: retain the entire pattern of
        # intersections against every loaded entity/tile mask, plus connector
        # options. This is stronger than grouping only by footprint size.
        collision = [bool(layers.intersection(q.get("layers", {}))) for q in obstacles]
        options = {k: v for k, v in p["collision_mask"].items() if k != "layers"}
        for d in (0, 4, 8, 12):
            signature = dict(
                collision=collision,
                options=options,
                box=rotate_box(p["collision_box"], d),
                secondary=rotate_box(p.get("secondary_collision_box"), d),
                grid=p["building_grid_bit_shift"],
                offgrid=p["flags"].get("placeable-off-grid", False),
                radius=p.get("mining_drill_radius"),
                resources=p.get("resource_categories"),
            )
            w, h = p["tile_width"], p["tile_height"]
            signature["alignment"] = [h % 2, w % 2] if d in (4, 12) else [w % 2, h % 2]
            rules = p.get("tile_buildability_rules")
            if rules:
                signature["tiles"] = [
                    {**r, "area": rotate_box(r["area"], d)} for r in rules
                ]
            kind = p["type"]
            if kind == "boiler":
                # Input-water/output-steam connection restrictions distinguish
                # opposite directions despite the symmetric collision box.
                signature["special"] = ["boiler", d]
            elif kind in ("rail-signal", "rail-chain-signal", "train-stop"):
                signature["special"] = ["signal" if "signal" in kind else kind, d]
            elif kind in (
                "gate",
                "straight-rail",
                "locomotive",
                "cargo-wagon",
                "fluid-wagon",
            ):
                signature["special"] = [
                    (
                        "rolling-stock"
                        if "wagon" in kind or kind == "locomotive"
                        else kind
                    ),
                    d % 8,
                ]
            elif kind == "spider-vehicle":
                signature["special"] = kind
            key = json.dumps(signature, sort_keys=True)
            groups[key].append([p["name"], d])
            if d == 0:
                entity_groups[key].append(p["name"])
    return list(entity_groups.values()), list(groups.values())


def main():
    entities, channels = classes()
    report = {
        "entity_class_count": len(entities),
        "channel_count": len(channels),
        "entity_classes": entities,
        "channels": channels,
    }
    (RESULTS / "buildability-classes-2.0.73.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for i, members in enumerate(entities):
        channel_ids = [
            j
            for j, group in enumerate(channels)
            if any(n == members[0] for n, d in group)
        ]
        print(i + 1, len(channel_ids), ", ".join(members))
    print(f"{len(entities)} entity classes; {len(channels)} directional channels")


if __name__ == "__main__":
    main()
