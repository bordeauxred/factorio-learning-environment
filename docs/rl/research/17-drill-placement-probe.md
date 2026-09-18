# D12 burner-drill placement probe

## Result

The two excursion failures were not caused by snapping, water, the character, a hidden tree, or an untracked obstacle. They were caused by Factorio's **manual build check requiring a compatible resource under a mining drill**. The RL placement clamp modeled ore as harmless—which is correct for collision—but omitted this positive support requirement. It therefore admitted empty grass for a burner mining drill.

I used only RCON 27004. I constructed `FactorioInstance(address="localhost", tcp_port=27004, fast=True, all_technologies_researched=False, inventory={})`, reset with `reset_position=True`, supplied five drills with `_set_inventory`, and found an empty grass site centered at `(-7.5,-7.5)`. Verification was verbatim:

```text
OPEN_AREA_VERIFY={"position":{"x":-7.5,"y":-7.5},"entities":["character@-7.5,-7.5"],"grass":121,"total":121}
```

Thus `find_entities_filtered` found only the teleported character and `count_tiles_filtered` found 121/121 grass tiles in the 11x11 search area.

## Grid probe

I tested every `(dx,dy)` in `[-2,2]^2`, directions `0,4,8,12`, and both integer and half-integer supplied centers: 200 tool calls total. Every success would have been picked up; all 200 were refused. The eight verbatim 5x5 result matrices (`X` = refused, rows are `dy=-2..2`) were identical:

```text
GRID integer dir=0  rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID integer dir=4  rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID integer dir=8  rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID integer dir=12 rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID half dir=0     rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID half dir=4     rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID half dir=8     rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID half dir=12    rows_dy_-2_to_2=XXXXX/XXXXX/XXXXX/XXXXX/XXXXX
GRID_COUNTS={"attempts": 200, "ok": 0, "raw_manual_blocked_before_tool": 200, "refused": 200}
```

Every refusal was queried separately. All had no entities/resources in the collision area, all-grass tiles, default `surface.can_place_entity=true`, manual `surface.can_place_entity=false`, and manual still false after moving the character ten tiles away. Representative exact tool text was:

```text
Could not place burner-mining-drill at (-10.0, -10.0), Cannot place burner-mining-drill at x=-10 y=-10 - something is in the way or terrain is unplaceable
```

The two reported excursion sites reproduced exactly. At `(152,-342)`, direction 4, the collision tiles were `(151,-343),(151,-342),(152,-343),(152,-342)`, all `grass-1`; entities/resources were empty, `manual=false`, `default=true`, and `away_manual=false`. The other target `(158,-334)`, direction 8, likewise covered `(157,-335),(157,-334),(158,-335),(158,-334)`, all `grass-1`, with the same booleans. Their original errors were therefore accurate but nonspecific: nothing occupied the sites; each lacked ore.

After full terrain and entity sync, the verbatim summary was:

```text
WORLD_SYNC={"entities": 0, "entity_records": 2, "obstacles": 5706, "ores": 34399, "terrain_records": 166495, "trees": 115601, "water_chunks": 3724}
```

For both failed footprints `WorldClient` reported `entity_origins=[]`, `obstacles=[]`, `ores=[]`, `trees=[]`, and `water=[]`. This identifies the missing clamp fact precisely: not missing occupancy, but missing **required resource coverage**.

## Enforced rule

`client.py` forwards the supplied `Position` unchanged. `server.lua` measures Euclidean build reach to that position, checks inventory, calls `avoid_entity`, then calls the wrapper around `surface.can_place_entity` with `build_check_type=manual`. A raw manual check treats the character collision box as blocking, but `avoid_entity` teleports the character diagonally until it no longer blocks. On ore, raw results were `character_on_centre=false, character_away=true`; the tool nevertheless placed successfully and left the player displaced. The clamp should therefore not reject the player tile as terminally occupied.

For this 2x2 drill, the prototype collision box is `[-0.69921875,0.69921875]^2`, mining radius is `0.99`, and resource category is `basic-solid`. Manual placement requires at least one compatible ore center in that mining square. Water and entities sharing the drill's collision layers (buildings, trees/rocks/cliffs, character before evasion, and colliding ground items) block; ore does not collide. Direction does not change this square footprint.

The creation call uses the supplied position as its center but Factorio applies build-grid snapping: integer `(-887,-196)` remained exact, while supplied half-center `(-886.5,-195.5)` created at `(-886,-195)`. Both conventions and all four directions succeeded on iron ore. This also proves the tool does not require *empty* terrain and does require ore: the same tool refused all empty grass but accepted ore (`ORE_TOOL_RESULT` actual `[-887.0,-196.0]`).

## Fix and verification

`fle/rl/env.py` now requires compatible cached resources within the base-game mining areas (burner 2x2, electric 5x5, pumpjack 1x1) before admitting an offset. The existing nearest-offset search now moves an unsupported drill request to the nearest truly supported offset and continues logging `requested_offset`, `executed_offset`, and `clamp_reason="place_footprint_or_reach"`. `fle/rl/observation.py` no longer masks the player tile, matching `avoid_entity`. The new offline test proves a burner request on grass clamps onto iron ore; the mask test now proves player-tile support while retaining entity-footprint exclusion.

```text
51 passed, 35 warnings in 0.55s
All checks passed!  # ruff, changed Python files
RESTORE={"speed":40,"paused":false}
```
