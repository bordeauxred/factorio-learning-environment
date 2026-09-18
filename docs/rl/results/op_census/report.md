# Operation census

The `seeded` fixture exists only to measure entity operations and is never a training start.

## regime=macro, fixture=empty

| op | n | ok | no_effect | no_support | tool_rejected | error |
|---|---:|---:|---:|---:|---:|---:|
| WAIT | 61 | 61 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| MOVE | 74 | 71 (95.9%) | 3 (4.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| HARVEST | 62 | 61 (98.4%) | 0 (0.0%) | 0 (0.0%) | 1 (1.6%) | 0 (0.0%) |
| CRAFT | 71 | 4 (5.6%) | 0 (0.0%) | 0 (0.0%) | 67 (94.4%) | 0 (0.0%) |
| PLACE | 62 | 8 (12.9%) | 0 (0.0%) | 53 (85.5%) | 1 (1.6%) | 0 (0.0%) |
| PICKUP | 58 | 0 (0.0%) | 5 (8.6%) | 53 (91.4%) | 0 (0.0%) | 0 (0.0%) |
| ROTATE | 56 | 0 (0.0%) | 0 (0.0%) | 47 (83.9%) | 9 (16.1%) | 0 (0.0%) |
| INSERT | 66 | 7 (10.6%) | 0 (0.0%) | 58 (87.9%) | 0 (0.0%) | 1 (1.5%) |
| EXTRACT | 68 | 2 (2.9%) | 0 (0.0%) | 66 (97.1%) | 0 (0.0%) | 0 (0.0%) |
| SET_RECIPE | 63 | 0 (0.0%) | 0 (0.0%) | 63 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| CONNECT | 68 | 0 (0.0%) | 0 (0.0%) | 68 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| RESEARCH | 59 | 49 (83.1%) | 0 (0.0%) | 0 (0.0%) | 10 (16.9%) | 0 (0.0%) |

| op | median wall s | p90 wall s | median game ticks | wall share |
|---|---:|---:|---:|---:|
| WAIT | 0.1743 | 0.2031 | 76 | 9.7% |
| MOVE | 0.7168 | 0.9971 | 371 | 26.3% |
| HARVEST | 2.0599 | 2.9820 | 1099 | 58.2% |
| CRAFT | 0.0619 | 0.0891 | 15 | 2.5% |
| PLACE | 0.0000 | 0.0455 | 0 | 0.2% |
| PICKUP | 0.0000 | 0.0000 | 0 | 0.3% |
| ROTATE | 0.0000 | 0.1093 | 0 | 0.5% |
| INSERT | 0.0000 | 0.1119 | 0 | 0.5% |
| EXTRACT | 0.0000 | 0.0000 | 0 | 0.1% |
| SET_RECIPE | 0.0000 | 0.0000 | 0 | 0.0% |
| CONNECT | 0.0000 | 0.0000 | 0 | 0.0% |
| RESEARCH | 0.0599 | 0.0788 | 13 | 1.7% |

### HARVEST headline

HARVEST ok: **61**; items acquired: **637**.

First acquired-item step: b5e760b7/episode 0: 10; b5e760b7/episode 1: 6; b5e760b7/episode 2: 37; b5e760b7/episode 3: 24; b5e760b7/episode 4: 5; b5e760b7/episode 5: 4; b5e760b7/episode 6: 6; b5e760b7/episode 7: 2; b5e760b7/episode 8: 20; b5e760b7/episode 9: 1; b5e760b7/episode 10: 10; b5e760b7/episode 11: 1

### Failure reasons

- WAIT: none

- MOVE: predicate_false=3 — `predicate_false`

- HARVEST: other=1 — `RCON JSON read returned invalid JSON: Cannot execute command. Error: LuaEntity API call when LuaEntity was invalid.`

  Top `other` messages: 1× `RCON JSON read returned invalid JSON: Cannot execute command. Error: LuaEntity API call when LuaEntity was invalid.`

- CRAFT: missing_ingredients=61 — `Failed to craft 20x burner-mining-drill because couldn't craft a required sub-ingredient for burner-mining-drill - iron-plate - Item iron-plate cannot be crafted (category: smelting). Recipe requires `; other=6 — `attempt to index field '?' (a nil value)`

  Top `other` messages: 6× `attempt to index field '?' (a nil value)`

- PLACE: no_placeable_in_inventory=53 — `no_placeable_in_inventory`; out_of_reach=1 — `Could not place stone-furnace at (52.0, -60.0), The target position is too far away to place the entity. The player position is 60.5, -54.5 and the target position is 52, -60. The distance is 10.12 an`

- PICKUP: no_entity=53 — `no_entity`; predicate_false=5 — `predicate_false`

- ROTATE: no_entity=47 — `no_entity`; other=9 — `Could not rotate stone-furnace.`

  Top `other` messages: 5× `Could not rotate stone-furnace.`; 4× `Could not rotate wooden-chest.`

- INSERT: no_entity=58 — `no_entity`; anchor_resolution_failed=1 — `anchor_resolution_failed: u994 wooden-chest at (8.5, 131.5)`

- EXTRACT: no_entity=64 — `no_entity`; entity_has_no_items=2 — `entity_has_no_items`

- SET_RECIPE: no_assembling_machine=63 — `no_assembling_machine`

- CONNECT: fewer_than_two_entities=66 — `fewer_than_two_entities`; no_connector_in_inventory=2 — `no_connector_in_inventory`

- RESEARCH: other=10 — `Failed to start research for steam-power`

  Top `other` messages: 6× `Failed to start research for electronics`; 4× `Failed to start research for steam-power`


## regime=macro, fixture=seeded

| op | n | ok | no_effect | no_support | tool_rejected | error |
|---|---:|---:|---:|---:|---:|---:|
| WAIT | 24 | 24 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| MOVE | 17 | 17 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| HARVEST | 20 | 20 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| CRAFT | 27 | 10 (37.0%) | 0 (0.0%) | 0 (0.0%) | 17 (63.0%) | 0 (0.0%) |
| PLACE | 24 | 20 (83.3%) | 1 (4.2%) | 0 (0.0%) | 3 (12.5%) | 0 (0.0%) |
| PICKUP | 22 | 0 (0.0%) | 21 (95.5%) | 0 (0.0%) | 0 (0.0%) | 1 (4.5%) |
| ROTATE | 23 | 12 (52.2%) | 2 (8.7%) | 0 (0.0%) | 9 (39.1%) | 0 (0.0%) |
| INSERT | 17 | 9 (52.9%) | 0 (0.0%) | 0 (0.0%) | 8 (47.1%) | 0 (0.0%) |
| EXTRACT | 28 | 1 (3.6%) | 0 (0.0%) | 27 (96.4%) | 0 (0.0%) | 0 (0.0%) |
| SET_RECIPE | 17 | 0 (0.0%) | 3 (17.6%) | 10 (58.8%) | 4 (23.5%) | 0 (0.0%) |
| CONNECT | 25 | 1 (4.0%) | 2 (8.0%) | 0 (0.0%) | 21 (84.0%) | 1 (4.0%) |
| RESEARCH | 12 | 11 (91.7%) | 0 (0.0%) | 0 (0.0%) | 1 (8.3%) | 0 (0.0%) |

| op | median wall s | p90 wall s | median game ticks | wall share |
|---|---:|---:|---:|---:|
| WAIT | 0.6088 | 6.0493 | 311 | 34.0% |
| MOVE | 0.6470 | 1.1261 | 348 | 9.7% |
| HARVEST | 1.8484 | 3.3811 | 985.5 | 33.9% |
| CRAFT | 0.0781 | 0.3304 | 20 | 3.2% |
| PLACE | 0.0647 | 0.1135 | 16 | 1.4% |
| PICKUP | 0.1180 | 0.1369 | 45.5 | 2.3% |
| ROTATE | 0.1186 | 0.1332 | 46 | 2.3% |
| INSERT | 0.1145 | 0.1318 | 44 | 1.6% |
| EXTRACT | 0.0000 | 0.0000 | 0 | 0.1% |
| SET_RECIPE | 0.0000 | 0.1419 | 0 | 0.8% |
| CONNECT | 0.4012 | 0.7706 | 201 | 9.9% |
| RESEARCH | 0.0619 | 0.1215 | 13.5 | 0.8% |

### HARVEST headline

HARVEST ok: **20**; items acquired: **202**.

First acquired-item step: db574ef9/episode 0: 18; db574ef9/episode 1: 32; db574ef9/episode 2: 0; db574ef9/episode 3: 9

### Failure reasons

- WAIT: none

- MOVE: none

- HARVEST: none

- CRAFT: missing_ingredients=17 — `Failed to craft 5x wooden-chest because couldn't craft a required sub-ingredient for wooden-chest - wood - recipe for wood doesn't exist, it is a raw resource that must be gathered first. Required woo`

- PLACE: cannot_place=2 — `Could not place transport-belt at (-3.5, -0.5), assembling-machine-1 at x=-3.5 y=0.5`; out_of_reach=1 — `Could not place burner-inserter at (-7.5, -7.5), The target position is too far away to place the entity. The player position is 0, 0 and the target position is -7.5, -7.5. The distance is 10.61 and t`; predicate_false=1 — `predicate_false`

- PICKUP: predicate_false=21 — `predicate_false`; anchor_resolution_failed=1 — `anchor_resolution_failed: u1042 wooden-chest at (3.5, 3.5)`

- ROTATE: other=9 — `Could not rotate stone-furnace.`; predicate_false=2 — `predicate_false`

  Top `other` messages: 5× `Could not rotate pipe.`; 2× `Could not rotate stone-furnace.`; 1× `Could not rotate wooden-chest.`; 1× `Could not rotate assembling-machine-1. Set the recipe first.`

- INSERT: no_suitable_slot=4 — `Could not insert: "Failed to insert transport-belt into pipe (type pipe) at position {x = -22.5, y = -105.5}. Attempted to insert 5 items. Entity might not accept this item or has no available space. `; other=4 — `Could not insert: "Could not find a nearby entity that can accept stone"`

  Top `other` messages: 1× `Could not insert: "Could not find a nearby entity that can accept stone"`; 1× `Could not insert: "Could not find a nearby entity that can accept assembling-machine-1"`; 1× `Could not insert: "Could not find a nearby entity that can accept pipe"`; 1× `Could not insert: LuaItemStack API call when LuaItemStack was invalid for read.`

- EXTRACT: entity_has_no_items=27 — `entity_has_no_items`

- SET_RECIPE: no_assembling_machine=10 — `no_assembling_machine`; other=4 — `Invalid entity type: parameter-9`; predicate_false=3 — `predicate_false`

  Top `other` messages: 2× `Invalid entity type: parameter-2`; 1× `Invalid entity type: parameter-9`; 1× `Invalid entity type: parameter-0`

- CONNECT: other=17 — `'BeltGroup' object has no attribute 'x'`; predicate_false=2 — `predicate_false`; not_in_inventory=2 — `Failed to connect {'SmallElectricPole'} from wooden-chest at x=72.5 y=16.5 to stone-furnace at x=3.0 y=0.0. You do not have enough small-electric-pole in you inventory to complete this connection. Req`; no_path=1 — `Failed to connect {'Pipe'} from burner-mining-drill at x=-16.0 y=-51.0 to burner-inserter at x=3.5 y=-1.5. Invalid path: nil`; connect_failed=1 — `Cannot connect to source inserter drop_position position x=3.5 y=-0.5 as it is already occupied by incompatible entities - ['stone-furnace at x=3.0 y=0.0'].`; anchor_resolution_failed=1 — `anchor_resolution_failed: u1093 wooden-chest at (3.5, 3.5)`

  Top `other` messages: 7× `'BeltGroup' object has no attribute 'x'`; 3× `<class 'fle.env.entities.BeltGroup'> is not a supported target object for fluid connection`; 2× `<class 'fle.env.entities.BeltGroup'> is not a supported source object for fluid connection`; 2× `'PipeGroup' object has no attribute 'x'`; 1× `Transport belts cannot be connected directly from a Prototype.StoneFurnace object as a source. You need to add an inserter that takes items from Prototype.StoneFurnace and use the inserter as a source`; 1× `Transport belts cannot be connected directly to a Prototype.StoneFurnace object as a target. You need to add an inserter that inputs items into Prototype.StoneFurnace and use the inserter as the targe`; 1× `Transport belts cannot be connected directly to a Prototype.WoodenChest object as a target. You need to add an inserter that inputs items into Prototype.WoodenChest and use the inserter as the target `

- RESEARCH: other=1 — `Failed to start research for electronics`

  Top `other` messages: 1× `Failed to start research for electronics`


## regime=naive, fixture=empty

| op | n | ok | no_effect | no_support | tool_rejected | error |
|---|---:|---:|---:|---:|---:|---:|
| WAIT | 59 | 59 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| MOVE | 74 | 74 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| HARVEST | 56 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 56 (100.0%) | 0 (0.0%) |
| CRAFT | 66 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 66 (100.0%) | 0 (0.0%) |
| PLACE | 58 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 58 (100.0%) | 0 (0.0%) |
| PICKUP | 64 | 0 (0.0%) | 0 (0.0%) | 64 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| ROTATE | 53 | 0 (0.0%) | 0 (0.0%) | 53 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| INSERT | 67 | 0 (0.0%) | 0 (0.0%) | 67 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| EXTRACT | 77 | 0 (0.0%) | 0 (0.0%) | 77 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| SET_RECIPE | 78 | 0 (0.0%) | 0 (0.0%) | 78 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| CONNECT | 61 | 0 (0.0%) | 0 (0.0%) | 61 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| RESEARCH | 55 | 1 (1.8%) | 0 (0.0%) | 0 (0.0%) | 54 (98.2%) | 0 (0.0%) |

| op | median wall s | p90 wall s | median game ticks | wall share |
|---|---:|---:|---:|---:|
| WAIT | 1.5627 | 6.0721 | 823 | 85.5% |
| MOVE | 0.1552 | 0.2018 | 69.5 | 6.6% |
| HARVEST | 0.0576 | 0.0786 | 15 | 2.0% |
| CRAFT | 0.0550 | 0.0725 | 15 | 2.2% |
| PLACE | 0.0502 | 0.0690 | 12.5 | 1.8% |
| PICKUP | 0.0000 | 0.0000 | 0 | 0.0% |
| ROTATE | 0.0000 | 0.0000 | 0 | 0.0% |
| INSERT | 0.0000 | 0.0000 | 0 | 0.0% |
| EXTRACT | 0.0000 | 0.0000 | 0 | 0.0% |
| SET_RECIPE | 0.0000 | 0.0000 | 0 | 0.0% |
| CONNECT | 0.0000 | 0.0000 | 0 | 0.0% |
| RESEARCH | 0.0561 | 0.0748 | 13 | 1.9% |

### HARVEST headline

HARVEST ok: **0**; items acquired: **0**.

First acquired-item step: db65a1bf/episode 0: never; db65a1bf/episode 1: never; db65a1bf/episode 2: never; db65a1bf/episode 3: never; db65a1bf/episode 4: never; db65a1bf/episode 5: never; db65a1bf/episode 6: never; db65a1bf/episode 7: never; db65a1bf/episode 8: never; db65a1bf/episode 9: never; db65a1bf/episode 10: never; db65a1bf/episode 11: never

### Failure reasons

- WAIT: none

- MOVE: none

- HARVEST: nothing_to_harvest=56 — `Could not harvest. Nothing within reach to harvest`

- CRAFT: recipe_disabled=59 — `Failed to craft 20x display-panel because recipe for display-panel is not unlocked yet (requires circuit-network technology). You need to research the technology first`; not_hand_craftable=4 — `Failed to craft 1x parameter-6 because Item parameter-6 cannot be crafted (category: parameters). Recipe requires a crafting machine or smelting in a furnace`; missing_ingredients=3 — `Failed to craft 1x iron-gear-wheel because couldn't craft a required sub-ingredient for iron-gear-wheel - iron-plate - Item iron-plate cannot be crafted (category: smelting). Recipe requires a craftin`

- PLACE: cannot_place=54 — `Could not place boiler at (5.0, -6.0), empty`; out_of_reach=4 — `Could not place medium-electric-pole at (7.0, -8.0), The target position is too far away to place the entity. The player position is 0, 0 and the target position is 7, -8. The distance is 10.63 and th`

- PICKUP: no_entity=64 — `no_entity`

- ROTATE: no_entity=53 — `no_entity`

- INSERT: no_entity=67 — `no_entity`

- EXTRACT: no_entity=77 — `no_entity`

- SET_RECIPE: no_entity=78 — `no_entity`

- CONNECT: fewer_than_two_entities=61 — `fewer_than_two_entities`

- RESEARCH: tech_prerequisite=54 — `Cannot start research for solar-panel-equipment. Missing prerequisites: modular-armor, solar-energy`


## regime=naive, fixture=seeded

| op | n | ok | no_effect | no_support | tool_rejected | error |
|---|---:|---:|---:|---:|---:|---:|
| WAIT | 25 | 25 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| MOVE | 19 | 18 (94.7%) | 1 (5.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| HARVEST | 21 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 21 (100.0%) | 0 (0.0%) |
| CRAFT | 25 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 25 (100.0%) | 0 (0.0%) |
| PLACE | 17 | 3 (17.6%) | 0 (0.0%) | 0 (0.0%) | 14 (82.4%) | 0 (0.0%) |
| PICKUP | 20 | 1 (5.0%) | 19 (95.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| ROTATE | 28 | 12 (42.9%) | 9 (32.1%) | 0 (0.0%) | 6 (21.4%) | 1 (3.6%) |
| INSERT | 23 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 22 (95.7%) | 1 (4.3%) |
| EXTRACT | 29 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 29 (100.0%) | 0 (0.0%) |
| SET_RECIPE | 15 | 0 (0.0%) | 2 (13.3%) | 0 (0.0%) | 13 (86.7%) | 0 (0.0%) |
| CONNECT | 22 | 0 (0.0%) | 5 (22.7%) | 0 (0.0%) | 17 (77.3%) | 0 (0.0%) |
| RESEARCH | 12 | 1 (8.3%) | 0 (0.0%) | 0 (0.0%) | 11 (91.7%) | 0 (0.0%) |

| op | median wall s | p90 wall s | median game ticks | wall share |
|---|---:|---:|---:|---:|
| WAIT | 1.5542 | 6.0605 | 825 | 58.2% |
| MOVE | 0.1726 | 1.0474 | 78 | 7.3% |
| HARVEST | 0.0690 | 0.0885 | 16 | 1.7% |
| CRAFT | 0.0601 | 0.0732 | 14 | 1.8% |
| PLACE | 0.0579 | 0.0773 | 14 | 1.2% |
| PICKUP | 0.1127 | 0.1317 | 45 | 2.7% |
| ROTATE | 0.1134 | 0.1303 | 46.5 | 3.8% |
| INSERT | 0.1117 | 0.1214 | 46 | 3.0% |
| EXTRACT | 0.1172 | 0.1370 | 46 | 4.0% |
| SET_RECIPE | 0.1126 | 0.1204 | 44 | 1.9% |
| CONNECT | 0.3971 | 0.8724 | 202 | 13.6% |
| RESEARCH | 0.0500 | 0.0670 | 13 | 0.7% |

### HARVEST headline

HARVEST ok: **0**; items acquired: **0**.

First acquired-item step: 634d2750/episode 0: never; 634d2750/episode 1: never; 634d2750/episode 2: never; 634d2750/episode 3: never

### Failure reasons

- WAIT: none

- MOVE: predicate_false=1 — `predicate_false`

- HARVEST: nothing_to_harvest=21 — `Could not harvest. Nothing within reach to harvest`

- CRAFT: recipe_disabled=23 — `Failed to craft 1x piercing-shotgun-shell because recipe for piercing-shotgun-shell is not unlocked yet (requires military-4 technology). You need to research the technology first`; missing_ingredients=2 — `Failed to craft 20x stone-furnace because couldn't craft a required sub-ingredient for stone-furnace - stone - recipe for stone doesn't exist, it is a raw resource that must be gathered first. Require`

- PLACE: cannot_place=13 — `Could not place chemical-plant at (-11.5, -42.5), wooden-chest=2, transport-belt=10, burner-inserter=4, small-electric-pole=4, pipe=10, coal=20, stone=10, iron-ore=20, iron-plate=20`; out_of_reach=1 — `Could not place underground-belt at (-2.5, 15.5), The target position is too far away to place the entity. The player position is 5.5, 8.5 and the target position is -2.5, 15.5. The distance is 10.63 `

- PICKUP: predicate_false=19 — `predicate_false`

- ROTATE: predicate_false=9 — `predicate_false`; other=6 — `Could not rotate stone-furnace.`; anchor_resolution_failed=1 — `anchor_resolution_failed: u1016 burner-inserter at (3.5, -1.5)`

  Top `other` messages: 2× `Could not rotate stone-furnace.`; 2× `Could not rotate small-electric-pole.`; 1× `Could not rotate wooden-chest.`; 1× `Could not rotate assembling-machine-1. Set the recipe first.`

- INSERT: other=12 — `The first argument must be a Prototype`; not_in_inventory=10 — `Could not insert: "No productivity-module to insert from your inventory"`; anchor_resolution_failed=1 — `anchor_resolution_failed: u1014 transport-belt at (1.5, 4.5)`

  Top `other` messages: 12× `The first argument must be a Prototype`

- EXTRACT: nothing_to_extract=18 — `Could not extract pumpjack from transport-belt at (2.5, 4.5): Could not find a valid transport-belt entity containing pumpjack`; other=11 — `'str' object has no attribute 'value'`

  Top `other` messages: 11× `'str' object has no attribute 'value'`

- SET_RECIPE: entity_not_found=7 — `Could not set recipe to arithmetic-combinatorNo building found that could have its recipe or filter set.`; other=6 — `Invalid entity type: belt-immunity-equipment`; predicate_false=2 — `predicate_false`

  Top `other` messages: 1× `Invalid entity type: belt-immunity-equipment`; 1× `Invalid entity type: explosive-rocket`; 1× `Invalid entity type: exoskeleton-equipment`; 1× `Invalid entity type: rocket-launcher`; 1× `Invalid entity type: logistic-robot`; 1× `Invalid entity type: flamethrower-ammo`

- CONNECT: other=15 — `<class 'fle.env.entities.BeltGroup'> is not a supported source object for fluid connection`; predicate_false=5 — `predicate_false`; not_in_inventory=1 — `Failed to connect {'SmallElectricPole'} from wooden-chest at x=3.5 y=3.5 to burner-mining-drill at x=-16.0 y=-51.0. You do not have enough small-electric-pole in you inventory to complete this connect`; no_path=1 — `Failed to connect {'Pipe'} from burner-mining-drill at x=-16.0 y=-51.0 to assembling-machine-1 at x=-3.5 y=0.5. Invalid path: nil`

  Top `other` messages: 4× `<class 'fle.env.entities.BeltGroup'> is not a supported source object for fluid connection`; 4× `'BeltGroup' object has no attribute 'x'`; 3× `<class 'fle.env.entities.BeltGroup'> is not a supported target object for fluid connection`; 1× `<class 'fle.env.entities.ElectricityGroup'> is not a supported source object for fluid connection`; 1× `Transport belts cannot be connected directly from a Prototype.StoneFurnace object as a source. You need to add an inserter that takes items from Prototype.StoneFurnace and use the inserter as a source`; 1× `Transport belts cannot be connected directly from a Prototype.WoodenChest object as a source. You need to add an inserter that takes items from Prototype.WoodenChest and use the inserter as a source e`; 1× `Transport belts cannot be connected directly from a Prototype.AssemblingMachine1 object as a source. You need to add an inserter that takes items from Prototype.AssemblingMachine1 and use the inserter`

- RESEARCH: tech_prerequisite=11 — `Cannot start research for rocket-fuel. Missing prerequisites: advanced-oil-processing, flammables`
