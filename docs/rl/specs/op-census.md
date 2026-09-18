# Spec: operation census harness (random policy, live server)

Status: to implement. Owner of the spec: main session. Implementer: Codex.

## Purpose

Produce the table the last attempt never produced: for each of the twelve
macro operations, driven by a uniform random policy on a live open-world
server, how often it succeeds and the distribution of why it fails. The table
is the environment specification. Nothing here is a learner.

Read `docs/rl/BRIEF.md` before writing code. Constraints that apply to this
work: do not modify anything under `fle/env/` or `fle/cluster/`; read every
tool client you call before calling it; no scripted policies; no reward logic.

## Live facts already measured (2026-09-15, seed 44340, spawn (0,0))

| Fact | Value |
|---|---|
| `character.resource_reach_distance` / `build_distance` | 2.7 / 10 tiles (read live, do not hard-code; read at startup) |
| Nearest tree / iron / copper / stone / coal from spawn | 48 / 53 / 68 / 81 / 92 tiles |
| Resource tiles within 40 tiles of spawn | 0 |
| `obs_terrain_full_sync()` | 2,601 chunks, 1.6 MB, 22.8 s. Do it once per server, reuse across episodes |
| `FactorioInstance.reset(reset_position=True, all_technologies_researched=False)` | 0.16 s |
| `move_to(nearest stone)` 81 tiles then `harvest_resource(pos, 5)` | 1.0 s + 0.5 s wall; inventory `stone: 5`; `craft_item("stone-furnace")` then works |
| `harvest_resource(player+3, 1)` from spawn | `Could not harvest. Nothing within reach to harvest` |
| `score()` after a manual harvest | `(12, 0)`: player score 12, automated 0 |

The probe script that measured these is at the end of this document; reuse
its connection pattern.

## Deliverables

1. `fle/rl/__init__.py` (empty).
2. `fle/rl/world.py`: `WorldClient`, a thin client over the PR #414 wire
   protocol (`obs_terrain_full_sync`, `obs_diff_full_sync`, `obs_all_drain`,
   all sent as `/sc <fn>()` RCON commands). Parse formats exactly as
   `tests/benchmarks/benchmark_tiered_obs.py::TieredClient` does, plus a parsed
   entity row: the `u<unit>,<name>,<x>,<y>,<direction>,<status>,<extras...>`
   row where extras are tagged fields (`E` energy, `P` progress, `R` recipe,
   `H` health, `W` w:h, `D` drop, `K` pickup, `N` net id, `T` temp,
   `I<idx>.<item>:<count>`, `F<fluid>:<amount>`). See
   `fle/cluster/scenarios/open_world/observation_diff.lua::rich_row`.
   Also provide:
   - `patches()`: connected components (4-neighbour) of same-name ore tiles
     from `ores`, each with `id` (deterministic: name plus min (x, y) tile),
     `name`, `n_tiles`, `bbox`, and `nearest_tile(px, py)`.
   - `read_reach()`, `read_player_pos()`, `read_tick()`, `read_current_research()`,
     `read_enabled_recipes()`, `read_research_state()` (name -> researched,
     enabled, prerequisites) as single RCON reads with `helpers.table_to_json`.
   - Player inventory via `namespace.inspect_inventory()` (returns a pydantic
     model with extra fields; `dict(inv)` gives name -> count).
3. `fle/rl/ops.py`: the twelve operations, argument samplers for two regimes,
   an executor, and an outcome classifier. Details below.
4. `tests/benchmarks/run_op_census.py`: the driver. CLI:
   `--port 27000 --regime {naive,macro} --fixture {empty,seeded} --episodes K
   --steps N --seed S --speed 10 --out <jsonl>`. Writes one JSON line per step,
   flushed every step (the last attempt lost an 8-hour run to missing logs).
   Prints a progress line every 25 steps with cumulative per-op ok counts.
5. `tests/benchmarks/summarize_op_census.py`: reads one or more JSONL files and
   writes a markdown report: per (regime, fixture) a table with one row per op:
   n, ok, no_effect, no_support, tool_rejected, error, as counts and percent;
   per op the top reason classes with counts and one example raw message each;
   per op median and p90 wall seconds and median game ticks; the HARVEST
   headline: total ok, items acquired, and for each episode the step index of
   the first acquired item or `never`. Also the share of total wall-clock per
   op (the WAIT 73.8 percent lesson).
6. `tests/rl/test_op_census_offline.py`: no-server tests for the row parser,
   `patches()`, `classify_error`, both samplers on a fake state (empty world
   gives `no_support` for the six entity ops; a state with one patch gives a
   HARVEST spec; naive HARVEST offsets are in [-8, 8]^2), and the effect
   verifier on hand-built before/after snapshots.

## Operations, samplers, and success predicates

Twelve ops: `WAIT, MOVE, HARVEST, CRAFT, PLACE, PICKUP, ROTATE, INSERT,
EXTRACT, SET_RECIPE, CONNECT, RESEARCH`. The driver picks the op uniformly at
random (1/12 each) in both regimes; the regime only changes how arguments are
sampled. A sampler returns either an `ActionSpec(op, args)` or
`NoSupport(reason)`; `NoSupport` is logged as outcome `no_support` and costs no
server call.

Quantity bins: naive uses `(1, 5, 20, -1)` for HARVEST (reproducing the old
contract; `-1` is expected to be rejected), `(1, 5, 20)` elsewhere. Macro uses
`(1, 5, 20)` everywhere. Directions: the four cardinals of
`fle.env.entities.Direction`. Offsets: integer `dx, dy` in `[-8, 8]`.

### naive regime (reproduces the old contract, expected to fail)

| Op | Arguments | Call |
|---|---|---|
| WAIT | seconds in (1, 5, 15, 60) | `namespace.sleep(seconds)` |
| MOVE | offset | `move_to(player + offset)` |
| HARVEST | offset, quantity | `harvest_resource(player + offset, quantity)` |
| CRAFT | recipe uniform over the 217 vocab recipes, quantity | `craft_item(name, quantity)` (client accepts a str) |
| PLACE | prototype uniform over the placeable allowlist below, offset, direction | `place_entity(Prototype, direction, player + offset, exact=True)` |
| PICKUP | entity uniform over tracked rows | `pickup_entity(entity_obj)` |
| ROTATE | entity, direction | `rotate_entity(entity_obj, direction)` |
| INSERT | entity, item uniform over the 251 vocab items, quantity | `insert_item(Prototype, entity_obj, quantity)` |
| EXTRACT | entity, item uniform over 251, quantity | `extract_item(Prototype, entity_obj, quantity)` |
| SET_RECIPE | entity, recipe uniform over 217 | `set_entity_recipe(entity_obj, name)` |
| CONNECT | two distinct entities, connector in (transport-belt, pipe, small-electric-pole) | `connect_entities(a, b, connection_type=Prototype)` |
| RESEARCH | technology uniform over 196 | `set_research(name)` |

Entity ops with zero tracked rows return `NoSupport("no_entity")`; CONNECT
with fewer than two rows returns `NoSupport("fewer_than_two_entities")`.

### macro regime (design intent: navigation absorbed, support-only masks)

| Op | Arguments | Execution |
|---|---|---|
| WAIT | ticks in (60,) if no tracked entity rows exist, else (60, 300, 900, 3600) | `sleep(ticks / 60)`; record true tick delta |
| MOVE | uniform over known 4x4-tile cells within 64 tiles of the player whose chunk is generated and whose tile is not water | `move_to(cell centre)` |
| HARVEST | target kind uniform in (patch, tree); patch uniform over `patches()` within 160 tiles (all resource names, including ones the character cannot hand-mine: the simulator decides); tree uniform over trees within 160 tiles; quantity | resolve the working tile (patch: tile nearest the player; tree: its tile); `move_to(tile)`; read live position; if distance to tile > reach - 0.1 record `approach_failed` and stop; else `harvest_resource(tile, quantity)` |
| CRAFT | recipe uniform over currently enabled force recipes, quantity | as naive |
| PLACE | prototype uniform over placeable items currently in inventory, else `NoSupport("no_placeable_in_inventory")`; offset; direction | position = player + offset, snapped to the legal centre for the prototype's `tile_width`/`tile_height` from the vocab (odd width means x + 0.5); `place_entity(..., exact=True)` |
| PICKUP | entity | as naive |
| ROTATE | entity, direction different from current | as naive |
| INSERT | entity, item uniform over items in the player inventory (else `NoSupport("empty_inventory")`), quantity | as naive |
| EXTRACT | entity, item uniform over items present in that entity's `I` fields (else `NoSupport("entity_has_no_items")`), quantity | as naive |
| SET_RECIPE | entity whose vocab type is `assembling-machine` (else `NoSupport`), recipe uniform over enabled recipes | as naive |
| CONNECT | two distinct entities, connector uniform over connector items in inventory (else `NoSupport("no_connector_in_inventory")`) | as naive |
| RESEARCH | technology uniform over those enabled, not researched, all prerequisites researched | as naive |

None of these encode preference. Every rule removes only what the simulator
refuses deterministically, or moves the player into reach, which the tool
already requires.

Placeable allowlist for naive PLACE (names that exist both in the vocab
`items[].place_result` and in `fle.env.game_types.Prototype`): burner-mining-drill,
electric-mining-drill, stone-furnace, steel-furnace, electric-furnace,
assembling-machine-1, assembling-machine-2, wooden-chest, iron-chest,
steel-chest, transport-belt, fast-transport-belt, underground-belt, splitter,
burner-inserter, inserter, long-handed-inserter, fast-inserter, pipe,
pipe-to-ground, small-electric-pole, medium-electric-pole, boiler,
steam-engine, lab, offshore-pump, pumpjack, chemical-plant, oil-refinery.
Verify each against the enum at import and drop with a logged warning any that
do not resolve.

Entity objects: `pickup_entity`, `rotate_entity`, `insert_item`, `extract_item`,
`set_entity_recipe`, `connect_entities` take `fle.env.entities.Entity`
instances, not rows. Resolve a sampled row to an object with
`namespace.get_entities(position=Position(x, y), radius=0.5)` filtered by
name, or `namespace.get_entity(Prototype, Position)` when the name resolves to
a `Prototype`. Read those clients first; `get_entity` returns `None` when
missing despite its annotation. If resolution fails, log outcome `error` with
reason class `anchor_resolution_failed`.

### Success predicates (effect verification, not return values)

Snapshot before and after every executed action: player inventory, the drained
entity rows, live player position, `game.tick`, and `score()`. Ok means the
effect happened; `no_effect` means the tool returned without raising but the
predicate is false.

| Op | ok iff |
|---|---|
| WAIT | `game.tick` advanced by at least 1; log requested vs actual ticks |
| MOVE | final live distance to the requested point <= 1.0 |
| HARVEST | total player inventory count increased; log `items_acquired` and the item names |
| CRAFT | inventory count of the recipe's main product increased |
| PLACE | count of rows with that name increased by 1 |
| PICKUP | the row's unit number is gone after the drain and inventory count of the item increased |
| ROTATE | the row's direction after equals the requested direction and differs from before |
| INSERT | inventory count of the item decreased |
| EXTRACT | inventory count of the item increased |
| SET_RECIPE | the row's `R` field after equals the requested recipe |
| CONNECT | no exception and total row count increased |
| RESEARCH | `force.current_research.name` after equals the requested technology |

Outcome statuses: `ok`, `no_effect`, `no_support`, `tool_rejected` (the tool
raised, message classified below), `approach_failed` (macro HARVEST only),
`error` (harness-side exception, e.g. type errors, anchor resolution; include
the traceback's last line).

### Error classification

`classify_error(op, message) -> str` maps the raised message to one of a
small set of classes by regex, keeping the raw message (first 200 chars) as
well: `out_of_reach`, `nothing_to_harvest`, `invalid_quantity`,
`no_path`, `cannot_place`, `not_in_inventory`, `missing_ingredients`,
`recipe_disabled`, `not_hand_craftable`, `unknown_name`, `entity_not_found`,
`no_suitable_slot`, `nothing_to_extract`, `connect_failed`,
`tech_researched`, `tech_prerequisite`, `tech_unknown`, `assertion`, `other`.
Start from the messages in the Lua servers under `fle/env/tools/agent/*/server.lua`
and the Python clients; anything unmatched is `other` and the summary lists
the top ten `other` raw messages per op so the taxonomy can be extended.

## Fixtures

`empty`: `instance.reset(reset_position=True, all_technologies_researched=False,
clear_entities=True)` with empty inventory. This is the open-play start and the
primary table.

`seeded`: a measurement-only fixture so the six entity ops have support. After
the empty reset, create via one RCON Lua block with `raise_built = true`:
a stone-furnace at (3, 0), a wooden-chest at (3, 3), an assembling-machine-1
at (-4, 0), three transport-belts at (0, 4), (1, 4), (2, 4), a burner-inserter
at (3, -2), and a burner-mining-drill on the nearest iron tile (read from
`patches()`); set player inventory via `namespace._set_inventory` or the
`set_inventory` admin tool (read it first) to `{coal: 20, iron-ore: 20,
iron-plate: 20, stone: 10, transport-belt: 10, burner-inserter: 4,
small-electric-pole: 4, pipe: 10, wooden-chest: 2}`. Document in the report
header that this fixture exists only to measure the entity ops and is never a
training start.

## Run protocol

Per run: connect `FactorioInstance(address="localhost", tcp_port=port,
fast=True, all_technologies_researched=False, inventory={}, reset_speed=speed)`;
one terrain full sync; then for each episode: reset, entity full sync, apply
fixture, then N steps. Each step: sample op, sample args, snapshot, execute,
`obs_all_drain()` and apply, snapshot, classify, log. Log fields per step:
`run_id, regime, fixture, seed, episode, step, op, args (JSON-safe), status,
reason_class, raw_error, wall_tool_s, wall_observe_s, ticks_before,
ticks_after, player_pos_before, player_pos_after, inventory_after (dict),
items_delta (dict), n_entities_after, score_player, score_automated,
target_distance` (macro MOVE and HARVEST). Leave the server unpaused at
`game.speed = speed` for the whole run; do not add pause logic.

Wrap each tool call in a 120 s wall timeout (thread with `concurrent.futures`),
logging `error` with class `timeout` if it fires, and re-sync entities after any
`error`.

## Acceptance

- `pytest -q tests/rl/test_op_census_offline.py` passes with no server.
- `ruff check fle/rl tests/benchmarks/run_op_census.py tests/benchmarks/summarize_op_census.py tests/rl/test_op_census_offline.py` passes.
- A live smoke run `--regime macro --fixture empty --episodes 1 --steps 24 --seed 1`
  against port 27000 completes, writes 24 JSON lines, and the summary script
  renders a table from it. Include the printed summary in your final report
  verbatim. If HARVEST never came up in 24 uniform draws, also run
  `--steps 48` once. Do not run anything longer than that; the main session
  runs the full census.
- Report every interface surprise you hit (a return type, a rounding, an
  exception path) in the final message, with `file:line`.

## Reference: the probe that produced the live facts

```python
inst = FactorioInstance(address="localhost", tcp_port=27000, fast=True,
                        all_technologies_researched=False, inventory={}, reset_speed=10)
ns = inst.namespace
inst.reset(reset_position=True, all_technologies_researched=False, clear_entities=True)
rc = inst.rcon_client
reach = json.loads(rc.send_command("/sc local c = storage.agent_characters[1] "
    "rcon.print(helpers.table_to_json({resource_reach=c.resource_reach_distance, "
    "build=c.build_distance, x=c.position.x, y=c.position.y}))"))
terrain = rc.send_command("/sc obs_terrain_full_sync()")   # 22.8 s, 1.6 MB
p = ns.nearest(Resource.Stone)                              # Position(60.5, -54.5)
ns.move_to(p)                                               # 1.0 s, lands exactly on p
ns.harvest_resource(p, quantity=5)                          # returns 5; inventory stone: 5
ns.craft_item("stone-furnace", 1)                           # returns 1
```
