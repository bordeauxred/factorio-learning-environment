# Operation census: what a random policy can do on a live server (2026-09-15)

This is the table the brief asked for first. One open-world server
(Factorio 2.0.73, map seed 44340, `game.speed = 10`), a uniform random policy
over the twelve macro operations, effects verified against the game state
after every action, and every step logged. Raw logs and the machine-generated
report are in `op_census/`. The harness is `tests/benchmarks/run_op_census.py`
over `fle/rl/ops.py` and `fle/rl/world.py`; its spec is `docs/rl/specs/op-census.md`.
Nothing in `fle/env/` or `fle/cluster/` was changed.

Two argument-sampling regimes were measured with the same op distribution
(1/12 each):

- **naive** reproduces the previous attempt's contract: HARVEST and MOVE take a
  player-relative offset in [-8, 8]^2, HARVEST quantities include the `-1`
  "all" bin, CRAFT/RESEARCH/INSERT/EXTRACT draw uniformly over the full vocab.
- **macro** is the design intent: HARVEST picks a known target anywhere within
  160 tiles (ore patch or tree) and absorbs the walk; every other op draws only
  from arguments the simulator does not deterministically refuse (item in
  inventory, entity present, recipe enabled and hand-craftable, technology
  unlocked with prerequisites researched).

Two start states: **empty** is the open-play start (empty inventory, no
entities, default research). **seeded** is a measurement fixture only, so the
six entity operations have something to act on: a furnace, chest, assembler,
three belts, a burner inserter near spawn, a burner drill on the nearest iron
tile, and a small inventory. It is never a training start.

## Live facts the table rests on

| Fact | Value |
|---|---|
| `character.resource_reach_distance` / `build_distance` | 2.7 / 10 tiles (read live) |
| Nearest tree / iron / copper / stone / coal from spawn | 48 / 53 / 68 / 81 / 92 tiles |
| Resource tiles within 40 tiles of spawn | 0 |
| Terrain full sync | 2,601 chunks, 1.6 MB, 22.8 s; reused across episodes |
| Episode reset | 0.16 s |

The last attempt's ±8 offset contract therefore had **zero** harvest support
from spawn on this map, not a small probability. Its 0/6,752 was certain.

## Headline

| Regime, start | Steps | HARVEST ok / attempts | Items acquired | Episodes with a first item | Best player PS | Automated PS |
|---|---:|---:|---:|---:|---:|---:|
| naive, empty | 768 (12 × 64) | **0 / 56** | 0 | 0 / 12 | 0 | 0 |
| macro, empty | 768 (12 × 64) | **61 / 62** | 637 | 12 / 12 (first item at step 1 to 37, median 6) | 297 | 0 |
| naive, seeded | 256 (4 × 64) | 0 / 21 | 0 | 0 / 4 | | 0 |
| macro, seeded | 256 (4 × 64) | 20 / 20 | 202 | 4 / 4 | 410 | 0 |

A random policy acquires a resource on a live server. The gate from the brief
is passed. Harvest success did not depend on distance: 9/9 at 0-40 tiles,
17/17 at 40-80, 11/11 at 80-120, 24/25 at 120-160. Walking is the cost, not
the risk: median HARVEST wall time was 2.1 s and 1,099 game ticks, and HARVEST
took 58% of the macro run's wall clock.

Automated PS stayed 0 in every run. That is expected: automation needs a fuelled
drill or a fuelled, fed furnace, and 64 uniform random steps never assembled
one. It is the learner's job, not the census's.

## The twelve operations, empty start

Counts are per op; `no_support` means the sampler found no legal argument and
no server call was made.

| Op | naive: n | ok | dominant failure | macro: n | ok | dominant failure |
|---|---:|---:|---|---:|---:|---|
| WAIT | 59 | 59 | | 61 | 61 | |
| MOVE | 74 | 74 | | 74 | 71 | 3 stopped 1.6-2.1 tiles short of a blocked cell |
| HARVEST | 56 | 0 | 56 "Nothing within reach to harvest" | 62 | 61 | 1 character death (see below) |
| CRAFT | 66 | 0 | 59 recipe locked, 4 pseudo-recipe, 3 missing ingredients | 71 | 4 | 61 missing ingredients (all need iron plate), 6 `recipe-unknown` crashes the craft tool |
| PLACE | 58 | 0 | 54 nothing in inventory, 4 beyond build reach | 62 | 8 | 53 no placeable held; 1 beyond build reach |
| PICKUP | 64 | 0 | 64 no entity | 58 | 0 (5 succeeded, see lag) | 53 no entity |
| ROTATE | 53 | 0 | 53 no entity | 56 | 0 | 47 no entity; 9 furnace/chest not rotatable |
| INSERT | 67 | 0 | 67 no entity | 66 | 7 | 58 no entity |
| EXTRACT | 77 | 0 | 77 no entity | 68 | 2 | 66 no entity or entity empty |
| SET_RECIPE | 78 | 0 | 78 no entity | 63 | 0 | 63 no assembler |
| CONNECT | 61 | 0 | 61 fewer than two entities | 68 | 0 | 66 fewer than two entities |
| RESEARCH | 55 | 1 | 54 prerequisites missing | 59 | 49 | 10 trigger technologies refused |

Wall-clock share, empty start: naive spent **85.5%** of its clock in WAIT
(median 1.56 s, p90 6.07 s) in a world where nothing runs. Macro spent 9.7% in
WAIT (60 ticks only while no entity exists), 58% in HARVEST, 26% in MOVE.

What the macro random policy did with what it harvested, without any script:
crafted stone furnaces and wooden chests, placed 8 of them, inserted stone into
a furnace and ore, wood and stone into chests, extracted from both, and queued
49 research selections. Items harvested: 372 wood, 90 stone, 83 copper ore,
72 iron ore, 20 coal.

## The twelve operations, seeded fixture (macro regime)

| Op | n | ok | Failures |
|---|---:|---:|---|
| WAIT | 24 | 24 | |
| MOVE | 17 | 17 | |
| HARVEST | 20 | 20 | |
| CRAFT | 27 | 10 | 17 missing ingredients (wood, plates) |
| PLACE | 24 | 20 | 2 collision, 1 beyond build reach, 1 placed but not visible in the next drain |
| PICKUP | 22 | 21 succeeded by inventory delta | all 21 logged `no_effect` because the row removal arrives one reconciler sweep later; 1 anchor already gone |
| ROTATE | 23 | 12 | 9 not rotatable: pipe, furnace, chest, assembler with no recipe |
| INSERT | 17 | 9 | 4 wrong slot type (belt into pipe), 4 "no nearby entity accepts stone" |
| EXTRACT | 28 | 1 | 27 entity has no items (nothing is fuelled) |
| SET_RECIPE | 17 | 0 logged | 3 logged `no_effect` with the recipe unset in the next drain; a live re-check shows the recipe is set and the row lags one sweep; 4 `parameter-N` pseudo-recipes refused; 10 no assembler in the drawn state |
| CONNECT | 25 | 1 | **17 crash inside `connect_entities` whenever an endpoint is a belt** (`'BeltGroup' object has no attribute 'x'`); 2 no connector held; 1 no path; 1 blocked drop position; 2 route found but built nothing |
| RESEARCH | 12 | 11 | 1 trigger technology |

The naive regime on the same fixture: HARVEST 0/21, CRAFT 0/25, INSERT 0/23,
EXTRACT 0/29 (uniform draws over 251 items and 217 recipes almost never name
something held or craftable), PLACE 3/17, ROTATE 12/28, RESEARCH 1/12.

## Environment facts the census established

These are properties of FLE and the observation protocol, each verified live
after the run, and each is an input to the action interface design.

1. **Harvest reach is 2.7 tiles and the error message does not distinguish
   "too far" from "nothing there"** (`harvest_resource/server.lua:469-478`).
   With navigation absorbed into HARVEST the reach gate never fired: 81 of 82
   macro harvests succeeded.
2. **Harvest quantities are advisory.** A tree yields 4 wood per `quantity=1`;
   `quantity=20` on a patch returned 16 to 22. Yield, not request, is the
   effect.
3. **Trigger technologies cannot be queued.** Factorio 2.0's `electronics` and
   `steam-power` have a `research_trigger` and `force.add_research` returns
   false for them even with prerequisites met. The pinned vocabulary has no
   trigger flag, so the research support rule needs one. The research queue
   holds 7 entries.
4. **The vocabulary export contains 11 pseudo-recipes** (`parameter-0..9`,
   `recipe-unknown`) and the force reports them enabled. `parameter-N` is
   refused with a clear error; `recipe-unknown` crashes `craft_item` with a
   Lua nil index. Hand-craft support = recipe enabled and category in the
   character's crafting categories (`{"crafting"}` live) removes all 11 and
   smelting, exactly as the craft tool itself checks.
5. **Rotatability is a prototype fact the observation does not expose.**
   Pipes, furnaces, chests refuse; an assembler refuses until it has a recipe;
   belts, drills, inserters rotate. Vocabulary needs a `supports_direction`
   flag, or the executor learns it from refusals.
6. **Two tools have a one-sweep observation lag.** `pickup_entity` destroys
   with `raise_destroy=false` (`pickup_entity/server.lua:81`) and
   `set_entity_recipe` does not touch the observation row, so their effects
   appear only on the next reconciler sweep, measured at about 0.2 to 0.3 s.
   PR #414 already gave `place`, `rotate` and `insert` instant visibility; these
   two need the same hook, or the step must drain after the sweep.
7. **`connect_entities` fails on belt endpoints.** Any CONNECT with a
   transport belt as source or target crashed in the client with
   `'BeltGroup' object has no attribute 'x'` (17 of 25 attempts). This is an
   FLE tool defect to reproduce in isolation before CONNECT is offered.
8. **The character can die.** Once in 768 macro steps, 250 tiles from spawn,
   the character became invalid and reappeared at (0, 0) with its inventory
   intact. `peaceful=True` does not prevent it. Episodes need a death signal.
9. **`move_to` stops short when the target cell is occupied**, 1.6 to 2.1
   tiles away in 3 of 74 macro moves. The arrival predicate must tolerate that
   or the executor must pick free cells.
10. **Placement reach is 10 tiles** and offsets in [-8, 8]^2 exceed it at the
    corners (5 refusals). Same lesson as harvest: absorb the approach.
11. **Support masks change the picture but do not solve the game.** At the
    empty start, six ops have no support 84 to 100% of the time until the
    policy has built something; CRAFT needs iron plates, which need a fuelled
    furnace. The ladder to automated PS is the learner's problem now, on top
    of an interface where every offered action can execute.

## What this changes in the plan

The brief's step 1 is done and step 2's central claim is verified: a visible
patch 40 tiles away, and 160 tiles away, is obtainable in one decision. The
interface for the learner should be the macro regime's contract with these
additions from the facts above: a trigger flag and rotatability flag in the
vocabulary, post-sweep draining or touch hooks for PICKUP and SET_RECIPE, CONNECT
withheld until the belt-endpoint crash is fixed, a death terminal, and approach
absorbed into PLACE as well as HARVEST. Then build the observation the
policy needs to choose among those targets, and only then a learner.

## Reproduction

```sh
.venv/bin/fle cluster start -n 1 -s open_world
.venv/bin/python tests/benchmarks/run_op_census.py --regime macro --fixture empty --episodes 12 --steps 64 --seed 11 --out docs/rl/results/op_census/macro_empty.jsonl
.venv/bin/python tests/benchmarks/summarize_op_census.py docs/rl/results/op_census/*.jsonl --out docs/rl/results/op_census/report.md
```

Runs: naive/empty 11:20-11:23, macro/empty 11:23-11:27, naive/seeded
11:27-11:29, macro/seeded 11:29-11:31 local time, seeds 11 and 12. Offline
tests: `pytest -q tests/rl/test_op_census_offline.py` (47 passed).
