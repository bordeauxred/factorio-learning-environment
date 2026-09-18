# SM-ARQ on FLE: overnight results, 2026-09-18

Semi-Markov autoregressive Q-learning on Factorio, run on one M3 Max.

- Base PR: **#414** (`worktree-tiered-observation`), never modified or merged from here.
- Exact starting SHA: **bcd15fdab095a078a97b5b66498c164d63950310**
- Experimental branch: **`sm-arq`**, worktree `/Users/robertmueller/Desktop/agents/fle-sm-arq`
- Machine: Apple M3 Max, 14 cores, 36 GB. Factorio runs in Docker (7 vCPU, 12.8 GB) under box64;
  the learner runs on the host through MPS.
- Interface contract: `fle/smarq/contract.py`. Spec: `docs/rl/specs/sm-arq.md`.

> Status while this file is being written: the simulator probe is complete and the
> numbers below are measured. Environment, learner and training-loop sections are
> filled in as each lands. Nothing in this file is an estimate unless it says so.

## 1. Simulator time semantics (measured)

Full report: `docs/rl/specs/sim-probe.md`. The design commitments in section 1 of the
brief are all achievable on this server, and two of them required choosing between
implementations:

| Question | Measurement | Consequence |
|---|---|---|
| Does pausing freeze the factory? | `game.tick_paused=true` held the tick at 45,046 and 45,114 across 3.0 s wall windows, with a fuelled furnace mid-craft; its crafting progress was unchanged | Inference and SGD run at a frozen decision boundary. RCON still answers while paused (`pong@45114`), so observations are read without spending game time |
| How fast can the simulator run? | speed 40 → **1,165.77 ticks/wall-s = 19.4 game-s/wall-s**; speeds 100, 400 and 1000 were all *slower* (1,109-1,129 ticks/s) | Execution speed is 40. The box64/ARM container is CPU-bound well below Factorio's speed cap |
| Can FAST_FORWARD land exactly? | RCON polling overshot by 7.0 ticks mean, 9 max. A Lua `on_nth_tick(1)` handler that pauses at a stored target tick landed **exactly, 0 overshoot in all 20 trials** | FAST_FORWARD uses the Lua target handler. Polling is not accurate enough for an SMDP duration |
| What does simulated time cost? | 3,600 ticks (60 game s) in 3.47 s wall; 36,000 ticks (600 game s) in 35.07 s. With one fuelled furnace: 3.55 s and 37.71 s | ~17 game-seconds per wall-second. A 30-game-minute episode is ~93 wall seconds of pure simulation |
| What does a decision boundary cost? | trivial RCON 2.60 ms mean / 5.23 p95; tiered observation drain+reconcile+tensors **2.48 ms** / 5.09 p95; `score()` **20.04 ms** / 36.68 p95; full boundary 21.89 ms / 40.26 p95 | The PR #414 observation is nearly free. `score()` dominates the boundary and is the thing to optimise if throughput binds |
| What does walking cost? | 10 tiles: 145 ticks; 30 tiles: 438; 60 tiles: 765 (0.16 / 0.42 / 0.76 wall s) | Navigation inside an action is affordable, and its simulated cost enters tau honestly |

Nothing sleeps in wall-clock time for the factory. Wall time appears in this document
only as a throughput metric.

### Score semantics, measured

`namespace.score()` returns `(production_score, automated_production_score)`.

| Stage | Production | Automated | Δ production | Δ automated |
|---|---:|---:|---:|---:|
| Baseline | 0 | 0 | 0 | 0 |
| Hand-mined one iron ore | 3 | −1 | +3 | **−1** |
| Hand-crafted one gear | 4 | −1 | +1 | 0 |
| Fuelled furnace produced autonomously | 5 | 0 | +1 | **+1** |

The automated score does what the brief wants: manual mining and manual crafting do not
pay, autonomous production does. It can go negative, so nothing downstream may assume
it is non-negative.

## 2. Environment

**Observation.** PR #414's tiered client, promoted into `fle/smarq/obs.py` unchanged in
layout: `grid (17,96,96)` of 3-tile cells, `entity_view (2048,38)` of the entities
nearest the character, `entity_mask`, `globals (10,)`. Added: `entity_ids`, the stable
Factorio unit numbers of the rows, so an entity pointer chosen at one boundary can be
resolved at execution and recognised in replay. On top of it, `fle/smarq/raster.py`
builds an **exact-tile egocentric raster** (default 96x96 tiles, 8 binary channels:
occupied, impassable, resource, belt, machine, tree/rock, player, in-build-reach)
derived only from state the client already holds. `in_build_reach` is an observation
channel and masks nothing.

**Action grammar.** Eleven verbs, decided autoregressively: MOVE_TO, MINE, CRAFT,
PLACE, PICKUP, ROTATE, INSERT, EXTRACT, SET_RECIPE, RESEARCH, FAST_FORWARD. Heads are
verb, prototype (87), position (9216 exact tiles at raster 96), direction (4), entity
pointer (2048), item (143), quantity (1/2/4/8/16/32/ALL), recipe (143), technology (61),
duration (1 s/10 s/60 s). Vocabularies are read from the live game and cached.

**Navigation semantics.** The executor may compute a reachable interaction position and
walk there; it never alters the requested coordinate. `PLACE(prototype, x, y, dir)`
preserves `(x, y)` exactly and fails as `blocked` / `unreachable` /
`insufficient_inventory` / `invalid_argument` rather than relocating. There is no
`nearest_buildable`, no `place_entity_next_to`, no snapping, no projection, and no
`connect_entities` macro anywhere in `fle/smarq/` (asserted by test and by grep).
One FLE behaviour had to be refused explicitly: `place_entity` silently relocates
offshore pumps, so SM-ARQ rejects that case instead of accepting a moved coordinate.

**Masks.** Structural only: placeable prototypes, live entities, entities with an
inventory for INSERT/EXTRACT, recipe-capable machines for SET_RECIPE, hand-craftable
recipes, unresearched technologies. Nothing is masked by inventory contents, occupancy,
reach, distance or quality, and the position head is never masked at all.

**Episode.** Fixed simulated-time budget (default 30 game minutes = 108,000 ticks) plus
an emergency cap of 1000 decisions. Repeated FAST_FORWARD consumes the same budget.

**Starting state.** FLE's standard open-play kit: 50 coal, 50 iron and 50 copper plates,
9 stone furnaces, 3 burner mining drills, 1 electric drill, 32 burner inserters, 50
belts, 1 assembling machine, boiler, steam engine, poles. **No ore.** Measured on these
maps, the nearest iron ore is **170 tiles away on port 27004 and 274 tiles on port
27000**. Automated production therefore requires a genuine journey: with a 96-tile
window the agent must chain several MOVE_TO hops before an ore tile is even addressable
by the position head. In game time that journey is cheap (about 13 ticks per tile, so
~2,200 ticks for 170 tiles, ~2% of a 30-minute episode); the difficulty is exploration,
not time.

## 2b. TOY-B: can this interface automate at all?

Before asking whether a learner can find it, the question is whether the action grammar
can express it. The scripted expert in `tests/benchmarks/smarq_live_demo.py` uses only
semantic actions with exact coordinates, on port 27004, starting beside an ore patch:

```
MOVE_TO      (-197,-229)                                   -> ok    tau=29t
PLACE        burner-mining-drill (-197,-231) SOUTH         -> ok    tau=9t
INSERT       slot 0, coal x8                               -> ok    tau=5t   automated -3
PLACE        stone-furnace (-197,-229) NORTH               -> ok    tau=5t
INSERT       slot 0, coal x8                               -> ok    tau=7t
FAST_FORWARD 60 s                                          -> ok    tau=3600t automated +66
FAST_FORWARD 60 s                                          -> ok    tau=3600t automated +76
```

Automated production score went **0 → 139 over 120 simulated seconds** (60.9 wall
seconds for the whole demonstration, 7,269 simulated ticks). The interface can automate,
with exact geometry and no build planner. Two honest caveats: the automated score first
dips to −3 because fuel consumption is charged before output is credited, and the
demonstration's own bookkeeping of entity slots is naive (rows are re-sorted nearest-first
each observation, so its second INSERT addressed the drill rather than the furnace, and
its EXTRACT of a plate that did not exist returned `tool_error`). The learner is not
affected by that: it re-chooses a pointer from the current observation at every boundary.

## 3. Algorithm

SMDP formulation actually implemented:

- A completed semantic action gives `(s, a, r, tau, s')` with `tau` in **simulated**
  game seconds.
- Continuous-time discount `Gamma(tau) = 2^(-tau/H)`, `H = 3600` simulated seconds by default.
- Target `r + Gamma(tau) * Q_target(s', a*)`, `a* = argmax_a Q_online(s', a)` taken through
  the same autoregressive maximisation the policy uses.
- Action construction is a chain of zero-time internal decisions: for those, `r = 0`,
  `tau = 0`, `Gamma = 1`, and a partial action bootstraps from the max over its next head.

_Network sizes and measured update rates to be filled when the learner lands._

## 4. M3 performance

_To be filled from `tests/benchmarks/smarq_throughput.py`._

## 5. Toy results

_TOY-A (SMDP correctness) and TOY-B (burner automation) to be filled._

## 6. Full results

| run | reward | scratch/demo | simulated horizon | semantic decisions | final production score | final automated score | automated / game minute | game-s per wall-s | wall clock |
|---|---|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | | | |

## 7. Failure analysis

_To be filled: manual grinding, invalid placement, Q divergence, exploration failure,
simulation bottlenecks._

## 8. Verdict

_To be chosen from A (clear swim) / B (weak swim) / C (infrastructure works, learning
unresolved) / D (drown), with the single highest-value next experiment._
