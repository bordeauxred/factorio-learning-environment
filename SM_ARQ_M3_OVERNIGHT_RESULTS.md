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
belts, 1 assembling machine, boiler, steam engine, poles. **No ore.** The nearest iron
ore is **53.5 tiles from spawn** at (−16, −51) on map seed 44340. Automated production
therefore needs a short journey first: the exact-tile window is ±48 tiles, so the ore
tile is not addressable by the position head until the agent has made one MOVE_TO hop
toward it. In game time the journey is cheap (~13 ticks per tile, under 1% of a
30-minute episode); the difficulty is exploration, not time.

> A measurement note worth recording, because it nearly went into this document as a
> fact: `find_entities_filtered` with a `limit` returns an *arbitrary* subset, not the
> nearest ones. Taking "the nearest" of a limited result set reported ore at 170 and 274
> tiles on two maps. Re-measured with a radius ladder and no limit, the true distance is
> 53.5 tiles, which agrees with the independently measured value from an earlier session.

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

**Network**, 1,965,735 parameters (budget was 10M), no recurrence, no transformer:

| Module | Parameters |
|---|---:|
| Grid encoder and pooling | 712,384 |
| Entity embeddings, MLP, pooling | 283,520 |
| Globals encoder | 4,864 |
| State encoder | 295,552 |
| Autoregressive context and discrete heads | 612,973 |
| Spatial head | 6,905 |
| Entity pointer | 49,537 |

The spatial head conditions the retained 96×96 feature map on verb and prototype
embeddings through FiLM and decodes one Q value per exact tile; it is never masked.
The pointer head scores each live entity embedding against the action context.
Categorical entity fields (type, recipe, item, fluid) go through embeddings rather than
being read as ordered floats.

**Learner**: double DQN, target network every 2000 updates, proportional PER
(alpha 0.6, beta 0.4→1.0), Huber loss, AdamW at 2e-4, batch 128, grad clip 10,
per-head epsilon-greedy exploration (never a uniformly random complete tuple).

## 4. M3 performance

Two numbers dominated every decision tonight.

**Learner throughput.** The first working learner computed its targets and its
autoregressive chain with a Python loop over the batch: **0.40 updates/second** on MPS
at batch 128, raster 96, 2048 entity slots — 51 sample-gradients per second, which would
have wasted the night. Vectorizing the batch (one batched online forward and one target
forward, per-head argmax as tensor ops) cut the per-update accelerator dispatches from
782 to 18 and gave **2.93 updates/second** measured through the real training loop on
MPS — 7.3× faster, 375 sample-gradients per second. That is the rate the runs below
actually used.

**Replay memory.** The first implementation stored 1.07 MiB per transition, so the
nominal 500k capacity from the brief would have needed 521 GiB and even 20k would have
needed 20.8 GiB on a 36 GiB host that already gives 12.8 GiB to Docker. Storing only
live entity rows, bit-packing the binary raster channels, and quantizing the grid to
uint8 with a per-channel affine plus light zlib brought it to **~26 KB per transition**
(25,924 bytes at 2048 entity slots in an early state), a 40× reduction, with a default
capacity of 30,000 costing about 2.7 GiB. The brief's 500k is simply not reachable on
this machine at this observation size, and that is now measured rather than assumed.

**Consequence for the replay ratio.** Environment collection runs at roughly 2.5 semantic
decisions per wall second per worker; the learner sustains 2.9 updates per wall second
shared across a run. A replay ratio of 4 would therefore throttle collection to under
one decision per second and the runs would see five episodes all night. The runs use
**ratio 0.5** — the highest rate that does not starve the environment, as the brief asks.

## 5. Toy results

**TOY-A, semi-Markov correctness** (`tests/smarq/test_smdp_correctness.py`, 10 passed):
`Gamma(tau) = 2^(-tau/H)` exactly, halving per horizon; ten FAST_FORWARD(60 s) discount
identically to one FAST_FORWARD(600 s), so the discount is per simulated second and not
per decision; Bellman targets match hand-computed numbers; internal autoregressive steps
use `Gamma = 1`; two routes to the same factory state differ in value only by the
simulated time they spend; a 3600-tick budget is exhausted by exactly one
FAST_FORWARD(60 s), so waiting cannot buy extra episode time.

**TOY-B, burner automation on a live server**: reported in section 2b. The scripted
expert took the automated score from 0 to 139 in 120 simulated seconds. A second variant
that also places a furnace on the drill's own drop tile is in
`fle/smarq/live_demos.py`; on the open-play map it reliably walks to the ore patch,
places a drill on an exact ore tile and fuels it, and earns its first automated points
within one FAST_FORWARD. The furnace step still fails as `blocked` because the drop tile
computed from the entity row falls inside the 2x2 drill footprint — a demonstration bug,
not an environment one, and the first thing to fix in the demo script.

The remaining TOY-B arms from the brief (random, scripted expert, scratch, demo-seeded,
compared on the same fixed budget) were not run: the night's time went into the two FULL
runs, which the brief gives priority.

## 6. Full results

Both runs are open play on the existing FLE world, reward = delta automated production
score, 30 simulated game minutes per episode, emergency cap 1000 decisions, H = 3600
simulated seconds, raster 96, batch 128, replay ratio 1.0, per-head epsilon annealed
over 5000 decisions, three workers each on their own Factorio server.

| run | reward | seeding | ports | status |
|---|---|---|---|---|
| FULL-1 `full1-scratch` | automated | scratch | 27000-27002 | running |
| FULL-2 `full2-demo` | automated | 6 live burner demonstrations at PER priority 10 | 27003-27005 | running |
| FULL-3 | production (ablation) | scratch | - | not run: all six servers are committed to FULL-1/2, which the brief prioritises |

Results are filled from `tests/benchmarks/smarq_status.py`; every number here is
measured, never projected.

| run | decisions | episodes | max automated | automated / game min | decisions per wall s | updates |
|---|---:|---:|---:|---:|---:|---:|
| _filling as the runs progress_ | | | | | | |

### What the scratch run does early

One thing is visible before any learning claim can be made: the scratch policy
accumulates **production score by hand** - 397 points within its first few hundred
decisions - while its **automated score stays at 0**. That is the brief's manual-grinding
hypothesis appearing unprompted, and it is exactly why the main objective is the
automated score rather than the general one.

## 7. Failure analysis

### What a random policy's actions actually do

Measured over the first 287 decisions of the live runs, by verb:

| Verb | n | Outcomes |
|---|---:|---|
| MOVE_TO | 44 | ok 100% |
| PLACE | 44 | insufficient_inventory 84%, tool_error 9%, blocked 4% |
| RESEARCH | 36 | tool_error 100% |
| FAST_FORWARD | 34 | ok 100% |
| MINE | 31 | tool_error 87%, insufficient_inventory 6%, ok 6% |
| CRAFT | 31 | tool_error 74%, insufficient_inventory 22%, ok 3% |
| PICKUP | 28 | tool_error 85%, ok 10% |
| ROTATE | 26 | insufficient_inventory 96%, ok 3% |
| EXTRACT | 7 | tool_error 57%, no_such_entity 43% |
| INSERT | 6 | insufficient_inventory 100% |

Two different things are visible here and they should not be confused.

**The environment is behaving correctly.** A uniformly random policy picks a prototype it
does not carry (84% of PLACE), mines tiles with nothing on them (87% of MINE), crafts
recipes whose ingredients it lacks, and picks up empty ground. These are exactly the
failed transitions the brief asks for: unmasked, zero-reward, and informative. Locomotion
and simulated waiting, the two things the environment is allowed to handle, succeed 100%
of the time.

**The failure taxonomy is miscalibrated.** `tool_error` is a catch-all for game messages
the classifier does not recognise, and it is swallowing failures that have proper names:
CRAFT without ingredients is `insufficient_inventory`, MINE on bare grass is an invalid
target, RESEARCH without science packs cannot succeed at all. ROTATE is worse than
uninformative — 96% of its failures are labelled `insufficient_inventory` because the
game's message contains a substring the inventory rule matches. This does not affect
learning (every failure earns the same ~0 reward either way) but it does affect what can
be read off the logs, and it should be fixed before the next run by matching on the
game's message catalogue rather than on substrings.

### Exploration is the binding constraint, and it is quantifiable

The chain to first automated reward is: hop toward ore (one MOVE_TO), select PLACE,
select `burner-mining-drill` from 87 placeable prototypes, select one of a few hundred
ore tiles out of 9,216 positions, then INSERT coal into that specific entity. Under
uniform per-head exploration the placement alone is roughly
(1/11)·(1/87)·(300/9216) ≈ 3·10⁻⁵ per decision, and it still has to be followed by the
right INSERT. At the measured ~1.4 decisions per wall second, a scratch run of a few
hours is not expected to stumble into it. That is the honest prior for FULL-1, and it is
the reason FULL-2 seeds replay with demonstrations of exactly that chain.

### A configuration error worth recording

The first launch used the default per-head epsilon schedule, which anneals over 100,000
decisions (200,000 for the position head). A run that collects ~10,000 decisions would
therefore have stayed above epsilon 0.9 for its whole life: it would have measured
exploration, not learning. Both runs were stopped after ~25 minutes and relaunched with
the decay matched to the decisions the run will really collect (`--epsilon-decay-decisions
5000`), so roughly the second half of each run acts on its Q-values.

## 8. Verdict

_Chosen when the runs stop; the evidence that will decide it is stated here in advance so
the choice is not made to fit the story._

- **A, clear swim**: a scratch or demo-seeded policy reaches automated production on its
  own and its automated score per game minute rises over training.
- **B, weak swim**: the demo-seeded policy repeats the automation chain after the
  demonstrations, even partially (drill placed on ore and fuelled), more often than
  chance.
- **C, infrastructure works, learning unresolved**: the interface and the SMDP are
  demonstrably correct - TOY-A passes, TOY-B automates, the learner learns on the toy
  environment - but neither live run shows automation within the compute available.
- **D, drown**: something in the formulation is broken rather than merely slow.

### What would be worth running next

The exploration arithmetic in section 7 makes the prediction sharp: the first automated
reward sits behind a five-decision conjunction whose random probability is about
3 in 100,000 per decision. Two experiments follow from that, and they are cheap:

1. **Start distribution, not algorithm.** Sample each episode's start beside an ore patch
   (the TOY-B lab setup, which needs only a teleport at reset) so the agent faces the
   automation rung without the travel prefix. If SM-ARQ learns the rung there and not in
   open play, the bottleneck is reaching ore, and the fix is a curriculum over start
   positions rather than a different learner.
2. **Make the demonstrations dense and let n-step carry them.** Six demonstrations is
   about 60 transitions in a 30,000-entry replay. Collecting a few hundred, and turning
   on the n-step semantic returns that are already implemented behind a flag, would put
   real weight behind the only trajectories that ever see reward.

Both are configuration changes to what already exists, which is the point of having
built the interface first.
