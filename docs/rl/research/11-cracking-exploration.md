# Cracking exploration in empty-start open play

Written 2026-09-14 against branch `feat/pufferlib-rl`. Read-only analysis of
`docs/rl/results/`, `fle/rl/`, `fle/env/tools/`, and the saved archive at
`.fle/runs/go_explore_pure/archive.pkl.gz`. No code was changed, no container
was started, nothing was trained.

The short version: the run logs say the agent has never acquired an item, and
they also say why. It is not credit assignment, it is not reward sparsity, and
it is only partly exploration. Two measured factors multiply to roughly
2 in 100,000 per action, and one of the two is a property of the action space
that can be removed by making the existing mask machinery tell the truth about
what the simulator will accept.

---

## 1. What the logs actually say

### 1.1 Zero harvests, ever, across every unscripted run

`harvest_resource` is the only operation that can put the first item in an
empty inventory. Across every unscripted open-play run in the repository it has
succeeded zero times.

| Run | HARVEST ok / attempts |
|---|---:|
| Go-Explore, pure, 296 cells (`go_explore_pure_posonly_296cells.jsonl`) | 0 / 1,800 |
| PPO open play (`ppo_open_play.jsonl`) | 0 / 4,197 |
| Go-Explore, no frontier term | 0 / 245 |
| Go-Explore, 16-tile bucket | 0 / 60 |
| Go-Explore, one cell | 0 / 156 |
| Go-Explore, reach 8 | 0 / 166 |
| Go-Explore, short excursions | 0 / 61 |
| Go-Explore, both scaled | 0 / 54 |
| Go-Explore, pure (35 s) | 0 / 13 |
| **Total unscripted** | **0 / 6,752** |

For contrast, the two runs seeded by the scripted policy harvested 41 times in
2,346 attempts, or 1.75 percent. The scripted seed's entire contribution at
this stage was standing the player next to ore. It did not teach crafting or
placement, it fixed geometry.

### 1.2 The error messages name the cause

In the 296-cell pure run, 1,797 of the 1,800 HARVEST attempts were rejected by
the game with the message `Could not harvest. Nothing within reach to harvest`.
The other 3 were rejected with `Given limit value is too small`.

`fle/env/tools/agent/harvest_resource/server.lua:469-472` gates on
`distance > player.resource_reach_distance` before it looks for anything at the
target tile. Factorio's default character `resource_reach_distance` is 2.7
tiles. The schema's offset head spans `dx, dy` in `[-8, 8]`, which is 289
positions, and only 21 of those 289 lie inside a radius of 2.7. So 92.7 percent
of the offsets available to HARVEST are rejected before the game even checks
whether a resource is there.

The remaining 3 attempts got past both the reach gate and the resource lookup
and were then killed by `QUANTITY_BINS[3] = -1`, the "all" bin, which
`find_entities_filtered` rejects as a limit. That bin is dead for HARVEST and,
per `candidate_scoring.md`, also for INSERT. It costs 25 percent of every
quantity draw.

### 1.3 Half of every action budget is provably null

`fle/rl/masks.py` builds masks from entity-row presence and the allowed
operation list only. Every other head is all ones. Nothing in the mask says
that an operation has no valid completion in the current state.

Outcome breakdown for the 21,540 steps of the 296-cell run:

| Outcome | Steps | Share |
|---|---:|---:|
| `preflight_invalid` | 10,835 | 50.3% |
| `tool_rejected` | 6,558 | 30.4% |
| `ok` | 4,147 | 19.3% |

Every one of the 10,835 preflight failures carried the single error
`real entity anchor required`. INSERT, EXTRACT, PICKUP, ROTATE, SET_RECIPE and
CONNECT all require a real entity anchor, and the world had no entities. The
sampler in `fle/rl/go_explore.py:319-330` picks the operation first, then finds
that the operation has no legal anchor, and burns the step.

Of the 19.3 percent that the game accepted, most changed nothing that matters.
PLACE was 0 / 1,742 because the inventory was empty. CRAFT reported `ok` 402
times, and all 295 archived cells still had an empty inventory set, so those
crafts produced nothing (`craft_item` returning 0 is not `False` or `None`, so
`fle/rl/action.py:183` accepts it). RESEARCH reported `ok` 106 times with no
lab and no science packs in the world. The only operations that did anything
real were MOVE (1,803 of 1,803) and WAIT (1,836 of 1,836).

So 16.7 percent of the action budget could conceivably have changed the world,
and all of it was movement or time.

### 1.4 Three quarters of the wall clock is spent waiting in an empty world

Per-operation wall-clock, measured from consecutive `elapsed_s` deltas within
each excursion of the 296-cell run:

| Operation and outcome | n | median s | total s | share of clock |
|---|---:|---:|---:|---:|
| WAIT / ok | 1,836 | 0.607 | 3,920.0 | **73.8%** |
| MOVE / ok | 1,803 | 0.172 | 651.3 | 12.3% |
| HARVEST / tool_rejected | 1,800 | 0.062 | 114.1 | 2.1% |
| ROTATE / preflight_invalid | 1,867 | 0.037 | 71.1 | 1.3% |
| INSERT / preflight_invalid | 1,827 | 0.037 | 69.6 | 1.3% |
| PLACE / tool_rejected | 1,732 | 0.039 | 69.4 | 1.3% |
| RESEARCH / tool_rejected | 1,695 | 0.040 | 69.0 | 1.3% |
| PICKUP / preflight_invalid | 1,818 | 0.037 | 68.9 | 1.3% |
| EXTRACT / preflight_invalid | 1,777 | 0.036 | 67.2 | 1.3% |
| CONNECT / preflight_invalid | 1,774 | 0.037 | 67.2 | 1.3% |
| SET_RECIPE / preflight_invalid | 1,762 | 0.037 | 66.8 | 1.3% |
| CRAFT / tool_rejected | 1,331 | 0.042 | 56.7 | 1.1% |
| CRAFT / ok | 402 | 0.043 | 18.0 | 0.3% |
| RESEARCH / ok | 106 | 0.041 | 4.4 | 0.1% |
| PLACE / preflight_invalid | 10 | 0.040 | 0.4 | 0.0% |

`DURATION_BINS = (1, 5, 15, 60)` target game seconds at `game.speed = 10`, so a
uniform WAIT costs 2.03 real seconds on average. The measured mean was 2.135.
The one operation that can produce the first item, HARVEST, received 2.1
percent of the clock. A ninety minute Go-Explore run is really a twenty-three
minute run with sixty-five minutes of sleep attached, and the sleep buys
nothing because nothing in the world is running.

Restore, by contrast, is nearly free: median 0.131 s over 337 restores, which
is 0.8 percent of the run. Restore is not the bottleneck, and any proposal that
trades more restores for fewer actions is affordable.

### 1.5 The archive is a map of walked ground

All 295 discovered cells in the 296-cell run had an empty `inventory` set, an
empty `built` set and an empty `automatic` set. Every discovery was a position
change. The cell function has six components and exactly one of them ever
moved. Discovery rate was 0.88 new cells per 64-action excursion, and 45
percent of excursions produced at least one.

### 1.6 The measurement that settles the diagnosis

I decoded the `iron_ore`, `copper_ore`, `coal`, `stone` and `uranium_ore`
channels of the stored fine grid for all 21,540 observations in
`.fle/runs/go_explore_pure/archive.pkl.gz`:

| Condition at a decision boundary | Boundaries | Share |
|---|---:|---:|
| Any resource tile anywhere in the 32x32 fine window (plus or minus 16 tiles) | 362 | 1.7% |
| Any resource tile inside the plus or minus 8 offset window | 194 | 0.9% |
| Any resource tile within harvest reach, radius 2.7 | 95 | 0.4% |

Among the 95 boundaries where a harvestable tile was in reach, the mean number
of in-reach resource tiles was 11.0 out of the 21 in-reach offsets. The
distribution was bimodal: 36 boundaries were standing fully inside a patch with
all 21 in-reach offsets on ore, and 36 were on the very edge with exactly 1.

That gives a clean factorization of the failure:

```
P(successful harvest per action)
  = P(a resource is within reach)              = 95 / 21,540      = 0.44%
  x P(op = HARVEST)                            = 1 / 12           = 8.3%
  x P(offset lands on an in-reach ore tile)    = 11.0 / 289       = 3.8%
  x P(quantity is not the dead -1 bin)         = 3 / 4            = 75%
  = 1.05e-5 per action
```

Times 21,540 actions, that predicts **0.23 expected successful harvests per
ninety minute run**. The observed count is 0, with 3 near misses that reached
the resource lookup. The model and the data agree.

---

## 2. Is the action space the blocker, or is it exploration?

Both are, and the factorization above says exactly how much each contributes
and which is cheaper to remove.

- The **coverage factor**, `P(a resource is within reach) = 0.44%`, is an
  exploration problem. The archive spends 99.6 percent of its boundaries
  nowhere near a resource.
- The **action factor**, `P(the sampled action is a valid harvest of an
  in-reach ore tile | a resource is in reach) = 0.24%`, is an action-space
  problem. It is a product of three independent losses: operation dilution over
  12 operations of which 6 are provably impossible, an offset head where 92.7
  percent of values are outside the game's reach gate, and a quantity bin the
  tool rejects.

Removing the action factor alone raises it from 0.24 percent to roughly 33
percent, a factor of about 140. Applied to the same 95 in-reach boundaries that
the existing archive already produced, that predicts **about 31 successful
harvests per ninety minute run instead of 0.23**. That is enough stone for a
furnace, and enough iron and coal to feed it.

Removing the coverage factor alone cannot do the same job. To get from 0.23
expected harvests to 30 through coverage alone, the agent would have to stand
within 2.7 tiles of a resource for 57 percent of all boundaries instead of 0.44
percent. No selection rule delivers that on a map where only 1.7 percent of
boundaries even see ore inside a 32-tile window.

So the honest answer to the question is: the action space is the blocker that
is currently cheapest to remove by two orders of magnitude, and coverage
becomes the binding constraint immediately afterwards. Both need work. Do the
action space first because it is a one-day change with a predicted 140x effect
on the measured bottleneck, and because until it is fixed no exploration
experiment can be interpreted. Every exploration result so far has been
measured through a 0.24 percent action-space filter, which means every one of
them was underpowered by a factor of 140.

### 2.1 The experiment that distinguishes them in ninety minutes

Run the existing Go-Explore loop with the support masks of proposal A and
nothing else changed. Same seed, same map, same cell function, same 64 uniform
actions per excursion, same selection weights, same duration bins.

- If HARVEST `ok` count rises above zero and at least one archived cell has a
  non-empty `inventory` set, the action space was the blocker.
- If the `ok` rate rises above 90 percent but HARVEST `ok` stays at 0, the
  action space was fixed and coverage is the blocker. Go to proposals C and D.
- If the `ok` rate does not rise above 90 percent, the mask implementation is
  wrong, not the hypothesis.

Primary endpoint: count of steps with `op == "HARVEST"` and
`outcome == "ok"`. Current value 0 of 1,800. Prediction under the change: 20 to
60. A result of 0 refutes proposal A outright.

---

## 3. The bitter-lesson boundary

The user has ruled out scripted bootstrapping and hand-written action-admission
rules. Proposal A sits on that line, so it deserves an explicit argument rather
than a quiet assumption.

There are two different things called a mask.

1. **Semantic admission**, which encodes a solution. "Place a burner drill on
   iron facing the chest." "Craft a stone furnace before a drill." This is
   scripting and it is correctly rejected. It transfers the designer's plan
   into the agent.
2. **Support**, which encodes what the simulator will deterministically refuse.
   "There is no entity, so INSERT has no legal anchor." "The tile is 7 tiles
   away and the reach gate is 2.7." "The inventory is empty, so no prototype can
   be placed." This encodes no preference at all. It only stops the sampler
   from proposing actions whose outcome is already known to be a no-op, and it
   says nothing about which of the remaining actions is good.

The repository already relies on category 2. `fle/rl/masks.py` masks anchor and
peer rows by entity presence, for exactly this reason. The proposal is to
finish that job, not to start a new kind of thing.

The literature treats this as table stakes rather than as a shortcut. Huang and
Ontanon (2020), "A Closer Look at Invalid Action Masking in Policy Gradient
Algorithms", show in microRTS that invalid action masking is not a minor
implementation detail: without it, policy gradient methods fail to learn at all
in large factored action spaces, and the gap grows with action space size.
Kanervisto, Scheller and Hautamaki (2020), "Action Space Shaping in Deep
Reinforcement Learning", measure across ViZDoom, Minecraft and StarCraft that
removing actions that cannot be used is often the difference between learning
and not learning. AlphaStar and OpenAI Five both mask structurally illegal
actions and neither is described as a scripted agent.

There is one constant in the proposal that is genuinely hand-written: the reach
radius 2.7. Do not hard-code it. Read `player.resource_reach_distance` and
`player.build_distance` from the server once at startup, the same way
`fle/rl/vocabulary.py` reads the prototype catalog. Then nothing in the mask is
a designer's number.

If even that is unacceptable, proposal G gives the learned version of the same
hypothesis, at roughly three times the implementation cost. My recommendation
is to run proposal A first as a diagnostic, because it answers the question in
one ninety minute slot, and then decide whether the final system keeps the mask
or replaces it with the learned head.

---

## 4. Proposals, ranked by expected value per unit of effort

Effort estimates assume familiarity with the existing code.

### A. Operation-conditional support masks. Effort: about one day. Rank 1.

**Mechanism.** Extend `fle/rl/masks.py` from a state-only mask to an
operation-conditional support mask, and make the operation mask itself a
function of whether any complete action exists for that operation.

Per head, using state the `ObservationClient` already holds:

| Head | Support rule | Source |
|---|---|---|
| `op` | Drop an operation if any required head has empty support | derived |
| `anchor`, `peer` | Already correct | `state.entities` |
| `offset` for HARVEST | Euclidean length at most `resource_reach_distance`, and `(x, y)` present in `state.ores` or the tree set | `state.ores`, server constant |
| `offset` for PLACE | Within `build_distance`, footprint not occupied | `state.entities`, `state.obstacles` |
| `prototype` | Count greater than zero in inventory | `state.inventory` |
| `item` for CRAFT | Recipe unlocked and ingredients present | `state.inventory`, `state.tech` |
| `item` for INSERT | Present in inventory | `state.inventory` |
| `recipe` | Unlocked | `state.tech` |
| `technology` | Not researched and prerequisites satisfied | `state.tech` |
| `quantity` | Drop `-1` where the tool rejects it; drop bins above the held count | tool contract, `state.inventory` |
| `direction`, `duration` | Never the sentinel when the operation reads them | schema |

Also fix the live sampler bug. `fle/rl/go_explore.py:325` reads
`chosen["op"] = int(rng.integers(len(OPERATIONS)))`, which ignores
`masks["op"]` entirely. `env.allowed_operations` is honoured by `build_masks`
and then discarded by `random_action`. That is a two-line fix and it is a
prerequisite for any operation-level masking to have an effect.

**Why this failure and not exploration in general.** It is aimed directly at
the three measured multipliers in section 1.6: operation dilution (12 to 3 at
the empty start), the 92.7 percent out-of-reach offsets, and the dead quantity
bin. It does not touch the reward, the cell function, the selection weights or
the excursion policy.

**Cost.** About 100 to 150 lines in `masks.py`, an operation argument threaded
through `ObservationClient.encode`, one server read for the two reach
constants, and about 10 lines in `go_explore.py`. The observation layout does
not change, so no schema fingerprint bump and no replay invalidation. Existing
`tests/rl/test_observation_action.py` coverage will need extending.

**Ninety minute measurement.** Endpoint: HARVEST `ok` count. Baseline 0 of
1,800. Prediction 20 to 60. Secondary: overall `ok` rate, baseline 19.3
percent, prediction above 90 percent; `preflight_invalid` rate, baseline 50.3
percent, prediction below 2 percent; archived cells with a non-empty
`inventory` set, baseline 0, prediction at least 1.

**Free pre-test, no Docker, minutes not hours.** The stored archive already
contains every observation from the 296-cell run. Build the mask function,
apply it offline to those 21,540 observations, and report the support set size
per head per boundary and the fraction of boundaries with a non-empty HARVEST
offset support. Section 1.6 says the answer should be 95 of 21,540, or 0.44
percent. If the offline mask says something different, the implementation is
wrong before any container starts. Run this before booking a ninety minute
slot.

### B. Stop paying for WAIT. Effort: hours. Rank 2.

**Mechanism.** Two independent levers.

First, raise `game.speed`. It is 10 in `run_go_explore.py:73`, and
`fle/agents/data/run_trace.py:40` already uses 20. WAIT's real cost is linear
in `1 / speed`. At speed 40 the mean WAIT falls from 2.135 s to 0.53 s, the
clock cost of WAIT falls from 3,920 s to 980 s, and the same ninety minutes
buys 2.24 times as many actions.

Second, restrict the WAIT duration support. When no entity in the world is in a
dynamic state, no amount of game time changes anything except the tick counter,
so only the shortest bin is supported. This is the same support argument as
proposal A applied to time.

**Why this failure.** Section 1.4. The single largest consumer of the
exploration budget is an operation that provably cannot change an empty world.
Compounded with proposal A, HARVEST attempts per ninety minutes go from 1,800
to roughly 16,000, which is a 9x on attempts on top of the 140x on per-attempt
success.

**Cost.** The speed change is one line plus a correctness gate. The duration
support is a few lines inside the same mask work as proposal A.

**Measurement.** A five minute throughput probe at `game.speed` 10, 20, 40 and
60, reporting actions per second, mean WAIT wall cost, and a correctness check
that a fuelled drill's automatic production counter advances by the expected
amount per game second at each speed. Endpoint: the highest speed at which the
counter stays exact.

**Uncertainty I want flagged.** I do not know the server's sustainable UPS
ceiling, and several FLE tool clients contain fixed `sleep(0.5)` long-poll
loops (`craft_item/client.py:53`, `harvest_resource/client.py:67`) that do not
scale with game speed. Above some speed those loops, not the simulation, will
set the floor. The probe measures this rather than assuming it.

### C. Weight archive selection by affordance, not by spatial frontier. Effort: half a day. Rank 3.

**Mechanism.** The current weight in `fle/rl/go_explore.py:244-249` includes
`(1 + 2 * (8 - neighbours))`, which ranges from 1 for a fully surrounded cell
to 17 for a cell with no archived neighbours. Every one of the 295 cells in the
pure run was a pure position cell, so this term is a pure spatial heuristic
operating on a purely spatial archive. It pushes selection towards empty ground
by construction.

That is the wrong direction for this problem. An ore-adjacent cell sitting
inside explored ground receives one seventeenth the weight of an empty frontier
cell, even though it is the only kind of cell from which the first item can be
obtained.

Replace the frontier term with an affordance term: weight a cell by the number
of distinct supported actions available from it that have never been executed
successfully anywhere in the archive. Under proposal A that number is computable
from the stored observation with no extra game calls. "Explore from where you
can do something nobody has done" is a criterion read from the observation, not
a plan.

**Why this failure.** Section 1.6 says the archive reached in-reach-of-ore
boundaries 95 times out of 21,540 and then walked away from them. Once
`REACH_TILES = 3` puts reachable resource types into the cell signature, those
become distinct cells, and the selection rule decides whether they get revisited
or drowned by 290 empty position cells.

**Cost.** Roughly 40 lines in `Archive.select`, plus the support-set size cached
on `ArchiveEntry` at capture time.

**Measurement.** Fraction of excursion starts whose cell has a non-empty
`reachable` set. Baseline from the current weights is unmeasured because the
296-cell run predates the `reachable` component, so measure it in the same run
as the control arm. Prediction: above 20 percent with the affordance term,
below 5 percent with the frontier term.

**Uncertainty.** The frontier term's actual effect is currently unmeasured. The
`go_explore_nofrontier.jsonl` run and the pure run are identical to the cell at
150 s, 300 s and 600 s (20, 24 and 57 cells in both), and the no-frontier run
ended at 626 s. So the existing logs neither support nor refute the frontier
term. Do not describe removing it as a fix to a known problem. It is an
untested heuristic that points the wrong way for this specific task.

### D. Put long-range resource information into the observation. Effort: one to two days. Rank 4.

**Mechanism.** `observation_diff.lua` exports ore tiles for every generated
chunk, so `client.ores` holds map-wide resource locations. The encoder then
throws almost all of it away: the fine grid renders resources only within plus
or minus 16 tiles, and `COARSE_CHANNELS` contains `known`, `entity_density`,
`working_fraction`, `no_power_fraction`, `starved_fraction` and
`output_blocked_fraction`, with no resource channel at all.

The consequence is measured in section 1.6: at 98.3 percent of boundaries the
observation contained no resource information whatsoever. The nearest iron on
this map is 53 tiles from spawn per the comment at `go_explore.py:22-24`. The
agent is doing a blind random walk because the observation gives it nothing to
walk towards.

Add resource density channels to the coarse grid, which covers plus or minus 64
tiles at 4-tile resolution. Five channels, one per resource type, 5 x 32 x 32 =
5,120 values on a 22,817-value observation.

**Why this failure.** No exploration algorithm can be directed towards
something the observation does not contain. This is the reason the archive
ladders outward as an undirected disc.

**Cost.** This is the most expensive proposal in the top five. It changes
`ObsSchema.blocks`, so the fingerprint changes, so every existing replay shard
and checkpoint is invalidated. The encoder change itself is small, and the data
is already in `client.ores`.

**Cheaper variant, if the schema churn is unwelcome.** Add a nearest-resource
lane to `globals`: for each of five resource types, relative `dx`, `dy` and log
distance to the nearest known tile. That is 15 floats computed directly from
`client.ores`. It is more hand-picked than a density channel, which is a real
cost against the bitter-lesson standard, but it fits in the existing `GLOBALS`
block if there is room and is an hour of work.

**Measurement.** Over a ninety minute run, the fraction of boundaries with a
resource within harvest reach. Baseline 0.44 percent. Prediction above 5
percent with coarse resource channels plus proposal C. Also report mean distance
from the player to the nearest known resource tile over time; under a blind
walk it should be flat, under a directed one it should trend down.

### E. Counts in the cell, not just sets. Effort: one hour. Rank 5.

**Mechanism.** `Cell.inventory` is a `frozenset` of item names with count
greater than zero. One stone and five hundred stone are the same cell. A stone
furnace needs five stone. So the archive has no rung between "obtained a
resource" and "obtained enough of it", which is exactly the rung the first
furnace needs.

Replace the inventory set with a set of `(name, log bucket)` pairs, with
buckets at 1, 2 to 4, 5 to 19, 20 to 99, 100 and above. Do the same for the
`automatic` component once production starts.

**Why this failure.** It is the cheapest change in this document and it targets
a specific structural gap in the ladder rather than exploration in general.

**Cost.** Five lines in `cell_from_client`, plus a line in `cell_record`.

**Risk.** It multiplies the archive. With position already generating 296 cells,
adding count buckets could produce cell explosion. Mitigate by ranking the
components: make progress components primary and subsample position cells within
a progress signature. Ecoffet et al. (2021) handle the same problem by
downsampling the observation until the archive size is manageable, and they
report that the archive size and the cell granularity are the two parameters
that matter most.

**Measurement.** Total cells, and cells broken down by which component is
non-trivial. Endpoint: at least one cell with an inventory bucket of 5 or more.

### F. Short excursions instead of 64 uniform actions. Effort: one hour. Rank 6.

**Mechanism.** `actions_per_segment` is already a parameter of `explore()`.
Restore costs 0.131 s median and a non-WAIT action costs about 0.070 s, so an
excursion of 4 actions carries only 32 percent restore overhead and an
excursion of 8 carries 19 percent. The current configuration spends 337
archive-guided selection decisions per ninety minutes. At 8 actions per
excursion, and with proposals A and B in place, the same clock buys roughly
4,000.

The first rung of this ladder is a single action: harvest an ore tile you are
already standing next to. For a problem whose first rung is one action deep,
breadth beats depth, and the archive already stores the depth.

**Better rule than a fixed length.** Terminate an excursion as soon as it
produces a discovery on a progress component (inventory, built, automatic,
researched), and continue through position-only discoveries, capped at 64. That
returns control to the archive exactly when something interesting happened, and
keeps depth available for multi-action sequences like harvesting five stone.

**Why this failure.** Section 1.5 says 0.88 discoveries per 64-action
excursion, and every single one was positional. A 64-action uniform random walk
from a restored cell is mostly a random walk back and forth over the same
ground.

**Cost.** One parameter and about 10 lines for the adaptive rule.

**Measurement.** Progress-component discoveries per wall-clock second, compared
at excursion lengths 4, 8, 16 and 64 in four 20-minute arms within one ninety
minute slot. Report both total cells per second and progress cells per second,
because they will disagree.

### G. A learned admissibility head. Effort: two to three days. Rank 7.

**Mechanism.** Every transition already carries `info["action_outcome"]` in
`{ok, tool_rejected, preflight_invalid, partial_mutation}`, and it is stored in
replay via `EpisodeTrace.action_outcomes`. Train a head to predict the outcome
class from the observation and the complete action tuple. Use its predicted
`P(ok)` to reweight the behaviour sampler, or as a learned mask with a
threshold.

**Why this failure, and why it is worth listing separately.** It tests exactly
the same hypothesis as proposal A, and it does so with no hand-written rules at
all. It is the bitter-lesson-clean version. The supervised signal is dense: the
296-cell run alone provides 21,540 labelled examples at a 19.3 percent positive
rate, which is four orders of magnitude denser than the reward. It also
generalises to cases proposal A's hand-written support cannot cover, such as
collision geometry and placement centre parity, which the server owns.

**Why it is rank 7 and not rank 1.** It costs three times as much to build, it
introduces a second network and a second loss, its errors are silent, and it
cannot be pre-tested offline in minutes the way proposal A can. Build it after
proposal A has told you whether the hypothesis is right. If proposal A works and
the user wants the hand-written support removed from the final system, this is
the replacement.

**Measurement.** Held-out AUC on outcome prediction from the existing archive,
measured offline with no container. Then, live, the HARVEST `ok` count under a
sampler reweighted by predicted `P(ok)`, against the proposal A arm.

### H. Count novelty over supported action identity, conditioned on the cell. Effort: half a day. Rank 8.

**Mechanism.** Weight the exploration sampler by
`1 / sqrt(1 + N(cell signature, action identity))` over the support set from
proposal A, where the action identity is the complete tuple's stable identity
rather than just the operation.

**Why this specific form.** The E1 arm in `exploration.md` is the single
largest measured effect in the whole history of this project: on-ore placements
17 of 80 to 68 of 80 with `p = 1.9e-16`, working drills 13 of 80 to 47 of 80
with `p = 3.8e-8`, and the first two four-ore successes ever observed from
scratch. E1 was count novelty over `(stage, operation)` plus inverse-square-root
counts over observed ore tile and direction tuples. This proposal is E1
generalised from the L2 fixture to open play, with the cell signature in place
of the hand-written four-bit stage.

**One design point that matters.** Count successes, not attempts. Under a
count-over-attempts rule, HARVEST has been tried 1,800 times and succeeded 0
times, so its novelty bonus would be near zero, which is exactly backwards. The
right counter is over actions that have never been executed successfully from a
cell like this one.

**Cost.** About 40 lines, replacing the uniform draw in
`go_explore.random_action`. It depends on proposal A for the support set.

**Measurement.** Distinct successfully-executed complete action identities per
ninety minutes, and HARVEST `ok` count, against a proposal-A-only control arm.

### I. Backward curriculum from the agent's own best trajectory. Effort: two to three days. Rank 9.

**Mechanism.** Once the archive holds any trajectory that reaches automatic
production, take that trajectory's stored states and start the learner from
state index `k` near the end. When success rate from `k` exceeds a threshold,
decrement `k`. This is Salimans and Chen (2018), "Learning Montezuma's Revenge
from a Single Demonstration", and it is the same idea as the robustification
phase of Ecoffet et al. (2021) and as reverse curriculum generation (Florensa
et al. 2017).

**Why it does not violate the constraint.** The trajectory comes from the
agent's own exploration, not from a human or a scripted policy. The mechanism
requires only `restore()`, which is already implemented and costs 0.131 s.

**Why it is rank 9.** It has no input yet. The only trajectory in the archive
that reaches automatic production is the scripted-seed one
(`go_explore_run1_23cells.jsonl`, automated score 35.0), and robustifying that
would inherit the scripting. This proposal becomes rank 2 the day proposal A
produces a self-discovered production trajectory, and it is worth building the
scaffolding in parallel for that reason.

**Measurement.** Start index `k` as a function of wall clock, and success rate
from `k`. A working backward curriculum shows `k` walking monotonically towards
0. A stalled one shows `k` stuck at the first index that needs a genuinely new
discovery.

### J. Goal-conditioned return, and cell-relabelled hindsight. Effort: high. Rank 10.

**Mechanism.** Policy-based Go-Explore replaces `restore()` with a
goal-conditioned policy trained to reach an archived cell, with the goal drawn
from the archive's own cell signature rather than from a hand-specified item and
threshold. Failed attempts are relabelled with the cell actually reached, which
is hindsight relabelling over a goal space the agent constructed. Related work:
Ecoffet et al. (2021) policy-based variant, Guo et al. (2020) "Memory Based
Trajectory-conditioned Policies for Learning from Sparse Rewards", and Ghosh et
al. (2021) "Learning to Reach Goals via Iterated Supervised Learning".

**Why this is explicitly not the rejected machinery.** The rejected design was
a hand-picked goal item plus a pass or fail threshold plus future HER on that
threshold, and `SUMMARY.md` correctly retires it on the grounds that the
production score is the metric. Here the goal is a cell the archive discovered,
the reward is "did you reach it", and nothing about the production-score
objective changes.

**Why it is rank 10 anyway.** The thing it buys is a return mechanism that
works when you cannot restore the simulator. This repository can restore the
simulator, reliably, in 0.131 s, which is half the cost of one average action.
Spend the effort on the explore phase, not on replacing a return phase that
already works. Revisit this only if restore reliability degrades or if the goal
becomes transfer to maps where the archive cannot be replayed.

### K. Intrinsic motivation over raw observations: RND, ICM, NGU. Effort: medium. Rank 11, predicted near zero value.

I am listing these to argue against them, with reasons specific to this
observation and this failure.

**RND** (Burda et al. 2018) gives a bonus for states where a random target
network is poorly predicted. The observation here is 22,817 values dominated by
a 12 x 32 x 32 player-centred fine grid and a 6 x 32 x 32 player-centred coarse
grid. Both re-render on every MOVE. RND's bonus would therefore be maximised by
moving to unseen terrain, which is precisely what the archive already achieves
for free and which section 1.5 shows produces only position cells. Worse, 80.7
percent of actions produce a next observation nearly identical to the current
one, because they were rejected. RND assigns those a bonus near zero, which is
correct and useless: it cannot tell you which action would have worked.

**ICM** (Pathak et al. 2017) has a sharper version of the same problem. Its
inverse model predicts the action from `(s, s')`. With 80.7 percent of actions
mapping to `s' ~= s`, the inverse model is unidentifiable over most of the
action space, and the forward-model error will be dominated by terrain scroll
during MOVE. I predict ICM concentrates its entire bonus on movement.

**NGU and Agent57** (Badia et al. 2020) use an episodic k-nearest-neighbour
bonus in an inverse-dynamics embedding, chosen so the embedding keeps only what
the agent controls. Here what the agent controls, at the empty start, is
position and nothing else. So the embedding collapses to position and NGU
reduces to the archive's existing behaviour.

The general pattern: every novelty measure defined over the raw observation
will rediscover position novelty, because position is the only dimension of the
observation the agent can currently move. This is not a hypothesis, it is a
restatement of section 1.5, where 295 of 295 discovered cells were positional.

**What this implies.** Novelty over the achievement tree, meaning the
`automatic` and `built` components, is degenerate for the opposite reason: it
has zero reached nodes and no gradient. Count-based novelty over the *supported
action set* (proposal H) is the only intrinsic-motivation variant I would fund,
and it is the only one with a large measured effect in this project's own
history.

### L. Skill discovery: DIAYN, DADS, LSD. Effort: high. Rank 12, predicted near zero value.

DIAYN (Eysenbach et al. 2019) maximises the mutual information between a latent
skill and the visited states under a learned discriminator. The discriminator
will separate skills along whichever state dimension varies most, and section
1.5 says that dimension is position, unanimously. DIAYN would therefore
rediscover "walk in eight directions". Skill discovery needs a state abstraction
that already ignores position, and if you have that abstraction you may as well
put it directly in the cell function, which costs a day instead of a month.

### M. Search over the restorable simulator. Effort: high. Rank 13, uncertain.

Worth naming because the restorable simulator is an unusual asset. At 0.131 s
per restore and 0.070 s per non-WAIT action, a 100-node lookahead costs about
ten seconds per decision. That is far too slow for online control, but it is
viable as an offline target generator: run a search from archive cells, store
the resulting trajectories, and train on them. Sampled MuZero (Hubert et al.
2021) is the closest fit for large factored action spaces, since it samples a
subset of actions at each node rather than enumerating them.

I am genuinely uncertain about this one. Its value depends entirely on whether
the branching factor after proposal A's support masks is small enough for search
to be informative, and that number is not yet measured. The offline pre-test in
proposal A produces exactly that number as a side effect: the mean support set
size per boundary. If it comes back under about 50 for the empty start, search
becomes interesting. If it comes back in the hundreds, it does not.

---

## 5. What I would do first, and why

**Proposal A: operation-conditional support masks, with the offline pre-test
run before any container is started.**

The reasons, in order:

1. It is the only proposal that addresses the measured 0 out of 6,752. Every
   other proposal in this document improves something that is currently being
   measured through a 0.24 percent action-space filter, so none of them can be
   evaluated until this one lands.
2. The predicted effect is 140x on the per-attempt success rate, giving
   about 31 expected harvests per ninety minutes against the current 0.23,
   using only the coverage the existing archive already achieved.
3. It can be pre-tested offline in minutes against 21,540 stored observations,
   with no Docker and no Factorio, so the ninety minute slot is only spent once
   the mask has been shown to behave as predicted.
4. It is about a day of work in one file, and it does not change the
   observation layout, so no replay shard or checkpoint is invalidated.
5. It is falsifiable cleanly: HARVEST `ok` count of 0 refutes it outright.

Run proposal B in the same slot, because raising `game.speed` is one line and
the WAIT duration support falls out of the same mask work. They are
independent, and their effects multiply.

If the combined run produces items, the next slot is proposals C, E and F
together, which are the archive-side changes and total about a day. If it
produces items but no automatic production, the slot after that is proposal I,
the backward curriculum from whatever trajectory the agent found.

---

## 6. Things I am uncertain about

- **The reach constant.** I used Factorio's default character
  `resource_reach_distance` of 2.7 tiles. FLE may configure a different value,
  and the 3 attempts that reached the resource lookup are slightly more than the
  0.6 my model predicts, which is weak evidence that the effective reach is
  larger or that trees and rocks are contributing. Read the constant from the
  server rather than trusting either number.
- **The 31 expected harvests.** That figure assumes the 95 in-reach boundaries
  from the 296-cell run carry over to a run with the support masks in place. The
  masks change the action distribution, which changes the walk, which changes
  the boundaries. It is a prediction from the old trajectory, not a simulation
  of the new one. Treat it as order-of-magnitude.
- **`REACH_TILES = 3` is untested.** The 296-cell run predates the `reachable`
  cell component. I do not know how many cells the new component generates or
  whether it interacts badly with the position component.
- **The frontier term.** Unmeasured, as section C explains. The pure and
  no-frontier runs are identical for their first 600 s.
- **Game speed ceiling.** Unmeasured. The fixed `sleep(0.5)` long-poll loops in
  several tool clients will set a floor that the simulation speed does not.
- **CRAFT reporting `ok` 402 times with an empty inventory.** I believe
  `craft_item` returned 0 and `dispatch` accepted it, because
  `fle/rl/action.py:183` tests `result is False or result is None` and 0 is
  neither. I did not confirm this against a live server. If it is right, the
  `ok` rate in every existing log is overstated and CRAFT should be counted as
  a rejection when it crafts zero items.
- **Whether a ninety minute budget is the right unit at all.** The two four-hour
  overnight runs in `overnight.md` showed that fifteen times the episodes bought
  nothing, which is evidence that these failures are structural rather than
  sample-limited. That supports short, sharply-targeted runs over long ones, and
  it is the reason every measurement in this document is designed to give a
  clear answer inside one slot.

---

## 7. References

- Ecoffet, Huang, Lehman, Stanley, Clune. First return, then explore. Nature
  590, 2021. Also the 2019 arXiv Go-Explore preprint.
- Salimans, Chen. Learning Montezuma's Revenge from a Single Demonstration.
  2018.
- Florensa, Held, Wulfmeier, Zhang, Abbeel. Reverse Curriculum Generation for
  Reinforcement Learning. CoRL 2017.
- Huang, Ontanon. A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms. 2020.
- Kanervisto, Scheller, Hautamaki. Action Space Shaping in Deep Reinforcement
  Learning. IEEE CoG 2020.
- Burda, Edwards, Storkey, Klimov. Exploration by Random Network Distillation.
  ICLR 2019.
- Pathak, Agrawal, Efros, Darrell. Curiosity-driven Exploration by
  Self-supervised Prediction. ICML 2017.
- Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos. Unifying
  Count-Based Exploration and Intrinsic Motivation. NeurIPS 2016.
- Badia et al. Never Give Up: Learning Directed Exploration Strategies. ICLR
  2020. Agent57, ICML 2020.
- Eysenbach, Gupta, Ibarz, Levine. Diversity is All You Need. ICLR 2019.
- Guo et al. Memory Based Trajectory-conditioned Policies for Learning from
  Sparse Rewards. NeurIPS 2020.
- Ghosh et al. Learning to Reach Goals via Iterated Supervised Learning. ICLR
  2021.
- Hubert et al. Learning and Planning in Complex Action Spaces (Sampled
  MuZero). ICML 2021.
- Hafner, Pasukonis, Ba, Lillicrap. Mastering Diverse Domains through World
  Models (DreamerV3). 2023. Noted here because it is the only existence proof I
  know of for learning a crafting tree from scratch without demonstrations, and
  because the Minecraft environment it used supplies a per-item milestone
  reward. The legitimate analogue in FLE is cell novelty over the item and
  entity tree, which is what the Go-Explore archive already is. The production
  score itself gives no such milestone, which is a real difference between the
  two settings and worth keeping in mind.
