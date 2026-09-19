# V0 shared contract — masked autoregressive Double DQN

Status: authoritative for tonight's V0. Spec owner: main session. Implementers:
parallel Codex sessions, one per package, disjoint file ownership.
Branch: `feat/rl-v0-masked-dqn` (our RL stack merged with Jack's buildability +
minimap observation work at `f08b0a9c`).

Every package reads this file. Where this file and a package spec disagree, this
file wins. Do not change this file; raise the conflict instead.

## 0. Non-negotiables

1. Bias toward exposing MORE state. Never drop an observation channel to save
   compute before the bottleneck has been measured.
2. Buildability is BOTH an observation (all 63 channels) AND an action mask.
3. The mask forbids only what the engine has freshly proven impossible:
   `certainly_blocked = (values == 0) & fresh`. Unknown, stale and
   out-of-window are ALLOWED. `BuildabilityCache.certainly_blocked_mask()` on
   this branch already implements exactly this; `legal_mask()` is gone.
4. Abstract locomotion. Do not abstract factory geometry.
5. No wall-clock sleeping anywhere in the RL execution path.
6. Execution always revalidates against Factorio. A mask is help, not truth.

## 1. Coordinate frame — one frame, three verbs

`LOCAL_SIDE = 64`, exactly one Factorio tile per cell, player-centred.

```
local_origin_x = floor(player_x) - 32
local_origin_y = floor(player_y) - 32
world_tile(x_cell, y_cell) = (local_origin_x + x_cell, local_origin_y + y_cell)
cell_index = y_cell * 64 + x_cell            # 0 .. 4095
```

The player always occupies cell `(32, 32)`. `MOVE_TO`, `MINE` and `PLACE` use
this identical index space and identical transform. There is no second location
domain, no coordinate regression, no candidate set, no octant move, no nearest
position search. The 64-tile window is also what enforces the intended ~32-tile
`MOVE_TO` limit: every reachable cell is within ±32 tiles by construction.

Tile anchoring: Factorio snaps a prototype to integer or half-tile centres by
footprint parity. Convert a chosen cell to a placement position with the
deterministic parity rule only. Nothing else may move the target.

## 2. Observation — fixed schema, every decision

`gymnasium.spaces.Dict`, identical keys and shapes at every neural decision,
including internal argument-prefix decisions. No action-dependent shapes.

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `local_exact` | `(C_LOCAL, 64, 64)` | float16 | exact geometry, 1 tile/cell, player-centred |
| `minimap` | `(16, 128, 128)` | float16 | Jack's 14 channels + `known` + `age` |
| `buildability` | `(63, 128, 128)` | **int8** | raw cache values: -1 unknown, 0 blocked, 1 legal |
| `build_age` | `(63, 16, 16)` | uint16 | per-8×8-block age in ticks, clipped to 65535; 65535 = never sampled |
| `entities` | `(32, 24)` | float32 | existing entity table, unchanged |
| `entity_mask` | `(32,)` | float32 | 1 = live row |
| `inventory` | `(251,)` | float32 | existing log1p counts |
| `research` | `(196,)` | float32 | existing binary tech vector |
| `recipes_enabled` | `(217,)` | float32 | existing binary vector |
| `globals` | `(G,)` | float32 | existing globals, plus the additions in §2.3 |
| `masks` | see §4 | uint8 | per-head support masks |

Buildability stays **int8 all the way into replay**. Cast at batch time only.
Never call `.astype(np.float32)` on it in the env or the replay writer: measured
274.5 µs of pure scalar conversion per call on this machine against 12.3 µs for
an int8 copy (`docs/rl/research/20-int8-cast-cost.md`).

### 2.1 `local_exact` channels

Exact, from the tiered protocol caches. One tile per cell. Order is fixed:

| Idx | Channel |
|---|---|
| 0 | water |
| 1 | iron ore present |
| 2 | copper ore present |
| 3 | coal present |
| 4 | stone present |
| 5 | uranium ore present |
| 6 | crude oil present |
| 7 | resource amount, `log1p(amount) / 16` |
| 8 | tree |
| 9 | rock / cliff |
| 10 | friendly entity occupancy |
| 11 | enemy entity occupancy |
| 12 | player |
| 13 | entity direction `sin` |
| 14 | entity direction `cos` |
| 15 | entity class id, `class_index / n_classes` |
| 16 | tile is outside any synced chunk (unknown terrain) |

`C_LOCAL = 17`. Occupancy uses the real footprint of each entity, not its
anchor tile. Channel 16 exists so "no data" is distinguishable from "empty".

### 2.2 `minimap` channels

Channels 0-13 are Jack's, verbatim, `visibility="generated"`. Then:

- 14 `known`: 1 where `sampled_ticks >= 0`, else 0.
- 15 `age`: `clip((tick - sampled_tick) / MINIMAP_AGE_SCALE, 0, 1)`, and 1.0
  where never sampled. `MINIMAP_AGE_SCALE = 3600`.

Channels 2-8 and 12 (the five ore sums, crude oil, tree count, enemy unit
count) are stored as `log1p(value)`, keeping -1 unknown at -1. They are
unbounded sums: measured live, one cell held 133,877 iron, and float16
saturates at 65,504, so storing them raw silently turned real resource
information into `+inf`. Undo with `expm1` if a consumer needs raw amounts.

Stale and unknown data is passed to the network, never silently refreshed.

### 2.3 `globals` additions

Append to the existing globals, in this order: current tick / 1e6, automated
production score / 1e3, general production score / 1e3, episode simulated
seconds / 3600, buildability known fraction, buildability fresh fraction,
minimap known fraction.

## 3. Action grammar

Verbs, in this index order:

```
0 MOVE_TO(location)
1 MINE(location, quantity)
2 PLACE(place_item, direction, location)
3 CRAFT(recipe, quantity)
4 PICKUP(entity)
5 ROTATE(entity, direction)
6 INSERT(entity, inventory_item, quantity)
7 EXTRACT(entity, contained_item, quantity)
8 SET_RECIPE(entity, recipe)
9 RESEARCH(technology)
10 FAST_FORWARD(duration)
```

Argument heads and their domains:

| Head | Size | Domain |
|---|---|---|
| `verb` | 11 | above |
| `location` | 4096 | §1 cell index |
| `place_item` | n_placeable | placeable prototypes, masked to those the player owns |
| `direction` | 4 | Factorio cardinals 0, 4, 8, 12 |
| `quantity` | 7 | 1, 2, 4, 8, 16, 32, ALL |
| `entity` | 32 | entity-table slot, resolved to a unit number |
| `inventory_item` | 251 | masked to items actually carried |
| `contained_item` | 251 | masked to the selected target's actual contents |
| `recipe` | 217 | existing recipe vocab |
| `technology` | 196 | existing tech vocab |
| `duration` | 3 | 1 s, 10 s, 60 s of simulated time |

Decode order is fixed per verb and every later head conditions on the earlier
selected arguments:

```
MOVE_TO      location
MINE         location, quantity
PLACE        place_item, direction, location      <-- buildability mask here
CRAFT        recipe, quantity
PICKUP       entity
ROTATE       entity, direction
INSERT       entity, inventory_item, quantity
EXTRACT      entity, contained_item, quantity
SET_RECIPE   entity, recipe
RESEARCH     technology
FAST_FORWARD duration
```

### 3.1 Automatic navigation

`MINE`, `PLACE`, `PICKUP`, `ROTATE`, `INSERT`, `EXTRACT` and `SET_RECIPE`
navigate themselves into interaction range using existing pathfinding. That
movement consumes simulated Factorio ticks, contributes to the SMDP duration,
and does NOT consume another policy action. `MOVE_TO` remains available for
deliberate relocation, exploration and recentring, and is never required first.

### 3.2 Exactness and identity

- `PLACE` calls `place_entity(..., exact=True)`. `nearest_buildable`,
  `place_entity_next_to`, `connect_entities`, `exact=False` and any
  nearest-valid search are forbidden for the policy path.
- The chosen tile must survive selection, navigation and placement unchanged
  apart from deterministic parity canonicalisation. Assert it.
- Entity-pointer actions must mutate the requested `unit_number`. Assert
  `requested_unit_number == mutated_unit_number`, log violations, and never
  silently redirect to a nearby compatible entity. The existing name-and-
  position fallbacks in `ops.py` must go.

## 4. Masks

Each mask is individually toggleable by config flag and individually logged.
Masks remove only deterministic, certain refusals. No strategy-dependent masks.

| Mask | Predicate |
|---|---|
| `place_location` | `certainly_blocked_mask(tick, PLACE_MAX_AGE_TICKS)` for the selected `(place_item, direction)` channel, cropped to the 64×64 local frame |
| `place_item` | prototype count in player inventory > 0 |
| `inventory_item` | carried count > 0 |
| `contained_item` | count in the selected target > 0 |
| `entity` | row is live |
| `recipe` | existing hand-craftable predicate |
| `technology` | existing prerequisite predicate |
| `verb` | a verb is masked only if every one of its argument domains is empty |

`PLACE_MAX_AGE_TICKS = 600`, configurable.

Cropping the mask: the buildability window is 128×128 with an 8-aligned origin
that follows the character with a 24-tile dead zone, so the 64×64 local window
is not always fully inside it. Compute the overlap, take the blocked bits there,
and leave every cell outside the overlap ALLOWED. A partially covered window is
normal and must not raise.

Channel lookup uses `BuildabilityCache.channel_for(name, direction)`. Diagonal
directions have no channel; the `direction` head therefore exposes cardinals
only. If a lookup raises `KeyError`, treat the whole location mask as allowed
and count it in `build/mask_lookup_miss`.

## 5. Timing, reward, episodes

- Duration `tau` is measured in simulated seconds: `(tick_after - tick_before) / 60`.
- Discount `Gamma(tau) = 2 ** (-tau / 3600)`.
- Reward for learning is the automated production score delta only:
  `reward = automated_score_after - automated_score_before`. The general
  production score is logged but never mixed into the objective.
- Internal argument-prefix transitions carry `reward = 0`, `duration = 0`,
  `discount = 1`. Only a completed semantic action carries real values.
- `FAST_FORWARD` advances simulated ticks at the maximum stable speed. Inference
  and SGD must not advance the world: pause at decision boundaries. Note that
  pausing does not remove the RCON service quantum, which is `1 / (60 * game.speed)`
  seconds, so keep `game.speed >= 10` while paused
  (`docs/rl/research/` and the sim-time probe).

## 6. Learner contract

Masked autoregressive Double DQN. Online net, target net, replay, Huber, AdamW,
gradient clipping, epsilon-greedy. PER only if the existing implementation is
stable. No PPO, RNN, transformer, distributional Q, HER or curiosity tonight.

Masks apply at BOTH acting and the Double-DQN target argmax:

```
q[forbidden] = -inf
a          = epsilon_greedy(q)                      # acting
next_q[next_forbidden] = -inf
a_star     = argmax(next_online_q)                  # target selection
target     = reward + Gamma(tau) * Q_target(next_state, a_star)
```

Epsilon exploration samples only from the currently unmasked domain of the head
being decoded. Never sample a malformed complete tuple.

Encoders: a separate CNN per map, then fusion.

```
local_exact  (17,64,64)   -> CNN -> embedding
minimap      (16,128,128) -> CNN -> embedding
buildability (63,128,128) -> CNN -> embedding + spatial features
entities     (32,24)      -> MLP + pooling
inventory / research / recipes / globals -> MLP
                 |
                 v
        fused state embedding -> autoregressive heads
```

The `location` head is a spatial Q map over the 64×64 frame, produced by fusing
local and buildability spatial features and a few convolutions. Resize or crop
the buildability pyramid into the 64×64 local frame. No attention. Target under
10M parameters.

## 7. Replay

Ring buffer storing each state once. Per-key dtypes exactly as §2. Buildability
int8, cast at batch time into a preallocated float32 buffer. No bitpacking.

Log `train/replay_bytes_per_transition` and `train/replay_bytes` at startup and
every log interval. If memory binds, reduce capacity; never drop channels.
Default capacity 8192, configurable.

## 8. W&B

Entity `2robert-mueller-none`, project `fle-sm-arq`. The organisation entity
`2robert-mueller-none-org` refuses runs, so do not use it. Run names
`v0-mask-scratch-<timestamp>` and `v0-mask-demo-<timestamp>`.

Config must record: git SHA, branch, Factorio version, seed, every observation
shape, vocab hashes, network config, replay config, epsilon schedule, every mask
toggle, buildability size and budget, freshness thresholds, minimap chunk
budget, reward definition, SMDP half-life.

Metric names are fixed by the run spec and must be used verbatim: the
`env/`, `action/`, `build/`, `train/`, `explore/` and `perf/` families.

## 9. Correctness gate

These assertions run before any training and must all pass:

1. `local_exact` is 64×64 and `MOVE_TO`/`MINE`/`PLACE` decode identical cells.
2. `MOVE_TO` cannot select a tile more than 32 tiles away.
3. Buildability reaches the network, as int8, with unknown (-1) preserved.
4. Freshness reaches the network; stale is distinguishable from fresh.
5. A fresh blocked PLACE cell is masked.
6. A fresh legal PLACE cell is allowed.
7. An unknown PLACE cell is allowed.
8. A stale PLACE cell is allowed.
9. The mask aligns spatially with the local frame under a deliberate origin
   offset, including partial overlap.
10. The correct prototype/direction channel is selected.
11. `PLACE` executes with `exact=True` and the target tile is unchanged after
    navigation.
12. Entity actions mutate the requested unit number.
13. `INSERT` uses carried items; `EXTRACT` uses the target's contents.
14. Masks apply to the Bellman argmax.
15. Epsilon exploration cannot select a masked action.
16. No wall-clock sleep in the execution path.
