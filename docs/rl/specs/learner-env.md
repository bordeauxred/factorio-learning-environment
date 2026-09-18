# Spec: learner-facing macro-action environment and first learners

Status: to implement. Spec owner: main session. Implementers: three Codex
sessions (A: environment, B: PufferLib PPO, C: off-policy learner). The
shared contract is already written and frozen for this round:
`fle/rl/schema.py` (heads, sizes, observation layout, mask packing) and
`fle/rl/fake_env.py` (server-free stand-in with the same tensors). Do not
change `schema.py` without saying so in your final report; B and C build
against it while A builds the real environment.

Read first: `docs/rl/BRIEF.md`, `docs/rl/results/op_census.md` (the measured
environment facts this design responds to), `docs/rl/specs/op-census.md`,
`fle/rl/ops.py`, `fle/rl/world.py`, `fle/rl/schema.py`. Rules: never modify
`fle/env/` or `fle/cluster/`; read every tool client before calling it; no
scripted policies; the reward is FLE's automated production score delta and
nothing else.

## Objective and metric

Reward per step `r_t = APS(t+1) - APS(t)` where `APS = namespace.score()[1]`,
the automated production score. Nothing is added: no shaping, no invalid-action
penalty, no bonus for the first item. The player score `score()[0]` is logged
as a diagnostic only. Report PS distributions per episode, never a pass rate.

## Measured speed facts (2026-09-15, port 27003)

| requested game.speed | realized UPS idle | move 53 tiles + harvest 5, wall s | true ticks |
|---:|---:|---:|---:|
| 10 | 401 | 1.53 | 502 |
| 20 | 445 | 0.77 | 224 |
| 40 | 491 | 0.46 | 195 |
| 60 | 476 | 0.34 | 125 |

The server saturates near 480 UPS under box64. Higher requested speed only
shortens the tool clients' cosmetic sleeps, so use speed 40 for training and
measure time in true ticks everywhere.

## Servers

Four open-world servers are up on RCON ports 27000 to 27003. Assignment:
A develops on 27000 and 27001; B and C develop only on `FakeMacroEnv` and use
no server; the main session runs live training after A lands. Never start or
stop containers.

## A. Environment: `fle/rl/env.py`, class `FleMacroEnv(gymnasium.Env)`

Constructor: `FleMacroEnv(port=27000, speed=40, max_steps=128,
max_ticks=108_000, seed=None, log_path=None)`. One instance owns one
`FactorioInstance` and one `WorldClient`; terrain full sync once per process.

Spaces: `observation_space = Box(-1, 1, (schema.OBS_SIZE,), float32)`,
`action_space = MultiDiscrete(schema.HEAD_DIMS)`. Masks live in the
observation tail at `schema.MASK_OFFSETS`.

### Operations and execution (11 ops; CONNECT withheld, see census fact 7)

Reuse `fle/rl/ops.py` executors where they fit; extend, do not duplicate.

| Op | Heads read | Execution |
|---|---|---|
| WAIT | duration | poll true `game.tick` (every 0.05 s) until start + ticks, wall cap 30 s; record actual ticks. Do not use the `sleep` tool: measured realized UPS is about 480 regardless of `game.speed`, so its wall sleep under-waits at high speed |
| MOVE | move_dir | `move_to(player + 16 tiles in the octant)`; if the cell is water or unknown, snap to the nearest known land cell in that direction; stopping within 2.5 tiles counts as arrival |
| HARVEST | target, quantity | resolve the slot's tile; `move_to(tile centre)`; check live distance <= reach - 0.1 else `approach_failed`; `harvest_resource(tile, quantity)` |
| CRAFT | recipe, quantity | `craft_item(name, quantity)` |
| PLACE | item, offset, direction | position = player + (dx, dy) snapped to the footprint's legal centre; `place_entity(prototype, direction, pos, exact=True)` |
| PICKUP | entity | if farther than build reach, `move_to` a free tile adjacent to it first; `pickup_entity(entity)` |
| ROTATE | entity, direction | approach as PICKUP; `rotate_entity` |
| INSERT | entity, item, quantity | approach; `insert_item(prototype, entity, quantity)` |
| EXTRACT | entity, item, quantity | approach; `extract_item` |
| SET_RECIPE | entity, recipe | approach; `set_entity_recipe` |
| RESEARCH | technology | `set_research(name)` |

Approach for entity ops is the same invariant as HARVEST: reach is the
executor's problem, not the policy's. Every tool call is wrapped in the census
harness's 120 s timeout.

After every action: `world.all_drain()`, then **a second drain 0.25 s later
if the op was PICKUP or SET_RECIPE** (census fact 6: their effects lag one
reconciler sweep), then snapshot. Effect verification reuses
`ops.verify_effect` with PICKUP judged by inventory delta.

### Masks (state-only support; `schema.MASK_OFFSETS`)

| Head | Mask rule |
|---|---|
| op | WAIT, MOVE always. HARVEST iff any target slot. CRAFT iff any hand-craftable enabled recipe (`ops.hand_craftable_recipes`). PLACE iff any placeable item held (vocab `place_result` resolves to a `Prototype`). PICKUP, ROTATE iff any entity slot. INSERT iff entity slot and non-empty inventory. EXTRACT iff any entity slot with items. SET_RECIPE iff any assembling-machine slot and any enabled recipe of category `crafting`. RESEARCH iff any available technology |
| target | slot filled |
| entity | slot filled |
| item | union: held items, items present in any entity slot's inventories |
| recipe | enabled and category in the character's crafting categories (read live once) |
| technology | enabled, not researched, all prerequisites researched, and **no research trigger** (read `t.prototype.research_trigger ~= nil` live once at startup and cache per name; census fact 3) |
| offset | (dx, dy) with Euclidean length <= build reach, whose tile in the local grid is not water and not occupied by a tracked entity footprint |
| direction | all four |
| quantity | all three |
| duration | 60 ticks always; 300/900/3600 only if any entity exists |
| move_dir | all eight |

Every head keeps index 0 admitted only if its rule admits it; if a rule admits
nothing the head is all zero and the op mask already excludes every op that
reads it. Assert at least WAIT is always admitted.

### Observation blocks (`schema.OBS_LAYOUT`)

- `globals` (16): episode tick / max_ticks; steps remaining / max_steps;
  player x/256, y/256; log1p(total inventory)/log1p(1000); log1p(n
  entities)/5; log1p(max(player PS, 0))/10; log1p(max(APS, 0))/10; last
  status one-hot (ok, no_effect, rejected); last op / 11; research in
  progress; any entity exists; reach/10; 2 zeros.
- `inventory` (251): log1p(count)/log1p(1000) by `schema.ITEM_INDEX`.
- `tech` (196): researched bits.
- `recipes_enabled` (217): enabled bits.
- `targets` (24 x 12): slots 0-15 the 16 nearest ore patches by
  `patch.nearest_tile` distance within 160 tiles; slots 16-23 the nearest
  tree in each compass octant within 160 tiles. Features: kind one-hot over
  `schema.TARGET_KINDS`, dx/160, dy/160, dist/160, log1p(n_tiles)/10 (trees:
  0), valid, 0.
- `entities` (32 x 24): 32 nearest tracked entities. Features: class one-hot
  over `schema.ENTITY_CLASSES` from the vocab entity `type`; dx/32, dy/32,
  dist/32 (clipped to 1); direction/16; status group one-hot over
  `schema.STATUS_GROUPS` using the vocab `statuses` code-to-name table (map
  working; no_fuel, no_power, low_power, no_minable_resources to
  no_fuel_or_power; no_ingredients, item_ingredient_shortage,
  waiting_for_source_items to no_input; full_output,
  waiting_for_space_in_destination to output_full; rest other); has recipe;
  recipe index/217; log1p(item total)/log1p(1000); log1p(coal+wood
  count)/log1p(100); valid; 0.
- `grid` (4 x 17 x 17): egocentric tiles within [-8, 8]: entity footprint,
  resource tile, tree or rock, water. Channel order `schema.GRID_CHANNELS`,
  row-major (channel, dy, dx).
- `masks` (1039) as above.

Slot order is stable within a step; a slot index means the same thing between
the observation and the action that follows it.

### Episode

`reset()`: `instance.reset(reset_position=True,
all_technologies_researched=False, clear_entities=True)`, speed set, entity
full sync, fixture empty, baseline `APS_0 = score()[1]`. `step()`: execute,
observe, `reward = APS - APS_prev`. `truncated` when `max_steps` or
`max_ticks` is reached. `terminated` when the character becomes invalid
(census fact 8; detect by a failed live position read or a position jump to
spawn with no MOVE); log it. `info` carries `op`, `status`, `reason_class`,
`score_player`, `score_automated`, `ticks`, `wall_s`, `items_delta`,
`n_entities`. With `log_path` set, write one JSON line per step and one
`{"episode_end": ...}` line per episode with final APS, max APS, player PS,
op histogram, status histogram, entities placed, items harvested.

### Acceptance for A

- `pytest -q tests/rl/test_env_offline.py`: mask rules on hand-built world
  states (empty world admits only WAIT, MOVE, RESEARCH; a world with one
  patch admits HARVEST; offsets beyond reach 10 are masked; trigger techs
  masked), observation layout round-trips, `random_valid_action` never picks a
  masked index.
- Live: `python -m fle.rl.env --port 27000 --episodes 3 --steps 64 --seed 1`
  drives `random_valid_action`, prints per-episode final APS, player PS, ok
  rate, invalid-combination rate (status `tool_rejected` or `no_effect` over
  non-WAIT steps), steps/s, and confirms HARVEST succeeded at least once.
  Include the output verbatim.
- ruff clean on new files.

## B. PufferLib PPO: `fle/rl/puffer/`

Phase 0, timebox 30 minutes of wall-clock: create `.venv-puffer` with Python
3.12, install `pufferlib==3.0.0` (sdist; needs a C compiler) and this repo
with `pip install -e . --no-deps` plus FLE's runtime deps at the versions
pufferlib pins (numpy 1.26.x, gymnasium 0.29.1). Verify `import fle.rl.env`
and `import pufferlib` in the same interpreter, then run one PPO update on
`FakeMacroEnv`. If the timebox expires without a working install, fall back
to a vendored CleanRL-style PPO (`fle/rl/ppo_cleanrl.py`, single file,
threads as the vector env since the real env is network-bound) in the main
`.venv`, and say so plainly in the report.

Policy (`fle/rl/puffer/policy.py`): encoder = per-block MLPs (globals+inventory
+tech+recipes concatenated -> 256; targets 24x12 -> shared row MLP -> masked
mean/max pool -> 128; entities 32x24 -> same -> 128; grid 4x17x17 -> 2 conv
layers -> 128), concatenated -> 512 -> 512. One linear head per
`schema.HEADS` entry; **slice masks from the observation tail and
`masked_fill(mask == 0, -1e8)` every head's logits before returning**
(PufferLib has no action masking; docs/rl/research/04-pufferlib.md section 3).
Sample heads independently (option A there). Value head on the trunk.

Trainer: PufferLib 3.0 clean_pufferl with `pufferlib.emulation.GymnasiumPufferEnv`
wrapping `FleMacroEnv` (or `FakeMacroEnv` with `--fake`), `num_envs` = number
of ports given, one env per port, serial or thread backend (never fork a
process holding an RCON socket). Defaults: rollout 128 steps per env, 4
minibatches, 2 epochs, lr 3e-4, gamma 0.99, gae 0.95, entropy 0.01, clip 0.2.
CLI: `python -m fle.rl.puffer.train --ports 27000,27001 --total-steps 20000
--run-name ppo-v0 --out runs/ppo-v0`. Log per update to `runs/<name>/metrics.jsonl`:
env steps, wall, loss terms, entropy per head, mean episode reward, final APS
per finished episode, player PS, op histogram of the update's actions,
invalid-combination rate. Save checkpoints every 10 updates.

Acceptance for B: on `FakeMacroEnv` with 4 envs, 20k steps, mean episode
reward rises clearly above the random baseline (report both numbers); masks
are respected (assert no sampled action index has mask 0 during a 2k-step
check); ruff clean. Do not touch a live server.

## C. Off-policy learner: `fle/rl/dqn.py`

CleanRL-style single file. Branching double DQN over `schema.HEADS`: shared
encoder as in B (write it once in `fle/rl/nets.py` and have B import it if
B's venv can import torch from the same source tree; otherwise duplicate and
note it), dueling value plus one advantage vector per head, centred per head.
Q of a complete action = V + mean over the op's active heads (`schema.OP_HEADS`)
of their advantages, plus the op head's advantage. Masks: `masked_fill(-1e8)`
before every argmax, both behaviour and target. Epsilon-greedy behaviour
over masked heads, epsilon 1.0 -> 0.1 over the first 30% of steps. n-step 3,
gamma 0.99, replay 100k transitions storing obs as float16 (masks recovered
from the stored obs), batch 64, learning starts 500, train every step, target
copy every 1000 updates, Adam 1e-4, Huber loss. Threads as the vector env for
live use, `--fake` for `FakeMacroEnv`. Log per 100 steps to
`runs/<name>/metrics.jsonl` with the same fields as B plus TD loss and mean
max-Q. CLI: `python -m fle.rl.dqn --ports 27002 --total-steps 20000 --run-name
dqn-v0 --out runs/dqn-v0`.

Acceptance for C: on `FakeMacroEnv`, 20k steps, mean episode reward rises
clearly above random (report both); a maximally masked batch yields finite
loss and gradients; ruff clean. No live server.

## What the main session does afterwards

Runs A's live smoke, then B and C on live servers for about 60 to 90 minutes
each, and writes `docs/rl/results/first_curves.md` with per-episode APS and
player PS distributions against the random baseline from the census, op
histograms over time, and what the curves say.
