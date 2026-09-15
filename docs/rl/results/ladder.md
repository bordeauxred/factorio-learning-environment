# Live RL learning ladder — 2026-09-13

One Factorio 2.0.73 `open_world` container ran at RCON port `27099`, map seed
`44340`, `game.speed=10`. Each episode reset entities, inventory, production
counters, research, and the observation epoch. The goal was **4 automatic
iron ore**. The learner was the repository's branching Double DQN with future
HER, MPS, batch size 64, one learner update per real transition, and **no
demonstrations**. Baseline actions were uniformly sampled from the currently
masked operation and its conditional argument masks. Greedy evaluations used
`epsilon=0` on fresh resets. This is one training seed (`20260913`), so it is
evidence for this fixed map and seed, not a multi-seed reliability estimate.

## L0 — fuelled drill, 8 actions

Reset placed a burner mining drill at `(106,-128)` on iron ore, facing east,
with **20 coal in its fuel inventory**. The player was teleported to
`(103,-128)` with an empty main inventory. A wooden chest was scripted at
`(107.5,-128.5)` to receive the drill's output. This receiver is part of the
exact task setup: without it, the drill reached the game's
`waiting_for_space_in_destination` state after 2 automatic ore and the goal
could not be reached. The setup and probe were performed while the game was
paused. The live probe started at 0 ore and one 60-game-second `WAIT` ended at
15 automatic ore, reward 1, with valid counters.

| Measurement | Real success | HER immediate positive transitions |
|---|---:|---:|
| Random baseline, 32 episodes | 16/32 (50%) | 0 |
| Untrained network, 32 fresh greedy episodes | 0/32 (0%) | 0 |
| Training, 100 episodes | 84/100 (84%) | 402/1262 HER copies |
| Trained checkpoint, 32 fresh greedy episodes | 32/32 (100%) | 0 |

All 32 trained evaluation episodes hit the goal on action 1. Their first
operation was `MOVE` in 32/32 episodes; the identically seeded untrained
network chose `EXTRACT` first in 32/32 and never produced ore. In the random
baseline, 16/32 episodes chose `WAIT`, but 8/16 successes did not choose
`WAIT`: normal non-WAIT actions also advance Factorio time. The trained
policy learned a successful time-advancing action, though it did not learn
the nominal `WAIT` solution.

| Training episodes | Real successes | Real first-hit rewards | HER immediate positives / copies | Fresh greedy evaluation after block |
|---:|---:|---:|---:|---:|
| 1–25 | 20/25 | 20 | 84/312 | 5/5 |
| 26–50 | 19/25 | 19 | 99/366 | 5/5 |
| 51–75 | 22/25 | 22 | 108/284 | 5/5 |
| 76–100 | 23/25 | 23 | 111/300 | 5/5 |

Training stored 800 real transitions and made 800 updates. The training phase,
including 20 periodic greedy episodes, lasted approximately 1327.4 wall
seconds, calculated from the logged start of the first training episode and
end of the last periodic evaluation. All 152 baseline, training, and periodic
episodes, all 32 final evaluations, and all 32 untrained evaluations logged
valid production counters. The JSONL grew from its header to a live first
transition before each run was released.

**Gate: PASS.** Fresh trained success was 32/32 versus 16/32 random, and the
identically initialized untrained network was 0/32. The 8-action maximum was
identical in baseline and evaluation. The checkpoint evaluation stopped at
the first hit; cumulative automatic production cannot lose a hit later in the
same episode. This avoided actions after an already decided success. The
training-phase JSONL has no `run_end` because the process was interrupted
after saving its checkpoint, before its redundant full-budget final evaluation;
the complete 32-episode checkpoint evaluation is in a separate JSONL.

Raw logs: [training and random baseline](ladder_l0.jsonl),
[trained checkpoint evaluation](ladder_l0_final_eval.jsonl),
[untrained evaluation](ladder_l0_untrained_eval.jsonl), and
[live setup probe](ladder_l0_probe.jsonl).

## L1 — unfuelled drill, 12 actions

Reset used the same iron site, player position, east-facing drill, and wooden
chest as L0, but the drill had **no fuel** and the player's main inventory held
**5 coal**. The player had no other items. The live scripted probe started at
0 automatic ore: `INSERT(drill, coal × 5)` returned reward 0 and outcome `ok`;
one 60-game-second `WAIT` then produced 15 automatic ore, reward 1, with valid
counters. The 12-action episode maximum was identical for baseline, training,
and final evaluation.

| Measurement | Real success | HER immediate positive transitions |
|---|---:|---:|
| Random baseline, 32 episodes | 1/32 (3.125%) | 0 |
| Training, 154 episodes | 81/154 | 376/1393 HER copies |
| Trained checkpoint, 32 fresh greedy episodes | 32/32 (100%) | 0 |

Each of the 32 final greedy episodes started with `INSERT`, then hit the goal
on a `MOVE` at action 2. The first periodic greedy block was 0/5: the policy
repeated `INSERT` and the game rejected all 12 attempts in each of those
episodes. The subsequent five blocks were each 5/5. This is a genuine
training curve, rather than only a favorable training-time epsilon result.

| Training episodes | Real successes | Real first-hit rewards | HER immediate positives / copies | Fresh greedy evaluation after block |
|---:|---:|---:|---:|---:|
| 1–25 | 2/25 | 2 | 11/69 | 0/5 |
| 26–50 | 7/25 | 7 | 49/272 | 5/5 |
| 51–75 | 17/25 | 17 | 68/252 | 5/5 |
| 76–100 | 15/25 | 15 | 62/240 | 5/5 |
| 101–125 | 17/25 | 17 | 74/240 | 5/5 |
| 126–150 | 19/25 | 19 | 96/288 | 5/5 |
| 151–154 | 4/4 | 4 | 16/32 | — |

Training stored 1848 real transitions, made 1848 updates, and lasted
1349.0960221660207 wall seconds, including 30 periodic greedy episodes. All
248 baseline, training, periodic, and final episodes logged valid production
counters. The JSONL log grew from its header to the first live transition
before training was released. Its `run_end.training_episodes` field incorrectly
reads 31 because the final-evaluation loop reused the training loop's Python
variable; the per-episode records contain 154 training episodes, and an
appended `metric_correction` row records the correction. The runner has been
fixed for subsequent runs.

**Gate: PASS.** Fresh trained success was 32/32 versus 1/32 random with
exploration off. This is still a single training seed on one fixed map.

Raw logs: [L1 baseline, training, and evaluation](ladder_l1.jsonl) and
[live setup probe](ladder_l1_probe.jsonl).

## L2 — drill and coal in inventory, 16 actions

Reset teleported the player to `(103,-128)` next to the same iron patch, with
**one burner mining drill and five coal in the main inventory**. No drill was
placed or fuelled at reset. The wooden output chest was at `(107.5,-128.5)`.
The live probe started at 0 automatic ore: `PLACE` of the drill at
`(106,-128)` facing east, `INSERT` of five coal, then one 60-game-second
`WAIT` all returned outcome `ok`; the final action produced 15 automatic ore,
reward 1, with valid counters. The episode maximum was 16 actions.

| Measurement | Real success | HER immediate positive transitions |
|---|---:|---:|
| Random baseline, 24 episodes | 0/24 (0%) | 0 |
| Training, 139 episodes | 0/139 (0%) | 4/36 HER copies |
| Trained checkpoint, 24 fresh greedy episodes | 0/24 (0%) | 0 |

| Training episodes | Real successes | Real first-hit rewards | HER immediate positives / copies | Fresh greedy evaluation after block |
|---:|---:|---:|---:|---:|
| 1–25 | 0/25 | 0 | 0/0 | 0/5 |
| 26–50 | 0/25 | 0 | 0/0 | 0/5 |
| 51–75 | 0/25 | 0 | 4/36 | 0/5 |
| 76–100 | 0/25 | 0 | 0/0 | 0/5 |
| 101–125 | 0/25 | 0 | 0/0 | 0/5 |
| 126–139 | 0/14 | 0 | 0/0 | — |

One training episode produced **2 automatic iron ore**, below the assigned
quota of 4. HER relabelled that episode to a reached subgoal and created 4
immediately positive transitions in 36 copies. This is **not** a real L2
success. All 139 training episodes received zero assigned-goal first-hit
reward. Training stored 2224 real transitions, made 2224 updates, and ran for
923.8911995840026 wall seconds. All 212 baseline, training, periodic, and
final episodes had valid production counters. The JSONL was verified growing
before the run was released.

The negative result is supported by these live checks:

1. **Reward and goal validity:** the `PLACE` → `INSERT` → `WAIT` probe reached
   15 automatic ore and returned reward 1; each training reset started at 0.
   The replay did see nonzero HER reward from the 2-ore trace, but no
   assigned-goal reward.
2. **Exploration and masks:** training selected `PLACE` 217 times and `WAIT`
   195 times, so neither operation was absent from exploration. At a fresh
   L2 reset, live operation-mask entries for both `WAIT` and `PLACE` were 1.
   The selected actions did not form the successful three-action dependency.
3. **Goal input:** on that same reset, changing only the goal from four iron
   ore to four copper ore changed the operation Q head by a maximum absolute
   0.049492448568344116; its argmax changed from `CONNECT` to `ROTATE`. The
   goal lane reaches the network, but these Q values did not yield a useful
   action for the iron goal.
4. **Action modelling:** the final trained policy chose `CONNECT` in all
   384/384 evaluation decisions. Its first tuple was
   `(10,2,1,0,0,0,0,0,0,0,0,0)` in 24/24 episodes and was rejected as
   `invalid connection` because connection argument 0 is the unused sentinel.
   All 384 final decisions were `preflight_invalid`, with 0 ore. Earlier
   periodic evaluations also chose `INSERT` with anchor 0, rejected as
   `real entity anchor required`. The marginal head masks allow these
   incompatible op/argument combinations. During training, 813 transitions
   were `preflight_invalid`, 835 were `tool_rejected`, 11 had partial mutation,
   and 565 were `ok`.

**Gate: FAIL.** The trained policy did not beat the 0/24 random baseline.
The immediate limits are a missing assigned-goal success signal for this
three-step task and a factorized action decoder that repeatedly chooses
argument sentinels invalid for its selected operation. This result does not
negate the L0 and L1 learning results. All three rungs used one training seed
and one fixed map.

Raw logs: [L2 baseline, training, and evaluation](ladder_l2.jsonl),
[live checkpoint diagnostic](ladder_l2_diagnostic.jsonl), and
[scripted live probe](ladder_l2_probe.jsonl).

`ruff check fle/rl/env.py tests/benchmarks/run_ladder.py` passed.
`pytest -q tests/rl/test_env_workers.py tests/rl/test_goals_replay_curriculum.py`
passed (21 tests). The one-service Compose project `fle_ladder_27099` was
brought down with `--remove-orphans`; `docker ps -a --filter
name=fle_ladder_27099` returned no containers. No cluster port was used.

## L2 with demonstrations — 2026-09-13

**Verdict: YES, demonstrations unlock L2 on this fixed map and seed.** The
demonstrations-only checkpoint reached the assigned goal in **32/32 fresh
greedy episodes**, versus **0/24** for the preceding L2 checkpoint without
demonstrations (and **24/24** over the matching first 24 evaluations). The
new random baseline was also **0/24**. Every successful
final episode began at 0 automatic ore, executed `PLACE → INSERT → WAIT` with
three `ok` outcomes, and ended at 15 automatic ore with reward 1 and valid
counters. Exploration was off. The 16-action maximum, map seed 44340, reset
position and inventory, chest, goal of 4 automatic iron ore, learner seed
20260913, MPS, batch size 64, one update per real transition, future HER
`k=4`, and 900-second training-window target matched the preceding L2 run.
The baseline used 24 episodes and the requested final evaluation used 32.

The replay seed came from `fle/rl/scripted.py` starting at its existing
`place_drill` phase under the L2 reset, rather than from the 44-action
empty-start archive. All **32/32 live demonstration episodes** performed
`PLACE(drill at 106,-128, east) → INSERT(drill, coal × 5) → WAIT(60 game
seconds)`, earned real assigned-goal reward 1, ended at 15 automatic ore,
and had valid counters before they entered replay. They seeded **96 real
transitions, 96 HER copies, and 32 immediately positive HER transitions**.
No learner updates or training episodes were counted as demonstrations.

| Training episodes | Real successes | Real first-hit rewards | HER immediate positives / copies | Fresh greedy evaluation after block |
|---:|---:|---:|---:|---:|
| 1–25 | 2/25 | 2 | 17/170 | 0/5 |
| 26–50 | 6/25 | 6 | 54/366 | 5/5 |
| 51–75 | 16/25 | 16 | 91/576 | 5/5 |
| 76–78 | 1/3 | 1 | 7/36 | — |

Training stored **1,248 new real transitions**, made **1,248 updates**, and
produced **25/78 real assigned-goal successes**. It added 1,148 HER copies,
169 with immediate positive reward, on top of the demonstration seed; final
replay size was 2,588 entries. The first greedy block placed and fuelled a
drill but stalled at 2 ore because its east-facing output missed the chest.
After 50 training episodes, all 10 remaining periodic greedy episodes hit
15 ore in three actions. The training phase from the first training-episode
start to the first final-evaluation start was **914.5226871240302 seconds**:
the 900-second loop finished its in-flight episode. The JSONL
`run_end.training_wall_s` value of 1134.3663822080125 incorrectly includes
the 32 final evaluations; an appended `metric_correction` records this, and
the runner now captures training time before final evaluation. The JSONL
grew from its header to the first live transition before the run was
released.

The following rates use **non-WAIT decisions** as the denominator and keep
preflight-invalid combinations separate from tool failures (`tool_rejected`
plus `partial_mutation`). These are the raw outcome classifications logged
by each run:

| Configuration and phase | Preflight invalid / non-WAIT (`invalid_combination_rate`) | Tool failure / non-WAIT (`tool_failure_rate`) |
|---|---:|---:|
| No demonstrations, L2 training | 813/2,029 (40.07%) | 846/2,029 (41.70%) |
| No demonstrations, final greedy | 384/384 (100%) | 0/384 (0%) |
| L2 demonstrations, random baseline | 20/339 (5.90%) | 186/339 (54.87%) |
| L2 demonstrations, training | 308/1,100 (28.00%) | 362/1,100 (32.91%) |
| L2 demonstrations, periodic greedy | 0/95 (0%) | 0/95 (0%) |
| L2 demonstrations, final greedy | 0/64 (0%) | 0/64 (0%) |

The demonstrations-only training rate exceeds the design's 15% concern
level, but the formal autoregressive-head trigger requires **10,000
non-WAIT transitions across three seeds after mask fixes**. This experiment
has 1,100 non-WAIT training decisions under the original masks and one seed.
Its final policy made no invalid tuples; the old policy's 384-decision
`CONNECT` collapse did not recur.

`CONNECT` had a cheap, definite mask bug. The old operation mask enabled it
whenever there were more than two selected rows. At the fresh L2 reset those
were the player character and the wooden chest, with no belt, pipe, or pole
in inventory. No useful connector triple existed, yet random baseline
episodes selected `CONNECT` on their first action; selecting the character
as an endpoint produced `FLE tool does not support prototype character`.
The operation mask now requires **two distinct non-character selected
entities and at least one connector material** before offering `CONNECT`.
This is a necessary local viability check; exact port and route feasibility
still belongs to Factorio. A regression test covers the L2 state. The
demonstrations-only process had already loaded the **original mask** before
this code change and was trained and evaluated with it.

The mask was tested separately on the **original 0/24 checkpoint**, without
retraining or demonstrations. With the tighter `CONNECT` operation mask, it
remained **0/24** and chose `RESEARCH` in all **384/384** decisions, each
with technology sentinel 0. Before the preflight classification fix, those
were logged as **0/384 preflight invalid and 384/384 tool failures**, all
`Technology PAD does not exist`. The same checkpoint under the same mask
after rejecting required catalog sentinel 0 in preflight remained **0/24**,
now with **384/384 preflight invalid and 0/384 tool failures**. This second
change only classified the same invalid action earlier; it did not change
the chosen tuple or reward. The tight mask eliminates the impossible
`CONNECT` choice at L2 reset but **does not prevent general invalid-tuple
collapse**: the independent argument head can still choose an unused
sentinel for an available operation. No invalid-action penalty was tested.
Some original-run sentinel failures were counted as tool failures, so its
raw `invalid_combination_rate` is a lower bound on semantic cross-head
invalidity.

Raw logs: [demonstrations-only baseline, seed, training, and final evaluation](ladder_l2_demonstrations.jsonl),
[mask-only old-checkpoint evaluation](ladder_l2_tight_mask_old_checkpoint.jsonl),
and [mask plus preflight-classification evaluation](ladder_l2_tight_mask_preflight_eval.jsonl).
The demonstrations-only checkpoint is `.fle/runs/ladder_l2_demonstrations.pt`.
All runs used one container at port 27099 at a time; each was torn down,
including after the RCON-startup failure before the final diagnostic retry.
