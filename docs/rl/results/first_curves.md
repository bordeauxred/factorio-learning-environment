# First learning curves on open play (2026-09-15 to 16)

Two snapshots: the 2-hour interim below, and the overnight final at the end.
DQN ran to its 20k-step budget (159 episodes); PPO to 331 episodes; random to
150. Regenerate with:

```sh
.venv/bin/python tests/benchmarks/plot_first_curves.py --out docs/rl/results/first_curves \
    ppo=runs/ppo-v0/env_27000.jsonl,runs/ppo-v0/env_27001.jsonl \
    dqn=runs/dqn-v0/env_27002.jsonl random=runs/random-v0/env_27003.jsonl
```

## Setup

Environment: `fle/rl/env.py` (`FleMacroEnv`), the macro-action contract in
`fle/rl/schema.py` and `docs/rl/specs/learner-env.md`. Eleven ops (CONNECT
withheld), independent action heads with state-only support masks packed into
the observation, navigation absorbed into HARVEST and the entity ops. Reward is
the change in FLE's automated production score, `score()[1]`, with nothing
added. Episodes are 128 macro steps or a character death. `game.speed = 40`
(realized UPS about 480 regardless of speed under box64), one open-world
server per env, empty start, map seed 44340.

| Arm | Learner | Servers | Config |
|---|---|---|---|
| random | `schema.random_valid_action` | 27003 | uniform over admitted indices |
| ppo | PufferLib 3.0 `PuffeRL`, masked multi-head policy (`fle/rl/puffer/`) | 27000, 27001 | rollout 128 x 2 envs, 4 minibatches, 2 epochs, lr 3e-4, gamma 0.99, gae 0.95, ent 0.01, clip 0.2 |
| dqn | branching double DQN (`fle/rl/dqn.py`) | 27002 | n-step 3, gamma 0.99, replay 100k, batch 64, eps 1.0 -> 0.1 over 6k steps, target copy 1000, Adam 1e-4 |

Throughput under three servers plus two trainers on this Mac: about 1 macro
step per second per server, so roughly 25 episodes per server-hour.

## Result at the snapshot

| Arm | Episodes | Episodes with automated PS > 0 | Positive values | Furnace fuelled and fed | Deaths |
|---|---:|---:|---|---:|---:|
| random | 59 | **0** | | 2 | 13 |
| ppo | 56 | 3 | 23, 18, 12 | 5 | 7 |
| dqn | 40 | **6** | 70, 35, 11, 6, 2, 1 | 10 | 6 |

DQN against random, 6/40 versus 0/59, is p = 0.003 by Fisher's exact test.
Four of DQN's six positive episodes are in its last third, its INSERT share
tripled from 2.2% to 8.3% across thirds, and its WAIT share halved. Its mean
episode reward went from -1.0 to +2.4. Every positive episode used the same
chain: craft a stone furnace, place it, insert coal, insert stone or ore,
keep inserting. Episode 27 (APS 35) and a later one (APS 70) were stone-brick
lines. The previous attempt's best number, 35.0, needed a scripted seed.

PPO moved the other way. Across thirds its HARVEST share fell 14.5% to 6.1%,
WAIT rose 20% to 31%, and median player PS fell 514 to 279. Its three positive
episodes came mid-run and did not persist. Its invalid-combination rate rose
from 0.33 to 0.39.

![automated PS per episode](first_curves/aps_per_episode.png)
![op share per block of 10 episodes](first_curves/op_share.png)

Player PS (hand work, diagnostic): random median 583, dqn 552, ppo 374.

## What the curves say

1. **The objective is learnable from the empty start, and DQN is learning
   it.** No scripted seed, no shaping, no goal conditioning. Six unscripted
   automation episodes, concentrated late, with the op distribution shifting
   toward INSERT and away from WAIT.
2. **PPO is drifting toward inaction, as the objective predicts.** Any hand
   action floors the automated score at -1, while doing nothing scores 0. On a
   near-flat reward the on-policy learner reduces harvesting and waits more.
   DQN's replay keeps its rare positive episodes alive; PPO's rollouts forget
   them. The off-policy learner is the right primary for this reward.
3. **The first automated act is paid with a negative reward.** Inserting ore
   and coal into a furnace immediately charges consumption (-7 in the traced
   episode), and the plate credits +7 two steps later. A furnace started after
   about step 115 of 128 never pays back. Horizon matters, and a longer episode
   would raise the positive count for every arm.
4. **Character death ends 15% of episodes** (26 of 155 across arms). Cause
   unidentified; peaceful mode is on. This is the largest unexplained loss of
   sample and needs a live investigation before longer runs.
5. **The policies destroy their own factories.** In most positive episodes a
   later PICKUP removed the working furnace. Nothing in the reward yet says
   why that is bad within the horizon.
6. **Independent heads waste a third of decisions.** PLACE fails 79% of the
   time because the item head is sampled without knowing the op (wood is not
   placeable); CRAFT fails 80% on missing ingredients. Both are deterministic
   refusals: a placeable-item head and a recursive ingredient check are
   support-mask changes, not preferences.

## What to change next, in order

1. Investigate and fix the character deaths.
2. Schema v1: PLACE gets its own placeable head; CRAFT mask checks the
   recursive ingredient closure; both learners rebuild against it.
3. Longer horizon (256 steps or a true-tick budget) so late furnaces pay back.
4. Keep DQN as the primary learner and run it for several hours across three
   servers; keep PPO as the comparison arm with the same schema.

Raw logs: `runs/{ppo-v0,dqn-v0,random-v0}/env_*.jsonl` (one line per step,
one `episode_end` per episode) and `runs/*/metrics.jsonl` per learner update.
Objective behaviour notes are also in `docs/rl/results/op_census.md`.

## Overnight final (2026-09-16 morning)

| Arm | Episodes | Final APS median [q1, q3] | max | Episodes APS > 0 | Player PS median |
|---|---:|---|---:|---:|---:|
| random | 150 | -1 [-1, -1] | 0 | 0 | 599 |
| ppo | 331 | -1 [-1, 0] | 23 | 3 | 62 |
| dqn | 159 | **54 [-1, 162]** | **723** | **109** | 653 |

DQN by 40-episode block: median APS -1, 41, 104, 389; positive episodes
6/40, 28/40, 38/40, 37/39; deaths 6, 2, 1, 0. Its last-block op mix is 38%
HARVEST, 32% INSERT, 3% WAIT, 3% MOVE. The best episode (APS 723, player PS
4,378) placed 4 furnaces, inserted stone 30 times and coal twice, harvested
1,417 items: a stone-brick smelting line run by hand-feeding. Across all 159
episodes the only things ever placed were stone furnaces (395) and wooden
chests (69); nothing else was ever crafted. That is the plateau: the next rung
(iron plates extracted, gears, a burner drill on ore, fuelled) is a long
conjunction epsilon-greedy does not find. PPO's final median player PS of 62
is the inaction collapse completed: it stopped harvesting almost entirely.

![automated PS per episode, final](first_curves/aps_per_episode.png)

**Death cause, found and reproduced:** worm turrets. Every death hotspot has
an enemy `turret` within 40 tiles; `surface.peaceful_mode` is false although
`FactorioInstance(peaceful=True)` is the default, and teleporting the
character to a hotspot killed it within 2 s. v1 restores the intended peaceful
default at reset (`docs/rl/specs/v1.md`).

## Automation ladder, v0 final (episodes reaching each rung)

| rung | ppo | dqn | random |
|---|---:|---:|---:|
| furnace placed | 24/331 | 129/159 | 35/150 |
| furnace fuelled | 10/331 | 117/159 | 17/150 |
| furnace fed | 18/331 | 126/159 | 21/150 |
| furnace fuelled and fed | 7/331 | 117/159 | 10/150 |
| plates extracted | 1/331 | 10/159 | 0/150 |
| gears crafted | 0/331 | 0/159 | 0/150 |
| drill crafted | 0/331 | 0/159 | 0/150 |
| drill placed | 0/331 | 0/159 | 0/150 |
| drill fuelled | 0/331 | 0/159 | 0/150 |
| chest placed | 84/331 | 22/159 | 88/150 |
| inserter placed | 0/331 | 0/159 | 0/150 |
| belt placed | 0/331 | 0/159 | 0/150 |
| APS > 0 | 3/331 | 109/159 | 0/150 |

The wall is at gears: plates were extracted in 10 DQN episodes, never enough to craft an iron gear wheel, so no drill was ever crafted. v1's recursive ingredient mask offers CRAFT gears only when the plates are in hand.
