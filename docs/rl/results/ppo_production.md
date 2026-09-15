# L2 on-policy PPO against automatic production score

**Verdict: no L2 learning from scratch in this single-seed run.** After 148
training episodes, the policy succeeded in **0/24 fresh greedy episodes**, the
same as the **0/24 fresh random baseline**. It never reached four automatic
iron ore during training (**0/148**). It did discover partial production: 13
training episodes made one or two automatic iron ore, but no episode exceeded
two. The final greedy policy repeated an unplaceable `PLACE` action, not the
required `PLACE → INSERT → advance time` sequence. This matches the earlier
no-demonstration DQN/HER result (**0/24**) and falls short of the
96-transition demonstration result (**32/32**).

## Trainer and setup

This was a **PufferLib-backed local PPO loop**, not the stock `PuffeRL` class
and not the DQN/HER learner. It used `pufferlib==3.0.0`'s
`pufferlib.pytorch.sample_logits` for masked `MultiDiscrete` action sampling
and log-probability recomputation, and its compiled
`compute_puff_advantage` for GAE. The local rollout and clipped PPO update in
[`run_ppo_production.py`](../../../tests/benchmarks/run_ppo_production.py)
allowed one Factorio server to provide periodic fresh greedy evaluations.
PufferLib installed from its source distribution into a separate trainer
virtualenv (`.fle/pufferlib-venv`) with NumPy 1.26.4 and Gymnasium 0.29.1;
the runner supplies FLE's legacy `np.bool` alias before importing FLE and
used PyTorch 2.14.0 in that isolated environment. This kept PufferLib's
incompatible version pins out of the FLE environment. The stock `PuffeRL`
collector was not adapted to the single-server evaluation schedule within
this experiment's time budget. The actual PPO optimizer, including its
clipped objective, is in the linked runner; these are **PufferLib components
inside a local PPO trainer**, not a stock PufferLib training run.

The run used one Factorio 2.0.73 `open_world` container at RCON **27098**,
map seed **44340**, `game.speed=10`, and learner seed **20260913**. Every L2
reset teleported the player to `(103,-128)` with one burner mining drill and
five coal in inventory, no placed drill, and a wooden receiving chest at
`(107.5,-128.5)`. The goal was four **automatic iron ore** in at most **16
actions**. There were **no demonstrations** or HER relabelling. The
observation was the schema's 22,815-float `Box` with 64 entity rows. The
policy used the repository's `BranchingDuelingQ` encoder as a shared
actor-critic backbone; its branch outputs became categorical logits and its
value output became the critic. PufferLib sampled all 12 action heads
independently, as its API requires. Masks came from the observation and were
applied as finite `-1e8` logits in float32. Every head retained at least one
valid entry, with the policy asserting this. The training operation set was
`WAIT`, `MOVE`, `PLACE`, `INSERT`, matching the L2 benchmark; the random
baseline drew from conditional valid-action masks. This difference in action
decoding matters for the diagnosis below.

PPO settings were four complete 16-step episodes per rollout (64 transitions),
four full-batch update epochs, Adam at `3e-4`, `gamma=0.99`, GAE
`lambda=0.95`, clip ratio `0.2`, value coefficient `0.5`, entropy coefficient
`0.01`, gradient norm cap `1.0`, and MPS float32. Greedy evaluation used
argmax independently per head, with exploration off. The run's JSONL grew to
a live `logging_probe` transition before training was released. A maximally
masked synthetic batch completed an actual PPO update with finite loss
`0.11540486663579941`; a toy GAE check also pinned PufferLib's shifted
reward-row convention, including the final action reward. The run completed
without an invalid production counter or nonfinite PPO loss.

## Ground-truth reward verification

`fle/rl/reward.py` implements sparse goal-first-hit reward, so the PPO runner
used the same score reward as the parallel DQN production-score arm:
`score_t+1[1] - score_t[1]` from `instance.first_namespace.score()`. The
second accessor is the **automated score**, which removes recorded manual
harvest and craft contributions from the priced net
production-minus-consumption balance. It is integer-floored by the Factorio
score script. L2 success was checked independently from the automatic
iron-ore counter, not inferred from this scalar.

The live scripted L2 probe used the exact reset and three required actions:

| Boundary | Automated score | Transition reward | Automatic iron ore |
|---|---:|---:|---:|
| Reset | 0 | — | 0 |
| `PLACE` drill at `(106,-128)`, east | 0 | 0 | 0 |
| `INSERT` five coal | -3 | -3 | 0 |
| `WAIT` 60 game seconds | 37 | +40 | 15 |

All three actions returned `ok` with valid counters. **Placing the drill did
not reduce the live score; inserting its coal did.** The negative fuel step
means this reward is informative after production starts, but does not give a
positive signal for every necessary precursor. Raw probe:
[`ppo_production_probe.jsonl`](ppo_production_probe.jsonl).

## Learning curve

Each row is a consecutive episode block. Training return is the sum of the
automatic-score deltas, with the per-episode mean in parentheses. Automatic
iron is the total produced in the block, with the per-episode mean in
parentheses. Periodic evaluation used five fresh greedy episodes after each
complete 24-episode block.

| Training episodes | L2 success | Training score return, sum (mean/episode) | Automatic iron, total (mean/episode) | Fresh greedy L2 success |
|---:|---:|---:|---:|---:|
| 1–24 | 0/24 | 0 (0.000) | 2 (0.083333) | 0/5 |
| 25–48 | 0/24 | -3 (-0.125) | 0 (0.000) | 0/5 |
| 49–72 | 0/24 | +6 (+0.250) | 4 (0.166667) | 0/5 |
| 73–96 | 0/24 | +6 (+0.250) | 7 (0.291667) | 0/5 |
| 97–120 | 0/24 | +9 (+0.375) | 6 (0.250000) | 0/5 |
| 121–144 | 0/24 | +6 (+0.250) | 6 (0.250000) | 0/5 |
| 145–148 | 0/4 | 0 (0.000) | 0 (0.000) | — |

Across all **148 training episodes**, score return summed to **+24**
(mean **0.16216216216216217** per episode), and automatic iron production
summed to **25** (mean **0.16891891891891891** per episode). Thirteen
episodes produced automatic iron; the maximum was **2** in one episode.
The first 2-ore episode placed and fuelled a drill and earned +3 net score,
but its output missed the receiving chest and the drill stalled. The final
evaluation made zero automatic iron.

| Fresh policy | L2 success | Automatic iron total | Mean automatic iron/episode |
|---|---:|---:|---:|
| Conditional-mask random, before training | 0/24 | 8 | 0.3333333333333333 |
| Trained PPO, greedy | 0/24 | 0 | 0 |

The 27.5-minute target produced **2,368 training transitions** and **148
optimizer steps** over **1663.6395023749792 s** of training wall time:
**1.4233852926787862 training steps/s** and **0.08896158079242414
updates/s**, including periodic evaluations. The full process, including
baseline, training, final evaluation, and setup, took
**2193.2076814169995 s** (36.55 minutes). The machine also ran another
Factorio container and trainer, so these are loaded-laptop rates. Logged
Factorio step calls consumed **1536.981717878778 s**; optimizer updates
consumed **7.261454004212283 s**. Factorio interaction, not PPO compute,
was the binding constraint.

## Failure diagnosis

The reward path worked: the scripted probe made 15 ore with a +37 net score,
and training saw 13 partial-production episodes. Exploration did sample the
required operations: across training, `PLACE` appeared **694** times,
`INSERT` **521**, and `WAIT` **586**. But `PLACE` plus fuel plus a legal
east-facing output into the chest is a rare joint sequence; no exploratory
episode reached four ore. Fuel insertion's -3 score also opposes its
immediate policy gradient before ore is produced. These are exploration and
credit-assignment limits of this reward on L2.

The **proximate greedy failure was action decoding**. Of 2,368 training
decisions, **971** were `preflight_invalid` and **250** were rejected by a
Factorio tool. The first greedy block repeated `INSERT` with the unused item
sentinel for all 80 decisions. The next four blocks repeated `PLACE` with
the unused prototype sentinel for all 320 decisions. In the sixth block,
the prototype became the real burner drill, but all 80 attempts targeted the
unplaceable `(101,-133)` tile. The final greedy policy repeated one tuple,
`(4,2,0,17,0,0,47,71,2,0,1,2)`, on **384/384** decisions across 24
episodes. Every one was rejected as an attempt to place the drill at
`(101,-133)`; it never reached `INSERT` or `WAIT`. Independent marginal head
masks cannot enforce a valid cross-head tuple or teach where the receiver
is. A conditional/autoregressive action decoder and stronger spatial
feasibility masking are the concrete next tests. This run does not show
that the production-score signal is wrong; it shows that **on-policy PPO
with this factorized PufferLib-compatible decoder did not learn L2 from
scratch** in this time and seed.

Raw episode/action/update records: [`ppo_production.jsonl`](ppo_production.jsonl).
The saved checkpoint is `.fle/runs/ppo_production.pt`.
The dedicated `fle_ppo_27098` container was stopped and removed after the
run; port 27099 was untouched. `ruff check
tests/benchmarks/run_ppo_production.py` passed.
