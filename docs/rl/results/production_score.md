# L2: sparse goal/HER versus automatic production reward

This is a one-seed (`20260913`), fixed-map (`44340`) head-to-head experiment on
Factorio 2.0.73. Each fresh L2 reset put one burner mining drill and five coal
in the player's inventory, teleported the player to `(103,-128)`, and placed a
wooden receiving chest at `(107.5,-128.5)`. The drill had to be placed on ore,
fuelled, and run within 16 actions. The success criterion was at least four
**automatic iron ore**. All arms used one container on RCON port 27099,
`game.speed=10`, the 22,815-value observation with 64 entity rows, the same
branching Double DQN, MPS, batch size 64, one update per real transition,
epsilon schedule, and no demonstrations. Greedy evaluation used epsilon zero
on fresh resets. The random baseline was run once with the control, since its
actions do not depend on reward. The control used sparse first-hit reward and
future HER (`k=4`); the score arm used the change in `automated_score`, with
the goal input zeroed and no HER. The positive-production variant likewise
had no goal input or HER and rewarded the increase in the calibrated
per-item automatic-production counters, summed over items.

## Live reward verification

The scripted L2 probe used the same paused reset and then `PLACE` at
`(106,-128)` facing east, `INSERT` five coal, and `WAIT` 60 game seconds.
Both FLE score accessors returned the values below; the second is
`automated_score`, which removes recorded hand-harvest and hand-craft
contributions. In this no-manual-action probe, the two accessors agreed.

| Boundary | Player score | Automated score | Automated score delta | Automatic iron ore |
|---|---:|---:|---:|---:|
| Reset | 0 | 0 | — | 0 |
| After `PLACE` | 0 | 0 | 0 | 0 |
| After `INSERT` | -3 | -3 | -3 | 0 |
| After `WAIT` | 37 | 37 | +40 | 15 |

Placement itself did **not** lower the score, despite removing the drill from
inventory. Fuel insertion did: the score is a priced net production-minus-
consumption balance. The 15 ore then increased it by 40, leaving a net +37.
The positive-production variant was required because the honest score-delta
reward penalizes the fuel action in the successful sequence. The raw counter
direction and manual-flow subtraction were established in the prior
[reset calibration](reset_optimization.md); this probe also checked that a
running drill increments the automatic iron-ore counter while placement and
fuel insertion do not. The scalar score remains a priced balance, not a
per-item proof of production, so L2 success uses the automatic iron-ore
counter independently.

## Learning curves and results

The fresh random baseline reached **0/24** L2 successes and produced eight
automatic ore total (0.333 per episode). It used the same repaired decoder
and masks as the learned policies. The historical no-demonstration L2 result
was also 0/24, but ran before the three decoder/schema fixes, so the fresh
control below is the clean comparator for this experiment.

### Sparse first-hit goal with future HER (control)

| Training episodes | L2 successes | Automatic ore produced | Mean ore/episode | HER immediately positive / copies | Fresh greedy evaluation after block |
|---:|---:|---:|---:|---:|---:|
| 1–25 | 0/25 | 10 | 0.400 | 21/112 | 0/5 |
| 26–50 | 0/25 | 2 | 0.080 | 5/53 | 0/5 |
| 51–75 | 0/25 | 8 | 0.320 | 19/174 | 0/5 |
| 76–88 | 0/13 | 6 | 0.462 | 12/48 | — |

The control made **1,408 updates** on 1,408 real transitions during a
**1,359.864-second** training phase, including the in-flight final episode.
Its 88 training episodes were **0/88** on the assigned 4-ore goal and produced
26 automatic ore total (**0.295/episode**). Partial traces generated 57
immediately positive HER copies among 387 copies, but no assigned-goal reward.
Its final fresh greedy evaluation was **0/24**. Every final episode executed
an `ok` `PLACE` and `ok` `INSERT`, then produced exactly **2 automatic ore**
while repeating `MOVE`; all 24 stopped below the 4-ore goal. Thus the repaired
decoder avoided the old invalid-action collapse and learned a partial
place-and-fuel sequence, but did not learn the receiver-aligned placement.

### Automated score delta (treatment)

| Training episodes | L2 successes | Automatic ore produced | Mean ore/episode | Net automated-score reward | Fresh greedy evaluation after block |
|---:|---:|---:|---:|---:|---:|
| 1–25 | 0/25 | 8 | 0.320 | +12 | 0/5 (0 ore) |
| 26–50 | 0/25 | 28 | 1.120 | +42 | 0/5 (2 ore each) |
| 51–74 | 0/24 | 22 | 0.917 | +33 | — |

The score arm made **1,184 updates** on 1,184 real transitions in the same
1,350-second training-window target, with no HER copies. Its **74 training
episodes** were **0/74** on L2, but produced 58 automatic ore total
(**0.784/episode**) and accumulated +87 net automated-score reward. The
training-episode count differs from the control because equal wall-clock
budgets do not imply equal numbers of environment steps. Greedy evaluation
was 0/5 after 25 episodes, choosing repeated `MOVE`, then 0/5 after 50,
choosing `PLACE → INSERT → WAIT` and producing exactly 2 ore in each episode.

The saved checkpoint reached **0/4** in a separate final set of fresh greedy
episodes. Each placed a north-facing drill away from the receiver, fuelled it,
produced exactly 2 automatic ore, and stalled while mostly moving and waiting.
The four episodes each ended at automated score +3: −3 from fuel, +6 from
the ore. Final evaluation was stopped after four identical 2-ore trajectories
to keep the required third arm inside the 90-minute total. The checkpoint
was already saved; the interrupted process therefore has no `run_end` row.
This final sample is smaller than the control's 24 and limits the strength of
the direct 0/24 versus 0/4 comparison.

### Positive automatic production only (additional treatment)

| Training episodes | L2 successes | Automatic ore produced | Mean ore/episode | Positive-production reward | Fresh greedy evaluation after block |
|---:|---:|---:|---:|---:|---:|
| 1–25 | 0/25 | 4 | 0.160 | +4 | 0/5 (0 ore) |
| 26–50 | 0/25 | 16 | 0.640 | +16 | 0/5 (2 ore each) |
| 51–75 | 0/25 | 40 | 1.600 | +40 | 0/5 (2 ore each) |
| 76–80 | 0/5 | 10 | 2.000 | +10 | — |

This variant made **1,280 updates** on 1,280 real transitions in
**1,359.261 seconds**, with no HER copies. Training reached **0/80** L2
successes and produced 70 automatic ore (**0.875/episode**). Its final fresh
greedy evaluation was **0/5**, again producing exactly 2 ore per episode
after an `ok` `PLACE` and `ok` `INSERT`. The five final episodes each earned
+2 positive-production reward. This removes the fuel-consumption penalty
from the reward, but does not solve the receiver-alignment problem.

## Head-to-head finding and failure mode

| Configuration | Training success | Mean automatic ore / training episode | Final fresh greedy L2 success | Final automatic ore / episode | Invalid combinations / non-WAIT training actions | Tool failures / non-WAIT training actions |
|---|---:|---:|---:|---:|---:|---:|
| Fresh random baseline | — | — | 0/24 | 0.333 | 0/243 (0%) | 44/243 (18.11%) |
| Sparse first-hit + future HER | 0/88 | 0.295 | 0/24 | 2.000 | 0/1,108 (0%) | 197/1,108 (17.78%) |
| Automated score delta, no goal/HER | 0/74 | 0.784 | 0/4 | 2.000 | 0/872 (0%) | 157/872 (18.00%) |
| Positive production only, no goal/HER | 0/80 | 0.875 | 0/5 | 2.000 | 0/944 (0%) | 177/944 (18.75%) |

**No: directly optimizing FLE's automated production-score delta did not
learn L2 from scratch in this run.** It more than doubled mean automatic
training output versus the sparse control (0.784 versus 0.295 ore/episode),
but neither it nor the positive-only variant reached the 4-ore goal in any
training episode or final greedy episode. The smaller final samples for the
treatments limit a statistical comparison with 0/24; they do document the
learned policy's repeated 2-ore failure on fresh resets of this fixed map.
The single training seed and map are not a multi-seed reliability estimate.

The reward is informative for **partial production** but still arrives only
after placement and fueling. The honest score additionally assigns −3 to
fueling before any ore appears. Across all three final policies, the first
actions were `PLACE → INSERT → MOVE`; the automatic counter rose to 2, then
stopped. The control placed a west-facing drill at `(107,-131)`, the score arm
a north-facing drill at `(108,-122)`, and the positive-only arm a west-facing
drill at `(108,-130)`. None delivered output into the scripted chest at
`(107.5,-128.5)`. In contrast, the live successful probe placed east-facing
at `(106,-128)` and reached 15 ore. The learned behavior is therefore a
productive but degenerate **place/fuel/stall** policy, not a sustained
`PLACE → INSERT → advance time` solution.

With the three recent fixes in place, the old measured **40.07%** preflight
invalid-combination rate fell to **0%** in all three current training arms.
Final greedy decisions were also 0/384 for the control, 0/44 for the score
arm, and 0/80 for the positive-only arm. That does not mean all actions were
useful. Around 18% of non-WAIT
training actions were rejected by Factorio, often for an unplaceable or
out-of-reach drill position or an incompatible `INSERT`. The binding failure
here is exploration of the precise placement/orientation needed to feed the
receiver, followed by weak reward discrimination among the much more common
2-ore placements. The factorized action representation still permits many
locally valid but physically poor placements; this experiment does not
isolate it from exploration as a sole cause.

Raw records: [live reward probe](production_score_probe.jsonl),
[control and random baseline](production_score_goal.jsonl),
[automated-score arm](production_score_score_delta.jsonl), and
[positive-production arm](production_score_positive_production.jsonl).
Every run's JSONL grew to a live first transition before release. The two
complete runs recorded `run_end`; the score arm's log ends after its fourth
final evaluation because it was stopped after saving the checkpoint. The
three runs used one container on port 27099; it was removed at completion,
and `docker ps -a --filter name=fle_production_27099` returned no containers.
The end-to-end experiment took approximately 84 minutes, below the
90-minute cap. `ruff check` and `git diff --check` passed. The RL replay,
policy and environment test selection passed (38 tests), including the new
stored-reward and replay-round-trip checks.

## Correction: the "monotone curve" was a block-size artifact (2026-09-14)

An earlier verbal summary of this run described the score-delta arm's training
curve as monotone, quoting mean automatic ore per 20-episode block as
0.30, 0.60, 1.10, 1.29. That framing does not survive checking and is withdrawn.

The per-episode data is **binary**: every one of the 74 training episodes ended
at either 0 or 2 automatic ore, because a drill either runs or it does not. Mean
ore per block is therefore just the fraction of episodes with a working drill,
scaled by two, and its smoothness depends entirely on the block size chosen:

| Block size | Mean ore per block | Monotone increasing |
|---:|---|---|
| 10 | 0.0, 0.6, 0.6, 0.6, 1.8, 0.4, 1.2, 1.5 | no |
| 15 | 0.27, 0.53, 1.07, 0.80, 1.29 | no |
| 20 | 0.30, 0.60, 1.10, 1.29 | yes |
| 25 | 0.32, 1.12, 0.92 | no |

Only block size 20 is monotone. Reporting that one is selection on the
smoothing parameter, which is the same error as smoothing a curve to look
better than the data.

**The underlying effect is real and should be stated this way instead.** The
fraction of training episodes producing a working drill rose from **24% in the
first half to 54% in the second half**, a difference of 0.595 mean ore with a
permutation-test p of **0.0090** over 10,000 shuffles. That is a significant
improvement under the dense production reward, against a sparse-goal arm whose
return was exactly 0.00 throughout. It is a real learning signal, not a smooth
learning curve, and the distinction matters because the agent is learning a
near-binary skill rather than improving a continuous quantity.
