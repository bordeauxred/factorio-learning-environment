# Where the deep-RL work stands, 2026-09-14

One sentence: **the agent learns real Factorio skills from scratch, and the
remaining gap is a single rare-discovery step that only demonstrations have
closed so far.**

## Learning results, all with exploration off on fresh episodes

| Task | Random | Trained | Note |
|---|---:|---:|---|
| L0, drill already running, 8 actions | 16/32 | 32/32 | untrained network 0/32 |
| L1, unfuelled drill + coal, 12 actions | 1/32 | **32/32** | learned INSERT then advance time |
| L2, must place the drill, 16 actions | 0/24 | 0/24 | from scratch, fails |
| L2 with 96 demonstration transitions | 0/24 | **32/32** | demonstrations close it |
| **L2 sub-skill: place on ore and fuel it** | **11/24** | **24/24** | from scratch, candidate scoring |

The last row is the strongest from-scratch result. With complete-action
candidate scoring, no demonstrations and the unmodified `delta PS` objective,
fresh greedy evaluation placed a drill on ore and fuelled it in **24/24**
episodes against **11/24** for a random policy over the same candidate set.
Training rose from 63/158 to 131/158 between halves, exact permutation
p = 2.68e-15, with quintiles 46%, 27%, 63%, 76%, 94%.

## What remains unsolved, and what it is

L2's four-ore goal requires the drill's output tile to feed the receiving
chest. A burner drill jams at `waiting_for_space_in_destination` after two ore
otherwise, which is genuine Factorio behaviour. Alignment occurred in
**1/316** episodes here and 5-6 per 800-1200 in the overnight runs. It never
improved with training in any arm.

The aligned action was always available: PLACE candidate recall was 1.0 at
every reset, so nothing was truncating it away. The agent simply never
discovers it often enough to learn it.

## Hypotheses tested and eliminated

| Hypothesis | Test | Result |
|---|---|---|
| Reward too sparse | dense `delta PS` vs sparse goal indicator | Helps. 24% to 54% working drills, p = 0.0090. Sparse gave exactly 0.00 return |
| Invalid action decoding | greedy-selection and sentinel fixes | Fixed. 40.07% to 0% invalid |
| Exploration too weak | count-based novelty, behaviour-only | Helps a lot. 13/80 to 47/80, p = 3.8e-8, first two four-ore successes |
| Not enough samples | two four-hour runs, 1,226 and 793 episodes | No effect. Quartiles flat, p = 1.00 and 0.51 |
| Action set too small | added ROTATE and PICKUP | Worse. Fuelled 52.7% to 15.2%, policy collapsed to MOVE |
| Additive Q cannot represent conjunctions | complete-action candidate scoring | Big gain on the sub-skill, no gain on alignment |
| Reward needs shaping | policy-invariant potential on alignment | No effect. The potential was as sparse as the goal, so it was zero on every transition |

## What this says

Four of the five stages of the task are learned reliably from scratch. The
fifth, spatial alignment of an output tile to a receiver, is a needle-in-a-
haystack discovery that no amount of reward density, exploration bonus,
compute, action-set breadth or action representation has converted into a
learned preference on its own. Demonstrations do convert it, which is the
result the original design predicted.

Nothing was tampered with to get here. The goal stayed at four ore, the reward
stayed FLE's published `PS(t) = sum of V(i)(P_i(t) - C_i(t))`, and intrinsic
exploration bonuses never entered replay rewards or Q targets.

One correction is recorded in `production_score.md`: an earlier claim that a
training curve was monotone was an artifact of block size and has been
withdrawn.

## Score implementation note

`fle/env/tools/agent/score/server.lua` matches the paper on seed prices, net
production minus consumption, and the square-root energy term. It differs on
the complexity multiplier: the paper specifies `1.025^(n-2)`, the Lua uses
`2^(n-1)`. Worth raising upstream.

## Correction, 2026-09-14: the goal threshold should never have been the metric

Every experiment above was scored against L2's "produce 4 automatic iron ore"
threshold. That threshold is a leftover from the goal-conditioned HER design,
which was abandoned when the reward became `delta PS`. Once the objective is
the production score, **the production score is the metric**, and a pass/fail
threshold measures the wrong thing.

Restated in PS units, the candidate-scoring run reads very differently:

| | Mean PS per episode |
|---:|---:|
| Random baseline | 0.38 |
| Trained, fresh greedy eval | **3.00** (median 3.00, min 3.00, max 3.00, n=24) |

Training quintiles: 1.38, 0.71, 1.90, 2.29, 2.81. That is a **7.9x gain on the
actual objective**, perfectly consistent across all 24 fresh episodes. It was
previously reported as "0/24 goal success", which was misleading.

PS = 3.00 is small only because the fixture caps it. Two ore at V = 3.1 minus
one coal at V = 3.0 is about 3.2, and the episode is 16 actions with a single
drill. The ceiling is the fixture, not the agent.

The same applies to the "alignment" problem. A jammed drill is a failure only
against a four-ore threshold. Against PS, a jammed drill is simply a small
positive, and the way to a larger PS is more production of any kind, which is
what the open track actually rewards.

`PS(t) = sum of V(i)(P_i(t) - C_i(t))` is unbounded and monotone in real
industrial progress. Maximizing it directly, with no goal and no threshold, is
the benchmark. That is now the running experiment.
