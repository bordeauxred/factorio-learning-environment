# Overnight L2 runs, 2026-09-14: both negative, failure now isolated

Two matched four-hour runs on L2 from scratch. Goal unchanged at 4 automatic
iron ore, reward unchanged as raw `delta PS`, count-based novelty exploration
affecting behaviour only, no demonstrations. One variable between them: the
action set. The MCP wrappers timed out at four hours but both training
processes ran to completion and their logs are intact.

| | Run A: + ROTATE + PICKUP | Run B: control, 4 ops |
|---|---:|---:|
| Training episodes | 1,226 | 793 |
| Fuelled drill | 186 (15.2%) | **418 (52.7%)** |
| Aligned drill | 6 (0.5%) | 5 (0.6%) |
| 4-ore goal, training | 1 | 3 |
| 4-ore goal, greedy eval | 0/60 | 0/35 |
| Goal rate 1st vs 2nd half | 0.002 vs 0.000, p = 1.00 | 0.003 vs 0.005, p = 0.51 |
| Distinct PLACE tuples tried | 786 | 382 |

## Three findings, all negative and all useful

**1. Four hours bought nothing over twenty-two minutes.** Neither run shows any
trend. Goal rate by quartile in Run A is 0/305, 1/305, 0/305, 0/305; in Run B it
is 1/197, 0/197, 1/197, 1/197. Permutation tests between halves give p = 1.00
and p = 0.51. Sample budget was a live hypothesis and it is now falsified at
this scale: roughly fifteen times the episodes produced no improvement.

**2. A larger action set made things worse, not better.** Run A had ROTATE and
PICKUP available, which was supposed to let it fix a misaligned drill. Instead
its fuelled-drill rate fell from 52.7% to 15.2%, and it reached fewer aligned
drills per episode than the control. The extra operations diluted exploration
without being used productively.

**3. The real failure is a degenerate attractor, not exploration.** Run A's
policy collapsed onto MOVE: 82% of training actions and **100% of greedy
evaluation actions**. Under `delta PS`, MOVE is always valid, always advances
time, and never pays a penalty. INSERT costs -3 for burned coal, and PLACE
risks producing nothing. With alignment occurring in about 0.5% of episodes,
the agent essentially never experiences the +40 payoff, so that value never
propagates back far enough to make the -3 worth paying. "Do nothing risky"
is a stable local optimum, and a bigger action space makes it easier to fall
into.

This does not contradict the earlier point that a discounted return should
handle an intermediate cost. It does, but only once the agent has experienced
the full sequence often enough for the value to propagate. At 0.5% it has not.

## What this points at

Exploration alone is insufficient. Count novelty already raised working-drill
episodes from 13/80 to 47/80 (p about 3.8e-8) and found the first two four-ore
successes, but neither longer training nor more actions converted that into a
policy.

The evidence now converges on the diagnosis in
`docs/rl/research/08-action-space-rethink.md`: the agent must discover a
specific conjunction of prototype, anchor, offset and direction, and the
additive Q, `V(s) + A_op + mean of active argument advantages`, cannot
represent that conjunction. Run A tried 786 distinct PLACE tuples and found
alignment six times. That is a search problem the current representation
cannot convert into a learned preference.

The recommended fix is already written up: enumerate complete, locally valid
action candidates at each boundary and score whole tuples, which makes the
Bellman max exact over the offered set and lets a small interaction network
represent "this prototype at this offset relative to this anchor feeds that
chest". Demonstrations remain the proven alternative, taking L2 from 0/24 to
32/32 with 96 transitions.

## Standing results unaffected

L0 32/32 against 16/32 random and 0/32 untrained, L1 32/32 against 1/32 random,
and L2 with demonstrations 32/32 against 0/24 all still hold. The from-scratch
L2 gap is what remains open.
