# L2 score-delta versus exact-readiness potential

## Paper and implementation audit

The FLE paper, Appendix A (p. 15 of `docs/assets/documents/paper.pdf`), defines
`PS(t) = Σ_i V(i)(P_i(t) − C_i(t))` over **items and fluids**. The score Lua in
`fle/env/tools/agent/score/server.lua`
forms the same net flow from Factorio's `input_counts − output_counts`, loops
over both item and fluid production statistics, prices each available entry,
and floors the final force score. It reads `game.surfaces[1]`, whereas the
paper's force-level equation does not specify a surface restriction; L2 has
only that surface. The live `PLACE → INSERT → WAIT` probe
reconciled to `0, −3, +40` automated-score deltas and 15 automatic iron ore:
one coal is charged at 3, and 15 ore contribute roughly `15×3.1 − 2×3` before
the scalar floor. The five paper seed prices match the Lua values exactly:
iron ore 3.1, copper ore 3.6, coal 3, stone 2.4, uranium ore 8.2; crude oil
0.2 also matches. Lua additionally seeds water and steam at 0.001, wood 3.2,
raw fish 100, and an unused `energy` price 1. Other unseeded raw resources get
2.5. The energy addition matches the appendix's
`ln(energy+1) × sqrt(ingredient cost)` form; Lua normalizes energy and
ingredients per product when constructing its recipe list.

**The price multiplier does not match the paper.** The appendix states
`α(n)=1.025^(n−2)` for `n` ingredients. A Lua `default_param()` helper contains
`ingredient_exponent=1.025`, but no score call uses that helper. Every active
no-argument `generate_price_list()` call instead falls back to
`default_ingredient_exponent()=2`. The forward recursion passes a recipe map
containing `n` ingredient keys **plus an `energy` key** to
`ingredient_multiplier`, which uses `table_size(recipe)−2`. Its effective
factor is thus `2^(n−1)` (for example, 2 for two ingredients), rather than
the paper's `1.025^(n−2)` (1 for two ingredients). The reverse missing-price
deduction applies the same helper to the original `recipe.ingredients` list,
so its exponent there is `n−2`, but this is an approximate fallback, not a
repair of the forward path. The pricing path takes the cheapest viable
allow-decomposition recipe, handles rocket-launch products separately, tries
reverse deduction and loop-forced fallback for unresolved prices, then skips
any flow with no price. These algorithmic details and per-product splitting
are not fully specified by the paper's compact equation.

The RL accessor reads the second score element, `automated_score`. Lua starts
with the floored player force score relative to a script-load baseline,
subtracts priced **recorded** manual harvest and net manual crafting since
that baseline, then floors again. The Python client may subtract its own
initial player-score baseline from the first element, not the automated one.
Thus the second element is not generally the paper's raw `PS(t)`; it is a
baseline-relative, manual-adjusted, floored variant. On this L2 fixture the
allowed operations exclude manual harvest and craft, and the live probe found
the two elements equal, so score deltas measure the repository's automatic
economic benchmark directly. Discounted sums of those deltas emphasize
earlier gains and are not identical to maximizing final undiscounted PS.

PR #414 adds the tiered observation wire and touches `INSERT`, `PLACE`, and
`ROTATE` for immediate observation refresh/build events. Its file list and
diff contain no score implementation change. These actions can still change
game production, which the existing score then measures. We deliberately did
not alter the benchmark's score pricing: doing so would make this run
incomparable with the prior score-delta arm. Consequently the experiment
targets the repository's published-metric implementation, with the paper/code
pricing discrepancy stated explicitly.
L2's live score changes involve only seeded iron ore and coal, so the
complex-recipe multiplier mismatch does not change those particular deltas;
the integer floor still applies.

## Exact potential and protocol

The server now emits `Y<P>:<F>` on **every** full sync and drain, computed from
live entities at the paused action boundary. `P` requires a player burner
mining drill's mining area to contain iron ore and its actual `drop_position`
to enter a player chest that `can_insert` one iron ore. `F` additionally
requires burning fuel remaining or a positive-fuel-value item in its fuel
inventory. Any eligible drill suffices; the server scans all drills rather
than the selected entity cache. The two bits occupy dedicated float32 lanes
and remain exact in replay; the schema fingerprint changes with the new block.
The live probe returned `1,1` for the scripted aligned fueled drill,
`0,0` after filling the chest, `1,0` after clearing the chest and fuel, and
`0,0` after destroying the drill. A missing or invalid exact record fails the
live environment boundary rather than silently using selected stale rows.

C1 uses `r=Δautomated_score`. C2 uses
`r'=Δautomated_score + β_t Φ(s') − Φ(s)`, with
`Φ(s)=20P(s)+20F(s)` and `β_t=0.99^((tick_next−tick_t)/300)`.
On actual episode truncation, `Φ(s')=0`; a goal hit alone does not erase it.
Both rewards are recomputed by replay from stored endpoints and observation
bytes. The shaping terms telescope to `−Φ(s_0)` over a complete discounted
episode; the unit test covers a fueled terminal state and save/load replay.
This invariance applies to the learner's **discounted score-delta return**;
that return already differs from undiscounted final paper PS because of
discounting and the audited Lua price/floor/manual adjustments.
The coal charge remains `−3` in the base score. For a placement-ready drill,
an insertion's shaped immediate reward is `40β_INSERT−23`, positive when
`β_INSERT>0.575`.

## Matched experiment

Both arms use the exact `ladder.md` L2 fixture: map seed 44340, one burner
drill and five coal in inventory, wooden receiving chest at
`(107.5,−128.5)`, 16 actions restricted to `WAIT`, `MOVE`, `PLACE`, `INSERT`,
and the unchanged **four automatic iron ore** success threshold. One
Factorio 2.0.73 container on port 27099, `game.speed=10`, the same branching
Double DQN and decoder, MPS batch 64, one update per real transition,
training seed 20260914, and no demonstrations or HER are used in both arms.
The fresh random baseline runs once before C1 and is shared because its
actions do not depend on reward. The observation is 22,817 float32 values in
both arms; the prior score arm had 22,815 and is historical context, not the
matched control. Each new training arm has a 1,500-second wall-clock window.

The fresh random baseline was **0/24** on four ore: 21 episodes ended at zero
ore and three at two, for six total and 0.250 mean ore. It fueled a drill in
3/24 episodes and had 0/226 preflight-invalid non-WAIT combinations.

### C1: unshaped `Δautomated_score`

C1 completed 82 training episodes, 1,312 real transitions and 1,312 learner
updates in 1,503.894 training-wall seconds (the final in-flight episode is
included). It had **0/82** four-ore successes, **46 total automatic ore**,
and **0.561 mean ore per episode**. The full final-ore distribution was
`0:58, 1:2, 2:22`. Thus the old run's exact 0/2 outcome pattern did not
hold here: two late-starting drills produced one ore. The binary event
“produced any automatic ore” occurred in **24/82 = 29.3%** of episodes.
It occurred in **5/41 = 12.2%** of the first half and **19/41 = 46.3%** of
the second. An exact two-sided permutation test of this binary event between
the fixed halves gives **p = 0.001311**. This is not a smoothed ore curve;
the corresponding raw ore totals are 9 and 37.

C1 successfully inserted coal into a drill in **24/82** training episodes,
reached aligned `P` in **0/82** and aligned fueled `F` in **0/82**, and had
**0/937** preflight-invalid non-WAIT decisions. The 15 periodic greedy
evaluations were 0/15 on four ore. Its final fresh greedy evaluation was
**0/24** on four ore: every episode fueled a drill and ended at exactly two
ore, for **2.000 mean ore**, with no aligned `P` or `F` event and **0/384**
preflight-invalid non-WAIT decisions. Total training net automated-score
deltas summed to +66; final-evaluation episode deltas summed to +72.

### C2: `Δautomated_score + βΦ(next) − Φ(now)`

C2 completed 90 training episodes, 1,440 real transitions and 1,440 learner
updates in 1,524.643 training-wall seconds (including its in-flight final
episode). It had **0/90** four-ore successes, **28 total automatic ore**,
and **0.311 mean ore per episode**. Its final-ore distribution was strictly
`0:76, 2:14`; the binary “produced any automatic ore” event occurred in
**14/90 = 15.6%** of episodes. The fixed halves were **7/45 = 15.6%** and
**7/45 = 15.6%**, each with 14 ore total. The exact two-sided permutation
test gives **p = 1.0**. No block size was selected to make a curve monotone.

C2 successfully inserted coal into a drill in **14/90** training episodes,
reached aligned `P` in **0/90** and aligned fueled `F` in **0/90**, and had
**0/1,120** preflight-invalid non-WAIT decisions. Its 15 periodic greedy
evaluations were 0/15 on four ore and all produced zero ore. The final fresh
greedy evaluation was **0/24** on four ore, **0/24** on any automatic ore,
**0/24** on drill fueling, and **0/384** on preflight-invalid non-WAIT
decisions. In each final episode it chose one successful `MOVE`, then repeated
an unplaceable `PLACE` at `(101,−133)` 15 times, yielding **360/384** Factorio
tool rejections. It had zero aligned readiness events. Total training net
automated-score deltas summed to +42; final-evaluation episode deltas summed
to 0.

**The shaping term was never activated on a collected C2 transition.** All
1,440 C2 training steps had `P=F=0` before and after, and their logged shaped
rewards equaled the base score deltas exactly. The same holds for its 15
periodic and 24 final episodes. This is a negative result about exploration
of the receiver-aligned state, not a test of whether a reached `P=1` state
would learn faster with this potential. The live scripted probe did reach
`P=F=1`, so the state and protocol are physically attainable and the lanes
can turn on.

## Verdict and limits

On this single seed, policy-invariant score shaping **did not improve behavior
on the unchanged four-ore objective**. Both arms had zero real goal successes
in training and 0/24 final greedy successes. C2 fueled drills less often
than C1 in training (**14/90 = 15.6%** versus **24/82 = 29.3%**) and in final
evaluation (**0/24** versus **24/24**). Its mean training ore was lower
(**0.311** versus **0.561**) despite more real transitions, and its final
greedy policy collapsed to a repeatedly rejected placement while C1's
produced two ore per episode. The potential cannot provide local credit if
exploration never discovers `P`; that is the binding observed limitation.
This one fixed map and one training seed do not establish an average effect
over seeds or layouts. Equal roughly 25-minute wall-clock windows yielded
different transition counts, so the outcome is a matched-time comparison,
not an equal-transition sample-efficiency claim. A separate 27098 container
appeared during the runs; it was not part of this experiment or modified by
it and could have affected wall-clock throughput.

Raw records: [live score probe](ps_shaping_probe.jsonl),
[C1 baseline, training and evaluation](ps_shaping_c1.jsonl), and
[C2 training and evaluation](ps_shaping_c2.jsonl). Each training JSONL was
confirmed to contain a live first transition before its logging gate was
released. Both complete runs recorded `run_end`; training-wall durations
were 1,503.894 and 1,524.643 seconds, respectively. The owned 27099
container was removed after completion and `docker ps -a --filter
name=fle_ps_shaping_27099` returned no containers. The experiment and audit
took approximately 69 minutes, below the 90-minute cap. `ruff check`,
`git diff --check`, and the RL test suite passed (68 passed, 1 skipped).
