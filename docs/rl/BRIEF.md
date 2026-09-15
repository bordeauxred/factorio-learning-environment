# Deep RL on FLE: brief for a fresh start

Read this first. It is the condensed result of one full attempt that failed, kept
because the measurements are worth more than the code was. All code from that
attempt has been deleted; the branch is byte-identical to PR #414 in `fle/`.
The old work is recoverable at tag `archive/rl-attempt-2026-09-15`.

## The goal

Train a deep RL agent to maximize FLE's published production score on open play:

`PS(t) = sum over items of V(i) * (P_i(t) - C_i(t))`

Use the **automated** variant, `score()[1]`, which subtracts hand-mined and
hand-crafted output. That is what FLE's own Inspect open-play scorer reports
(`fle/eval/inspect/integration/solver.py:160,594`). PS is unbounded and monotone
in real industrial progress, so **PS is both the objective and the metric**.
Report PS distributions, never a pass/fail rate against a threshold.

For scale: an LLM agent driving FLE's Python tool API launches a rocket in under
two hours. The RL attempt described here produced 2 iron ore.

## The one thing that killed the last attempt

**The action space could not reach the first resource.** Measured across nine
unscripted runs: **0 successful harvests in 6,752 attempts.**

- `harvest_resource` gates server-side on `player.resource_reach_distance`,
  default 2.7 tiles (`fle/env/tools/agent/harvest_resource/server.lua:470`).
- The action's offset head spanned +/-8 tiles, so only **21 of 289** offsets were
  physically reachable. 1,797 of 1,800 rejections read
  "Nothing within reach to harvest".
- A quantity bin of `-1` meaning "all" is rejected outright by the tool, and
  killed the only attempts that ever passed the reach gate.

Offline decode of 21,540 stored observations gave the factorization
`0.44% coverage x 8.3% op x 3.8% offset x 75% quantity = 1.05e-5` per action,
predicting 0.23 harvests per 90 minutes. Observed 0. Model and data agree.

Everything else measured in that attempt was measured through this filter.

Two further structural defects from the same audit:

- **50.3% of all steps were preflight-invalid** with a single cause: six of
  twelve operations require a real entity anchor, and open play starts with an
  empty world.
- **73.8% of wall-clock went to WAIT** in a world where nothing was running.
  HARVEST, the only operation that can produce a first item, got 2.1%.

**Do not write a line of learner code until a random policy can acquire a
resource on a live server.** Verify that first, with the game running.

## What is true about the environment

- `fle/env/` is **fine**. It is the API LLM agents use to launch rockets. The
  last attempt's failure was entirely in the tensor interface built on top.
  Do not modify `fle/env/tools/` without a specific, measured reason.
- PR #414's observation protocol is a good foundation. Drain costs ~2.1 ms.
  Architecture is sound; the benchmark's tensor sizing is not a spec.
- `GameState` save/restore works, at ~0.13 s once terrain is reused. It captures
  world state but **not** the production statistics ledger, so any restored state
  starts from a fresh score baseline. Compare only identity, position and
  orientation when checking a restore; entity **contents and recipe are volatile**
  (a smelting furnace changes every tick and reports a derived recipe).
- A cold reset with full terrain sync is expensive. Reusing terrain across
  episodes took reset from 24.8 s to 0.126 s. Get this right early.
- Live-calibrated and trustworthy: raw `input_counts` is production,
  `output_counts` is consumption. Hand-mining and hand-crafting earn **zero**
  automated credit; a burner drill and a stone furnace earn positive credit.
- The checked-in vocabulary at `data/rl/vocab/factorio-2.0.73-base-v1.json` is a
  real live export: 105 prototypes, 253 items, 219 recipes, 197 technologies,
  67 status codes. Vocabularies are **not** statically enumerable from the repo.
- Known discrepancy worth reporting upstream: the paper specifies a complexity
  multiplier of `1.025^(n-2)`; `score/server.lua` uses `2^(n-1)`.

## What was tried, and what it cost

| Approach | Outcome |
|---|---|
| PPO, entropy bonus, open play | 186 episodes, PS **0.00**, zero entities built |
| DQN + HER, goal-conditioned | Worked on toy fixtures, irrelevant on open play |
| Count-based novelty, behaviour-only | Real gain: working-drill episodes 13/80 to 47/80, p=3.8e-8 |
| Four-hour runs | Completely flat; sample budget is not the blocker |
| Larger action set (+ROTATE/PICKUP) | **Worse.** Policy collapsed onto MOVE 100% |
| Potential-based shaping on the hard step | No effect; the potential was as sparse as the goal |
| Go-Explore, achievement cells only | **1 cell, ever** |
| Go-Explore, + coarse position cells | **296 cells** from a bare root, reached the ore patch, still **0 items** |
| Go-Explore, scripted seed | 23 cells, PS **35.0**, crafted gears and a burner drill |

The only configuration that ever automated anything used a **scripted seed**,
which the user rejected on bitter-lesson grounds. Every unscripted configuration
scored 0.00.

## Hard-won rules

1. **Verify the environment on a live server before any learner.** The single
   check that matters: can a random policy acquire an item? Baseline 0/6,752.
2. **Read the interface before writing against it.** Most of the lost time came
   from guessing: an offset scale that broke a working component, a restore
   contract that asserted time stood still, a verification script that read the
   wrong tuple index and then the wrong dict key.
3. **Support masks only.** Remove what the simulator deterministically refuses.
   Never encode what a designer thinks is a good idea. The user rejects dynamic
   candidate generation and semantic admission rules.
4. **No scripted bootstrapping.** Every time it was used, the problem reappeared
   further up, and it capped depth at whatever the script demonstrated.
5. **No reward tampering.** Keep the published objective. Intrinsic bonuses may
   affect the behaviour policy only, never the replay reward or Q target.
6. **Never select a smoothing parameter that flatters a curve.** One "monotone"
   curve was monotone only at one block size out of four, and had to be withdrawn.
7. **Judge stochastic processes on long windows.** Discovery was called
   "plateaued" twice on windows that were far too short.

## Where to start

1. Stand up a container and measure, with a random policy, how often each of the
   twelve operations succeeds and why it fails. Publish the table. That table is
   the environment specification, and the last attempt never produced it.
2. Fix whatever makes the first resource unobtainable. The design intent from the
   project owner is a parameterized-action MDP with **macro-actions that absorb
   navigation**, so reach is an internal execution invariant rather than a
   coordinate the policy must guess. A visible patch 40 tiles away should be
   obtainable in one decision.
3. Re-verify: random policy acquires an item. Only then launch learners.
4. Run PPO and an off-policy learner on the same objective, report PS
   distributions against a random baseline, and report recipe depth reached.

## Reference

Design and audits are in `docs/rl/`. The most load-bearing files:

- `research/01-action-space.md`, `research/02-observation-space.md`: code-grounded
  audits of what FLE exposes, with `file:line` citations.
- `research/11-cracking-exploration.md`: the audit that found the 0/6,752 harvest
  failure and the probability factorization.
- `research/12-astra-env-modelling.md`: environment redesign specification. Its
  diagnosis is sound; its implementation was not completed successfully and was
  deleted with the rest.
- `results/`: measured outcomes of every experiment listed above.
