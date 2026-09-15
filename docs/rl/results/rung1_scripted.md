# Rung 1 scripted live bootstrap — 2026-09-13

**Gate: YES for scripted demonstrations.** The prior random-policy result remains
NO: 800 live transitions produced no automatic output. A normal masked scripted
trajectory now completes an empty-inventory iron line on the live server, and
`ReplayBuffer.relabel_future` produces non-zero copies from real observations.
This clears the specific rung-1 HER-candidate gate in [SCALING](../SCALING.md).
It does not show that an unseeded random policy can bootstrap or that a learner
can improve. No gradient updates or training were run.

## Method and action-space result

One open-world Factorio 2.0.73 container was run on RCON port 27097, with
`game.speed=10`, empty inventory, locked research, and the fixed scenario map
seed 44340. `GoalConditionedEnv.reset` cleared entities and inventory before
each episode. The behavior goal was 100 iron plates. All actions came from
`ScriptedBootstrap.next_action`; the runner independently checked every tuple
head against `build_masks(state, vocabulary, operation)`, called `plan_action`,
then passed the tuple to normal `GoalConditionedEnv.step` and `dispatch`.

**No action-space gap was observed in the completed iron-line sequence.** In
the saved four-episode batch, all **176/176 emitted actions** passed the real
conditional masks and all **176/176** dispatched with outcome `ok`; there were
0 masked actions, 0 `preflight_invalid`, 0 `tool_rejected`, and 0
`partial_mutation`. The mix was 108 MOVE, 12 HARVEST, 24 CRAFT, 8 PLACE,
12 INSERT, 8 WAIT, and 4 EXTRACT. Furnace placement, coal and ore insertion,
plate extraction, drill placement and fueling were all expressible. No
`SET_RECIPE` was needed for the stone furnace. The four same-seed validation
episodes added another 176/176 masked-and-`ok` actions.

The first two live probes exposed two scripted-policy assumptions:

1. With the original 32-tile resource search, the first reset showed 19,744
   resource tiles but no iron within that radius; nearest iron was at
   `(-16,-51)`, about 53.5 tiles from spawn `(0,0)`. The policy emitted zero
   actions and stopped with `no nearby iron ore`. Its default search radius
   was increased to 256 in `fle/rl/scripted.py`.
2. With the wider search, the policy mined 20 iron ore and travelled toward
   stone, but action 17 attempted `HARVEST` at `(55.5,-62.5)` from
   `(48.5,-62.5)`. The tuple passed the mask and normal dispatch, then the
   real tool rejected it: `Could not harvest. Nothing within reach to harvest`.
   `_position` now emits MOVE until the target is within one tile before
   HARVEST. This is a scripted reach error, not a missing action. The
   marginal `masks.py` offset mask admits uncertain geometry by design; the
   fake `harvest_resource` has no reach check, so it did not expose this error.

No change to `action.py` or `masks.py` was required. These probes show that a
mask-accepted action can still be tool-rejected for live reach. The original
fake-only 30 ore / 38 plate / 48 HER result is not a live measurement.

## Live production, timing, and HER

The final, saved batch used policy/reset seeds 10 through 13. Each episode
completed 44 actions and produced 15 automatic iron ore and 34 automatic iron
plates. Immediately before its last WAIT, each had 0 automatic ore and 20
automatic plates; that WAIT added 15 ore and 14 plates. Thus the live
drill-to-furnace line itself produced both items, beyond the hand-fed initial
smelting. Hand-mined ore, stone and coal registered zero automatic output.

| Seed | Actions | Automatic ore | Automatic plates | Future-positive boundary pairs | HER copies | Episode wall s, including reset | Game ticks | Game s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 44 | 15 | 34 | 351 | 165 | 48.59029024996562 | 12800 | 213.33333333333334 |
| 11 | 44 | 15 | 34 | 353 | 167 | 48.62826466700062 | 12861 | 214.35 |
| 12 | 44 | 15 | 34 | 351 | 165 | 47.08917370799463 | 12888 | 214.8 |
| 13 | 44 | 15 | 34 | 351 | 165 | 48.47037175000878 | 12858 | 214.3 |
| **Total** | **176** | **60** | **136** | **1406** | **662** | **192.77810037496965** | **51407** | **856.7833333333333** |

For one concrete episode, seed 10 took 48.59029024996562 wall seconds,
including a 25.356430334039032-second reset; game time was 12,800 ticks or
213.33333333333334 game seconds. Cold creation of the first connection in the
initial probe took 28.8803754170076 seconds, separately from episode wall
time. The final batch's warm connection took 0.2950200419872999 seconds.

The repeated-seed check ran four additional live episodes with seed 7. All
four completed the **identical 44-action sequence** and ended at 15 automatic
ore and 34 automatic plates, with 0 rejected actions. They produced 351
future-positive pairs and 165 HER copies each. Their game ticks were
`[12761, 12821, 12668, 12863]` and wall seconds were
`[48.654626375006046, 48.93825875001494, 47.16315358300926, 48.06462466699304]`.
The outcome and action sequence reproduced on this fixed map; exact timing
did not. `env.reset(seed=7)` does not change the scenario's map seed, so this
is not a test across different world generations.

## Ordinary replay and measured size

The saved batch used the same `env.trace.replay_kwargs()` and
`ReplayBuffer.add_episode` path as ordinary agent experience, followed by
`relabel_future` with its default `k=4`. There is no demonstration flag or
separate buffer. All four stored episodes are full and counter-valid. The
saved [replay artifact](rung1_scripted_replay.npz) reloads with **176 original
transitions, 662 HER overlays, and 838 total entries**. All stored action
outcomes are `ok`; 26 of the HER entries have an immediate positive reward
when decoded with `ReplayBuffer.get` (others may precede their relabelled
first hit). This confirms that real trajectories are relabelled by the normal
implementation.

This artifact uses the original 512-row, 38,495-float schema. The 64-row schema
supersedes it for new training; its fingerprint correctly rejects this file.
The archived NPZ and its compressed boundaries remain readable as historical
data under the original schema.

At 176 real transitions and 662 HER entries, the artifact is **74,977 bytes
on disk**. The live buffer contained **89,556 bytes of compressed observation
boundaries** and a **16,777,216-byte priority tree** at configured capacity
200,000. A protocol-5 pickle of the in-memory buffer was **16,935,723 bytes**.
In a fresh load process, resident memory rose from 117,063,680 to 142,884,864
bytes (delta **25,821,184 bytes**); `tracemalloc` reported 17,227,316 current
allocated bytes after load. These are measurements of a small, sparse
bootstrap factory. A straight-line projection of its pickle bytes excluding
the fixed tree is roughly 197 MB at 200,000 real transitions with the same
HER ratio, but it is **not** a mature-factory sizing result. The design's
10–14 GiB estimate is neither observed here nor ruled out for larger,
denser factory states.

Raw per-step logs: [initial no-iron probe](rung1_scripted_first.jsonl),
[out-of-reach probe](rung1_scripted_second.jsonl),
[first completed episode](rung1_scripted_third.jsonl),
[same-seed repeats](rung1_scripted_seeded.jsonl), and
[saved replay batch](rung1_scripted_replay.jsonl).

## Validation and cleanup

`pytest -q tests/rl`: 54 passed, 1 skipped. `ruff check` on
`fle/rl/scripted.py` and `tests/benchmarks/benchmark_scripted_live.py` passed.
The one-service Compose project `fle_scripted_27097` was brought down with
`--remove-orphans`; `docker ps -a --filter name=fle_scripted_27097` returned
no containers. Ports 27098 and 27099 were not used by this run.
