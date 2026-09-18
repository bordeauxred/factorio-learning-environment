# Deep RL on FLE: where to look

Five documents carry the current state. Everything else under `docs/rl/` is
history or detail.

| Read this for | File |
|---|---|
| Results and insights, updated as runs finish | `results/overnight_v1.md` |
| What algorithm to use next, and why (Codex research memo, 2026-09-16) | `research/13-algorithms-for-open-play.md` |
| The environment the learner sees (ops, heads, masks, observation, horizon) | `specs/learner-env.md`, then `specs/v1.md` for the v1 changes |
| Measured environment facts (reach, deaths, tool defects, score behaviour) | `results/op_census.md` |
| The running timeline, one entry per half hour | `results/overnight_v1_log.md` |
| Compute recipe for the Google Cloud VM (machine, setup, smoke test, launch commands) | `specs/compute-recipe.md` |
| The local dashboard (current arms on top, archived pre-11:20 history below) | `results/dashboard.html`, rebuilt by `tests/benchmarks/refresh_plots.sh` |
| Archived plots of everything before the 2026-09-17 11:20 fresh start | `results/overnight_v1_relocation_era/` (v4 arms and day-1 anchor); `results/first_curves/` (day 1) |

Also there when needed: `BRIEF.md` (the rules and the history of the failed
attempt), `results/first_curves.md` (the first day's curves and the v0 plateau),
`research/14-deep-isometry-rl.md` (assessed, verdict skip for now),
`specs/overnight-plan.md` (the plan and its Codex review).

## Runs and code

- Env: `fle/rl/env.py`; contract `fle/rl/schema.py`; learner `fle/rl/dqn.py`;
  greedy or stochastic eval `fle/rl/eval_policy.py`; PufferLib PPO
  `fle/rl/puffer/` (superseded, collapsed to inaction).
- Logs: `runs/<arm>/env_<port>.jsonl` (one line per step, one `episode_end`
  per episode), `runs/<arm>/metrics.jsonl`, checkpoints pruned to the newest
  two.
- Status: `python tests/benchmarks/night_status.py`. Plots and ladder:
  `python tests/benchmarks/plot_first_curves.py --out <dir> label=path,...`.

## Support-only execution guards

A support mask may remove only an action that the simulator will refuse
deterministically regardless of the policy's intent. An executor may navigate
to a target selected by the policy, as HARVEST and entity operations do, but it
must never replace a sampled target, entity slot, item, or PLACE offset with a
different one. Unsupported sampled combinations are zero-reward `no_support`
steps without a tool call; other infeasible combinations reach the tool and
become ordinary `tool_rejected` steps with the game's message.

PLACE therefore means exactly the snapped centre for `player + (dx, dy)`. Its
guard checks build reach at that centre and the selected prototype's rotated
footprint against water, entities, trees, rocks, and cliffs. The character tile
remains supported because `place_entity` moves the character aside. Ore is not
part of this guard: a mining drill sampled over grass is sent to the game and
rejected there. PLACE never searches for another offset and never approaches
ore.

The remaining execution guards are classified as follows:

| Guard | Classification | Default |
|---|---|---|
| Quantity (`--clamp-quantity`) | Judgement-call argument clamp: use the largest observable, coverable configured quantity no greater than the sampled quantity for the same recipe/item | On |
| Item (`--clamp-item`) | Policy-choice substitution when enabled; otherwise a missing/unaccepted INSERT item or absent EXTRACT item is `no_support` | Off |
| Entity anchor (`--clamp-anchor`) | Policy-choice substitution when enabled, including ROTATE; otherwise an empty, stale, or non-rotatable sampled slot is `no_support` | Off |
| CONNECT endpoints (`--clamp-connect`) | Policy-choice substitution when enabled; otherwise stale, identical, or connector-incompatible sampled endpoints are `no_support` | Off |
| Rejection cooldown (`--support-cooldowns`) | Support mask for an exactly repeated deterministic refusal until visible support state changes | On |

The quantity clamp remains on for now because it preserves the sampled
recipe/item and only reduces its count, but ingredient and item counts are
visible in the observation, so this is explicitly a judgement call rather than
a pure support-mask fact. `--quantity-aware-support` remains a compatible alias
for `--clamp-quantity`. Every quantity-clamped step logs
`requested_quantity` and `executed_quantity`.

After a rejected tool call, the exact complete action (the op and every head it
reads) is blocked until the mask-state fingerprint changes: player tile,
player inventory, or entity set. RESEARCH also excludes the current and queued
technologies as deterministic no-ops. No PLACE/PICKUP inverse is masked.

## Arm names

Arms are named by their learner and exploration, not by letters. Current runs (fresh
start 11:20 on the relocation-free environment) live under their own names:
runs/dqn-per, runs/dqn-per-ucbexplorer, runs/qrdqn-per-ucbexplorer-frontier,
runs/rainbow-optimisticper-n5. The earlier v4 runs of the same arms were
renamed runs/ctrl-v4-relocation-era, runs/armB-v4-relocation-era,
runs/armD-v4-relocation-era, runs/rainbow-v4-relocation-era; dqn-per-v1env =
runs/ctrl-v1 (day-1 anchor). Kept checkpoints: runs/_kept/. The full table is at the end of
`results/overnight_v1.md`.
