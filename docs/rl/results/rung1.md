# Rung 1 — does the loop close (2026-09-13)

**Gate: NO.** The live random policy completed 13 episodes and 800 transitions, but
0 episodes produced automatic output, yielding 0 future-positive boundary pairs
and 0 HER relabels. No gradient updates or live training were run. The prescribed
next step is better scripted empty-start demonstrations or a readiness signal,
not more servers.

## Method and raw data

Open-world Factorio 2.0.73, one server on RCON 27099, `game.speed=10`, empty
inventory and locked research. The policy uniformly samples a currently enabled
operation and then independently samples its conditional marginal action masks.
WAIT uses 1- or 5-game-second targets. Budgets were one 32-action smoke episode,
ten 64-action episodes, and two post-fix 64-action episodes, with seeds 177, 178,
and 500 respectively. Every transition was flushed to JSONL; the file was checked
after the first 32-action episode (36 lines) before the longer sample. Raw logs:
[initial and ten-episode sample](rung1_one.jsonl),
[post-fix sample](rung1_postfix.jsonl), [targeted probe](rung1_probe.json), and
[two-worker sample](rung1_two.jsonl). Values in tables below are rounded; the
JSONL retains full precision.

## Loop correctness

All 13 episodes reset and ended at their action budget with `truncated=True`.
All 13 trace tick sequences (800 transitions) were monotonic, with 0
counter-invalid episodes.
The episode tick is derived from the v2 drain's true `game.tick`, not
`storage.elapsed_ticks`. In a direct live probe, `storage.elapsed_ticks` was set to
987654321; a WAIT advanced true `game.tick` from 51576 to 51644, the trace also
recorded 51644, the episode tick was 68, and overshoot was 8 ticks. A furnace
with unit 635 appeared in a full sync (`[634, 635]`), then disappeared after
reset (`[634]`): 0 ghost rows survived.

The one-container post-fix sample classified 100 actions as `ok` and 28 as
`tool_rejected`; 0 were `preflight_invalid` or `partial_mutation`. The rejected
HARVEST actions reported “Nothing within reach to harvest.” A rejected RESEARCH
call was also classified as a tool failure. The original live classifier had
marked 5 rejected HARVEST calls as `partial_mutation` across the first 672
transitions because it fingerprinted unrelated entity row changes while the game
ran. The fix scopes the before/after fingerprint to state the attempted macro
can change. The post-fix sample showed 0 such false partial outcomes; the offline
partial-mutation fixture still passes.

The first benchmark version mislabeled drain response bytes as total RCON bytes.
Those 672 raw rows now carry the accurate `drain_response_bytes` name; full
command-plus-reply payload bytes were instrumented for the 128 post-fix rows
and are reported below. An initial active-time UPS estimate could exceed the
requested 600 UPS because it omitted RCON wall time; the logged derived value
was corrected to true tick delta divided by whole-step wall time.

**Superseded for warm episode-reset throughput:** [reset optimization](reset_optimization.md) measured the legacy 64-action rate at 1.880480 steps/s including resets and the optimized rate at 6.975047 steps/s. The 1.873 steps/s below remains the original rung-1 measurement, not a current warm-reset estimate.

## One-container latency and throughput

Post-fix sample: 128 transitions, two 64-action episodes. `snapshot` is client
wire-record parsing; server-side snapshot serialization is included in `drain`.
`rcon_bytes` counts UTF-8 command and reply payload bytes, excluding TCP/RCON
framing. The effective WAIT UPS proxy is true tick delta divided by transition
wall time, not a server-reported UPS counter.

| Phase | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|
| Unpause | 3.393 | 5.415 | 3.513 |
| Tool dispatch (includes WAIT polling) | 75.899 | 524.635 | 130.537 |
| Pause | 2.028 | 10.273 | 2.889 |
| Drain | 2.099 | 3.136 | 2.085 |
| Snapshot parse | 0.004 | 0.007 | 0.004 |
| Client reconcile | 0.013 | 0.030 | 0.020 |
| Encode, excluding masks | 3.110 | 4.143 | 3.324 |
| Masks | 0.746 | 1.420 | 0.858 |
| Reward | 0.067 | 0.109 | 0.073 |
| Whole step | 86.039 | 535.040 | 143.318 |

| Quantity | Measured value |
|---|---:|
| Episode wall time, including resets and policy sampling | 68.339 s |
| Reset time, two resets | 49.686 s (25.142, 24.545 s) |
| Step wall time, 128 steps | 18.345 s |
| Steps/s including resets | 1.873 |
| Steps/s inside `env.step` only | 6.978 |
| True game ticks advanced | 9641 |
| Game-seconds/wall-second including resets | 2.351 |
| RCON payload bytes/transition, p50 / p95 | 626.5 / 2178.6 B |
| Drain reply bytes/transition, p50 / p95 | 75.0 / 194.3 B |
| Entity count, p50 / p95 | 1 / 1 |
| Selected relevant fraction, p50 / p95 | 1.0 / 1.0 (vacuous; no relevant machines) |
| WAIT transitions | 33 |
| WAIT overshoot, p50 / p95 / max | 4 / 6.4 / 9 ticks |
| Effective WAIT UPS, p50 / p95 | 555.445 / 577.816 |

Cold first contact was measured separately: creating the first
`FactorioInstance` took 35.832 s, and its first reset plus sync took another
23.350 s. This excludes Docker container startup. A later warm connection took
0.273 s. The ten 64-action episodes spent 233.779 s in resets and 74.001 s in
steps; reset cost is substantial at this short budget.

## Random-output and HER gate

| Cohort | Episodes | Actions | Episodes with automatic output | Future-positive pairs | HER relabels |
|---|---:|---:|---:|---:|---:|
| Initial smoke, seed 177 | 1 | 32 | 0 | 0 | 0 |
| Random sample, seed 178 | 10 | 640 | 0 | 0 | 0 |
| Post-fix sample, seed 500 | 2 | 128 | 0 | 0 | 0 |
| **Total** | **13** | **800** | **0** | **0** | **0** |

Per-episode HER relabels, in run order: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.
Per-episode future-positive pairs were also `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.
Automatically produced goal items: **none** of coal, copper ore, iron ore, or
iron plate. The policy never completed a HARVEST; from empty inventory the mask
only enabled WAIT, MOVE, HARVEST, and RESEARCH. It therefore never reached the
craft/place sequence needed for automatic production.

Across all random episodes, `invalid_combination_rate` (`preflight_invalid` /
all actions) was **0/800 = 0%**. `tool_failure_rate` (`tool_rejected` or
`partial_mutation` / all actions) was **208/800 = 26%**. In the post-fix subset,
the corresponding rates were **0/128 = 0%** and **28/128 = 21.875%**.
These are separate signals: the 15% trigger
for replacing independent action heads was not met, while tool rejection from
random HARVEST positions remains high.

## Two-container worker pool

Two open-world servers on ports 27099 and 27098 were driven through
`AsyncWorkerPool`. Both workers were allowed to finish reset before timing
started. Worker 0 repeatedly used a 1-game-second WAIT; worker 1 used a
15-game-second WAIT. This deliberately unequal workload tests whether a slow
worker blocks a fast one; it is not a matched-policy scaling comparison with
the random one-container sample.

| Quantity | Measured value |
|---|---:|
| Active transitions | 48 |
| Active wall time, excluding both cold starts/resets | 10.248 s |
| Aggregate steps/s | 4.684 |
| Worker 0 / worker 1 transitions | 45 / 3 |
| Fast-worker results before slow worker's first result | 13 |
| Action outcomes | 48 `ok` |
| Time until worker 0 / worker 1 first reset completed | 29.070 / 60.443 s |

The 13 fast-worker results before the first slow result show that completed
transitions arrive without a lockstep batch barrier. The two-worker aggregate
throughput was lower than the one-env `env.step`-only random throughput, but
the action mix and cold/reset treatment differ, so this run does not establish
a scaling factor.

## Validation and cleanup

`pytest -q tests/rl`: 53 passed, 1 skipped (live vocabulary test requires an
opt-in host). `ruff check` on the four edited/added Python files passed.
`docker compose -p fle_rung1 -f /tmp/fle-rung1-compose.yml down
--remove-orphans` removed both containers and the Compose network.
`docker ps --filter name=fle_rung1` returned no running containers.
