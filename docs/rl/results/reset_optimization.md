# RL episode reset optimization (2026-09-13)

## Measured bottleneck before the change

The repeatable, one-container legacy benchmark ran three 64-action episodes with the same seed and action sampler as rung 1. Its resets took **25.115589708 s, 25.094677708 s, and 24.465367917 s**. The first reset checks the vocabulary once; the next two show the recurring cost. All phases below were timed at the call site in `GoalConditionedEnv.reset`; the `server_reset` subcall was separately decomposed with Python method wrappers in a three-reset live probe. Values are milliseconds; small gaps to the total are Python bookkeeping and the entity-row check.

| Reset phase | Legacy episode 0 | Legacy episode 1 | Legacy episode 2 |
|---|---:|---:|---:|
| Shared `FactorioInstance.reset` plus RL counter-origin reset | 225.380 | 165.926 | 173.962 |
| Pause | 4.246 | 2.530 | 2.537 |
| Vocabulary check | 273.235 | 0.000 | 0.000 |
| Full-sync RCON call | **24512.882** | **24833.935** | **24197.680** |
| Parse full-sync records | 3.637 | 4.209 | 4.564 |
| Client cache rebuild | 85.881 | 65.428 | 65.702 |
| Redundant paused drain after full sync | 5.196 | 6.679 | 5.474 |
| Goal sampling and validation | 0.246 | 1.379 | 1.472 |
| Observation encoding | 3.877 | 4.296 | 4.079 |
| **Whole reset** | **25115.575** | **25094.655** | **24465.357** |

The legacy full-sync reply was **1,626,993 bytes** on each of these resets. It re-encoded every generated terrain chunk even though the map had not been replaced. Parsing and applying those bytes cost only 70–90 ms; the 24–25 s is in the RCON full-sync call. A separate initial live probe measured whole resets of **32.871160 s, 39.306550 s, and 30.950239 s**; their full-sync RCON calls took **32.276081 s, 39.031974 s, and 30.640058 s**. This variation changes the absolute saving, not the attribution. In that probe, the reset Lua tool took **0.113761 s, 0.154044 s, and 0.163465 s**; Python namespace reset, default game-control restoration, and score accounting together were small. Task/goal setup is under 2 ms on warm resets.

The expensive work was a fresh terrain snapshot, not server startup, map generation, task setup, or the FLE admin reset. The second paused drain was redundant after a paused full sync but saved only about 5–7 ms.

## Change and episode boundary

The RL environment now takes a full terrain snapshot on its first reset, then reuses that terrain cache on later resets of the same server. The new `obs_all_full_sync(false)` still replaces **all entity rows and dynamic protocol records**, but skips terrain chunk enumeration and leaves terrain event buffers intact. The client clears its entities, inventory, automatic counters, raw production and consumption counters, flow history, and technology cache, while retaining terrain. The `!full` marker and an exact equality check against the new `u` rows still reject ghost entities. The second drain was removed from the optimized reset.

The shared FLE reset remains the authoritative cleanup for player teleport, inventory, built entities, alerts, paths, production statistics, manual harvest/craft ledgers, force/research state, and `storage.elapsed_ticks`. The RL call passes `regenerate_resources=False` so ore/resource amounts on the persistent map remain consistent with the retained terrain cache. The new argument defaults to `True` through `FactorioInstance.reset` and the reset tool, preserving non-RL behavior. `reuse_terrain=False` on `GoalConditionedEnv` selects the original resource regeneration, complete terrain sync, and post-sync drain; it also supplied the matched legacy benchmark.

Each successful RL reset calls `obs_diff_reset_episode()` to establish a new automatic-counter origin, creates a new `ObservationClient` state for all non-terrain data, increments `epoch`, and starts the observation's episode tick from the current **true `game.tick`**. The goal generator still runs once each episode. A persistent map means mined resources remain depleted across episodes; this is suitable for the present open-track run but is not a fresh-map distribution. If map variation or inexhaustible starting patches become necessary, rotate two pre-warmed servers with different seeds and periodically refresh them outside the measured episode path. Do not claim a fresh map from this optimization.

## Live correctness checks

The [direct boundary probe](reset_correctness.json) deliberately created a furnace with unit **977**, moved the player to **(10, 6)**, gave **7 iron ore**, injected **12** raw iron-plate production and **5** manual units (3 harvested, 2 crafted), and verified `A_iron-plate = 7` before reset. After the optimized reset, unit 977 was absent, the player was at **(0, 0)** with empty inventory, automatic and raw production maps were empty (zero baseline), both manual ledgers were `{}`, and the epoch and trace epoch advanced from **1 to 2**. The encoded episode tick was **0**.

With `storage.elapsed_ticks` set to **987654321**, a live WAIT advanced true `game.tick` from **12861 to 12933**; the reported and trace episode ticks were both **72**, and `counter_valid` stayed true. The offline HER test `test_her_excludes_cross_epoch_and_invalid_episodes` passed, so the newly advanced epoch retains the replay exclusion rule. A separate live shared-reset check set a crude-oil resource to 7: the RL option left it at **7**, while a default `FactorioInstance.reset` restored it to **300000**. This verifies the old reset semantics remain available.

`pytest -q tests/rl` passed after the final cache test (54 passed, 1 skipped); the focused cache and environment tests also passed (12 passed). `ruff check` passed on the changed Python files. No learner or gradient update ran.

## Matched throughput

The [one-container legacy](reset_before_one.jsonl) and [optimized](reset_after_one_matched.jsonl) runs used the existing random-action benchmark, seed **500**, three **64-action** episodes, one server on **27099**, and `game.speed=10`. The table uses episodes 1 and 2 so both sides represent warm server/client episodes; the rate includes each episode's reset and action sampling. Initial full-terrain acquisition is shown below separately. The [two-container legacy](reset_before_two.jsonl) and [optimized](reset_after_two.jsonl) runs used the benchmark's matched-worker mode: ports **27099/27098**, two **64-action** episodes per worker, both taking a 1-game-second WAIT. Its active window begins after both initial full resets and includes each worker's later reset. One- and two-container absolute rates have different action mixes and should not be compared as scaling factors.

| Configuration | Reset seconds before | Reset seconds after | Steps/s incl. warm resets before | Steps/s incl. warm resets after |
|---|---:|---:|---:|---:|
| One container, 2 warm 64-action episodes | 25.094678, 24.465368 (mean **24.780023**) | 0.133461, 0.118068 (mean **0.125764**) | **1.880480** (128 / 68.067739 s) | **6.975047** (128 / 18.351131 s) |
| Two containers, 2 episodes/worker; one warm reset each | 25.514856, 25.609939 (mean **25.562397**) | 0.121612, 0.115770 (mean **0.118691**) | **5.891098** (256 / 43.455397 s) | **15.450562** (256 / 16.568977 s) |

The warm per-episode reset fell by about **197×** for one container and **215×** across the two measured workers. The observed end-to-end warm throughput gains were **3.709×** and **2.623×**, respectively; step cost still dominates once reset is cheap. On the first reset of a fresh client, the full terrain snapshot still took **25.115590 s before vs 24.395043 s after** in the one-container matched runs. Counting those first resets, all three one-container episodes yielded **1.862182 vs 3.656205 steps/s** (192 steps / 103.104860 vs 52.513468 s), excluding `FactorioInstance` construction. Counting the concurrent cold starts and first resets, both two-container runs yielded **3.024065 vs 6.132411 steps/s** (256 steps / 84.654273 vs 41.745406 s).

The optimized warm sync returned **4,761 bytes** and took **17.542 ms** and **13.357 ms** in the matched one-container run. Its full first sync returned **1,626,993 bytes** and took **23.939206 s**. Raw JSONL retains phase timings, ticks, outcomes, and per-step timings. A two-worker attempt made immediately after starting the second container failed RCON authentication during startup; its [two-line log](reset_two_startup_failure.jsonl) is separate from the successful matched run.

Both containers and the Compose network were removed after the measurements; `docker ps --filter name=fle_reset_opt` returned no containers.
