# Semi-Markov simulation probe

Measured on 2026-09-18 against the live headless server on RCON port 27004. The exact measurement invocation was:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/benchmarks/smarq_sim_probe.py --port 27004
```

The environment variable prevented Python bytecode; the shorter command is also accepted. Timings use `time.perf_counter()`. The probe used `FactorioInstance(fast=True, all_technologies_researched=False, inventory={}, reset_speed=10)` and `reset_position=True`. It restored port 27004 to an unpaused, speed-10 fresh reset at exit.

## Q1: pause correctness

`game.tick_paused=true` froze both tested states. The working-state setup was one stone furnace at `(4,0)` with 50 coal and 50 iron ore. Before pausing it reported status `1`, crafting progress `0.3072916667`, and zero output plates. Those values were unchanged afterward.

| State | Tick before | Tick after | Tick delta | Measured wall interval (s) |
|---|---:|---:|---:|---:|
| Fresh reset | 45,046 | 45,046 | 0 | 3.0062 |
| Fuelled furnace | 45,114 | 45,114 | 0 | 3.0120 |

RCON remained operational while paused: a command returned `pong@45114`. Thus the decision boundary can be frozen while the controller reads state and issues RCON commands.

## Q2: speed ceiling

Each row is a three-second near-empty-map window. “Game-s/wall-s” is realized ticks/s divided by 60.

| Requested speed | Engine-reported speed | Tick delta | Wall (s) | Ticks/wall-s | Game-s/wall-s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 181 | 3.0161 | 60.01 | 1.00 |
| 10 | 10 | 1,761 | 3.0032 | 586.38 | 9.77 |
| 40 | 40 | 3,500 | 3.0023 | **1,165.77** | **19.43** |
| 100 | 100 | 3,335 | 3.0067 | 1,109.20 | 18.49 |
| 400 | 400 | 3,355 | 3.0104 | 1,114.46 | 18.57 |
| 1000 | 1000 | 3,400 | 3.0115 | 1,129.02 | 18.82 |

The measured ceiling occurs at requested speed 40. Increasing the request to 100, 400, or 1000 did not increase realized throughput; all three were below the 1,165.77 ticks/s peak. Subsequent tests therefore used speed 40.

## Q3: exact tick advance

Both methods were run for ten trials at each horizon. The polling method read `game.tick` repeatedly and paused after observing the target. The event method registered `script.on_nth_tick(1)`, compared `event.tick` with a target stored in `storage`, paused in Lua, and removed the handler after the block.

| Method | Requested ticks | Mean overshoot | Max overshoot | Mean wall (s) | Max wall (s) |
|---|---:|---:|---:|---:|---:|
| RCON polling | 600 | 7.00 | 9 | 0.5760 | 0.5917 |
| RCON polling | 3,600 | 7.00 | 9 | 3.4289 | 3.7909 |
| Lua `on_nth_tick(1)` | 600 | **0.00** | **0** | 0.5698 | 0.5922 |
| Lua `on_nth_tick(1)` | 3,600 | **0.00** | **0** | 3.5070 | 3.7456 |

All 20 Lua trials landed exactly. `FAST_FORWARD(N)` should use the Lua target handler; RCON polling is measurably inexact.

## Q4: cost versus factory size

The factory state was exactly one stone furnace with 50 coal and 50 iron ore. Durations were sequential within each state: 3,600 ticks, then another 36,000 ticks. The furnace ended with 50 plates, progress zero, and status `18`, so its input was exhausted during the long window.

| State | Simulated ticks | Wall (s) | Game-s/wall-s | Overshoot |
|---|---:|---:|---:|---:|
| Fresh reset | 3,600 | 3.4731 | 17.28 | 0 |
| Fresh reset | 36,000 | 35.0730 | 17.11 | 0 |
| One fuelled furnace | 3,600 | 3.5461 | 16.92 | 0 |
| One fuelled furnace | 36,000 | 37.7063 | 15.91 | 0 |

## Q5: score semantics

`namespace.score()` returned a two-element tuple `(production_score, automated_production_score)`. Two iron plates were supplied directly before the baseline without changing production statistics. The agent then hand-mined one iron ore, hand-crafted one iron gear, and finally ran a fuelled furnace for exactly 600 ticks; it produced three plates. A baseline score call cost 37.877 ms. The 30-sample paused measurement in Q6 gives the steadier cost estimate.

| Stage | Production | Automated | Production delta | Automated delta | Score call (ms) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0 | 0 | 0 | 0 | 37.877 |
| After hand-mining one iron ore | 3 | -1 | +3 | -1 | 17.159 |
| After hand-crafting one gear | 4 | -1 | +1 | 0 | 32.839 |
| After autonomous furnace | 5 | 0 | +1 | +1 | 12.421 |

Production includes manual work. Automated score excluded the hand-craft increment and increased with furnace output; the measured manual-mining stage yielded `-1`, so consumers must not assume this component is nonnegative.

## Q6: paused-boundary latency

Thirty samples were taken with the game paused. Tensor timing includes `obs_all_drain()`, entity/terrain reconciliation, and `TensorClient.observation()` for the `(17,96,96)` grid, `(2048,38)` view, and globals.

| Operation | n | Mean (ms) | p95 (ms) | Mean wall (s) |
|---|---:|---:|---:|---:|
| Trivial RCON round trip | 30 | 2.596 | 5.231 | 0.002596 |
| Tensor drain + reconcile + view | 30 | 2.480 | 5.093 | 0.002480 |
| `namespace.score()` | 30 | 20.037 | 36.678 | 0.020037 |
| Full tensor + score boundary | 30 | 21.888 | 40.256 | 0.021888 |

## Q7: navigation cost

These measurements used existing `move_to` at speed 40 and read live `game.tick` around the call. FLE fast mode also records modeled movement duration in `storage.elapsed_ticks`, so both counters are shown.

| Requested tiles | Actual displacement | `game.tick` delta | Modeled action ticks | Wall (s) | Final position |
|---:|---:|---:|---:|---:|---:|
| 10 | 10.51 | 145 | 74 | 0.1572 | `(10.50,0.50)` |
| 30 | 30.50 | 438 | 214 | 0.4194 | `(30.50,0.50)` |
| 60 | 60.50 | 765 | 424 | 0.7646 | `(60.50,0.50)` |

For SMDP action duration, `storage.elapsed_ticks` is the movement model emitted by fast-mode `move_to`; live `game.tick` additionally advances while path requests, RCON work, and the tool’s wall wait execute.
