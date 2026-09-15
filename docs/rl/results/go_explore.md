# Go-Explore open-play restore gate: blocked

The 2-hour Go-Explore run was **not released**. A live save–restore round trip on
the isolated Factorio 2.0.73 server at RCON **27098** did not reproduce
automatic-production counters or automated score. Continuing would make archive
cells and replay rewards false. The separate PPO container on 27099 was not used.

## Intended archive and dataset

[`fle/rl/go_explore.py`](../../../fle/rl/go_explore.py) defines a cell as the
cumulative tuple of (1) item types ever produced automatically, (2) entity
types ever built, and (3) researched technology names. Those mark progress in
the discrete recipe and research tree. Position, tick, and inventory counts are
excluded because they vary without a new achievement. There is no score bucket:
score chooses the representative within a cell instead of multiplying cells
for output from the same line. The initial `automation-science-pack` research
bit, granted by FLE's reset, is present in every root signature.

Each entry holds `GameState`, the transition-index trajectory to its first
discovery (or a later better representative), automated score, visits,
discoveries from that cell, and last productive selection. A higher-scoring
representative wins; equal scores prefer the shorter trajectory. Selection
weight is
`(1+visits)^−1/2 × (1+min(discoveries,5)/(1+age/10)) × (1+log(1+max(score,0)))`.
The 12-operation policy samples operations uniformly and structural arguments
from valid masks. Each excursion is bounded to 32 actions. Accepted cell paths
form the offline dataset; their stored consecutive score endpoints give the
real `score_after − score_before` reward. The exporter writes an ordinary
`ReplayBuffer` shard loadable by `fle/rl/replay.py`, with no goal conditioning,
HER, or relabelling. Archive and replay checkpoints are configured every
900 seconds, with atomic replacement.

## Live round trip

The gate used a **diagnostic** fuelled burner mining drill and wooden chest to
ensure a nonzero automatic counter. These were not an exploration start state.
After one WAIT action, `GameState.from_instance` saved the paused state.
`instance.reset(game_state=...)` restored it, followed by a pause, exporter
episode reset, new client full sync, and an incremented environment epoch.

| Measure | Before save | After restore |
| --- | ---: | ---: |
| Automatic iron ore | 15 | 0 |
| Raw iron-ore production | 15 | 0 |
| Automated score | 37.0 | −3.0 |
| Player stone inventory | 3 | 3 |
| Drill fuel (coal) | 19 | 16 |
| Chest iron ore | 8 | 15 |
| Player position | (105, −128) | (105, −128) |
| Researched technology | automation-science-pack | automation-science-pack |

The drill and chest entity types and positions returned, but their contents
changed during restore. The injected fake observation entity row was removed,
and the environment epoch advanced from 1 to 2. Save cost was **0.055857 s**;
reset, pause, exporter reset, and full observation sync together cost
**24.780407 s**. The serialized `GameState` was **68,741 bytes**. These are one
live cadence measurement, not a throughput distribution. A separate first
round trip measured `instance.reset` itself at 0.119617 s, without the full
observation sync; it showed the same loss of 15 automatic iron ore and score
37.0 → −3.0. Thus **0 of 2** observed round trips preserved the required
production counters.

The implementation explains the counter loss: `GameState.from_instance` saves
entities, inventory, research, namespaces, and messages, but not force
production statistics or the manual-production ledger. The reset action calls
`reset_production_stats()`, which clears both statistics and manual ledgers
(`fle/env/tools/admin/reset/server.lua:31`,
`fle/env/tools/admin/get_production_stats/server.lua:42-53`). Consequently an
archive state cannot recover its `automated_score`, and even a valid-looking
entity restoration changes the reward basis. This is a blocking restore
failure, so no counter offset or reward relabelling was applied. The snapshot
also invokes `save_entity_state` with its default `resource_entities=False`,
so mined resource amounts are not part of `GameState`; that is another obstacle
to exact archive replay after harvesting.

## Exploration and offline-data measurements

| Measurement | Result |
| --- | ---: |
| Valid archive cells at 0 min | 0 |
| Valid archive cells at stop | 0 |
| Discovery rate | Not measurable; gate stopped before exploration |
| Deepest recipe-chain point | None from a valid open-play run |
| Items automatically produced by exploration | None |
| Entities built by exploration | None |
| Maximum automated score in a valid cell | Not measured (random baseline: 0.00) |
| Dataset cells / transitions | 0 / 0 |
| Archive / replay bytes | 0 / 0 |

The diagnostic drill's 15 iron ore and score 37.0 are excluded from these
figures because it was pre-placed solely to test restore. The JSONL grew from
the `run_start` row to the `restore_probe` and `blocked` rows; the long loop was
never released. The probe data and exact before/after values are in
[`go_explore_restore_probe.json`](go_explore_restore_probe.json), with the run
events in [`go_explore.jsonl`](go_explore.jsonl). A synthetic archive export
loaded through `ReplayBuffer.load` and recovered one transition with reward
2.0; no live replay shard was created.

The runner is [`tests/benchmarks/run_go_explore.py`](../../../tests/benchmarks/run_go_explore.py).
It requires port 27098 and exits with status 2 when the restore gate fails.
`ruff check`, Python compilation, `git diff --check`, and the replay load-back
check passed. The existing `tests/rl/test_observation_action.py` suite had
9 passes and 3 failures against unrelated in-progress observation/mask edits;
`tests/rl/test_env_workers.py` could not collect because the already-deleted
`fle.rl.goals` module is still imported there. The Go-Explore container was
removed after the probe; the PPO container's ID remained
`b309d8bb59241d2479b4b0740d3e3ad1a047fa424cfe5d2e478dd9e2effdfc45`
and it remained running.
