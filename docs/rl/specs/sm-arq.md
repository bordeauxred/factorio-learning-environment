# Spec: SM-ARQ — semi-Markov autoregressive Q-learning on FLE

Status: to implement, overnight 2026-09-18. Spec owner: main session.
Base: PR #414 (`worktree-tiered-observation`), SHA `bcd15fdab095a078a97b5b66498c164d63950310`.
Branch: `sm-arq`, worktree `/Users/robertmueller/Desktop/agents/fle-sm-arq`.
PR #414 itself is never modified or merged from here.

The frozen interface is `fle/smarq/contract.py`. Every session imports from it
and from nothing else of another session's code. Read it before anything else.

## The question

Can a replay-based value learner learn Factorio through semantic,
variable-duration actions, where the neural policy chooses the exact industrial
intervention and the exact geometry, while deterministic navigation handles
locomotion?

## Non-negotiable commitments

1. **The policy owns geometry.** No candidate generation, no `nearest_buildable`,
   no snapping, no projection, no `place_entity_next_to`, no automatic belt or
   pipe routing, no `connect_entities` macro, no automatic spacing. A requested
   tile is placed at exactly that tile or the action fails.
2. **Locomotion is abstract.** Any action may navigate to a reachable
   interaction position for the target the policy chose. Navigation costs
   simulated time and never alters the requested target.
3. **Time is simulated.** The game is paused at every decision boundary. Policy
   inference and learner updates must not advance a single tick. Execution runs
   the simulator as fast as it will go. Nothing ever sleeps for a simulated
   duration.
4. **Failure is a learning signal.** Masks remove type and syntax
   impossibilities only. Occupancy, reachability, affordability, layout quality
   and strategic pointlessness are never masked; they produce failed
   transitions with a `failure_reason`.
5. **Reward is the published objective.** `reward = automated_production_score_after
   - automated_production_score_before`. No shaping, no intrinsic terms added to
   the reward. Both scores are always logged.

## Session split (disjoint file ownership)

| Session | Owns | Needs a live server |
|---|---|---|
| S0 (main) | `fle/smarq/contract.py`, `fle/smarq/fake.py` | no |
| S1 | `fle/smarq/{vocab,obs,raster,actions,sim,env}.py`, `tests/smarq/test_{actions,env_offline,sim_live}.py` | yes, ports 27000-27003 |
| S2 | `fle/smarq/{net,policy,replay,learner}.py`, `tests/smarq/test_{net,replay,learner}.py` | no |
| S3 | `fle/smarq/{demos,train,logging_}.py`, `tests/smarq/test_train_loop.py`, `tests/benchmarks/smarq_throughput.py` | later |
| S4 | `tests/benchmarks/smarq_sim_probe.py`, `docs/rl/specs/sim-probe.md` | yes, ports 27004-27005 |

No session edits a file it does not own. No session edits `fle/env/`,
`fle/cluster/`, or anything else that PR #414 touched.

## S1. Environment

### Simulated-time control (`sim.py`)

`SimClock` wraps `FactorioInstance`:

- `boundary()` — pause the game, return the current tick. Every decision
  boundary starts here. While paused, RCON reads still work.
- `run_until(predicate, max_ticks)` — unpause at the configured execution speed,
  let the game run while a tool call executes, pause again, return elapsed ticks.
- `advance_ticks(n)` — advance exactly `n` ticks and re-pause, using the most
  exact mechanism the probe (`docs/rl/specs/sim-probe.md`) found. Report the
  realized overshoot in every step's info dict.
- Every method records `duration_ticks`, `duration_game_seconds =
  duration_ticks / 60`, and `wall_seconds`.

Execution speed comes from the probe's ceiling measurement, not from a guess.
No `time.sleep` anywhere except a short poll interval while the simulator is
genuinely running.

### Observation (`obs.py`, `raster.py`)

Promote the PR #414 `TensorClient` into `fle/smarq/obs.py` as an importable
class (copy it in, do not edit the benchmark files; keep the channel layout and
feature indices identical and cite the benchmark file it came from). It gives
`grid (17,96,96)`, `entity_view (2048,38)`, `entity_mask`, `globals (10,)`.

Add `entity_ids (2048,)`: the stable Factorio unit numbers of the rows, so an
entity pointer chosen at collection time can be resolved at execution time and
recognised in replay.

`raster.py` builds the exact-tile egocentric raster from state the client
already holds — never from a build-planner query. Side length configurable,
default 96, target 288; channels exactly `RASTER_CHANNEL_NAMES`. The window is
centred on the character's tile; `raster_origin` is reported with every
observation so a position index decodes to the same absolute tile later.

### Action grammar (`actions.py`, `env.py`)

Verbs, heads, head order, quantity buckets and durations are in the contract.
Implement decode (head indices → `Action` with exact absolute tile) and execute.

Executor rules per verb, all through existing FLE tools:

- `PLACE(prototype, tile, direction)` — preserve the tile exactly; compute a
  reachable interaction position; navigate; attempt `place_entity` at exactly
  that tile; on refusal classify `blocked` / `unreachable` /
  `insufficient_inventory` / `invalid_argument`. Never retry at another tile.
- `MINE(tile, quantity)`, `MOVE_TO(tile)` — same navigation rule.
- `INSERT/EXTRACT/PICKUP/ROTATE/SET_RECIPE(entity_slot, …)` — resolve the slot
  to a live unit number; if it died or moved, fail with `no_such_entity`.
- `CRAFT(recipe, quantity)`, `RESEARCH(technology)` — no navigation.
- `FAST_FORWARD(duration)` — exactly 60/600/3600 ticks, executed as fast as the
  simulator manages, no wall-clock sleeping.

`ALL` resolves at execution time to the largest amount the game will accept for
that exact action; log `requested_quantity` and `executed_quantity`.

### Masks

Structural only, as the contract's `Masks` docstring states. Specifically
allowed: placeable prototypes for PLACE; live entities for the entity pointer;
entities with an inventory for INSERT/EXTRACT; recipe-capable machines for
SET_RECIPE with the recipes that machine's category accepts; hand-craftable
recipes for CRAFT; existing and not-yet-researched technologies for RESEARCH.
Specifically forbidden: masking by inventory contents, occupancy, reach,
distance, or quality. The POSITION head is never masked.

### Episode

Fixed simulated-time horizon, default 30 game minutes = 108000 ticks, plus an
emergency cap of 1000 semantic decisions. Repeated FAST_FORWARD consumes the
same budget as everything else and cannot extend the episode.

### Acceptance (S1)

`pytest tests/smarq -k "not live"`, `ruff check fle/smarq tests/smarq`, and a
live run printed verbatim:
`.venv/bin/python -m fle.smarq.env --port 27000 --episodes 2 --decisions 60 --seed 1`
reporting for each episode: decisions, simulated ticks, wall seconds,
game-seconds per wall-second, both scores, failure-reason histogram, and proof
that ticks did not advance while paused.

## S2. Network, policy, replay, learner

Parameter budget under 10M. No recurrence, no large transformer.

- Grid encoder: 17×96×96 → conv 17→32 3x3, 32→64 3x3 s2, 64→128 3x3 s2, two
  small residual blocks; keep the spatial map, also pool to `z_grid` (128-256).
- Entity encoder: shared MLP over the 38 features → 128-d per entity, masked;
  keep per-entity embeddings for the pointer head; masked mean and max → ~256.
  Categorical fields (type, recipe, item, fluid) go through embeddings, not as
  ordered floats. Counts `log1p`, energy logged, positions relative, progress in [0,1].
- Globals: 10 → 64 → 64.
- `z_state`: concat → MLP → 256 or 512.
- Autoregressive Q heads exactly as in the contract's `HEAD_SEQUENCE`. Each
  head conditions on `z_state` and on embeddings of every argument already
  chosen. The spatial head conditions the retained feature map on verb and
  prototype embeddings and upsamples to the exact raster resolution; it is a
  Q-value per exact tile and is never masked.
- Entity pointer head scores each entity embedding against the action context.

Learner: double DQN, target network, prioritized replay (alpha 0.6, beta
0.4→1.0), Huber loss, AdamW, lr 2e-4, batch 128, grad clip 10, target update
every ~2000 updates, replay capacity 500k.

SMDP targets: for a completed action, `target = r + 2**(-tau/H) * Q_target(s', a*)`
with `a* = argmax_a Q_online(s', a)` chosen through the same autoregressive
maximisation, tau in simulated seconds, H default 3600. For internal
autoregressive steps, `r = 0`, `tau = 0`, `Gamma = 1`, and the value of a
partial action is the max over its next head.

Exploration: epsilon-greedy per head, never a uniformly random complete tuple.
The spatial head explores by sampling an exact tile in the window.

### Acceptance (S2)

Offline tests only, against `fle.smarq.fake.FakeSemanticEnv`: parameter count
under 10M; a gradient step changes parameters; target network updates on
schedule; PER sampling and priority update are correct; the SMDP target uses
simulated duration and a longer duration gives a smaller discount; internal
transitions use Gamma=1; replay round-trips through disk.

## S3. Training loop, demonstrations, logging

Single process: N environment workers in threads (RCON is I/O bound) feeding one
replay; learner on MPS in the main thread; configurable replay ratio (4, 8, 16).
Checkpoint every N updates. Resume from checkpoint.

Demonstrations use the same semantic interface with exact coordinates and no
privileged helper; they enter replay as ordinary off-policy transitions with
elevated initial priority. 50-200 trajectories if they are cheap.

Log per semantic step: tick, simulated seconds, wall seconds, game-seconds per
wall-second, both scores and both deltas, verb, heads, failure reason,
requested/executed quantity, Q of the chosen action, TD error, priority. Per
episode: return, final and max automated score, automated score per game
minute, decisions, failure histogram. Per 100 updates: loss, grad norm, mean
max Q, epsilon, replay ratio, transitions/wall-second, simulated seconds per
wall-second.

## Runs

| Run | Learner | Reward | Horizon | Seed |
|---|---|---|---|---|
| TOY-A | correctness harness | — | — | — |
| TOY-B | random / scripted / scratch / demo-seeded | automated | fixed small budget | 1 |
| FULL-1 | SM-ARQ scratch | automated | 30 game min | 1 |
| FULL-2 | SM-ARQ demo-seeded | automated | 30 game min | 1 |
| FULL-3 | SM-ARQ scratch | production (ablation) | 30 game min | 1 |

FULL-1 and FULL-2 have priority. Results go to
`SM_ARQ_M3_OVERNIGHT_RESULTS.md` at the repository root.
