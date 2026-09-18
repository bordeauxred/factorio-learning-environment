# SM-ARQ on FLE: overnight results, 2026-09-18

Semi-Markov autoregressive Q-learning on Factorio, run on one M3 Max.

- Base PR: **#414** (`worktree-tiered-observation`), never modified or merged from here.
- Exact starting SHA: **bcd15fdab095a078a97b5b66498c164d63950310**
- Experimental branch: **`sm-arq`**, worktree `/Users/robertmueller/Desktop/agents/fle-sm-arq`
- Machine: Apple M3 Max, 14 cores, 36 GB. Factorio runs in Docker (7 vCPU, 12.8 GB) under box64;
  the learner runs on the host through MPS.
- Interface contract: `fle/smarq/contract.py`. Spec: `docs/rl/specs/sm-arq.md`.

> Status while this file is being written: the simulator probe is complete and the
> numbers below are measured. Environment, learner and training-loop sections are
> filled in as each lands. Nothing in this file is an estimate unless it says so.

## 1. Simulator time semantics (measured)

Full report: `docs/rl/specs/sim-probe.md`. The design commitments in section 1 of the
brief are all achievable on this server, and two of them required choosing between
implementations:

| Question | Measurement | Consequence |
|---|---|---|
| Does pausing freeze the factory? | `game.tick_paused=true` held the tick at 45,046 and 45,114 across 3.0 s wall windows, with a fuelled furnace mid-craft; its crafting progress was unchanged | Inference and SGD run at a frozen decision boundary. RCON still answers while paused (`pong@45114`), so observations are read without spending game time |
| How fast can the simulator run? | speed 40 → **1,165.77 ticks/wall-s = 19.4 game-s/wall-s**; speeds 100, 400 and 1000 were all *slower* (1,109-1,129 ticks/s) | Execution speed is 40. The box64/ARM container is CPU-bound well below Factorio's speed cap |
| Can FAST_FORWARD land exactly? | RCON polling overshot by 7.0 ticks mean, 9 max. A Lua `on_nth_tick(1)` handler that pauses at a stored target tick landed **exactly, 0 overshoot in all 20 trials** | FAST_FORWARD uses the Lua target handler. Polling is not accurate enough for an SMDP duration |
| What does simulated time cost? | 3,600 ticks (60 game s) in 3.47 s wall; 36,000 ticks (600 game s) in 35.07 s. With one fuelled furnace: 3.55 s and 37.71 s | ~17 game-seconds per wall-second. A 30-game-minute episode is ~93 wall seconds of pure simulation |
| What does a decision boundary cost? | trivial RCON 2.60 ms mean / 5.23 p95; tiered observation drain+reconcile+tensors **2.48 ms** / 5.09 p95; `score()` **20.04 ms** / 36.68 p95; full boundary 21.89 ms / 40.26 p95 | The PR #414 observation is nearly free. `score()` dominates the boundary and is the thing to optimise if throughput binds |
| What does walking cost? | 10 tiles: 145 ticks; 30 tiles: 438; 60 tiles: 765 (0.16 / 0.42 / 0.76 wall s) | Navigation inside an action is affordable, and its simulated cost enters tau honestly |

Nothing sleeps in wall-clock time for the factory. Wall time appears in this document
only as a throughput metric.

### Score semantics, measured

`namespace.score()` returns `(production_score, automated_production_score)`.

| Stage | Production | Automated | Δ production | Δ automated |
|---|---:|---:|---:|---:|
| Baseline | 0 | 0 | 0 | 0 |
| Hand-mined one iron ore | 3 | −1 | +3 | **−1** |
| Hand-crafted one gear | 4 | −1 | +1 | 0 |
| Fuelled furnace produced autonomously | 5 | 0 | +1 | **+1** |

The automated score does what the brief wants: manual mining and manual crafting do not
pay, autonomous production does. It can go negative, so nothing downstream may assume
it is non-negative.

## 2. Environment

_To be filled when the live semantic environment lands._

## 3. Algorithm

SMDP formulation actually implemented:

- A completed semantic action gives `(s, a, r, tau, s')` with `tau` in **simulated**
  game seconds.
- Continuous-time discount `Gamma(tau) = 2^(-tau/H)`, `H = 3600` simulated seconds by default.
- Target `r + Gamma(tau) * Q_target(s', a*)`, `a* = argmax_a Q_online(s', a)` taken through
  the same autoregressive maximisation the policy uses.
- Action construction is a chain of zero-time internal decisions: for those, `r = 0`,
  `tau = 0`, `Gamma = 1`, and a partial action bootstraps from the max over its next head.

_Network sizes and measured update rates to be filled when the learner lands._

## 4. M3 performance

_To be filled from `tests/benchmarks/smarq_throughput.py`._

## 5. Toy results

_TOY-A (SMDP correctness) and TOY-B (burner automation) to be filled._

## 6. Full results

| run | reward | scratch/demo | simulated horizon | semantic decisions | final production score | final automated score | automated / game minute | game-s per wall-s | wall clock |
|---|---|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | | | |

## 7. Failure analysis

_To be filled: manual grinding, invalid placement, Q divergence, exploration failure,
simulation bottlenecks._

## 8. Verdict

_To be chosen from A (clear swim) / B (weak swim) / C (infrastructure works, learning
unresolved) / D (drown), with the single highest-value next experiment._
