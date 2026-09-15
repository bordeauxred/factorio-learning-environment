# Compute scaling ladder

Compute is earned, not assumed. Each rung has an entry cost, a specific result it
must produce, and a gate that must pass before the next rung is funded. A rung
that fails its gate is not retried at larger scale; the failure is diagnosed
first, because scaling a broken signal only buys a more expensive wrong answer.

## Measured, 2026-09-13, M3 Max

These replace estimates. Real vocabulary, real observation size.

| Quantity | Value |
|---|---|
| Observation size | 38,495 floats (design predicted 38,000 to 40,000) |
| Policy parameters | 2,489,450 |
| Batch-256 forward, CPU | 166.6 ms |
| Batch-256 forward, MPS | 23.6 ms |
| Full update (online fwd + target fwd + backward + Adam), MPS | 69.3 ms |
| Learner throughput, MPS | 14.4 updates/s |

**This contradicts the design.** `DESIGN.md` asserts that Factorio simulation, not
the RL stack, limits throughput. On this machine that is false at the replay ratios
we want:

| Replay ratio | Env transitions/s the learner can absorb | Workers it can feed at ~1 s/transition |
|---:|---:|---:|
| 4 | 3.6 | ~3.6 |
| 8 | 1.8 | ~1.8 |
| 16 | 0.9 | ~0.9 |

The Docker VM's 7 CPUs would allow roughly 6 Factorio servers, but the MPS learner
can only keep 2 to 4 of them busy depending on replay ratio. **On the M3 Max the
GPU is the bottleneck, not Factorio.** Two consequences:

1. Starting configuration should be replay ratio 4 with 2 to 3 workers, not ratio 8
   with more. Raising worker count beyond what the learner absorbs buys nothing.
2. The "Factorio is always the bottleneck" claim only becomes true on a machine with
   a real GPU. That is an argument for rung 3's x86 box on *both* axes, not just CPU.

Note MPS is 7x faster than CPU here, so CPU-only training is not viable.

## Ladder

## Rung 0 — correctness, no training

**Hardware:** the M3 Max. 1 Factorio container, no GPU work beyond smoke tests.

**Cost:** minutes.

**Must produce:**
- The real vocabulary artifact in `data/rl/vocab/`, two exports hashing identically.
- A yes/no answer, with numbers, on whether automatic production counts are
  trustworthy: hand-mining and hand-crafting must contribute zero, drill and
  furnace output must contribute positively.
- Protocol v2 container tests passing, and PR #414's existing tests not regressed.
- A finite loss on a maximally masked batch.

**Gate:** automatic counts verified correct. Everything downstream reads them, and
a wrong count is invisible to every later metric. If this fails, nothing else runs.

## Rung 1 — does the loop close

**Hardware:** M3 Max. 1 container, 1 worker, MPS.

**Cost:** an hour or two of wall time.

**Must produce:**
- A random policy completing full episodes against a real server without leaks,
  hangs or ghost rows.
- Measured per-step latency broken down by phase, and real steps/s for one env.
  Every throughput number in `DESIGN.md` is currently an estimate; this replaces
  the first of them with a measurement.
- HER producing non-zero relabelled transitions from real trajectories, and the
  count of future-positive candidates per episode.

**Gate:** HER candidate count is greater than zero on real data. Risk 2 in the
design is that sparse HER can only relabel trajectories that produced something,
and from an empty start that may be nothing at all. If it is zero, the fix is
better scripted demonstrations or a goal-conditioned readiness term, not more
servers.

**Measured gate status (2026-09-13): YES with live scripted demonstrations.**
The random-policy sample remained at zero, but four live scripted episodes
produced 1,406 future-positive boundary pairs and 662 HER copies from 176
ordinary replay transitions. See [rung1_scripted.md](results/rung1_scripted.md).

## Rung 2 — does it learn anything

**Hardware:** M3 Max. 4 to 6 containers, 1 GPU (MPS).

**Cost:** overnight. Docker's VM currently has 7 CPUs and 13 GB, which caps this
rung; raise it in Docker Desktop settings before starting.

**Must produce:**
- A learning curve on the easiest curriculum goals, against a random baseline and
  a scripted baseline. Both baselines are required: beating random proves almost
  nothing in an environment where random almost never succeeds.
- Real success rate and HER success rate reported **separately**. Conflating them
  is the easiest way to fool ourselves here, since HER success is guaranteed to
  look good by construction.
- `invalid_combination_rate` measured. The design's trigger for replacing
  independent heads with autoregressive ones is 15%.

**Gate:** real success rate on assigned goals is meaningfully above the scripted
baseline on at least the easiest goal band. If the agent cannot learn "produce 16
iron ore" with dense HER relabelling, adding servers will not help.

## Rung 3 — ablations and the replay-ratio question

**Hardware:** a native x86 Linux box. 16 to 32 cores, 64 to 128 GB, one GPU.

**Cost:** days.

**Why leave the Mac here:** on arm64 the cluster runs Factorio under `box64`
emulation (`fle/cluster/run_envs.py:94-97`), so the most expensive component in the
system pays an emulation tax. Native x86 removes it and roughly quintuples usable
parallelism. This is the rung where that stops being a nuisance and starts being
the limit.

**Must produce:**
- Replay ratio 4 / 8 / 16 at matched fresh steps, with and without periodic resets.
- Greedy branch argmax versus a top-3-per-head beam, to settle whether the
  factorized argmax approximation is costing real performance.
- GOID curriculum versus uniform goal sampling.
- `game.speed` 10 / 20 / 40 with realized UPS, to find the actual wall-time lever.

**Gate:** at least one ablation produces a clear, reproducible win. Ablations that
all come out flat mean the signal is too weak to tune against, which sends us back
to rung 2 rather than forward.

## Rung 4 — the real run

**Hardware:** whatever rung 3's measurements say is needed, not a guess.

Only fund this once rungs 0 through 3 have passed. By then every number in the
design's sizing table has been replaced by a measurement, and the container count
follows from the measured steps/s and the measured sample efficiency rather than
from the estimate of "0.5M to 1M fresh steps" that the design currently carries as
**UNVERIFIED**.

## Standing rules

1. **No long run without a written checklist and explicit sign-off.** Configuration
   parameters listed, cost and time estimated, purpose stated.
2. **Verify logging is actually writing before any run longer than an hour.** An
   unlogged overnight run is a wasted night.
3. **Measure before enlarging the model.** Report input size, parameter count and
   forward latency first. The current architecture is deliberately boring; make it
   less boring only against evidence.
4. **One variable per experiment.** With this few runs available, a confounded
   result is worse than no result.
