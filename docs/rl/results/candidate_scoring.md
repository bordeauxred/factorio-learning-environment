# Complete-action candidate scoring on L2 from scratch

The experiment uses the unchanged L2 task: produce four automatic iron ore in
at most 16 actions. The fixed Factorio 2.0.73 map has seed 44340; reset gives
the player one burner mining drill and five coal beside an iron patch and a
wooden output chest. One `open_world` container uses RCON port 27099 at
`game.speed=10`.

The learner starts with seed 20260913, a new candidate-mode network, and empty
replay. It uses joint complete-action Q scores, Double DQN with an exact maximization over
the offered successor candidates, a batch of 64 on MPS, and one update per real
transition. Only WAIT, MOVE, PLACE, and INSERT are allowed. The reward stored
in replay and used for Q targets is raw `delta PS`; there is no shaping, HER,
or demonstration. Count-based stage/operation and PLACE-tuple novelty affects
behavior sampling only. Epsilon decreases linearly from 1.0 to 0.1 over
9,000 training seconds. A fresh baseline samples uniformly from the offered
complete actions for 24 episodes, and the final checkpoint gets 24 fresh
greedy episodes. Network and replay checkpoints are scheduled every 900
training seconds and committed at the next episode boundary, as well as after
each 100 training episodes. `OMP_NUM_THREADS=1` is set throughout.

On the epsilon branch, operation novelty weights are inverse square-root
stage/operation counts, and PLACE novelty weights are inverse square-root
counts of offered complete PLACE identities. The earlier E1 exploration
sampled observed ore tiles and directions, so this test compares full
candidate-mode behavior with the overnight arms, not scorer weights in an
otherwise identical proposal distribution. The fresh random baseline also
uses the offered candidate set.

The primary endpoint is the fraction of training episodes whose exact
readiness lane ever reports an ore-to-receiver aligned drill. Fuelled drill
and four-ore success are secondary episode fractions. The report compares
contiguous first and second halves with an exact two-sided permutation test
for binary outcomes, conditional on total successes. Quartiles are four
contiguous, near-equal episode groups; no smoothing blocks are selected.

An initial live pilot exposed an INSERT candidate with the action schema's
`-1` “all” count. Factorio rejects that count. The generator now omits this
redundant candidate, and the reported run starts from a new network and empty
replay after a container restart. The pilot is excluded from every result.

The first live training reset offers 467 candidates, 428 of them PLACE. This
differs from the 113-candidate offline fixture because the live iron patch is
larger. The dynamically selected setup's exact aligned action is offered at
PLACE rank 265 (one-indexed); a cap of 100 would remove that exact action.
The default PLACE cap of 512 is retained, and live candidate recall is reported
below.

The runner initially reserved 60 final episodes. After the episode-200
periodic check showed roughly 88 seconds per greedy episode, the final sample
was fixed at 24 **before final evaluation began**, matching the earlier 0/24
from-scratch comparison and leaving time for this report. The decision was
stated before evaluation; its `eval_budget_revision` row was restored after
the runner's open file handle overwrote the concurrent append. No episode
record was changed.

## Results

The run completed **316 training episodes** and 5,056 real training decisions in 2.502 training hours (**0.561 steps/s** per training-phase wall second, including periodic evaluation). Epsilon went from 1.000 to 0.101. There were 16 network-and-replay checkpoint records; the largest gap between committed records was 933.9 seconds.

The final recoverable pair is `.fle/runs/candidate_l2/model_316.pt` and `.fle/runs/candidate_l2/replay_316.npz`.

On the saved 467-candidate boundary, 100 warmed greedy decisions on MPS took 5.63 ms median and 6.11 ms at the 95th percentile. That is about 0.3% of the 1.78 seconds per real training decision end to end, so action selection does not dominate. The four-hour control collected roughly 0.88 training steps/s; the lower rate here also reflects long WAIT actions and learner updates. The microbenchmark excludes Q-update cost.

| Measure | Training | Fresh random baseline | Fresh greedy final |
|---|---:|---:|---:|
| Aligned drill | 1/316 (0.3%) | 0/24 (0.0%) | 0/24 (0.0%) |
| Fuelled drill | 194/316 (61.4%) | 11/24 (45.8%) | 24/24 (100.0%) |
| Four-ore goal | 0/316 (0.0%) | 0/24 (0.0%) | 0/24 (0.0%) |

The exact two-sided permutation test for **final four-ore success versus fresh random** gives p = 1. Previous from-scratch greedy arms were 0/24, 0/35, and 0/60. The two overnight training arms reached an aligned drill in 6/1,226 (0.5%) and 5/793 (0.6%) episodes.

| Outcome | First half | Second half | Exact permutation p |
|---|---:|---:|---:|
| Aligned drill | 1/158 (0.6%) | 0/158 (0.0%) | 1 |
| Fuelled drill | 63/158 (39.9%) | 131/158 (82.9%) | 2.676e-15 |
| Four-ore goal | 0/158 (0.0%) | 0/158 (0.0%) | 1 |

Quartiles below are contiguous, near-equal groups of episode numbers. No rolling or selected block average is used.

| Quartile (episode numbers) | Aligned drill | Fuelled drill | Four-ore goal |
|---|---:|---:|---:|
| Q1 (0–78) | 1/79 (1.3%) | 34/79 (43.0%) | 0/79 (0.0%) |
| Q2 (79–157) | 0/79 (0.0%) | 29/79 (36.7%) | 0/79 (0.0%) |
| Q3 (158–236) | 0/79 (0.0%) | 58/79 (73.4%) | 0/79 (0.0%) |
| Q4 (237–315) | 0/79 (0.0%) | 73/79 (92.4%) | 0/79 (0.0%) |

**Candidate availability.** The dynamically selected aligned action was present at 316/316 (100.0%) training resets. Mean candidate-set size across episode boundaries was 76.4 (observed range 30–467). Mean offered/available recall was WAIT 1.000, MOVE 0.636, PLACE 1.000, INSERT 1.000. Training tried 95 distinct PLACE tuples.

**Action outcomes.** Training operations were {'MOVE': 1451, 'WAIT': 2324, 'INSERT': 1031, 'PLACE': 250}; outcomes were {'ok': 5020, 'tool_rejected': 36}, with a total raw score delta of 576.0. Final greedy operations were {'PLACE': 24, 'INSERT': 120, 'MOVE': 48, 'WAIT': 192}; outcomes were {'ok': 384}, with final ore counts {2: 24}. All 379 logged episodes had 0 invalid-counter records and 0 score-delta mismatches. Supervisor restarts: 0.

## Verdict

**No: complete-action candidate scoring alone did not close the from-scratch L2 gap.** The final greedy policy reached the four-ore goal in 0/24, the same observed rate as random (0/24) and the prior from-scratch greedy arms.

The final policy fuelled a drill in 24/24 (100.0%) and ended at exactly two ore in 24/24 episodes. It learned the place–fuel–mine prefix but not the receiver alignment needed for four ore.

The aligned action was available at reset and PLACE recall was 1.000, so truncation did not hide the setup. The live set was much larger than the offline fixture, and sparse aligned experience gave the scorer little evidence for ranking the ore/receiver conjunction. At episode 100, greedy choice repeated one physically rejected PLACE despite its zero return; at episode 200, it shifted to a valid but unaligned two-ore drill. The remaining issue is physical PLACE admission and learned ranking of aligned placements, not representational capacity alone.

The half-period permutation p values are descriptive: training episodes share a changing policy and are not independent randomized arms. This is one fixed map and one learner seed.

Validation: the production JSONL was observed growing from a live first transition to a full episode before the long loop was released. The final network and candidate replay reload, `OMP_NUM_THREADS=1 pytest -q tests/rl` passed 75 tests with one skip, and Ruff, Python compilation, and `git diff --check` passed.

Raw records: [candidate_scoring.jsonl](candidate_scoring.jsonl). Machine-readable analysis: [candidate_scoring_summary.json](candidate_scoring_summary.json). Latency data: [candidate_scoring_latency.json](candidate_scoring_latency.json).

Container teardown: `docker ps -a --filter name=fle_candidate_27099` returned 0 containers. `docker ps` returned no running containers at final verification.
