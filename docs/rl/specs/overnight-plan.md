# Overnight plan, 2026-09-16 night: eight hours, six servers

Draft by the main session; reviewed by Codex; final version below the review.

## What we know going in

- v0 plain branching DQN learned a hand-fed stone-brick furnace line from the
  empty start: median APS 389 over its last 39 episodes, max 723, 109/159
  positive. PPO collapsed to inaction. Random: 0/150.
- v0's wall: plates extracted in 10/159 episodes, gears crafted 0/159, so no
  drill ever. The reward is flat until the gear -> drill -> fuel conjunction
  pays.
- v1 env (landed): peaceful reset (zero deaths), a placeable head (PLACE ok
  76-100%), recursive CRAFT ingredient mask (CRAFT ok 100% at quantity 1),
  256-step horizon.
- v1 learner (landing): PER, Bootstrapped DQN with randomized priors,
  multi-server threads, resume, greedy eval.
- Throughput: about 1 macro step/s/server; 8 h x 6 servers = about 170k steps
  total, about 650 episodes of 256 steps.
- Rules: reward untouched; exploration touches the behaviour policy only;
  support masks only; no scripts.

## Arms (draft)

| Arm | Servers | Learner | Purpose |
|---|---|---|---|
| A boot | 27000, 27001 | bootdqn + PER, K=8, prior 1.0, eps floor 0.02, 60k steps | primary deep-exploration arm |
| B ctrl | 27004 | plain dqn (v0 hyperparameters) on env v1, 30k steps | isolates the env v1 effect from the learner change |
| C rand | 27003 | random valid actions, 200 episodes | v1 baseline |
| D ucb | 27002 | dqn + count-based behaviour bonus (UCB over (op, primary arg) counts, behaviour only) if D2 can add it in 30 min; else a second boot seed | historically the largest measured exploration gain in this project |
| E eval | 27005 | hourly greedy evaluation of the latest A checkpoint, 4 episodes | exploitation readout, separates behaviour from greedy policy |

## Night loop (main session, every 45 min)

1. Every run alive? Relaunch with `--resume` if it exited early or finished.
2. Regenerate curves and the automation ladder for all arms.
3. Note the first episode per arm that crafts a gear, a drill, places a drill,
   fuels a drill; snapshot that trace.
4. If arm A's greedy eval is clearly ahead of its behaviour policy, keep
   going; if A is behind B after 4 h on the ladder, switch A's second server to
   arm D's configuration.
5. At the end: `docs/rl/results/overnight_v1.md`.

## Open questions for the review

- Is bootstrapped DQN with priors the right first deep-exploration method here,
  or would count-based behaviour bonuses over supported actions dominate on
  this budget?
- Horizon: 256 steps or longer for the night?
- What single extra log would make the morning analysis decisive?

## Codex review (2026-09-16), condensed

Ranked by information per server-hour: UCB > matched control > bootstrapped >
eval > random. Points adopted: (a) count-UCB over supported (op, primary
argument) pairs is the best bet, not a fallback, because the recursive masks
expose CRAFT gear and CRAFT drill exactly when feasible; (b) the control must
be plain epsilon-DQN with identical PER, otherwise A vs B confounds
exploration, replay and env; (c) K=8 is oversized on CPU with about 30 acting
episodes per head, use K=4, batch 32, one update per two steps; (d) random v1
is a 12-episode census, then that server does greedy eval every 90 min; (e)
protect lucky episodes with a milestone archive (16 episodes per deepest rung
plus 32 highest-APS, 25% of every batch), PER alpha 0.5, beta 0.4 -> 1, priority
cap at the running 99th percentile; (f) keep 256 steps and the 216k-tick cap,
revisit 512 only if first hits cluster past step 220; (g) log per action:
exploratory flag, standardized Q, UCB bonus or acting head, chosen (op,
primary) pair; log per rung: first-hit episode, step, tick, final APS, and
the rung's repeat probability afterwards; (h) guards: replay 40k, torch threads
1, a 60 s step timeout plus a log-staleness watchdog, resume must persist the
milestone archive, keep latest plus milestone checkpoints.

## Final plan

| Arm | Servers | Learner | Steps |
|---|---|---|---|
| ucb | 27000, 27001 | dqn + PER + count-UCB behaviour, z(Q) + 1.5 sqrt(log(N+1)/(n+1)), eps 0.02, force each newly supported pair once | 60k |
| ctrl | 27004 | dqn + identical PER, eps 1.0 -> 0.05 over 15k then 0.05 | 30k |
| boot | 27002, 27005 | bootdqn K=4, prior 1.0, PER, batch 32, update every 2 steps | 60k |
| census/eval | 27003 | random v1, 12 episodes; then greedy eval of the latest ucb and boot checkpoints, 4 episodes each, every 90 min | |

Common: n-step 5, gamma 0.999, lr 1e-4, target copy 2000 updates, replay 40k,
batch 64, milestone archive 25% of batch, checkpoint every 2k steps, speed 40,
256 steps / 216k ticks.

Night loop every 45 min: `python tests/benchmarks/night_status.py`; relaunch
with `--resume` any run whose env log is older than 3 min or whose process is
gone; run the eval slot; note ladder firsts; at the end write
`docs/rl/results/overnight_v1.md` with curves, ladder, first-hit table and
repeat probabilities.
