# Rung 2a — one-goal live DQN/HER training (2026-09-13)

## Registered configuration before the run

Question: can a goal-conditioned DQN seeded with live scripted demonstrations and trained with HER produce 4 automatic iron ore on an assigned goal, beating a random baseline? This is the easiest useful rung-2 setting. The scripted policy produced 15 automatic ore per episode in the four saved live demonstrations, so quota 4 is reachable within the 64-action budget. The full rung-2 scripted-baseline gate in `SCALING.md` is stricter; this 2a experiment tests the requested random-baseline gate first.

| Setting | Value |
|---|---|
| Goal | One item, `iron-ore: 4`, fixed for baseline, training, and evaluation |
| Map/server | Open-world Factorio 2.0.73, map seed 44340, one Docker container, RCON 27099, `game.speed=10` |
| Episode budget | 64 actions; automatic output read from live production counters |
| Replay seed | `rung1_scripted_replay.npz`: 176 real live transitions plus 662 saved HER overlays; real demonstration goal lane retargeted from `iron-plate: 100` to `iron-ore: 4` without changing state/action/tick/outcome data |
| Algorithm | Goal-conditioned branching Double DQN, 3-step returns, prioritized replay, future HER `k=4`, target sync every 2,000 updates, Adam learning rate `1e-4`, gradient clipping at 10 |
| Learner | MPS, batch 256, 2 updates per live training transition; 352 warm-start updates (2 per seeded real transition) |
| Exploration | Episode-level epsilon starts at 0.5, decreases linearly with training wall time to 0.05; greedy evaluation uses epsilon 0 |
| Baseline/evaluation | 16 random episodes first; 3 fresh greedy episodes every 20 training episodes; 16 fresh greedy episodes after training |
| Duration | 2,520 seconds (42 minutes) of training phase, including periodic greedy evaluations; total task budget under 75 minutes |
| Logging and recovery | Flushed JSONL for run/episodes/every update; runner pauses after first live transition until JSONL growth is verified; checkpoint after warm-start, every 10 training episodes, and at training end |

The measured one-container warm throughput is 6.975047 steps/s and the measured MPS learner throughput is 14.4 updates/s. At replay ratio 2, 6.98 steps/s demands about 13.96 updates/s, close to learner capacity. Ratio 8 would demand about 55.8 updates/s and stall environment collection; it is inappropriate here. Actual end-to-end rates and the binding constraint will be measured below.

## Results

Pending live run.
