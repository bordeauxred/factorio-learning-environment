# Direct production-score maximization in empty-start open play

This run uses a thin adapter over the RL `GoalConditionedEnv` reset and action
path to implement the empty-start contract of `open_play`/`DefaultTask`.
`open_play_production` is an Inspect evaluation configuration, not a
constructible Gym task; the Inspect solver itself opens `open_play`. The
adapter resets the open-world Factorio instance with empty inventory, no
placed entities, and no research pre-unlocked. It does no
world setup. The legacy observation schema retains a goal lane of zeros for
compatibility, but the policy is goal-free (`goal_conditioned=False`), there
is no target or threshold, and no goal reward or HER is used.

The reward stored in replay and used for Q targets is raw `delta PS` from
FLE's player production score: the floored, priced net item and fluid
production balance. This includes manual collection and crafting. FLE's
separate automatic item counters subtract manual harvested and crafted
output from production counts; these are reported to distinguish factory
progress from hand work. The metric here is final PS per episode, not a
success fraction or production-rate estimate.

The fixed map uses Factorio 2.0.73, seed 44340, one open-world container and
RCON port 27099 at `game.speed=10`. The branch is `feat/pufferlib-rl`.
Episodes have 320 decisions: at the earlier 86 ms/step estimate that is
about 27.5 seconds of action time, while the live run includes reset,
candidate generation, tool latency, waits, and learning. The complete
action candidate generator offers WAIT, MOVE, HARVEST, CRAFT, PLACE, PICKUP,
ROTATE, INSERT, EXTRACT, SET_RECIPE, CONNECT, and RESEARCH as state permits.
Open-play additions permit directional movement without visible ore,
resource approach moves from known terrain, placement near the player, and
furnace/assembler ingredient insertion. A fresh random baseline samples
uniformly from the same candidate generator. The main learner uses a
complete-action Q scorer, Double DQN, no demonstrations, and count-based
operation/action novelty only in behavior sampling. Novelty does not modify
replay rewards or Q targets. Epsilon declines linearly from 1.0 toward 0.1
over 9,000 training seconds. `OMP_NUM_THREADS=1` is set.

The JSONL log was inspected after its first live transition and observed to
grow before the verification barrier was released. The supervisor restarts
the container and resumes from the latest atomic network/replay pair on
failure. Checkpoints are scheduled every 900 training seconds and at each
completed training episode. The pre-run port and Docker checks found no
other service running.

The eight fresh random episodes all used 320 actions. Their final PS values,
in episode order, were **381, 0, 0, 0, 0, 0, 583, 0** (mean **120.5**,
median **0**, maximum **583**). The positive runs collected resources by
hand; the baseline built **zero entities** and produced **zero items
automatically**. All eight resets had empty inventory and no setup entities,
and all eight summed stepwise `delta PS` values matched final minus initial
PS exactly.
