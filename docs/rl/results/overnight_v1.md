# Overnight v1: four off-policy arms on six servers (2026-09-16, 01:38 to 08:30)

Environment v1 (`fle/rl/env.py`): peaceful reset with enemies removed (zero
deaths all night), a dedicated placeable head for PLACE, a recursive
ingredient mask for CRAFT, 256-step episodes, game speed 40, empty start,
map seed 44340. Reward is the raw change in FLE's automated production score.
Learner: `fle/rl/dqn.py` (branching double DQN family). Plan and Codex review
in `docs/rl/specs/overnight-plan.md`; the half-hourly log in
`overnight_v1_log.md`; plots and tables in `overnight_v1/`.

## Arms

| Arm | Servers | Exploration | Hyperparameters | Steps by 08:23 |
|---|---|---|---|---:|
| ctrl (v0 hp) | 27004 | epsilon 1.0 -> 0.1 over 30% of budget | PER a=0.6, n-step 3, gamma 0.99, replay 100k, target 1000, batch 64 | 37.9k (resumed once at 30k) |
| ctrl_m (matched) | 27003 | epsilon 1.0 -> 0.05 over 15k | PER a=0.5 b0=0.4 cap p99, n-step 5, gamma 0.999, target 2000, replay 40k, archive 25% | 30.0k (finished) |
| ucb | 27000, 27001 | count-UCB over (op, primary arg), c=1.5, eps floor 0.02, each new pair tried once | same as ctrl_m | 52.2k |
| boot | 27002, 27005 | bootstrapped DQN K=4, randomized priors 1.0, eps floor 0.10 | PER, v0 defaults otherwise | 42.3k |
| random | 27003 | uniform over admitted indices | 12-episode census | |

Boot was restarted once at 02:23: with an eps floor of 0.02 it was degenerate
(61% RESEARCH, zero INSERT, no furnace fuelled in 20 episodes). The 0.10 floor
fixed it. ctrl_s2, a second seed of ctrl, ran 19 episodes before yielding its
server to ucb.

## Headline

| Arm | Episodes | Final APS median [q1, q3] | max | Episodes APS > 0 | Last-20 median |
|---|---:|---|---:|---:|---:|
| random v1 | 12 | -1 [-2, -1] | 8 | 2 | |
| ctrl (v0 hp) | 148 | 23.5 [0, 106] | **1189** | 111 (75%) | **177** |
| ctrl_m | 118 | 19.5 [-1, 46] | 121 | 73 (62%) | 40 |
| ucb | 206 | 2.5 [-1, 21] | 116 | 109 (53%) | 21 |
| boot | 168 | **65.5 [-1, 180]** | 596 | 119 (71%) | 143 |

For scale: v0 last night ended at a last-20 median of 567 after 159 episodes
on a 128-step horizon and a single server, but never left the stone-brick
line. The v1 arms are at 100 to 180 after 150 episodes on 256 steps and are
still climbing (see `overnight_v1/aps_per_episode.png`).

![APS per episode](overnight_v1/aps_per_episode.png)
![op share](overnight_v1/op_share.png)

## Automation ladder and what gets forgotten

Episodes reaching each rung (from `overnight_v1/ladder.md`):

| rung | ctrl | ctrl_m | ucb | boot | random |
|---|---:|---:|---:|---:|---:|
| furnace fuelled and fed | 121/148 | 92/118 | 133/206 | 123/168 | 7/12 |
| plates extracted | 32/148 | 28/118 | 78/206 | 38/168 | 0/12 |
| gears crafted | 0/148 | 2/118 | **21/206** | 0/168 | 0/12 |
| drill crafted | 0/148 | 1/118 | **5/206** | 1/168 | 0/12 |
| inserter placed | 0/148 | 3/118 | 12/206 | 0/168 | 0/12 |
| belt placed | 0/148 | 4/118 | 13/206 | 0/168 | 0/12 |
| drill placed | 0 | 0 | 0 | 0 | 0 |

Repeat rate after first discovery (`overnight_v1/rung_repeat.md`):

| rung | ctrl | ctrl_m | ucb | boot |
|---|---:|---:|---:|---:|
| furnace fuelled and fed | 0.82 | 0.81 | 0.64 | 0.79 |
| plates extracted | 0.22 | 0.25 | 0.37 | 0.26 |
| gears crafted | never | 0.01 | 0.11 | never |
| drill crafted | never | 0.00 | 0.03 | 0.00 (found at ep 165) |

Two rungs are learned (fuel and feed a furnace, extract plates). Everything
past plates is **found then forgotten**: after the first gear craft, ctrl_m
repeated it once in 94 episodes; ucb, whose counts keep pulling it back,
repeated it 20 times in 184 but never converted it into score. The
milestone archive (25% of every batch from milestone episodes) did not turn
those episodes into a repeated policy on this budget.

## What the arms taught

1. **All four off-policy arms learn from the empty start; none needed a
   script.** Positive episodes: 53 to 75% per arm against 17% for random on
   the same environment.
2. **Exploitation wins the score race, exploration wins the ladder.** The
   plain v0-hyperparameter control has the best last-20 median and the 1,189
   episode (nine furnaces, 87 iron-ore inserts, score rising linearly). UCB
   reaches gears, drills, inserters and belts far more often than any other
   arm and has the worst median. Bootstrapped DQN sits between: second-best
   score, deep rung reached once.
3. **The matched control under-performs the v0 control.** n-step 5, gamma
   0.999, the 40k replay and the priority cap together cost roughly 4x in
   last-20 median versus the v0 settings. That comparison is one seed each
   and should be read as "the Codex-recommended settings are not a free
   win", not as a verdict.
4. **A drill has been crafted 7 times and placed 0 times.** After the craft,
   every episode drifted into cheap loops: wooden-chest crafting sprees
   (ctrl_m ep54: 25 chest crafts in a row after the drill), place/pickup
   cycles on an inserter (ucb ep22: 120 steps). These are zero-reward
   actions that are always admitted, and nothing in the value function yet
   prefers PLACE drill over them.
5. **The quantity head is still independent of the ingredient check.** The
   CRAFT mask certifies one unit; 5x and 20x requests still fail on
   ingredients (`firearm-magazine x5` in the ucb trace). Small, fixable.
6. **Deaths: zero all night.** The peaceful reset closed the 15% sample loss.
7. **Throughput was the binding constraint:** about 1 macro step per second
   per server with six servers and three learners, so the whole night bought
   about 160k environment steps in total.

## Recommendations for the day

1. Keep **ctrl (v0 hp) + PER** as the exploitation baseline and **ucb** as
   the discovery arm, and try the obvious hybrid: UCB counts with a larger
   exploitation weight (c 0.5) or UCB only over the deep rungs' ops.
2. Make the archive count: raise the archive fraction for the deepest rung
   and log how often archived gear or drill transitions are actually sampled
   (Codex's "found then forgotten" diagnostic, now measured as 0.01 to 0.11).
3. Fix the quantity head: mask quantities that the recursive ingredient
   check cannot cover.
4. Evaluate the greedy policies (running now on 27003) and compare with the
   behaviour policies above; if greedy is clearly ahead, the eps floors are
   too high late in training.
5. Consider a second map seed before believing any of the absolute numbers.

## Greedy evaluation (08:23 to 08:52, port 27003, 4 episodes each)

| Checkpoint | Greedy APS (all 4 episodes) | Behaviour last-20 median at that time |
|---|---:|---:|
| ctrl (v0 hp) @ 38k steps | **574** | 177 |
| boot @ 42k steps (head-averaged Q) | 38 | 143 |
| ucb @ 52k steps (Q without the bonus) | -1 | 21 |

Three readings. First, the four greedy episodes are identical per
checkpoint: with a deterministic policy and a deterministic reset, evaluation
episodes are replicas, so greedy numbers are one sample each, not a
distribution; future evals need seed variation or a small stochastic floor.
Second, dqn-per's greedy policy (574) is far above its behaviour
median (177): its epsilon floor of 0.1 was costing it most of its score late
in training. Third, UCB's Q on its own does nothing (-1): every one of its 126
positive episodes came from the count bonus, which means its deep-rung
discoveries were never absorbed into the value function on this budget. The
morning ordering by greedy score is ctrl > boot > ucb, the reverse of the
ladder ordering.

## Day 2 (2026-09-16 afternoon): the drill wall was a masking gap

Across the 9 episodes that ever crafted a burner mining drill, the policy
attempted to place it 55 times, and the tool refused every attempt with
"something is in the way or terrain is unplaceable", always at the same target
within an episode (one episode retried the identical tile 24 times in a row).
Two causes, both in our code: the PLACE offset mask did not exclude the
character's own tile, so the snapped 2x2 centre collided with the player; and
the offset head is a greedy secondary head, so the same blocked tile was
re-selected. Neither exploration nor value learning was the blocker at this
rung. Fixed (D6): player tile excluded from the offset support, secondary
heads resampled for first-time or just-refused pairs, frontier snapshots at the
gear rung, 512-step explorer horizon. Arms B and D restarted on it at 18:35.

Other day-2 results: the quantile head (QR-DQN, 51 quantiles) learns on the
same curve as plain DQN; NoisyNets are the failing Rainbow component here
(two flat runs); the frontier archive sampler gave no gain and cost 40%
throughput; frontier return (restore the agent's own deep state, 64-step
excursions) is implemented and live-verified (restore 0.83 s, position and
inventory exact) and is armed on qrdqn-per-ucbexplorer-frontier. A disk-full incident at 12:45 cost
three learners their replay buffers; checkpoints are now pruned automatically.

## Day 2 close (2026-09-17 00:45): what runs overnight and why

Trace inspection (`research/15-trace-inspection.md`) found that after the
placement fix the next wall was the quantity head: gear and drill crafts were
requested at x5 or x20 and refused for missing plates, so first-try
exploration credit was spent on impossible requests; excursions from the
first-gear state started too early; loops concentrated on the UCB ports; the
anchor holds plates and never spends them. Implemented (D7): quantity-aware
support with clamping, smallest supported quantity on first-time pairs,
snapshots when the drill craft first becomes admissible with that craft as the
first excursion action, refusal cooldowns and research no-op masks. Then (D8)
CONNECT restored as op 12 with position endpoints, schema macro-v2, per the
owner's rule that no buildable capability is withheld; refusals score zero.

Overnight arms, all fresh on macro-v2 with the D7 guards, 100k-step budgets:
B (plain DQN, epsilon + rung-UCB explorer at 512 steps), D (quantile head +
the same behaviours + frontier return from the drill-admissible state), QR
(quantile head, epsilon). The v1 anchor keeps running as the score reference
(last-20 median about 1,400 at 73k steps). Decision metric unchanged: root
episodes with a drill placed, fuelled and producing.

## Two-day wrap-up (2026-09-17, 01:20)

### Biggest successes

1. **Unscripted automation from the empty start.** Two days ago the project's
   best was 2 iron ore. Now a plain branching double DQN with prioritized
   replay, given FLE's raw automated production score as its only reward,
   reaches a last-20 median of about 1,500 per 256-step episode (max 1,929)
   after 75k environment steps, from a policy that started random. No
   demonstrations, no shaping, no goal conditioning.
2. **The environment became measurable and honest.** Operation census, the
   automation ladder, greedy evaluation, per-action provenance logs, and a
   dashboard. Every failure of the last two days was found in the logs and
   named: reach, turrets, the placement mask, the quantity head.
3. **Exploration findings that transfer.** PPO collapses to inaction on this
   reward; off-policy with replay is required. Count-based UCB over (rung, op,
   argument) discovers deep rungs (gears, drills, inserters, belts) an order
   of magnitude more often than epsilon or bootstrapped heads, but does not
   exploit them; halving its bonus recovers exploitation. NoisyNets fail;
   quantile heads work and reach gears under plain epsilon.
4. **Infrastructure that outlives this attempt:** peaceful reset, macro
   environment with support masks, frontier snapshot and restore in under a
   second, quantity clamping, cooldowns, CONNECT restored with position
   endpoints, PufferLib integration, and about 100 offline tests.

### What we learned, distilled

- The score is learnable but the objective's shape matters: consumption is
  charged before production credits, hand work floors it at -1, so gamma 0.99
  dislikes investment and late furnaces never pay back.
- The deep rungs were never a value-learning wall. Three consecutive engineering
  gaps blocked them: harvest reach (day 0), placement on the character's tile
  (day 2), and craft quantities the ingredients could not cover (day 2). Each
  looked like "exploration is hard" until the traces were read.
- Independent action heads cost roughly a third of decisions on conjunctive
  actions; the fixes so far are support masks and clamps, and the structural
  fix is an autoregressive head.
- Throughput, about 1 macro step per second per server under x86 emulation,
  caps every arm at about 30k steps per 8 hours. The methods are being
  compared at one tenth of the budget they are usually judged at.

### Options for tonight (already running unless noted)

Running on six servers with 100k-step budgets, all on the CONNECT-enabled
schema with the quantity and cooldown guards:

- **dqn-per-ucbexplorer**: plain DQN, epsilon port plus rung-UCB explorer port at 512 steps.
  Tests whether the guards convert gears into drills without a value change.
- **qrdqn-per-ucbexplorer-frontier**: quantile head, same behaviours, plus frontier return from the
  state where the drill craft first becomes admissible, with the craft forced
  as the first excursion action. The most direct attack on the drill rung.
- **QR**: quantile head, plain epsilon. The clean value-axis comparison.
- **Anchor**: yesterday's plain DQN on the old schema, still climbing, kept as
  the score reference and to see where hand-fed smelting saturates.

Alternatives not taken tonight, ready when wanted: a mask-free control arm
(refusals cost a zero-reward step, nothing hidden), gamma 0.999 or 1.0 on the
telescoping objective, autoregressive action heads, and Go-Explore style
return from every rung rather than only the drill state. The morning read:
`dashboard.html` for curves, this file for insights, the log for the timeline.
The decision metric is unchanged: a drill placed, fuelled and producing in a
root episode.

## Night 2 (2026-09-17, 01:00 to 02:30): the environment layer debugged from traces

Two trace audits of the first 13k steps on the CONNECT-enabled schema
(`research/16-env-trace-audit.md`, plus my own read) found that about one
decision in five was wasted by our layer, not by FLE's tools: the item head was
a union over all entities so INSERT/EXTRACT picked items the anchor lacked
(396 refusals logged without a tool call); the refusal cooldown was a
post-selection guard rather than a mask (632 wasted steps, one MOVE repeated 57
times); ROTATE was offered on furnaces and chests (572 refusals and silent
no-ops); PLACE masked one tile, not the footprint (56 tree, rock, cliff and
water collisions); crude oil and stale trees were offered as harvest targets
(71); CONNECT allowed the same entity as both endpoints; EXTRACT contents
changed during approach; a harvest that acquired items was logged as refused
because the tool raised afterwards; and the fixed 216k tick cap truncated every
512-step explorer episode at 67 to 310 steps. Rewards telescoped exactly and
no op had a p95 wall time above 10 s.

All of it fixed in two Codex sessions (D9, D10), 125 offline tests, schema
unchanged. Live smoke on a spare server: 250 of 256 random steps executed as
intended, zero rejected crafts, zero cooldown waste, PLACE 90%+, CRAFT 100%.
The four remaining refusals were an FLE insert-tool bug on full slots and two
tree collisions the clamp cannot see. All four arms restarted fresh as v4 at
02:30 on this environment with seeds 9 to 12 and 100k-step budgets. This is the
first clean comparison of the methods; everything before it was measured
through one environment defect or another.

## Night 2, 05:50: the last drill wall was a game rule, not a bug

Frontier return worked on the first try on the fixed environment: two
excursions restored the drill-admissible state, opened with the forced drill
craft, and both crafted a drill. Both placements were refused on tiles our
clamp judged free. A 200-call live probe settled it: Factorio's manual build
check refuses a mining drill unless ore lies within its mining area, and every
drill placement any agent attempted since day 1 was on bare ground. Our clamp
now requires ore coverage for drills and moves an unsupported request to the
nearest offset over ore; this is support, since the game refuses everything
else. All arms restarted on it at 05:50. The decision metric (drill placed,
fuelled, producing in a root episode) is for the first time physically
reachable by the policies.

## 07:40: first drill placed on ore

dqn-per-ucbexplorer (plain DQN, epsilon behaviour) placed a burner mining drill on ore four
times in one episode, the first drill placements in the project, each one
through the new ore-coverage clamp. Nothing was inserted into the drill, so it
produced nothing; the remaining rung is INSERT coal into the drill, an
admitted action the policy has not yet chosen. The environment now permits the
whole first automation chain end to end; from here it is the learner's job.

## Morning 3 (2026-09-17, 09:30): the fixed environment learns faster

Seven hours after the v4 restart on the fully fixed environment, every arm is
ahead of where the same configuration stood at the same step count on the
broken one. Last-20 medians of automated PS: Rainbow with optimistic PER and
n-step 5 at 454 (max 870), plain DQN with an explorer/exploiter split at 428
(max 1,169), quantile head with frontier return at 191 (max 1,024), plain
control at 133. Rainbow, flat in both earlier attempts, learns here; the
difference is the environment, not the algorithm. Drill placed on ore once
(dqn-per-ucbexplorer), never fuelled yet; the frontier forcing window on qrdqn-per-ucbexplorer-frontier is armed for
the next drill-rung transition.


## Arm names (2026-09-17)

Descriptive names replace the letters used earlier in this file and in the
log. Run directories on disk keep their launch names.

| Name | Run directory | Configuration |
|---|---|---|
| dqn-per | runs/dqn-per (fresh 11:20; relocation era in runs/ctrl-v4-relocation-era) | branching double DQN, dueling, 3-step, PER a=0.6, epsilon 1.0 to 0.1 over 18k |
| dqn-per-ucbexplorer | runs/dqn-per-ucbexplorer (relocation era in runs/armB-v4-relocation-era) | same learner; port 27000 epsilon 0.2 to 0.05, port 27001 rung-keyed count-UCB explorer at 512 steps |
| qrdqn-per-ucbexplorer-frontier | runs/qrdqn-per-ucbexplorer-frontier (relocation era in runs/armD-v4-relocation-era) | QR-DQN (51 quantiles) + PER; explorer/exploiter as above; frontier return from the drill-admissible state; 24-step forcing window at the drill rung |
| rainbow-optimisticper-n5 | runs/rainbow-optimisticper-n5 (relocation era in runs/rainbow-v4-relocation-era) | QR-51, noisy nets (eps floor 0.02), PER a=0.6 with optimism K=3 and episode-return bonus B=2, n-step 5 |
| dqn-per-v1env | runs/ctrl-v1 | day-1 anchor, old environment, kept as historical reference |

## The algorithms, in enough detail to reproduce

All arms share one learner, `fle/rl/dqn.py`, and one environment. The value
function is a branching Q-network: a shared encoder over the 4,047-float
observation (per-block MLPs for globals, inventory, technology and recipe bits;
a shared row MLP with masked mean and max pooling over the 24 target slots and
the 32 entity slots; a small conv over the 4x17x17 egocentric grid; concatenated
to 512), then one dueling output per action head (14 heads in schema v2: op,
target, entity, item, placeable, recipe, technology, offset, direction,
quantity, duration, move_dir, peer, connector). The Q of a complete action is
the shared value plus the op head's advantage plus the mean advantage of the
heads that op reads. Targets are double-DQN: the online network picks the
argmax per head under the masks, the target network evaluates it. Masks come
from the observation tail and set refused indices to -1e8 before every argmax.

**dqn-per** is that network with 3-step returns, gamma 0.99, PER (alpha 0.6,
beta 0.4 to 1), replay 100k, batch 64, Adam 1e-4, target copy every 1,000
updates, epsilon 1.0 to 0.1 over 18k steps, plus a milestone archive that
supplies 25% of each batch from episodes that reached a ladder rung or scored
in the top 32.

**rainbow-optimisticper-n5** replaces the scalar heads with quantile regression
(51 quantiles per head; the branching combination is applied per quantile and
the mean quantile is used for argmax and masks), replaces epsilon with
factorized NoisyLinear output layers (fresh noise per acting step and per
batch), uses 5-step returns, and changes only how replay is sampled: the
priority of a transition is (|delta| * K)^alpha when its TD error is positive
(the value was underestimated) and |delta|^alpha otherwise, with K=3; and when
an episode ends, all its transitions' priorities are multiplied by 1 + 2 *
min(APS/1000, 1). Rewards and targets are untouched; only which transitions get
replayed changes. This is the Osband-style optimism the owner asked for, placed
in the sampler. It was flat twice on the broken environment and is the best arm
on the fixed one, so the earlier verdict against noisy nets is withdrawn.

**Count-based UCB explorer** (the ucbexplorer arms): one learner, two servers.
One server acts epsilon-greedily (0.2 to 0.05). The other keeps counts
n(rung, op, primary) where the primary is the op's main argument (recipe for
CRAFT, placeable for PLACE, item for INSERT/EXTRACT, target kind for HARVEST,
technology for RESEARCH, and so on) and the rung is the deepest automation
rung the episode has reached so far (furnace fed, plates extracted, gears,
drill crafted, drill placed, drill fuelled). At each step it scores every
admitted (op, primary) pair by z(Q) + c * sqrt(log(N+1)/(n+1)) with c decaying
from 0.75 to 0.15 over 40k steps, where z(Q) standardizes the Q values across
admitted pairs so the bonus is not drowned as Q grows into the hundreds; any
pair with n=0 at the current rung is tried once first. Secondary heads are
greedy except that a first try uses the smallest admitted quantity and a random
direction. Rung keying means PLACE drill becomes a new pair the moment a drill
is first crafted. A forcing window (24 steps after reaching the drill rung)
picks uniformly among pairs tried at most twice. Both servers write to the
same replay; the learner trains on both. This is exploration in the behaviour
policy only; nothing enters the targets.

**Frontier return** (the frontier arm): when an episode first reaches the
state where crafting a drill is admissible, the environment snapshots the game
(FLE's GameState save, about 0.1 s) and with probability 0.5 the explorer's
next episode restores it and runs a 64-step excursion whose first action is the
drill craft. Excursions are tagged, kept out of the root-episode metrics, and
their transitions enter replay normally with real rewards relative to the
restored baseline. This is Go-Explore's return step using only states the
agent itself reached.

**What is still open on the algorithm side:** independent heads cannot
condition an argument on the sampled op (an autoregressive head is the
structural fix), gamma 0.99 discounts the late credit of any investment on a
telescoping objective (gamma 0.999 or 1.0 is the clean comparison), and every
arm has seen about 30k steps, an order of magnitude below where these methods
are usually compared.

## Trace read, 2026-09-17 11:30 (current arms, since the 02:00 restart)

Two findings. First, **the fuel rung has been reached**: in its last 20
episodes dqn-per-ucbexplorer inserted coal into a burner mining drill 19
times. This happened under the relocation code that is now being removed, so
it does not count for the ladder until it recurs on the MDP-faithful
environment, but it shows the policy chooses the fuel step once a drill
stands. Second, **Rainbow wins with a different economy, not tidier
execution**: its last 20 episodes are 49% harvest and 33% insert, and 684 of
its inserts are stone into furnaces, a stone-brick line; dqn-per runs an
iron-and-coal line (273 iron ore, 373 coal inserts). Bricks pay about twice
as much per episode here (median 726 vs 154), and Rainbow's episodes end at
a median of 174 steps because long harvests spend the 60-game-minute cap
faster. The optimistic sampler, which replays positive surprises three times
as often, is the plausible reason it locked onto the higher-value product;
the plain arms are still climbing toward it.

Refusal profile is now dominated by two FLE tool behaviours the masks
deliberately do not model: a furnace already holding one ore refuses another
(a state-dependent rule the observation exposes, so the policy can learn it),
and the insert tool's own error on a full fuel slot. Wasted-step rates are 3
to 10% per arm, down from 20% two days ago.

## Environment versions under the v4 curves (read this before comparing)

The v4 arms started fresh at 02:00 on 2026-09-17 on the schema macro-v2
environment with the D9/D10 support fixes. Every later environment change
restarted the affected arms from their last checkpoint (weights kept, replay
buffer lost), so one curve spans several environment versions. Dotted
vertical markers on the APS plot show the boundaries per arm:

| Marker | Time | Change | Arms |
|---|---|---|---|
| ore rule | 05:50 | drill placement requires ore under it, implemented as relocation to the nearest ore offset | all |
| UCB place key | 07:20 | UCB explorer keys PLACE by placeable | frontier arm |
| forcing window | 09:05 | 24-step forcing window at the drill rung | frontier arm |
| walk-to-ore | 10:50 | drills approach the nearest ore before placing (relocation) | all |

The score curves are unaffected in substance (no arm's score came from
drills), but the drill-rung counts from 05:50 on were obtained under
relocation, which the owner ruled out at 11:05 as a modification of the
policy's action. The D16 reversal removes both relocations; after it lands,
all four arms start fresh so that the next comparison is within one
environment and one weight history.

## 11:20, 2026-09-17: fresh start on the honest environment

The relocation ruling is implemented (D16). Every executor substitution of a
sampled offset, item, anchor or endpoint is gone; an infeasible action is a
zero-reward refusal the policy must learn from, as the LLM agents experience
`place_entity` errors. The only argument clamp left is quantity (reduce a
craft or transfer to the largest coverable count of the same recipe or item),
kept behind a flag that defaults on and documented as a judgement call, since
the inventory counts are visible to the policy. Support masks remove only
deterministic refusals; navigation is performed only to a target the policy
chose; the egocentric grid's resource channel is verified to align with the
offset head, so placing a drill over ore is a learnable two-step plan
(harvest at a patch, then place within reach).

All four arms started fresh at 11:20 with new seeds. Everything measured
before this on the drill rung is labelled relocation-era and kept in
`runs/*-relocation-era` for reference only. From here the comparison is
within one environment version and one weight history per arm.

## Correction (11:30): the relocation-era "fuelled drill" was the executor's

Codex's action-log read (`research/18-behaviour-trace-read.md`) shows that of
the 19 coal inserts into the placed drill, the policy sampled coal only four
times; the item clamp substituted coal for the stone, plates, furnaces and
boilers it actually chose. That milestone is withdrawn as an agent
achievement. Unrelocated drill placement was physically feasible in 55% of
the post-craft positions, but only one of seven sampled placements was
correct as sampled. The fresh arms started at 11:20 run without any
substitution, so their drill counts, when they come, are the agent's.
Rainbow's edge is confirmed as a different economy executed with less waste:
4.7 APS per decision against 0.75 for dqn-per.

## 11:40: restarted once more, on D17, so the comparison stays within one environment

Codex's D17 landed twenty minutes after the 11:20 start (147 tests, ruff
clean, fake-env smoke and resume verified). It changes the environment only
by two deterministic-refusal masks, but a mask change is an environment
change, and no checkpoint existed yet, so all four arms were killed and
started fresh again at 11:40 with the same seeds and flags. The 15-minute
D16 fragments are in `runs/_scratch/*-d16-15min`. D17 contains:

- **UCB behaviour time persisted per port.** The exploration coefficient
  now resumes from its checkpointed value instead of resetting to 0.75 on
  every restart (memo 18 found it had reset twice). Forcing at the drill
  rung now counts distinct complete trials (op, primary, secondary heads),
  so identical invalid variants are not retried.
- **Ore-aware sampling of forced drill placements, without relocation.**
  When a first try or the forcing window picks PLACE with a resource-requiring
  placeable, the offset head is sampled only among admitted offsets whose
  observation tile shows ore; if none is visible the sampled offset stands
  and the action fails. Forced INSERT into a drill samples the item only among
  held fuels. The executed action is always exactly the sampled one; this is
  behaviour-policy sampling, not substitution, and greedy actions are
  untouched.
- **Two deterministic-refusal masks.** EXTRACT from an entity slot with no
  items, and EXTRACT of a mining drill's fuel (the tool cannot read that
  slot), are excluded; the op mask drops EXTRACT only when no visible entity
  has anything extractable, and a first `no_support` of this kind sets the
  cooldown at once (memo 18 counted 80 repeats on one port).
- **CONNECT effect predicate** accepts connector consumption without an
  exception, after a second drain, so pole-laying is no longer logged as
  `predicate_false`.
- **Telemetry.** UCB action rows carry the admitted candidate-set size and
  the number of pairs untried at the current rung, so "what is left to try"
  is recoverable from the logs.

## 12:22: two Rainbows, macro actions against bare actions

Owner's decision at 12:15: Rainbow is the strongest learner on every
environment version it has run on, so the four-arm comparison is stopped and
the compute goes to one question: does absorbing navigation into actions
matter? Two runs of the identical learner and seed:

- **rainbow macro actions** (`runs/rainbow-optimisticper-n5-macro`, ports
  27003 to 27005, started 12:22): the environment as it stands. HARVEST walks
  to the chosen resource, entity operations walk to the chosen entity, MOVE
  exists too.
- **rainbow bare actions** (`runs/rainbow-optimisticper-n5-bare`, ports 27000
  to 27002, starts when D18 lands): no navigation inside any action. HARVEST
  and entity operations act only within reach, MOVE is the only way to move,
  and support masks drop out-of-reach targets and entities as deterministic
  refusals. This is the setting in which the random policy of the op census
  never obtained a single resource (the nearest ore is about 53 tiles from
  spawn and the 17x17 grid cannot see it); whether Rainbow can learn to walk
  there from the target-head features is the experiment.

Both runs use three servers each, stepped concurrently by one learner, and
a 300k-step budget. The 30-minute D17 fragments of the four arms are in
`runs/_scratch/*-d17-30min`; their only content is that Rainbow and dqn-per
had a furnace fuelled and fed in episode 0 and the UCB arms by episode 2.

## The program as of 16:23, 2026-09-17

### What runs

| run | ports | since | learner | action regime | budget |
|---|---|---|---|---|---|
| rainbow macro actions | 27003, 27004, 27005 | 12:22 | Rainbow (QR-51, noisy nets, optimistic PER K=3, episode-return bonus, n-step 5, epsilon floor 0.02), seed 24 | macro: HARVEST and entity ops walk to the chosen target | 300k steps |
| rainbow bare actions | 27000, 27001, 27002 | 16:18 | identical | bare: no navigation inside any action, MOVE (16 tiles, 8 directions) is the only way to move; out-of-reach targets and entities masked as deterministic refusals | 300k steps |

The bare twin started four hours after the macro run because D18 took that
long to land on a saturated machine; the two are compared at equal step
counts, not equal wall time.

### What the macro Rainbow has achieved (99 episodes, 11.1k steps)

- APS median 104 (q1 -1, q3 225), max 619, 63 of 99 episodes positive. Last
  twenty: 231, 43, 40, 110, 43, 156, 564, 224, 225, 224, 268, 225, 380, 88,
  619, 457, 522 and three at -1. Player score median 1036.
- Ladder: furnace placed 91/99, fuelled and fed 68/99, chest placed 7/99.
  Plates extracted 0, gears 0, drill 0. Every point of APS is hand-fed
  smelting: 478 furnaces placed, 8,671 iron ore and 3,637 coal inserted, 352
  furnaces and 117 chests crafted, nothing else crafted.
- The best episodes (APS 522 to 619) are a harvest-feed shuttle: 40 to 65
  HARVEST, 30 to 37 INSERT, 30 to 60 MOVE, a handful of furnaces, 11 to 21
  coal and 15 to 23 iron ore inserts. The same economy Rainbow found on
  every earlier environment version, now on the honest one.
- Op share: INSERT 30%, HARVEST 27%, MOVE 20%, WAIT 6%, PLACE 6%, RESEARCH
  4%, PICKUP 3%, EXTRACT 3%, CRAFT 2%. Status: ok 80%, tool_rejected 14%,
  no_support 6%. The refusals are dominated by INSERT into a furnace whose
  input slot holds a different ore or is full (743 no_suitable_slot, 648
  content conflicts), which are game-contingent and learnable.
- Learner: Q values at 92 and TD loss at 7.2 and rising, epsilon at floor
  0.02 (noisy nets carry the exploration).

### Episode definition, found in the traces

All 101 completed episodes ended on the tick cap (216,000 ticks, one game
hour), never on the 256-step cap, at a median of 115 decisions (55 to 159).
A macro step costs 1,000 to 1,500 ticks of game time because wall latency
on this machine is 1 to 2.3 s per tool call at about 480 UPS. So the
episode is one game hour, and the number of decisions in it depends on
server speed. For the production score this is the fair definition (the
score accrues in game time), and both Rainbows share it, but it should be
stated as the design rather than discovered: an episode is one game hour,
at most 256 decisions.

### What conveniences remain (the "cheating" inventory)

Executor and observation, in order of how much they help the agent:

1. **Navigation inside actions (macro regime only).** HARVEST walks to the
   chosen resource; PICKUP, ROTATE, INSERT, EXTRACT and SET_RECIPE walk to
   the chosen entity. The target is always the policy's; the walk is the
   executor's. The bare run removes this entirely.
2. **Resource radar.** The HARVEST target head lists the nearest resource
   patches out to 160 tiles with direction and distance, beyond the 17x17
   grid. The LLM agents get the same through nearest_resource. Present in
   both regimes; it is what makes bare navigation learnable at all.
3. **Quantity clamp** (flag on): a CRAFT or transfer of a sampled recipe or
   item is reduced to the largest coverable count. Same recipe and item,
   smaller number. Logged on every step as requested and executed quantity.
   Judgement call, could be turned off.
4. **Support masks** remove only deterministic refusals: unheld items,
   uncraftable recipes, unplaceable offsets (reach, water, footprint), empty
   extraction slots, drill fuel slot, out-of-reach targets in bare. Ore under
   a drill is not masked; a drill on grass is sent and refused.
5. **Cooldown mask:** an exactly repeated refused action is masked until
   the visible state changes. Loop breaker, deterministic by construction.
6. **Peaceful mode and enemy clearing at reset.** Owner-approved.
7. **Replay sampling:** optimistic PER and a 25% milestone archive change
   which transitions are replayed, never their rewards or targets.

Not present in either running arm: item, anchor, offset or endpoint
substitution (removed D16), forced first tries and ore-aware offset
sampling (UCB machinery, not used by the epsilon Rainbow), frontier
snapshot and restore, any reward shaping.

### State and action space

Observation: 4,047 floats. Player inventory counts, the 2,048 nearest
entities' rows reduced to 32 slots (class, position, direction, contents,
fuel), 8 HARVEST target slots (resource type, dx, dy, distance), research
state, an egocentric 4x17x17 grid (entities, resources, obstacles, water),
and the support masks for every head packed in the tail.

Action: 14 independent heads. op (12 ops), target (8), entity (32), item,
placeable, recipe, technology, offset (17x17 with the character tile
admitted), direction (4), quantity (1, 5, 20), duration (WAIT ticks),
move_dir (8), peer (32), connector. Each op reads only its own heads.

### Next

Compare the two Rainbows at equal steps (20k, 50k, 100k) on APS
distribution and ladder; read the bare traces early for whether MOVE
trajectories reach ore at all. Codex trace read of the macro run is in
`research/19-rainbow-macro-trace-read.md` when it lands. The compute box
recipe stands: 48 to 64 vCPU x86 Linux, no GPU, 16 to 24 servers, one
learner per arm, seeds of the winning regime.

### Trace read of the macro Rainbow (Codex, 16:20 snapshot; `research/19-rainbow-macro-trace-read.md`)

- **Economy.** All five best episodes (APS 457 to 619) are one to six stone
  furnaces hand-fed with iron ore and coal; across them 1,508 iron ore and
  472 coal went in, outputs stayed in the furnaces. No drill anywhere.
- **Learning.** Per port, first-20 to last-20 APS mean: 24 to 240, 50 to
  172, 120 to 221. Over 20-episode blocks the policy moved from 34% MOVE
  and 17% WAIT to 43% HARVEST, 24% INSERT and 0.2% WAIT, while refusals fell
  from 22% to 8% of steps. Q rose 0.16 to 91.5 and TD loss 0.5 to 7.2
  alongside returns; the optimistic sampler holds positive priorities at
  about twice negative.
- **Conveniences counted.** 4,375 executor navigations on non-MOVE steps
  (HARVEST 2,924, INSERT 1,256, PICKUP 174), 728 quantity clamps (INSERT
  565, CRAFT 153). Zero substitutions of item, anchor, peer or offset.
- **Waste.** 46% of steps produced no item delta and zero reward. 177
  adjacent PLACE then PICKUP pairs. 77 runs of identical refusals with 457
  repeats, the worst 84 identical wrong-input INSERTs in one episode.
- **Defect found: the exact-action cooldown never fires.** cooldown_hit and
  cooldown_masked are zero on all 10,975 steps. The most likely cause is
  that the support fingerprint changes every step while a furnace smelts,
  invalidating the cooldown immediately. D19 dispatched to reproduce and
  fix, plus an observation-backed INSERT op mask (no visible entity accepts
  any held item given its input) and PLACE executed_offset telemetry.
- **FLE tool behaviour, not ours:** 136 "inventory full" errors with 12 to
  19 of 50 slots occupied and 221 invalid LuaItemStack reads.

## Result of the macro-versus-bare comparison (stopped 2026-09-18 03:13 by the owner)

Same Rainbow learner and seed, D18 environment, three servers each, one
game hour per episode.

| run | steps | episodes | last-20 APS median | all-episode median | max | positive |
|---|---:|---:|---:|---:|---:|---:|
| rainbow macro actions | 40k | 408 | 1347 | 731 | 1457 | 371/408 |
| rainbow bare actions | 34k | 442 | 0 | 0 | 0 | 0/442 |

**Macro.** The median rose monotonically for eleven hours (104 at 11k
steps, 457 at 14k, 883 at 18k, 1013 at 20k, 1300 at 36k) and had just
left a plateau at 1300 when stopped. Every point came from hand-fed stone
furnaces smelting iron ore with coal; the ladder never went past
"furnace fuelled and fed". Codex's trace read (research/19) confirmed
zero argument substitutions, 4,375 executor navigations and 728 quantity
clamps as the only conveniences used. Kept checkpoints at 20k, 30k, 40k.

**Bare.** With no navigation inside actions and the automated score as
reward, the policy first collapsed to MOVE, WAIT and RESEARCH (hand work
is charged -1 and never repaid), then around 20k steps began sustained
harvesting bursts (72 HARVEST in an episode, 1,061 coal; another episode
crafted 43 furnaces and placed 12), then receded again. It learned
navigation to resources, harvesting, crafting and placing from MOVE
alone, but never inserted fuel and ore into one placed furnace, so the
automated score stayed at 0 or -1 in all 442 episodes. The 20k and final
checkpoints are kept.

**Reading.** Under the published automated score, absorbing navigation
into actions is worth the difference between 1347 and 0 at this budget;
the bare regime's missing link is one INSERT pair with no reward gradient
toward it. The open option, not run: bare regime rewarded on the raw
production score and measured on the automated score, to test whether
denser reward buys the furnace loop or only buys harvesting.
