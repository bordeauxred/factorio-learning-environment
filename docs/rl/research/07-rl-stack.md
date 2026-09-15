# RL stack for goal-conditioned off-policy training with HER

Researched 2026-09-13. The requirement changed: we need an **off-policy algorithm with a
replay buffer** so we can do **Hindsight Experience Replay** over a **goal-conditioned**
policy, on the **open track**. That requirement settles several questions that were open
before.

## 1. PufferLib is out, and not marginally

`pufferlib/pufferl.py` on branch 3.0 contains exactly one algorithm: a single `PuffeRL`
class implementing PPO, with `compute_puff_advantage` for GAE. There is **no replay buffer,
no Q-network, no target network, no soft update, and no off-policy algorithm of any kind**.
Verified by scanning every class and function definition in that file.

So PufferLib cannot do HER. This is not a dependency problem that can be patched; the
capability does not exist. Combined with the previously documented facts (§8: unsatisfiable
`numpy<2.0` / `gymnasium<=0.29.1` pins against FLE's floors, sdist-only, large dependency
tail; §3: no action masking; §7: a multi-discrete sampling path its own authors mark
`TODO: Double check this`), there is no remaining argument for it.

**Decision: drop PufferLib.** We keep exactly one thing from that line of work, which is the
observation-must-be-a-flat-Box discipline, and we keep it because it is good practice for
batching, not because a library forces it.

## 2. The candidates, judged against what we actually need

We need, simultaneously: a **factorized (MultiDiscrete) action space**, **per-head invalid
action masking**, **goal conditioning (UVFA-style: the goal is an input to the value
function)**, **a replay buffer with HER relabelling**, and tolerance for an environment whose
step takes 0.1 to 2 seconds and lives behind a TCP socket.

| Option | HER | Off-policy | MultiDiscrete + masking | Verdict |
|---|---|---|---|---|
| **PufferLib 3.0** | No | **No** | No masking | Out, see §1 |
| **Stable-Baselines3** | `HerReplayBuffer` built in | SAC, TD3, DQN | **No.** SAC is Box-only, DQN is Discrete-only. No factorized discrete off-policy policy exists | Out for our action space |
| **Tianshou** | `HERReplayBuffer` (future strategy only, online sampling only, needs Dict obs with `observation`/`achieved_goal`/`desired_goal`, takes a `compute_reward_fn`) | Yes, incl. `DiscreteSAC` | Partial. `DiscreteSAC` assumes a single `Discrete` head; factorized heads need a custom policy. Masking support in the DQN family via a `mask` convention; for `DiscreteSAC` it is UNVERIFIED | **Viable option**, with custom policy |
| **rejax / Stoix / purejaxrl** | varies | yes | n/a | **Out.** These jit and vmap the *whole training loop including env.step*. Our step is an RCON round trip to a Factorio server. It cannot be jitted or vmapped. Wrong tool, not a close call |
| **CleanRL-derived single file** | we write it | we write it | we write it | **Recommended primary** |

### Recommendation: vendor a single-file trainer, keep Tianshou as the named fallback

The honest situation is that **no library supports factorized + masked + goal-conditioned +
HER out of the box**. Under every option we write the policy network and the relabelling
reward ourselves. Given that, a vendored CleanRL-style single file wins on three counts:

1. **No dependency conflict at all.** CleanRL is designed to be copied, not installed. The
   numpy/gymnasium problem simply does not arise.
2. **Full control of the masking path**, which is where the subtle correctness bugs live
   (see the NaN trap in `04-pufferlib.md` §7 for what happens when a library's sampling path
   meets our masks).
3. **Full control of HER relabelling**, which for our goal space needs a custom
   `compute_reward` over stored achieved-goal vectors anyway.

Cost: roughly 800 to 1,200 lines we own and must test. That is real but bounded, and the
alternative is fighting a library's assumptions at every one of the four axes above.

Tianshou stays named as the batteries-included option if we would rather inherit a tested
buffer and trainer and only write the policy. Its `HERReplayBuffer` restriction to the
`future` strategy is acceptable, since `future` is the strategy the HER paper found best.

## 3. Algorithm choice: branching double DQN over discrete SAC

Two credible families for a factorized discrete action space.

**(a) Branching / factorized double DQN** (in the spirit of Tavakoli et al., Branching
Dueling Q-Networks). One Q head per action dimension over a shared trunk, with the shared
state value and per-branch advantages. Greedy action is the per-branch argmax, which is an
approximation to the joint argmax but a well-studied one.

**(b) Discrete SAC** (Christodoulou 2019) with factorized heads.

**Recommend (a).** The deciding argument is masking. In Q-learning, masking is exact: set
invalid actions' Q values to a large negative before both the behaviour argmax and the
bootstrap target max, and invalid actions are never selected and never enter a target. In
discrete SAC the entropy term and the soft-value term are sums over the action distribution,
so masking has to be threaded correctly through the temperature objective and the target
soft value as well; discrete SAC is also known to be temperature-sensitive. For a first
working system on an expensive environment, exactness and fewer knobs win.

Build (a) with the standard sample-efficiency stack: **n-step returns** (n=3), **double
Q**, **dueling**, **prioritized replay**, and **a high replay ratio**.

### The replay ratio is the point

This is the part that changes the compute story. The previous PPO plan needed roughly 10M
environment steps and the sizing table in `DESIGN.md` §5 showed that costing 21 to 79
containers for a 72-hour run. PPO is on-policy: every sample is used for a handful of
gradient steps and then thrown away.

Off-policy replay reuses every transition many times. With a high update-to-data ratio and
the periodic-reset trick from the replay-ratio literature (SR-SPR, D'Oro et al., which uses
network resets to avoid the primacy bias that otherwise caps the usable replay ratio), sample
efficiency improves by roughly an order of magnitude on discrete control benchmarks.
UNVERIFIED that this transfers to Factorio, but the direction is not in doubt, and HER
multiplies it again by manufacturing successful transitions out of failures.

**This is the strongest argument for the pivot.** It is not only a nicer research framing; it
is what makes the hardware budget plausible. The compute moves from "simulate Factorio 10M
times" to "simulate Factorio ~0.5M to 1M times and do many gradient steps per step", and
gradient steps are cheap and GPU-parallel while Factorio steps are neither.

## 4. Goal generation: keep the GOID criterion, question the GAN

The ask was Goal-GAN style: sample a goal, let the agent pursue it, relabel on failure.

Goal-GAN (Held et al. 2018) has two separable parts. The part that matters is the **GOID
criterion**: train on *goals of intermediate difficulty*, empirically those whose current
success rate falls in roughly [0.1, 0.9]. Too-easy goals give no gradient, too-hard goals give
no reward. The part that is incidental is the **LSGAN** used to generate such goals.

For Factorio the GAN is a poor fit and I would not start there:

- Our goal space is a **sparse non-negative integer vector over an item vocabulary**
  ("produce at least 40 iron plates and 10 gears automatically"). GANs model continuous
  densities; generating discrete count vectors means generating floats and rounding, which
  loses exactly the structure that matters.
- We get a free generator that a GAN would have to learn: the **achieved-goal distribution
  already in the replay buffer**. Every episode produces a real, reachable, correctly-typed
  goal vector for free. Sampling and perturbing those is strictly better-conditioned than
  learning a generator from scratch.

**Recommend:** curriculum by **achieved-goal resampling with a difficulty band**. Maintain
per-goal (or per-goal-cluster) success-rate estimates, sample training goals whose estimated
success rate is in the intermediate band, and extend the frontier by scaling up achieved
goals (multiply counts, add a new item) rather than by generating from noise. This is the
GOID criterion with a non-parametric generator.

Keep Goal-GAN as an explicit option to implement second, once the non-parametric curriculum
gives a baseline to beat. If the user wants the GAN specifically, it slots into the same
interface: anything that returns a goal vector is a goal generator.

## 5. HER is unusually well-suited to this environment

Worth stating, because it is the reason the pivot is right and not merely different.

- **The achieved goal is already computed.** The design already tracks per-item automatic
  production counts with manual crafting and hand-mining subtracted. That vector *is* `m(s)`.
- **Achieved goals are monotone within an episode.** Cumulative automatic production never
  decreases, so once a goal is achieved it stays achieved. This makes the `final` and
  `future` relabelling strategies clean and makes goal-reaching a first-hitting-time problem
  rather than a maintenance problem. Note this is true of *counts* and false of *rates*,
  which is a reason to prefer count goals.
- **Failure is informative in exactly HER's sense.** An agent told to make circuits that
  instead builds a working iron-plate line has not wasted the episode; it has produced a
  perfect demonstration for the goal "make iron plates". In a sparse open-ended game this is
  the single largest source of usable signal.
- **Reward recomputation is offline and cheap.** `r(s', g) = 1[m(s') >= g]` on the support of
  `g` is a vector comparison over stored data, no environment interaction.

## 6. Open track versus task-based: HER dissolves the choice

The user flagged this as a decision to make. With goal conditioning it largely stops being
one.

Train goal-conditioned on the **open track**, where the agent has an unbounded state space
and the curriculum generates its own goals. Then **evaluate** on the bounded lab throughput
tasks by simply setting the desired goal to that task's target vector. A "task" is a fixed
goal; the goal-conditioned policy is a superset.

That gives one training regime and a clean, comparable evaluation against the existing FLE
task suite, which is what makes the result legible to anyone who already knows FLE.
Recommend training open, evaluating on both self-generated held-out goals and the existing
named tasks.

## 7. Open items

- UNVERIFIED: whether a branching argmax is adequate for our action factorization, or whether
  cross-head coupling (the anchor and the prototype are strongly dependent) degrades it enough
  to need a sequential/autoregressive Q decomposition. Measure against a random and a scripted
  baseline before concluding.
- UNVERIFIED: achievable replay ratio before instability, and whether periodic resets help here.
- UNVERIFIED: Tianshou `DiscreteSAC` masking support, if we take the library route.
- The time-cost and potential-shaping terms from the PPO design interact awkwardly with sparse
  goal-conditioned HER rewards. HER wants a clean sparse indicator. Decide whether shaping is
  retained as a goal-conditioned potential or dropped in favour of pure sparse plus HER.
