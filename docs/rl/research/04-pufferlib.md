# PufferLib integration: what the library actually constrains

Researched 2026-09-13 against the upstream repo and PyPI. Every claim below is verified
against source unless marked UNVERIFIED.

## 1. Version landscape — pick 3.0, not 5.0

| Line | Where | Layout | Verdict for us |
|---|---|---|---|
| **3.0.0** | PyPI latest stable (`pip install pufferlib`) | Python package `pufferlib/` with `pufferlib.py`, `emulation.py`, `vector.py`, `pufferl.py`, `models.py`, `pytorch.py`, `spaces.py` | **Target this.** |
| **5.0** | GitHub default branch; release tag `5.0-experiments` published 2026-09-13 | No Python package at all. Top level is `config/ ocean/ src/ vendor/`; `src/` is `algo.cu constellation.c ocean.cu protein.cu puffercpu.c pufferenv.h pufferl.cu` | **Wrong target.** C/CUDA-first; env contract is a C header (`pufferenv.h`), training is `./puffer train`. |
| 2.0.x | PyPI | Older Python layout | Superseded. |

PufferLib 5.0 assumes the environment is C code compiled into the trainer and stepped in
shared memory. Our environment is a Python process issuing RCON commands to a Factorio
server over TCP. There is no path to expressing that as a `pufferenv.h` implementation, so
5.0 is out. **Pin `pufferlib==3.0.0`.**

Note this also resets expectations about what PufferLib buys us — see §4.

## 2. Hard API constraints (PufferLib 3.0, `pufferlib/pufferlib.py:45-71`)

`PufferEnv.__init__` validates, and raises `APIUsageError` otherwise:

- `single_observation_space` **must be a `Box`**. Not a `Dict`, not a `Tuple`.
- `single_action_space` must be `Discrete`, `MultiDiscrete`, or `Box`.
- Must define `single_*` spaces, *not* `observation_space` / `action_space` (those are
  synthesized as joint spaces).
- `num_agents >= 1` required.
- `reset(seed)` / `step(actions)` / `close()` are the implemented surface; native envs
  handle their own resets (`done` property returns `False`).

**Consequence 1 — the flat observation vector is mandatory, not a stylistic choice.**
The proposal's "concatenate everything into one float32 vector and feed an MLP" is exactly
what the native API requires. A `Dict` observation would force us through
`GymnasiumPufferEnv` emulation, which flattens it anyway.

**Consequence 2 — `MultiDiscrete` is the right action space** for the factorized heads and
is natively supported.

## 3. There is NO built-in action masking — this is the big one

Two distinct things are called "mask" in PufferLib and only one exists:

- `env.masks` (`pufferlib.py:29,42,111`) is an **agent-liveness mask** of shape
  `(num_agents,)`. It tells the trainer which agents produced a real transition this step.
  It is *not* an action mask.
- **Action masking is absent.** `pufferl.py:298` carries the literal comment
  `# Note: We are not yet handling masks in this version`, and the sampling call
  `pufferlib.pytorch.sample_logits(logits)` (`pufferl.py:274`, and `:388` with
  `action=mb_actions` for log-prob recomputation during the update) takes no mask argument.

So we implement masking ourselves. Because the observation must be a `Box`, **the action
mask has to be packed into the observation vector** and sliced back out inside the policy.
That is the only channel the API gives us.

### The workable pattern

The policy contract is `forward(observations, state) -> (logits, value)` with
`encode_observations` / `decode_actions` hooks (`models.py:65-98`). Since the policy sees
the raw observation in *both* the rollout path and the update path, it can:

1. slice the mask block out of the observation,
2. compute per-head logits,
3. set invalid entries to `-inf` **before returning**,
4. let PufferLib's unmodified `sample_logits` build `Categorical`s over already-masked logits.

This is correct in both rollout and update because the same masked logits are reconstructed
from the stored observations. **No fork of PufferLib is needed for this.**

### What this pattern cannot do: autoregressive conditional masking

The proposal wants
`π(a|s) = π(op|s)·π(arg₁|s,op)·π(arg₂|s,op,arg₁)·…`, where each head's mask depends on the
values *sampled* for earlier heads in the same step. PufferLib samples every head
independently from one list of logits in a single external call, so a head cannot condition
on a sibling's sampled value. This is a genuine architectural fork in the road:

- **Option A — independent heads, state-only masks.** Each head is masked by what is valid
  in state `s` marginalized over the other heads (e.g. `recipe_id` masked to recipes valid
  for *some* currently-selectable machine). Invalid *combinations* still get sampled; the
  env rejects them as no-ops with the invalid penalty. Zero PufferLib modification.
- **Option B — vendor `pufferl.py` and replace the sampling path.** `pufferl.py` is a
  single ~1000-line file; vendoring it to add autoregressive masked sampling (and matching
  log-prob recomputation against `mb_actions`) is mechanically straightforward but creates
  a maintenance fork and touches the PPO update's log-prob/entropy math, which is where
  subtle correctness bugs live.

Recommendation: **A for V0, B as a measured upgrade** once we can quantify the invalid-combination
rate from real rollouts. Ship A, log the rate, and only pay for B if the number is bad.
UNVERIFIED: what that rate actually is — it depends on the masking scheme and cannot be
predicted from source.

## 4. What PufferLib is actually buying us here

PufferLib's headline value is throughput for environments that step in microseconds. Ours
steps in milliseconds-to-seconds and is bound by a Factorio server, so the C-level
optimizations are irrelevant. What we genuinely get:

1. A working, single-file, well-exercised PPO implementation (`pufferl.py`).
2. Multiprocessing vectorization with shared-memory buffers (`vector.py`) across N
   Factorio containers.
3. A conventional training CLI, sweeps (`sweep.py`), and a policy/model scaffold.

That is worth having, but it means **PufferLib is not on the critical path for performance**.
The bottleneck is Factorio simulation time. Any claim that "PufferLib makes this fast" would
be wrong, and the design doc should not make it.

## 5. Dependencies not currently in the repo

`pufferlib` and `torch` are both absent from the project venv and from `pyproject.toml`
(`pyproject.toml:39-40` has only `numpy>=2.2.3` and `gymnasium>=1.3.0`). They must be added
under an optional extra (e.g. `[project.optional-dependencies] rl = [...]`) so the core FLE
install stays light — FLE's LLM-agent users should not be made to install torch.

## 6. Open items

- UNVERIFIED: exact `sample_logits` behaviour for a `MultiDiscrete` list of logits tensors
  (shape/order contract). Must be read before writing the policy.
- UNVERIFIED: whether `vector.Multiprocessing` tolerates per-env step latencies in the
  hundreds of milliseconds without timeout, and whether it can be configured to avoid
  lockstep stalls when one env is much slower than the others (a real risk here — a
  `move_to` step is far slower than a `CRAFT` step).
- UNVERIFIED: whether `pufferlib==3.0.0` has a wheel for this platform. `pip download
  pufferlib==3.0.0 --only-binary :all:` on macOS/py3.12 found no matching distribution;
  the sdist build was not completed during this research. Installation must be validated
  early — it is a cheap check that de-risks everything downstream.

## 7. AMENDMENT — verified `sample_logits` contract and a NaN trap

Read from `pufferlib/pytorch.py:189-219` on branch 3.0. This resolves the open item in §6
and adds a hard implementation constraint.

**Multi-discrete is supported via a list of per-head logits tensors, and heads may have
different sizes.** The multi-discrete path pads the list to a common width:

```python
else: #multi-discrete
    logits = torch.nn.utils.rnn.pad_sequence(
        [l.transpose(0,1) for l in logits],
        batch_first=False,
        padding_value=-torch.inf
    ).permute(1,2,0)
```

Padding with `-inf` is how ragged head sizes are made rectangular, so our heads do not all
need the same cardinality. Good.

### The trap: a fully-masked head produces NaN loss

After padding, the code does `logits - logits.logsumexp(dim=-1, keepdim=True)` and
`logits_to_probs(logits)` (`pytorch.py:213-214`). If we mask by writing `-inf` into invalid
entries and **every** entry of some head is invalid, that head's row is all `-inf`, its
`logsumexp` is `-inf`, and the probabilities become `NaN`.

On the rollout path this is silently papered over — `torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)`
runs only inside `if action is None:` (`pytorch.py:216-217`). On the **update** path, where
`action=mb_actions` is supplied (`pufferl.py:388`), that repair does not run, so the NaN
propagates into `log_prob` and the PPO loss. The failure therefore appears as a corrupted
gradient during training, not as a crash during rollout, which is the worst debugging shape.

**Two rules follow, and both belong in the implementation:**

1. **Every head must always have at least one valid entry.** Give each argument head a
   reserved index 0 meaning "unused / no-op" that is never masked out. For an op that does
   not read a given head, the mask leaves only index 0 open. This also gives the heads a
   well-defined value to store in the rollout buffer for unused arguments, which the PPO
   update needs for log-prob recomputation.
2. **Mask with a large finite negative, not `-inf`.** Use `-1e8` (or `torch.finfo.min/2`).
   It behaves identically for sampling while keeping `logsumexp` finite, and it does not
   collide with the `-inf` the padding path writes.

Add an assertion in the policy that no head is fully masked, and a test that runs one PPO
update over a batch containing a maximally-masked observation and asserts the loss is finite.

### Additional risk

`pytorch.py:204` carries the upstream comment `# TODO: Double check this` directly above the
multi-discrete branch. PufferLib's own authors flag this path as unverified. We are relying
on it, so our test suite should pin the behaviour we depend on (ragged head sizes, padding
semantics, log-prob recomputation matching between rollout and update) rather than trusting it.

## 8. AMENDMENT — hard dependency conflict with FLE (verified, blocking)

Read from the PyPI metadata for `pufferlib==3.0.0` and `pyproject.toml`.

`pufferlib==3.0.0` declares, among others:

```
numpy<2.0
gym<=0.23
gymnasium<=0.29.1
pettingzoo<=1.24.1
shimmy[gym-v21]
torch, psutil, pynvml, rich, imageio, pyro-ppl, heavyball, neptune, wandb
```

FLE declares (`pyproject.toml:39-40`):

```
numpy>=2.2.3
gymnasium>=1.3.0
```

**These are directly unsatisfiable in one environment.** `numpy<2.0` against `numpy>=2.2.3`,
and `gymnasium<=0.29.1` against `gymnasium>=1.3.0`. A plain `pip install pufferlib` into the
FLE venv will either fail to resolve or silently downgrade numpy and gymnasium underneath FLE.

Two further facts make this worse:

- **`pufferlib==3.0.0` ships as sdist only.** The PyPI release has exactly one file,
  `pufferlib-3.0.0.tar.gz`, with no wheels for any platform. Installation compiles C
  extensions from source, which is why `pip download --only-binary :all:` found no
  distribution on macOS/py3.12.
- **It drags in a large dependency tail** including `torch`, `wandb`, `neptune`, `pyro-ppl`
  and `heavyball`. This is not a light addition to a repo whose current RL-adjacent
  dependency set is numpy plus gymnasium.

### How bad is the conflict really

Evidence that it is probably resolvable rather than fatal:

- FLE's use of gymnasium is shallow: `gym.register`, `gym.make`, `gymnasium.spaces`
  (`fle/env/gym_env/registry.py:7,78`, `fle/env/gym_env/environment.py:3,5`). Those APIs are
  stable across 0.29 and 1.x, so the `gymnasium<=0.29.1` pin is unlikely to break FLE in
  practice. UNVERIFIED whether `gym.register`'s exact signature as called at
  `registry.py:78` is 0.29-compatible.
- No numpy-2-only API usage was found in `fle/`. The one relevant line is a backwards shim,
  `np.bool8 = np.dtype(np.bool)` (`fle/env/gym_env/environment.py:31`), which restores a name
  numpy 2 removed. That shim suggests the `numpy>=2.2.3` floor is a convenience pin rather
  than a hard requirement. UNVERIFIED whether FLE's full test suite passes on numpy 1.26.

So the likely resolutions, cheapest first:

1. **Separate trainer venv.** The PPO trainer imports `pufferlib` and `torch`; the env worker
   imports `fle`. If they are separate processes they can be separate environments. This is
   the least invasive option and it fits the vectorized architecture, but it needs the
   observation/action transport between them to be plain arrays, which it already is.
2. **Relax FLE's floors for the `rl` extra** and verify FLE's tests on numpy 1.26 /
   gymnasium 0.29. Cheap to try, and if it passes it is the simplest single-env setup.
3. **Install pufferlib from git with patched pins.** PufferLib's pins are historically
   over-tight; `numpy<2.0` in a 2026 release is very likely stale rather than meaningful.
   Creates a fork to maintain.

### Consequence for the design

This reframes the "why PufferLib" question, and the design doc should say so honestly.
Combining §4 with this section: PufferLib gives us a PPO implementation and multiprocessing
vectorization, contributes nothing to throughput for a network-bound environment, has no
action masking, has a multi-discrete sampling path its own authors flag as unchecked, ships
without wheels, and conflicts with FLE's two existing pins. The alternative — a vendored
CleanRL-style PPO of roughly 300 lines plus our own worker pool — is a real option and should
be named as one, with the recommendation made explicitly rather than assumed.

**Phase 0 of implementation must be: install `pufferlib==3.0.0` alongside FLE and run one
PPO update on a toy env.** It is a few hours of work and it de-risks everything downstream.
If that fails, the vendored-PPO path is chosen on evidence instead of taste.
