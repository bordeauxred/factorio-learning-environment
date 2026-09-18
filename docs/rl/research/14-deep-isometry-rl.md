# Deep isometry RL: does the Rosseau repo help our open play learner?

Verdict up front: skip for now, keep as one cheap arm for later. The repo is about training deep value networks without collapse. Our learner is one hidden layer deep and our failures are in exploration and credit assignment. Nothing in the repo addresses those.

Access note. The repo is private and every web URL returned 404. The `gh` CLI, logged in as this machine's GitHub account (bordeauxred), has read access, so I read the files through `gh api`. Nothing was cloned into the project.

## 1. What the repo is

The repo is a research workspace, not a library. Facts from the GitHub API on 2026-09-16:

- Private, Apache 2.0, 0 stars, 0 forks, one branch (`master`).
- Created 2026-06-11T22:48:04Z, last push 2026-09-10T21:07:07Z, about 1,290 commits (Arosseau 1,203, a `claude` author 87).
- 2,815 files. 191 are Python, 751 are shell launchers, 596 are result CSVs. `RESEARCH_LOG.md` is 1.59 MB and 17,182 lines.
- Framework is JAX and Flax. The code lives in `isorl/`. The main Atari trainer `isorl/pqn_atari.py` is 305 KB in one file.
- 16 test files under `tests/` (e.g. `tests/test_muon_scope_brax_tree.py`, `tests/test_simba2.py`). No pyproject or requirements file.

The owner is Andries Rosseau (VUB). The README states the mission as making "actually deep (4 to 128+ layer) networks train stably AND pay off in GENERAL reinforcement learning" with first-order optimizers, target ICLR 2027. The README cites arXiv 2606.09762, "Preserving Plasticity in Continual Learning via Dynamical Isometry" (Rosseau, Robert Müller, Ann Nowé, ICML 2026), as the sister project on plasticity, and says this repo is the depth complement with no overlap.

The paper draft is `paper/iclr2026/deep_isometry_rl.tex`, titled "Deep Reinforcement Learning Without Collapse: End-to-End Isometry from Initialization to Optimizer". Its abstract claims a 32-layer network improves the median score over the standard baseline by 47% on 29 Atari games. That abstract is stale. Commit f1615050 (2026-09-10) says "correction of the +48% claim; Atari verdict = robustness, not score-with-depth", and the program conclusion in `CATALOG.md` says "depth beyond ~8 isometric blocks is never required" on Atari. The current claim is that the recipe makes depth harmless and makes learning rate less sensitive.

Domains tested: Atari (PQN and PPO), DeepMind Control via Brax (SAC, TD3, PPO), Craftax-Symbolic, Octax, Procgen, MinAtar, and JaxGCRL. Craftax is the closest to us. There, depth paid: greedy return 18.7, 21.6, 25.5 for 1, 2, and 33 blocks at 100M steps, width 1024 (log entry XCIII, 2026-08-25, 1 to 2 seeds per cell). The per-achievement table in `analysis/craftax_achievements_20260826.md` shows the deep policy collected iron in 44.2% of episodes against 5.1% for the shallow one. The SAC ladder on DMC (n of at least 3) went x1.11, x1.44, x1.50, x1.62 for 2, 4, 8, 16 blocks over the published baseline.

## 2. The core idea

The network. A layer is isometric when the singular values of its input-output Jacobian are close to one, so the signal neither shrinks nor grows as it passes through. The claim is that TD training breaks this faster than supervised training because the network's own outputs are the targets. The recipe (`CATALOG.md`, "Recipe of record") is: an encoder, then a Dense-512 entry projection followed by a parameter-free RMS norm, then L residual blocks of the form `x + (1/L) * SReLU(RMS(Dense(x)))`, then one ReLU and a linear Q head. RMS(z) is `z / sqrt(mean(z^2) + 1e-6)` with no learnable gain. SReLU(z) is `max(z, thr)` with thr a learnable per-channel threshold initialised at -1. Dense weights use orthogonal initialisation with gain sqrt(2). The 1/L branch scale keeps the sum of L branches at unit size. The code is `ResidualBlock` in `isorl/deep_mlp.py`, and `_append_const_feature` adds a constant-1 input so an all-zero observation never puts the norm at its singular point.

The optimizer. Every 2-D weight matrix, including the Q head, is trained with Muon (`isorl/muon.py`). Muon keeps a momentum average of the gradient, orthogonalises that matrix with five Newton-Schulz iterations so all its singular values become about one, and steps by the learning rate times that matrix. Biases, thresholds and conv kernels go to an Adam group with epsilon about 8e-8/L. The learning rate is 5e-4 annealed to zero. The ICML paper's own method is different. AdamO adds a decoupled penalty `||W^T W - I||_F^2` (or `W W^T - I` on the smaller side) with lambda 1e-3, applied outside Adam's moment estimates (`isorl/adamo.py`). The repo tested that penalty and found it redundant once the init gain is compensated, so the recipe sets `ORTHO_LAMBDA=0`.

## 3. Fit to our setting

The repo addresses one problem, which is training deep value networks under bootstrapped targets without the trunk collapsing. It does not address exploration, value propagation across long zero-reward chains, or sample efficiency at low data. Two side results are useful to us anyway. An RND bonus at scale 0.5 was "TOXIC at both depths", and EMA targets were "neither necessary nor sufficient" (`CATALOG.md`).

Our network in `fle/rl/dqn.py` has 1.26M parameters. After the per-block encoders the trunk is a single `Linear(640, 512)` plus ReLU, and the 12 heads are linear. There is no depth to stabilise. Our learner does about 60k updates in 8 hours. The plasticity benchmarks in the ICML paper run 20 task changes. Nothing in `overnight_v1.md` points to a trainability failure. The greedy control policy reached 574 APS, and the UCB arm's Q of -1 came from discoveries never entering the value function, which `13-algorithms-for-open-play.md` attributes to sampling, the 0.99 discount and the independent argument heads.

The one link is Craftax. A tech tree is our structure too, and the deep policy's advantage there sat at the deepest reached tier. But that run used 100M steps with 1,024 parallel environments. We have 60k steps. The repo contains no test in a low-data regime.

Integration cost is low. In `fle/rl/nets.py`, add a trunk module of K residual blocks as above, about 40 lines, plus orthogonal init with gain sqrt(2). In `fle/rl/dqn.py`, replace `torch.optim.Adam` (line 1836) with `torch.optim.Muon`, which exists in the installed torch 2.14.0, for 2-D matrices and keep Adam for the rest, about 30 lines. Newton-Schulz on a few 512 by 512 matrices per update is negligible against one environment step per second. One risk is that our learner uses replay, a target network and PER, while the recipe was tuned for PQN with none of those. The repo's SAC and TD3 arms do use replay and target networks and the recipe held there, so the transfer is plausible.

The control experiment is three arms on two servers each, with identical replay, PER, exploration, seeds and step budget:

- A: current trunk with Adam.
- B: 4 isometric blocks with Muon.
- C: 4 isometric blocks with Adam, to separate depth from optimizer.

Report held-out greedy median APS over 16 map seeds, ladder conversion rates, repeat rate after the first gear and drill craft, and Q-value spread. Adopt only if B beats A on held-out APS and repeat rate.

## 4. Verdict

Skip now. Try as one arm later. The repo is a serious, well-logged single-owner workspace, but its subject is depth, and depth is not our bottleneck. Our next runs should test the exploration and replay changes in `13-algorithms-for-open-play.md`. Once an arm produces a rewarding suffix and value propagation becomes the limit, the isometric trunk with Muon costs about a day to add and is a reasonable arm to include.
