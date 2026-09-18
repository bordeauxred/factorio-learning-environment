"""A server-free stand-in for FleMacroEnv with the exact same tensor contract.

Learners develop and unit-test against this; it produces self-consistent
observations and masks from ``fle.rl.schema`` and a small deterministic reward
signal so that a correct learner visibly improves on it. It is not a model of
Factorio and must never be reported as one.

Reward rule: the hidden state holds one "good" op and one "good" argument index
for that op's first head; choosing both yields +1, the op alone +0.1. The masks
always admit the good pair, so improvement over a random policy is achievable
and measurable. Episodes last ``horizon`` steps.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np

from fle.rl import schema as S


@dataclass(frozen=True)
class FakeFrontierState:
    """In-memory equivalent of ``FrontierState`` for offline plumbing tests."""

    step_index: int
    max_steps: int
    good_op: int
    good_arg: int
    rng_state: dict
    entity_hash: str
    observation: np.ndarray


class FakeMacroEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 256,
        seed: int | None = None,
        mask_density: float = 0.3,
        regime: str = "macro",
    ):
        if regime not in {"macro", "bare"}:
            raise ValueError(f"Unknown action regime {regime}")
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(S.OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(S.HEAD_DIMS)
        self.horizon = horizon
        self.mask_density = mask_density
        self.regime = regime
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._episode_max_steps = horizon
        self._good_op = 0
        self._good_arg = 0
        self._excursion = False
        self._restore_hash: str | None = None
        self._excursion_steps = 0
        self._excursion_budget: int | None = None

    def _observe(self) -> np.ndarray:
        obs = np.zeros(S.OBS_SIZE, dtype=np.float32)
        a, _ = S.OBS_LAYOUT["globals"]
        obs[a] = self._t / self._episode_max_steps
        obs[a + 1] = self._good_op / len(S.OPS)          # the fake state reveals its secret
        obs[a + 2] = self._good_arg / 64.0
        # random but valid masks; the good pair always admitted; index 0 open for every head
        for m0, m1 in S.MASK_OFFSETS.values():
            size = m1 - m0
            mask = self._rng.random(size) < self.mask_density
            mask[0] = True
            obs[m0:m1] = mask
        op_name = S.OPS[self._good_op]
        obs[S.MASK_OFFSETS["op"][0] + self._good_op] = 1.0
        heads = S.OP_HEADS[op_name]
        if heads:
            h0, _ = S.MASK_OFFSETS[heads[0]]
            obs[h0 + self._good_arg] = 1.0
        self._last_obs = obs.copy()
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        options = {} if options is None else dict(options)
        restore = options.get("restore")
        if restore is None:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._t = 0
            self._episode_max_steps = self.horizon
            self._good_op = int(self._rng.integers(len(S.OPS)))
            heads = S.OP_HEADS[S.OPS[self._good_op]]
            limit = S.HEAD_SIZES[heads[0]] if heads else 1
            self._good_arg = int(self._rng.integers(min(limit, 64)))
            self._excursion = False
            self._restore_hash = None
            self._excursion_steps = 0
            self._excursion_budget = None
        else:
            if not isinstance(restore, FakeFrontierState):
                raise TypeError("options['restore'] must be a FakeFrontierState")
            budget = options.get("budget")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise ValueError("a restored reset requires a positive integer budget")
            if restore.step_index >= restore.max_steps:
                raise ValueError("frontier has no remaining episode step budget")
            self._t = restore.step_index
            self._episode_max_steps = restore.max_steps
            self._good_op = restore.good_op
            self._good_arg = restore.good_arg
            self._rng.bit_generator.state = copy.deepcopy(restore.rng_state)
            self._excursion = True
            self._restore_hash = restore.entity_hash
            self._excursion_steps = 0
            self._excursion_budget = min(budget, restore.max_steps - self._t)
        observation = (
            self._observe()
            if restore is None
            else np.asarray(restore.observation, dtype=np.float32).copy()
        )
        self._last_obs = observation.copy()
        return observation, {
            "excursion": self._excursion,
            "restore_hash": self._restore_hash,
        }

    def snapshot(self) -> FakeFrontierState:
        payload = json.dumps(
            [self._t, self._good_op, self._good_arg], separators=(",", ":")
        )
        return FakeFrontierState(
            step_index=self._t,
            max_steps=self._episode_max_steps,
            good_op=self._good_op,
            good_arg=self._good_arg,
            rng_state=copy.deepcopy(self._rng.bit_generator.state),
            entity_hash=hashlib.sha256(payload.encode()).hexdigest(),
            observation=self._last_obs.copy(),
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        op = int(action[S.HEADS.index("op")])
        reward = 0.0
        if op == self._good_op:
            reward = 0.1
            heads = S.OP_HEADS[S.OPS[op]]
            if not heads or int(action[S.HEADS.index(heads[0])]) == self._good_arg:
                reward = 1.0
        self._t += 1
        if self._excursion:
            self._excursion_steps += 1
        terminated = False
        truncated = self._t >= self._episode_max_steps or (
            self._excursion_budget is not None
            and self._excursion_steps >= self._excursion_budget
        )
        info = {"op": S.OPS[op], "status": "ok" if reward > 0 else "no_effect",
                "score_automated": 0.0, "score_player": 0.0, "deaths": 0,
                "excursion": self._excursion, "restore_hash": self._restore_hash,
                "regime": self.regime}
        return self._observe(), reward, terminated, truncated, info
