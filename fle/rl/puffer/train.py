"""Train the FLE macro policy with PufferLib 3.0 PPO."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.pufferl
import pufferlib.pytorch
import pufferlib.spaces
import torch

from fle.rl import schema as S
from fle.rl.fake_env import FakeMacroEnv
from fle.rl.puffer.policy import MaskedMultiHeadPolicy


class EpisodeStatsWrapper(gym.Wrapper):
    """Expose completed-episode diagnostics through ordinary step info."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_reward = 0.0

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self.episode_reward
            info["episode_final_aps"] = float(info.get("score_automated", 0.0))
            info["episode_player_ps"] = float(info.get("score_player", 0.0))
        # PufferLib 3.0 records `terminated` but drops `truncated` from its
        # advantage computation. Treat either boundary as terminal for PPO.
        return observation, reward, terminated or truncated, False, info


class ThreadVectorEnv:
    """Minimal PufferLib vector API using one thread per Gymnasium environment."""

    def __init__(self, creators: list[Callable[[], gym.Env]], seed: int):
        self.envs = [
            pufferlib.emulation.GymnasiumPufferEnv(env_creator=creator, seed=seed + index)
            for index, creator in enumerate(creators)
        ]
        driver = self.envs[0]
        self.single_observation_space = driver.single_observation_space
        self.single_action_space = driver.single_action_space
        self.num_agents = len(self.envs)
        self.agents_per_batch = self.num_agents
        self.action_space = pufferlib.spaces.joint_space(
            self.single_action_space, self.num_agents
        )
        self.observation_space = pufferlib.spaces.joint_space(
            self.single_observation_space, self.num_agents
        )
        self._executor = ThreadPoolExecutor(max_workers=self.num_agents)
        self._observations = np.zeros(
            (self.num_agents, *self.single_observation_space.shape),
            dtype=self.single_observation_space.dtype,
        )
        self._rewards = np.zeros(self.num_agents, dtype=np.float32)
        self._terminals = np.zeros(self.num_agents, dtype=bool)
        self._truncations = np.zeros(self.num_agents, dtype=bool)
        self._masks = np.ones(self.num_agents, dtype=bool)
        self._env_ids = np.arange(self.num_agents)
        self._infos: list[dict] = []

    def _copy(self, index: int) -> None:
        env = self.envs[index]
        self._observations[index] = env.observations[0]
        self._rewards[index] = env.rewards[0]
        self._terminals[index] = env.terminals[0]
        self._truncations[index] = env.truncations[0]

    def async_reset(self, seed: int | None = None) -> None:
        seeds = [None if seed is None else seed + index for index in range(self.num_agents)]
        results = list(self._executor.map(lambda pair: pair[0].reset(seed=pair[1]), zip(self.envs, seeds)))
        self._infos = [info for _, info in results if info]
        for index in range(self.num_agents):
            self._copy(index)

    @staticmethod
    def _advance(pair):
        env, action = pair
        if env.done:
            return env.reset()
        return env.step(action)

    def send(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        results = list(self._executor.map(self._advance, zip(self.envs, actions)))
        self._infos = []
        for result in results:
            info = result[-1]
            if info:
                self._infos.append(info)
        for index in range(self.num_agents):
            self._copy(index)

    def recv(self):
        return (
            self._observations,
            self._rewards,
            self._terminals,
            self._truncations,
            self._infos,
            self._env_ids,
            self._masks,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()


class LocalLogger:
    def __init__(self, run_name: str):
        self.run_id = run_name

    def log(self, logs, step) -> None:
        del logs, step

    def close(self, model_path) -> None:
        del model_path


class FlePuffeRL(pufferlib.pufferl.PuffeRL):
    """PufferLib PPO with its interactive terminal dashboard disabled."""

    def print_dashboard(self, clear: bool = False) -> None:
        del clear


def _parse_ports(value: str) -> list[int]:
    if not value:
        return []
    try:
        ports = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("ports must be comma-separated integers") from error
    if not ports:
        raise argparse.ArgumentTypeError("at least one port is required")
    return ports


def _fake_creator(seed: int) -> Callable[[], gym.Env]:
    return lambda: EpisodeStatsWrapper(FakeMacroEnv(seed=seed))


def _live_creator(port: int, log_dir: Path | None = None) -> Callable[[], gym.Env]:
    def create() -> gym.Env:
        from fle.rl.env import FleMacroEnv

        log_path = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"env_{port}.jsonl"
        return EpisodeStatsWrapper(FleMacroEnv(port=port, speed=40, log_path=log_path))

    return create


def make_vector_env(args) -> ThreadVectorEnv:
    if args.fake:
        creators = [_fake_creator(args.seed + index) for index in range(args.num_envs)]
    else:
        if not args.ports:
            raise ValueError("--ports is required unless --fake is used")
        log_dir = Path(args.out) if args.out else None
        creators = [_live_creator(port, log_dir) for port in args.ports]
    return ThreadVectorEnv(creators, args.seed)


def make_config(args, num_envs: int, output_dir: Path) -> dict:
    batch_size = args.rollout_steps * num_envs
    if batch_size % args.minibatches:
        raise ValueError("rollout batch must be divisible by --minibatches")
    return {
        "env": "fle-macro",
        "seed": args.seed,
        "torch_deterministic": True,
        "cpu_offload": False,
        "device": args.device,
        "optimizer": "adam",
        "anneal_lr": False,
        "precision": "float32",
        "total_timesteps": args.total_steps,
        "learning_rate": args.learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "update_epochs": args.update_epochs,
        "clip_coef": 0.2,
        "vf_coef": 0.5,
        "vf_clip_coef": 0.2,
        "max_grad_norm": 0.5,
        "ent_coef": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "data_dir": str(output_dir / "checkpoints"),
        "checkpoint_interval": 10,
        "batch_size": batch_size,
        "minibatch_size": batch_size // args.minibatches,
        "max_minibatch_size": batch_size // args.minibatches,
        # Feedforward policy: the horizon only shapes PufferLib's buffer, and it
        # must divide the minibatch (with 2 envs the minibatch is 64 < 128).
        "bptt_horizon": min(args.rollout_steps, batch_size // args.minibatches),
        "compile": False,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "prio_alpha": 0.0,
        "prio_beta0": 0.0,
        "use_rnn": False,
    }


def random_baseline(episodes: int, seed: int) -> float:
    env = FakeMacroEnv(seed=seed)
    rng = np.random.default_rng(seed)
    returns = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total = 0.0
        while True:
            action = S.random_valid_action(observation, rng)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def evaluate_fake(policy: MaskedMultiHeadPolicy, episodes: int, seed: int) -> float:
    env = FakeMacroEnv(seed=seed)
    returns = []
    policy.eval()
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total = 0.0
        while True:
            with torch.no_grad():
                tensor = torch.as_tensor(observation).unsqueeze(0)
                logits, _ = policy(tensor)
                action = np.asarray([head.argmax(dim=-1).item() for head in logits])
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    policy.train()
    return float(np.mean(returns))


def assert_masks_respected(
    policy: MaskedMultiHeadPolicy, steps: int, num_envs: int, seed: int
) -> None:
    envs = [FakeMacroEnv(seed=seed + index) for index in range(num_envs)]
    observations = np.stack([env.reset()[0] for env in envs])
    device = next(policy.parameters()).device
    for step in range(steps):
        with torch.no_grad():
            logits, _ = policy(torch.as_tensor(observations, device=device))
            actions, _, _ = pufferlib.pytorch.sample_logits(logits)
        actions_np = actions.cpu().numpy()
        for env_index, (observation, action) in enumerate(zip(observations, actions_np)):
            for head_index, head in enumerate(S.HEADS):
                start, _ = S.MASK_OFFSETS[head]
                assert observation[start + action[head_index]] > 0.5, (
                    f"masked action sampled at step={step}, env={env_index}, head={head}"
                )
            next_observation, _, terminated, truncated, _ = envs[env_index].step(action)
            if terminated or truncated:
                next_observation, _ = envs[env_index].reset()
            observations[env_index] = next_observation
    for env in envs:
        env.close()


def _update_metrics(learner, stats, start_time: float) -> dict:
    observations = learner.observations.reshape(-1, S.OBS_SIZE)
    with torch.no_grad():
        logits, _ = learner.uncompiled_policy(observations)
    entropies = learner.uncompiled_policy.entropy_by_head(logits)
    actions = learner.actions.reshape(-1, len(S.HEADS)).cpu().numpy()
    op_histogram = Counter(S.OPS[int(index)] for index in actions[:, 0])

    ops = list(stats.get("op", []))
    statuses = list(stats.get("status", []))
    eligible = [(op, status) for op, status in zip(ops, statuses) if op != "WAIT"]
    invalid = sum(status in {"tool_rejected", "no_effect"} for _, status in eligible)
    episode_rewards = [float(value) for value in stats.get("episode_reward", [])]
    losses = dict(learner.losses)
    total_loss = (
        losses.get("policy_loss", 0.0)
        + learner.config["vf_coef"] * losses.get("value_loss", 0.0)
        - learner.config["ent_coef"] * losses.get("entropy", 0.0)
    )
    return {
        "update": learner.epoch,
        "env_steps": learner.global_step,
        "wall": time.time() - start_time,
        "losses": losses,
        "total_loss": total_loss,
        "policy_loss": losses.get("policy_loss"),
        "value_loss": losses.get("value_loss"),
        "entropy": losses.get("entropy"),
        "entropy_per_head": entropies,
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else None,
        "final_aps": [float(value) for value in stats.get("episode_final_aps", [])],
        "player_ps": [float(value) for value in stats.get("episode_player_ps", [])],
        "op_histogram": dict(op_histogram),
        "invalid_combination_rate": invalid / len(eligible) if eligible else 0.0,
    }


def train(args) -> MaskedMultiHeadPolicy:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.out or Path("runs") / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    vector_env = make_vector_env(args)
    policy = MaskedMultiHeadPolicy(vector_env).to(args.device)
    config = make_config(args, vector_env.num_agents, output_dir)
    learner = FlePuffeRL(
        config, vector_env, policy, logger=LocalLogger(args.run_name)
    )
    start_time = time.time()
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            while learner.global_step < args.total_steps:
                stats = learner.evaluate()
                stats = {key: list(values) for key, values in stats.items()}
                learner.last_log_time = time.time() - 1.0
                learner.train()
                metrics = _update_metrics(learner, stats, start_time)
                metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
                metrics_file.flush()
    finally:
        learner.close()

    if args.fake and args.mask_check_steps:
        assert_masks_respected(policy, args.mask_check_steps, args.num_envs, args.seed + 20_000)
        print(f"MASK_CHECK steps={args.mask_check_steps} envs={args.num_envs} violations=0")
    if args.fake and args.eval_episodes:
        baseline = random_baseline(args.eval_episodes, args.seed + 30_000)
        trained = evaluate_fake(policy.cpu(), args.eval_episodes, args.seed + 30_000)
        print(
            f"ACCEPTANCE episodes={args.eval_episodes} random_baseline={baseline:.6f} "
            f"trained_mean_reward={trained:.6f}"
        )
    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=_parse_ports, default=[])
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--run-name", default="ppo-v0")
    parser.add_argument("--out")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mask-check-steps", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.total_steps <= 0 or args.num_envs <= 0:
        raise SystemExit("--total-steps and --num-envs must be positive")
    train(args)


if __name__ == "__main__":
    main()
