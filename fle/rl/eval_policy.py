"""Evaluate a DQN checkpoint greedily while retaining environment JSONL logs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from fle.rl import schema as S
from fle.rl.dqn import (
    RUNG_NAMES,
    EpisodeMilestones,
    action_details,
    greedy_policy_actions,
    policy_from_checkpoint,
    reset_network_noise,
)
from fle.rl.fake_env import FakeMacroEnv


def evaluate(args: argparse.Namespace) -> list[dict]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_num_threads(args.torch_threads)
    network, checkpoint = policy_from_checkpoint(
        args.checkpoint, device, noisy_eval=args.noisy_eval
    )
    rng = np.random.default_rng(args.seed)

    if args.fake:
        env = FakeMacroEnv(seed=args.seed, regime=args.action_regime)
        log_path = None
    else:
        from fle.rl.env import FleMacroEnv

        log_path = (
            Path(args.log_path)
            if args.log_path
            else Path(args.checkpoint).resolve().parent / f"eval_env_{args.port}.jsonl"
        )
        env = FleMacroEnv(
            port=args.port,
            speed=args.speed,
            seed=args.seed,
            log_path=log_path,
            regime=args.action_regime,
        )

    results = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            done = False
            final_info = {}
            steps = 0
            milestones = EpisodeMilestones()
            while not done:
                old_obs = obs
                if rng.random() < args.eps:
                    action = S.random_valid_action(old_obs, rng)
                else:
                    if args.noisy_eval:
                        reset_network_noise(network)
                    with torch.no_grad():
                        tensor = torch.as_tensor(
                            old_obs[None], dtype=torch.float32, device=device
                        )
                        action = greedy_policy_actions(network, tensor)[0].cpu().numpy()
                obs, reward, terminated, truncated, final_info = env.step(action)
                milestones.observe(
                    action_details(old_obs, action),
                    str(final_info.get("status", "")),
                    steps,
                )
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated
            record = {
                "episode": episode,
                "reward": total_reward,
                "score_automated": float(final_info.get("score_automated", 0.0)),
                "score_player": float(final_info.get("score_player", 0.0)),
                "steps": steps,
                "regime": args.action_regime,
                "deepest_rung": (
                    RUNG_NAMES[milestones.deepest]
                    if milestones.deepest >= 0
                    else None
                ),
                "rungs_reached": [
                    RUNG_NAMES[index]
                    for index, reached in enumerate(milestones.reached)
                    if reached
                ],
            }
            results.append(record)
            print(json.dumps(record), flush=True)
    finally:
        env.close()

    rewards = np.asarray([row["reward"] for row in results], dtype=np.float64)
    automated_scores = np.asarray(
        [row["score_automated"] for row in results], dtype=np.float64
    )
    reward_q25, reward_q75 = np.quantile(rewards, [0.25, 0.75])
    score_q25, score_q75 = np.quantile(automated_scores, [0.25, 0.75])
    summary = {
        "algo": checkpoint.get("algo", "dqn"),
        "checkpoint_env_steps": int(checkpoint["env_steps"]),
        "regime": args.action_regime,
        "episodes": args.episodes,
        "eps": args.eps,
        "noisy_eval": args.noisy_eval,
        "mean_reward": float(rewards.mean()),
        "median_reward": float(np.median(rewards)),
        "reward_iqr": float(reward_q75 - reward_q25),
        "reward_q25": float(reward_q25),
        "reward_q75": float(reward_q75),
        "mean_score_automated": float(automated_scores.mean()),
        "median_score_automated": float(np.median(automated_scores)),
        "score_automated_iqr": float(score_q75 - score_q25),
        "score_automated_q25": float(score_q25),
        "score_automated_q75": float(score_q75),
        "ladder_attainment": {
            rung: {
                "episodes": sum(rung in row["rungs_reached"] for row in results),
                "rate": sum(rung in row["rungs_reached"] for row in results)
                / args.episodes,
            }
            for rung in RUNG_NAMES
        },
        "log_path": str(log_path) if log_path is not None else None,
    }
    print("evaluation " + json.dumps(summary, sort_keys=True), flush=True)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=27000)
    parser.add_argument("--action-regime", choices=("macro", "bare"), default="macro")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--noisy-eval", action="store_true")
    parser.add_argument("--log-path")
    parser.add_argument("--speed", type=float, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0 <= args.eps <= 1:
        parser.error("--eps must be between 0 and 1")
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
