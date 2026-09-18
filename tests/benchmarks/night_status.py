"""One-screen status of all overnight runs: processes, progress, ladder firsts.

Usage: python tests/benchmarks/night_status.py [--runs runs] [--since-min 45]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from tests.benchmarks.plot_first_curves import LADDER_RUNGS, ladder_from_steps


def processes() -> list[str]:
    out = subprocess.run(["pgrep", "-fl", "fle.rl"], capture_output=True, text=True, check=False).stdout
    return [line.split(None, 1)[1][:110] for line in out.strip().splitlines() if line.strip()]


def scan_run(run_dir: Path, since_s: float) -> dict:
    info = {"run": run_dir.name, "envs": [], "episodes": 0, "recent_episodes": 0, "aps": [], "firsts": {}}
    ep_index = 0
    for env_log in sorted(run_dir.glob("env_*.jsonl")):
        steps: list[dict] = []
        n_eps = 0
        last_mtime = env_log.stat().st_mtime
        with env_log.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "episode_end" in rec:
                    body = rec["episode_end"]
                    ladder = ladder_from_steps(steps)
                    aps = float(body.get("final_aps", 0.0))
                    info["aps"].append(aps)
                    for rung, hit in ladder.items():
                        if hit and rung not in info["firsts"]:
                            info["firsts"][rung] = (env_log.name, ep_index, aps)
                    steps = []
                    n_eps += 1
                    ep_index += 1
                else:
                    steps.append(rec)
        info["envs"].append((env_log.name, n_eps, round((time.time() - last_mtime) / 60, 1)))
        info["episodes"] += n_eps
    metrics = run_dir / "metrics.jsonl"
    if metrics.exists():
        last = None
        with metrics.open() as f:
            for line in f:
                if line.strip():
                    last = line
        if last:
            rec = json.loads(last)
            info["metrics"] = {k: rec.get(k) for k in ("env_steps", "epsilon", "td_loss", "mean_max_q", "mean_episode_reward", "acting_head") if k in rec}
            info["metrics_age_min"] = round((time.time() - metrics.stat().st_mtime) / 60, 1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--since-min", type=float, default=45)
    args = ap.parse_args()
    print("processes:")
    for p in processes():
        print("  ", p)
    for run_dir in sorted(args.runs.iterdir()):
        if not run_dir.is_dir() or not any(run_dir.glob("env_*.jsonl")):
            continue
        info = scan_run(run_dir, args.since_min * 60)
        aps = info["aps"]
        recent = aps[-20:]
        med = sorted(recent)[len(recent) // 2] if recent else None
        print(f"\n== {info['run']}: episodes {info['episodes']}, last-20 APS median {med}, max {max(aps) if aps else None}, positive {sum(a > 0 for a in aps)}/{len(aps)}")
        for name, n, age in info["envs"]:
            print(f"   {name}: {n} episodes, log age {age} min")
        if "metrics" in info:
            print(f"   metrics ({info['metrics_age_min']} min old): {info['metrics']}")
        firsts = info["firsts"]
        print("   ladder firsts:", ", ".join(f"{r}@ep{firsts[r][1]}" for r in LADDER_RUNGS if r in firsts) or "none")


if __name__ == "__main__":
    main()
