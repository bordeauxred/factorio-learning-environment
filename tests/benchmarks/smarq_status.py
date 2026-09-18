"""Summarize an SM-ARQ run directory: what the agent did and what it earned.

    .venv/bin/python tests/benchmarks/smarq_status.py runs/full1-scratch [...]

Everything here reads the JSONL the training loop already writes, so it is safe
to run against a live run.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially flushed final line
    return rows


def summarize(run_dir: Path) -> None:
    steps: list[dict] = []
    for path in sorted(run_dir.glob("steps_*.jsonl")):
        steps.extend(read_jsonl(path))
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    metrics = read_jsonl(run_dir / "metrics.jsonl")

    print(f"\n=== {run_dir.name} ===")
    if not steps:
        print("  no steps logged yet")
        return

    sim_seconds = sum(s.get("simulated_seconds_elapsed", 0.0) for s in steps)
    wall = sum(s.get("wall_seconds", 0.0) for s in steps)
    auto = [s.get("automated_production_score", 0.0) for s in steps]
    prod = [s.get("production_score", 0.0) for s in steps]
    gains = [s for s in steps if s.get("delta_automated_production_score", 0.0) > 0]

    print(f"  decisions           {len(steps)}")
    print(f"  episodes finished   {len(episodes)}")
    print(f"  simulated time      {sim_seconds/60:.1f} game minutes")
    print(f"  wall time in steps  {wall/60:.1f} minutes")
    print(f"  automated score     max {max(auto):.0f}, last {auto[-1]:.0f}")
    print(f"  production score    max {max(prod):.0f}, last {prod[-1]:.0f}")
    print(f"  steps with automated gain: {len(gains)} ({100*len(gains)/len(steps):.1f}%)")

    verbs = collections.Counter(s.get("verb") for s in steps)
    total = sum(verbs.values())
    print("  verbs: " + ", ".join(f"{v} {100*n/total:.0f}%" for v, n in verbs.most_common()))

    reasons = collections.Counter(s.get("failure_reason") for s in steps)
    print("  outcomes: " + ", ".join(f"{r} {100*n/total:.0f}%" for r, n in reasons.most_common()))

    placements = [s for s in steps if s.get("verb") == "PLACE"]
    if placements:
        ok = sum(1 for s in placements if s.get("success"))
        blocked = sum(1 for s in placements if s.get("failure_reason") == "blocked")
        print(
            f"  PLACE: {len(placements)} attempts, {100*ok/len(placements):.0f}% accepted, "
            f"{100*blocked/len(placements):.0f}% blocked"
        )

    if episodes:
        best = max(episodes, key=lambda e: e.get("final_automated_score", 0.0))
        per_min = [e.get("automated_score_per_game_minute", 0.0) for e in episodes]
        print(
            f"  best episode: automated {best.get('final_automated_score', 0):.0f} "
            f"in {best.get('decisions', 0)} decisions, end={best.get('end_reason')}"
        )
        print(f"  automated per game minute: max {max(per_min):.2f}, mean {sum(per_min)/len(per_min):.2f}")

    if metrics:
        last = metrics[-1]
        print(
            f"  learner: {last.get('updates', 0)} updates, loss {last.get('loss', 0):.4f}, "
            f"mean max Q {last.get('mean_max_q', 0):.3f}, epsilon {last.get('epsilon', 0):.3f}, "
            f"replay {last.get('replay_size', 0)}"
        )
        print(
            f"  throughput: {last.get('transitions_per_wall_second', 0):.2f} decisions/wall-s, "
            f"{last.get('updates_per_second', 0):.2f} updates/wall-s, "
            f"{last.get('simulated_seconds_per_wall_second', 0):.1f} game-s/wall-s"
        )


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [Path("runs/full1-scratch"), Path("runs/full2-demo")]
    for path in dirs:
        summarize(path)


if __name__ == "__main__":
    main()
