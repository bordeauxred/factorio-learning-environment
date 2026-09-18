"""Plots for an SM-ARQ run, from the JSONL the training loop writes.

    .venv/bin/python tests/benchmarks/smarq_plots.py --runs runs/full1-scratch runs/full2-demo --out docs/rl/results/sm_arq

Produces the figures the brief asks for:
  1. automated production score against simulated game time
  2. automated production score against environment transitions
  3. automated production score against wall-clock time
  4. general production score against automated production score
  5. failed-action and invalid-placement rate over time
  6. action-type distribution
  7. learner diagnostics (loss, mean max Q, epsilon)

Safe to run against a live run; partially written final lines are skipped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2a6f97", "#d4761a", "#4f8a5b", "#9b4f96"]


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
                continue
    return rows


def load_run(run_dir: Path) -> dict:
    steps: list[dict] = []
    for path in sorted(run_dir.glob("steps_*.jsonl")):
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            row["_worker_index"] = index
        steps.extend(rows)
    # One global ordering by tick is misleading across workers, so order by the
    # position within each worker and then interleave: the x axis is "decisions
    # this run has taken", which is what the brief asks for.
    steps.sort(key=lambda r: (r.get("_worker_index", 0), r.get("worker", 0)))
    return {
        "name": run_dir.name,
        "steps": steps,
        "episodes": read_jsonl(run_dir / "episodes.jsonl"),
        "metrics": read_jsonl(run_dir / "metrics.jsonl"),
    }


def cumulative_game_minutes(steps: list[dict]) -> np.ndarray:
    per_step = np.array([s.get("wall_seconds", 0.0) for s in steps])
    del per_step
    ticks = np.array([s.get("tick", 0) for s in steps], dtype=float)
    # Ticks restart per episode boundary only in the sense that the server keeps
    # counting; differences within a worker are the honest simulated durations.
    deltas = np.diff(ticks, prepend=ticks[0] if len(ticks) else 0.0)
    deltas[deltas < 0] = 0.0
    return np.cumsum(deltas) / 60.0 / 60.0


def plot_runs(runs: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    runs = [r for r in runs if r["steps"]]
    if not runs:
        return written

    def new_axes(title: str, xlabel: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        return fig, ax

    # 1-3: automated score against three different clocks.
    for key, xlabel, filename in (
        ("game", "simulated game minutes", "automated_vs_game_time.png"),
        ("steps", "semantic decisions", "automated_vs_transitions.png"),
        ("wall", "wall-clock minutes", "automated_vs_wall_time.png"),
    ):
        fig, ax = new_axes(
            "Automated production score", xlabel, "automated production score"
        )
        for color, run in zip(COLORS, runs):
            steps = run["steps"]
            auto = np.array([s.get("automated_production_score", 0.0) for s in steps])
            if key == "game":
                x = cumulative_game_minutes(steps)
            elif key == "steps":
                x = np.arange(len(steps))
            else:
                x = np.cumsum([s.get("wall_seconds", 0.0) for s in steps]) / 60.0
            ax.plot(x, auto, color=color, linewidth=1.2, label=run["name"])
            ax.plot(x, np.maximum.accumulate(auto), color=color, linewidth=0.8,
                    linestyle="--", alpha=0.7)
        ax.legend(frameon=False, fontsize=8)
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    # 4: general against automated.
    fig, ax = new_axes(
        "General production score against automated", "production score", "automated production score"
    )
    for color, run in zip(COLORS, runs):
        steps = run["steps"]
        ax.scatter(
            [s.get("production_score", 0.0) for s in steps],
            [s.get("automated_production_score", 0.0) for s in steps],
            s=5, alpha=0.4, color=color, label=run["name"], edgecolors="none",
        )
    lim = ax.get_xlim()
    ax.plot(lim, lim, color="#999999", linewidth=0.7, linestyle=":")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "production_vs_automated.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # 5: failure rates over time.
    fig, ax = new_axes("Failed actions, rolling", "semantic decisions", "fraction of decisions")
    window = 200
    for color, run in zip(COLORS, runs):
        steps = run["steps"]
        failed = np.array([0.0 if s.get("success") else 1.0 for s in steps])
        placement = np.array(
            [1.0 if (s.get("verb") == "PLACE" and not s.get("success")) else 0.0 for s in steps]
        )
        if len(failed) >= window:
            kernel = np.ones(window) / window
            ax.plot(np.convolve(failed, kernel, "valid"), color=color, linewidth=1.2,
                    label=f"{run['name']} any failure")
            ax.plot(np.convolve(placement, kernel, "valid"), color=color, linewidth=0.9,
                    linestyle="--", label=f"{run['name']} failed PLACE")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "failure_rate.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # 6: action distribution.
    fig, ax = new_axes("Action-type distribution", "", "share of decisions")
    verbs = sorted({s.get("verb") for run in runs for s in run["steps"] if s.get("verb")})
    width = 0.8 / max(1, len(runs))
    for index, (color, run) in enumerate(zip(COLORS, runs)):
        counts = Counter(s.get("verb") for s in run["steps"])
        total = max(1, sum(counts.values()))
        ax.bar(
            np.arange(len(verbs)) + index * width,
            [counts.get(v, 0) / total for v in verbs],
            width=width, color=color, label=run["name"],
        )
    ax.set_xticks(np.arange(len(verbs)) + 0.4 - width / 2)
    ax.set_xticklabels(verbs, rotation=40, ha="right", fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "action_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # 7: learner diagnostics.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=140)
    for color, run in zip(COLORS, runs):
        metrics = run["metrics"]
        if not metrics:
            continue
        updates = [m.get("updates", 0) for m in metrics]
        axes[0].plot(updates, [m.get("loss", 0.0) for m in metrics], color=color, linewidth=1.1,
                     label=run["name"])
        axes[1].plot(updates, [m.get("mean_max_q", 0.0) for m in metrics], color=color, linewidth=1.1)
        axes[2].plot(updates, [m.get("epsilon", 0.0) for m in metrics], color=color, linewidth=1.1)
    for axis, title in zip(axes, ("Huber loss", "mean max Q", "epsilon")):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("learner updates", fontsize=8)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "learner_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["runs/full1-scratch", "runs/full2-demo"])
    parser.add_argument("--out", default="docs/rl/results/sm_arq")
    args = parser.parse_args()
    runs = [load_run(Path(r)) for r in args.runs]
    for run in runs:
        print(f"{run['name']}: {len(run['steps'])} steps, {len(run['episodes'])} episodes")
    written = plot_runs(runs, Path(args.out))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
