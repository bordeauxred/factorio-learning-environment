"""Plot first learning curves from FleMacroEnv episode logs.

Reads one or more env JSONL logs (one per run label), each containing per-step
lines and one ``{"episode_end": ...}`` line per episode, and writes:

  <out>/aps_per_episode.png        final automated PS per episode, dots + rolling median
  <out>/player_ps_per_episode.png  player PS per episode (diagnostic)
  <out>/op_share.png               small multiples: share of each op per block of episodes
  <out>/invalid_rate.png           invalid-combination rate per episode
  <out>/summary.md                 medians and quartiles, first half vs second half

Usage:
  python tests/benchmarks/plot_first_curves.py --out docs/rl/results/first_curves \
      ppo=runs/ppo-v0/env_27000.jsonl,runs/ppo-v0/env_27001.jsonl \
      dqn=runs/dqn-v0/env_27002.jsonl random=runs/random-v0/env_27003.jsonl

Field names are read defensively: any of final_aps / score_automated /
aps_final for the automated score, player_ps / score_player for the player
score, op_hist / op_histogram for the op histogram.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fle.rl.schema import OPS

SERIES = {"ppo": "#2a78d6", "dqn": "#eb6834", "random": "#7a7973", "dqn-per": "#2a78d6", "rainbow-optimisticper-n5": "#e34948", "dqn-per-ucbexplorer": "#eb6834", "qrdqn-per-ucbexplorer-frontier": "#1baf7a", "dqn-per-v1env": "#b5b3ad", "rainbow-macro": "#e34948", "rainbow-bare": "#2a78d6"}
FALLBACK = ["#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_episodes(paths: list[Path]) -> list[dict]:
    episodes = []
    for path in paths:
        step_ops: list[str] = []
        step_status: list[str] = []
        step_records: list[dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "episode_end" in rec or rec.get("kind") == "episode_end":
                    body = rec.get("episode_end") if isinstance(rec.get("episode_end"), dict) else rec
                    op_hist = _first(body, "op_hist", "op_histogram")
                    if not op_hist and step_ops:
                        op_hist = {op: step_ops.count(op) for op in set(step_ops)}
                    status_hist = _first(body, "status_hist", "status_histogram")
                    if not status_hist and step_status:
                        status_hist = {s: step_status.count(s) for s in set(step_status)}
                    non_wait = sum(v for k, v in (op_hist or {}).items() if k != "WAIT") or 1
                    invalid = sum(v for k, v in (status_hist or {}).items() if k in ("tool_rejected", "no_effect", "approach_failed"))
                    episodes.append({
                        "ladder": ladder_from_steps(step_records),
                        "source": path.name,
                        "aps": float(_first(body, "final_aps", "score_automated", "aps_final", "final_score_automated", default=0.0)),
                        "max_aps": float(_first(body, "max_aps", "max_score_automated", default=0.0)),
                        "player_ps": float(_first(body, "player_ps", "score_player", "final_score_player", default=0.0)),
                        "steps": int(_first(body, "steps", "n_steps", default=len(step_ops))),
                        "op_hist": op_hist or {},
                        "invalid_rate": invalid / non_wait,
                        "wall_end": _first(body, "wall", "wall_s", "time"),
                    })
                    step_ops, step_status, step_records = [], [], []
                else:
                    op = rec.get("op")
                    if op:
                        step_ops.append(op)
                        step_status.append(rec.get("status", ""))
                        step_records.append(rec)
    # interleave sources by index so multi-env runs share one episode axis
    by_source: dict[str, list[dict]] = {}
    for ep in episodes:
        by_source.setdefault(ep["source"], []).append(ep)
    merged: list[dict] = []
    longest = max((len(v) for v in by_source.values()), default=0)
    for i in range(longest):
        for src in sorted(by_source):
            if i < len(by_source[src]):
                merged.append(by_source[src][i])
    return merged


LADDER_RUNGS = (
    "furnace placed", "furnace fuelled", "furnace fed", "furnace fuelled and fed",
    "plates extracted", "gears crafted", "drill crafted", "drill placed", "drill fuelled",
    "chest placed", "inserter placed", "belt placed", "APS > 0",
)
FUELS = {"coal", "wood"}
SMELTABLE = {"iron-ore", "copper-ore", "stone"}


def ladder_from_steps(steps: list[dict]) -> dict[str, bool]:
    """Which rungs of the automation ladder an episode reached, from ok steps only."""
    placed, crafted, inserted, extracted = set(), set(), set(), set()
    for r in steps:
        if r.get("status") != "ok":
            continue
        a = r.get("args") or {}
        op = r.get("op")
        if op == "PLACE":
            placed.add(a.get("item") or a.get("prototype"))
        elif op == "CRAFT":
            crafted.add(a.get("recipe"))
        elif op == "INSERT":
            inserted.add((a.get("item"), (a.get("anchor") or {}).get("name")))
        elif op == "EXTRACT":
            extracted.add(a.get("item"))
    fuel_f = any(i in FUELS and n == "stone-furnace" for i, n in inserted)
    feed_f = any(i in SMELTABLE and n == "stone-furnace" for i, n in inserted)
    return {
        "furnace placed": "stone-furnace" in placed,
        "furnace fuelled": fuel_f,
        "furnace fed": feed_f,
        "furnace fuelled and fed": fuel_f and feed_f,
        "plates extracted": bool(extracted & {"iron-plate", "copper-plate", "stone-brick"}),
        "gears crafted": "iron-gear-wheel" in crafted,
        "drill crafted": "burner-mining-drill" in crafted,
        "drill placed": "burner-mining-drill" in placed,
        "drill fuelled": any(i in FUELS and n == "burner-mining-drill" for i, n in inserted),
        "chest placed": bool(placed & {"wooden-chest", "iron-chest"}),
        "inserter placed": "burner-inserter" in placed,
        "belt placed": "transport-belt" in placed,
        "APS > 0": False,  # filled from the episode record below
    }


def write_ladder(runs: dict[str, list[dict]], out: Path):
    lines = ["# Automation ladder: episodes reaching each rung", "", "| rung | " + " | ".join(runs) + " |", "|---|" + "---:|" * len(runs)]
    for rung in LADDER_RUNGS:
        cells = []
        for eps in runs.values():
            hit = sum((e["aps"] > 0) if rung == "APS > 0" else e["ladder"].get(rung, False) for e in eps)
            cells.append(f"{hit}/{len(eps)}")
        lines.append(f"| {rung} | " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def rolling_median(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(statistics.median(values[lo:i + 1]))
    return out


def style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel("episode", color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


BOUNDARIES: dict[str, list[tuple[int, str]]] = {}  # label -> [(episode_index, text)]


def plot_metric(runs: dict[str, list[dict]], key: str, title: str, ylabel: str, out: Path, window: int):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    for i, (label, eps) in enumerate(runs.items()):
        color = SERIES.get(label, FALLBACK[i % len(FALLBACK)])
        ys = [e[key] for e in eps]
        xs = list(range(1, len(ys) + 1))
        ax.scatter(xs, ys, s=14, color=color, alpha=0.35, linewidths=0)
        if len(ys) >= 2:
            ax.plot(xs, rolling_median(ys, window), color=color, linewidth=2, label=f"{label} (rolling median, w={window})")
    for label, marks in BOUNDARIES.items():
        color = SERIES.get(label, "#7a7973")
        for ep_index, text in marks:
            ax.axvline(ep_index, color=color, linestyle=":", linewidth=1, alpha=0.7)
            ax.text(ep_index, ax.get_ylim()[1] * 0.98, text, rotation=90, va="top", ha="right", fontsize=7, color=color, alpha=0.9)
    style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_op_share(runs: dict[str, list[dict]], out: Path, block: int):
    cols = 4
    rows = (len(OPS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.4 * rows), facecolor=SURFACE, sharex=False)
    for k, op in enumerate(OPS):
        ax = axes[k // cols][k % cols]
        for i, (label, eps) in enumerate(runs.items()):
            color = SERIES.get(label, FALLBACK[i % len(FALLBACK)])
            shares, xs = [], []
            for b in range(0, len(eps), block):
                chunk = eps[b:b + block]
                total = sum(sum(e["op_hist"].values()) for e in chunk) or 1
                shares.append(sum(e["op_hist"].get(op, 0) for e in chunk) / total)
                xs.append(b + len(chunk) / 2)
            ax.plot(xs, shares, color=color, linewidth=2, label=label)
        style(ax, op, "share")
        ax.set_ylim(0, 1)
    for k in range(len(OPS), rows * cols):
        axes[k // cols][k % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.suptitle(f"Share of each op per block of {block} episodes", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def quartiles(values: list[float]) -> str:
    if not values:
        return "n=0"
    q = statistics.quantiles(values, n=4) if len(values) >= 4 else [min(values), statistics.median(values), max(values)]
    return f"n={len(values)} median {statistics.median(values):.1f} [q1 {q[0]:.1f}, q3 {q[-1]:.1f}] max {max(values):.1f}"


def write_summary(runs: dict[str, list[dict]], out: Path):
    lines = ["# First curves: summary", "", "| run | half | final APS | player PS | invalid rate | episodes with APS > 0 |", "|---|---|---|---|---:|---:|"]
    for label, eps in runs.items():
        half = len(eps) // 2
        for name, part in (("first", eps[:half]), ("second", eps[half:]), ("all", eps)):
            if not part:
                continue
            inv = statistics.median(e["invalid_rate"] for e in part)
            pos = sum(e["aps"] > 0 for e in part)
            lines.append(f"| {label} | {name} | {quartiles([e['aps'] for e in part])} | {quartiles([e['player_ps'] for e in part])} | {inv:.2f} | {pos}/{len(part)} |")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path[,path...]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--boundary", action="append", default=[], help="label=episode_index=text, repeatable; env-version boundary markers")
    args = ap.parse_args()
    for spec in args.boundary:
        label, idx, text = spec.split("=", 2)
        BOUNDARIES.setdefault(label, []).append((int(idx), text))
    args.out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[dict]] = {}
    for spec in args.runs:
        label, _, paths = spec.partition("=")
        runs[label] = load_episodes([Path(p) for p in paths.split(",") if p])
        print(f"{label}: {len(runs[label])} episodes")
    plot_metric(runs, "aps", "Final automated production score per episode", "automated PS", args.out / "aps_per_episode.png", args.window)
    plot_metric(runs, "player_ps", "Player production score per episode (diagnostic, includes hand work)", "player PS", args.out / "player_ps_per_episode.png", args.window)
    plot_metric(runs, "invalid_rate", "Invalid-combination rate per episode (rejected or no-effect over non-WAIT steps)", "rate", args.out / "invalid_rate.png", args.window)
    plot_op_share(runs, args.out / "op_share.png", args.block)
    write_summary(runs, args.out / "summary.md")
    write_ladder(runs, args.out / "ladder.md")


if __name__ == "__main__":
    main()
