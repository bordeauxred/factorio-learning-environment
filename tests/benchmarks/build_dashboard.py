"""Build a single local HTML dashboard for the RL runs.

Usage: python tests/benchmarks/build_dashboard.py --out docs/rl/results/dashboard.html \
           --plots docs/rl/results/overnight_v1 [--runs runs] [--log docs/rl/results/overnight_v1_log.md]

Embeds the four plots as base64 PNGs, renders the summary and ladder tables,
a live table of the running arms (from night_status's scan), and the tail of
the half-hourly log. Static file; reopen or refresh after regenerating.
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import statistics
import time
from pathlib import Path

from tests.benchmarks.night_status import scan_run

# Human-readable names for run directories (the directories keep their launch names).
DISPLAY_NAMES = {
    "dqn-per": "dqn-per (plain DQN + PER, epsilon)",
    "dqn-per-ucbexplorer": "dqn-per-ucbexplorer (DQN + PER; epsilon port + rung-UCB explorer, 512 steps)",
    "qrdqn-per-ucbexplorer-frontier": "qrdqn-per-ucbexplorer-frontier (QR-DQN + PER; explorer/exploiter; frontier return)",
    "rainbow-optimisticper-n5": "rainbow-optimisticper-n5 (QR-51, noisy nets, PER with optimism K=3, return bonus, n-step 5)",
    "rainbow-optimisticper-n5-macro": "rainbow macro actions (Rainbow: QR-51, noisy nets, optimistic PER K=3, n-step 5; HARVEST and entity ops walk to their target; 3 servers)",
    "rainbow-optimisticper-n5-bare": "rainbow bare actions (same learner; no navigation inside actions, MOVE is the only way to move; 3 servers)",
    "ctrl-v1": "dqn-per-v1env (day-1 anchor, old environment)",
    "random-v1": "random policy",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
LINE = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"


def img_tag(path: Path) -> str:
    if not path.exists():
        return f"<p class='muted'>missing {html.escape(path.name)}</p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{data}' alt='{html.escape(path.stem)}'>"


def md_table_to_html(md: str) -> str:
    rows = [line for line in md.splitlines() if line.startswith("|")]
    if not rows:
        return "<p class='muted'>no table</p>"
    out = ["<table>"]
    for i, row in enumerate(rows):
        if re.match(r"^\|\s*-", row):
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{html.escape(c).replace('**', '')}</{tag}>" for c in cells) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def live_table(runs_dir: Path, active: list[str]) -> str:
    out = ["<table><tr><th>arm</th><th>episodes</th><th>steps</th><th>last-20 median APS</th><th>max</th><th>positive</th><th>deepest rung</th><th>log age</th></tr>"]
    order = ["furnace fuelled and fed", "plates extracted", "gears crafted", "drill crafted", "drill placed", "drill fuelled"]
    for name in active:
        run_dir = runs_dir / name
        if not run_dir.exists():
            continue
        info = scan_run(run_dir, 0)
        aps = info["aps"]
        recent = aps[-20:]
        med = statistics.median(recent) if recent else None
        deepest = "none"
        for rung in order:
            if rung in info["firsts"]:
                deepest = rung
        age = max((e[2] for e in info["envs"]), default=None)
        steps = info.get("metrics", {}).get("env_steps")
        stale = " class='warn'" if age is not None and age > 4 else ""
        out.append(
            f"<tr{stale}><td>{html.escape(DISPLAY_NAMES.get(name, name))}</td><td>{info['episodes']}</td><td>{steps}</td>"
            f"<td>{med}</td><td>{max(aps) if aps else ''}</td><td>{sum(a > 0 for a in aps)}/{len(aps)}</td>"
            f"<td>{deepest}</td><td>{age} min</td></tr>"
        )
    out.append("</table>")
    return "\n".join(out)


def sparkline_svg(values: list[float], color: str, width: int = 320, height: int = 60) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * (width - 4) + 2
        y = height - 2 - (v - lo) / span * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(pts)}'/>"
        f"<text x='2' y='12' font-size='10' fill='{INK2}'>{lo:.0f}</text>"
        f"<text x='2' y='{height - 4}' font-size='10' fill='{INK2}'></text>"
        f"<text x='{width - 40}' y='12' font-size='10' fill='{INK2}'>{hi:.0f}</text></svg>"
    )


def rolling(values: list[float], w: int = 10) -> list[float]:
    return [statistics.median(values[max(0, i - w + 1): i + 1]) for i in range(len(values))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plots", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--log", type=Path, default=Path("docs/rl/results/overnight_v1_log.md"))
    ap.add_argument("--history", type=Path, default=Path("docs/rl/results/overnight_v1_relocation_era"),
                    help="archived plots from before the 11:20 fresh start (relocation-era and day-1 runs)")
    ap.add_argument("--active", default="rainbow-optimisticper-n5-macro,rainbow-optimisticper-n5-bare")
    args = ap.parse_args()
    active = [a for a in args.active.split(",") if a]
    colors = {"dqn-per": BLUE, "dqn-per-ucbexplorer": ORANGE, "qrdqn-per-ucbexplorer-frontier": AQUA, "rainbow-optimisticper-n5": RED, "rainbow-optimisticper-n5-macro": RED, "rainbow-optimisticper-n5-bare": BLUE}

    sparks = []
    for name in active:
        run_dir = args.runs / name
        if not run_dir.exists():
            continue
        info = scan_run(run_dir, 0)
        med = rolling([float(a) for a in info["aps"]])
        sparks.append(f"<div class='spark'><div class='label'><span class='dot' style='background:{colors.get(name, RED)}'></span>{html.escape(DISPLAY_NAMES.get(name, name).split(' (')[0])} rolling-median APS, {len(med)} episodes</div>{sparkline_svg(med, colors.get(name, RED))}</div>")

    summary = (args.plots / "summary.md").read_text() if (args.plots / "summary.md").exists() else ""
    ladder = (args.plots / "ladder.md").read_text() if (args.plots / "ladder.md").exists() else ""
    history_summary = (args.history / "summary.md").read_text() if (args.history / "summary.md").exists() else ""
    history_ladder = (args.history / "ladder.md").read_text() if (args.history / "ladder.md").exists() else ""
    log_text = args.log.read_text() if args.log.exists() else ""
    log_tail = "\n".join(log_text.splitlines()[-60:])

    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>FLE RL runs</title>
<style>
body{{margin:0;padding:24px;background:{SURFACE};color:{INK};font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif;max-width:1400px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} .muted{{color:{INK2}}}
table{{border-collapse:collapse;margin:8px 0;font-size:13px}} th,td{{border-bottom:1px solid {LINE};padding:4px 10px;text-align:left}} th{{color:{INK2};font-weight:600}}
tr.warn td{{background:#fff3e6}}
img{{max-width:100%;border:1px solid {LINE};border-radius:6px;margin:6px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .spark{{display:inline-block;margin:6px 18px 6px 0;vertical-align:top}}
.label{{font-size:12px;color:{INK2};margin-bottom:2px}} .dot{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px;vertical-align:middle}}
pre{{background:#f4f3f0;padding:12px;border-radius:6px;font-size:12px;overflow-x:auto;white-space:pre-wrap}}
</style></head><body>
<h1>FLE deep RL: run dashboard</h1>
<p class='muted'>Generated {time.strftime('%Y-%m-%d %H:%M')}. Two Rainbow runs, same learner and seed, D18 environment: macro actions since 2026-09-17 12:22 (ports 27003 to 27005) and bare actions since 16:18 (ports 27000 to 27002), three servers each. Compare at equal step counts, not equal wall time. Automated production score (APS) per 256-step episode unless noted. Reward is the raw APS delta; nothing shaped.</p>
<h2>Running arms</h2>
{live_table(args.runs, active)}
<div>{''.join(sparks)}</div>
<h2>Curves</h2>
<div class='grid'>
<div>{img_tag(args.plots / 'aps_per_episode.png')}</div>
<div>{img_tag(args.plots / 'player_ps_per_episode.png')}</div>
</div>
{img_tag(args.plots / 'op_share.png')}
<div class='grid'>
<div><h2>Automation ladder (episodes reaching each rung)</h2>{md_table_to_html(ladder)}</div>
<div><h2>Summary (first half, second half, all)</h2>{md_table_to_html(summary)}</div>
</div>
<h2>Before the fresh start: 2026-09-15 to 2026-09-17 12:22 (archived, not comparable to the curves above)</h2>
<p class='muted'>Runs on the earlier environment versions: dqn-per-v1env is the day-1 anchor (v1 environment, 304 episodes);
the four v4 arms ran from 2026-09-17 02:00 to 11:20 across D9 to D15, including the relocation clamp from 05:50 that
substituted items, anchors and drill offsets (withdrawn by owner ruling, D16). Their drill milestones are the executor's, not the
agent's. Full account: docs/rl/results/overnight_v1.md sections "Night 2", "Environment versions under the v4 curves" and
"Correction (11:30)"; day 1 in docs/rl/results/first_curves.md; trace audits in docs/rl/research/15 to 18.</p>
<div class='grid'>
<div>{img_tag(args.history / 'aps_per_episode.png')}</div>
<div>{img_tag(args.history / 'player_ps_per_episode.png')}</div>
</div>
<div class='grid'>
<div><h2>Archived ladder</h2>{md_table_to_html(history_ladder)}</div>
<div><h2>Archived summary</h2>{md_table_to_html(history_summary)}</div>
</div>
<h2>Log, last 60 lines</h2>
<pre>{html.escape(log_tail)}</pre>
<p class='muted'>Source files: docs/rl/results/overnight_v1.md (insights), docs/rl/research/13-algorithms-for-open-play.md (algorithms), docs/rl/results/overnight_v1_log.md (timeline).</p>
</body></html>"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
