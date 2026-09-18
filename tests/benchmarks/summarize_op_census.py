"""Render operation-census JSONL logs as a markdown report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fle.rl.ops import OPS

STATUSES = ("ok", "no_effect", "no_support", "tool_rejected", "error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def read_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def cell(count: int, total: int) -> str:
    percentage = count * 100 / total if total else 0.0
    return f"{count} ({percentage:.1f}%)"


def count_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| op | n | ok | no_effect | no_support | tool_rejected | error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for op in OPS:
        rows = [record for record in records if record["op"] == op]
        counts = Counter(record["status"] for record in rows)
        columns = [op, str(len(rows))] + [cell(counts[status], len(rows)) for status in STATUSES]
        lines.append("| " + " | ".join(columns) + " |")
    return "\n".join(lines)


def timing_table(records: list[dict[str, Any]]) -> str:
    total_wall = sum(
        float(record.get("wall_tool_s", 0)) + float(record.get("wall_observe_s", 0))
        for record in records
    )
    lines = [
        "| op | median wall s | p90 wall s | median game ticks | wall share |",
        "|---|---:|---:|---:|---:|",
    ]
    for op in OPS:
        rows = [record for record in records if record["op"] == op]
        walls = [
            float(record.get("wall_tool_s", 0)) + float(record.get("wall_observe_s", 0))
            for record in rows
        ]
        ticks = [record.get("ticks_after", 0) - record.get("ticks_before", 0) for record in rows]
        op_wall = sum(walls)
        share = op_wall * 100 / total_wall if total_wall else 0.0
        median_wall = statistics.median(walls) if walls else 0.0
        median_ticks = statistics.median(ticks) if ticks else 0.0
        lines.append(
            f"| {op} | {median_wall:.4f} | {percentile(walls, 0.9):.4f} | "
            f"{median_ticks:g} | {share:.1f}% |"
        )
    return "\n".join(lines)


def reason_sections(records: list[dict[str, Any]]) -> list[str]:
    sections = ["### Failure reasons", ""]
    for op in OPS:
        failures = [record for record in records if record["op"] == op and record["status"] != "ok"]
        reasons = Counter(record.get("reason_class") or "unspecified" for record in failures)
        if not reasons:
            sections.extend([f"- {op}: none", ""])
            continue
        examples: dict[str, str] = {}
        for record in failures:
            reason = record.get("reason_class") or "unspecified"
            raw = record.get("raw_error") or reason
            examples.setdefault(reason, str(raw)[:200])
        rendered = "; ".join(
            f"{reason}={count} — `{examples[reason]}`" for reason, count in reasons.most_common()
        )
        sections.extend([f"- {op}: {rendered}", ""])

        other_raw = Counter(
            str(record.get("raw_error") or "")[:200]
            for record in failures
            if record.get("reason_class") == "other"
        )
        if other_raw:
            top = "; ".join(f"{count}× `{raw}`" for raw, count in other_raw.most_common(10))
            sections.extend([f"  Top `other` messages: {top}", ""])
    return sections


def harvest_headline(records: list[dict[str, Any]]) -> list[str]:
    harvests = [record for record in records if record["op"] == "HARVEST"]
    ok = sum(record["status"] == "ok" for record in harvests)
    acquired = sum(
        sum(max(0, int(delta)) for delta in record.get("items_delta", {}).values())
        for record in harvests
        if record["status"] == "ok"
    )
    by_episode: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[(record.get("run_id", ""), int(record["episode"]))].append(record)
    firsts = []
    for (run_id, episode), episode_rows in sorted(by_episode.items()):
        first = next(
            (
                row["step"]
                for row in sorted(episode_rows, key=lambda candidate: candidate["step"])
                if row["op"] == "HARVEST"
                and sum(max(0, int(value)) for value in row.get("items_delta", {}).values()) > 0
            ),
            None,
        )
        label = str(first) if first is not None else "never"
        firsts.append(f"{run_id[:8]}/episode {episode}: {label}")
    return [
        "### HARVEST headline",
        "",
        f"HARVEST ok: **{ok}**; items acquired: **{acquired}**.",
        "",
        "First acquired-item step: " + "; ".join(firsts),
    ]


def render(records: list[dict[str, Any]]) -> tuple[str, str]:
    if not records:
        raise ValueError("No census records found")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["regime"], record["fixture"])].append(record)

    report = [
        "# Operation census",
        "",
        "The `seeded` fixture exists only to measure entity operations and is never a training start.",
    ]
    printed_tables = []
    for (regime, fixture), rows in sorted(groups.items()):
        heading = f"## regime={regime}, fixture={fixture}"
        counts = count_table(rows)
        printed_tables.extend([heading, "", counts])
        report.extend(["", heading, "", counts, "", timing_table(rows), ""])
        approach_failed = sum(row["status"] == "approach_failed" for row in rows)
        if approach_failed:
            report.extend([f"Macro HARVEST approach_failed outcomes: **{approach_failed}**.", ""])
        report.extend(harvest_headline(rows))
        report.extend([""] + reason_sections(rows))
    return "\n".join(report).rstrip() + "\n", "\n".join(printed_tables).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report, tables = render(read_records(args.inputs))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(tables, end="")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
