from __future__ import annotations

import json

import pytest

from fle.smarq.logging_ import (
    EPISODE_FIELDS,
    METRIC_FIELDS,
    STEP_FIELDS,
    JsonlLogger,
    episode_record,
    tail_status,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the repository's live-server autouse fixture for offline tests."""
    yield


def _step(index: int) -> dict:
    return {
        "tick": index * 60,
        "simulated_seconds_elapsed": float(index),
        "wall_seconds": 0.01,
        "game_seconds_per_wall_second": 100.0,
        "production_score": 1.0,
        "automated_production_score": 2.0,
        "delta_production_score": 0.0,
        "delta_automated_production_score": 1.0,
        "verb": "FAST_FORWARD",
        "heads": [-1] * 9,
        "failure_reason": "ok",
        "success": True,
        "requested_quantity": None,
        "executed_quantity": None,
        "q_value": 0.5,
        "td_error": 0.2,
        "per_priority": 1.2,
        "exploratory": False,
        "epsilon_per_head": {"verb": 0.1},
    }


def test_each_step_is_visible_before_episode_close(tmp_path):
    logger = JsonlLogger(tmp_path, fsync_every=1)
    for index in range(5):
        logger.log_step(0, _step(index))
    # Simulate inspecting a process killed in the middle of its episode.
    lines = (tmp_path / "steps_0.jsonl").read_text().splitlines()
    assert len(lines) == 5
    assert json.loads(lines[-1])["tick"] == 240
    logger.close()


def test_episode_rate_uses_simulated_not_wall_time():
    record = episode_record(
        episode_return=3,
        final_automated_score=120,
        max_automated_score=120,
        final_production_score=4,
        decisions=10,
        simulated_ticks=2 * 60 * 60,
        wall_seconds=0.001,
        failure_reason_histogram={"ok": 10},
        verb_histogram={"MINE": 10},
        end_reason="tick_budget",
    )
    assert record["automated_score_per_game_minute"] == 60.0
    assert set(EPISODE_FIELDS) <= record.keys()


def test_tail_status_and_documented_fields(tmp_path, capsys):
    with JsonlLogger(tmp_path) as logger:
        logger.log_step(2, _step(1))
        episode = episode_record(
            episode_return=1,
            final_automated_score=2,
            max_automated_score=2,
            final_production_score=0,
            decisions=1,
            simulated_ticks=60,
            wall_seconds=1,
            failure_reason_histogram={"ok": 1},
            verb_histogram={"FAST_FORWARD": 1},
            end_reason="decision_cap",
        )
        logger.log_episode(episode)
        metric = {field: 0 for field in METRIC_FIELDS}
        metric.update({"updates": 4, "replay_size": 1})
        logger.log_metrics(metric)
    status = tail_status(tmp_path)
    assert "decisions=1" in status
    assert "updates=4" in capsys.readouterr().out
    assert set(STEP_FIELDS) <= json.loads(
        (tmp_path / "steps_2.jsonl").read_text().splitlines()[0]
    ).keys()
