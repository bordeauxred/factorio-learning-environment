from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.benchmarks import vm_throughput_sweep as sweep


STRING_FIELDS = {"machine", "regime", "status", "error"}
INTEGER_FIELDS = {
    "vcpus",
    "servers",
    "vcpus_used",
    "steps_total",
    "episodes_completed",
    "deaths",
    "log_bytes_total",
}
NULLABLE_FLOAT_FIELDS = {"marginal_steps_per_s_per_vcpu"}


class _OfflineInstance:
    """Keep the repository's global live-server autouse fixture offline here."""

    def reset(self, **kwargs) -> None:
        del kwargs


@pytest.fixture(scope="session")
def instance() -> _OfflineInstance:
    return _OfflineInstance()


def _args(out: Path, *extra: str):
    return sweep.parse_args(
        [
            "--servers",
            "1",
            "--seconds",
            "0.1",
            "--warmup",
            "0",
            "--out",
            str(out),
            "--dry-run",
            "--no-learner-probe",
            *extra,
        ]
    )


def _raising_worker(*args) -> None:
    del args
    raise RuntimeError("injected worker failure")


class TeardownSpy:
    def __init__(self) -> None:
        self.teardowns = 0

    def start(self, servers: int, scenario: str) -> None:
        del servers, scenario

    def teardown(self) -> None:
        self.teardowns += 1


def test_dry_run_writes_well_formed_records(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    result = sweep.main(
        [
            "--servers",
            "2,4",
            "--seconds",
            "2",
            "--warmup",
            "0",
            "--out",
            str(out),
            "--dry-run",
            "--no-learner-probe",
        ]
    )
    assert result == 0
    records = [
        json.loads(line) for line in (out / "sweep.jsonl").read_text().splitlines()
    ]
    assert [record["servers"] for record in records] == [2, 4]
    for record in records:
        assert set(sweep.POINT_FIELDS) <= record.keys()
        assert record["status"] == "ok"
        for field in sweep.POINT_FIELDS:
            value = record[field]
            if field in STRING_FIELDS:
                assert isinstance(value, str), field
            elif field in INTEGER_FIELDS:
                assert isinstance(value, int) and not isinstance(value, bool), field
            elif field in NULLABLE_FLOAT_FIELDS and value is None:
                pass
            else:
                assert isinstance(value, float) and not isinstance(value, bool), field
        assert record["decisions_per_episode"] == pytest.approx(256.0)


def test_cost_math() -> None:
    usd_per_1k_steps, steps_per_usd = sweep.cost_metrics(10.0, 0.75)
    assert usd_per_1k_steps == pytest.approx(0.75 / 36.0, abs=1e-9)
    assert steps_per_usd == pytest.approx(48_000.0)
    price_per_vcpu_hour, used_cost = sweep.used_resource_cost(10.0, 0.75, 90, 10)
    assert price_per_vcpu_hour == pytest.approx(0.75 / 90)
    assert used_cost == pytest.approx(10 * (0.75 / 90) / 36.0)


def test_worker_env_uses_production_episode_defaults(tmp_path: Path) -> None:
    calls = []

    class SpyEnv:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    sweep._make_worker_env(
        dry_run=False,
        port=27000,
        speed=40,
        seed=1,
        regime="macro",
        max_steps=None,
        log_path=tmp_path / "env_27000.jsonl",
        live_env_class=SpyEnv,
    )
    assert "max_steps" not in calls[0]
    assert "max_ticks" not in calls[0]

    sweep._make_worker_env(
        dry_run=False,
        port=27000,
        speed=40,
        seed=1,
        regime="macro",
        max_steps=17,
        log_path=None,
        live_env_class=SpyEnv,
    )
    assert calls[1]["max_steps"] == 17
    assert "max_ticks" not in calls[1]


def test_marginal_and_knee_arithmetic(tmp_path: Path) -> None:
    args = sweep.parse_args(
        [
            "--servers",
            "2,4,6,8",
            "--seconds",
            "1",
            "--out",
            str(tmp_path),
            "--dry-run",
        ]
    )
    records = []
    for servers, steps_per_s in ((2, 20.0), (4, 36.0), (6, 40.0), (8, 42.0)):
        record = sweep._base_record(args, servers)
        whole_cost, steps_per_usd = sweep.cost_metrics(steps_per_s, args.price_per_hour)
        _, used_cost = sweep.used_resource_cost(
            steps_per_s, args.price_per_hour, args.vcpus, servers
        )
        record.update(
            status="ok",
            error="",
            steps_per_s=steps_per_s,
            steps_per_s_per_server=steps_per_s / servers,
            usd_per_1k_steps=whole_cost,
            steps_per_usd=steps_per_usd,
            usd_per_1k_steps_used=used_cost,
        )
        records.append(record)

    annotated = sweep.with_marginal_metrics(records)
    assert [record["marginal_steps_per_s_per_vcpu"] for record in annotated] == [
        None,
        8.0,
        2.0,
        1.0,
    ]
    assert sweep.knee_server_count(records) == 4

    fixture = tmp_path / "sweep.jsonl"
    fixture.write_text("".join(json.dumps(record) + "\n" for record in records))
    summary = sweep.render_summary(fixture)
    assert "Aggregate throughput is maximal at N=8" in summary
    assert "Used-vCPU cost is minimal at N=2" in summary
    assert "The throughput knee is N=4" in summary
    assert "use 4 servers per" in summary


def test_resume_preserves_existing_record(tmp_path: Path) -> None:
    out = tmp_path / "resume"
    args = _args(out)
    original = sweep._base_record(args, 1)
    original.update(status="ok", error="", steps_total=17, steps_per_s=170.0)
    out.mkdir()
    line = json.dumps(original, sort_keys=True) + "\n"
    (out / "sweep.jsonl").write_text(line)

    resumed = _args(out, "--resume")
    sweep.run_sweep(resumed, cluster=sweep.DryRunCluster())
    assert (out / "sweep.jsonl").read_text() == line


def test_summary_marks_failed_row(tmp_path: Path) -> None:
    out = tmp_path / "summary"
    args = _args(out)
    good = sweep._base_record(args, 1)
    good.update(status="ok", error="", steps_total=10, steps_per_s=100.0)
    failed = sweep._base_record(args, 2)
    failed.update(status="server_start_failed", error="RCON timeout")
    sweep_path = tmp_path / "fixture.jsonl"
    sweep_path.write_text(
        "".join(json.dumps(record) + "\n" for record in (good, failed))
    )

    rendered = sweep.render_summary(sweep_path)
    assert "| 2 | server_start_failed | - |" in rendered
    assert "Aggregate throughput is maximal at N=1" in rendered


def test_worker_failure_records_point_and_tears_cluster_down(tmp_path: Path) -> None:
    out = tmp_path / "failure"
    spy = TeardownSpy()
    records = sweep.run_sweep(
        _args(out),
        cluster=spy,
        worker_target=_raising_worker,
    )
    assert records[0]["status"] == "worker_failed"
    assert records[0]["error"]
    assert spy.teardowns >= 2


@pytest.mark.parametrize("keep_logs", [False, True])
def test_point_logs_are_removed_unless_kept(tmp_path: Path, keep_logs: bool) -> None:
    out = tmp_path / "logs"
    extra = ("--keep-logs",) if keep_logs else ()
    args = _args(out, *extra)

    def logging_point_runner(args, servers, **kwargs):
        del kwargs
        point_dir = sweep._point_dir(args, servers)
        point_dir.mkdir(parents=True, exist_ok=True)
        log_path = point_dir / "env_27000.jsonl"
        log_path.write_text('{"step": 1}\n')
        record = sweep._base_record(args, servers)
        record.update(
            status="ok",
            error="",
            steps_total=1,
            steps_per_s=10.0,
            steps_per_s_per_server=10.0,
            log_bytes_total=log_path.stat().st_size,
        )
        return record

    records = sweep.run_sweep(
        args,
        cluster=sweep.DryRunCluster(),
        point_runner=logging_point_runner,
    )
    log_path = sweep._point_dir(args, 1) / "env_27000.jsonl"
    assert log_path.exists() is keep_logs
    assert records[0]["log_bytes_total"] > 0
