# Spec: VM throughput profiler (steps/s and steps/s per dollar)

Written 2026-09-17. Purpose: before committing GCP credits to multi-day
Rainbow runs, measure on real hardware how many concurrent Factorio
headless servers a given VM sustains, where aggregate macro-step
throughput peaks, and what a macro step costs in dollars. Companion to
`docs/rl/specs/compute-recipe.md`, which guessed 24 servers on a
`c3d-highcpu-60`. That guess is now retired: the machine was unbuildable
under this project's quota and the sizing rule turned out to be
0.7 x vCPUs. Results: `docs/rl/results/vm_throughput.md`.

## Deliverables

1. `tests/benchmarks/vm_throughput_sweep.py` - the sweep driver.
2. `tests/benchmarks/vm_bootstrap.sh` - one-shot VM setup, idempotent.
3. `tests/benchmarks/test_vm_throughput_sweep.py` - offline tests.

No changes to `fle/rl/env.py`, `fle/rl/dqn.py`, or anything under
`fle/cluster/`. Read from them, do not edit them.

## 1. `vm_throughput_sweep.py`

### CLI

```
--servers 8,16,24,32,40,48,64   sweep points, ascending
--seconds 300                   measured window per point (after warmup)
--warmup 30                     seconds discarded at the start of each point
--scenario open_world
--regime macro|bare             passed to FleMacroEnv(regime=...)
--speed 40                      FleMacroEnv speed
--machine c3d-highcpu-90        label only, recorded in output
--price-per-hour 0.75           VM USD/hour, recorded and used for cost math
--vcpus 90                      required outside --dry-run; VM vCPU count
--max-steps N                   optional episode-step override; default uses FleMacroEnv's 256
--out runs/throughput/<label>
--learner-probe / --no-learner-probe   default on
--log-env / --no-log-env        default on; match production per-step JSONL I/O
--keep-logs                     retain per-point env JSONL files (default deletes them)
--resume                        skip points already in sweep.jsonl
--dry-run                       no Docker, no RCON: use fle.rl.fake_env
```

### Per sweep point

1. Tear down any running cluster, then start N servers with the same code
   path `fle cluster start -n N -s <scenario>` uses
   (`fle.cluster.run_envs`). Record map-generation wall seconds.
2. Wait until RCON answers on all N ports (ports 27000..27000+N-1),
   retrying with backoff, hard timeout 600 s. A point that never comes up
   is recorded with `"status": "server_start_failed"` and the sweep
   continues.
3. Spawn exactly N worker processes (`multiprocessing`, spawn start
   method), one per port. Each worker builds an `FleMacroEnv` on its port
   and loops `S.random_valid_action(obs, rng)` -> `env.step(...)` in the
   same shape `fle/rl/env.py::main` does, including the CRAFT
   quantity-0 probe, until a shared deadline. Workers report back per-step
   wall times (as summary stats, not the raw list) and step counts. By
   default the environment receives no `max_steps` or `max_ticks` override,
   matching the production learner's 256-step, 216,000-tick episode. An
   explicit `--max-steps` is passed as `max_steps` only, preserving
   `FleMacroEnv`'s default tick-budget scaling.
4. Sample every 5 s during the window: system-wide CPU percent per core
   (psutil), 1-minute load average, total RSS of the Factorio containers,
   free RAM.
5. Measure game speed honestly: each worker records elapsed game ticks
   (the `elapsed_ticks` path exercised by
   `tests/benchmarks/test_elapsed_ticks.py`) over the window, so we can
   report ticks/s/server, i.e. real UPS under load, not just steps/s.
6. Match production action logging by writing one file per worker at
   `<out>/point-<N>/env_<port>.jsonl`. Record their total size, then remove
   the JSONL files after the point unless `--keep-logs` is set.
7. Tear the cluster down before the next point, whatever happened.

### Recorded per point (one JSON object per line in `sweep.jsonl`)

```
machine, price_per_hour, vcpus, price_per_vcpu_hour,
servers, vcpus_used, seconds, regime,
mapgen_s, steps_total, steps_per_s, steps_per_s_per_server,
episodes_completed, decisions_per_episode, reset_s_mean,
step_wall_p50, step_wall_p90, step_wall_p99,
ticks_per_s_per_server, cpu_pct_mean, load1_mean,
rss_gb_total, rss_gb_per_server, ram_free_gb_min,
ok_rate, invalid_rate, deaths,
usd_per_1k_steps, steps_per_usd, usd_per_1k_steps_used,
marginal_steps_per_s_per_vcpu, log_bytes_total, status, error
```

`steps_per_usd = steps_per_s * 3600 / price_per_hour`.
This whole-VM metric is proportional to `steps_per_s` within one sweep because
the hourly VM price is fixed, so their optima are identical by construction.
It remains useful for comparing machine types.

For capacity selection, price the CPUs occupied by the Factorio containers:
`price_per_vcpu_hour = price_per_hour / vcpus`, `vcpus_used = servers`, and
`usd_per_1k_steps_used = vcpus_used * price_per_vcpu_hour /
(steps_per_s * 3.6)`. The compose template pins one CPU per server.
`marginal_steps_per_s_per_vcpu` is the change in `steps_per_s` divided by the
change in `vcpus_used` between consecutive successful points; it is null for
the first successful point. Workers also report completed episodes, mean
decisions per completed episode, and mean reset wall time so the sweep includes
the same episode-boundary cost and episode-length behavior as production.

### 2. Learner probe

Separate, runs once per VM, not per sweep point. For
`torch_threads in (1, 2, 4)`: run the Rainbow configuration from
`docs/rl/specs/compute-recipe.md` against `--fake` (no servers) for a
fixed 2,000 learner updates and record updates/s and RSS. Write
`learner.json`. This is what decides vCPUs per learner and how many
servers one learner can keep fed.

### 3. `summary.md`

A markdown table of all points plus, in prose:
- the N that maximises aggregate steps/s,
- the N that minimises used-vCPU USD per 1,000 steps,
- the whole-VM steps/USD optimum, noting that it equals the aggregate optimum
  at a fixed VM price,
- the throughput knee: the largest N whose marginal steps/s per vCPU is at
  least half the average steps/s per vCPU at that point,
- ticks/s/server at that N versus at N=8, so throughput collapse is visible,
- decisions per completed episode and mean reset time,
- learner updates/s per thread count, and the implied servers-per-learner
  at the measured steps/s,
- a one-line recommendation for `compute-recipe.md` using the knee N.

Keep the tone of the existing results files in `docs/rl/results/`: plain
sentences, numbers with units, no adjectives.

## `vm_bootstrap.sh`

Idempotent, safe to re-run. Installs docker.io, docker-compose-v2, git,
tmux, htop; adds the user to the docker group; installs uv; clones or
pulls the repo at `feat/pufferlib-rl`; `uv venv .venv --python 3.12`;
`uv pip install -e .`; runs `pytest -q tests/rl` and stops with a clear
message if that fails. It must NOT start any cluster or sweep on its own.

## Tests (must run offline, no Docker, under 10 s)

- `--dry-run` sweep over `--servers 2,4 --seconds 2 --warmup 0` produces a
  `sweep.jsonl` with one well-formed record per point and every field
  above present and correctly typed.
- The cost math: a point with `steps_per_s=10, price_per_hour=0.75`
  yields `usd_per_1k_steps == 0.75 / 36.0` within 1e-9.
- `--resume` on an existing `sweep.jsonl` skips completed points and
  leaves their records untouched.
- Summary rendering from a fixture `sweep.jsonl` containing one failed
  point does not crash and marks that row as failed.
- A worker that raises is recorded as a failed point, and the cluster
  teardown still runs (assert via a fake teardown spy).
- The default worker construction does not override `max_steps` or `max_ticks`;
  an explicit `--max-steps` still leaves `max_ticks` at its environment default.
- Marginal throughput and knee selection match a fixture with known values.
- Per-point environment logs are removed after a point unless `--keep-logs`
  is set.

## Constraints

- Python 3.12, stdlib plus what the repo already depends on
  (numpy, psutil if already present - check `pyproject.toml`; if psutil is
  absent, read `/proc/stat` and `/proc/meminfo` directly rather than
  adding a dependency).
- The script must be safe to kill: SIGINT tears the cluster down and
  writes whatever points completed.
- Never write into `docs/` from the script; write into `--out`.
