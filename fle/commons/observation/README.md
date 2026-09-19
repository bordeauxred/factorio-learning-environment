# Progressive buildability observations

For the separate 512×512-tile, 14-channel minimap and combined-client usage,
see [MINIMAP.md](MINIMAP.md). It contains no buildability channels.

The tiered `open_world` observation protocol can progressively populate the
63 buildability channels over the **128×128-tile buildability area**.
The existing 17-channel context grid remains at its original three-tile
resolution. Buildability is a separate tile-resolution view, with its own
world origin, because a single bit per 3×3 cell cannot represent exact anchors.

The channel manifest is for base Factorio **2.0.73**, four cardinal directions,
and new construction (occupied placement sites cannot be replaced or rotated).
Other mod sets require regenerating and validating the classes. The 63-channel
mapping is a validated conservative reduction, not a proven global minimum.

## Use with the existing tensor client

Run from the repository root with its Python dependencies installed and
`PYTHONPATH=.:tests/benchmarks`. The server must run this checkout's
`fle/cluster/scenarios/open_world` scenario on base Factorio 2.0.73 (no additional
mods); an older running scenario does not acquire the new Lua globals merely
by updating the Python checkout. `rc` below is your connected Factorio RCON
client. Enable once per episode, then keep the same client across polls.

```python
from benchmark_tensor_obs import TensorClient

client = TensorClient()
client.apply_entity(rc.send_command("/sc obs_diff_full_sync()") or "")
client.apply_terrain(rc.send_command("/sc obs_terrain_full_sync()") or "")
client.buildability.configure(rc)  # 128×128 window, 256 engine checks/poll

# Repeat this section for each observation:
response = rc.send_command("/sc obs_all_drain()") or ""
entities, _, terrain = response.partition("~")
client.apply_entity(entities)
client.apply_terrain(terrain)

grid, globals_, entities, entity_mask, build = client.observation(
    include_buildability=True
)
# build["values"]: int8 (63, 128, 128): -1 unknown, 0 blocked, 1 legal at sample
# build["sampled_ticks"]: int64 (63, 16, 16), one tick per 8×8 tile block
# build["origin"]: world (x, y) of array tile [0, 0]
# build also includes surface, force, current tick, generation and manifest ID.

channel = client.buildability.channel_for("pipe-to-ground", 0)  # engine north
# Unknown and samples older than 120 ticks are excluded:
legal = client.buildability.legal_mask(client.tick, max_age_ticks=120)
```

`TensorClient` is currently the reference implementation in
`tests/benchmarks/benchmark_tensor_obs.py`, matching the pre-existing tensor
protocol. The reusable buildability decoder and manifest live in this package.
The default `observation()` return value remains the existing four-tuple;
requesting buildability adds a metadata dictionary with zero-copy arrays.
Copy these arrays when retaining observations across later drains.

For tile index `(x, y)`, the world tile is `origin + (x, y)`. Candidate entity
positions use prototype width/height parity offsets (integer versus half-tile
anchors), rotated with the requested direction. Rails/vehicles can additionally
snap according to the engine's placement rules. Engine directions are
`0/4/8/12`; do not pass FLE's tool-facing direction enum without conversion.
The manifest records every entity/direction alias, including the four distinct
underground-pipe masks. `rail` is accepted as an alias for `straight-rail`.

For the existing benchmark CLI:

```sh
PYTHONPATH=. python tests/benchmarks/benchmark_tensor_obs.py \
  --port 27099 --buildability --buildability-budget 256
```

## Work scheduling and storage

- Enable explicitly with `obs_buildability_configure{...}`; unused environments
  do no buildability work. `enabled=false` disables and clears the client view.
- Each block contains 64 tile decisions for one channel. The server evaluates
  `can_place_entity(manual) AND can_place_entity(ghost_revive)`, at most 128
  engine checks per block. `check_budget=256` permits two blocks per drain.
  Budgets must be multiples of 128, from 128 through 8192.
- Initial traversal fills nearby blocks first, evaluating all channels of a
  block before proceeding outward. Dirty work and refreshes of the nearest
  nine blocks alternate with the normal traversal so distant work progresses.
- `refresh_ticks=600` makes samples eligible for refresh. This is **not a bound
  on maximum sample age**: total refresh time depends on the work budget,
  polling rate, window size and invalidation traffic.
- The window follows the tracked character, recenters after a 24-tile dead
  zone, and aligns its origin to eight tiles. Overlapping client/server blocks
  survive recentering. When using the full window, the tensor client's context
  grid follows the same origin so both views cover exactly the same area.
  The server evicts blocks outside the current window.
  A surface/force change starts a fresh generation.
- Boolean blocks use two 32-bit words on the wire. The client uses about
  **1.11 MiB** for its int8 tile values and per-block int64 timestamps, separate
  from existing observation arrays. Float32 expansion and model execution are
  downstream costs, not part of the cache's polling benchmark.

The default window contains 16,128 channel-blocks. At the default two blocks per
poll it needs **at least 8,064 polls** for initial coverage, before refresh or
invalidation work. This is progressive coverage, not a complete fresh snapshot
on the first poll. Raise the budget for faster filling at greater polling cost.
The optional server `size` must be a multiple of 16 in `[16,288]`; the public
default is 128×128. The existing coarse context grid still covers 288×288 tiles.

## Changes, freshness and recovery

Build/mine/die/rotate/clone events, tile events, depletion events, and
`obs_diff_touch(entity)` invalidate affected cached areas immediately on the
next drain. Both old and new footprints are invalidated for tracked movement.
Relevant fluid changes invalidate all fluid-dependent channels across the
window: changing a connection outside the local block can alter distant fluid
systems. The global dependency includes fluid-bearing members of shared classes,
not just the chosen representative.

The existing reconciler compares placement-relevant state (position, direction,
footprint, force, fluid types and recipe) independently of quantized entity
rows. Routine energy/count/progress updates do not invalidate these masks.
Eventless changes and moving untracked entities are eventually detected through
rolling engine queries. Bulk scripts that bypass events should call
`obs_buildability_invalidate()`; scripts editing an entity can use the existing
`obs_diff_touch(entity)` hook.

**A cached bit is exact at its sampled tick, not guaranteed current.** Inventory
and player reach are separate gates. Use sample ages to exclude old values and
revalidate the chosen placement against the engine when executing the action.
The cache does not claim complete instant invalidation of arbitrary script
mutations or fluid simulation changes.

`obs_buildability_full_sync()` begins a new client generation and progressively
replays valid cached blocks, reusing engine results rather than synchronously
recomputing the viewport. Apply its returned records using `apply_terrain()` and
continue normal drains. Reconfigure for a new episode/server after a world
restore. Like the original destructive drain protocol, this is an ordered,
single-consumer stream; it is not a multi-agent broadcast protocol.

## Wire format

Buildability records share the terrain half of `entities~terrain`, so existing
clients can ignore the new tags without changing framing:

| Record | Meaning |
|---|---|
| `B<manifest>,<generation>,<surface>,<force>,<x0>,<y0>,<size>` | Initialize or move the window |
| `Boff,<generation>` | Disable and clear |
| `b<generation>,<channel>,<cx>,<cy>,<tick>,<hex>` | 64 values for world block `(cx,cy)` |
| `R<generation>,<cx0>,<cy0>,<cx1>,<cy1>` | Invalidate all channels in an inclusive block rectangle |
| `J<generation>,<channel>` | Invalidate this channel across the viewport |

Channel IDs are zero-based. Hex is two eight-digit words: bits 0–31 then 32–63,
row-major within an 8×8 block. Window metadata precedes invalidations, which
precede newly sampled blocks. Generation checks reject records from earlier
resets. The manifest ID rejects incompatible channel mappings.

## Validation and performance

```sh
PYTHONPATH=. python -m pytest tests/observation_diff/ -o addopts='' -q
PYTHONPATH=. python tests/benchmarks/benchmark_progressive_buildability.py
```

Both use a dedicated disposable Factorio container. Tests cover all-channel
engine parity, budget enforcement, unknown/expired masks, entity/tile/fluid
invalidation, recentering, surface changes, replay, fairness and silent changes.
Raw 128×128 timings are saved in
`tests/benchmarks/results/progressive-buildability-128-2026-09-18.json`.
The previous 288×288 baseline is preserved in
`tests/benchmarks/results/progressive-buildability-2026-09-18.json`.

Measured on 2026-09-18 with base Factorio 2.0.73 under Docker/box64 on an
ARM64 host: 1,000 chests, paused world, 128×128 window, 20 warmup polls
and 100 measured polls per configuration. Times include RCON, decoding and
the tensor client's observation assembly.

| Maximum engine checks/poll | Median | p95 |
|---|---:|---:|
| Disabled (before) | 2.75 ms | 3.60 ms |
| 128 | 2.95 ms | 4.51 ms |
| 256 (default) | 3.74 ms | 5.27 ms |
| 1024 | 4.97 ms | 6.61 ms |
| Disabled (after) | 2.71 ms | 3.58 ms |

The default adds approximately 1.0 ms to median polling time in this initial
fill workload. The previous 288×288 run measured 3.52 ms median / 4.79 ms p95
at the same default budget. This smaller window does not demonstrate a
per-poll speedup: engine work per poll remains bounded by the same budget.
It reduces cache storage and minimum initial fill work by 80.25% (5.06×).
These measurements do not characterize a running factory's
invalidation traffic, full-cache refresh latency, or action execution latency.

Client-side rasterisation and owned float32 training samples are profiled
separately in
[`training-rasterisation-128-2026-09-18.md`](../../../tests/benchmarks/results/training-rasterisation-128-2026-09-18.md).
Run `PYTHONPATH=. python tests/benchmarks/benchmark_training_rasterisation.py`
to reproduce the CPU/NumPy measurements. With 1,000 entities, the live polling
client stages (decode, observation assembly and owned sample) measured
0.44 ms median / 0.52 ms p95, excluding RCON. A full grid-and-table rebuild
measured 8.12 ms median. Framework conversion, GPU transfer and model execution
are excluded.
