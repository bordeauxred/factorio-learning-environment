# Exact spatial buildability investigation — 2026-09-18

Base: tiered observation commit `bcd15fd`, branch `feat/refine-observation-state`.

## Channel count

Resolving the unique FLE `Prototype` item names through Factorio 2.0.73's
`item.place_result` produces **79 entity types** (catalog included in JSON).
This excludes non-building items and virtual groups and maps `rail` to
`straight-rail`. With the four cardinal orientations exposed by FLE, an
explicit, unshared representation is **79 × 4 = 316 binary spatial channels**.
This is a sufficient indexing scheme for those entity/direction actions,
not a proven minimum number of distinct placement-rule classes. Some masks
can share storage/computation, but equality on one map does not prove that
they have identical rules on every map. Diagonal rail actions or additional
placement modes require extending the action catalog.

An alternative policy factorization is entity first, then position/direction:
only **four conditional channels** need to be returned for the selected entity.
That changes policy architecture; it does not expose every entity's mask in
the initial observation.

The existing grid is `(17, 96, 96)` with **3×3 tiles per cell**. A single
boolean at this resolution cannot specify which individual tile anchors are
legal. Keep a separate tile-resolution buildability tensor, e.g.
`(316, 21, 21)` locally, or `(316, 288, 288)` across the existing field of view.
The latter occupies about **100 MiB float32**, **25 MiB uint8**, or **3.12 MiB
bit-packed**, before metadata. Alternatively nine within-cell offsets per
entity/direction would require 2,844 extra channels on the 96×96 grid.

Candidate anchors must use prototype-specific placement alignment (including
integer/half-integer offsets and special rail rules). A tile lattice does not
represent every arbitrary fractional position accepted by scripting. Define
the policy's discrete anchor mapping explicitly and validate those exact
coordinates. The benchmark uses tile-width/height parity offsets, rotated
with direction, and lets the engine reject invalid candidates; it is not a
complete rail/vehicle anchor enumerator.

## Exactness boundary

Use `surface.can_place_entity` with the entity, exact position, engine direction,
force, and `defines.build_check_type.manual`, matching the main placement
wrapper in `fle/env/mods/utils.lua`. Do not substitute a land/ore/shore heuristic:
footprints, directional water constraints, mining coverage, collision layers,
and rail-specific constraints matter.

These measurements cover **spatial placement checks**, not inventory/reach or
an end-to-end guarantee that the FLE tool succeeds. Combine spatial legality
with per-agent inventory and reach at decision time. FLE placement paths can
move the character or search nearby positions; an exact-anchor action should
not silently relocate the requested build. Keep placement mode (ordinary
build versus replacement/ghost) explicit.

Existing observation data is insufficient to derive an authoritative mask
locally: terrain records discard some geometry, fields are quantized, and
updates can lag. Invalidation must cover moving colliders, all relevant entity
and tile edits, resource depletion, and eventless scripted mutations. The
existing periodic reconciler alone cannot guarantee exact masks on every tick.
Revalidate at action execution because a live world can change after polling.

## Measured latency

Actual local Docker runs using `factoriotools/factorio:2.0.73`, box64 on arm64,
fresh open_world scenario, map seed 44340, speed 10 then **world paused** during
measurement. No populated-factory workload was added. Six measured repetitions
after one discarded warmup per case. Baseline and mask observations alternate.
Numbers include one RCON request, all engine placement checks, ASCII mask
serialization/transfer, existing observation reconciliation/tensors, and mask
conversion to float32. Masks are recomputed fully each time. All channels in a
case are batched into one command; these are not per-tile network round trips.

| Anchor grid | Channels | Checks/poll | Median observation + masks | Paired observation baseline |
|---|---:|---:|---:|---:|
| 21×21, tile step 1 | 1 | 441 | 4.15 ms | 2.83 ms |
| 21×21, tile step 1 | 4 | 1,764 | 8.76 ms | 2.23 ms |
| 21×21, tile step 1 | 8 | 3,528 | 16.26 ms | 1.98 ms |
| 21×21, tile step 1 | 316 | 139,356 | 774.63 ms | 4.20 ms |
| 96×96, tile step 1 | 8 | 73,728 | 316.04 ms | 2.79 ms |
| 96×96, tile step 3 (sparse anchors) | 1 | 9,216 | 76.36 ms | 3.05 ms |
| 96×96, tile step 3 (sparse anchors) | 8 | 73,728 | 328.43 ms | 1.84 ms |
| 288×288, tile step 1 | 1 | 82,944 | 348.23 ms | 2.67 ms |
| 288×288, tile step 1 | 8 | 663,552 | 5,311.56 ms | 5.18 ms |

The eight representative channels are iron chest, burner drill, electric drill,
pumpjack (north each), and offshore pump in all four cardinal directions.
The four-channel local case is **accumulator in four directions**; its time
must not be claimed as a universal four-channel cost for every entity.
The 316-channel local case checks the entire catalog. A 21×21 square is a
candidate neighborhood around a 10-tile reach, not a circular reach mask;
agent fractional position and actual reach must still be applied.

These are baseline implementation measurements, **not a lower bound** and not
production-integrated mask latency. Bit packing, buffer reuse, grouping proven
equivalent rules, caching static constraints, and recomputing affected regions
can improve them. Their exact performance has not been measured here. Dense
full-field results should not be extrapolated linearly to all 316 channels.

## Recommendation

Keep the 17-channel context grid. Select a build entity first and request four
engine-validated direction masks over reachable candidate anchors. A measured
example costs ~8.8 ms total versus ~2.2 ms baseline, rather than ~775 ms for
the full local catalog. Cache only constraints with complete invalidation and
retain a final authoritative check before executing the action. If the policy
requires simultaneous all-entity masks, budget for 316 logical channels and
benchmark an optimized cache with dynamic-factory correctness tests before
promising millisecond observation latency.

## Reproduce

Run from this checkout with the existing Python environment (numpy,
factorio_rcon and pytest) and Docker access:

```sh
python tests/benchmarks/benchmark_buildability.py
python tests/benchmarks/benchmark_buildability.py --local
```

Each command starts and removes its own container through the existing isolated
observation fixture. JSON files alongside this report contain raw per-sample
timings, machine platform, and the resolved catalog. The benchmark asserts the
returned mask length; it does not yet test equivalence with every FLE action.

API reference: https://lua-api.factorio.com/latest/classes/LuaSurface.html#can_place_entity
(API documentation is latest; live measurements explicitly use 2.0.73.)
