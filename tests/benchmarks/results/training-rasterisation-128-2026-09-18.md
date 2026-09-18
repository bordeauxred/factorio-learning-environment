# Training rasterisation profile — 128×128 buildability

CPU/NumPy measurements on macOS-26.3.1-arm64-arm-64bit-Mach-O. Base Factorio 2.0.73, paused world, 1,000 iron chests. Each measurement has 20 warmup iterations and 100 timed iterations. p95 uses NumPy percentile interpolation. RCON is excluded from CPU-only measurements.

## Live incremental polling

The default budget supplies two 8×8 channel-blocks per poll. The paused workload has no ongoing entity mutations. Each poll decodes the stream, constructs the nearest-entity view, and copies an owned float32 sample.

| Stage | Median ms | p95 ms |
|---|---:|---:|
| rcon | 2.962 | 4.173 |
| decode_apply | 0.032 | 0.053 |
| observation | 0.123 | 0.160 |
| owned_float32_snapshot | 0.287 | 0.309 |
| client_total | 0.439 | 0.519 |
| total_with_rcon | 3.408 | 4.662 |

Totals are measured directly; separately calculated medians need not add up.

## CPU-only rasterisation and training conversion

| Operation | Median ms | p95 ms |
|---|---:|---:|
| observation | 0.075 | 0.084 |
| snapshot_existing_observation | 0.292 | 0.320 |
| buildability_float32_cast | 0.268 | 0.302 |
| fresh_legal_mask | 0.385 | 0.433 |
| observation_and_snapshot | 0.354 | 0.437 |
| full_grid_rebuild | 5.838 | 6.390 |
| full_grid_and_table_rebuild | 8.120 | 8.505 |
| apply_0_entity_upserts | 0.000 | 0.000 |
| train_sample_0_entity_upserts | 0.359 | 0.404 |
| apply_1_entity_upserts | 0.006 | 0.007 |
| train_sample_1_entity_upserts | 0.365 | 0.411 |
| apply_10_entity_upserts | 0.056 | 0.061 |
| train_sample_10_entity_upserts | 0.419 | 0.461 |
| apply_100_entity_upserts | 0.563 | 0.601 |
| train_sample_100_entity_upserts | 0.928 | 1.002 |
| apply_1000_entity_upserts | 5.660 | 5.803 |
| train_sample_1000_entity_upserts | 6.121 | 6.316 |
| flat_float32_feature_vector | 0.282 | 0.300 |

Entity-update tests replay captured real rows as forced upserts. They measure cost versus update count, not actual factory mutation rates. `train_sample_*` includes applying those rows, observation assembly and an owned float32 snapshot; it excludes terrain/buildability record decoding and RCON. Full rebuild timings include existing terrain caches (196 water chunks, 1,938 ore tiles, 5,037 trees, 185 obstacles), with only in-window contributions rasterised.

## Tensor layout and ownership

- Context: float32 `(17, 96, 96)`, returned flattened, covering 288×288 world tiles at three-tile resolution.
- Buildability: int8 `(63, 128, 128)` in the cache, cast to float32 for this training sample. Values retain −1 unknown, 0 blocked and 1 legal when sampled.
- Nearest entities: float32 `(2048, 38)` and `(2048,)` mask; globals: float32 `(10,)`.
- Sample ticks: int64 `(63, 16, 16)`, copied alongside buildability origin and episode/surface metadata.
- The context and buildability views have different extents and potentially different centers. Keep their coordinate metadata; do not concatenate them as spatial channels without resampling.
- Owned numerical arrays consume 5,204,008 bytes (4.96 MiB) per sample. Allocations and copying are included. Python metadata overhead is excluded.
- `observation()` alone exposes shared cache arrays; retaining these without copying is unsafe for a rollout buffer. The benchmark snapshot checks equal values and independent storage.

At capture time 15,360 of 1,032,192 buildability entries were sampled. Casts, copies and masks process the entire allocated tensor, including unknown entries. This is not a complete/fresh cache coverage benchmark. Age filtering is timed separately and is not included in the float32 snapshot timings.

## Findings

The incremental path avoids a full raster rebuild. With few updates, float32 buildability expansion dominates CPU-only snapshot conversion. At 100 forced entity updates, cProfile identifies entity row parsing, table updates and subtract/add grid contributions as the largest aggregate cost. Rows are currently parsed separately for the table and spatial contribution. These are potential optimisation targets; this profiling change does not alter the implementation.

PyTorch is not installed in the measured environment. These results cover NumPy tensors, not framework wrapping, GPU transfers, batched rollout storage or model execution. Feature normalisation and categorical-ID encoding are policy-specific and excluded. cProfile instrumentation is only used for the separate hotspot report, not the latency samples.

## Reproduce

```sh
PYTHONPATH=. python tests/benchmarks/benchmark_training_rasterisation.py
```

This starts and removes a dedicated Factorio container. Outputs: [raw samples](training-rasterisation-128-2026-09-18.json), [hotspot profile](training-rasterisation-128-2026-09-18.profile.txt).
