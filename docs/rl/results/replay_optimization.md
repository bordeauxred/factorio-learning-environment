# Replay batch assembly: offline CPU profile and fix

Measured 2026-09-13 on the loaded 38.65 GB Apple host. No Factorio process,
training loop, MPS operation, or device transfer was run by this investigation.
Device-side timing and end-to-end training throughput remain unmeasured.

## Method

The live observation schema is 38,495 float32 values (153,980 bytes). Batch size
was 256 and n-step horizon was 3. The first fixture was the checked-in
`rung1_scripted_replay.npz` (838 entries, 176 real transitions, 180 stored
boundaries; 79.0% of entries were HER after relabeling). The second had 1,024
real transitions in sixteen 64-action episodes, with binary grids and dense
random float32 entity features. Both used the real vocabulary and observation
width. Timings below are medians of six pre-change or eight post-change batches
after a warmup batch, with one CPU thread for BLAS. Component timings are
isolated measurements and their medians need not sum exactly to the total.

The pre-change path was profiled *before editing replay*: priority traversal,
entry/episode lookup, zlib decompression, copying decoded entries, `get`,
three-step return assembly, and both `np.stack` calls were timed separately.
The post-change hot path was profiled as priority sampling plus
`assemble_batch`, with its lookup, n-step metadata, vector gathers, and HER
goal writes measured separately. Benchmark scripts ran from `/tmp` and used
only synthetic arrays and the checked-in archive.

| CPU time per batch 256 | Demonstration, before | Demonstration, after | Dense synthetic, before | Dense synthetic, after |
| --- | ---: | ---: | ---: | ---: |
| Learner batch assembly including priority | 664.962 ms | 45.312 ms | 903.728 ms | 45.045 ms |
| Priority traversal | 0.422 ms | 0.466 ms | 0.340 ms | 0.365 ms |
| Entry/episode lookup | 0.041 ms | 0.030 ms | 0.046 ms | 0.031 ms |
| zlib decompression inside `get` and n-step | 589.833 ms | 0 | 813.457 ms | 0 |
| Decoded-entry copies (before) | 11.760 ms | — | 14.126 ms | — |
| `get` including decode/reward (before) | 151.333 ms | — | 209.059 ms | — |
| n-step including decode/reward (before) | 505.460 ms | — | 679.810 ms | — |
| Both `np.stack` calls (before) | 10.407 ms | — | 7.183 ms | — |
| n-step metadata/reward (after) | — | 34.763 ms | — | 34.764 ms |
| Current/next vector gathers (after) | — | 8.246 / 4.218 ms | — | 8.792 / 5.610 ms |
| HER goal overlay writes (after) | — | 0.505 ms | — | 0.015 ms |

The hypothesis was right about assembly, but the main mechanism was repeated
**decompression**, rather than Python index lookup or `np.stack`. The old
learner decoded two full boundaries per sampled entry, then up to six more to
compute a three-step return, even though reward only reads the exact achieved
lanes. On the demonstration fixture, decompression was 590 of 665 ms.

## Change and correctness

Replay now stores each real transition's boundary in a single contiguous
float32 array. An episode keeps slot references and one final boundary; HER
copies keep only goal overlays. The learner samples priority indices, gathers
current and n-step next states into two reusable host arrays, and writes HER
goals into those gathered rows. N-step rewards read directly from stored
boundaries without decoding or copying full observations. The per-entry `get`
and `n_step` APIs remain available. Saves retain the existing compressed NPZ
layout, and loads migrate that layout into the contiguous store in memory.

All observation lanes remain **bit-exact float32**, especially `achieved`,
`goal`, and episode tick. We considered the design's uint8-grid/float16-feature
layout: it could lower RAM, but float16 changes normalized policy values and
some grid channels are not guaranteed binary. It therefore fails the requested
full-batch numerical identity guarantee. No lossy conversion was introduced.
The reusable arrays are ordinary host arrays. Pinned host memory was not used:
the active device is MPS, which has no supported CUDA-style pinned transfer
path; testing MPS transfers would contend with the live experiment.

Offline verification: `tests/rl/` passed (56 passed, 1 opt-in live test skipped).
New tests compare assembled batches to per-entry `get`/`n_step` at identical
indices, check exact-lane bytes after store and save/load, and compare selected
loaded demonstration observations to the original NPZ's decompressed bytes.

## Memory limit

At this schema width, the contiguous store costs **153,980 bytes per real
transition**, plus one 153,980-byte final boundary per episode. At 64 actions
per episode this is **156,386 bytes/transition** for observation storage,
before metadata. The two reusable batch-256 observation arrays add 78,837,760
bytes per learner. A 200,000-transition store alone would need 30.80 GB for
transition rows (about 31.28 GB including 64-action episode finals), versus
only 6.71–9.16 GB reported as available during these measurements. This
machine cannot safely hold the design's 200,000-transition full-float32 buffer.

The pre-change dense fixture held 67,335,543 compressed observation bytes for
1,040 boundaries, or about 65,758 bytes per real transition. Its measured RSS
increase during construction was 119,111,680 bytes for 1,024 transitions
(116,320 bytes/transition, including Python and temporary allocations). The
checked-in sparse demonstration archive's boundaries were only 89,556 bytes
total, showing how content-dependent the old compression was. The new dense
fixture allocated 157,675,520 bytes for its 1,024 transition rows, with a
244,842,496-byte measured construction RSS increase including temporary input
arrays and allocator retention. HER copies allocate no observation rows.

Fifty thousand raw float32 rows would require 7.70 GB before final boundaries,
metadata, and batch arrays. Available RAM fluctuated from 6.71 to 9.16 GB
while the live run was active, so allocating a 50,000-row stress fixture could
have caused swapping or contention; profiling used 1,024 rows and projected
from their measured bytes. At the low observed 6.71 GB headroom, the arithmetic
upper bound is about 42,900 transitions before any reserve. Reserving 2 GB for
the live run and transient allocations gives a **practical projected cap of
about 30,000 transitions** on this machine, not a measured saturation point.
The default 200,000 capacity is a virtual reservation until rows are inserted;
it must be reduced for an actual long-running experiment here.
