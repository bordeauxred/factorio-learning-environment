# INT8 buildability cast cost

Keep buildability as `int8` in the rollout buffer and cast whole batches on the accelerator; on this host that reduces the CPU step from 409.9 µs to a 12.5 µs ownership-preserving copy and reduces stored bytes fourfold. If a CPU `float32` view is immediately required, use a retained output buffer and Accelerate `vDSP_vflt8`, which took 63.8 µs, or ordinary NumPy assignment at 274.5 µs if a platform-specific dependency is unacceptable. The NumPy bottleneck is an unvectorised scalar ARM64 conversion loop plus about 135 µs of fresh-page work, not the 8,192-element buffered-cast machinery and not attainable memory bandwidth.

## Evidence

The reproduction used 400 timed samples per operation after 30 warmups, `time.perf_counter_ns`, disabled cyclic GC during timing, and validated the Accelerate, Torch, and lookup outputs against the source. macOS has no supported process-affinity API, so the process was not pinned; measurements were made in one otherwise idle standalone run. Values below are from Python 3.12.0, NumPy 2.3.4, macOS 15.1 on the stated M3 Max. “GB/s” divides the 4,128,768 output bytes by median time, matching the rate definition in the prompt.

| Operation | Median (µs) | p95 (µs) | Min (µs) | Effective GB/s |
|---|---:|---:|---:|---:|
| `src.astype(np.float32)`, fresh | 409.916 | 510.059 | 255.958 | 10.07 |
| `dst[...] = src`, retained destination | 274.542 | 303.557 | 255.791 | 15.04 |
| `big_f32.copy()`, fresh | 169.083 | 220.925 | 130.667 | 24.42 |
| `dst[...] = big_f32`, retained destination | 49.458 | 61.375 | 45.375 | 83.48 |
| `np.zeros(..., float32)` | 123.855 | 173.442 | 99.750 | — |
| `np.empty(...).fill(1.0)` | 162.459 | 225.002 | 123.875 | — |

These confirm the supplied table within ordinary run-to-run variation: deviations are +1.7%, -2.3%, -7.1%, +0.9%, -6.2%, and -3.3%, respectively. The retained float copy is 5.55 times faster than the retained cast, close to the stated 5.7 times. The one claim not reproduced is NumPy 2.5.2’s 4.6% improvement: the same script under the supplied second environment measured 406.312 µs (p95 538.354, min 260.208), only 0.9% below 2.3.4 and well inside observed allocation noise.

Ranked alternatives from the 2.3.4 run are:

| Alternative | Median (µs) | p95 (µs) | Min (µs) | Speedup vs 403 µs |
|---|---:|---:|---:|---:|
| Accelerate `vDSP_vflt8`, retained output | 63.750 | 74.968 | 59.333 | 6.32× |
| Four-thread `np.copyto`, row blocks | 130.666 | 156.556 | 117.833 | 3.08× |
| Torch `.to(float32)`, 10 CPU threads | 134.021 | 149.216 | 46.333 | 3.01× |
| 512K-element chunks, retained output | 276.167 | 296.268 | 275.333 | 1.46× |
| Eight-buffer rotating pool | 282.688 | 334.731 | 255.833 | 1.43× |
| `np.copyto(..., casting="unsafe")` | 285.625 | 326.731 | 255.917 | 1.41× |
| 16K-element chunks, retained output | 300.209 | 321.087 | 278.083 | 1.34× |
| Staged `int8 -> int16 -> float32` | 303.084 | 331.936 | 276.750 | 1.33× |
| 256-entry table with `np.take` | 3,966.917 | 4,588.330 | 3,553.792 | 0.10× |

Chunking offers no conversion advantage: a 2.5 MiB source-plus-destination working set merely matches the contiguous loop, while an 80 KiB working set adds Python/slice overhead. Staging adds another scalar pass, and general indexed gather makes the lookup table particularly poor. Multithreading works because NumPy releases the GIL for this large transfer, but consumes four cores and has scheduling overhead. Torch is installed here and competitive using its default ten threads; it was not installed in the 2.5.2 environment. Accelerate is the fastest CPU result and uses one retained output, though its framework call is macOS-specific.

## Mechanism

`astype` allocates a destination and reaches NumPy’s assignment/casting path. In the installed `_multiarray_umath` binary, `_raw_array_assign_array` prepares a two-array raw iterator, calls `PyArray_GetDTypeTransferFunction`, and invokes the selected transfer loop. Aligned, C-contiguous `int8` and `float32` operands select `__aligned_contig_cast_byte_to_float` (the generated `_aligned_contig_cast` family).

Disassembly is decisive: each iteration performs `ldrsb` for one byte, scalar `scvtf s0,w11`, scalar `str s0`, then updates pointers/count and branches. There are no NEON vector loads, widening conversions, or vector stores. `np.show_config()` says the 2.3.4 wheel was built with NEON, NEON_FP16, NEON_VFPV4, and ASIMD baseline support, with ASIMDHP/ASIMDFHM dispatch targets; 2.5.2 reports the same baseline and adds ASIMDDP to its dispatch list. Thus the wheel includes ARM SIMD infrastructure, but this legacy dtype-transfer loop does not use it.

The contiguous dimensions are coalesced and the returned loop is called for the full raw span. `NPY_BUFSIZE` is indeed 8192 in `ndarraytypes.h`, but this operation does not allocate or cycle through that buffer. The 16K and 512K manual-chunk results also show no hidden cache-size win.

This is not memory-bound. A cast has at least 1,032,192 bytes read plus 4,128,768 written and only one conversion per element: 0.2 conversions per byte. At 274.5 µs the minimum traffic rate is 18.8 GB/s. The same process copies 8,257,536 aggregate bytes for retained float-to-float assignment in 49.5 µs (167 GB/s), and `vDSP_vflt8` performs the same cast traffic in 63.8 µs (80.9 GB/s). Both demonstrate ample bandwidth; scalar conversion throughput is limiting NumPy.

Allocation is separately visible. Fresh `astype` costs 135.4 µs more than retained assignment; zeroing 4,128,768 bytes costs 123.9 µs. A persistent anonymous mapping casts in 274.5 µs, but `MADV_DONTNEED` before each cast raises that to 422.1 µs as pages must be supplied again. `MADV_WILLNEED` takes 282.2 µs and provides no improvement. A retained eight-buffer pool remains effective at 282.7 µs; repeatedly allocating and freeing does not, because macOS may make large freed pages reusable/discardable. A pool therefore helps only while its buffers remain live, and its capacity must cover all in-flight observations.

## Recommendation

| Client choice | CPU work per step | Minimum bytes touched | Rollout bytes per step |
|---|---:|---:|---:|
| Current fresh NumPy cast | 409.916 µs | 5,160,960 plus page provisioning | 4,128,768 |
| NumPy cast into retained buffer | 274.542 µs | 5,160,960 | 4,128,768 if retained per step |
| Keep `int8` with an owned copy | 12.500 µs | 2,064,384 | 1,032,192 |
| Accelerate into retained `float32` | 63.750 µs | 5,160,960 | 4,128,768 if retained per step |

The rollout path should take the third choice. It saves approximately 3 MiB per step (about 4 MiB versus 1 MiB) and defers a large contiguous batch conversion to the device that will consume it. The 12.5 µs figure includes an `int8.copy()`; transferring ownership could avoid even that copy, but the current mutable observation contract should not be assumed to permit it. Reusing one float buffer is safe only for immediate consumption, not for a rollout whose earlier steps must remain unchanged; a rotating float pool restores the full four-byte-per-cell memory cost.

Finally, the reference machine’s 365–461 µs whole-snapshot result cannot be attributed to CPU generation, macOS version, or NumPy version from these data. NumPy version is specifically disfavored by the 0.9% comparison; CPU and OS were not varied independently. Running this exact script plus `np.show_config()` and disassembling the reference wheel’s selected byte-to-float loop would distinguish faster hardware, a different cast implementation, and a benchmark/ownership difference.
