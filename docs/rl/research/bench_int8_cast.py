"""Benchmark int8-to-float32 conversion for the RL buildability raster."""

from __future__ import annotations

import ctypes
import gc
import mmap
import platform
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

SHAPE = (63, 128, 128)
ELEMENTS = int(np.prod(SHAPE))
INPUT_BYTES = ELEMENTS
OUTPUT_BYTES = ELEMENTS * np.dtype(np.float32).itemsize
SAMPLES = 400
WARMUP = 30
REFERENCE_BASELINE_US = 403.0


@dataclass
class Result:
    name: str
    median_us: float
    p95_us: float
    min_us: float


def measure(name: str, operation: Callable[[], object]) -> Result:
    for _ in range(WARMUP):
        operation()
    samples = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        result = operation()
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / 1_000)
        del result
    values = np.asarray(samples)
    return Result(
        name,
        float(np.median(values)),
        float(np.percentile(values, 95)),
        float(np.min(values)),
    )


def chunked_copy(
    source: np.ndarray, destination: np.ndarray, chunk_elements: int
) -> Callable[[], np.ndarray]:
    source_flat = source.reshape(-1)
    destination_flat = destination.reshape(-1)

    def copy() -> np.ndarray:
        for start in range(0, source_flat.size, chunk_elements):
            stop = min(start + chunk_elements, source_flat.size)
            destination_flat[start:stop] = source_flat[start:stop]
        return destination

    return copy


def rotating_pool_copy(
    source: np.ndarray, pool_size: int
) -> tuple[Callable[[], np.ndarray], list[np.ndarray]]:
    pool = [np.empty(SHAPE, dtype=np.float32) for _ in range(pool_size)]
    index = 0

    def copy() -> np.ndarray:
        nonlocal index
        destination = pool[index]
        index = (index + 1) % pool_size
        destination[...] = source
        return destination

    return copy, pool


def add_accelerate(
    operations: list[tuple[str, Callable[[], object]]],
    source: np.ndarray,
    keepalive: list[object],
) -> str:
    if platform.system() != "Darwin":
        return "unavailable (not macOS)"
    path = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
    try:
        accelerate = ctypes.CDLL(path)
        vflt8 = accelerate.vDSP_vflt8
    except (AttributeError, OSError) as error:
        return f"unavailable ({error})"
    vflt8.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_long,
        ctypes.c_ulong,
    ]
    vflt8.restype = None
    destination = np.empty(SHAPE, dtype=np.float32)
    source_pointer = source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    destination_pointer = destination.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def convert() -> np.ndarray:
        vflt8(source_pointer, 1, destination_pointer, 1, ELEMENTS)
        return destination

    convert()
    np.testing.assert_array_equal(destination, source)
    operations.append(("Accelerate vDSP_vflt8 -> reused", convert))
    keepalive.extend([accelerate, destination, source_pointer, destination_pointer])
    return "available"


def add_torch(
    operations: list[tuple[str, Callable[[], object]]], source: np.ndarray
) -> str:
    try:
        import torch
    except ImportError:
        return "not installed"
    source_tensor = torch.from_numpy(source)

    def convert():
        return source_tensor.to(torch.float32)

    converted = convert()
    np.testing.assert_array_equal(converted.numpy(), source)
    operations.append(
        (f"torch .to(float32), {torch.get_num_threads()} threads", convert)
    )
    return f"{torch.__version__} ({torch.get_num_threads()} threads)"


def main() -> None:
    rng = np.random.default_rng(20260918)
    source = rng.integers(-128, 128, size=SHAPE, dtype=np.int8)
    float_source = source.astype(np.float32)
    destination = np.empty(SHAPE, dtype=np.float32)
    int8_destination = np.empty(SHAPE, dtype=np.int8)
    intermediate = np.empty(SHAPE, dtype=np.int16)
    lookup = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float32)
    keepalive: list[object] = []

    def assign() -> np.ndarray:
        destination[...] = source
        return destination

    def copyto() -> np.ndarray:
        np.copyto(destination, source, casting="unsafe")
        return destination

    def allocate_and_copyto() -> np.ndarray:
        result = np.empty(SHAPE, dtype=np.float32)
        np.copyto(result, source, casting="unsafe")
        return result

    def assign_float32() -> np.ndarray:
        destination[...] = float_source
        return destination

    def fill_empty() -> np.ndarray:
        result = np.empty(SHAPE, dtype=np.float32)
        result.fill(1.0)
        return result

    def staged() -> np.ndarray:
        intermediate[...] = source
        destination[...] = intermediate
        return destination

    def take_lookup() -> np.ndarray:
        np.take(lookup, source.view(np.uint8), out=destination)
        return destination

    def copy_int8() -> np.ndarray:
        int8_destination[...] = source
        return int8_destination

    pool_copy, pool = rotating_pool_copy(source, 8)
    keepalive.append(pool)

    anonymous_map = mmap.mmap(-1, OUTPUT_BYTES)
    mapped_destination = np.ndarray(SHAPE, dtype=np.float32, buffer=anonymous_map)

    def mmap_copy() -> np.ndarray:
        mapped_destination[...] = source
        return mapped_destination

    def mmap_willneed_copy() -> np.ndarray:
        anonymous_map.madvise(mmap.MADV_WILLNEED)
        mapped_destination[...] = source
        return mapped_destination

    def mmap_discard_copy() -> np.ndarray:
        anonymous_map.madvise(mmap.MADV_DONTNEED)
        mapped_destination[...] = source
        return mapped_destination

    operations: list[tuple[str, Callable[[], object]]] = [
        ("astype float32 -> fresh", lambda: source.astype(np.float32)),
        ("empty + np.copyto int8 -> fresh", allocate_and_copyto),
        ("assignment int8 -> reused", assign),
        ("np.copyto int8 -> reused", copyto),
        ("rotating pool[8] int8 -> float32", pool_copy),
        ("mmap buffer int8 -> float32", mmap_copy),
        ("mmap MADV_WILLNEED + cast", mmap_willneed_copy),
        ("mmap MADV_DONTNEED + cast", mmap_discard_copy),
        (
            "chunk 16K elems (80 KiB working set)",
            chunked_copy(source, destination, 16_384),
        ),
        (
            "chunk 512K elems (2.5 MiB working set)",
            chunked_copy(source, destination, 524_288),
        ),
        ("staged int8 -> int16 -> float32", staged),
        ("256-entry lookup with np.take", take_lookup),
        ("float32.copy() -> fresh", float_source.copy),
        ("assignment float32 -> reused", assign_float32),
        ("int8.copy() -> fresh", source.copy),
        ("assignment int8 -> int8 reused", copy_int8),
        ("zeros float32 -> fresh", lambda: np.zeros(SHAPE, dtype=np.float32)),
        ("empty float32 then fill(1)", fill_empty),
        ("empty float32 only (untouched)", lambda: np.empty(SHAPE, dtype=np.float32)),
    ]

    executor = ThreadPoolExecutor(max_workers=4)
    source_blocks = np.array_split(source, 4, axis=0)
    destination_blocks = np.array_split(destination, 4, axis=0)

    def threaded_copy() -> np.ndarray:
        futures = [
            executor.submit(np.copyto, target, block, casting="unsafe")
            for target, block in zip(destination_blocks, source_blocks, strict=True)
        ]
        for future in futures:
            future.result()
        return destination

    operations.insert(7, ("4-thread np.copyto over row blocks", threaded_copy))
    torch_status = add_torch(operations, source)
    accelerate_status = add_accelerate(operations, source, keepalive)

    previous_gc_state = gc.isenabled()
    gc.disable()
    try:
        results = [measure(name, operation) for name, operation in operations]
    finally:
        if previous_gc_state:
            gc.enable()
        executor.shutdown()
        anonymous_map.close()

    print("int8 -> float32 buildability benchmark")
    print(f"Python: {sys.version.split()[0]}; NumPy: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Shape: {SHAPE}; elements: {ELEMENTS:,}")
    print(f"Input: {INPUT_BYTES:,} B; float32 output: {OUTPUT_BYTES:,} B")
    print(f"Samples: {SAMPLES}; warmup: {WARMUP}; clock: perf_counter_ns")
    print("Affinity: not pinned (macOS has no supported process-affinity API)")
    print(f"Torch: {torch_status}; Accelerate vDSP_vflt8: {accelerate_status}")
    print()
    print(
        f"{'operation':<45} {'median us':>10} {'p95 us':>10} "
        f"{'min us':>10} {'GB/s out':>10} {'vs 403 us':>10}"
    )
    print("-" * 100)
    for result in results:
        rate = OUTPUT_BYTES / (result.median_us * 1_000)
        speedup = REFERENCE_BASELINE_US / result.median_us
        print(
            f"{result.name:<45} {result.median_us:10.3f} "
            f"{result.p95_us:10.3f} {result.min_us:10.3f} "
            f"{rate:10.2f} {speedup:8.2f}x"
        )


if __name__ == "__main__":
    main()
