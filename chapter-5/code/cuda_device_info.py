#!/usr/bin/env python3
"""
Print CUDA device properties and a tile-width recommendation for tiled matmul.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys

import torch


def _get_attr(obj, name: str, default="N/A"):
    return getattr(obj, name, default)


def _load_cudart():
    for lib in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    return None


def _cuda_attr(device: int, attr_id: int):
    cudart = _load_cudart()
    if cudart is None:
        return "N/A"

    # int cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
    fn = cudart.cudaDeviceGetAttribute
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int

    out = ctypes.c_int()
    err = fn(ctypes.byref(out), attr_id, device)
    if err != 0:
        return "N/A"
    return int(out.value)


def calculate_optimal_tile_width(props, m: int, n: int, k: int) -> int:
    max_threads_per_block = int(_get_attr(props, "max_threads_per_block", 1024))
    max_threads_dim = _get_attr(props, "max_threads_dim", (1024, 1024, 64))
    shared_mem_per_block = int(_get_attr(props, "shared_memory_per_block", 0))

    by_threads = int(math.sqrt(max_threads_per_block))
    by_block_dim = min(int(max_threads_dim[0]), int(max_threads_dim[1]))
    by_shared_mem = (
        int(math.sqrt(shared_mem_per_block / (2 * 4))) if shared_mem_per_block > 0 else 1
    )
    by_problem = min(m, n, k)

    max_valid = max(1, min(by_threads, by_block_dim, by_shared_mem, by_problem, 32))

    tile = 1
    while (tile << 1) <= max_valid:
        tile <<= 1
    return tile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show CUDA device properties and recommended tiled-matmul width."
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--mnk",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        default=(1024, 1024, 1024),
        help="Matrix dimensions used for tile recommendation (default: 1024 1024 1024)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return 1

    device_count = torch.cuda.device_count()
    if args.device < 0 or args.device >= device_count:
        print(f"Invalid --device {args.device}. Available range: [0, {device_count - 1}]")
        return 1

    props = torch.cuda.get_device_properties(args.device)
    m, n, k = args.mnk
    tile = calculate_optimal_tile_width(props, m, n, k)

    # CUDA runtime attributes (fallback for fields not exposed by torch props).
    max_threads_per_block = _cuda_attr(args.device, 1)
    max_block_dim_x = _cuda_attr(args.device, 2)
    max_block_dim_y = _cuda_attr(args.device, 3)
    max_block_dim_z = _cuda_attr(args.device, 4)
    max_grid_dim_x = _cuda_attr(args.device, 5)
    max_grid_dim_y = _cuda_attr(args.device, 6)
    max_grid_dim_z = _cuda_attr(args.device, 7)
    shared_mem_per_block = _cuda_attr(args.device, 8)
    regs_per_block = _cuda_attr(args.device, 12)
    clock_rate_khz = _cuda_attr(args.device, 13)
    memory_clock_rate_khz = _cuda_attr(args.device, 36)
    memory_bus_width_bits = _cuda_attr(args.device, 37)
    l2_cache_size_bytes = _cuda_attr(args.device, 38)
    shared_mem_per_sm = _cuda_attr(args.device, 81)
    regs_per_sm = _cuda_attr(args.device, 82)

    # Prefer CUDA-runtime values when available; fallback to torch props.
    max_threads_per_block = (
        max_threads_per_block
        if max_threads_per_block != "N/A"
        else _get_attr(props, "max_threads_per_block")
    )
    max_threads_dim = (
        (max_block_dim_x, max_block_dim_y, max_block_dim_z)
        if max_block_dim_x != "N/A"
        else _get_attr(props, "max_threads_dim")
    )
    max_grid_size = (
        (max_grid_dim_x, max_grid_dim_y, max_grid_dim_z)
        if max_grid_dim_x != "N/A"
        else _get_attr(props, "max_grid_size")
    )
    shared_mem_per_block = (
        shared_mem_per_block
        if shared_mem_per_block != "N/A"
        else _get_attr(props, "shared_memory_per_block")
    )
    shared_mem_per_sm = (
        shared_mem_per_sm
        if shared_mem_per_sm != "N/A"
        else _get_attr(props, "shared_memory_per_multiprocessor")
    )
    regs_per_block = (
        regs_per_block if regs_per_block != "N/A" else _get_attr(props, "regs_per_block")
    )
    regs_per_sm = (
        regs_per_sm
        if regs_per_sm != "N/A"
        else _get_attr(props, "regs_per_multiprocessor")
    )
    memory_clock_rate_khz = (
        memory_clock_rate_khz
        if memory_clock_rate_khz != "N/A"
        else _get_attr(props, "memory_clock_rate")
    )
    memory_bus_width_bits = (
        memory_bus_width_bits
        if memory_bus_width_bits != "N/A"
        else _get_attr(props, "memory_bus_width")
    )
    l2_cache_size_bytes = (
        l2_cache_size_bytes
        if l2_cache_size_bytes != "N/A"
        else _get_attr(props, "l2_cache_size")
    )

    print(f"CUDA devices: {device_count}")
    print(f"Selected device: {args.device}")
    print()
    print(f"name: {_get_attr(props, 'name')}")
    print(f"compute capability: {_get_attr(props, 'major')}.{_get_attr(props, 'minor')}")
    print(f"total memory (bytes): {_get_attr(props, 'total_memory')}")
    print(f"SM count: {_get_attr(props, 'multi_processor_count')}")
    print(f"max threads per block: {max_threads_per_block}")
    print(f"max threads dim: {max_threads_dim}")
    print(f"max grid size: {max_grid_size}")
    print(f"warp size: {_get_attr(props, 'warp_size')}")
    print(f"shared memory per block (bytes): {shared_mem_per_block}")
    print(f"shared memory per multiprocessor (bytes): {shared_mem_per_sm}")
    print(f"regs per block: {regs_per_block}")
    print(f"regs per multiprocessor: {regs_per_sm}")
    print(f"clock rate (kHz): {clock_rate_khz}")
    print(f"memory clock rate (kHz): {memory_clock_rate_khz}")
    print(f"memory bus width (bits): {memory_bus_width_bits}")
    print(f"L2 cache size (bytes): {l2_cache_size_bytes}")
    print()
    print(f"Recommended tile width for MxNxK={m}x{n}x{k}: {tile}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
