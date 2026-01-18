"""
CUDA Matrix Multiplication Kernel Profiler

Profile different matrix multiplication kernel implementations in CUDA via PyTorch.
Based on GPU MODE Lecture 1 profiling techniques.
"""

from pathlib import Path
from typing import Literal, Callable
import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile, ProfilerActivity
import argparse

CUDA_FILE_PATH = Path("matmul_kernels.cu")

CPP_SOURCE = """
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_row_per_thread(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_col_per_thread(torch::Tensor A, torch::Tensor B);
"""

SUPPORTED_KERNELS = Literal["naive", "row_per_thread", "col_per_thread", "pytorch"]
DEFAULT_KERNELS = ["pytorch", "naive", "row_per_thread", "col_per_thread"]


def time_cuda_function(
    func: Callable,
    *args,
    num_warmup: int = 5,
    num_iterations: int = 10,
    **kwargs,
) -> tuple[float, float, list[float]]:
    """
    Measure execution time of a CUDA function using torch.cuda.Event.

    Since CUDA is asynchronous, we can't use Python's time module.
    Instead, we use PyTorch's CUDA events to measure GPU time accurately.

    Returns:
        tuple: (mean_time_ms, std_time_ms, all_times)
    """
    # Warmup runs to initialize CUDA context and caches
    for _ in range(num_warmup):
        func(*args, **kwargs)

    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = func(*args, **kwargs)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    return mean_time, std_time, times


def profile_with_autograd(
    func: Callable,
    *args,
    kernel_name: str = "kernel",
    **kwargs,
) -> str:
    """
    Profile a function using torch.autograd.profiler.

    Returns a formatted table string showing CPU/CUDA times.
    """
    # Warmup
    for _ in range(3):
        func(*args, **kwargs)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        func(*args, **kwargs)

    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)


def profile_with_torch_profiler(
    func: Callable,
    *args,
    kernel_name: str = "kernel",
    output_trace: str | None = None,
    num_iterations: int = 5,
    **kwargs,
) -> str:
    """
    Profile using torch.profiler with optional Chrome trace export.

    The trace can be viewed at chrome://tracing/
    """
    # Warmup
    for _ in range(3):
        func(*args, **kwargs)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(num_iterations):
            func(*args, **kwargs)

    if output_trace:
        prof.export_chrome_trace(output_trace)
        print(f"Chrome trace exported to: {output_trace}")

    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)


def compile_cuda_kernels(cuda_file_path: Path, cpp_source: str, verbose: bool = True):
    """Compile and load the CUDA matmul kernels."""
    cuda_source = cuda_file_path.read_text()

    if verbose:
        print("Compiling CUDA kernels...")

    module = load_inline(
        name="matmul_kernels",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul_naive", "matmul_row_per_thread", "matmul_col_per_thread"],
        verbose=verbose,
        extra_cuda_cflags=["-O3"],
    )

    if verbose:
        print("Compilation complete!")

    return module


def create_test_matrices(
    M: int, N: int, K: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random test matrices A (M x K) and B (K x N) on GPU."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")
    return A, B


def verify_correctness(
    result: torch.Tensor,
    expected: torch.Tensor,
    kernel_name: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """Verify kernel output against PyTorch reference."""
    is_close = torch.allclose(result, expected, rtol=rtol, atol=atol)
    max_diff = (result - expected).abs().max().item()

    if is_close:
        print(f"  ✓ {kernel_name}: PASSED (max diff: {max_diff:.2e})")
    else:
        print(f"  ✗ {kernel_name}: FAILED (max diff: {max_diff:.2e})")

    return is_close


def get_kernel_func(module, kernel_name: str) -> Callable | None:
    """Get the kernel function by name."""
    kernel_map = {
        "pytorch": torch.matmul,
        "naive": module.matmul_naive,
        "row_per_thread": module.matmul_row_per_thread,
        "col_per_thread": module.matmul_col_per_thread,
    }
    return kernel_map.get(kernel_name)


def run_benchmark(
    module,
    A: torch.Tensor,
    B: torch.Tensor,
    kernels: list[str],
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> dict:
    """Run timing benchmark for specified kernels."""
    results = {}

    for kernel_name in kernels:
        func = get_kernel_func(module, kernel_name)
        if func is None:
            print(f"  Warning: Unknown kernel '{kernel_name}', skipping")
            continue

        mean_ms, std_ms, _ = time_cuda_function(
            func, A, B, num_warmup=num_warmup, num_iterations=num_iterations
        )
        results[kernel_name] = {"mean_ms": mean_ms, "std_ms": std_ms}
        print(f"  {kernel_name:20s}: {mean_ms:10.3f} ± {std_ms:.3f} ms")

    # Calculate speedups relative to PyTorch
    if "pytorch" in results:
        pytorch_time = results["pytorch"]["mean_ms"]
        for name, data in results.items():
            if name != "pytorch":
                speedup = pytorch_time / data["mean_ms"]
                results[name]["speedup_vs_pytorch"] = speedup

    return results


def main(
    cuda_file_path: Path,
    cpp_source: str,
    matrix_sizes: list[tuple[int, int, int]],
    kernels: list[str],
    profile_mode: str = "timing",
    export_trace: bool = False,
    num_warmup: int = 5,
    num_iterations: int = 20,
    verbose: bool = True,
):
    """
    Main profiling routine.

    Args:
        matrix_sizes: List of (M, N, K) tuples for matrix dimensions
        kernels: List of kernel names to benchmark
        profile_mode: 'timing', 'autograd', 'profiler', or 'all'
        export_trace: Whether to export Chrome traces
    """
    # Compile CUDA kernels
    module = compile_cuda_kernels(cuda_file_path, cpp_source, verbose=verbose)

    all_results = {}

    for M, N, K in matrix_sizes:
        print(f"\n{'=' * 60}")
        print(f"Matrix size: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
        print(f"{'=' * 60}")

        A, B = create_test_matrices(M, N, K)

        # Compute reference result
        reference = torch.matmul(A, B)

        # Verify correctness
        print("\nCorrectness check:")
        for kernel_name in kernels:
            if kernel_name == "pytorch":
                continue
            func = get_kernel_func(module, kernel_name)
            if func is None:
                continue
            result = func(A, B)
            verify_correctness(result, reference, kernel_name)

        # Timing benchmark
        if profile_mode in ("timing", "all"):
            print(
                f"\nTiming benchmark ({num_iterations} iterations, {num_warmup} warmup):"
            )
            results = run_benchmark(module, A, B, kernels, num_warmup, num_iterations)
            all_results[f"{M}x{N}x{K}"] = results

            # Print speedup summary
            if "pytorch" in results:
                print("\nSpeedup vs PyTorch:")
                for name, data in results.items():
                    if "speedup_vs_pytorch" in data:
                        speedup = data["speedup_vs_pytorch"]
                        faster_slower = "faster" if speedup > 1 else "slower"
                        print(f"  {name:20s}: {speedup:.3f}x ({faster_slower})")

        # Autograd profiler
        if profile_mode in ("autograd", "all"):
            print("\n--- Autograd Profiler ---")
            for kernel_name in kernels:
                func = get_kernel_func(module, kernel_name)
                if func is None:
                    continue
                print(f"\n[{kernel_name}]")
                table = profile_with_autograd(func, A, B)
                print(table)

        # Torch profiler with optional trace export
        if profile_mode in ("profiler", "all"):
            print("\n--- Torch Profiler ---")
            for kernel_name in kernels:
                func = get_kernel_func(module, kernel_name)
                if func is None:
                    continue
                print(f"\n[{kernel_name}]")
                trace_file = (
                    f"trace_{kernel_name}_{M}x{N}x{K}.json" if export_trace else None
                )
                table = profile_with_torch_profiler(func, A, B, output_trace=trace_file)
                print(table)

    # Summary
    if all_results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for size, results in all_results.items():
            print(f"\n{size}:")
            for kernel, data in results.items():
                line = f"  {kernel:20s}: {data['mean_ms']:.3f} ms"
                if "speedup_vs_pytorch" in data:
                    line += f" ({data['speedup_vs_pytorch']:.3f}x vs pytorch)"
                print(line)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CUDA matrix multiplication kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic timing comparison at 1024x1024
  python matmul_profile.py -s 1024

  # Multiple sizes
  python matmul_profile.py -s 512 1024 2048 4096

  # Full profiling with Chrome trace export
  python matmul_profile.py -s 1024 --profile all --trace

  # Specific M,N,K dimensions
  python matmul_profile.py --mnk 1024 512 768

  # Only compare specific kernels
  python matmul_profile.py -s 2048 -k naive row_per_thread pytorch

Available kernels:
  - pytorch        : PyTorch's native torch.matmul (cuBLAS)
  - naive          : One thread per output element
  - row_per_thread : One thread computes entire output row
  - col_per_thread : One thread computes entire output column
        """,
    )
    parser.add_argument(
        "-s",
        "--sizes",
        type=int,
        nargs="+",
        default=[1024],
        help="Matrix sizes (square matrices, default: 1024)",
    )
    parser.add_argument(
        "--mnk",
        type=int,
        nargs=3,
        action="append",
        metavar=("M", "N", "K"),
        help="Specific M N K dimensions (can be repeated)",
    )
    parser.add_argument(
        "-k",
        "--kernels",
        nargs="+",
        default=DEFAULT_KERNELS,
        help=f"Kernels to benchmark (default: {' '.join(DEFAULT_KERNELS)})",
    )
    parser.add_argument(
        "-p",
        "--profile",
        choices=["timing", "autograd", "profiler", "all"],
        default="timing",
        help="Profiling mode (default: timing)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Export Chrome traces (use with --profile profiler or all)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build matrix sizes list
    matrix_sizes = []

    # Square matrices from -s/--sizes
    for s in args.sizes:
        matrix_sizes.append((s, s, s))

    # Custom M,N,K from --mnk
    if args.mnk:
        for mnk in args.mnk:
            matrix_sizes.append(tuple(mnk))

    main(
        cuda_file_path=CUDA_FILE_PATH,
        cpp_source=CPP_SOURCE,
        matrix_sizes=matrix_sizes,
        kernels=args.kernels,
        profile_mode=args.profile,
        export_trace=args.trace,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        verbose=not args.quiet,
    )
