from pathlib import Path

import pytest
import torch

from matmul_profile import compile_cuda_kernels


@pytest.fixture(scope="session")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    return compile_cuda_kernels(
        cuda_file_path=Path("matmul_kernels.cu"),
        cpp_source=(
            "torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);\n"
            "torch::Tensor matmul_row_per_thread(torch::Tensor A, torch::Tensor B);\n"
            "torch::Tensor matmul_col_per_thread(torch::Tensor A, torch::Tensor B);\n"
            "torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);\n"
            "torch::Tensor matmul_tiled_8(torch::Tensor A, torch::Tensor B);\n"
            "torch::Tensor matmul_tiled_16(torch::Tensor A, torch::Tensor B);\n"
            "torch::Tensor matmul_tiled_32(torch::Tensor A, torch::Tensor B);\n"
        ),
        verbose=False,
    )

def _run_case(module, M: int, N: int, K: int, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    reference = torch.matmul(A, B)

    for name in ("matmul_naive", "matmul_row_per_thread", "matmul_col_per_thread", "matmul_tiled"):
        kernel = getattr(module, name)
        result = kernel(A, B)
        assert torch.allclose(result, reference, rtol=rtol, atol=atol), (
            f"{name} failed for {M}x{K} @ {K}x{N}"
        )

def test_square_256(cuda_module) -> None:
    _run_case(cuda_module, 256, 256, 256)

def test_rectangular(cuda_module) -> None:
    _run_case(cuda_module, 128, 96, 64)

def test_non_multiple_of_tile(cuda_module) -> None:
    # Exercise boundary handling in tiled kernel (TILE_WIDTH=16).
    _run_case(cuda_module, 17, 19, 23)
