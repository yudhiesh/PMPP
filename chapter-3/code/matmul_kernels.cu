/*
 * CUDA Matrix Multiplication Kernels
 *
 * Contains multiple implementations for benchmarking:
 * - naive: Basic element-per-thread approach
 * - row_per_thread: Each thread computes one row of output
 * - col_per_thread: Each thread computes one column of output
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

// =============================================================================
// Naive Matrix Multiplication Kernel
// Each thread computes one element of the output matrix C
// C[i,j] = sum_k(A[i,k] * B[k,j])
// =============================================================================
__global__ void matmul_naive_kernel(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C, int M, int N,
                                    int K) {
  // Calculate output position
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// =============================================================================
// Row-per-Thread Matrix Multiplication Kernel
// Each thread computes one entire row of the output matrix C
// For matrix multiplication C(MxN) = A(MxK) × B(KxN):
// * A is M × K
// * B is K × N
// * C is M × N
// =============================================================================
__global__ void matmul_row_per_thread_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C, int M,
                                             int N, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    for (int i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + i];
      }
      C[row * N + i] = sum;
    }
  }
}

// =============================================================================
// Column-per-Thread Matrix Multiplication Kernel
// Each thread computes one entire column of the output matrix C
// =============================================================================
__global__ void matmul_col_per_thread_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C, int M,
                                             int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < N) {
    for (int i = 0; i < M; ++i) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + col];
      }
      C[i * N + col] = sum;
    }
  }
}

// =============================================================================
// Wrapper Functions
// =============================================================================

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
  // A: (M, K), B: (K, N) -> C: (M, N)
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::empty({M, N}, A.options());

  dim3 threads_per_block(16, 16);
  dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                  (M + threads_per_block.y - 1) / threads_per_block.y);

  matmul_naive_kernel<<<num_blocks, threads_per_block>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  return C;
}

torch::Tensor matmul_row_per_thread(torch::Tensor A, torch::Tensor B) {
  // A: (M, K), B: (K, N) -> C: (M, N)
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::empty({M, N}, A.options());

  // 1D grid: one thread per row
  const int threads_per_block = 256;
  const int num_blocks = (M + threads_per_block - 1) / threads_per_block;

  matmul_row_per_thread_kernel<<<num_blocks, threads_per_block>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  return C;
}

torch::Tensor matmul_col_per_thread(torch::Tensor A, torch::Tensor B) {
  // A: (M, K), B: (K, N) -> C: (M, N)
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::empty({M, N}, A.options());

  // 1D grid: one thread per column
  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  matmul_col_per_thread_kernel<<<num_blocks, threads_per_block>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  return C;
}
