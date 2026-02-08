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
#include <algorithm>
#include <cmath>
#include <thread>
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

// Each thread block computes one TILE x TILE output tile.
template <int TILE>
__global__ void matmul_tiled_kernel(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C, int M, int N,
                                    int K, size_t ads_size_bytes,
                                    size_t bds_size_bytes) {
  (void)bds_size_bytes;
  extern __shared__ float shared_mem[];
  float *Ads = shared_mem;
  float *Bds = reinterpret_cast<float *>(
      reinterpret_cast<char *>(shared_mem) + ads_size_bytes);

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE + threadIdx.y;
  int Col = bx * TILE + threadIdx.x;

  float Pvalue = 0.0f;
  const int num_phases = (K + TILE - 1) / TILE;
  for (int phase = 0; phase < num_phases; ++phase) {
    const int a_col = phase * TILE + tx;
    const int b_row = phase * TILE + ty;
    const int idx = ty * TILE + tx;

    Ads[idx] = (Row < M && a_col < K) ? A[Row * K + a_col] : 0.0f;
    Bds[idx] = (b_row < K && Col < N) ? B[b_row * N + Col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      Pvalue += Ads[ty * TILE + k] * Bds[k * TILE + tx];
    }
    __syncthreads();
  }
  if (Row < M && Col < N) {
    C[Row * N + Col] = Pvalue;
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

template <int TILE>
torch::Tensor matmul_tiled_impl(torch::Tensor A, torch::Tensor B) {
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

  dim3 threads_per_block(TILE, TILE);
  dim3 num_blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  const size_t ads_size_bytes = TILE * TILE * sizeof(float);
  const size_t bds_size_bytes = TILE * TILE * sizeof(float);
  const size_t shared_mem_size_bytes = ads_size_bytes + bds_size_bytes;

  matmul_tiled_kernel<TILE><<<num_blocks, threads_per_block,
                              shared_mem_size_bytes>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K,
      ads_size_bytes, bds_size_bytes);

  return C;
}

int calculate_optimal_tile_width(int M, int N, int K) {
  TORCH_CHECK(M > 0 && N > 0 && K > 0, "Matrix dimensions must be positive");

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ",
              cudaGetErrorString(err));

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  TORCH_CHECK(err == cudaSuccess, "cudaGetDeviceProperties failed: ",
              cudaGetErrorString(err));

  const int by_threads =
      static_cast<int>(std::floor(std::sqrt(prop.maxThreadsPerBlock)));
  const int by_block_dim =
      std::min(prop.maxThreadsDim[0], prop.maxThreadsDim[1]);
  const int by_shared_mem = static_cast<int>(std::floor(std::sqrt(
      static_cast<double>(prop.sharedMemPerBlock) / (2.0 * sizeof(float)))));
  const int by_problem = std::min({M, N, K});

  int max_valid = std::min({by_threads, by_block_dim, by_shared_mem, by_problem,
                            32}); // dispatcher supports TILE in [1, 32]
  max_valid = std::max(1, max_valid);

  // Choose the largest power-of-two tile width <= max_valid.
  int tile_width = 1;
  while ((tile_width << 1) <= max_valid) {
    tile_width <<= 1;
  }

  // Prefer 16x16 when valid (good baseline for many GPUs).
  if (max_valid >= 16) {
    tile_width = std::max(tile_width, 16);
  }

  return tile_width;
}

torch::Tensor matmul_tiled_dynamic(torch::Tensor A, torch::Tensor B,
                                   int tile_width);

// Default tiled kernel (backwards compatible).
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  const int tile_width = calculate_optimal_tile_width(M, N, K);
  return matmul_tiled_dynamic(A, B, tile_width);
}

torch::Tensor matmul_tiled_auto(torch::Tensor A, torch::Tensor B) {
  return matmul_tiled(A, B);
}

// Explicit tiled variants for runtime selection.
torch::Tensor matmul_tiled_8(torch::Tensor A, torch::Tensor B) {
  return matmul_tiled_impl<8>(A, B);
}

torch::Tensor matmul_tiled_16(torch::Tensor A, torch::Tensor B) {
  return matmul_tiled_impl<16>(A, B);
}

torch::Tensor matmul_tiled_32(torch::Tensor A, torch::Tensor B) {
  return matmul_tiled_impl<32>(A, B);
}

torch::Tensor matmul_tiled_dynamic(torch::Tensor A, torch::Tensor B,
                                   int tile_width) {
  TORCH_CHECK(tile_width > 0, "tile_width must be > 0");
  TORCH_CHECK(tile_width <= 32,
              "tile_width must be <= 32 to keep tile_width^2 <= 1024 threads");

  switch (tile_width) {
  case 1:
    return matmul_tiled_impl<1>(A, B);
  case 2:
    return matmul_tiled_impl<2>(A, B);
  case 3:
    return matmul_tiled_impl<3>(A, B);
  case 4:
    return matmul_tiled_impl<4>(A, B);
  case 5:
    return matmul_tiled_impl<5>(A, B);
  case 6:
    return matmul_tiled_impl<6>(A, B);
  case 7:
    return matmul_tiled_impl<7>(A, B);
  case 8:
    return matmul_tiled_impl<8>(A, B);
  case 9:
    return matmul_tiled_impl<9>(A, B);
  case 10:
    return matmul_tiled_impl<10>(A, B);
  case 11:
    return matmul_tiled_impl<11>(A, B);
  case 12:
    return matmul_tiled_impl<12>(A, B);
  case 13:
    return matmul_tiled_impl<13>(A, B);
  case 14:
    return matmul_tiled_impl<14>(A, B);
  case 15:
    return matmul_tiled_impl<15>(A, B);
  case 16:
    return matmul_tiled_impl<16>(A, B);
  case 17:
    return matmul_tiled_impl<17>(A, B);
  case 18:
    return matmul_tiled_impl<18>(A, B);
  case 19:
    return matmul_tiled_impl<19>(A, B);
  case 20:
    return matmul_tiled_impl<20>(A, B);
  case 21:
    return matmul_tiled_impl<21>(A, B);
  case 22:
    return matmul_tiled_impl<22>(A, B);
  case 23:
    return matmul_tiled_impl<23>(A, B);
  case 24:
    return matmul_tiled_impl<24>(A, B);
  case 25:
    return matmul_tiled_impl<25>(A, B);
  case 26:
    return matmul_tiled_impl<26>(A, B);
  case 27:
    return matmul_tiled_impl<27>(A, B);
  case 28:
    return matmul_tiled_impl<28>(A, B);
  case 29:
    return matmul_tiled_impl<29>(A, B);
  case 30:
    return matmul_tiled_impl<30>(A, B);
  case 31:
    return matmul_tiled_impl<31>(A, B);
  case 32:
    return matmul_tiled_impl<32>(A, B);
  default:
    TORCH_CHECK(false, "Unsupported tile_width: ", tile_width);
  }
}
