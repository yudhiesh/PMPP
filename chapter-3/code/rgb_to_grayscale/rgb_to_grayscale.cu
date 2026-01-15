#include <cassert>
#include <cuda_runtime.h>
#include <torch/types.h>

const int CHANNELS = 3;

__global__ void rgbToGrayscaleConversion(float *Pin, float *Pout, int width,
                                         int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;
    float r = Pin[rgbOffset];
    float g = Pin[rgbOffset + 1];
    float b = Pin[rgbOffset + 2];
    Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor rgb_to_grayscale(torch::Tensor matrix) {
  assert(matrix.device().type() == torch::kCUDA);
  assert(matrix.dtype() == torch::kFloat32);
  const auto height = matrix.size(0);
  const auto width = matrix.size(1);
  auto result = torch::empty({height, width}, matrix.options());
  const int THREADS = 16;
  dim3 threadsPerBlock(THREADS, THREADS);
  dim3 numberOfBlocks(cdiv(width, threadsPerBlock.x),
                      cdiv(height, threadsPerBlock.y));

  rgbToGrayscaleConversion<<<numberOfBlocks, threadsPerBlock>>>(
      matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
  return result;
}
