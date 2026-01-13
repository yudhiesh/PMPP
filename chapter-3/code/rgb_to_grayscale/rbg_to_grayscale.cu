#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

const int CHANNELS = 3;

__global__ void colorToGrayscaleConversion(unsigned char *Pout,
                                           unsigned char *Pin, int width,
                                           int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;
    unsigned char r = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];
    Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}
