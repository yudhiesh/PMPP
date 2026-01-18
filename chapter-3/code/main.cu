#include <cuda_runtime.h>
#include <torch/extension.h>

enum class KernelType { GRAYSCALE = 0, BLUR = 1 };

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__global__ void rgbToGrayscaleConversion(float *Pin, float *Pout, int width,
                                         int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int idx = (row * width + col) * 3;
    float r = Pin[idx];
    float g = Pin[idx + 1];
    float b = Pin[idx + 2];
    Pout[row * width + col] = 0.299f * r + 0.587f * g + 0.114f * b;
  }
}

__global__ void blurKernel(float *Pin, float *Pout, int width, int height,
                           int blurRadius) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    // Process each RGB channel
    for (int channel = 0; channel < 3; ++channel) {
      float pixVal = 0.0f;
      int pixels = 0;

      for (int blurRow = -blurRadius; blurRow < blurRadius + 1; ++blurRow) {
        for (int blurCol = -blurRadius; blurCol < blurRadius + 1; ++blurCol) {
          int curRow = row + blurRow;
          int curCol = col + blurCol;

          if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
            // pixel_number = (row * width + col)
            // memory position = pixel_number * 3
            // channel_offset = c
            pixVal += Pin[(curRow * width + curCol) * 3 + channel];
            ++pixels;
          }
        }
      }

      Pout[(row * width + col) * 3 + channel] = pixVal / pixels;
    }
  }
}

torch::Tensor process_image(torch::Tensor matrix,
                            const std::string &kernel_name, int param) {
  TORCH_CHECK(matrix.device().is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(matrix.dtype() == torch::kFloat32, "Input must be float32");

  const auto height = matrix.size(0);
  const auto width = matrix.size(1);

  const int THREADS = 16;
  dim3 threadsPerBlock(THREADS, THREADS);
  dim3 numberOfBlocks(cdiv(width, threadsPerBlock.x),
                      cdiv(height, threadsPerBlock.y));

  if (kernel_name == "grayscale") {
    auto result = torch::empty({height, width}, matrix.options());
    rgbToGrayscaleConversion<<<numberOfBlocks, threadsPerBlock>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
    return result;
  } else if (kernel_name == "blur") {
    auto result = torch::empty_like(matrix);
    blurKernel<<<numberOfBlocks, threadsPerBlock>>>(matrix.data_ptr<float>(),
                                                    result.data_ptr<float>(),
                                                    width, height, param);
    return result;
  }

  TORCH_CHECK(false, "Unknown kernel: " + kernel_name);
  return matrix; // Unreachable, but satisfies compiler
}
