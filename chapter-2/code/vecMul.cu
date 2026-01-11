#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "vecMul.h"

#define cudaCheckError(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;

  cudaCheckError(cudaMalloc((void **)&A_d, size));
  cudaCheckError(cudaMalloc((void **)&B_d, size));
  cudaCheckError(cudaMalloc((void **)&C_d, size));

  cudaCheckError(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (int)ceil(n / (float)threadsPerBlock);

  vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());

  cudaCheckError(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

  cudaCheckError(cudaFree(A_d));
  cudaCheckError(cudaFree(B_d));
  cudaCheckError(cudaFree(C_d));
}
