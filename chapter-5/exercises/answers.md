# CUDA Programming Exercises

## Exercise 1
Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

## Exercise 2
Draw the equivalent of Fig. 5.7 for a 8 × 8 matrix multiplication with 2 × 2 tiling and 4 × 4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

## Exercise 3
What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of Fig. 5.9?

## Exercise 4
Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

## Exercise 5
For our tiled matrix-matrix multiplication kernel, if we use a 32 × 32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?

## Exercise 6
Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

## Exercise 7
In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

## Exercise 8
Consider performing a matrix multiplication of two input matrices with dimensions N × N. How many times is each element in the input matrices requested from global memory when:

**a.** There is no tiling?

**b.** Tiles of size T × T are used?

## Exercise 9
A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.

**a.** Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second

**b.** Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second

## Exercise 10
To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.

```cuda
01  dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
02  dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
03  BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);
04
05  __global__ void
06  BlockTranspose(float* A_elements, int A_width, int A_height)
07  {
08      __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
09
10      int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
11      baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
12
13      blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
14
15      A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
16  }
```

**a.** Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?

**b.** If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

**e.** What is the amount of shared memory used per block (in bytes)?

**f.** What is the floating-point to global memory access ratio of the kernel (in OP/B)?

## Exercise 11
Consider the following CUDA kernel and the corresponding host function that calls it:

```cuda
01  __global__ void foo_kernel(float* a, float* b) {
02      unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03      float x[4];
04      __shared__ float y_s;
05      __shared__ float b_s[128];
06      for(unsigned int j = 0; j < 4; ++j) {
07          x[j] = a[i*blockDim.x*gridDim.x + j];
08      }
09      if(threadIdx.x == 0) {
10          y_s = 7.4f;
11      }
12      b_s[threadIdx.x] = b[i];
13      __syncthreads();
14      b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
15             + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)*128];
16  }
17
18  void foo(int* a_d, int* b_d) {
19      unsigned int N = 1024;
20      foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
21  }
```

**a.** How many versions of the variable `i` are there?

**b.** How many versions of the array `x[]` are there?

**c.** How many versions of the variable `y_s` are there?

**d.** How many versions of the array `b_s[]` are there?

## Exercise 12
Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

**a.** The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.

**b.** The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.
