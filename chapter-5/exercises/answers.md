# CUDA Programming Exercises

## Exercise 1

Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

**Solution**:

Shared memory provides benefit only when there is data reuse between threads (reuse ratio > 1). In matrix addition, since each element is accessed by exactly one thread, there is no commonality in data access patterns, and shared memory adds overhead without reducing global memory bandwidth.

Arithmetic intensity: Matrix addition has 1 FLOP per 12 bytes (assuming FP32), making it highly memory-bound. Shared memory cannot improve this because the bottleneck is the total amount of data that must be moved from global memory, not the latency of individual accesses.

## Exercise 2

**Solution**:

### 2×2 Tiling Configuration

For an 8×8 matrix multiplication C = A × B using 2×2 tiles:

**Tile Organization**:

- Number of tiles along each dimension: 8÷2 = 4
- Total number of tiles: 4×4 = 16 tiles

**Data Requirements per Tile**:
To compute one 2×2 output tile of C:

- From A: 2×8 slice (2 rows, all 8 columns) = 16 elements
- From B: 8×2 slice (all 8 rows, 2 columns) = 16 elements
- Total per tile: 32 elements

**Total Memory Traffic**:

- 16 tiles × 32 elements/tile = **512 elements**

### 4×4 Tiling Configuration

For an 8×8 matrix multiplication using 4×4 tiles:

**Tile Organization**:

- Number of tiles along each dimension: 8÷4 = 2
- Total number of tiles: 2×2 = 4 tiles

**Data Requirements per Tile**:
To compute one 4×4 output tile of C:

- From A: 4×8 slice (4 rows, all 8 columns) = 32 elements
- From B: 8×4 slice (all 8 rows, 4 columns) = 32 elements
- Total per tile: 64 elements

**Total Memory Traffic**:

- 4 tiles × 64 elements/tile = **256 elements**

### Baseline (No Tiling)

Without tiling:

- 64 output elements (8×8)
- Each element requires: 8 elements from A + 8 elements from B = 16 elements
- Total memory traffic: 64 × 16 = **1024 elements**

### Bandwidth Reduction Verification

| Configuration | Total Elements | Reduction Factor | Tile Dimension (T) |
| ------------- | -------------- | ---------------- | ------------------ |
| No tiling     | 1024           | 1× (baseline)    | -                  |
| 2×2 tiling    | 512            | 2×               | 2                  |
| 4×4 tiling    | 256            | 4×               | 4                  |

**Verification**: The reduction factor equals the tile dimension in both cases:

- 2×2 tiling: 1024 ÷ 512 = 2 (matches tile dimension of 2)
- 4×4 tiling: 1024 ÷ 256 = 4 (matches tile dimension of 4)

**Conclusion**: The reduction in global memory bandwidth is indeed **proportional to the tile dimension size**. For an N×N matrix with T×T tiles, each input element is reused T times, resulting in a bandwidth reduction factor of T.

Draw the equivalent of Fig. 5.7 for a 8 × 8 matrix multiplication with 2 × 2 tiling and 4 × 4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

## Exercise 3

What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of Fig. 5.9?

**Solution**:

✓ Both cause silent failures with wrong numerical results (not crashes)

✓ First barrier prevents RAW (read-after-write) hazards - threads reading shared memory before other threads finish writing to it

✓ Second barrier prevents WAR (write-after-read) hazards - threads overwriting shared memory before other threads finish reading from it

The key insight: `syncthreads()` ensures all threads reach the same point before any proceed. It's a coordination mechanism that prevents these race conditions.

## Exercise 4

Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

**Solution**:

Registers are private to each thread, while shared memory is visible to all threads within a block.
This difference in scope makes shared memory essential for algorithms that require inter-thread communication and data sharing.
Why this matters:
When a value is stored in a thread's registers, no other thread can access it. However, when a value is stored in shared memory, all threads in the block can read and write to it.

Example - Tiled Matrix Multiplication:
In tiled matrix multiplication, each thread loads one element from global memory into shared memory. Then, all threads in the block must access the entire tile to compute their outputs. For instance:

Thread 0 loads Mds[0][0] from global memory
Thread 15 needs to read Mds[0][0] to compute its dot product
This is only possible because the data is in shared memory

If the data were stored in Thread 0's registers, Thread 15 would have no way to access it, making collaborative algorithms like tiling impossible to implement.
Conclusion: Shared memory enables threads within a block to collaborate by sharing data, which is essential for many parallel algorithms including tiling, reductions, and scans.

## Exercise 5

For our tiled matrix-matrix multiplication kernel, if we use a 32 × 32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?

**Solution**:

From the general tiling principle: the bandwidth reduction factor equals the tile dimension (T).
For a T×T tile:

Without tiling: Each element from M and N is read N times (once per output element in its row/column)
With T×T tiling: Each element is read N/T times (once per tile in its row/column)
Reduction factor = N/(N/T) = T

For 32×32 tiles:

Each element of M and N is loaded once into shared memory per tile
That element is then reused 32 times by the threads computing the output tile
Reduction factor = 32×

This means that with 32×32 tiling, the global memory bandwidth required is reduced to 1/32 of the bandwidth needed without tiling.

## Exercise 6

Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

**Solution**:

Local variables are indeed private to each thread - they're stored in registers (or spilled to local memory if you run out of registers), and each thread operates on its own independent copy.
So the calculation is:

1000 thread blocks × 512 threads per block = 512,000 versions

Each of those 512,000 threads gets its own version of the local variable throughout the kernel's lifetime.

## Exercise 7

In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

**Solution**:

Shared memory is shared across all threads within a thread block therefore we would be have threadBlockDim numbers of copies

## Exercise 8

Consider performing a matrix multiplication of two input matrices with dimensions N × N. How many times is each element in the input matrices requested from global memory when:

**a.** There is no tiling?

**Solution**:

Without tiling: N reads per element

**b.** Tiles of size T × T are used?

**Solution**:

With tiling: N/T reads per element
Reduction factor: T

## Exercise 9

A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.

**a.** Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second

**Solution**:

Arithmetic Intensity: 36 FLOPs / (7 × 4 bytes) = 1.286 FLOPS/byte

Memory can sustain: 100 GB/s × 1.286 = 128.6 GFLOPS
Compute capability: 200 GFLOPS
Memory-bound (limited to 128.6 GFLOPS)

**b.** Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second

**Solution**:

Memory can sustain: 250 GB/s × 1.286 = 321.5 GFLOPS
Compute capability: 300 GFLOPS
Compute-bound (limited to 300 GFLOPS)

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

**Solution**:

When BLOCK_WIDTH = 1:

The block contains exactly 1 thread (1×1)
That single thread executes line 13, writes to blockA[0][0]
Then the same thread executes line 15, reads from blockA[0][0]
No race condition possible - a thread can't race against itself!
It's sequential execution within that thread

For BLOCK_WIDTH ≥ 2:

Multiple threads in the block create the RAW hazard you identified
The code breaks due to missing synchronization

**b.** If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

**Solution**:

```cuda
13  blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
14  __syncthreads();
15  A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
```

Eliminate the read-after-write hazard where when we try to swap in the index around we ensure that all threads have set their values before the next step to read from the same place.

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

**Solution**:

**Grid Configuration:**

- Blocks: (1024 + 128 - 1) / 128 = 9 blocks
- Threads per block: 128
- Total threads: 9 × 128 = 1,152 threads

**a. Versions of variable `i`:** **1,152**

- One per thread (local variable, no `__shared__`)

**b. Versions of array `x[]`:** **1,152**

- One per thread (local array stored in registers/local memory)

**c. Versions of variable `y_s`:** **9**

- One per block (`__shared__` qualifier)

**d. Versions of array `b_s[]`:** **9**

- One per block (`__shared__` qualifier)

**e. Shared memory per block:** **516 bytes**

- `y_s`: 1 float × 4 bytes = 4 bytes
- `b_s[128]`: 128 floats × 4 bytes = 512 bytes
- Total: 4 + 512 = 516 bytes

**f. OP/B ratio:** **0.417 OP/B**

- **FLOPs:** 5 multiplies + 5 adds = 10 FLOPs per thread
- **Global memory bytes:**
  - Lines 6-8: 4 reads from `a[]` = 16 bytes
  - Line 12: 1 read from `b[]` = 4 bytes
  - Line 14: 1 write to `b[]` = 4 bytes
  - Total: 24 bytes
- **Ratio:** 10 / 24 = 0.417 OP/B

**Bug Found:** Line 15 has `b_s[(threadIdx.x + 3)*128]` which computes indices 384, 512, 640, etc. - far beyond the declared size of 128. This causes out-of-bounds memory access.

## Exercise 12

Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

**a.** The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.

**b.** The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.

**Solution**:

**Hardware Limits:**

- 2048 threads/SM
- 32 blocks/SM
- 65,536 registers/SM
- 96 KB (98,304 bytes) shared memory/SM

## Part (a): 64 threads/block, 27 registers/thread, 4 KB shared memory/block

**Constraint Analysis:**

1. **Thread constraint:**
   - Max blocks = ⌊2048 / 64⌋ = **32 blocks**

2. **Block constraint:**
   - Hardware limit = **32 blocks**

3. **Register constraint:**
   - Registers per block = 64 × 27 = 1,728 registers
   - Max blocks = ⌊65,536 / 1,728⌋ = **37 blocks**

4. **Shared memory constraint:**
   - Per block = 4 KB = 4,096 bytes
   - Max blocks = ⌊98,304 / 4,096⌋ = **24 blocks**

**Answer:**

- **Limiting factor:** Shared memory (24 blocks)
- **Full occupancy?** **No** - can only achieve 24 blocks instead of 32

## Part (b): 256 threads/block, 31 registers/thread, 8 KB shared memory/block

**Constraint Analysis:**

1. **Thread constraint:**
   - Max blocks = ⌊2048 / 256⌋ = **8 blocks**

2. **Block constraint:**
   - Hardware limit = **32 blocks**

3. **Register constraint:**
   - Registers per block = 256 × 31 = 7,936 registers
   - Max blocks = ⌊65,536 / 7,936⌋ = **8 blocks**

4. **Shared memory constraint:**
   - Per block = 8 KB = 8,192 bytes
   - Max blocks = ⌊98,304 / 8,192⌋ = **12 blocks**

**Answer:**

- **Limiting factors:** Thread constraint AND Register constraint (both at 8 blocks)
- **Full occupancy?** **No** - can only achieve 8 blocks instead of 32
