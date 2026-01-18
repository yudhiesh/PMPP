# Exercises

## 1. Matrix Multiplication Kernel Variations

In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

**a.** Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.

**b.** Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.

**c.** Analyze the pros and cons of each of the two kernel designs.

| Design            | Reads from A             | Reads from B             | Writes to C   |
| ----------------- | ------------------------ | ------------------------ | ------------- |
| Row-per-thread    | Not coalesced            | Broadcast (same element) | Not coalesced |
| Column-per-thread | Broadcast (same element) | Coalesced                | Coalesced     |

The "broadcast" case is actually fine — when all threads read the same address, the GPU handles this efficiently.

**Why the massive performance differences:**

| Kernel           | Threads            | Work per thread        | GPU Utilization |
| ---------------- | ------------------ | ---------------------- | --------------- |
| pytorch (cuBLAS) | Optimized          | Tiled + shared mem     | Maximum         |
| naive            | 1,024 × 1,024 = 1M | 1 dot product (K muls) | Good            |
| row_per_thread   | 1,024              | N × K = 1M ops         | **Terrible**    |
| col_per_thread   | 1,024              | M × K = 1M ops         | **Terrible**    |

**Key insights:**

1. **row_per_thread and col_per_thread severely underutilize the GPU** - You're launching only 1,024 threads when modern GPUs can handle millions. Most SMs sit idle.

2. **Why col_per_thread is ~3x faster than row_per_thread** - Memory coalescing:

   ```
   row_per_thread: A[row * K + k]  → row varies across warp → strided reads (bad)
                   B[k * N + i]    → same for all threads → broadcast (ok)

   col_per_thread: A[i * K + k]    → same for all threads → broadcast (ok)
                   B[k * N + col]  → col varies across warp → coalesced reads (good!)
   ```

   When threads 0-31 in a warp read consecutive columns of B, that's a single coalesced 128-byte transaction. Strided reads require multiple transactions.

3. **naive is only 6x slower than cuBLAS** - Good parallelism (1M threads), but no shared memory tiling means repeated global memory reads.

**The takeaway:** These row/col-per-thread kernels are pedagogical examples showing why parallelism strategy matters enormously on GPUs. You want maximum threads with coalesced memory access patterns. The tiled shared-memory approach (which cuBLAS uses extensively) is the way to get close to peak performance.

---

## 2. Matrix-Vector Multiplication

A matrix-vector multiplication takes an input matrix **B** and a vector **C** and produces one output vector **A**. Each element of the output vector **A** is the dot product of one row of the input matrix **B** and **C**, that is:

$$A[i] = \sum_j B[i][j] \cdot C[j]$$

For simplicity we will handle only square matrices whose elements are single-precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters:

- Pointer to the output matrix
- Pointer to the input matrix
- Pointer to the input vector
- The number of elements in each dimension

Use one thread to calculate an output vector element.

---

## 3. CUDA Kernel Analysis

Consider the following CUDA kernel and the corresponding host function that calls it:

_[See image]_

**a.** What is the number of threads per block?

**b.** What is the number of threads in the grid?

**c.** What is the number of blocks in the grid?

**d.** What is the number of threads that execute the code on line 05?

---

## 4. 2D Matrix Indexing

Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at **row 20** and **column 10**:

**a.** If the matrix is stored in row-major order.

**b.** If the matrix is stored in column-major order.

---

## 5. 3D Tensor Indexing

Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at **x=10**, **y=20**, and **z=5**.
