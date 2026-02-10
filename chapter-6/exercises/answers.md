## Exercise 1: Shared Memory for Matrix Addition

**Question:** Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption?

**Solution:**

Shared memory provides benefit only when there is **data reuse** between threads (reuse ratio > 1). In matrix addition, each element of the input matrices is accessed by exactly **one thread** — there is no commonality in data access patterns between threads. Loading data into shared memory would simply add overhead (the shared memory write + read) without reducing any global memory traffic, since each value is used once and then discarded.

**Arithmetic intensity analysis:** Matrix addition performs 1 FLOP (one addition) per output element, requiring 3 memory accesses (2 reads + 1 write) × 4 bytes = 12 bytes of global memory traffic. This gives an arithmetic intensity of **1/12 ≈ 0.083 OP/B**, making it deeply memory-bound. Shared memory cannot improve this because the bottleneck is the *total volume* of data that must move from global memory, not the latency or redundancy of individual accesses.

**Key insight:** Shared memory is only useful when multiple threads need the same data. In matrix addition, the access pattern is strictly 1:1 — each element maps to exactly one thread. Contrast this with matrix multiplication, where each element participates in an entire row or column of dot products, creating massive reuse opportunities.

---

## Exercise 2: BLOCK_SIZE Values for Fully Coalesced Access in Tiled Matrix Multiplication

**Question:** For tiled matrix multiplication, of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely avoid uncoalesced accesses to global memory? (Consider only square blocks.)

**Solution:**

**BLOCK_SIZE must be a multiple of the warp size (32).**

To understand why, recall how the tiled matmul kernel loads tiles from global memory. From the kernel in Figure 6.11 (replicated from Figure 5.7):

**Loading Md (Line 9):** Each thread loads `Md[Row * Width + m*TILE_WIDTH + tx]`, where `tx = threadIdx.x`. For threads in the same warp with consecutive `threadIdx.x` values, they access consecutive memory locations — this is coalesced.

**Loading Nd (Line 10):** Each thread loads `Nd[(m*TILE_WIDTH + ty) * Width + Col]`, where `Col = bx*TILE_WIDTH + tx`. Again, the only term varying across threads in the same warp is `tx`, so consecutive `threadIdx.x` → consecutive addresses → coalesced.

**The critical requirement:** Threads are linearized into warps as `linear_id = threadIdx.y * blockDim.x + threadIdx.x`. A warp consists of 32 consecutive threads by linear ID. For all threads in a warp to have consecutive `threadIdx.x` values, the block's x-dimension (BLOCK_SIZE) must be a **multiple of 32**.

**What goes wrong if BLOCK_SIZE is not a multiple of 32:**

Consider BLOCK_SIZE = 16. Thread linearization gives:
- Threads 0–15: `(ty=0, tx=0..15)` — Row 0 of the block
- Threads 16–31: `(ty=1, tx=0..15)` — Row 1 of the block

A single warp of 32 threads spans **two rows** of the thread block. When loading Md, threads 0–15 access `Md[Row_0 * Width + ...]` while threads 16–31 access `Md[Row_1 * Width + ...]`. These addresses are `Width` elements apart — **not consecutive, therefore uncoalesced**.

**Valid BLOCK_SIZE values:** Since we need square blocks and BLOCK_SIZE must be a multiple of 32, the candidate is **BLOCK_SIZE = 32** (giving 32 × 32 = 1024 threads per block). Larger multiples like 64 would give 4096 threads per block, exceeding hardware limits. Note that on the G80/GT200, the maximum threads per block was 512, so even BLOCK_SIZE = 32 exceeded the limit on those architectures. On modern GPUs with 1024 threads per block limit, BLOCK_SIZE = 32 is the unique valid choice.

**Practical note:** This doesn't mean BLOCK_SIZE = 16 performs terribly — modern GPU memory systems (Compute Capability 1.2+) handle misaligned and strided accesses more gracefully with segment-based transactions. But BLOCK_SIZE = 32 is the only value that guarantees **complete** coalescing.

---

## Exercise 3: Coalesced vs. Uncoalesced Memory Access Analysis

**Question:** Consider the following CUDA kernel and determine whether each memory access is coalesced, uncoalesced, or not applicable (shared memory).

```cuda
__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // Line 02
    __shared__ float a_s[256];                                 // Line 03
    __shared__ float bc_s[4 * 256];                            // Line 04
    a_s[threadIdx.x] = a[i];                                   // Line 05
    for (unsigned int j = 0; j < 4; ++j) {                     // Line 06
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];  // Line 07
    }                                                           // Line 08
    __syncthreads();                                            // Line 09
    d[i + 8] = a_s[threadIdx.x];                               // Line 10
    e[i*8] = bc_s[threadIdx.x * 4];                            // Line 11
}
```

**Assumptions:** 1D blocks with `blockDim.x = 256`. Threads in the same warp have consecutive `threadIdx.x` values (e.g., 0, 1, 2, ..., 31 for the first warp).

### a. Access to array `a` on Line 05: `a[i]`

**Coalesced.**

The index is `i = blockIdx.x * blockDim.x + threadIdx.x`. For threads in the same warp, `blockIdx.x * blockDim.x` is a constant, and `threadIdx.x` varies consecutively. Therefore adjacent threads access adjacent elements: `a[base + 0]`, `a[base + 1]`, ..., `a[base + 31]`.

### b. Access to array `a_s` on Line 05: `a_s[threadIdx.x]`

**Not applicable** — `a_s` is declared as `__shared__` memory. Coalescing is a global memory concept; shared memory uses a bank-based access mechanism instead.

### c. Access to array `b` on Line 07: `b[j*blockDim.x*gridDim.x + i]`

**Coalesced.**

For a fixed iteration `j`, the index becomes `j*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x`. The only term that varies across threads in a warp is `threadIdx.x`, which is consecutive. So adjacent threads access adjacent memory locations.

### d. Access to array `c` on Line 07: `c[i*4 + j]`

**Uncoalesced.**

For a fixed `j`, the index is `(blockIdx.x * blockDim.x + threadIdx.x) * 4 + j`. Adjacent threads access elements with a **stride of 4**:
- Thread 0: `c[base*4 + j]`
- Thread 1: `c[(base+1)*4 + j]` = `c[base*4 + 4 + j]`
- Thread 2: `c[(base+2)*4 + j]` = `c[base*4 + 8 + j]`

The addresses are 4 elements (16 bytes) apart — not consecutive. This is a classic strided access pattern.

### e. Access to array `bc_s` on Line 07: `bc_s[j*256 + threadIdx.x]`

**Not applicable** — `bc_s` is shared memory. Coalescing does not apply.

### f. Access to array `a_s` on Line 10: `a_s[threadIdx.x]`

**Not applicable** — `a_s` is shared memory. Coalescing does not apply.

### g. Access to array `d` on Line 10: `d[i + 8]`

**Coalesced.**

The index is `blockIdx.x * blockDim.x + threadIdx.x + 8`. The constant offset of 8 shifts all accesses uniformly but does not affect the relative spacing between threads. Adjacent threads still access adjacent elements: `d[base + 8]`, `d[base + 9]`, ..., `d[base + 39]`. The hardware handles the alignment offset efficiently (Compute 1.2+).

### h. Access to array `bc_s` on Line 11: `bc_s[threadIdx.x * 4]`

**Not applicable** — `bc_s` is shared memory. Coalescing does not apply. (Note: while coalescing doesn't apply, this access pattern with stride 4 *would* cause shared memory bank conflicts, which is a separate performance concern.)

### i. Access to array `e` on Line 11: `e[i*8]`

**Uncoalesced.**

The index is `(blockIdx.x * blockDim.x + threadIdx.x) * 8`. Adjacent threads access elements with a **stride of 8**:
- Thread 0: `e[base * 8]`
- Thread 1: `e[(base+1) * 8]` = `e[base*8 + 8]`
- Thread 2: `e[(base+2) * 8]` = `e[base*8 + 16]`

The addresses are 8 elements (32 bytes) apart. This is even worse than the stride-4 pattern in part (d), as each warp's accesses are spread over a much larger memory region.

### Summary Table

| Part | Array | Line | Memory Type | Index Pattern | Verdict |
|------|-------|------|-------------|---------------|---------|
| a | `a` | 05 | Global | `i` (stride 1) | **Coalesced** |
| b | `a_s` | 05 | Shared | `threadIdx.x` | **N/A** |
| c | `b` | 07 | Global | `const + i` (stride 1) | **Coalesced** |
| d | `c` | 07 | Global | `i*4 + j` (stride 4) | **Uncoalesced** |
| e | `bc_s` | 07 | Shared | `j*256 + threadIdx.x` | **N/A** |
| f | `a_s` | 10 | Shared | `threadIdx.x` | **N/A** |
| g | `d` | 10 | Global | `i + 8` (stride 1) | **Coalesced** |
| h | `bc_s` | 11 | Shared | `threadIdx.x*4` | **N/A** |
| i | `e` | 11 | Global | `i*8` (stride 8) | **Uncoalesced** |

**Key takeaway:** The determining factor for coalescing is whether adjacent threads (consecutive `threadIdx.x`) produce consecutive memory addresses. Any multiplicative factor on `threadIdx.x` in the address calculation (like `i*4` or `i*8`) creates a strided pattern that breaks coalescing. Additive constants (like `i + 8`) are harmless because they shift all addresses uniformly.

---

## Exercise 4: Floating Point to Global Memory Access Ratio (OP/B)

**Question:** What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix multiplication kernels?

Consider matrices A(M, K) and B(K, N), producing output C(M, N) where each element requires a dot product of length K.

### a. Simple kernel (Chapter 3) — No optimizations

**FLOPs per output element:** Each element of C requires K multiply-accumulate operations. Each multiply-accumulate is 2 FLOPs (one multiply + one add), giving **2K FLOPs**.

**Global memory loads per output element:** Each thread independently loads K elements from a row of A and K elements from a column of B, totaling **2K elements = 8K bytes** (at 4 bytes per float).

**OP/B = 2K / 8K = 0.25 OP/B**

This is remarkably low and independent of matrix dimensions. For every byte loaded from global memory, only a quarter of a floating-point operation is performed. The kernel is deeply memory-bandwidth bound, wasting enormous potential compute throughput.

### b. Tiled kernel (Chapter 5) — 32×32 shared memory tiles

**FLOPs per output element:** Still **2K FLOPs** — tiling doesn't change the amount of computation, only how data is accessed.

**Global memory analysis:** A thread block computes a 32×32 tile of the output (1024 elements). The computation proceeds in K/32 phases. In each phase:
- The block cooperatively loads one 32×32 tile from A (1024 elements)
- The block cooperatively loads one 32×32 tile from B (1024 elements)
- Total per phase: 2048 elements loaded from global memory

Across all K/32 phases: **2048 × (K/32) = 64K elements** loaded for the entire block.

Per output element: **64K / 1024 = K/16 elements = K/4 bytes**.

**OP/B = 2K / (K/4) = 8.0 OP/B**

This is a **32× improvement** over the naive kernel. The factor of 32 is exactly the tile dimension — each element loaded from global memory is reused by 32 threads in the block (one full row or column of the tile). Tiling converts redundant independent global memory accesses into a single cooperative load followed by shared memory reuse.

### c. Tiled kernel with thread coarsening — 32×32 tiles, coarsening factor 4

**FLOPs per output element:** Still **2K FLOPs** — coarsening doesn't change the computation per element.

**Global memory analysis with coarsening:** With a coarsening factor of 4, each thread computes 4 output elements that share the **same row** of A but use different columns of B. In each of the K/32 phases, per thread:

- **1 load from A's tile** — shared across all 4 output elements (same row)
- **4 loads from B's tile** — one for each output element (different columns)
- **5 total loads for 4 output elements**

Per output element per phase: 5/4 = 1.25 loads.

Across all K/32 phases per output element: **1.25 × (K/32) = 5K/128 elements = 5K/32 bytes**.

**OP/B = 2K / (5K/32) = 64/5 = 12.8 OP/B**

### Comparison Summary

| Kernel | Loads/Element | OP/B | Improvement vs. Naive |
|--------|-------------|------|----------------------|
| (a) Naive | 2K | 0.25 | 1× (baseline) |
| (b) Tiled 32×32 | K/16 | 8.0 | 32× |
| (c) Tiled + Coarsening (4) | 5K/128 | 12.8 | 51.2× |

**Key insights:**

1. **Tiling provides the dominant improvement** (32×) by enabling data reuse across threads via shared memory. The improvement factor equals the tile dimension.

2. **Coarsening provides a more modest additional gain** (1.6× over tiled) by exploiting asymmetric reuse *within* a single thread — the A tile element is loaded once but used for 4 different dot products.

3. **The OP/B ratio is independent of matrix size** for all three kernels — it depends only on the tile size and coarsening factor. This means the optimization benefit is consistent regardless of problem scale.

4. **Diminishing returns:** Each optimization layer adds less than the previous. Tiling eliminates redundancy across threads; coarsening eliminates redundancy within a thread's workload. The remaining inefficiency would require fundamentally different approaches (e.g., register tiling, data prefetching) to address further.
