# PMPP Chapter 4: Compute Architecture and Scheduling
## Complete Exercise Solutions

---

## Question 1: Kernel Analysis — Warps, Divergence, and SIMD Efficiency

### Kernel Code

```cuda
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;                        // Line 04
    }
    if(i%2 == 0) {
        a[i] = b[i]*2;                          // Line 07
    }
    for(unsigned int j = 0; j < 5 - (i%3); ++j) {
        b[i] += j;                              // Line 10
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<<(N + 128 - 1)/128, 128>>>(a_d, b_d);
}
```

### Grid Configuration

From the kernel launch:
- **Block size:** 128 threads per block
- **Grid size:** ⌈1024/128⌉ = 8 blocks
- **Total threads:** 8 × 128 = 1024 threads

---

### Part (a): Warps per Block

> **Answer: 4 warps per block**

A warp always contains 32 threads:

```
128 threads ÷ 32 threads/warp = 4 warps
```

---

### Part (b): Warps in the Grid

> **Answer: 32 warps**

```
4 warps/block × 8 blocks = 32 warps
```

---

### Part (c): Analysis of Line 04

```cuda
if(threadIdx.x < 40 || threadIdx.x >= 104) {
    b[i] = a[i] + 1;
}
```

**Key Observation:** The condition uses `threadIdx.x` (local index 0–127), so the pattern is identical in every block.

**Threads executing line 04:** threadIdx.x ∈ {0–39} ∪ {104–127}

| Warp | threadIdx.x | Threads Executing | Active? | Divergent? |
|------|-------------|-------------------|---------|------------|
| 0    | 0–31        | 32 (all satisfy `< 40`) | Yes | No |
| 1    | 32–63       | 8 (only 32–39 satisfy `< 40`) | Yes | **Yes** |
| 2    | 64–95       | 0 (none satisfy either condition) | No | No |
| 3    | 96–127      | 24 (only 104–127 satisfy `≥ 104`) | Yes | **Yes** |

**Per block:** 3 active warps, 2 divergent warps

#### Answers

| Question | Answer | Reasoning |
|----------|--------|-----------|
| **(i)** Active warps in grid | **24** | 3 active warps/block × 8 blocks |
| **(ii)** Divergent warps in grid | **16** | 2 divergent warps/block × 8 blocks |
| **(iii)** SIMD efficiency, Warp 0 | **100%** | 32/32 threads execute |
| **(iv)** SIMD efficiency, Warp 1 | **25%** | 8/32 threads execute |
| **(v)** SIMD efficiency, Warp 3 | **75%** | 24/32 threads execute |

---

### Part (d): Analysis of Line 07

```cuda
if(i%2 == 0) {
    a[i] = b[i]*2;
}
```

**Key Observation:** The condition uses the global index `i`. Within any warp, threads have 32 consecutive `i` values — exactly 16 even and 16 odd.

**Result:** Every warp splits 50/50 between threads executing and skipping.

#### Answers

| Question | Answer | Reasoning |
|----------|--------|-----------|
| **(i)** Active warps in grid | **32** | Every warp has threads with even `i` |
| **(ii)** Divergent warps in grid | **32** | Every warp has a 16/16 split |
| **(iii)** SIMD efficiency, Warp 0 | **50%** | 16/32 threads execute |

---

### Part (e): Analysis of the Loop (Lines 09–11)

```cuda
for(unsigned int j = 0; j < 5 - (i%3); ++j) {
    b[i] += j;
}
```

**Key Observation:** Iteration count depends on `i % 3`:

| i % 3 | Loop Bound | Iterations |
|-------|------------|------------|
| 0     | j < 5      | 5          |
| 1     | j < 4      | 4          |
| 2     | j < 3      | 3          |

Within any warp, `i % 3` cycles through 0, 1, 2, 0, 1, 2... so threads have varying iteration counts.

**Iteration-by-Iteration Analysis:**

| Iteration | i%3 = 0 | i%3 = 1 | i%3 = 2 | Divergent? |
|-----------|---------|---------|---------|------------|
| j = 0     | ✓       | ✓       | ✓       | No         |
| j = 1     | ✓       | ✓       | ✓       | No         |
| j = 2     | ✓       | ✓       | ✓       | No         |
| j = 3     | ✓       | ✓       | ✗ done  | **Yes**    |
| j = 4     | ✓       | ✗ done  | ✗ done  | **Yes**    |

#### Answers

| Question | Answer | Reasoning |
|----------|--------|-----------|
| **(i)** Iterations with no divergence | **3** | All threads execute at least 3 iterations (j = 0, 1, 2) |
| **(ii)** Iterations with divergence | **2** | Threads finish at different times (j = 3, 4) |

---

## Question 2: Vector Addition Grid Size

**Problem:** Vector length is 2000, each thread calculates one output element, block size is 512 threads. How many threads will be in the grid?

### Solution

Number of blocks (using ceiling division):

```
Blocks = ⌈2000 / 512⌉ = ⌈3.906⌉ = 4
```

Total threads:

```
Threads = 4 × 512 = 2048
```

> **Answer: 2048 threads**

**Key Insight:** We launch 2048 threads for 2000 elements. The extra 48 threads fail the boundary check (`if (i < n)`) and do no work. This is the standard pattern for handling arbitrary data sizes.

---

## Question 3: Warp Divergence at Boundary

**Problem:** For Question 2, how many warps have divergence due to the boundary check?

### Solution

**Grid Layout:**

| Block | Global Thread Range | Processing Elements |
|-------|---------------------|---------------------|
| 0     | 0–511               | 0–511 (all valid)   |
| 1     | 512–1023            | 512–1023 (all valid)|
| 2     | 1024–1535           | 1024–1535 (all valid)|
| 3     | 1536–2047           | 1536–1999 (partial) |

**Analysis of Block 3:**

The boundary at element 2000 falls at local thread index 464 (since 2000 − 1536 = 464).

| Warp | Local Threads | Global Threads | Behavior |
|------|---------------|----------------|----------|
| 14   | 448–479       | 1984–2015      | Threads 1984–1999 pass, 2000–2015 fail → **Divergent** |
| 15   | 480–511       | 2016–2047      | All 32 threads fail → No divergence |

> **Answer: 1 warp**

**Key Insight:** Divergence only occurs when threads *within the same warp* take different paths. Warp 15 uniformly fails the check — no divergence.

---

## Question 4: Barrier Synchronization Overhead

**Problem:** 8 threads execute before a barrier with times (μs): 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9. What percentage of total execution time is spent waiting?

### Solution

**Step 1:** Barrier releases when the slowest thread arrives:

```
t_barrier = max(all times) = 3.0 μs
```

**Step 2:** Calculate wait times:

| Thread | Execution Time | Wait Time |
|--------|----------------|-----------|
| 0      | 2.0            | 1.0       |
| 1      | 2.3            | 0.7       |
| 2      | 3.0            | 0.0       |
| 3      | 2.8            | 0.2       |
| 4      | 2.4            | 0.6       |
| 5      | 1.9            | 1.1       |
| 6      | 2.6            | 0.4       |
| 7      | 2.9            | 0.1       |
| **Total** |             | **4.1**   |

**Step 3:** Calculate percentage:

```
Total execution time = 8 × 3.0 = 24.0 μs

Waiting percentage = (4.1 / 24.0) × 100 = 17.08%
```

> **Answer: 17.08%**

**Key Insight:** Barrier cost scales with work imbalance. Nearly 1/6 of compute resources sit idle, illustrating why kernels should distribute work evenly.

---

## Question 5: Omitting `__syncthreads()` with 32-Thread Blocks

**Problem:** A programmer claims that with 32 threads per block (one warp), they can skip `__syncthreads()`. Is this a good idea?

> **Answer: No, this is not a good idea.**

**The Programmer's Logic:** A warp executes in lockstep (SIMT), so 32 threads are implicitly synchronized.

**Why This Fails:**

| Issue | Explanation |
|-------|-------------|
| **Memory Fence** | `__syncthreads()` ensures shared memory writes are visible to other threads. Without it, the compiler may reorder operations or cache values in registers. |
| **Portability** | Warp size (32) is a hardware characteristic, not a guarantee. Future architectures could change it. |
| **Maintainability** | If someone changes the block size to 64, the code silently breaks with no compiler warning. |
| **Programming Model** | CUDA only guarantees synchronization through explicit primitives. Relying on warp behavior uses undocumented implementation details. |

**Best Practice:** Always use `__syncthreads()` when synchronization is needed, regardless of block size.

---

## Question 6: Maximizing Threads per SM

**Problem:** SM supports up to 1536 threads and 4 blocks. Which configuration maximizes threads?

- (a) 128 threads/block
- (b) 256 threads/block
- (c) 512 threads/block
- (d) 1024 threads/block

### Solution

Both constraints must be satisfied — the tighter one wins:

| Option | Threads/Block | Block Limit | Thread Limit | Actual Blocks | Total Threads |
|--------|---------------|-------------|--------------|---------------|---------------|
| (a)    | 128           | 4           | ⌊1536/128⌋ = 12 | 4          | 512           |
| (b)    | 256           | 4           | ⌊1536/256⌋ = 6  | 4          | 1024          |
| (c)    | 512           | 4           | ⌊1536/512⌋ = 3  | 3          | **1536**      |
| (d)    | 1024          | 4           | ⌊1536/1024⌋ = 1 | 1          | 1024          |

> **Answer: (c) 512 threads per block**

**Key Insight:** Options (a) and (b) are *block-limited*; (c) and (d) are *thread-limited*. The optimal point is where constraints balance. Larger blocks aren't always better — 1024 threads/block wastes capacity due to the floor effect.

---

## Question 7: SM Occupancy Analysis

**Problem:** SM supports up to 64 blocks and 2048 threads. Which assignments are possible, and what is their occupancy?

### Solution

Occupancy = Actual Threads / 2048

| Option | Blocks | Threads/Block | Total Threads | Valid? | Occupancy |
|--------|--------|---------------|---------------|--------|-----------|
| (a)    | 8      | 128           | 1024          | ✓      | **50%**   |
| (b)    | 16     | 64            | 1024          | ✓      | **50%**   |
| (c)    | 32     | 32            | 1024          | ✓      | **50%**   |
| (d)    | 64     | 32            | 2048          | ✓      | **100%**  |
| (e)    | 32     | 64            | 2048          | ✓      | **100%**  |

### Key Insights

1. **All configurations are valid** — none exceed either constraint.

2. **Multiple paths to same occupancy** — Options (a), (b), (c) achieve 50% with very different structures.

3. **Choosing between 100% options:** Option (e) with 64 threads/block is generally preferable over (d):
   - Better shared memory utilization per block
   - Lower scheduling overhead (32 vs. 64 blocks)
   - More flexible 2D tile shapes (8×8 possible)
   - Better preparation for tiled algorithms

---

## Summary of Key Concepts

| Concept | Key Takeaway |
|---------|--------------|
| **Warps** | Always 32 threads; the fundamental scheduling unit |
| **Divergence** | Occurs when threads *within the same warp* take different paths |
| **SIMD Efficiency** | (Active threads / 32) × 100%; measures warp utilization |
| **Grid Sizing** | Use ceiling division; extra threads handle boundary checks |
| **Barrier Overhead** | Scales with work imbalance; design for even distribution |
| **`__syncthreads()`** | Never skip it — provides both barrier and memory fence |
| **Occupancy** | Balance multiple constraints; larger blocks aren't always better |
| **Block Size Selection** | Consider shared memory, scheduling overhead, and 2D tile shapes |

