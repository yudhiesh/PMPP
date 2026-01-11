# Repository Guidelines

## Project Structure & Module Organization
- `vecMul.cu` holds the CUDA vector-add implementation and GoogleTest cases.
- `vecMul` is a built binary (do not edit by hand; recompile when needed).

## Build, Test, and Development Commands
- `nvcc vecMul.cu -lgtest -lpthread -o vecMul` builds the CUDA test binary (adjust library paths if your GTest install is non-standard).
- `./vecMul` runs the GoogleTest suite locally.

## Coding Style & Naming Conventions
- Indentation: 2 spaces, braces on the same line.
- CUDA kernels use `camelCase` with `Kernel` suffix (e.g., `vecAddKernel`).
- Host helpers use `camelCase` (e.g., `vecAdd`), constants in `UPPER_SNAKE_CASE` when added.
- Prefer simple, explicit variable names for GPU buffers (`A_d`, `B_d`, `C_d`) to mirror host arrays.

## Testing Guidelines
- Framework: GoogleTest (`#include <gtest/gtest.h>`).
- Test cases live in `vecMul.cu` and follow `TEST(SuiteName, CaseName)` with descriptive names (e.g., `VecAddTest, LargeArray`).
- Add tests for edge cases (small sizes, zeros, negatives) and validate selected indices for large arrays.

## Commit & Pull Request Guidelines
- Git history only shows `Initial commit`, so no established commit convention yet.
- Use short, imperative commit messages (e.g., "Add bounds check to kernel").
- PRs should summarize the change, include test results (`./vecMul`), and note any CUDA/toolchain requirements.

## Security & Configuration Tips
- Ensure your CUDA toolkit and compatible GPU drivers are installed before building.
- If GTest is not installed system-wide, document the include/lib paths used for your build.
