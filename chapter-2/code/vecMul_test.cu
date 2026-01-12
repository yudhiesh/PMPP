#include <gtest/gtest.h>
#include <stdlib.h>

#include "vecMul.h"

// ============ GOOGLE TEST CASES ============

TEST(VecAddTest, BasicAddition) {
  int N = 10;
  float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  float B[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  float C[10] = {0};

  vecAdd(A, B, C, N);

  float expected[] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 110};

  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(C[i], expected[i], 0.001);
  }
}

TEST(VecAddTest, AllZeros) {
  int N = 100;
  float *A = (float *)calloc(N, sizeof(float));
  float *B = (float *)calloc(N, sizeof(float));
  float *C = (float *)malloc(N * sizeof(float));

  vecAdd(A, B, C, N);

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(C[i], 0.0f);
  }

  free(A);
  free(B);
  free(C);
}

TEST(VecAddTest, NegativeNumbers) {
  int N = 5;
  float A[] = {-1, -2, -3, -4, -5};
  float B[] = {1, 2, 3, 4, 5};
  float C[5] = {0};

  vecAdd(A, B, C, N);

  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(C[i], 0.0f, 0.001);
  }
}

TEST(VecAddTest, LargeArray) {
  int N = 100000;
  float *A = (float *)malloc(N * sizeof(float));
  float *B = (float *)malloc(N * sizeof(float));
  float *C = (float *)malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    A[i] = i * 0.5f;
    B[i] = i * 1.5f;
  }

  vecAdd(A, B, C, N);

  for (int i = 0; i < N; i += 1000) {
    EXPECT_NEAR(C[i], A[i] + B[i], 0.001);
  }

  free(A);
  free(B);
  free(C);
}

TEST(VecAddTest, SingleElement) {
  float A[] = {42.0f};
  float B[] = {58.0f};
  float C[1] = {0};

  vecAdd(A, B, C, 1);

  EXPECT_NEAR(C[0], 100.0f, 0.001);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
