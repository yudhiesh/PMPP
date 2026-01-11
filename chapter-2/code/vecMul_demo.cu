#include <stdio.h>
#include <stdlib.h>

#include "vecMul.h"

int main() {
  int N = 10;
  float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  float B[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  float *C = (float *)calloc(N, sizeof(float));

  vecAdd(A, B, C, N);

  printf("Result:\\n");
  for (int i = 0; i < N; i++) {
    printf("%0.1f ", C[i]);
  }
  printf("\\n");

  free(C);
  return 0;
}
