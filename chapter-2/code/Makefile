NVCC ?= nvcc
CUDA_ARCH ?= native

DEMO_BIN := vecMul_demo
TEST_BIN := vecMul_test

SRC := vecMul.cu
DEMO_SRC := vecMul_demo.cu
TEST_SRC := vecMul_test.cu

GTEST_LIBS := -lgtest -lgtest_main -lpthread

.PHONY: all demo test clean

all: demo

demo: $(DEMO_BIN)

test: $(TEST_BIN)

$(DEMO_BIN): $(SRC) $(DEMO_SRC) vecMul.h
	$(NVCC) -arch=$(CUDA_ARCH) $(SRC) $(DEMO_SRC) -o $@

$(TEST_BIN): $(SRC) $(TEST_SRC) vecMul.h
	$(NVCC) -arch=$(CUDA_ARCH) $(SRC) $(TEST_SRC) -o $@ $(GTEST_LIBS)

clean:
	rm -f $(DEMO_BIN) $(TEST_BIN)
