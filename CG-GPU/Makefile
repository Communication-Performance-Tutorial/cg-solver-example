# =============================================================================
# Makefile for the GPU Conjugate Gradient Tutorial
#
# Usage:
#   module load rocm          # loads ROCm (hipcc, rocSPARSE, rocBLAS)
#   module load rocm-*/ucx-*/ompi/*   # loads GPU-Aware MPI
#   make
#   mpirun -n 4 ./cg_gpu src/Dubcova2.pm
# =============================================================================

ROCM_PATH ?= /opt/rocm

# hipcc wraps clang for HIP; it knows where ROCm headers/libs are
HIPCC = $(ROCM_PATH)/bin/hipcc

CXXFLAGS = -O3 -std=c++17

# Pull MPI compile/link flags directly from the mpicc wrapper.
# This automatically picks up whichever MPI is loaded via the module system,
# including the GPU-Aware OpenMPI build (--with-rocm).
MPI_CFLAGS := $(shell mpicc --showme:compile 2>/dev/null)
MPI_LFLAGS := $(shell mpicc --showme:link   2>/dev/null)

INCLUDES = -I$(ROCM_PATH)/include -Isrc $(MPI_CFLAGS)
LDFLAGS  = -L$(ROCM_PATH)/lib $(MPI_LFLAGS)

# rocSPARSE and rocBLAS for GPU linear algebra; rccl for GPU-native collectives
LIBS     = -lrocsparse -lrocblas -lrccl

TARGET   = cg_gpu
SRC      = src/cg.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

# Quick sanity run: 4 ranks, all four methods
run: $(TARGET)
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm isend
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm staged
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm alltoallv
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm rccl

clean:
	rm -f $(TARGET)
