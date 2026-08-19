# Distributed CG Solver Example

A distributed (MPI) Conjugate Gradient (CG) solver used as a tutorial for studying **communication performance** on GPU clusters. The same sparse linear system is solved two ways — a plain CPU reference implementation and a GPU-accelerated port with **five interchangeable communication strategies** — so you can directly measure what GPU-Aware MPI, RCCL, and different collective/point-to-point patterns cost or save.

The test problem is `Dubcova2` (65 536 × 65 536, symmetric positive-definite), partitioned across MPI ranks and solved with the standard CG iteration:

```
r₀ = b − A x₀
p₀ = r₀
for i = 0, 1, 2, …:
    α  = (rᵢ, rᵢ) / (Apᵢ, pᵢ)
    x  += α pᵢ
    r  = b − A x        (recomputed every 8 iterations for numerical stability)
    β  = (rᵢ₊₁, rᵢ₊₁) / (rᵢ, rᵢ)
    p  = r + β p
```

Each iteration needs a distributed sparse matrix–vector product (SpMV), which requires exchanging "ghost" values between neighboring ranks — that halo exchange, plus the `MPI_Allreduce` for the dot products, is the communication this tutorial exists to study.

---

## Directories

| Directory | What it is |
|---|---|
| [`CG-CPU/`](CG-CPU/README.md) | **Reference implementation.** Plain C++ + MPI, `std::vector` for vectors, hand-written SpMV/axpy/dot loops. No GPU, no external math libraries. Written by Amanda Bienz, UNM ([source](https://github.com/bienz2/CG)). |
| [`CG-GPU/`](CG-GPU/README.md) | **GPU port.** Same algorithm, but vectors live on the GPU (`hipMalloc`), SpMV/dot/axpy use rocSPARSE/rocBLAS, and the halo exchange can run through any of **5 communication variants**: CPU-staged Isend/Irecv, GPU-Aware Isend/Irecv, RCCL, and CPU-staged or GPU-Aware `MPI_Alltoallv`. |

`CG-GPU/` is built directly on top of `CG-CPU/` — reading the two side by side shows exactly what changes when a solver moves from the CPU to the GPU, and then what changes again as GPU-Aware MPI and RCCL are layered on top of that.

---

## Quick start

```bash
# CPU reference
cd CG-CPU
module load rocm/6.4.1
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4
make
salloc -N 1 --ntasks=4 --gpus-per-task=1 -p SH5_MI300A_CPX
mpirun -n 4 ./cg_cpu src/Dubcova2.pm
exit

# GPU solver
cd ../CG-GPU
module load rocm/6.4.1
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4
make
salloc -N 1 --ntasks=4 --gpus-per-task=1 -p SH5_MI300A_CPX
mpirun -n 4 ./cg_gpu src/Dubcova2.pm rccl
exit
```

See each directory's README for the full build/run instructions, environment setup, and a breakdown of every communication variant.

---

## What this tutorial is for

Comparing runs across `CG-CPU/` and the different `CG-GPU/` communication variants answers questions like:

- How much does GPU-Aware MPI save over staging through host (pinned) buffers?
- Is RCCL (GPU-native, Infinity-Fabric-aware) faster than MPI point-to-point for this halo pattern?
- Does replacing many small point-to-point messages with one `MPI_Alltoallv` collective help or hurt?
- Do SDMA engines or blit kernels move ghost data faster for this message size (`HSA_ENABLE_SDMA`)?

`CG-GPU/` also has scripts for sweeping these variants automatically (`run_test.sh`) and for profiling the run with TAU — see [`CG-GPU/README.md`](CG-GPU/README.md#profiling-with-tau) for per-call MPI/ROCm timing and the communication matrix.
