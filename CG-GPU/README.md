# Distributed Conjugate Gradient GPU-Based Example

A GPU-accelerated, MPI-distributed Conjugate Gradient (CG) solver.

This directory is a direct companion to `CG-CPU/` — it solves the same problem with the same algorithm but moves every numerical
operation to the GPU and uses GPU-Aware MPI so communication buffers live in GPU memory the whole time.

---

## What changed from `CG-CPU/`

| `CG-CPU/`                            | `CG-GPU/`                                             |
|--------------------------------------|-------------------------------------------------------|
| `std::vector<double>` for vectors    | `hipMalloc` / `hipMemcpy` for device vectors          |
| Hand-written SpMV loop               | `rocsparse_spmv` (rocSPARSE generic API)              |
| Hand-written gather (pack send buf)  | `rocsparse_dgthr` (rocSPARSE gather)                  |
| Hand-written dot product             | `rocblas_ddot` (rocBLAS)                              |
| Hand-written axpy / scale            | `rocblas_daxpy` / `rocblas_dscal` (rocBLAS)           |
| `MPI_Irecv/Isend` with host buffers  | `MPI_Irecv/Isend` with **GPU pointers** (GPU-Aware)   |
| `MPI_Allreduce` on host scalar       | Same (result copied from GPU → host, then allreduced) |

The two header files (`sparse_mat.hpp`, `par_binary_IO.hpp`) are **identical** to the originals — file I/O and comm pattern setup
stay on the CPU.

---

## File layout

```
CG-Tutorial/
├── Makefile
├── README.md
└── src/
    ├── cg.cpp            ← GPU solver (the interesting file)
    ├── sparse_mat.hpp    ← CPU structs + MPI comm setup  (same as CG/)
    ├── par_binary_IO.hpp ← parallel binary matrix reader (same as CG/)
    └── Dubcova2.pm       ← test matrix (65 536 × 65 536, SPD)
```

---

## Environment Setup

```bash
module load rocm/6.4.1
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4
```

ROCm provides `hipcc`, rocSPARSE, rocBLAS, and RCCL.  The OpenMPI build must be compiled with ROCm/UCX support so that MPI can send and
receive GPU pointers directly (GPU-Aware MPI).

---

## How to build

```bash
cd CG-GPU
make
```

---

## How to run

Pass the matrix file as the first argument and the communication method as the second (default: `staged`):

```bash
mpirun -n 4 ./cg_gpu src/Dubcova2.pm staged            # (default) Isend/Irecv through CPU host buffers
mpirun -n 4 ./cg_gpu src/Dubcova2.pm isend             # Isend/Irecv with GPU buffers (GPU-Aware)
mpirun -n 4 ./cg_gpu src/Dubcova2.pm rccl              # RCCL ncclSend/ncclRecv
mpirun -n 4 ./cg_gpu src/Dubcova2.pm alltoallv_staged  # MPI_Alltoallv through CPU host buffers
mpirun -n 4 ./cg_gpu src/Dubcova2.pm alltoallv         # MPI_Alltoallv with GPU buffers (GPU-Aware)
```

All five methods produce the same numerical result (the CG algorithm is identical; only the spmv data exchange differs).

Expected output (any method):

```
method=staged  ranks=4  gpus_visible=24
Initial residual: 197.025
172 iterations to converge
2-norm of residual: 0.000196
CG solve time:      0.4724 s  (0.0028 s/iter)
```

The iteration count, residual, and solve time vary slightly between runs because the right-hand side is seeded from a random initial guess.
The solve time covers the CG loop only (matrix read, upload, and setup are excluded) and is reported as the maximum across all MPI ranks.

### GPU assignment

The solver assigns one GPU per MPI rank using:

```cpp
hipSetDevice(rank % num_gpus);
```

On a node with 4 MPI ranks and multiple visible GPUs, each rank gets a distinct GPU.  To control which physical GPUs are used, set
`ROCR_VISIBLE_DEVICES` before launching, or use a GPU affinity script with `mpirun --map-by NUMA`.  See the [Affinity part 2 blog post](https://rocm.blogs.amd.com/software-tools-optimization/affinity/part-2/README.html)
for guidance on matching MPI ranks to GPU NUMA domains.

---

## Communication variants

### Method 1 — `staged` (CPU-buffered Isend/Irecv)

The simplest baseline: all packing, communication, and unpacking go through the CPU.  MPI operates on host pointers.

1. Copy the full local vector from GPU to host (`hipMemcpy D→H`)
2. Pack send values into a pinned host buffer (CPU loop)
3. `MPI_Irecv` / `MPI_Isend` on host pointers
4. After wait, copy received values back to GPU (`hipMemcpy H→D`)

### Method 2 — `isend` (GPU-Aware Isend/Irecv)

Same point-to-point structure as `staged`, but ghost values never touch the CPU.  `rocsparse_dgthr` packs the send buffer directly on the
GPU, and `MPI_Isend` / `MPI_Irecv` operate on GPU pointers.  On-proc SpMV overlaps with in-flight messages.

Comparing `staged` vs. `isend` shows the cost of the D→H + H→D copies that GPU-Aware MPI eliminates.

### Method 3 — `rccl` (RCCL ncclSend / ncclRecv)

RCCL (ROCm Collective Communications Library) provides GPU-native communication that can exploit Infinity Fabric / NVLink paths without any CPU
involvement.  Sends and receives are grouped with `ncclGroupStart` / `ncclGroupEnd` and launched on a dedicated `rccl_stream`.  On-proc SpMV
runs on the default GPU stream concurrently — true GPU-GPU overlap.

### Method 4 — `alltoallv_staged` (CPU-buffered MPI_Alltoallv)

Replaces the many point-to-point messages with one collective, but uses explicit host staging:

1. `rocsparse_dgthr` packs into `d_sendbuf` on the GPU
2. `hipMemcpy D→H` → `h_sendbuf`
3. `MPI_Alltoallv` on host pointers
4. `hipMemcpy H→D` → `d_recvbuf`
5. On-proc SpMV overlaps with step 3 (CPU blocks in MPI, GPU computes)

### Method 5 — `alltoallv` (GPU-Aware MPI_Alltoallv)

Same collective as `alltoallv_staged` but passes GPU pointers directly to `MPI_Alltoallv` — no host copies needed.  On-proc SpMV is submitted 
to the GPU before the collective blocks the CPU, so both run in parallel.

Comparing `alltoallv_staged` vs. `alltoallv` isolates the PCIe round-trip cost (D→H + H→D) that GPU-Aware MPI avoids.

---

## Key concepts

### 1. One GPU per MPI rank

```cpp
hipSetDevice(rank % num_gpus);
```

Ranks round-robin across available GPUs.  On a node with 4 MI300A partitions and 4 MPI ranks, each rank gets its own GPU.

### 2. rocSPARSE SpMV instead of a CPU loop

```cpp
// CPU (original):
for (int i = 0; i < A.n_rows; i++)
    for (int j = A.rowptr[i]; j < A.rowptr[i+1]; j++)
        b[i] += alpha * A.data[j] * x[A.col_idx[j]];

// GPU (this file):
rocsparse_spmv(handle, rocsparse_operation_none,
               &alpha, A.descr, vec_x, &beta, vec_b, ...);
```

The rocSPARSE descriptor wraps the CSR arrays that were uploaded with `hipMemcpy`.  rocSPARSE picks the best SpMV algorithm for the hardware.

### 3. GPU-Aware MPI — sending/receiving GPU buffers directly

```cpp
// Pack the send buffer on the GPU (no host copy needed)
rocsparse_dgthr(handle, send_size, d_x, d_sendbuf, d_send_idx, ...);
hipDeviceSynchronize();   // gather must finish before MPI reads d_sendbuf

// Send and receive GPU pointers — the MPI runtime handles the rest
MPI_Isend(d_sendbuf + offset, count, MPI_DOUBLE, dest, ...);
MPI_Irecv(d_recvbuf + offset, count, MPI_DOUBLE, src,  ...);
```

Without GPU-Aware MPI you would have to copy from GPU→host before sending and host→GPU after receiving.  With GPU-Aware MPI you skip both copies.

### 4. rocBLAS for dot / axpy / scale

```cpp
rocblas_ddot (blas_handle, n, d_a, 1, d_b, 1, &local_sum);
rocblas_daxpy(blas_handle, n, &alpha, d_y, 1, d_x, 1);   // d_x += alpha*d_y
rocblas_dscal(blas_handle, n, &alpha, d_x, 1);            // d_x *= alpha
```

After `rocblas_ddot`, a single `hipDeviceSynchronize()` makes the result visible on the CPU before `MPI_Allreduce` reduces it across ranks.

---

## Requirements

- ROCm ≥ 6.3 (rocSPARSE, rocBLAS, hipcc)
- GPU-Aware MPI (OpenMPI/UCX built with ROCm support — see module list on
  the login banner)
