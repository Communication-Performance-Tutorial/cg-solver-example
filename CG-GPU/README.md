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

## SDMA engines vs. blit kernels (`HSA_ENABLE_SDMA`)

When GPU-Aware MPI (`isend`, `alltoallv`) passes a GPU pointer to the MPI runtime, UCX must physically move the data out of device memory.
ROCm gives it two mechanisms to do that:

| Mechanism | `HSA_ENABLE_SDMA` | How it works |
|---|---|---|
| **SDMA engines** | `1` (default) | Dedicated hardware DMA controllers transfer data between GPU memory and the fabric. They run independently of the shader engines and do not consume compute resources. |
| **Blit kernels** | `0` | ROCm dispatches a small compute shader (a "blit") to copy the data using the GPU's shader engines. No dedicated DMA hardware is used. |

Set the variable before `mpirun` to switch between them:

```bash
# SDMA engines (default)
HSA_ENABLE_SDMA=1 mpirun -n 8 --bind-to none bash set_affinity_mi300a.sh ./cg_gpu src/Dubcova2.pm isend

# Blit kernels
HSA_ENABLE_SDMA=0 mpirun -n 8 --bind-to none bash set_affinity_mi300a.sh ./cg_gpu src/Dubcova2.pm isend
```

`run_test_7.13.sh` sweeps both values automatically for `isend` and `alltoallv` and labels each run in the log output:

```
=== isend  HSA_ENABLE_SDMA=1  (sdma) ===
=== isend  HSA_ENABLE_SDMA=0  (blit_kernel) ===
=== alltoallv  HSA_ENABLE_SDMA=1  (sdma) ===
=== alltoallv  HSA_ENABLE_SDMA=0  (blit_kernel) ===
```

### When does each win?

- **SDMA** tends to win for **large, infrequent transfers** where having the shader engines free to overlap computation matters more
  than raw copy throughput.
- **Blit kernels** tend to win for **small, latency-sensitive transfers** where the shader engines are otherwise idle and the lower
  software overhead of a compute dispatch beats the DMA engine's start-up cost.  On MI300A unified memory, blit kernels often outperform
  SDMA for the modest ghost-zone message sizes typical of sparse solvers.

The right choice is workload- and hardware-dependent; the sweep in `run_test_7.13.sh` surfaces the difference empirically.

---

## Profiling with TAU

TAU intercepts MPI and, with `-rocm`, ROCm/HIP GPU calls via `LD_PRELOAD` — this directly measures the **communication** that GPU-only tools (rocprofv3, roofline, etc.) don't see: per-call MPI time, the per-rank point-to-point **communication matrix**, and (with `-rocm`) the ROCm runtime/kernel time in the same profile. No special instrumented build is needed — `tau_exec` wraps the plain `./cg_gpu` binary at run time.

```bash
module load rocm/6.4.3
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4
module load tau/dev      # layered on top of rocm — load it last

make
mpirun -n 4 --bind-to none bash set_affinity_mi300a.sh \
    tau_exec -T MPI,ROCM -rocm ./cg_gpu src/Dubcova2.pm rccl

pprof            # text summary: per-call MPI + ROCm/HSA time, merged across ranks
```

`run_tau.sh` automates this end-to-end as an `sbatch` job (build → run under
`tau_exec` → `pprof` summary), since the login node has no GPU:

```bash
sbatch run_tau.sh                              # profile 4 ranks, method=rccl
METHOD=isend sbatch run_tau.sh                  # profile a different comm. variant
TRACE=1 sbatch run_tau.sh                       # also write an OTF2 trace + a Perfetto/Chrome JSON trace
sbatch -p PPAC_MI300A_SPX --gpus=4 --ntasks=4 run_tau.sh   # override partition/GPU count
```

### Perfetto / `chrome://tracing` timeline (`TRACE=1`)

With `TRACE=1`, `run_tau.sh` merges the per-rank raw traces (`tau_treemerge.pl`) and converts the merged trace with
`tau_trace2json ... -chrome` into `$EXPDIR/perfetto_<method>.json` — a standard Chrome Trace Event JSON file. Open
it at [ui.perfetto.dev](https://ui.perfetto.dev) (drag-and-drop the file) or `chrome://tracing` to see every
MPI call, ROCm/HSA runtime call, and GPU kernel launch as a timeline per rank/thread, with flow arrows (`s`/`f`
events) connecting each `MPI_Isend` to its matching `MPI_Irecv` — this is the easiest way to *see* overlap (or
the lack of it) between communication and compute, rather than reading it out of aggregate call counts.

> **Not valid strict JSON — by design.** The file is a bare `[ {...}, {...}, ... ,` with no closing `]` and a
> trailing comma; this is the documented legacy Chrome Trace Event Format convention (any trace tooling can crash
> mid-write and still leave a loadable file), and both Perfetto UI and `chrome://tracing` handle it natively. A
> strict `json.load()` will reject it — that's expected, not corruption.
>
> **These traces are large.** A 4-rank, ~170-iteration run produces a ~400 MB JSON (~2.9M events after
> `MPI_Init`, `hsa_*` runtime calls, and per-iteration kernels are all captured). Perfetto UI can still load it,
> but expect it to take a while; for a smaller trace, target fewer iterations/ranks or a smaller matrix.

Verified on AAC6/MI300A (ROCm 6.4.3, `tau/dev`): the profile below shows real per-call MPI time (`MPI_Init`, `MPI_Allreduce`, …) merged with ROCm HSA runtime and kernel time (`rocsparse::gthr_kernel`, `rocblas_dot_kernel_*`, …) in one `pprof` summary:

```
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
100.0           34        8,976           1           1    8976004 .TAU application
 67.3        6,038        6,038           1           0    6038301 MPI_Init()
  0.2            7           20         306         306         69 MPI_Allreduce()
  0.0       0.0688       0.0688       25.05           0          3 [ROCm Kernel] rocsparse::gthr_kernel<...>
```

> **Known teardown quirk (ROCm 6.4.3 + `tau/dev`).** The instrumented run reliably
> prints the CG solve result and writes every `profile.*` file, then the *process
> teardown* aborts with `corrupted size vs. prev_size in fastbins` (a rocprofsdk +
> glibc interaction, not a measurement failure) — `run_tau.sh` reads the profile
> anyway and does not treat that nonzero exit as a hard failure.
>
> **`pprof`/`paraprof` read `$PROFILEDIR` themselves.** Run them from `CG-GPU/`
> with `PROFILEDIR` set to the experiment dir — *don't* `cd` into that directory
> first, or they'll look for a nonexistent nested `<dir>/<dir>/profile.*`.
> `tau_treemerge.pl`/`tau2otf2` (used for `TRACE=1`) are the opposite: they don't
> consult `$TRACEDIR` at all and must be run from inside the experiment directory.

For the GUI (`paraprof`, per-call bar charts + the communication matrix), or to convert a `TRACE=1` run to OTF2 and view it in a trace viewer, open a remote graphical session (`man aac6_vnc` / `man aac6_novnc` / `man aac6_x11`) — `paraprof` needs a JRE, which may not be installed on every login node.

---

## Requirements

- ROCm ≥ 6.3 (rocSPARSE, rocBLAS, hipcc)
- GPU-Aware MPI (OpenMPI/UCX built with ROCm support — see module list on
  the login banner)
- TAU (`module load tau/dev`, loaded after `rocm`) for MPI+ROCm profiling/tracing
