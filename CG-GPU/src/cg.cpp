// =============================================================================
// Distributed GPU Conjugate Gradient — five communication variants
//
// Select a parallel SpMV communication scheme at run-time:
//
//   ./cg_gpu matrix.pm staged            (default) Isend/Irecv through CPU buffers
//   ./cg_gpu matrix.pm isend                       Isend/Irecv with GPU buffers (GPU-Aware)
//   ./cg_gpu matrix.pm rccl                        RCCL ncclSend/ncclRecv, GPU buffers
//   ./cg_gpu matrix.pm alltoallv_staged            MPI_Alltoallv through CPU buffers
//   ./cg_gpu matrix.pm alltoallv                   MPI_Alltoallv with GPU buffers (GPU-Aware)
//
// All five variants produce identical numerical output.  The differences are
// purely in how ghost values are exchanged between MPI ranks.
//
// Requirements:
//   ROCm (hipcc, rocSPARSE, rocBLAS, RCCL), GPU-Aware OpenMPI
// =============================================================================

#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <rccl/rccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>

// =============================================================================
// Error-checking macros
// =============================================================================
#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t _e = (call);                                              \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error '%s' at %s:%d\n",                   \
                    hipGetErrorString(_e), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, -1);                                  \
        }                                                                    \
    } while (0)

#define ROCSPARSE_CHECK(call)                                                \
    do {                                                                     \
        rocsparse_status _s = (call);                                        \
        if (_s != rocsparse_status_success) {                                \
            fprintf(stderr, "rocSPARSE error %d at %s:%d\n",                \
                    (int)_s, __FILE__, __LINE__);                            \
            MPI_Abort(MPI_COMM_WORLD, -1);                                  \
        }                                                                    \
    } while (0)

#define ROCBLAS_CHECK(call)                                                  \
    do {                                                                     \
        rocblas_status _s = (call);                                          \
        if (_s != rocblas_status_success) {                                  \
            fprintf(stderr, "rocBLAS error %d at %s:%d\n",                  \
                    (int)_s, __FILE__, __LINE__);                            \
            MPI_Abort(MPI_COMM_WORLD, -1);                                  \
        }                                                                    \
    } while (0)

#define RCCL_CHECK(call)                                                     \
    do {                                                                     \
        ncclResult_t _r = (call);                                            \
        if (_r != ncclSuccess) {                                             \
            fprintf(stderr, "RCCL error '%s' at %s:%d\n",                  \
                    ncclGetErrorString(_r), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, -1);                                  \
        }                                                                    \
    } while (0)

// =============================================================================
// upload_mat — copy one CPU sparse block to GPU + build rocSPARSE state
// =============================================================================
static void upload_mat(const Mat& cpu, GPUMat& gpu,
                       rocsparse_handle handle,
                       int vec_x_size, int vec_y_size)
{
    gpu.n_rows = cpu.n_rows;
    gpu.n_cols = cpu.n_cols;
    gpu.nnz    = cpu.nnz;

    if (cpu.nnz == 0) {
        gpu.d_rowptr = nullptr;  gpu.d_colidx = nullptr;  gpu.d_data = nullptr;
        gpu.descr    = nullptr;  gpu.d_spmv_buf = nullptr; gpu.spmv_buf_size = 0;
        return;
    }

    HIP_CHECK(hipMalloc(&gpu.d_rowptr, (cpu.n_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu.d_colidx,  cpu.nnz        * sizeof(int)));
    HIP_CHECK(hipMalloc(&gpu.d_data,    cpu.nnz        * sizeof(double)));

    HIP_CHECK(hipMemcpy(gpu.d_rowptr, cpu.rowptr.data(),
                        (cpu.n_rows + 1) * sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu.d_colidx, cpu.col_idx.data(),
                        cpu.nnz * sizeof(int),             hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu.d_data,   cpu.data.data(),
                        cpu.nnz * sizeof(double),          hipMemcpyHostToDevice));

    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &gpu.descr, gpu.n_rows, gpu.n_cols, gpu.nnz,
        gpu.d_rowptr, gpu.d_colidx, gpu.d_data,
        rocsparse_indextype_i32, rocsparse_indextype_i32,
        rocsparse_index_base_zero, rocsparse_datatype_f64_r));

    // Query rocSPARSE SpMV workspace size via temporary vector descriptors
    double *d_tmp_x, *d_tmp_y;
    HIP_CHECK(hipMalloc(&d_tmp_x, vec_x_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tmp_y, vec_y_size * sizeof(double)));

    rocsparse_dnvec_descr tmp_x, tmp_y;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &tmp_x, vec_x_size, d_tmp_x, rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &tmp_y, vec_y_size, d_tmp_y, rocsparse_datatype_f64_r));

    const double one = 1.0, zero = 0.0;
    ROCSPARSE_CHECK(rocsparse_spmv(
        handle, rocsparse_operation_none,
        &one, gpu.descr, tmp_x, &zero, tmp_y,
        rocsparse_datatype_f64_r, rocsparse_spmv_alg_default,
        rocsparse_spmv_stage_buffer_size, &gpu.spmv_buf_size, nullptr));

    HIP_CHECK(hipMalloc(&gpu.d_spmv_buf, gpu.spmv_buf_size));

    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(tmp_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(tmp_y));
    HIP_CHECK(hipFree(d_tmp_x));
    HIP_CHECK(hipFree(d_tmp_y));
}

// =============================================================================
// upload_par_mat — upload the full parallel matrix + all comm scheme state
// =============================================================================
static void upload_par_mat(const ParMat& cpu, GPUParMat& gpu,
                           rocsparse_handle handle, int rank, int num_procs)
{
    upload_mat(cpu.on_proc,  gpu.on_proc,  handle,
               cpu.local_cols, cpu.local_rows);
    upload_mat(cpu.off_proc, gpu.off_proc, handle,
               cpu.off_proc_num_cols, cpu.local_rows);

    // ── GPU comm buffers (variants 1, 3, 4) ──────────────────────────────────
    if (cpu.send_comm.size_msgs > 0) {
        HIP_CHECK(hipMalloc(&gpu.d_sendbuf,
                            cpu.send_comm.size_msgs * sizeof(double)));
        HIP_CHECK(hipMalloc(&gpu.d_send_idx,
                            cpu.send_comm.size_msgs * sizeof(int)));
        HIP_CHECK(hipMemcpy(gpu.d_send_idx,
                            cpu.send_comm.idx.data(),
                            cpu.send_comm.size_msgs * sizeof(int),
                            hipMemcpyHostToDevice));
    } else {
        gpu.d_sendbuf  = nullptr;
        gpu.d_send_idx = nullptr;
    }

    if (cpu.recv_comm.size_msgs > 0)
        HIP_CHECK(hipMalloc(&gpu.d_recvbuf,
                            cpu.recv_comm.size_msgs * sizeof(double)));
    else
        gpu.d_recvbuf = nullptr;

    // ── Pinned host buffers (variant 2: staged) ───────────────────────────────
    // hipHostMalloc allocates page-locked (pinned) memory, which enables fast
    // DMA transfers without extra staging inside the driver.
    if (cpu.send_comm.size_msgs > 0)
        HIP_CHECK(hipHostMalloc(&gpu.h_sendbuf,
                                cpu.send_comm.size_msgs * sizeof(double),
                                hipHostMallocDefault));
    else
        gpu.h_sendbuf = nullptr;

    if (cpu.recv_comm.size_msgs > 0)
        HIP_CHECK(hipHostMalloc(&gpu.h_recvbuf,
                                cpu.recv_comm.size_msgs * sizeof(double),
                                hipHostMallocDefault));
    else
        gpu.h_recvbuf = nullptr;

    // ── MPI_Alltoallv arrays (variant 3) ─────────────────────────────────────
    // Expand the sparse send/recv comm pattern into full per-rank arrays.
    gpu.a2a_sendcounts.assign(num_procs, 0);
    gpu.a2a_sdispls   .assign(num_procs, 0);
    gpu.a2a_recvcounts.assign(num_procs, 0);
    gpu.a2a_rdispls   .assign(num_procs, 0);

    for (int i = 0; i < cpu.send_comm.n_msgs; i++) {
        int p = cpu.send_comm.procs[i];
        gpu.a2a_sendcounts[p] = cpu.send_comm.ptr[i+1] - cpu.send_comm.ptr[i];
        gpu.a2a_sdispls[p]    = cpu.send_comm.ptr[i];
    }
    for (int i = 0; i < cpu.recv_comm.n_msgs; i++) {
        int p = cpu.recv_comm.procs[i];
        gpu.a2a_recvcounts[p] = cpu.recv_comm.ptr[i+1] - cpu.recv_comm.ptr[i];
        gpu.a2a_rdispls[p]    = cpu.recv_comm.ptr[i];
    }

    // ── RCCL communicator + stream (variant 4) ────────────────────────────────
    // All ranks share the same ncclUniqueId, broadcast from rank 0.
    ncclUniqueId nccl_id;
    if (rank == 0) RCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    RCCL_CHECK(ncclCommInitRank(&gpu.rccl_comm, num_procs, nccl_id, rank));

    // Dedicated HIP stream for RCCL so it can overlap with GPU compute
    HIP_CHECK(hipStreamCreate(&gpu.rccl_stream));

    // CPU comm pointers
    gpu.send_comm = const_cast<Comm*>(&cpu.send_comm);
    gpu.recv_comm = const_cast<Comm*>(&cpu.recv_comm);
}

// =============================================================================
// spmv (single block) — b = alpha * A * x + beta * b  on the GPU
// =============================================================================
static void spmv(double alpha, GPUMat& A, double* d_x,
                 double beta,  double* d_b,
                 rocsparse_handle handle)
{
    if (A.nnz == 0) return;

    rocsparse_dnvec_descr vec_x, vec_b;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_x, A.n_cols, d_x, rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_b, A.n_rows, d_b, rocsparse_datatype_f64_r));

    ROCSPARSE_CHECK(rocsparse_spmv(
        handle, rocsparse_operation_none,
        &alpha, A.descr, vec_x, &beta, vec_b,
        rocsparse_datatype_f64_r, rocsparse_spmv_alg_default,
        rocsparse_spmv_stage_compute, &A.spmv_buf_size, A.d_spmv_buf));

    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_b));
}

// =============================================================================
// Variant 1 — staged through CPU (host) buffers
//
// The simplest baseline: all packing, communication, and unpacking touch the
// CPU.  MPI operates on host pointers.  Two extra GPU↔CPU copies per SpMV
// show the overhead that GPU-Aware MPI (variants 2, 3, 5) eliminates.
// =============================================================================
static void spmv_staged(double alpha, GPUParMat& A, double* d_x,
                         double beta,  double* d_b,
                         rocsparse_handle handle)
{
    int   tag  = 0;
    Comm& recv = *A.recv_comm;
    Comm& send = *A.send_comm;

    // Copy full local vector from GPU to a temporary host buffer
    int n_cols = A.on_proc.n_cols;
    std::vector<double> h_x(n_cols);
    HIP_CHECK(hipMemcpy(h_x.data(), d_x,
                        n_cols * sizeof(double), hipMemcpyDeviceToHost));

    // Pack send buffer on CPU
    for (int i = 0; i < send.size_msgs; i++)
        A.h_sendbuf[i] = h_x[send.idx[i]];

    // Post receives and sends into host memory
    for (int i = 0; i < recv.n_msgs; i++)
        MPI_Irecv(A.h_recvbuf + recv.ptr[i],
                  recv.ptr[i+1] - recv.ptr[i],
                  MPI_DOUBLE, recv.procs[i], tag,
                  MPI_COMM_WORLD, &recv.req[i]);

    for (int i = 0; i < send.n_msgs; i++)
        MPI_Isend(A.h_sendbuf + send.ptr[i],
                  send.ptr[i+1] - send.ptr[i],
                  MPI_DOUBLE, send.procs[i], tag,
                  MPI_COMM_WORLD, &send.req[i]);

    // On-proc SpMV overlaps with in-flight communication
    spmv(alpha, A.on_proc, d_x, beta, d_b, handle);

    if (recv.n_msgs)
        MPI_Waitall(recv.n_msgs, recv.req.data(), MPI_STATUSES_IGNORE);

    // Copy received ghost values from host to GPU, then run off-proc SpMV
    if (recv.size_msgs > 0)
        HIP_CHECK(hipMemcpy(A.d_recvbuf, A.h_recvbuf,
                            recv.size_msgs * sizeof(double),
                            hipMemcpyHostToDevice));

    spmv(alpha, A.off_proc, A.d_recvbuf, 1.0, d_b, handle);

    if (send.n_msgs)
        MPI_Waitall(send.n_msgs, send.req.data(), MPI_STATUSES_IGNORE);
}

// =============================================================================
// Variant 2 — non-blocking Isend/Irecv with GPU buffers (GPU-Aware MPI)
//
// Ghost values never touch the CPU:
//   • rocsparse_dgthr gathers the send values on the GPU
//   • MPI_Isend/Irecv operate directly on GPU pointers
//   • on-proc SpMV overlaps with in-flight MPI messages
//
// Compare with variant 1 (staged) to see the GPU↔CPU copy cost directly.
// =============================================================================
static void spmv_isend_irecv(double alpha, GPUParMat& A, double* d_x,
                              double beta,  double* d_b,
                              rocsparse_handle handle)
{
    int   tag  = 0;
    Comm& recv = *A.recv_comm;
    Comm& send = *A.send_comm;

    // Post receives into GPU memory
    for (int i = 0; i < recv.n_msgs; i++)
        MPI_Irecv(A.d_recvbuf + recv.ptr[i],
                  recv.ptr[i+1] - recv.ptr[i],
                  MPI_DOUBLE, recv.procs[i], tag,
                  MPI_COMM_WORLD, &recv.req[i]);

    // Gather send values on GPU:  d_sendbuf[i] = d_x[ d_send_idx[i] ]
    if (send.size_msgs > 0) {
        ROCSPARSE_CHECK(rocsparse_dgthr(
            handle, send.size_msgs, d_x, A.d_sendbuf, A.d_send_idx,
            rocsparse_index_base_zero));
        HIP_CHECK(hipDeviceSynchronize());   // gather must finish before MPI reads d_sendbuf
    }

    // Post sends from GPU memory
    for (int i = 0; i < send.n_msgs; i++)
        MPI_Isend(A.d_sendbuf + send.ptr[i],
                  send.ptr[i+1] - send.ptr[i],
                  MPI_DOUBLE, send.procs[i], tag,
                  MPI_COMM_WORLD, &send.req[i]);

    // On-proc SpMV overlaps with in-flight communication
    spmv(alpha, A.on_proc, d_x, beta, d_b, handle);

    if (recv.n_msgs)
        MPI_Waitall(recv.n_msgs, recv.req.data(), MPI_STATUSES_IGNORE);

    spmv(alpha, A.off_proc, A.d_recvbuf, 1.0, d_b, handle);

    if (send.n_msgs)
        MPI_Waitall(send.n_msgs, send.req.data(), MPI_STATUSES_IGNORE);
}

// =============================================================================
// Variant 3 — RCCL ncclSend / ncclRecv
//
// RCCL (ROCm Collective Communications Library) provides GPU-native point-to-
// point communication.  On MI300A (unified memory) and multi-GPU nodes it can
// use Infinity Fabric / NVLink paths that bypass the CPU entirely.
//
// Overlap: RCCL runs on rccl_stream while on-proc SpMV runs on the default
// GPU stream — true GPU-GPU overlap between communication and computation.
// =============================================================================
static void spmv_rccl(double alpha, GPUParMat& A, double* d_x,
                       double beta,  double* d_b,
                       rocsparse_handle handle)
{
    Comm& recv = *A.recv_comm;
    Comm& send = *A.send_comm;

    // Gather send values on the default GPU stream
    if (send.size_msgs > 0) {
        ROCSPARSE_CHECK(rocsparse_dgthr(
            handle, send.size_msgs, d_x, A.d_sendbuf, A.d_send_idx,
            rocsparse_index_base_zero));
        // Sync so that rccl_stream sees the gathered values.
        // For finer-grained sync, an event + hipStreamWaitEvent can replace
        // this full device sync.
        HIP_CHECK(hipDeviceSynchronize());
    }

    // Launch RCCL sends and receives as a single grouped operation on rccl_stream
    RCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < recv.n_msgs; i++)
        RCCL_CHECK(ncclRecv(A.d_recvbuf + recv.ptr[i],
                            recv.ptr[i+1] - recv.ptr[i],
                            ncclDouble, recv.procs[i],
                            A.rccl_comm, A.rccl_stream));
    for (int i = 0; i < send.n_msgs; i++)
        RCCL_CHECK(ncclSend(A.d_sendbuf + send.ptr[i],
                            send.ptr[i+1] - send.ptr[i],
                            ncclDouble, send.procs[i],
                            A.rccl_comm, A.rccl_stream));
    RCCL_CHECK(ncclGroupEnd());   // kicks off all sends/recvs on rccl_stream

    // On-proc SpMV on the default stream — overlaps with RCCL on rccl_stream
    spmv(alpha, A.on_proc, d_x, beta, d_b, handle);

    // Wait for RCCL to deliver ghost values before off-proc SpMV reads them
    HIP_CHECK(hipStreamSynchronize(A.rccl_stream));

    spmv(alpha, A.off_proc, A.d_recvbuf, 1.0, d_b, handle);
}

// =============================================================================
// Variant 4 — MPI_Alltoallv with explicit host staging
//
// Same collective structure as variant 5 (GPU-Aware MPI_Alltoallv), but the
// send/recv buffers are explicitly copied through pinned host memory before and
// after the call — the "classic" approach without GPU-Aware MPI.
//
// Comparing variant 4 vs. variant 5 isolates the PCIe round-trip cost
// (D→H + H→D) that GPU-Aware MPI avoids by passing device pointers directly.
//
// Overlap: the on-proc SpMV is submitted to the GPU before the CPU blocks in
// MPI_Alltoallv, so diagonal computation and communication run in parallel.
// =============================================================================
static void spmv_alltoallv_staged(double alpha, GPUParMat& A, double* d_x,
                                   double beta,  double* d_b,
                                   rocsparse_handle handle)
{
    Comm& send = *A.send_comm;

    // Gather send values on GPU into d_sendbuf, then copy to pinned host buffer
    if (send.size_msgs > 0) {
        ROCSPARSE_CHECK(rocsparse_dgthr(
            handle, send.size_msgs, d_x, A.d_sendbuf, A.d_send_idx,
            rocsparse_index_base_zero));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(A.h_sendbuf, A.d_sendbuf,
                            send.size_msgs * sizeof(double),
                            hipMemcpyDeviceToHost));
    }

    // Submit on-proc SpMV to GPU (async); CPU continues to MPI_Alltoallv
    spmv(alpha, A.on_proc, d_x, beta, d_b, handle);

    // Exchange ghost values via host buffers (non-GPU-Aware MPI).
    // While the CPU blocks here, the GPU runs on-proc SpMV in parallel.
    MPI_Alltoallv(A.h_sendbuf, A.a2a_sendcounts.data(), A.a2a_sdispls.data(),
                  MPI_DOUBLE,
                  A.h_recvbuf, A.a2a_recvcounts.data(), A.a2a_rdispls.data(),
                  MPI_DOUBLE, MPI_COMM_WORLD);

    // Copy ghost values pinned host → GPU recv buffer
    Comm& recv = *A.recv_comm;
    if (recv.size_msgs > 0)
        HIP_CHECK(hipMemcpy(A.d_recvbuf, A.h_recvbuf,
                            recv.size_msgs * sizeof(double),
                            hipMemcpyHostToDevice));

    spmv(alpha, A.off_proc, A.d_recvbuf, 1.0, d_b, handle);
}

// =============================================================================
// Variant 5 — MPI_Alltoallv with GPU buffers (GPU-Aware MPI)
//
// Replaces many point-to-point messages with one collective call.  The
// send/receive buffer layout is pre-computed as per-rank count/displacement
// arrays.  Buffers are GPU pointers — no host copies needed.
//
// Overlap: on-proc SpMV is submitted to the GPU before MPI_Alltoallv blocks
// the CPU, so the GPU computes the diagonal contribution while the CPU waits
// for the collective to complete.
// =============================================================================
static void spmv_alltoallv(double alpha, GPUParMat& A, double* d_x,
                            double beta,  double* d_b,
                            rocsparse_handle handle)
{
    Comm& send = *A.send_comm;

    // Gather send values on GPU
    if (send.size_msgs > 0) {
        ROCSPARSE_CHECK(rocsparse_dgthr(
            handle, send.size_msgs, d_x, A.d_sendbuf, A.d_send_idx,
            rocsparse_index_base_zero));
        HIP_CHECK(hipDeviceSynchronize());
    }

    // Submit on-proc SpMV to GPU (async); CPU continues to MPI_Alltoallv
    spmv(alpha, A.on_proc, d_x, beta, d_b, handle);

    // Exchange all ghost values in one collective (GPU-Aware MPI).
    // While the CPU is blocked here, the GPU runs on-proc SpMV in parallel.
    MPI_Alltoallv(A.d_sendbuf, A.a2a_sendcounts.data(), A.a2a_sdispls.data(),
                  MPI_DOUBLE,
                  A.d_recvbuf, A.a2a_recvcounts.data(), A.a2a_rdispls.data(),
                  MPI_DOUBLE, MPI_COMM_WORLD);

    spmv(alpha, A.off_proc, A.d_recvbuf, 1.0, d_b, handle);
}

// =============================================================================
// inner_product — global dot(a,b) via rocBLAS + MPI_Allreduce
// =============================================================================
static double inner_product(double* d_a, double* d_b, int n,
                            rocblas_handle blas_handle)
{
    double local_sum;
    ROCBLAS_CHECK(rocblas_ddot(blas_handle, n, d_a, 1, d_b, 1, &local_sum));
    HIP_CHECK(hipDeviceSynchronize());
    MPI_Allreduce(MPI_IN_PLACE, &local_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return local_sum;
}

static void axpy(double alpha, double* d_x, double* d_y, int n,
                 rocblas_handle h)
{
    ROCBLAS_CHECK(rocblas_daxpy(h, n, &alpha, d_y, 1, d_x, 1));
}

static void scale(double alpha, double* d_x, int n, rocblas_handle h)
{
    ROCBLAS_CHECK(rocblas_dscal(h, n, &alpha, d_x, 1));
}

static void free_gpu_mat(GPUMat& gpu)
{
    if (gpu.nnz == 0) return;
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(gpu.descr));
    HIP_CHECK(hipFree(gpu.d_rowptr));
    HIP_CHECK(hipFree(gpu.d_colidx));
    HIP_CHECK(hipFree(gpu.d_data));
    HIP_CHECK(hipFree(gpu.d_spmv_buf));
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // -------------------------------------------------------------------------
    // Parse arguments
    // -------------------------------------------------------------------------
    const char* filename = "Dubcova2.pm";
    if (argc > 1) filename = argv[1];

    // Choose communication variant (default: staged)
    const char* method = (argc > 2) ? argv[2] : "staged";

    // Map the method string to a function pointer
    using spmv_fn = void(*)(double, GPUParMat&, double*, double, double*,
                             rocsparse_handle);
    spmv_fn par_spmv = nullptr;

    if      (!strcmp(method, "staged"))           par_spmv = spmv_staged;
    else if (!strcmp(method, "isend"))            par_spmv = spmv_isend_irecv;
    else if (!strcmp(method, "rccl"))             par_spmv = spmv_rccl;
    else if (!strcmp(method, "alltoallv_staged")) par_spmv = spmv_alltoallv_staged;
    else if (!strcmp(method, "alltoallv"))        par_spmv = spmv_alltoallv;
    else {
        if (rank == 0)
            fprintf(stderr,
                "Unknown method '%s'.\n"
                "Choose: staged | isend | rccl | alltoallv_staged | alltoallv\n",
                method);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // -------------------------------------------------------------------------
    // GPU setup: assign one GPU per MPI rank
    // -------------------------------------------------------------------------
    int num_gpus;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    HIP_CHECK(hipSetDevice(rank % num_gpus));

    if (rank == 0)
        printf("method=%s  ranks=%d  gpus_visible=%d\n",
               method, num_procs, num_gpus);

    rocsparse_handle sparse_handle;
    rocblas_handle   blas_handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&sparse_handle));
    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle));

    // -------------------------------------------------------------------------
    // Read matrix and form MPI communication pattern
    // -------------------------------------------------------------------------
    ParMat A;
    readParMatrix(filename, A);
    form_comm(A);

    // -------------------------------------------------------------------------
    // Upload matrix + all communication scheme state to GPU
    // -------------------------------------------------------------------------
    GPUParMat gA;
    upload_par_mat(A, gA, sparse_handle, rank, num_procs);

    // -------------------------------------------------------------------------
    // Allocate device vectors
    // -------------------------------------------------------------------------
    int n = A.local_rows;

    double *d_x, *d_b;
    HIP_CHECK(hipMalloc(&d_x, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, n * sizeof(double)));

    srand(time(NULL) + rank);
    std::vector<double> h_x(n);
    std::generate(h_x.begin(), h_x.end(), [](){ return (double)rand() / RAND_MAX; });
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), n * sizeof(double), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(d_b, 0, n * sizeof(double)));
    par_spmv(1.0, gA, d_x, 0.0, d_b, sparse_handle);      // b = A * x_random
    HIP_CHECK(hipMemset(d_x, 0, n * sizeof(double)));       // x = 0 (initial guess)

    // -------------------------------------------------------------------------
    // CG variables
    // -------------------------------------------------------------------------
    double *d_r, *d_p, *d_Ap;
    HIP_CHECK(hipMalloc(&d_r,  n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_p,  n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Ap, n * sizeof(double)));

    HIP_CHECK(hipMemcpy(d_r, d_b, n * sizeof(double), hipMemcpyDeviceToDevice));
    par_spmv(-1.0, gA, d_x, 1.0, d_r, sparse_handle);     // r = b - A*x

    HIP_CHECK(hipMemcpy(d_p, d_r, n * sizeof(double), hipMemcpyDeviceToDevice));

    double rr_inner = inner_product(d_r, d_r, n, blas_handle);
    double norm_r   = sqrt(rr_inner);
    double tol      = 1e-6;

    if (rank == 0) printf("Initial residual: %lg\n", norm_r);
    if (norm_r != 0.0) tol *= norm_r;

    int recompute_r = 8;
    int max_iter    = (int)(1.3 * A.global_rows) + 2;
    int iter        = 0;

    // -------------------------------------------------------------------------
    // CG loop
    // -------------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);          // synchronize all ranks before timing
    double t_start = MPI_Wtime();

    while (norm_r > tol && iter < max_iter)
    {
        par_spmv(1.0, gA, d_p, 0.0, d_Ap, sparse_handle);

        double App_inner = inner_product(d_Ap, d_p, n, blas_handle);
        if (App_inner < 0.0) {
            if (rank == 0) printf("Indefinite matrix! Aborting.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        double alpha = rr_inner / App_inner;

        axpy(alpha, d_x, d_p, n, blas_handle);

        if ((iter % recompute_r) && iter > 0) {
            axpy(-alpha, d_r, d_Ap, n, blas_handle);
        } else {
            HIP_CHECK(hipMemcpy(d_r, d_b, n * sizeof(double), hipMemcpyDeviceToDevice));
            par_spmv(-1.0, gA, d_x, 1.0, d_r, sparse_handle);
        }

        double next_inner = inner_product(d_r, d_r, n, blas_handle);
        double beta       = next_inner / rr_inner;

        scale(beta, d_p, n, blas_handle);
        axpy(1.0, d_p, d_r, n, blas_handle);

        rr_inner = next_inner;
        norm_r   = sqrt(rr_inner);
        iter++;
    }

    HIP_CHECK(hipDeviceSynchronize());    // ensure all GPU work is done before stopping timer
    double t_elapsed = MPI_Wtime() - t_start;

    // Reduce to the maximum time across all ranks (the slowest rank sets the wall time)
    double t_max;
    MPI_Reduce(&t_elapsed, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (iter == max_iter) printf("Max iterations reached.\n");
        else                   printf("%d iterations to converge\n", iter);
        printf("2-norm of residual: %lg\n", norm_r);
        printf("CG solve time:      %.4f s  (%.4f s/iter)\n",
               t_max, t_max / iter);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    HIP_CHECK(hipFree(d_x));  HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_r));  HIP_CHECK(hipFree(d_p));  HIP_CHECK(hipFree(d_Ap));

    free_gpu_mat(gA.on_proc);
    free_gpu_mat(gA.off_proc);

    if (gA.d_sendbuf)  HIP_CHECK(hipFree(gA.d_sendbuf));
    if (gA.d_recvbuf)  HIP_CHECK(hipFree(gA.d_recvbuf));
    if (gA.d_send_idx) HIP_CHECK(hipFree(gA.d_send_idx));
    if (gA.h_sendbuf)  HIP_CHECK(hipHostFree(gA.h_sendbuf));
    if (gA.h_recvbuf)  HIP_CHECK(hipHostFree(gA.h_recvbuf));

    RCCL_CHECK(ncclCommDestroy(gA.rccl_comm));
    HIP_CHECK(hipStreamDestroy(gA.rccl_stream));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(sparse_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(blas_handle));

    MPI_Finalize();
    return 0;
}
