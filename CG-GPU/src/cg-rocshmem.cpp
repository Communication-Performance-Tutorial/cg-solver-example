// =============================================================================
// Distributed GPU Conjugate Gradient — rocSHMEM ghost exchange
//
// Same CG solver as cg.cpp, but ghost values are exchanged with rocSHMEM
// one-sided puts (rocshmem_putmem_on_stream) into symmetric GPU memory.
//
// Usage:
//   mpirun -n 4 ./cg_gpu_rocshmem src/Dubcova2.pm
//
// All runtime communication (ghost exchange, dot-product reduction, offset
// setup, timing barrier/reduction) uses rocSHMEM. MPI is only used for
// process bootstrap and the CPU-side matrix I/O / comm-pattern setup shared
// with every other CG-GPU variant (sparse_mat.hpp, par_binary_IO.hpp).
//
// Requirements:
//   ROCm (hipcc, rocSPARSE, rocBLAS), rocSHMEM, GPU-Aware OpenMPI (bootstrap
//   and matrix setup only)
// =============================================================================

#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <rocshmem/rocshmem.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>

using namespace rocshmem;

// Not exposed in the public rocshmem.hpp, but exported by librocshmem: the
// host-side context the library itself uses for its no-ctx host wrappers
// (e.g. rocshmem_barrier_all). We reuse it instead of calling
// rocshmem_ctx_create(), whose default host-context pool (MAX_NUM_HOST_
// CONTEXTS=1) is already consumed by this very context.
namespace rocshmem { extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT; }

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

#define ROCSHMEM_CHECK(call)                                                 \
    do {                                                                     \
        int _r = (call);                                                     \
        if (_r != ROCSHMEM_SUCCESS) {                                        \
            fprintf(stderr, "rocSHMEM error %d at %s:%d\n",                 \
                    _r, __FILE__, __LINE__);                                 \
            MPI_Abort(MPI_COMM_WORLD, -1);                                  \
        }                                                                    \
    } while (0)

// =============================================================================
// GPUParMat — rocSHMEM variant (symmetric comm buffers on the GPU)
// =============================================================================
struct RocshmemGPUParMat {
    GPUMat  on_proc;
    GPUMat  off_proc;

    double *d_sendbuf;     // rocshmem_malloc — packed values to send
    double *d_recvbuf;     // rocshmem_malloc — ghost values received
    int    *d_send_idx;    // hipMalloc — gather indices for rocsparse_dgthr

    // Offset in each remote PE's d_recvbuf where our puts must land.
    std::vector<int> send_dest_off;

    hipStream_t comm_stream;

    Comm *send_comm;
    Comm *recv_comm;
};

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
// exchange_put_displs — learn remote recv offsets for rocSHMEM puts
//
// Uses a single rocshmem_alltoallmem_on_stream instead of per-neighbour MPI
// messages. Each PE contributes one int per peer: the offset in its own
// recv buffer where ghost data from that peer lands (-1 if none). After the
// alltoall, PE i's block for peer j equals PE j's contributed block for PE
// i — i.e. exactly the offset PE i must use when it later puts data into
// PE j's recv buffer.
// =============================================================================
static void exchange_put_displs(const ParMat& A, int num_procs,
                                 hipStream_t stream,
                                 std::vector<int>& send_dest_off)
{
    send_dest_off.assign(A.send_comm.n_msgs, 0);

    std::vector<int> my_offsets(num_procs, -1);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
        my_offsets[A.recv_comm.procs[i]] = A.recv_comm.ptr[i];

    int *d_src = static_cast<int *>(rocshmem_malloc(num_procs * sizeof(int)));
    int *d_dst = static_cast<int *>(rocshmem_malloc(num_procs * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_src, my_offsets.data(), num_procs * sizeof(int),
                        hipMemcpyHostToDevice));

    rocshmem_alltoallmem_on_stream(ROCSHMEM_TEAM_WORLD, d_dst, d_src,
                                   sizeof(int), stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<int> peer_offsets(num_procs);
    HIP_CHECK(hipMemcpy(peer_offsets.data(), d_dst, num_procs * sizeof(int),
                        hipMemcpyDeviceToHost));

    rocshmem_free(d_src);
    rocshmem_free(d_dst);

    for (int j = 0; j < A.send_comm.n_msgs; j++)
        send_dest_off[j] = peer_offsets[A.send_comm.procs[j]];
}

// =============================================================================
// upload_par_mat — upload matrix + rocSHMEM symmetric comm buffers
// =============================================================================
static void upload_par_mat(const ParMat& cpu, RocshmemGPUParMat& gpu,
                           rocsparse_handle handle, rocshmem_ctx_t ctx,
                           int num_procs)
{
    upload_mat(cpu.on_proc,  gpu.on_proc,  handle,
               cpu.local_cols, cpu.local_rows);
    upload_mat(cpu.off_proc, gpu.off_proc, handle,
               cpu.off_proc_num_cols, cpu.local_rows);

    // rocshmem_malloc is collective and must request the same size on every
    // PE to keep symmetric-heap offsets aligned across ranks. Ghost-buffer
    // sizes vary with partitioning, so allocate the global max (via a
    // rocSHMEM max-reduce) and use only the local prefix of each buffer.
    int local_send = cpu.send_comm.size_msgs;
    int local_recv = cpu.recv_comm.size_msgs;

    int *d_local  = static_cast<int *>(rocshmem_malloc(sizeof(int)));
    int *d_global = static_cast<int *>(rocshmem_malloc(sizeof(int)));

    HIP_CHECK(hipMemcpy(d_local, &local_send, sizeof(int), hipMemcpyHostToDevice));
    ROCSHMEM_CHECK(rocshmem_ctx_int_max_reduce(ctx, ROCSHMEM_TEAM_WORLD, d_global, d_local, 1));
    int max_send;
    HIP_CHECK(hipMemcpy(&max_send, d_global, sizeof(int), hipMemcpyDeviceToHost));

    HIP_CHECK(hipMemcpy(d_local, &local_recv, sizeof(int), hipMemcpyHostToDevice));
    ROCSHMEM_CHECK(rocshmem_ctx_int_max_reduce(ctx, ROCSHMEM_TEAM_WORLD, d_global, d_local, 1));
    int max_recv;
    HIP_CHECK(hipMemcpy(&max_recv, d_global, sizeof(int), hipMemcpyDeviceToHost));

    rocshmem_free(d_local);
    rocshmem_free(d_global);

    gpu.d_sendbuf = (max_send > 0)
        ? static_cast<double *>(rocshmem_malloc(max_send * sizeof(double)))
        : nullptr;
    if (max_send > 0 && !gpu.d_sendbuf) {
        fprintf(stderr, "rocshmem_malloc failed for send buffer\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    gpu.d_recvbuf = (max_recv > 0)
        ? static_cast<double *>(rocshmem_malloc(max_recv * sizeof(double)))
        : nullptr;
    if (max_recv > 0 && !gpu.d_recvbuf) {
        fprintf(stderr, "rocshmem_malloc failed for recv buffer\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (local_send > 0) {
        HIP_CHECK(hipMalloc(&gpu.d_send_idx, local_send * sizeof(int)));
        HIP_CHECK(hipMemcpy(gpu.d_send_idx,
                            cpu.send_comm.idx.data(),
                            local_send * sizeof(int),
                            hipMemcpyHostToDevice));
    } else {
        gpu.d_send_idx = nullptr;
    }

    HIP_CHECK(hipStreamCreate(&gpu.comm_stream));

    exchange_put_displs(cpu, num_procs, gpu.comm_stream, gpu.send_dest_off);

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
// spmv_rocshmem — ghost exchange via rocSHMEM one-sided puts
//
// 1. Gather send values on the GPU (rocsparse_dgthr → symmetric send buffer)
// 2. Issue rocshmem_putmem_on_stream for each neighbour on comm_stream
// 3. Overlap on-proc SpMV on the default stream
// 4. Quiet the rocSHMEM stream (local put completion), then barrier across
//    all PEs (global put completion) before reading the recv buffer
// =============================================================================
static void spmv_rocshmem(double alpha, RocshmemGPUParMat& A, double* d_x,
                          double beta,  double* d_b,
                          rocsparse_handle handle)
{
    Comm& send = *A.send_comm;

    if (send.size_msgs > 0) {
        ROCSPARSE_CHECK(rocsparse_dgthr(
            handle, send.size_msgs, d_x, A.d_sendbuf, A.d_send_idx,
            rocsparse_index_base_zero));
        HIP_CHECK(hipDeviceSynchronize());
    }

    for (int j = 0; j < send.n_msgs; j++) {
        int pe = send.procs[j];
        size_t bytes = static_cast<size_t>(send.ptr[j + 1] - send.ptr[j])
                     * sizeof(double);
        char *src  = reinterpret_cast<char *>(A.d_sendbuf)
                   + static_cast<size_t>(send.ptr[j]) * sizeof(double);
        char *dest = reinterpret_cast<char *>(A.d_recvbuf)
                   + static_cast<size_t>(A.send_dest_off[j]) * sizeof(double);
        rocshmem_putmem_on_stream(dest, src, bytes, pe, A.comm_stream);
    }

    spmv(alpha, A.on_proc, d_x, beta, d_b, handle);

    rocshmem_quiet_on_stream(A.comm_stream);
    HIP_CHECK(hipStreamSynchronize(A.comm_stream));

    // quiet only guarantees *this* PE's outgoing puts have landed; a barrier
    // is required so every PE waits until *all* neighbours' puts (into this
    // PE's recv buffer) have also landed before it is safe to read them.
    rocshmem_barrier_all();

    spmv(alpha, A.off_proc, A.d_recvbuf, 1.0, d_b, handle);
}

// =============================================================================
// inner_product — global dot(a,b) via rocBLAS + rocSHMEM sum-reduce
//
// d_reduce_in/d_reduce_out are persistent, single-element symmetric-heap
// scalars (rocshmem_malloc'd once in main) reused across every call.
// =============================================================================
static double inner_product(double* d_a, double* d_b, int n,
                            rocblas_handle blas_handle, rocshmem_ctx_t ctx,
                            double* d_reduce_in, double* d_reduce_out)
{
    double local_sum;
    ROCBLAS_CHECK(rocblas_ddot(blas_handle, n, d_a, 1, d_b, 1, &local_sum));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(d_reduce_in, &local_sum, sizeof(double), hipMemcpyHostToDevice));
    ROCSHMEM_CHECK(rocshmem_ctx_double_sum_reduce(ctx, ROCSHMEM_TEAM_WORLD,
                                                  d_reduce_out, d_reduce_in, 1));
    double global_sum;
    HIP_CHECK(hipMemcpy(&global_sum, d_reduce_out, sizeof(double), hipMemcpyDeviceToHost));
    return global_sum;
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

    const char* filename = "Dubcova2.pm";
    if (argc > 1) filename = argv[1];

    int num_gpus;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    HIP_CHECK(hipSetDevice(rank % num_gpus));

    rocshmem_init();

    // Reuse rocSHMEM's own default host context rather than creating a new
    // one (see comment near the top of the file).
    rocshmem_ctx_t ctx = rocshmem::ROCSHMEM_HOST_CTX_DEFAULT;

    // Persistent single-element symmetric-heap scalars reused by every
    // rocSHMEM reduction (dot products during the CG loop, and the final
    // timing reduction).
    double *d_reduce_in  = static_cast<double *>(rocshmem_malloc(sizeof(double)));
    double *d_reduce_out = static_cast<double *>(rocshmem_malloc(sizeof(double)));

    if (rank == 0)
        printf("method=rocshmem  ranks=%d  gpus_visible=%d\n",
               num_procs, num_gpus);

    rocsparse_handle sparse_handle;
    rocblas_handle   blas_handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&sparse_handle));
    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle));

    ParMat A;
    readParMatrix(filename, A);
    form_comm(A);

    RocshmemGPUParMat gA;
    upload_par_mat(A, gA, sparse_handle, ctx, num_procs);

    int n = A.local_rows;

    double *d_x, *d_b;
    HIP_CHECK(hipMalloc(&d_x, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, n * sizeof(double)));

    srand(time(NULL) + rank);
    std::vector<double> h_x(n);
    std::generate(h_x.begin(), h_x.end(), [](){ return (double)rand() / RAND_MAX; });
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), n * sizeof(double), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(d_b, 0, n * sizeof(double)));
    spmv_rocshmem(1.0, gA, d_x, 0.0, d_b, sparse_handle);
    HIP_CHECK(hipMemset(d_x, 0, n * sizeof(double)));

    double *d_r, *d_p, *d_Ap;
    HIP_CHECK(hipMalloc(&d_r,  n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_p,  n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Ap, n * sizeof(double)));

    HIP_CHECK(hipMemcpy(d_r, d_b, n * sizeof(double), hipMemcpyDeviceToDevice));
    spmv_rocshmem(-1.0, gA, d_x, 1.0, d_r, sparse_handle);

    HIP_CHECK(hipMemcpy(d_p, d_r, n * sizeof(double), hipMemcpyDeviceToDevice));

    double rr_inner = inner_product(d_r, d_r, n, blas_handle, ctx,
                                    d_reduce_in, d_reduce_out);
    double norm_r   = sqrt(rr_inner);
    double tol      = 1e-6;

    if (rank == 0) printf("Initial residual: %lg\n", norm_r);
    if (norm_r != 0.0) tol *= norm_r;

    int recompute_r = 8;
    int max_iter    = (int)(1.3 * A.global_rows) + 2;
    int iter        = 0;

    rocshmem_barrier_all();
    double t_start = MPI_Wtime();

    while (norm_r > tol && iter < max_iter)
    {
        spmv_rocshmem(1.0, gA, d_p, 0.0, d_Ap, sparse_handle);

        double App_inner = inner_product(d_Ap, d_p, n, blas_handle, ctx,
                                          d_reduce_in, d_reduce_out);
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
            spmv_rocshmem(-1.0, gA, d_x, 1.0, d_r, sparse_handle);
        }

        double next_inner = inner_product(d_r, d_r, n, blas_handle, ctx,
                                          d_reduce_in, d_reduce_out);
        double beta       = next_inner / rr_inner;

        scale(beta, d_p, n, blas_handle);
        axpy(1.0, d_p, d_r, n, blas_handle);

        rr_inner = next_inner;
        norm_r   = sqrt(rr_inner);
        iter++;
    }

    HIP_CHECK(hipDeviceSynchronize());
    double t_elapsed = MPI_Wtime() - t_start;

    // Reuse the persistent reduce scalars for the final max-time reduction.
    HIP_CHECK(hipMemcpy(d_reduce_in, &t_elapsed, sizeof(double), hipMemcpyHostToDevice));
    ROCSHMEM_CHECK(rocshmem_ctx_double_max_reduce(ctx, ROCSHMEM_TEAM_WORLD,
                                                  d_reduce_out, d_reduce_in, 1));
    double t_max;
    HIP_CHECK(hipMemcpy(&t_max, d_reduce_out, sizeof(double), hipMemcpyDeviceToHost));

    if (rank == 0) {
        if (iter == max_iter) printf("Max iterations reached.\n");
        else                   printf("%d iterations to converge\n", iter);
        printf("2-norm of residual: %lg\n", norm_r);
        printf("CG solve time:      %.4f s  (%.4f s/iter)\n",
               t_max, t_max / iter);
    }

    HIP_CHECK(hipFree(d_x));  HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_r));  HIP_CHECK(hipFree(d_p));  HIP_CHECK(hipFree(d_Ap));

    free_gpu_mat(gA.on_proc);
    free_gpu_mat(gA.off_proc);

    if (gA.d_sendbuf)  rocshmem_free(gA.d_sendbuf);
    if (gA.d_recvbuf)  rocshmem_free(gA.d_recvbuf);
    if (gA.d_send_idx) HIP_CHECK(hipFree(gA.d_send_idx));

    HIP_CHECK(hipStreamDestroy(gA.comm_stream));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(sparse_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(blas_handle));

    rocshmem_free(d_reduce_in);
    rocshmem_free(d_reduce_out);
    // ctx is rocSHMEM's own default host context — do not destroy it here;
    // it is torn down internally by rocshmem_finalize().

    rocshmem_finalize();
    MPI_Finalize();
    return 0;
}
