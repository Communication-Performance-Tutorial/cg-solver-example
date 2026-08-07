#!/bin/bash
#SBATCH --job-name=cg_gpu_tau
#SBATCH --partition=PPAC_MI300A_CPX
#SBATCH --exclusive
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --time=00:20:00
#SBATCH --output=cg_tau_test_%j.log
# =============================================================================
# TAU profiling + tracing for the GPU Conjugate Gradient solver (CG-GPU)
#
# Wraps the already-built ./cg_gpu with `tau_exec`, which intercepts MPI
# (-T MPI) and, with -rocm, ROCm/HIP GPU calls via LD_PRELOAD. No special
# instrumented build is needed (unlike Score-P's compiler instrumentation).
#
# The login node has no GPU, so both the build and the run happen on a GPU
# compute node inside this allocation.
#
# Submit:
#   sbatch run_tau.sh                    # profile only (pprof text summary)
#   TRACE=1 sbatch run_tau.sh             # also collect an OTF2 trace + a
#                                         # Perfetto/Chrome JSON trace (open at
#                                         # https://ui.perfetto.dev)
#   METHOD=isend sbatch run_tau.sh        # profile a different comm. variant
#
# Override the partition / GPU count on the command line, e.g.:
#   sbatch -p PPAC_MI300A_SPX --gpus=4 --ntasks=4 run_tau.sh
#
# Env overrides:
#   ROCM_MODULE      rocm module to load       (default: rocm/6.4.3)
#   OPENMPI_MODULE   openmpi module to load    (default: openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4)
#   METHOD           comm. variant to profile  (default: rccl)
#   MATRIX           matrix file               (default: src/Dubcova2.pm)
#   TRACE            1 = also write an OTF2 trace (default: 0, profile only)
#   TAU_DIR_PREFIX   experiment dir name       (default: tau_cg_<method>)
# =============================================================================

set -u
date
echo "Node(s): ${SLURM_JOB_NODELIST:-<none - running outside SLURM>}"

# SLURM copies the batch script to a spool dir; resolve CG-GPU/ from where
# sbatch was invoked (submit from inside CG-GPU/).
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "FAIL: cannot cd to submit dir"; exit 1; }

# ---------------------------------------------------------------------------
# Modules: ROCm + GPU-Aware OpenMPI + TAU (TAU is layered on top of the rocm
# module, so it must load after ROCm is on the module path).
# ---------------------------------------------------------------------------
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/6.4.3}"
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}" 2>/dev/null || module load openmpi
module load tau/dev
module list 2>&1

command -v tau_exec >/dev/null || { echo "FAIL: tau_exec not on PATH"; exit 1; }
echo "[info] $(tau_exec -help 2>&1 | head -1) | $(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*')"

# ---------------------------------------------------------------------------
# Build (plain build -- tau_exec instruments at run time, no recompile needed)
# ---------------------------------------------------------------------------
echo "=== Building cg_gpu ==="
make clean && make || { echo "FAIL: build failed"; exit 1; }
[ -x ./cg_gpu ] || { echo "FAIL: cg_gpu not produced"; exit 1; }

METHOD=${METHOD:-rccl}
MATRIX=${MATRIX:-src/Dubcova2.pm}
NUM_RANKS=${SLURM_NTASKS:-4}
EXPDIR=${TAU_DIR_PREFIX:-tau_cg_${METHOD}}
rm -rf "$EXPDIR"
mkdir -p "$EXPDIR"

# ---------------------------------------------------------------------------
# TAU measurement configuration
#   PROFILEDIR       -- where profile.<rank>.0.0 files land (pprof/paraprof)
#   TAU_COMM_MATRIX   -- record the per-rank MPI point-to-point comm matrix
#   TAU_TRACE/TRACEDIR -- optional OTF2-mergeable event trace (TRACE=1)
# ---------------------------------------------------------------------------
export PROFILEDIR="$EXPDIR"
export TAU_COMM_MATRIX=1
if [ "${TRACE:-0}" = "1" ]; then
    export TAU_TRACE=1
    export TRACEDIR="$EXPDIR"
else
    unset TAU_TRACE TRACEDIR
fi

echo "=== Running: method=$METHOD ranks=$NUM_RANKS matrix=$MATRIX (TRACE=${TRACE:-0}) ==="
mpirun -n "$NUM_RANKS" --bind-to none bash set_affinity_mi300a.sh \
    tau_exec -T MPI,ROCM -rocm ./cg_gpu "$MATRIX" "$METHOD"
STATUS=$?
# TAU/ROCm can abort with "corrupted size vs. prev_size in fastbins" during
# library teardown (a known rocprofsdk + glibc interaction on ROCm 6.4.x) even
# though every profile.* file was flushed to disk correctly beforehand -- don't
# fail the job on a nonzero mpirun exit alone.

echo
echo "=== pprof text summary ($EXPDIR) ==="
# pprof/paraprof read $PROFILEDIR themselves -- run from CG-GPU/, NOT from
# inside $EXPDIR (cd'ing there while PROFILEDIR is still set makes pprof look
# for a nonexistent nested "$EXPDIR/$EXPDIR/profile.*").
if ls "$EXPDIR"/profile.* >/dev/null 2>&1; then
    pprof
else
    echo "WARN: no profile.* files in $EXPDIR -- measurement likely aborted before writing output"
fi

if [ "${TRACE:-0}" = "1" ]; then
    echo
    echo "=== Merging TAU trace -> OTF2 + Perfetto/Chrome JSON ($EXPDIR) ==="
    if ls "$EXPDIR"/tautrace.*.trc >/dev/null 2>&1; then
        # Unlike pprof, tau_treemerge.pl/tau2otf2/tau_trace2json do NOT honor
        # $TRACEDIR to locate input files -- they glob tautrace.*.trc /
        # events.*.edf in the current directory, so cd into $EXPDIR (no
        # PROFILEDIR-style trap here: these tools don't consult
        # $PROFILEDIR/$TRACEDIR at all). tau_treemerge.pl produces the merged
        # tau.trc/tau.edf that both downstream converters read.
        (
            cd "$EXPDIR" \
            && tau_treemerge.pl \
            && tau2otf2 tau.trc tau.edf "otf2_${METHOD}" \
            && tau_trace2json tau.trc tau.edf -chrome -o "perfetto_${METHOD}.json"
        ) || echo "WARN: OTF2/Perfetto conversion failed"
    else
        echo "WARN: no tautrace.*.trc files in $EXPDIR -- trace not written"
    fi
fi

echo
echo "=== Artifacts ==="
echo "  $PWD/$EXPDIR/profile.*              -- pprof (above) or paraprof (GUI: per-call"
echo "                                         bar charts + comm. matrix, needs a JRE)"
if [ "${TRACE:-0}" = "1" ]; then
echo "  $PWD/$EXPDIR/otf2_${METHOD}/        -- OTF2 event trace, open in a trace viewer"
echo "  $PWD/$EXPDIR/perfetto_${METHOD}.json -- Chrome/Perfetto trace, open at"
echo "                                         https://ui.perfetto.dev (or chrome://tracing)"
fi

date
exit $STATUS
