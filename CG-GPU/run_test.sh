#!/bin/bash
#SBATCH --job-name=cg_gpu_6.4.3
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --time=00:10:00
#SBATCH --output=cg_6.4.3_test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null
module load rocm/6.4.3 openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4

# SLURM copies the batch script to a spool dir; resolve CG-GPU/ from where
# sbatch was invoked (submit from inside CG-GPU/).
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "FAIL: cannot cd to submit dir"; exit 1; }

make clean && make

# --bind-to none: let set_affinity_mi300a.sh own all CPU and GPU pinning.
# The affinity script sets ROCR_VISIBLE_DEVICES=local_rank (one GPU per rank)
# and pins each rank to the CPU cores sharing the same memory domain.
# One rank == one GPU, so size everything off however many tasks/GPUs SLURM
# actually granted this job (allows overriding --ntasks/--gres on the sbatch
# command line to fit whatever node is available, not just an 8-GPU node).
NUM_RANKS=${SLURM_NTASKS:-8}
export NUM_GPUS=${NUM_RANKS}
export NUM_CPUS=$(( NUM_RANKS * ${SLURM_CPUS_PER_TASK:-4} ))
AFFINITY="./set_affinity_mi300a.sh"
MPIRUN="mpirun -n ${NUM_RANKS} --bind-to none bash ${AFFINITY}"

# Non-GPU-aware methods: no dependency on the SDMA vs blit-kernel path
for m in staged alltoallv_staged rccl; do
    echo "=== $m ==="
    ${MPIRUN} ./cg_gpu src/Dubcova2.pm $m
    echo
done

# GPU-aware MPI methods: sweep SDMA engines vs blit kernels.
#
#   HSA_ENABLE_SDMA=1  (default) — ROCm uses hardware SDMA engines for
#                                   GPU-to-GPU / GPU-to-host DMA transfers.
#   HSA_ENABLE_SDMA=0            — SDMA engines are disabled; ROCm falls back
#                                   to blit kernels (compute-shader copies),
#                                   which use the shader engines instead of the
#                                   dedicated DMA hardware.
#
# Comparing the two reveals whether the SDMA path or the blit-kernel path
# is faster for the message sizes produced by this matrix/decomposition.
for sdma in 1 0; do
    if [ "$sdma" -eq 1 ]; then
        label="sdma"
    else
        label="blit_kernel"
    fi
    for m in isend alltoallv; do
        echo "=== ${m}  HSA_ENABLE_SDMA=${sdma}  (${label}) ==="
        HSA_ENABLE_SDMA=${sdma} ${MPIRUN} ./cg_gpu src/Dubcova2.pm $m
        echo
    done
done
