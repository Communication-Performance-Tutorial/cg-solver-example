#!/bin/bash
#SBATCH --job-name=cg_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --partition=192C4G1H_MI300A_RHEL9_A1_CPX
#SBATCH --time=00:10:00
#SBATCH --output=cg_test_cpx_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

module swap PrgEnv-cray/8.7.0 PrgEnv-amd-openmpi/openmpi-5.0.10-ofi-7.14.0

# --bind-to none: let set_affinity_mi300a.sh own all CPU and GPU pinning.
# The affinity script sets HIP_VISIBLE_DEVICES=local_rank (one GPU per rank)
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
