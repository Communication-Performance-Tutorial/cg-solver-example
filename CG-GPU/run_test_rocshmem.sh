#!/bin/bash
#SBATCH --job-name=cg_gpu_rocshmem
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=cg_rocshmem_test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cg-solver-example and micro-benchmarks are sibling repos checked out under the
# same parent directory; reuse the rocSHMEM build/install from the sibling repo.
ROCSHMEM_DIR="$(cd "${SCRIPT_DIR}/../../micro-benchmarks/rocshmem" && pwd)"

export ROCSHMEM_INSTALL="${ROCSHMEM_INSTALL:-$HOME/rocshmem}"
export ROCSHMEM_PATH="${ROCSHMEM_INSTALL}"
ROCSHMEM_LIB="${ROCSHMEM_INSTALL}/lib/librocshmem.a"

# Build rocSHMEM if needed (see micro-benchmarks/rocshmem/README.md).
# First build takes ~20 min; installs to ~/rocshmem by default.
if [[ ! -f "$ROCSHMEM_LIB" ]]; then
    echo "=== Build rocSHMEM (first run: ~20 min) ==="
    bash "${ROCSHMEM_DIR}/build.sh"
else
    echo "=== rocSHMEM already built at ${ROCSHMEM_INSTALL} ==="
fi

module load rocm/7.13.0 openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4

cd "${SCRIPT_DIR}"

export ROCM_PATH="$(dirname "$(dirname "$(command -v hipcc)")")"

echo "=== Environment ==="
echo "ROCM_PATH=${ROCM_PATH}"
echo "ROCSHMEM_PATH=${ROCSHMEM_PATH}"
echo "hipcc: $(command -v hipcc)"
echo "mpirun: $(command -v mpirun)"
echo

echo "=== Build cg_gpu_rocshmem ==="
make clean
make cg_gpu_rocshmem
echo

# --bind-to none: let set_affinity_mi300a.sh own all CPU and GPU pinning.
AFFINITY="${SCRIPT_DIR}/set_affinity_mi300a.sh"
MPIRUN="mpirun -n 8 --bind-to none bash ${AFFINITY}"

echo "=== rocshmem ==="
${MPIRUN} ./cg_gpu_rocshmem src/Dubcova2.pm
echo

# Reference: GPU-aware MPI isend on the same matrix/decomposition.
echo "=== isend (reference) ==="
make cg_gpu
${MPIRUN} ./cg_gpu src/Dubcova2.pm isend
echo
