#!/bin/bash
#SBATCH --job-name=cg_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --output=cg_cpu_test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null
module load rocm/6.4.3 openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4

# SLURM copies the batch script to a spool dir; resolve CG-CPU/ from where
# sbatch was invoked (submit from inside CG-CPU/).
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "FAIL: cannot cd to submit dir"; exit 1; }

make clean && make

mpirun -n ${SLURM_NTASKS:-4} ./cg_cpu src/Dubcova2.pm
