#!/bin/bash
#SBATCH --job-name=cg_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --output=cg_cpu_test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

module swap PrgEnv-cray/8.7.0 PrgEnv-amd-openmpi/openmpi-5.0.10-ofi-7.14.0

mpirun -n ${SLURM_NTASKS:-4} ./cg_cpu src/Dubcova2.pm
