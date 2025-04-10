#!/bin/bash
#SBATCH --output=fast.out
#SBATCH --error=fast.err
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

module load PrgEnv-gnu cpe-cuda cuda python
export PYTHONPATH=$PYTHONPATH:/global/homes/a/akaptano/deepxde_copy/
module load tensorflow/2.9.0
export HDF5_USE_FILE_LOCKING='FALSE'
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1

srun -n1 python3 fast_solovev_equil.py 


