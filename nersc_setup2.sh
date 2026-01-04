#!/bin/bash
# -----------------------------
# NERSC HPC environment setup
# -----------------------------
salloc -N1 -t 01:00:00

module purge
# Restore system defaults first
source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh

# Load required modules
module load python
module load cray-mpich
# module load cray-hdf5-parallel   # optional if using h5py with MPI

# -----------------------------
# Build mode and GPU flag
# -----------------------------
export BUILD_MODE=nersc
export INSTALL_PYCUDA=0

# -----------------------------
# Verify MPI
# -----------------------------
echo "MPI compiler:"
which mpicc
mpicc -show

# -----------------------------
# Configure Python environment
# -----------------------------
make configure

# -----------------------------
# Finished
# -----------------------------
echo "NERSC environment setup complete."
